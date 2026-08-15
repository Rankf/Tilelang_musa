#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/fop.h>
#include <tl_templates/musa/accelerated_ops.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C) {
  __musa_async_bar_record(6);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float c_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc_0);
    tl::prefetch_tma_descriptor(B_desc_1);
  }
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((256 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((256 <= ((int)threadIdx.x))) {
    for (int k = 0; k < 4; ++k) {
      __musa_async_wait(((k % 3) + 4), ((k / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(((k % 3) + 1), 8192);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, ((k % 3) + 1), (&(((half_t*)a_shared)[((k % 3) * 4096)])), (k * 64), (((int)blockIdx.y) * 64), 64, 64);
        tl::mbarrier_arrive_expect_tx(((k % 3) + 1), 8192);
        tl::tma_load<SmemSwizzleGranularity::B32, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, ((k % 3) + 1), (&(((half_t*)b_shared)[((k % 3) * 4096)])), (((int)blockIdx.x) * 64), (k * 64), 64, 64);
      }
    }
  }
  if ((((int)threadIdx.x) < 256)) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(tl_f4*)(c_local + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
    }
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      __musa_async_wait(((k_1 % 3) + 1), (k_1 / 3));
      tl::gemm_ss<64, 64, 64, 8, 1, 0, 0, 0, 64, 64, 0, 0, true, 32, 64, 64>((&(((half_t*)a_shared)[((k_1 % 3) * 4096)])), (&(((half_t*)b_shared)[((k_1 % 3) * 4096)])), (&(c_local[0])), 0);
      __musa_async_arrive(((k_1 % 3) + 4));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      C[(((((((((int)blockIdx.y) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((i_1 >> 3) * 4096)) + (((((int)threadIdx.x) & 127) >> 3) * 256)) + (((int)blockIdx.x) * 64)) + ((i_1 & 7) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)c_local[i_1]);
    }
  }
}

