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

extern "C" __global__ void matmul_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) matmul_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C) {
  __musa_async_bar_record(6);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  float c_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc_0);
    tl::prefetch_tma_descriptor(B_desc_1);
  }
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((128 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((128 <= ((int)threadIdx.x))) {
    for (int k = 0; k < 8; ++k) {
      __musa_async_wait(((k % 3) + 4), (((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(((k % 3) + 1), 16384);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, ((k % 3) + 1), (&(((half_t*)a_shared)[((k % 3) * 8192)])), (k * 64), (((int)blockIdx.y) * 128), 64, 128);
        tl::mbarrier_arrive_expect_tx(((k % 3) + 1), 16384);
        tl::tma_load<SmemSwizzleGranularity::B32, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, ((k % 3) + 1), (&(((half_t*)b_shared)[((k % 3) * 8192)])), (((int)blockIdx.x) * 128), (k * 64), 128, 64);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(tl_f4*)(c_local + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
    }
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      __musa_async_wait(((k_1 % 3) + 1), ((k_1 % 6) / 3));
      tl::gemm_ss<128, 128, 64, 4, 1, 0, 0, 0, 64, 128, 0, 0, true, 128, 128, 64>((&(((half_t*)a_shared)[((k_1 % 3) * 8192)])), (&(((half_t*)b_shared)[((k_1 % 3) * 8192)])), (&(c_local[0])), 0);
      __musa_async_arrive(((k_1 % 3) + 4));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 128; ++i_1) {
      C[((((((((int)blockIdx.y) * 65536) + ((i_1 >> 4) * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.x) * 128)) + ((i_1 & 15) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)c_local[i_1]);
    }
  }
}

