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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C) {
  __musa_async_bar_record(2);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b = ((void*)((char*)buf_dyn_shmem + 0));
  void* a = ((void*)((char*)buf_dyn_shmem + 8192));
  float out[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc_0);
    tl::prefetch_tma_descriptor(B_desc_1);
  }
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((128 <= ((int)threadIdx.x))) {
    for (int k = 0; k < 24; ++k) {
      __musa_async_wait(2, ((k & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(1, 8192);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, 1, (&(((half_t*)a)[0])), (k * 64), (((int)blockIdx.y) * 64), 64, 64);
        tl::mbarrier_arrive_expect_tx(1, 8192);
        tl::tma_load<SmemSwizzleGranularity::B32, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, 1, (&(((half_t*)b)[0])), (((int)blockIdx.x) * 64), (k * 64), 64, 64);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(tl_f4*)(out + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
    }
    for (int k_1 = 0; k_1 < 24; ++k_1) {
      __musa_async_wait(1, (k_1 & 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true, 64, 64, 64>((&(((half_t*)a)[0])), (&(((half_t*)b)[0])), (&(out[0])), 0);
      __musa_async_arrive(2);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      C[((((((((int)blockIdx.y) * 65536) + ((i_1 >> 3) * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 64)) + ((i_1 & 7) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)out[i_1]);
    }
  }
}

