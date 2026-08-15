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

extern "C" __global__ void main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, int* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, int* __restrict__ C) {
  __musa_async_bar_record(2);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b = ((void*)((char*)buf_dyn_shmem + 0));
  void* a = ((void*)((char*)buf_dyn_shmem + 8192));
  int out[128];
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
    for (int k = 0; k < 16; ++k) {
      __musa_async_wait(2, ((k & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(1, 8192);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, 1, (&(((signed char*)a)[0])), (k * 64), (((int)blockIdx.y) * 128), 64, 128);
        tl::mbarrier_arrive_expect_tx(1, 8192);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, 1, (&(((signed char*)b)[0])), (((int)blockIdx.x) * 128), (k * 64), 128, 64);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      int broadcast_var = 0;
      *(int4*)(out + (i * 4)) = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      __musa_async_wait(1, (k_1 & 1));
      tl::gemm_ss<128, 128, 64, 4, 1, 0, 0, 0, 64, 128, 0, 0, true, 128, 128, 64>((&(((signed char*)a)[0])), (&(((signed char*)b)[0])), (&(out[0])), 0);
      __musa_async_arrive(2);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 128; ++i_1) {
      C[((((((((int)blockIdx.y) * 131072) + ((i_1 >> 4) * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_1 & 15) * 8)) + (((int)threadIdx.x) & 7))] = out[i_1];
    }
  }
}

