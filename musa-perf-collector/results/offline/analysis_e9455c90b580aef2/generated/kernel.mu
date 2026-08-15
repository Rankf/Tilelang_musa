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

extern "C" __global__ void kernel_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C, const int* __restrict__ offset);
extern "C" __global__ void __launch_bounds__(384, 1) kernel_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, __attribute__((grid_constant)) const MUtensorDescriptor B_desc_1, half_t* __restrict__ C, const int* __restrict__ offset) {
  __musa_async_bar_record(4);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  int offset_local_producer_ws[1];
  int offset_local[1];
  float acc[16];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((256 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((256 <= ((int)threadIdx.x))) {
    offset_local_producer_ws[0] = offset[0];
    for (int kk = 0; kk < 4; ++kk) {
      __musa_async_wait(((kk & 1) + 3), ((kk >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(((kk & 1) + 1), 4096);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, ((kk & 1) + 1), (&(((half_t*)a_shared)[((kk & 1) * 2048)])), (kk * 32), ((((int)blockIdx.y) * 64) + (offset_local_producer_ws[0] * 64)), 32, 64);
      }
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((kk & 1) + 1), 4096);
        tl::tma_load<SmemSwizzleGranularity::B32, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, ((kk & 1) + 1), (&(((half_t*)b_shared)[((kk & 1) * 2048)])), (((int)blockIdx.x) * 64), (kk * 32), 32, 32);
        tl::tma_load<SmemSwizzleGranularity::B32, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(B_desc_1, ((kk & 1) + 1), (&(((half_t*)b_shared)[(((kk & 1) * 2048) + 1024)])), ((((int)blockIdx.x) * 64) + 32), (kk * 32), 32, 32);
      }
    }
  }
  if ((((int)threadIdx.x) < 256)) {
    offset_local[0] = offset[0];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(tl_f4*)(acc + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
    }
    for (int kk_1 = 0; kk_1 < 4; ++kk_1) {
      __musa_async_wait(((kk_1 & 1) + 1), (kk_1 >> 1));
      tl::gemm_ss<64, 64, 32, 4, 2, 0, 0, 0, 32, 64, 0, 0, true, 64, 32, 32>((&(((half_t*)a_shared)[((kk_1 & 1) * 2048)])), (&(((half_t*)b_shared)[((kk_1 & 1) * 2048)])), (&(acc[0])), 0);
      __musa_async_arrive(((kk_1 & 1) + 3));
    }
    if (((((int)blockIdx.y) + offset_local[0]) < 2) && (0 <= (((int)blockIdx.y) + offset_local[0]))) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        if (0 <= (((int)blockIdx.y) + offset_local[0])) {
          if ((((int)blockIdx.y) + offset_local[0]) < 2) {
            C[((((((((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)offset_local[(int64_t)0]) * (int64_t)8192)) + ((((int64_t)i_1) >> (int64_t)2) * (int64_t)2048)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)3) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)32)) + ((((int64_t)i_1) & (int64_t)3) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)7))] = ((half_t)acc[i_1]);
          }
        }
      }
    }
  }
}

