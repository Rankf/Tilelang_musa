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

extern "C" __global__ void l1_tma_probe_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) l1_tma_probe_kernel(const float* __restrict__ A, float* __restrict__ C) {
  __musa_async_bar_record(2);
  extern __shared__ __align__(4096) float shared[];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((128 <= ((int)threadIdx.x))) {
    for (int __1 = 0; __1 < 1; ++__1) {
      __musa_async_wait(2, 1);
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(1, 16384);
        tl::tma_load<CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>((&(shared[0])), (&(A[(((int)blockIdx.x) * 4096)])), 1, 16384);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    for (int __2 = 0; __2 < 1; ++__2) {
      __musa_async_wait(1, 0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store<CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>((&(C[(((int)blockIdx.x) * 4096)])), (&(shared[0])), 16384);
        tl::tma_store_arrive();
        tl::tma_store_wait<0>();
      }
      __musa_async_arrive(2);
    }
  }
}

