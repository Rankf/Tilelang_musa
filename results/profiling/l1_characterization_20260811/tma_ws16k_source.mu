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
  float accum[1];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((128 <= ((int)threadIdx.x))) {
    for (int __1 = 0; __1 < 8; ++__1) {
      __musa_async_wait(2, ((__1 & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(1, 16384);
        tl::tma_load<CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>((&(shared[0])), (&(A[(((int)blockIdx.x) * 4096)])), 1, 16384);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    accum[0] = 0x0p+0f/*0.000000e+00*/;
    for (int __2 = 0; __2 < 8; ++__2) {
      __musa_async_wait(1, (__2 & 1));
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        int broadcast_var = 0;
          int4 v_ = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          tl_f4 v__1 = tl::add4((tl_f4{static_cast<float>(accum[0]), static_cast<float>(accum[0]), static_cast<float>(accum[0]), static_cast<float>(accum[0])}), *(tl_f4*)(shared + (i * 4)));
          accum[v_.x] = v__1[0];
          accum[v_.y] = v__1[1];
          accum[v_.z] = v__1[2];
          accum[v_.w] = v__1[3];
      }
      __musa_async_arrive(2);
    }
    C[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = accum[0];
  }
}

