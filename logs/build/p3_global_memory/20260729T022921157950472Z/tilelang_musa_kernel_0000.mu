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

extern "C" __global__ void shared_read_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) shared_read_kernel(const float* __restrict__ A, float* __restrict__ C) {
  extern __shared__ __align__(4096) float tile[];
  float acc[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    tl::cp_async_gs<16>((&(tile[((i * 512) + (((int)threadIdx.x) * 4))])), (&(A[(((((int)blockIdx.x) * 2048) + (i * 512)) + (((int)threadIdx.x) * 4))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(tl_f4*)(acc + (i_1 * 4)) = *(tl_f4*)(tile + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    *(tl_f4*)(C + (((((int)blockIdx.x) * 2048) + (i_2 * 512)) + (((int)threadIdx.x) * 4))) = *(tl_f4*)(acc + (i_2 * 4));
  }
}

