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

extern "C" __global__ void main_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ A, float* __restrict__ C) {
  float x[1];
  float y[1];
  extern __shared__ __align__(4096) float workspace[];
  x[0] = A[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
  y[0] = 0x0p+0f/*0.000000e+00*/;
  y[0] = (y[0] + x[0]);
  y[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(y[0], (&(workspace[0])));
  C[((int)blockIdx.x)] = y[0];
}

