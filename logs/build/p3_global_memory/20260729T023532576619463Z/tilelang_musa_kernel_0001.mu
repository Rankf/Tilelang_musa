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

extern "C" __global__ void read_sum_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) read_sum_kernel(const float* __restrict__ A, float* __restrict__ C) {
  float acc[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int k = 0; k < 64; ++k) {
    acc[0] = (acc[0] + A[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 64)) + k)]);
  }
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = acc[0];
}

