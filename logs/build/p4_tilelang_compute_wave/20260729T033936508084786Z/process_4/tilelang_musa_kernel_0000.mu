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

extern "C" __global__ void fixed_grid_compute_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) fixed_grid_compute_kernel(const float* __restrict__ A, float* __restrict__ C) {
  float acc[1];
  acc[0] = A[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))];
  for (int k = 0; k < 2048; ++k) {
    acc[0] = ((acc[0] * 0x1.00068db8bac71p+0f/*1.000100e+00*/) + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/);
  }
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = acc[0];
}

