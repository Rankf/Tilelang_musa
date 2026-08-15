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

extern "C" __global__ void unary_kernel(float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(32, 1) unary_kernel(float* __restrict__ C) {
  C[((int)threadIdx.x)] = 0x1p+0f/*1.000000e+00*/;
}

