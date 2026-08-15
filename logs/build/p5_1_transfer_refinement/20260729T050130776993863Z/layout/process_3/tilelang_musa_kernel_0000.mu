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

extern "C" __global__ void strided_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) strided_kernel(const float* __restrict__ A, float* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    C[(((((int)blockIdx.x) * 2048) + (i * 256)) + ((int)threadIdx.x))] = A[(((((int)blockIdx.x) * 4096) + (i * 512)) + (((int)threadIdx.x) * 2))];
  }
}

