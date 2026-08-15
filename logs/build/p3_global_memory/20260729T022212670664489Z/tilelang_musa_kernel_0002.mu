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

extern "C" __global__ void fixed_grid_copy_kernel(const float* __restrict__ A, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) fixed_grid_copy_kernel(const float* __restrict__ A, float* __restrict__ C) {
  for (int k = 0; k < 4; ++k) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      *(tl_f4*)(C + ((((k * 2097152) + (((int)blockIdx.x) * 2048)) + (i * 1024)) + (((int)threadIdx.x) * 4))) = *(tl_f4*)(A + ((((k * 2097152) + (((int)blockIdx.x) * 2048)) + (i * 1024)) + (((int)threadIdx.x) * 4)));
    }
  }
}

