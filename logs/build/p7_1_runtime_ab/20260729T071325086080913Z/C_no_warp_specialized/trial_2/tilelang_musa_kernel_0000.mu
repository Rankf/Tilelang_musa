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

extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b = ((void*)((char*)buf_dyn_shmem + 0));
  void* a = ((void*)((char*)buf_dyn_shmem + 8192));
  float out[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(tl_f4*)(out + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)a)[(((i_1 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_1 & 1) * 8) + (((int)threadIdx.x) >> 4))) * 8))])), (&(A[((((((int)blockIdx.y) * 98304) + (i_1 * 24576)) + ((((int)threadIdx.x) >> 3) * 1536)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)b)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ (((int)threadIdx.x) >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(B[((((i_2 * 16384) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 23; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads_lm();
    tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true, 64, 64, 64>((&(((half_t*)a)[0])), (&(((half_t*)b)[0])), (&(out[0])), 0);
    __syncthreads_lm();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)a)[(((i_3 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_3 & 1) * 8) + (((int)threadIdx.x) >> 4))) * 8))])), (&(A[((((((((int)blockIdx.y) * 98304) + (i_3 * 24576)) + ((((int)threadIdx.x) >> 3) * 1536)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)b)[((((i_4 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ (((int)threadIdx.x) >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(B[((((((k * 65536) + (i_4 * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads_lm();
  tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true, 64, 64, 64>((&(((half_t*)a)[0])), (&(((half_t*)b)[0])), (&(out[0])), 0);
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    C[((((((((int)blockIdx.y) * 65536) + ((i_5 >> 3) * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 64)) + ((i_5 & 7) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)out[i_5]);
  }
}

