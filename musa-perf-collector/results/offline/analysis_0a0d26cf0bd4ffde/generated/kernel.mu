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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ out);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ out) {
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  float acc[64];
  float out_local[8];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(tl_f4*)(acc + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)a_shared)[(((i_1 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))])), (&(A[((i_1 * 2048) + (((int)threadIdx.x) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)b_shared)[(((i_2 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))])), (&(B[((i_2 * 2048) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads_lm();
  tl::gemm_ss<128, 128, 64, 4, 2, 0, 1, 0, 64, 64, 0, 0, true, 128, 64, 64>((&(((half_t*)a_shared)[0])), (&(((half_t*)b_shared)[0])), (&(acc[0])), 0);
  __syncthreads_lm();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    out_local[i_3] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      out_local[i_3] = (out_local[i_3] + acc[((i_3 * 8) + rv)]);
    }
    out_local[i_3] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(out_local[i_3], (&(((float*)workspace)[0])));
    out_local[i_3] = tl::AllReduce<tl::SumOp, 8, 1, 0>::run(out_local[i_3]);
  }
  if ((((((int)threadIdx.x) & 7) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      out[((i_4 * 16) + ((((int)threadIdx.x) & 127) >> 3))] = out_local[i_4];
    }
  }
}

