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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ a, half_t* __restrict__ out);
extern "C" __global__ void __launch_bounds__(32, 1) kernel_kernel(const half_t* __restrict__ a, half_t* __restrict__ out) {
  half_t frag[2];
  tl_h2 out_frag_pack[1];
  half_t out_frag[1];
  *(tl_h2*)(frag + 0) = *(tl_h2*)(a + (((int)threadIdx.x) * 2));
  half_t broadcast_var = -std::numeric_limits<half_t>::infinity();
  out_frag_pack[0] = (tl_h2{make_tl_h_elem(half_t(static_cast<float>(broadcast_var))), make_tl_h_elem(half_t(static_cast<float>(broadcast_var)))});
  out_frag_pack[0] = tl::max2(out_frag_pack[0], *(tl_h2*)(frag + 0));
  out_frag[0] = mutlass::fast_max(tl_h_elem_to_half((out_frag_pack[0])[0]), tl_h_elem_to_half((out_frag_pack[0])[1]));
  out_frag[0] = tl::AllReduce<tl::MaxOp, 32, 1, 0>::run(out_frag[0]);
  if (((int)threadIdx.x) == 0) {
    out[0] = out_frag[0];
  }
}

