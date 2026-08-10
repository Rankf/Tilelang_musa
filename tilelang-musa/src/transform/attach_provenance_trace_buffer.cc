/*!
 * \file attach_provenance_trace_buffer.cc
 * \brief Append the bounded provenance Trace buffer to a cloned PrimFunc ABI.
 */

#include "support/check.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

namespace {

PrimFunc AttachTraceBuffer(PrimFunc f, String trace_buffer_name,
                           Integer trace_extent) {
  ICHECK_GT(trace_extent->value, 0)
      << "AttachProvenanceTraceBuffer extent must be positive";
  ICHECK(!trace_buffer_name.empty())
      << "AttachProvenanceTraceBuffer name must not be empty";

  for (const Var &param : f->params) {
    ICHECK_NE(param->name_hint, trace_buffer_name)
        << "AttachProvenanceTraceBuffer parameter name already exists: "
        << trace_buffer_name;
  }
  for (const auto &[param, buffer] : f->buffer_map) {
    ICHECK_NE(buffer->name, trace_buffer_name)
        << "AttachProvenanceTraceBuffer buffer name already exists: "
        << trace_buffer_name;
  }

  Array<Var> params = f->params;
  Map<Var, Buffer> buffer_map = f->buffer_map;
  Var trace_param(trace_buffer_name, DataType::Handle());
  Buffer trace_buffer = decl_buffer(
      {IntImm(DataType::Int(32), trace_extent->value)}, DataType::Int(64),
      trace_buffer_name, "global");
  const int64_t parameter_index = static_cast<int64_t>(params.size());
  params.push_back(trace_param);
  buffer_map.Set(trace_param, trace_buffer);

  PrimFunc result(params, f->body, f->ret_type, buffer_map, f->attrs, f->span);
  result = WithAttr(std::move(result), "tl.provenance_trace_abi_version",
                    Integer(1));
  result = WithAttr(std::move(result), "tl.provenance_trace_extent",
                    trace_extent);
  result = WithAttr(std::move(result), "tl.provenance_trace_parameter_index",
                    Integer(parameter_index));
  result = WithAttr(std::move(result), "tl.provenance_trace_buffer_name",
                    trace_buffer_name);
  return result;
}

} // namespace

tvm::transform::Pass AttachProvenanceTraceBuffer(String trace_buffer_name,
                                                  Integer trace_extent) {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return AttachTraceBuffer(std::move(f), trace_buffer_name, trace_extent);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AttachProvenanceTraceBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AttachProvenanceTraceBuffer",
                        AttachProvenanceTraceBuffer);
}

} // namespace tl
} // namespace tvm

