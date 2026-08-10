/*!
 * \file provenance_annotation.cc
 * \brief Stamp every source TileOp with a deterministic compiler provenance ordinal.
 *
 * The ordinal is attached to the Call annotation before pipeline expansion and
 * LowerTileOp. Cloned pipeline instances deliberately retain the same ordinal,
 * expressing a one-to-many lowering relation rather than losing source identity.
 */

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

static constexpr const char *kProvenanceOrdinal = "tl_provenance_ordinal";
static constexpr const char *kProvenanceLoopOrdinal =
    "tl_provenance_loop_ordinal";

class ProvenanceAnnotator : public StmtExprMutator {
public:
  static PrimFunc Annotate(PrimFunc f) {
    ProvenanceAnnotator annotator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = annotator.VisitStmt(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    // This deliberately uses the same preorder lexical numbering as the
    // offline collector's execution-context visitor.  A loop ordinal is
    // source provenance, not a lowered-loop heuristic: later passes may add
    // implementation loops, but only this source annotation is accepted by
    // deep iteration sampling.
    For annotated = GetRef<For>(op);
    auto *node = annotated.CopyOnWrite();
    ICHECK(!node->annotations.count(kProvenanceLoopOrdinal))
        << "source loop already has provenance ordinal";
    node->annotations.Set(kProvenanceLoopOrdinal, Integer(next_loop_ordinal_++));
    node->body = VisitStmt(op->body);
    return annotated;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    // A TileOp is an Evaluate(Call(...)) statement.  Do not stamp its nested
    // region/access calls: their presence is an implementation detail and
    // would make this ordinal differ from the source TileOp ordinal reported
    // by the offline collector.
    Evaluate evaluated =
        Downcast<Evaluate>(StmtExprMutator::VisitStmt_(op));
    const CallNode *call_node = evaluated->value.as<CallNode>();
    if (call_node == nullptr ||
        !ParseOperator(GetRef<Stmt>(op)).defined()) {
      return evaluated;
    }
    Call call = GetRef<Call>(call_node);
    if (call->annotations.count(kProvenanceOrdinal)) {
      return evaluated;
    }
    auto annotations = call->annotations;
    annotations.Set(kProvenanceOrdinal, Integer(next_ordinal_++));
    return Evaluate(
        Call(call->dtype, call->op, call->args, annotations, call->span));
  }

  int64_t next_ordinal_{0};
  int64_t next_loop_ordinal_{0};
};

} // namespace

tvm::transform::Pass ProvenanceAnnotation() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ProvenanceAnnotator::Annotate(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProvenanceAnnotation", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProvenanceAnnotation", ProvenanceAnnotation);
}

} // namespace tl
} // namespace tvm
