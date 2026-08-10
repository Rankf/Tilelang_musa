/*!
 * \file provenance_trace_injector.cc
 * \brief Bounded two-point clock64 tracing selected by compiler provenance.
 *
 * This pass is deliberately narrow: callers supply an existing int64 Trace
 * input buffer and one source TileOp ordinal already proven by
 * ProvenanceAnnotation.  It emits a begin/end pair of partition-leader stores
 * around each selected lowered AttrStmt (or one source envelope).  It never allocates memory,
 * inserts a barrier, or falls back to structural matching.
 */

#include "support/check.h"

#include <tvm/ir/cast.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

#include <array>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

static constexpr const char *kProvenanceOrdinal = "tl.provenance_ordinal";
static constexpr const char *kProvenanceLoopOrdinal =
    "tl_provenance_loop_ordinal";

using BlockCoordinate = std::array<int64_t, 3>;

class ThreadBindingFinder : public StmtExprVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      Capture(iv, op->value, "threadIdx.x", &thread_idx_[0], &thread_extent_[0]);
      Capture(iv, op->value, "threadIdx.y", &thread_idx_[1], &thread_extent_[1]);
      Capture(iv, op->value, "threadIdx.z", &thread_idx_[2], &thread_extent_[2]);
      Capture(iv, op->value, "blockIdx.x", &block_idx_[0], nullptr);
      Capture(iv, op->value, "blockIdx.y", &block_idx_[1], nullptr);
      Capture(iv, op->value, "blockIdx.z", &block_idx_[2], nullptr);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::array<Optional<Var>, 3> thread_idx_;
  std::array<Optional<PrimExpr>, 3> thread_extent_;
  std::array<Optional<Var>, 3> block_idx_;

private:
  static void Capture(const IterVar &iv, const PrimExpr &extent,
                      const char *thread_tag, Optional<Var> *destination,
                      Optional<PrimExpr> *extent_destination) {
    if (iv->thread_tag != thread_tag) {
      return;
    }
    ICHECK(!destination->defined() || destination->value().same_as(iv->var))
        << "ProvenanceTraceInjection found multiple " << thread_tag
        << " bindings";
    *destination = iv->var;
    if (extent_destination != nullptr) {
      ICHECK(!extent_destination->defined() ||
             StructuralEqual()(extent_destination->value(), extent))
          << "ProvenanceTraceInjection found inconsistent " << thread_tag
          << " extents";
      *extent_destination = extent;
    }
  }
};

class ProvenanceMatchCounter : public StmtVisitor {
public:
  explicit ProvenanceMatchCounter(int64_t selected_ordinal)
      : selected_ordinal_(selected_ordinal) {}

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kProvenanceOrdinal) {
      if (const auto *ordinal = op->value.as<IntImmNode>()) {
        if (ordinal->value == selected_ordinal_) {
          ++count_;
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  int64_t count_{0};

private:
  int64_t selected_ordinal_;
};

class ProvenanceTraceInjector : public StmtExprMutator {
public:
  ProvenanceTraceInjector(Buffer trace_buffer,
                          std::array<Optional<Var>, 3> thread_idx,
                          Optional<PrimExpr> thread_x_extent,
                          std::array<Optional<Var>, 3> block_idx,
                          std::vector<BlockCoordinate> selected_blocks,
                          int64_t selected_ordinal, int64_t trace_slot_offset,
                          int64_t selected_loop_ordinal,
                          int64_t selected_iteration, int64_t total_matches,
                          std::string trace_semantics)
      : trace_buffer_(std::move(trace_buffer)),
        thread_idx_(std::move(thread_idx)),
        thread_x_extent_(std::move(thread_x_extent)),
        block_idx_(std::move(block_idx)),
        selected_blocks_(std::move(selected_blocks)),
        selected_ordinal_(selected_ordinal), trace_slot_offset_(trace_slot_offset),
        selected_loop_ordinal_(selected_loop_ordinal),
        selected_iteration_(selected_iteration), total_matches_(total_matches),
        trace_semantics_(std::move(trace_semantics)) {}

  static PrimFunc Inject(PrimFunc f, String trace_buffer_name,
                         Integer selected_ordinal, Integer trace_slot_offset,
                         Integer selected_loop_ordinal,
                         Integer selected_iteration,
                         Array<Integer> selected_blocks_flat,
                         String trace_semantics) {
    ICHECK_GE(trace_slot_offset->value, 0)
        << "ProvenanceTraceInjection trace slot offset must be non-negative";
    const bool has_iteration_selector = selected_iteration->value >= 0;
    ICHECK_EQ(has_iteration_selector, selected_loop_ordinal->value >= 0)
        << "ProvenanceTraceInjection requires both source loop ordinal and "
           "non-negative relative iteration, or neither";
    ICHECK(trace_semantics == "single" || trace_semantics == "source_envelope" ||
           trace_semantics == "lowered_instance")
        << "ProvenanceTraceInjection semantics must be single, source_envelope, "
           "or lowered_instance";
    ICHECK(!selected_blocks_flat.empty())
        << "ProvenanceTraceInjection requires at least one selected block";
    ICHECK_EQ(selected_blocks_flat.size() % 3, 0U)
        << "ProvenanceTraceInjection block selector must be flattened x/y/z triples";
    std::vector<BlockCoordinate> selected_blocks;
    std::set<BlockCoordinate> unique_blocks;
    selected_blocks.reserve(selected_blocks_flat.size() / 3);
    for (size_t index = 0; index < selected_blocks_flat.size(); index += 3) {
      BlockCoordinate coordinate{selected_blocks_flat[index]->value,
                                 selected_blocks_flat[index + 1]->value,
                                 selected_blocks_flat[index + 2]->value};
      ICHECK_GE(coordinate[0], 0)
          << "ProvenanceTraceInjection block x must be non-negative";
      ICHECK_GE(coordinate[1], 0)
          << "ProvenanceTraceInjection block y must be non-negative";
      ICHECK_GE(coordinate[2], 0)
          << "ProvenanceTraceInjection block z must be non-negative";
      ICHECK(unique_blocks.insert(coordinate).second)
          << "ProvenanceTraceInjection selected blocks must be unique";
      selected_blocks.push_back(coordinate);
    }
    Optional<Buffer> trace_buffer;
    for (const auto &[param, buffer] : f->buffer_map) {
      if (buffer->name == trace_buffer_name) {
        trace_buffer = buffer;
        break;
      }
    }
    ICHECK(trace_buffer.defined())
        << "ProvenanceTraceInjection requires an existing input buffer named `"
        << trace_buffer_name << "`";
    ICHECK_EQ(trace_buffer.value()->dtype, DataType::Int(64))
        << "ProvenanceTraceInjection Trace buffer must have dtype int64";
    ICHECK_EQ(trace_buffer.value()->shape.size(), 1U)
        << "ProvenanceTraceInjection Trace buffer must be one-dimensional";
    ProvenanceMatchCounter counter(selected_ordinal->value);
    counter(f->body);
    if (trace_semantics == "single") {
      ICHECK_EQ(counter.count_, 1)
          << "ProvenanceTraceInjection single semantics requires exactly one "
             "lowered TileOp with ordinal "
          << selected_ordinal;
    } else {
      ICHECK_GT(counter.count_, 0)
          << "ProvenanceTraceInjection found no lowered TileOp with ordinal "
          << selected_ordinal;
      ICHECK(!has_iteration_selector)
          << "multi-instance provenance tracing does not yet combine with "
             "iteration selection";
    }
    const auto *extent = trace_buffer.value()->shape[0].as<IntImmNode>();
    const int64_t instance_factor =
        trace_semantics == "lowered_instance" ? counter.count_ : 1;
    const int64_t required_extent =
        trace_slot_offset->value + instance_factor *
            static_cast<int64_t>(selected_blocks.size()) * 2;
    ICHECK(extent && extent->value >= required_extent)
        << "ProvenanceTraceInjection Trace buffer is too small for requested trace slots";

    ThreadBindingFinder finder;
    finder(f->body);
    ICHECK(finder.thread_idx_[0].defined())
        << "ProvenanceTraceInjection requires a threadIdx.x binding";
    ICHECK(finder.thread_extent_[0].defined())
        << "ProvenanceTraceInjection requires a threadIdx.x extent";
    ProvenanceTraceInjector injector(trace_buffer.value(), finder.thread_idx_,
                                     finder.thread_extent_[0],
                                     finder.block_idx_, std::move(selected_blocks),
                                     selected_ordinal->value,
                                     trace_slot_offset->value,
                                     selected_loop_ordinal->value,
                                     selected_iteration->value, counter.count_,
                                     std::string(trace_semantics));
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = injector.VisitStmt(f->body);
    ICHECK_EQ(injector.selected_matches_, counter.count_)
        << "ProvenanceTraceInjection match count changed during mutation";
    return f;
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    loop_stack_.push_back(op);
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    loop_stack_.pop_back();
    return visited;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kWarpSpecializationScope) {
      const auto *branch = op->body.as<IfThenElseNode>();
      ICHECK(branch != nullptr && branch->else_case.defined())
          << "ProvenanceTraceInjection requires a two-branch warp "
             "specialization scope";
      Array<IntImm> partitions = Downcast<Array<IntImm>>(op->node);
      ICHECK_EQ(partitions.size(), 2U)
          << "warp specialization scope must describe producer and consumer "
             "thread extents";
      ICHECK_GT(partitions[0]->value, 0);
      ICHECK_GT(partitions[1]->value, 0);
      const auto *thread_extent = thread_x_extent_.value().as<IntImmNode>();
      ICHECK(thread_extent != nullptr &&
             thread_extent->value == partitions[0]->value + partitions[1]->value)
          << "warp specialization partition extents must sum to threadIdx.x extent";
      bool has_partition_condition = false;
      if (const auto *condition = branch->condition.as<GENode>()) {
        const auto *consumer_extent = condition->b.as<IntImmNode>();
        has_partition_condition =
            condition->a.same_as(thread_idx_[0].value()) &&
            consumer_extent != nullptr &&
            consumer_extent->value == partitions[1]->value;
      } else if (const auto *condition = branch->condition.as<LENode>()) {
        const auto *consumer_extent = condition->a.as<IntImmNode>();
        has_partition_condition =
            condition->b.same_as(thread_idx_[0].value()) &&
            consumer_extent != nullptr &&
            consumer_extent->value == partitions[1]->value;
      }
      ICHECK(has_partition_condition)
          << "warp specialization scope must branch on threadIdx.x >= "
             "consumer extent";

      Optional<PrimExpr> previous_leader = thread_x_leader_;
      // Producer threads occupy the upper partition and consumer threads the
      // lower partition.  Timestamping must use the first active lane in each
      // partition; a process-wide threadIdx.x == 0 guard silently misses the
      // producer-side clone.
      thread_x_leader_ = PrimExpr(partitions[1]);
      Stmt producer = VisitStmt(branch->then_case);
      thread_x_leader_ = make_const(thread_idx_[0].value().dtype(), 0);
      Stmt consumer = VisitStmt(branch->else_case.value());
      thread_x_leader_ = previous_leader;

      Stmt body = IfThenElse(branch->condition, producer, consumer, branch->span);
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    if (op->attr_key != kProvenanceOrdinal) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *ordinal = op->value.as<IntImmNode>();
    if (ordinal == nullptr || ordinal->value != selected_ordinal_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const int64_t instance_index = selected_matches_++;
    const ForNode *iteration_loop = nullptr;
    if (selected_iteration_ >= 0) {
      iteration_loop = FindSelectedSourceLoop();
    }
    Stmt lowered_body = VisitStmt(op->body);
    Stmt tagged_body = AttrStmt(op->node, op->attr_key, op->value, lowered_body);
    if (trace_semantics_ == "single") {
      return SeqStmt({TimestampStores(trace_slot_offset_, 0, iteration_loop),
                      tagged_body,
                      TimestampStores(trace_slot_offset_, 1, iteration_loop)});
    }
    if (trace_semantics_ == "source_envelope") {
      Array<Stmt> sequence;
      if (instance_index == 0) {
        sequence.push_back(TimestampStores(trace_slot_offset_, 0, nullptr));
      }
      sequence.push_back(tagged_body);
      if (instance_index + 1 == total_matches_) {
        sequence.push_back(TimestampStores(trace_slot_offset_, 1, nullptr));
      }
      return sequence.size() == 1 ? sequence[0] : SeqStmt(sequence);
    }
    const int64_t instance_offset =
        trace_slot_offset_ + instance_index *
            static_cast<int64_t>(selected_blocks_.size()) * 2;
    return SeqStmt({TimestampStores(instance_offset, 0, nullptr), tagged_body,
                    TimestampStores(instance_offset, 1, nullptr)});
  }

  const ForNode *FindSelectedSourceLoop() const {
    for (auto it = loop_stack_.rbegin(); it != loop_stack_.rend(); ++it) {
      const ForNode *loop = *it;
      auto annotation = loop->annotations.Get(kProvenanceLoopOrdinal);
      if (!annotation.has_value()) {
        continue;
      }
      const auto *ordinal = annotation.value().as<IntImmNode>();
      ICHECK(ordinal != nullptr)
          << "source loop provenance annotation must be an integer";
      if (ordinal->value != selected_loop_ordinal_) {
        continue;
      }
      ICHECK_EQ(loop->kind, ForKind::kSerial)
          << "iteration sampling only permits a source serial loop";
      ICHECK(!loop->thread_binding.defined())
          << "iteration sampling cannot select a thread-binding loop";
      if (const auto *extent = loop->extent.as<IntImmNode>()) {
        ICHECK_LT(selected_iteration_, extent->value)
            << "selected relative iteration is outside source loop extent";
      }
      return loop;
    }
    ICHECK(false) << "selected source loop ordinal " << selected_loop_ordinal_
                  << " does not enclose TileOp ordinal " << selected_ordinal_;
    return nullptr;
  }

  PrimExpr LeaderPredicate() const {
    PrimExpr leader = thread_x_leader_.defined()
                          ? thread_x_leader_.value()
                          : make_const(thread_idx_[0].value().dtype(), 0);
    leader = cast(thread_idx_[0].value().dtype(), leader);
    PrimExpr predicate = thread_idx_[0].value() == leader;
    for (size_t axis = 1; axis < thread_idx_.size(); ++axis) {
      if (thread_idx_[axis].defined()) {
        predicate = predicate && thread_idx_[axis].value() ==
            IntImm(thread_idx_[axis].value().dtype(), 0);
      }
    }
    return predicate;
  }

  PrimExpr BlockPredicate(const BlockCoordinate &coordinate) const {
    PrimExpr predicate = IntImm(DataType::Bool(), 1);
    for (size_t axis = 0; axis < block_idx_.size(); ++axis) {
      if (!block_idx_[axis].defined()) {
        ICHECK_EQ(coordinate[axis], 0)
            << "selected block uses an unavailable blockIdx axis";
        continue;
      }
      predicate = predicate && block_idx_[axis].value() ==
          IntImm(block_idx_[axis].value().dtype(), coordinate[axis]);
    }
    return predicate;
  }

  Stmt TimestampStores(int64_t base_slot, int64_t boundary,
                       const ForNode *iteration_loop) const {
    Array<Stmt> stores;
    for (size_t block_index = 0; block_index < selected_blocks_.size(); ++block_index) {
      PrimExpr clock = Call(DataType::Int(64), builtin::call_extern(),
                            {StringImm("clock64")});
      int64_t slot = base_slot + static_cast<int64_t>(block_index) * 2 + boundary;
      Stmt store = BufferStore(trace_buffer_, clock,
                               {IntImm(DataType::Int(32), slot)});
      PrimExpr predicate = LeaderPredicate() && BlockPredicate(selected_blocks_[block_index]);
      if (iteration_loop != nullptr) {
        PrimExpr selected_value = iteration_loop->min +
            IntImm(iteration_loop->loop_var.dtype(), selected_iteration_);
        predicate = predicate && iteration_loop->loop_var == selected_value;
      }
      stores.push_back(IfThenElse(predicate, store));
    }
    if (stores.size() == 1) {
      return stores[0];
    }
    return SeqStmt(stores);
  }

  Buffer trace_buffer_;
  int64_t selected_ordinal_;
  int64_t trace_slot_offset_;
  int64_t selected_loop_ordinal_;
  int64_t selected_iteration_;
  int64_t total_matches_;
  std::string trace_semantics_;
  std::array<Optional<Var>, 3> thread_idx_;
  Optional<PrimExpr> thread_x_extent_;
  std::array<Optional<Var>, 3> block_idx_;
  std::vector<BlockCoordinate> selected_blocks_;
  std::vector<const ForNode *> loop_stack_;
  Optional<PrimExpr> thread_x_leader_;
  int selected_matches_{0};
};

} // namespace

tvm::transform::Pass ProvenanceTraceInjection(String trace_buffer_name,
                                               Integer selected_ordinal,
                                               Integer trace_slot_offset,
                                               Integer selected_loop_ordinal,
                                               Integer selected_iteration,
                                               Array<Integer> selected_blocks_flat,
                                               String trace_semantics) {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ProvenanceTraceInjector::Inject(std::move(f), trace_buffer_name,
                                           selected_ordinal, trace_slot_offset,
                                           selected_loop_ordinal,
                                           selected_iteration,
                                           selected_blocks_flat,
                                           trace_semantics);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProvenanceTraceInjection", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProvenanceTraceInjection",
                        ProvenanceTraceInjection);
}

} // namespace tl
} // namespace tvm
