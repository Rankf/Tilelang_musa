"""Compiler-exact online instrumentation plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import OnlineConfig


PLAN_SCHEMA = "online-instrumentation-plan-v1"
TRACE_BUFFER_NAME = "__musa_perf_trace"


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _tilelang_revision() -> str | None:
    try:
        import tilelang
        root = Path(tilelang.__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    if not path.is_dir():
        raise RuntimeError("online exact lineage requires an offline artifact directory with manifest")
    lineage = path / "metadata" / "lineage.json"
    manifest = path / "manifest.json"
    if not lineage.is_file() or not manifest.is_file():
        raise RuntimeError("lineage artifact must contain metadata/lineage.json and manifest.json")
    return lineage, manifest


def _selected_blocks(
    selector: dict[str, Any] | None, launch_context: dict[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    resolved = selector or {"indices": [0]}
    if set(resolved) - {"indices"} or "indices" not in resolved:
        raise RuntimeError("P6 MVP online block selection requires exact linear indices")
    indices = [int(value) for value in resolved["indices"]]
    if not indices or len(set(indices)) != len(indices) or any(value < 0 for value in indices):
        raise RuntimeError("online block indices must be non-empty, unique, and non-negative")
    axes = launch_context.get("block_axes", {})
    extents = tuple(axes.get(f"blockIdx.{axis}", 1) for axis in "xyz")
    if any(not isinstance(value, int) or value <= 0 for value in extents):
        raise RuntimeError("online block selection requires a static positive compiler grid")
    candidate_count = extents[0] * extents[1] * extents[2]
    if any(value >= candidate_count for value in indices):
        raise RuntimeError("online block selector is outside the compiler-observed static grid")
    coordinates = []
    for value in indices:
        x = value % extents[0]
        y = (value // extents[0]) % extents[1]
        z = value // (extents[0] * extents[1])
        coordinates.append((x, y, z))
    return tuple(coordinates)


@dataclass(frozen=True)
class InstrumentationPlan:
    baseline_primfunc_sha256: str
    tilelang_revision: str
    target: str
    execution_backend: str
    source_op_ids: tuple[str, ...]
    source_ordinals: tuple[int, ...]
    source_kinds: tuple[str, ...]
    lowered_instance_counts: tuple[int, ...]
    lowering_semantics: str
    launch_context: dict[str, Any]
    selected_blocks: tuple[tuple[int, int, int], ...]
    trace_extent: int
    normalized_out_idx: tuple[int, ...]
    slot_layout: tuple[dict[str, Any], ...]
    plan_sha256: str
    lineage_artifact: str

    @property
    def trace_bytes(self) -> int:
        return self.trace_extent * 8

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "baseline_primfunc_sha256": self.baseline_primfunc_sha256,
            "tilelang_revision": self.tilelang_revision,
            "target": self.target,
            "execution_backend": self.execution_backend,
            "source_op_ids": list(self.source_op_ids),
            "source_ordinals": list(self.source_ordinals),
            "source_kinds": list(self.source_kinds),
            "lowered_instance_counts": list(self.lowered_instance_counts),
            "lowering_semantics": self.lowering_semantics,
            "launch_context": self.launch_context,
            "selected_blocks": [list(value) for value in self.selected_blocks],
            "trace_extent": self.trace_extent,
            "trace_bytes": self.trace_bytes,
            "normalized_out_idx": list(self.normalized_out_idx),
            "trace_buffer_name": TRACE_BUFFER_NAME,
            "slot_layout": list(self.slot_layout),
            "lineage_artifact": self.lineage_artifact,
            "plan_sha256": self.plan_sha256,
        }


def build_plan(config: OnlineConfig, request: Any, *, target: str, execution_backend: str) -> InstrumentationPlan:
    """Resolve op IDs to one-to-one compiler ordinals and freeze every slot."""
    if config.lineage_artifact is None:
        raise RuntimeError("compiler-exact online instrumentation requires lineage_artifact")
    lineage_path, manifest_path = _artifact_paths(config.lineage_artifact)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel_metadata_path = config.lineage_artifact / "metadata" / "kernel.json"
    if not kernel_metadata_path.is_file():
        raise RuntimeError("lineage artifact is missing compiler-observed kernel launch metadata")
    kernel_metadata = json.loads(kernel_metadata_path.read_text(encoding="utf-8"))
    launch_context = kernel_metadata.get("launch_context")
    if not isinstance(launch_context, dict):
        raise RuntimeError("lineage artifact has no compiler-observed launch context")
    if lineage.get("status") != "compiler_exact_lineage" or not lineage.get(
        "exact_runtime_instrumentation_allowed"):
        raise RuntimeError("lineage artifact does not permit compiler-exact instrumentation")
    if manifest.get("input_ir_sha256") != request.prim_func_sha256:
        raise RuntimeError(
            "lineage artifact input IR hash does not match the online PrimFunc")
    current_revision = _tilelang_revision()
    offline_revision = manifest.get("tilelang_revision")
    if not current_revision or offline_revision != current_revision:
        raise RuntimeError(
            f"lineage artifact TileLang revision mismatch: offline={offline_revision}, online={current_revision}")

    mappings = {str(item.get("source_op_id")): item for item in lineage.get("mappings", [])}
    selected: list[tuple[int, str, str, int]] = []
    for op_id in config.trace_op_ids:
        mapping = mappings.get(op_id)
        if mapping is None or not mapping.get("eligible_for_exact_runtime_instrumentation"):
            raise RuntimeError(f"requested op_id lacks compiler-exact lineage: {op_id}")
        ordinal = mapping.get("source_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise RuntimeError(f"requested op_id has invalid source ordinal: {op_id}")
        lowered_count = mapping.get("lowered_instance_count")
        final_count = mapping.get("final_device_instance_count")
        if not isinstance(lowered_count, int) or lowered_count <= 0 or final_count != lowered_count:
            raise RuntimeError(f"requested op_id lacks a stable lowered instance count: {op_id}")
        if config.lowering_semantics == "single" and lowered_count != 1:
            raise RuntimeError(f"requested op_id is not a one-to-one lowered instance: {op_id}")
        selected.append((ordinal, op_id, str(mapping.get("source_kind", "unknown")), lowered_count))
    selected.sort()
    if not selected or len({item[0] for item in selected}) != len(selected):
        raise RuntimeError("selected online TileOps must resolve to unique source ordinals")
    blocks = _selected_blocks(config.block_selector, launch_context)
    instance_total = (
        sum(item[3] for item in selected)
        if config.lowering_semantics == "lowered_instance"
        else len(selected))
    trace_extent = instance_total * len(blocks) * 2
    if trace_extent > config.budget.max_trace_records or trace_extent * 8 > config.budget.max_trace_bytes:
        raise RuntimeError("online instrumentation plan exceeds trace record/byte budget")

    slot_layout: list[dict[str, Any]] = []
    next_slot = 0
    for ordinal, op_id, kind, instance_count in selected:
        instance_indices: tuple[int | None, ...] = (
            tuple(range(instance_count))
            if config.lowering_semantics == "lowered_instance" else (None,))
        for instance_index in instance_indices:
            for block in blocks:
                for boundary in ("begin", "end"):
                    slot = next_slot
                    next_slot += 1
                    slot_layout.append({
                        "slot": slot,
                        "op_id": op_id,
                        "ordinal": ordinal,
                        "source_kind": kind,
                        "lowered_instance_index": instance_index,
                        "lowering_semantics": config.lowering_semantics,
                        "block": list(block),
                        "boundary": boundary,
                    })
    base = {
        "schema": PLAN_SCHEMA,
        "baseline_primfunc_sha256": request.prim_func_sha256,
        "tilelang_revision": current_revision,
        "target": target,
        "execution_backend": execution_backend,
        "source_op_ids": [item[1] for item in selected],
        "source_ordinals": [item[0] for item in selected],
        "source_kinds": [item[2] for item in selected],
        "lowered_instance_counts": [item[3] for item in selected],
        "lowering_semantics": config.lowering_semantics,
        "launch_context": launch_context,
        "selected_blocks": [list(value) for value in blocks],
        "trace_extent": trace_extent,
        "trace_bytes": trace_extent * 8,
        "normalized_out_idx": list(request.normalized_out_idx),
        "trace_buffer_name": TRACE_BUFFER_NAME,
        "slot_layout": slot_layout,
        "lineage_artifact": str(lineage_path),
    }
    digest = _canonical_hash(base)
    return InstrumentationPlan(
        baseline_primfunc_sha256=request.prim_func_sha256,
        tilelang_revision=current_revision,
        target=target,
        execution_backend=execution_backend,
        source_op_ids=tuple(item[1] for item in selected),
        source_ordinals=tuple(item[0] for item in selected),
        source_kinds=tuple(item[2] for item in selected),
        lowered_instance_counts=tuple(item[3] for item in selected),
        lowering_semantics=config.lowering_semantics,
        launch_context=launch_context,
        selected_blocks=blocks,
        trace_extent=trace_extent,
        normalized_out_idx=tuple(request.normalized_out_idx),
        slot_layout=tuple(slot_layout),
        plan_sha256=digest,
        lineage_artifact=str(lineage_path),
    )


def prepare_instrumented_primfunc(request: Any, plan: InstrumentationPlan) -> Any:
    """Clone the source function through the native Trace ABI pass."""
    import tilelang
    from tilelang import tvm

    attrs = getattr(request.prim_func, "attrs", None)
    symbol = str(attrs["global_symbol"]) if attrs and "global_symbol" in attrs else "main"
    module = tvm.IRModule({symbol: request.prim_func})
    transformed = tilelang.transform.AttachProvenanceTraceBuffer(
        TRACE_BUFFER_NAME, plan.trace_extent)(module)
    instrumented = transformed[symbol]
    # The source eager PrimFunc commonly carries tilelang_out_idx=[-1].  Once
    # Trace is appended, -1 would identify Trace.  Rewrite this attr only on
    # the clone; the baseline function and its cache identity remain untouched.
    return instrumented.with_attr(
        "tilelang_out_idx", list(plan.normalized_out_idx))


def instrumented_pass_configs(request: Any, plan: InstrumentationPlan) -> dict[str, Any]:
    result = dict(request.pass_configs or {})
    result.update({
        "tl.provenance_trace_enable": True,
        "tl.provenance_trace_ordinals": list(plan.source_ordinals),
        "tl.provenance_trace_buffer": TRACE_BUFFER_NAME,
        "tl.provenance_trace_blocks": [value for block in plan.selected_blocks for value in block],
        "tl.provenance_trace_semantics": plan.lowering_semantics,
        "tl.provenance_trace_instance_counts": list(plan.lowered_instance_counts),
    })
    return result


def load_selected_tileop_context(plan: InstrumentationPlan) -> dict[str, dict[str, Any]]:
    """Load exact source metadata for trace enrichment and verify the join."""
    lineage_path = Path(plan.lineage_artifact)
    tileops_path = lineage_path.parent / "tile_ops.jsonl"
    if not tileops_path.is_file():
        raise RuntimeError(f"exact lineage artifact is missing TileOp metadata: {tileops_path}")
    requested = set(plan.source_op_ids)
    records: dict[str, dict[str, Any]] = {}
    for line in tileops_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        op_id = str(record.get("op_id"))
        if op_id in requested:
            if op_id in records:
                raise RuntimeError(f"duplicate TileOp metadata for {op_id}")
            records[op_id] = record
    if set(records) != requested:
        raise RuntimeError(f"missing TileOp metadata for {sorted(requested - set(records))}")
    expected = dict(zip(plan.source_op_ids, plan.source_ordinals))
    for op_id, record in records.items():
        if record.get("ordinal") != expected[op_id]:
            raise RuntimeError(f"TileOp metadata ordinal mismatch for {op_id}")
    return records
