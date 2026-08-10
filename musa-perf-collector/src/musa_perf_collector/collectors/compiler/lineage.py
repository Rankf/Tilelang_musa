"""Auditable source-TileOp to lowered-IR provenance evidence.

Structural anchors are retained as a conservative fallback for compilers that
do not carry identity through lowering.  A TileLang compiler provenance pass
can instead stamp every top-level source TileOp with
``tl_provenance_ordinal``.  Exact lineage is granted only when that ordinal is
complete and survives both TileOp lowering and final device lowering.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any


_LOWERED_ANCHORS = {"gemm": ("T.tl_gemm(", "_gemm_sqmma"), "cumsum": ("tl::CumSum",)}
_ATTR_STMT_RE = re.compile(r'"tl\.provenance_ordinal",\s*(\d+)')
_CALL_ANNOTATION_RE = re.compile(r'\btl_provenance_ordinal=(\d+)')


def _count_any(text: str, anchors: tuple[str, ...]) -> int:
    return sum(text.count(anchor) for anchor in anchors)


def _annotation_counts(text: str) -> Counter[int]:
    # Before LowerTileOp the printer shows Call annotations as
    # ``tl_provenance_ordinal=N``.  After lowering it shows AttrStmt values
    # as ``"tl.provenance_ordinal", N``.  Both are compiler-owned forms of
    # the same provenance token and are intentionally checked at their
    # corresponding pipeline stages.
    return Counter(
        int(value)
        for value in (
            _ATTR_STMT_RE.findall(text) + _CALL_ANNOTATION_RE.findall(text)
        )
    )


def _compiler_exact_lineage(
    tile_ops: list[dict[str, Any]], snapshots: dict[str, str]
) -> dict[str, Any] | None:
    """Return exact lineage only for a complete compiler-provenance chain."""
    source_ordinals = [op.get("ordinal") for op in tile_ops]
    if not source_ordinals or any(not isinstance(item, int) for item in source_ordinals):
        return None
    if len(set(source_ordinals)) != len(source_ordinals):
        return {
            "status": "compiler_provenance_incomplete",
            "mappings": [],
            "exact_runtime_instrumentation_allowed": False,
            "reason": "source TileOp ordinals are absent or not unique",
        }

    stage_counts = {
        stage: _annotation_counts(snapshots.get(stage, ""))
        for stage in ("S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S60_FINAL_DEVICE_IR")
    }
    if not any(stage_counts.values()):
        return None

    source_set = set(source_ordinals)
    missing = {
        stage: sorted(source_set - set(counts))
        for stage, counts in stage_counts.items()
    }
    unexpected = {
        stage: sorted(set(counts) - source_set)
        for stage, counts in stage_counts.items()
    }
    if any(missing.values()) or any(unexpected.values()):
        return {
            "status": "compiler_provenance_incomplete",
            "mappings": [],
            "exact_runtime_instrumentation_allowed": False,
            "reason": "compiler provenance ordinal coverage is incomplete",
            "missing_ordinals": missing,
            "unexpected_ordinals": unexpected,
        }

    lowered_counts = stage_counts["S40_TILE_OP_LOWERED"]
    final_counts = stage_counts["S60_FINAL_DEVICE_IR"]
    mappings = []
    for op in tile_ops:
        ordinal = op["ordinal"]
        source_loop_context = [
            {
                "lexical_loop_ordinal": int(str(loop["loop_id"]).rsplit("/", 1)[1]),
                "min": loop.get("min"),
                "extent": loop.get("extent"),
                "extent_value": loop.get("extent_value"),
                "kind": loop.get("kind"),
                "thread_binding": loop.get("thread_binding"),
            }
            for loop in op.get("loop_context", [])
            if str(loop.get("loop_id", "")).startswith("lexical_loop/")
        ]
        mappings.append({
            "source_op_id": op["op_id"],
            "source_kind": op.get("kind", "unknown"),
            "source_ordinal": ordinal,
            "source_snapshot": "S00_INPUT_TILE_IR",
            "target_snapshot": "S40_TILE_OP_LOWERED",
            "target_provenance_ordinal": ordinal,
            "source_loop_context": source_loop_context,
            "lowered_instance_count": lowered_counts[ordinal],
            "final_device_instance_count": final_counts[ordinal],
            "status": "compiler_exact_lineage",
            "confidence": "exact",
            "eligible_for_exact_runtime_instrumentation": True,
            "reason": "compiler provenance ordinal matches source metadata and survives final device lowering",
        })
    return {
        "status": "compiler_exact_lineage",
        "mappings": mappings,
        "exact_runtime_instrumentation_allowed": True,
        "reason": "compiler provenance ordinals are complete and preserved through S60_FINAL_DEVICE_IR",
        "compiler_provenance": {
            "annotation": "tl.provenance_ordinal",
            "stage_instance_counts": {stage: dict(sorted(counts.items())) for stage, counts in stage_counts.items()},
        },
    }


def structural_candidates(tile_ops: list[dict[str, Any]], snapshots: dict[str, str]) -> dict[str, Any]:
    """Return exact compiler lineage when proven; otherwise structural candidates."""
    exact = _compiler_exact_lineage(tile_ops, snapshots)
    if exact is not None:
        return exact

    lowered = snapshots.get("S40_TILE_OP_LOWERED", "")
    mappings: list[dict[str, Any]] = []
    source_by_kind: dict[str, list[dict[str, Any]]] = {}
    for op in tile_ops:
        source_by_kind.setdefault(str(op.get("kind")), []).append(op)
    for kind, ops in source_by_kind.items():
        anchors = _LOWERED_ANCHORS.get(kind)
        if anchors is None:
            continue
        count = _count_any(lowered, anchors)
        for op in ops:
            unique = len(ops) == 1 and count == 1
            mappings.append({
                "source_op_id": op["op_id"], "source_kind": kind,
                "source_snapshot": "S00_INPUT_TILE_IR", "target_snapshot": "S40_TILE_OP_LOWERED",
                "target_anchors": list(anchors), "target_anchor_count": count,
                "status": "structural_candidate", "confidence": "high" if unique else "ambiguous",
                "eligible_for_exact_runtime_instrumentation": False,
                "reason": ("unique structural anchor, but compiler does not preserve source op_id provenance"
                           if unique else "source or lowered anchor cardinality is not one-to-one"),
            })
    return {"status": "structural_candidates_only", "mappings": mappings,
            "exact_runtime_instrumentation_allowed": False,
            "reason": "No compiler pass currently carries source op_id into lowered IR; candidates cannot satisfy exact lineage."}
