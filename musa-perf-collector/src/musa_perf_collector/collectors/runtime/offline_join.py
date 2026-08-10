"""Fail-closed enrichment of runtime trace slots with offline TileOp metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"offline runtime association is missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"offline runtime association is missing required artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_paths(value: Any) -> tuple[Path, Path]:
    path = Path(str(value)).expanduser().resolve()
    if path.is_dir():
        return path, path / "metadata" / "lineage.json"
    if path.name != "lineage.json" or path.parent.name != "metadata":
        raise RuntimeError("lineage_artifact file must be metadata/lineage.json")
    return path.parent.parent, path


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if value is None or value in result:
            raise RuntimeError(f"offline runtime association requires unique {label}: {value}")
        result[value] = record
    return result


def _operation_details(op: dict[str, Any]) -> dict[str, Any]:
    keys = ("gemm", "reduction", "copy", "fill", "atomic", "scan", "opaque")
    return {key: op[key] for key in keys if key in op}


def associate(case: Any, trace: list[dict[str, Any]], deep_config: dict[str, Any]) -> dict[str, Any]:
    """Join exact runtime slots to their exact offline TileOp records.

    Requesting an artifact makes association part of the deep evidence gate.
    Absence of an artifact is explicit and never triggers heuristic matching.
    """
    artifact_value = deep_config.get("lineage_artifact")
    if artifact_value is None:
        return {"status": "not_requested", "reason": "no lineage_artifact was supplied",
                "tileops": [], "trace_records": []}

    root, lineage_path = _artifact_paths(artifact_value)
    manifest = _read_json(root / "manifest.json")
    lineage_artifact = _read_json(lineage_path)
    if manifest.get("analysis_status") != "complete" or manifest.get("device_execution") is not False:
        raise RuntimeError("offline runtime association requires a complete CPU-only offline artifact")
    if (lineage_artifact.get("status") != "compiler_exact_lineage" or
            not lineage_artifact.get("exact_runtime_instrumentation_allowed")):
        raise RuntimeError("offline runtime association requires compiler exact lineage")

    selection = (case.trace_metadata or {}).get("selection_evidence") or {}
    if selection.get("mode") != "offline_compiler_exact_lineage":
        raise RuntimeError("runtime trace does not carry offline compiler-exact selection evidence")
    evidence_path = Path(str(selection.get("lineage_artifact", ""))).resolve()
    if evidence_path != lineage_path:
        raise RuntimeError("runtime selection evidence references a different lineage artifact")
    for evidence_key, manifest_key in (("offline_tilelang_revision", "tilelang_revision"),
                                       ("offline_input_ir_sha256", "input_ir_sha256")):
        observed = selection.get(evidence_key)
        expected = manifest.get(manifest_key)
        if observed is not None and expected is not None and observed != expected:
            raise RuntimeError(f"runtime selection evidence {evidence_key} does not match offline manifest")

    tile_ops = _read_jsonl(root / "metadata" / "tile_ops.jsonl")
    metrics = _read_jsonl(root / "metrics" / "workload_per_invocation.jsonl")
    dag = _read_json(root / "metadata" / "dag.json")
    pipeline = _read_json(root / "metadata" / "pipeline.json")
    tile_by_id = _unique(tile_ops, "op_id", "offline TileOp op_id")
    tile_by_ordinal = _unique(tile_ops, "ordinal", "offline TileOp ordinal")
    metric_by_id = _unique(metrics, "op_id", "offline metric op_id")
    mapping_by_id = _unique(lineage_artifact.get("mappings", []), "source_op_id", "compiler lineage op_id")
    runtime_by_slot = _unique(case.lineage or [], "trace_slot", "runtime lineage trace_slot")
    trace_by_slot = _unique(trace, "trace_slot", "runtime trace_slot")
    if set(runtime_by_slot) != set(trace_by_slot):
        raise RuntimeError("runtime trace slots and runtime lineage slots differ")

    selected_ids: list[str] = []
    enriched_records: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    edges = list(dag.get("edges", []))
    pipeline_loops = {str(item.get("loop_id")): item for item in pipeline.get("pipeline_loops", [])}
    for slot in sorted(trace_by_slot):
        runtime_lineage = runtime_by_slot[slot]
        op_id = runtime_lineage.get("source_op_id")
        ordinal = runtime_lineage.get("source_ordinal")
        if not isinstance(op_id, str) or op_id not in tile_by_id or op_id not in mapping_by_id:
            raise RuntimeError(f"runtime slot {slot} lacks an exact offline source_op_id")
        op = tile_by_id[op_id]
        mapping = mapping_by_id[op_id]
        if (mapping.get("status") != "compiler_exact_lineage" or
                not mapping.get("eligible_for_exact_runtime_instrumentation")):
            raise RuntimeError(f"offline TileOp {op_id} is not individually eligible for exact instrumentation")
        if ordinal != op.get("ordinal") or ordinal != mapping.get("source_ordinal"):
            raise RuntimeError(f"runtime slot {slot} source ordinal conflicts with offline TileOp lineage")
        if mapping.get("source_kind") is not None and mapping.get("source_kind") != op.get("kind"):
            raise RuntimeError(f"offline TileOp {op_id} kind conflicts with compiler lineage")
        if tile_by_ordinal.get(ordinal, {}).get("op_id") != op_id:
            raise RuntimeError(f"offline ordinal {ordinal} does not resolve back to runtime op_id {op_id}")
        runtime_family = runtime_lineage.get("canonical_family")
        if runtime_family is not None and runtime_family != op.get("canonical_family"):
            raise RuntimeError(f"runtime slot {slot} family conflicts with offline TileOp metadata")
        metric = metric_by_id.get(op_id)
        if metric is None:
            raise RuntimeError(f"offline TileOp {op_id} is missing per-invocation metrics")
        if op_id not in contexts:
            loop_ids = [str(item.get("loop_id")) for item in op.get("loop_context", [])]
            contexts[op_id] = {
                "source_op_id": op_id,
                "source_ordinal": ordinal,
                "kind": op.get("kind"),
                "canonical_family": op.get("canonical_family"),
                "raw_op_name": op.get("raw_op_name"),
                "raw_ir": op.get("raw_ir"),
                "statement_path": op.get("statement_path"),
                "support_level": op.get("support_level"),
                "operation_details": _operation_details(op),
                "reads": op.get("reads", []),
                "writes": op.get("writes", []),
                "loop_context": op.get("loop_context", []),
                "condition_context": op.get("condition_context", []),
                "execution": op.get("execution", {}),
                "metrics_per_invocation": metric,
                "dependencies": {
                    "incoming": [edge for edge in edges if edge.get("to") == op_id],
                    "outgoing": [edge for edge in edges if edge.get("from") == op_id],
                },
                "pipeline_context": [pipeline_loops[loop_id] for loop_id in loop_ids if loop_id in pipeline_loops],
                "lineage": mapping,
            }
            selected_ids.append(op_id)
        enriched_records.append({**trace_by_slot[slot], "tileop": contexts[op_id]})

    requested_ids = {str(item) for item in deep_config.get("trace_op_ids", [])}
    if requested_ids != set(selected_ids):
        raise RuntimeError("runtime-associated TileOps do not exactly match requested trace_op_ids")
    evidence_selected = {
        (str(item.get("source_op_id")), item.get("source_ordinal"))
        for item in selection.get("selected", [])
    }
    associated = {(op_id, contexts[op_id]["source_ordinal"]) for op_id in selected_ids}
    if evidence_selected != associated:
        raise RuntimeError("runtime selection evidence does not exactly match associated TileOps")
    return {
        "status": "complete",
        "association_mode": "compiler_exact_op_id_ordinal_slot",
        "artifact": {
            "root": str(root), "lineage": str(lineage_path),
            "analysis_id": manifest.get("analysis_id"), "kernel_id": manifest.get("kernel_id"),
            "tilelang_revision": manifest.get("tilelang_revision"),
            "input_ir_sha256": manifest.get("input_ir_sha256"),
        },
        "selected_source_op_ids": selected_ids,
        "tileop_count": len(contexts),
        "trace_record_count": len(enriched_records),
        "tileops": [contexts[op_id] for op_id in selected_ids],
        "trace_records": enriched_records,
    }
