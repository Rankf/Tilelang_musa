"""Fail-closed resolution of runtime trace targets from offline lineage artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iteration_selector(deep_config: dict[str, Any]) -> dict[str, int] | None:
    has_loop = "trace_loop_ordinal" in deep_config
    has_iteration = "trace_iteration" in deep_config
    if has_loop != has_iteration:
        raise RuntimeError(
            "iteration sampling requires both trace_loop_ordinal and trace_iteration"
        )
    if not has_loop:
        return None
    selector = {
        "source_loop_ordinal": int(deep_config["trace_loop_ordinal"]),
        "relative_iteration": int(deep_config["trace_iteration"]),
    }
    if selector["source_loop_ordinal"] < 0 or selector["relative_iteration"] < 0:
        raise RuntimeError("iteration sampling requires non-negative source loop ordinal and iteration")
    return selector


def resolve(
    deep_config: dict[str, Any], *, workload_config: dict[str, Any] | None = None
) -> tuple[list[int], dict[str, Any] | None]:
    """Resolve ordered exact ordinals from explicit op_ids or use legacy ordinals.

    A lineage artifact is authoritative only when it itself declares complete
    compiler exact lineage.  Family-name matching is deliberately unsupported:
    it would reintroduce ambiguity for repeated TileOps.
    """
    selector = _iteration_selector(deep_config)
    artifact_path = deep_config.get("lineage_artifact")
    if artifact_path is None:
        return [int(item) for item in deep_config.get("trace_source_ordinals", [4])], (
            {"mode": "explicit_compiler_exact_workload_contract",
             "iteration_selector": selector} if selector is not None else None
        )
    path = Path(str(artifact_path)).expanduser().resolve()
    lineage_path = path / "metadata" / "lineage.json" if path.is_dir() else path
    if not lineage_path.is_file():
        raise RuntimeError(f"lineage artifact does not exist: {lineage_path}")
    artifact = json.loads(lineage_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {}
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"lineage artifact is missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if workload_config is not None:
            artifact_config = manifest.get("input_config", {})
            mismatches = {
                key: {"offline": artifact_config.get(key), "runtime": workload_config.get(key)}
                for key in ("M", "N", "K", "threads", "reduction")
                if key in artifact_config and key in workload_config and artifact_config.get(key) != workload_config.get(key)
            }
            if mismatches:
                raise RuntimeError(f"lineage artifact workload configuration mismatch: {mismatches}")
    if artifact.get("status") != "compiler_exact_lineage" or not artifact.get("exact_runtime_instrumentation_allowed"):
        raise RuntimeError("lineage artifact does not permit exact runtime instrumentation")
    requested = [str(item) for item in deep_config.get("trace_op_ids", [])]
    if not requested:
        raise RuntimeError("lineage_artifact requires explicit trace_op_ids; family matching is not allowed")
    mappings = {str(item.get("source_op_id")): item for item in artifact.get("mappings", [])}
    selected = []
    for op_id in requested:
        mapping = mappings.get(op_id)
        if mapping is None or not mapping.get("eligible_for_exact_runtime_instrumentation"):
            raise RuntimeError(f"requested op_id lacks exact lineage: {op_id}")
        ordinal = mapping.get("source_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise RuntimeError(f"requested op_id has invalid source ordinal: {op_id}")
        if selector is not None:
            matching_loop = [
                item for item in mapping.get("source_loop_context", [])
                if item.get("lexical_loop_ordinal") == selector["source_loop_ordinal"]
            ]
            if len(matching_loop) != 1:
                raise RuntimeError(
                    f"requested op_id is not enclosed by exact source loop ordinal "
                    f"{selector['source_loop_ordinal']}: {op_id}"
                )
            loop = matching_loop[0]
            if loop.get("thread_binding") is not None or "serial" not in str(loop.get("kind", "")).lower():
                raise RuntimeError("iteration sampling only permits a non-thread-bound serial source loop")
            extent = loop.get("extent_value")
            if isinstance(extent, int) and selector["relative_iteration"] >= extent:
                raise RuntimeError("selected iteration is outside the exact source loop extent")
        selected.append((ordinal, op_id, str(mapping.get("source_kind", "unknown"))))
    selected.sort()
    if len({item[0] for item in selected}) != len(selected):
        raise RuntimeError("requested op_ids resolve to duplicate source ordinals")
    return [item[0] for item in selected], {
        "mode": "offline_compiler_exact_lineage",
        "lineage_artifact": str(lineage_path),
        "offline_tilelang_revision": manifest.get("tilelang_revision"),
        "offline_input_ir_sha256": manifest.get("input_ir_sha256"),
        "selected": [{"source_ordinal": ordinal, "source_op_id": op_id, "source_kind": kind}
                     for ordinal, op_id, kind in selected],
        "iteration_selector": selector,
    }
