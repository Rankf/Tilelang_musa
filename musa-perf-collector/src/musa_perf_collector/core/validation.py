"""Offline-static completeness gates and lightweight artifact validation."""

from __future__ import annotations

from typing import Any


REQUIRED_SNAPSHOTS = {
    "S00_INPUT_TILE_IR", "S10_PIPELINE_PLANNED", "S20_PIPELINE_INJECTED",
    "S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S50_HOST_DEVICE_SPLIT", "S60_FINAL_DEVICE_IR",
}


def validate_offline_static(*, snapshots: dict[str, str], kernel_source: str, coverage: dict[str, Any], tile_ops: list[dict[str, Any]], dag: dict[str, Any], domains: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, op_id: str | None = None) -> None:
        value = {"severity": severity, "code": code, "message": message}
        if op_id:
            value["op_id"] = op_id
        diagnostics.append(value)

    missing = sorted(REQUIRED_SNAPSHOTS - set(snapshots))
    if missing:
        issue("blocker", "MISSING_REQUIRED_SNAPSHOT", f"missing snapshots: {', '.join(missing)}")
    if not kernel_source:
        issue("error", "EMPTY_MUSA_SOURCE", "generated MUSA C source is empty")
    if coverage["unexported_action_calls"]:
        issue("blocker", "UNEXPORTED_TILEOP_CALL", f"{coverage['unexported_action_calls']} action TileOps have no record")
    schema_valid = True
    required_record_fields = {"op_id", "raw_op_name", "canonical_family", "support_level", "reads", "writes", "execution"}
    if coverage.get("exported_action_records") != len(tile_ops) or coverage.get("observed_action_calls") != len(tile_ops) + coverage.get("unexported_action_calls", 0):
        schema_valid = False
        issue("error", "SCHEMA_VALIDATION_FAILED", "TileOp coverage counts do not match exported records")
    op_ids = [op.get("op_id") for op in tile_ops]
    if len(op_ids) != len(set(op_ids)):
        schema_valid = False
        issue("error", "SCHEMA_VALIDATION_FAILED", "TileOp IDs are not unique")
    for op in tile_ops:
        if not required_record_fields.issubset(op):
            schema_valid = False
            issue("error", "SCHEMA_VALIDATION_FAILED", "TileOp record misses required fields", op.get("op_id"))
            continue
        if op["support_level"] == "opaque":
            issue("blocker", "UNSUPPORTED_TILEOP_ADAPTER", f"{op['raw_op_name']} was retained as opaque", op["op_id"])
        if op["support_level"] == "invalid":
            issue("error", "INVALID_TILEOP_RECORD", f"{op['raw_op_name']} has invalid metadata", op["op_id"])
        if op["execution"]["calls_per_block_expr"] in (None, ""):
            issue("blocker", "UNKNOWN_EXECUTION_MULTIPLICITY", "missing per-block multiplicity", op["op_id"])
        for code in op.get("diagnostic_codes", []):
            if code == "UNREGISTERED_TILEOP_CALL":
                issue("blocker", code, f"{op['raw_op_name']} has no TileLang builder", op["op_id"])
            elif code == "TARGET_VARIANT_MISMATCH":
                issue("blocker", code, f"{op['raw_op_name']} is a non-MUSA explicit variant", op["op_id"])
    node_ids = {node["op_id"] for node in dag["nodes"]}
    for edge in dag["edges"]:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            issue("error", "DAG_INVALID_ENDPOINT", "DAG edge references a missing node")
    if dag["unknown_effect_barriers"]:
        issue("blocker", "UNKNOWN_ACCESS_EFFECT", "opaque actions prevent a complete dependence graph")
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in dag["edges"]:
        if edge["from"] in adjacency and edge["to"] in adjacency:
            adjacency[edge["from"]].append(edge["to"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        result = any(has_cycle(child) for child in adjacency[node])
        visiting.remove(node)
        visited.add(node)
        return result

    if any(has_cycle(node) for node in node_ids):
        issue("error", "DAG_CYCLE_IN_SINGLE_ITERATION", "single-iteration DAG contains a cycle")
    execution_domain_consistency = True
    for domain in (domains["per_block"], domains["per_kernel_launch"]):
        expected = "per_block" if domain is domains["per_block"] else "per_kernel_launch"
        if domain.get("execution_domain") != expected:
            execution_domain_consistency = False
            issue("error", "MIXED_EXECUTION_DOMAIN", f"expected {expected} aggregate")
        if domain["symbolic_multiplicity_ops"]:
            # Symbolic expressions are valid static output; this is information,
            # not a completeness failure.
            issue("info", "SYMBOLIC_EXECUTION_MULTIPLICITY", f"{domain['execution_domain']} includes symbolic counts")
        if domain["unknown_access_effect_ops"]:
            issue("blocker", "UNKNOWN_ACCESS_EFFECT", f"{domain['execution_domain']} has unknown access effects")
    has_error = any(item["severity"] == "error" for item in diagnostics)
    has_blocker = any(item["severity"] == "blocker" for item in diagnostics)
    gates = {
        "artifact_capture": not missing and bool(kernel_source),
        "tileop_lossless_export": coverage["unexported_action_calls"] == 0,
        "tileop_semantic_coverage": not any(op["support_level"] in {"opaque", "invalid"} for op in tile_ops),
        "execution_domain_consistency": execution_domain_consistency,
        "dag_valid": not dag["unknown_effect_barriers"] and not has_error,
        "schema_valid": schema_valid,
    }
    return {
        "profile": "offline_static_v1",
        "status": "failed" if has_error else ("partial" if has_blocker else "complete"),
        "complete_gate": gates,
        "coverage": coverage,
        "diagnostics": diagnostics,
        "expected_unknown_fields": ["physical_register_count", "S5000_cache_hit_rate", "S5000_effective_bandwidth", "absolute_latency_prediction"],
        # Runtime evidence is deliberately excluded from this CPU-only profile,
        # not globally unsupported. It is emitted under runtime_measurement_v1.
        "runtime_collectors_excluded_from_offline_profile": ["runtime_event", "torch_profiler", "mupti"],
        "device_execution": False,
    }
