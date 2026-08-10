"""Conservative DAG and execution-domain static workload analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.evidence import MODEL_DERIVED_UNVALIDATED, SEMANTIC_DERIVED, UNKNOWN
from ...core.ids import stable_id


def _regions(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _scale(value: int, factor: int | None) -> int | None:
    return value * factor if isinstance(factor, int) else None


def _combined_expression(numeric: int, terms: list[str]) -> int | str:
    if not terms:
        return numeric
    return " + ".join((([str(numeric)] if numeric else []) + terms))


def _family(op: dict[str, Any]) -> str:
    return str(op.get("canonical_family", op.get("kind", "unknown")))


def _support_level(op: dict[str, Any]) -> str:
    return str(op.get("support_level", "structured"))


def _execution(op: dict[str, Any]) -> dict[str, Any]:
    return op.get("execution", {
        "calls_per_block_value": 1,
        "calls_per_block_expr": "1",
        "calls_per_kernel_value": 1,
        "calls_per_kernel_expr": "1",
    })


def build_dag(tile_ops: list[dict[str, Any]]) -> dict[str, Any]:
    writers: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    unknown_effect_barriers: list[str] = []
    for op in tile_ops:
        if not isinstance(op.get("reads"), list) or not isinstance(op.get("writes"), list):
            unknown_effect_barriers.append(op["op_id"])
            continue
        for region in _regions(op.get("reads")):
            producer = writers.get(region["buffer"])
            if producer:
                edges.append({
                    "edge_id": stable_id("edge", [producer, op["op_id"], region["buffer"], "data"]),
                    "from": producer,
                    "to": op["op_id"],
                    "buffer": region["buffer"],
                    "kind": "data",
                    "confidence": "may_depend_buffer_match",
                })
        for region in _regions(op.get("writes")):
            writers[region["buffer"]] = op["op_id"]
    return {
        "nodes": [{"op_id": op["op_id"], "kind": op["kind"], "family": _family(op), "ordinal": op["ordinal"], "loop_context": op.get("loop_context", [])} for op in tile_ops],
        "edges": edges,
        "unknown_effect_barriers": unknown_effect_barriers,
        "scope": "single input Tile IR snapshot; buffer-name matching is conservative and loop-carried dependencies are not expanded in v0",
        "evidence_level": SEMANTIC_DERIVED,
    }


def _per_invocation_metric(op: dict[str, Any]) -> dict[str, Any]:
    reads = op.get("reads")
    writes = op.get("writes")
    unknown_access = not isinstance(reads, list) or not isinstance(writes, list)
    logical_read_bytes = sum(region.get("logical_bytes", 0) for region in _regions(reads) if isinstance(region.get("logical_bytes"), int))
    logical_write_bytes = sum(region.get("logical_bytes", 0) for region in _regions(writes) if isinstance(region.get("logical_bytes"), int))
    unknown_byte_fields = sum(
        not isinstance(region.get("logical_bytes"), int)
        for region in [*_regions(reads), *_regions(writes)]
    ) + (1 if unknown_access else 0)
    flops = op.get("gemm", {}).get("flops", 0)
    if not isinstance(flops, int):
        flops = 0
    transfers: dict[str, int] = defaultdict(int)
    ambiguous: list[dict[str, Any]] = []
    if _family(op) in {"copy", "transform_copy"} and len(_regions(reads)) == 1 and len(_regions(writes)) == 1:
        source, destination = _regions(reads)[0], _regions(writes)[0]
        source_bytes, destination_bytes = source.get("logical_bytes"), destination.get("logical_bytes")
        if isinstance(source_bytes, int) and source_bytes == destination_bytes:
            transfers[f"{source['scope']}->{destination['scope']}"] += source_bytes
        else:
            ambiguous.append({
                "from_scope": source.get("scope"), "to_scope": destination.get("scope"),
                "source_logical_bytes": source_bytes, "destination_logical_bytes": destination_bytes,
                "reason": "endpoint logical byte counts differ or are symbolic",
            })
    return {
        "op_id": op["op_id"],
        "kind": op["kind"],
        "canonical_family": _family(op),
        "execution_domain": "per_invocation",
        "flops": flops,
        "flops_dense_equivalent": op.get("gemm_sparse", {}).get("flops_dense_equivalent"),
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_write_bytes,
        "logical_total_bytes": logical_read_bytes + logical_write_bytes,
        "scope_transfer_bytes": dict(transfers),
        "ambiguous_scope_transfers": ambiguous,
        "unknown_access_effect": unknown_access,
        "unknown_byte_fields": unknown_byte_fields,
        "support_level": _support_level(op),
        "calls_per_block_value": _execution(op)["calls_per_block_value"],
        "calls_per_block_expr": _execution(op)["calls_per_block_expr"],
        "calls_per_kernel_value": _execution(op)["calls_per_kernel_value"],
        "calls_per_kernel_expr": _execution(op)["calls_per_kernel_expr"],
        "evidence_level": SEMANTIC_DERIVED,
    }


def workload_per_invocation(tile_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_per_invocation_metric(op) for op in tile_ops]


def aggregate_workload(per_op: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    if domain not in {"per_block", "per_kernel_launch"}:
        raise ValueError(f"unsupported aggregate domain: {domain}")
    value_key = "calls_per_block_value" if domain == "per_block" else "calls_per_kernel_value"
    expression_key = "calls_per_block_expr" if domain == "per_block" else "calls_per_kernel_expr"
    result: dict[str, Any] = {
        "execution_domain": domain,
        "flops": 0,
        "logical_read_bytes": 0,
        "logical_write_bytes": 0,
        "logical_total_bytes": 0,
        "scope_transfer_bytes": defaultdict(int),
        "ambiguous_scope_transfers": [],
        "unknown_byte_fields": 0,
        "unknown_access_effect_ops": [],
        "symbolic_multiplicity_ops": [],
        "symbolic_terms": {"flops": [], "logical_read_bytes": [], "logical_write_bytes": [], "logical_total_bytes": []},
        "symbolic_scope_transfer_terms": defaultdict(list),
        "evidence_level": SEMANTIC_DERIVED,
    }
    for metric in per_op:
        count = metric[value_key]
        if not isinstance(count, int):
            result["symbolic_multiplicity_ops"].append({"op_id": metric["op_id"], "expr": metric[expression_key]})
            for field in ("flops", "logical_read_bytes", "logical_write_bytes", "logical_total_bytes"):
                if metric[field]:
                    result["symbolic_terms"][field].append(f"({metric[field]}) * ({metric[expression_key]})")
            for scope, bytes_ in metric["scope_transfer_bytes"].items():
                result["symbolic_scope_transfer_terms"][scope].append(f"({bytes_}) * ({metric[expression_key]})")
            result["unknown_byte_fields"] += metric["unknown_byte_fields"]
            if metric["unknown_access_effect"]:
                result["unknown_access_effect_ops"].append(metric["op_id"])
            continue
        for field in ("flops", "logical_read_bytes", "logical_write_bytes", "logical_total_bytes"):
            result[field] += _scale(metric[field], count) or 0
        for scope, bytes_ in metric["scope_transfer_bytes"].items():
            result["scope_transfer_bytes"][scope] += _scale(bytes_, count) or 0
        result["ambiguous_scope_transfers"].extend(metric["ambiguous_scope_transfers"])
        result["unknown_byte_fields"] += metric["unknown_byte_fields"]
        if metric["unknown_access_effect"]:
            result["unknown_access_effect_ops"].append(metric["op_id"])
    for field, terms in result["symbolic_terms"].items():
        result[field] = _combined_expression(result[field], terms)
    scope_transfers: dict[str, int | str] = {}
    for scope in set(result["scope_transfer_bytes"]) | set(result["symbolic_scope_transfer_terms"]):
        scope_transfers[scope] = _combined_expression(result["scope_transfer_bytes"].get(scope, 0), result["symbolic_scope_transfer_terms"].get(scope, []))
    result["scope_transfer_bytes"] = scope_transfers
    result.pop("symbolic_scope_transfer_terms")
    total = result["logical_total_bytes"]
    global_bytes = sum(
        bytes_ for scope, bytes_ in result["scope_transfer_bytes"].items()
        if (scope.startswith("global->") or scope.endswith("->global")) and isinstance(bytes_, int)
    )
    # Endpoint transfers do not include a dtype-converting store. Add known
    # global endpoint accesses from the per-op metrics for a stable logical AI.
    global_access_bytes = 0
    for metric in per_op:
        count = metric[value_key]
        if not isinstance(count, int):
            continue
        # This field is intentionally conservative: only regions known by the
        # adapter contribute, and is recomputed below by callers if needed.
        global_access_bytes += 0
    result["arithmetic_intensity_all_logical"] = result["flops"] / total if isinstance(result["flops"], int) and isinstance(total, int) and total else (f"({result['flops']}) / ({total})" if total else None)
    result["arithmetic_intensity_global_logical"] = None
    result["global_endpoint_transfer_bytes"] = global_bytes
    return result


def attach_global_access_bytes(per_op: list[dict[str, Any]], tile_ops: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    """Attach global read/write logical bytes without confusing them with transfers."""
    global_by_op: dict[str, int] = {}
    for op in tile_ops:
        total = sum(region.get("logical_bytes", 0) for region in [*_regions(op.get("reads")), *_regions(op.get("writes"))] if region.get("scope") == "global" and isinstance(region.get("logical_bytes"), int))
        global_by_op[op["op_id"]] = total
    for aggregate in aggregates:
        key = "calls_per_block_value" if aggregate["execution_domain"] == "per_block" else "calls_per_kernel_value"
        numeric = sum(global_by_op.get(metric["op_id"], 0) * metric[key] for metric in per_op if isinstance(metric[key], int))
        symbolic = [f"({global_by_op.get(metric['op_id'], 0)}) * ({metric[key.replace('_value', '_expr')]})" for metric in per_op if not isinstance(metric[key], int) and global_by_op.get(metric["op_id"], 0)]
        global_bytes = _combined_expression(numeric, symbolic)
        aggregate["global_logical_bytes"] = global_bytes
        aggregate["arithmetic_intensity_global_logical"] = aggregate["flops"] / global_bytes if isinstance(aggregate["flops"], int) and isinstance(global_bytes, int) and global_bytes else (f"({aggregate['flops']}) / ({global_bytes})" if global_bytes else None)


def workload_domains(tile_ops: list[dict[str, Any]]) -> dict[str, Any]:
    per_invocation = workload_per_invocation(tile_ops)
    per_block = aggregate_workload(per_invocation, "per_block")
    per_kernel = aggregate_workload(per_invocation, "per_kernel_launch")
    attach_global_access_bytes(per_invocation, tile_ops, [per_block, per_kernel])
    return {
        "per_invocation": per_invocation,
        "per_block": per_block,
        "per_kernel_launch": per_kernel,
        "evidence_level": SEMANTIC_DERIVED,
    }


def workload_demand(tile_ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility index; mixed execution-domain numbers are intentionally gone."""
    domains = workload_domains(tile_ops)
    return {
        "status": "execution_domain_index",
        "deprecated_reason": "mixed per-representative-tile metrics were removed",
        "per_block": domains["per_block"],
        "per_kernel_launch": domains["per_kernel_launch"],
        "evidence_level": SEMANTIC_DERIVED,
    }


def reuse_distance(tile_ops: list[dict[str, Any]], capacities: list[int]) -> dict[str, Any]:
    def loop_identity(loop: dict[str, Any]) -> str:
        return str(loop.get("loop_id", f"{loop.get('var')}:{loop.get('extent')}"))

    def expand_segment(ops: list[dict[str, Any]], depth: int, coordinates: dict[str, int], result: list[tuple[dict[str, Any], dict[str, int]]]) -> bool:
        """Expand contiguous lexical loop groups for bounded numeric schedules."""
        index = 0
        while index < len(ops):
            op = ops[index]
            if op.get("condition_context"):
                return False
            context = op.get("loop_context", [])
            if len(context) <= depth:
                result.append((op, dict(coordinates)))
                index += 1
                continue
            loop = context[depth]
            identity = loop_identity(loop)
            group: list[dict[str, Any]] = []
            while index < len(ops):
                candidate_context = ops[index].get("loop_context", [])
                if len(candidate_context) <= depth or loop_identity(candidate_context[depth]) != identity:
                    break
                group.append(ops[index])
                index += 1
            extent = loop.get("extent_value")
            minimum = loop.get("min")
            try:
                start = int(minimum)
            except (TypeError, ValueError):
                return False
            if not isinstance(extent, int) or extent < 0:
                return False
            for value in range(start, start + extent):
                coordinates[loop["var"]] = value
                if not expand_segment(group, depth + 1, coordinates, result):
                    return False
            coordinates.pop(loop["var"], None)
        return True

    def region_key(region: dict[str, Any], coordinates: dict[str, int]) -> str:
        def substitute(value: Any) -> str:
            text = str(value)
            for variable, numeric in coordinates.items():
                text = text.replace(variable, str(numeric))
            return text
        return f"{region['buffer']}:{[substitute(index) for index in region['indices']]}:{[substitute(shape) for shape in region['shape']]}"

    expanded: list[tuple[dict[str, Any], dict[str, int]]] = []
    expansion_ok = expand_segment(tile_ops, 0, {}, expanded)
    # A bounded schedule prevents an enormous compile-time artifact for large
    # static loops.  It is safe to omit the curve rather than silently sample.
    if len(expanded) > 100_000:
        expansion_ok = False
        expanded = []
    last_seen: dict[str, int] = {}
    distances: list[dict[str, Any]] = []
    global_reads: list[tuple[int, dict[str, Any], dict[str, int]]] = []
    for ordinal, (op, coordinates) in enumerate(expanded):
        seen_in_op: set[str] = set()
        for region in [*_regions(op.get("reads")), *_regions(op.get("writes"))]:
            key = region_key(region, coordinates)
            if key in last_seen and key not in seen_in_op:
                distances.append({"reuse_key": key, "from_event": last_seen[key], "to_event": ordinal, "static_op_id": op["op_id"], "iteration_coordinates": coordinates, "op_distance": ordinal - last_seen[key], "logical_bytes": region.get("logical_bytes"), "scope": region.get("scope")})
            seen_in_op.add(key)
        for region in [*_regions(op.get("reads")), *_regions(op.get("writes"))]:
            last_seen[region_key(region, coordinates)] = ordinal
        for region in _regions(op.get("reads")):
            if region.get("scope") == "global":
                global_reads.append((ordinal, region, coordinates))
    capacity_sweep: list[dict[str, Any]] = []
    if expansion_ok:
        compulsory_global_reads = sum(region.get("logical_bytes", 0) for _, region, _ in global_reads if isinstance(region.get("logical_bytes"), int))
        for capacity in capacities:
            last_global: dict[str, int] = {}
            potential_reused = 0
            for event, region, coordinates in global_reads:
                key = region_key(region, coordinates)
                if key in last_global and isinstance(region.get("logical_bytes"), int) and region["logical_bytes"] <= capacity:
                    potential_reused += region["logical_bytes"]
                last_global[key] = event
            capacity_sweep.append({
                "assumed_effective_capacity_bytes": capacity,
                "potential_reused_global_read_bytes": potential_reused,
                "potential_global_read_traffic_bytes": max(0, compulsory_global_reads - potential_reused),
                "evidence_level": MODEL_DERIVED_UNVALIDATED,
            })
    return {
        "reuse_events": distances,
        "capacity_sweep": capacity_sweep,
        "status": "complete_bounded_static_schedule" if expansion_ok else "partial",
        "expanded_event_count_per_block": len(expanded) if expansion_ok else None,
        "limitations": [
            "capacity sweep is a bounded static schedule model, not a measured hardware cache result",
            "does not model S5000 cache capacity, associativity, or replacement policy",
        ],
        "evidence_level": MODEL_DERIVED_UNVALIDATED if distances else UNKNOWN,
    }


def upper_bound_formula(demand: dict[str, Any]) -> dict[str, Any]:
    return {
        "variables": {"P_compute_effective": "effective compute throughput supplied only after calibration", "BW_scope_effective": "effective bandwidth for each observed scope transfer supplied only after calibration"},
        "formulae": ["T_compute >= FLOPs / P_compute_effective", "T_transfer(scope) >= logical_bytes(scope) / BW_scope_effective(scope)", "T_pipeline >= max(overlappable components) + sum(serial components)"],
        "demand_vector": demand,
        "absolute_latency_prediction": None,
        "evidence_level": MODEL_DERIVED_UNVALIDATED,
    }
