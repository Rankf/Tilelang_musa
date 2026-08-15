"""Self-contained TileLang/TIR static analysis for TileSight.

This module deliberately depends only on the TIR passed by the caller and the
TileSight architecture object.  It does not import a profiler or an external
metadata collector.  Facts which cannot be proved from the input IR are kept
as ``unknown``/diagnostics instead of being guessed.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from typing import Any, Iterable


SCHEMA_VERSION = "tilesight/tile-metadata-v1"


def _tir_module():
    """Import the TileLang TVM dialect lazily.

    TileLang-MUSA exposes its TIR dialect as ``tvm.tirx`` while upstream TVM
    commonly exposes ``tvm.tir``.  Supporting both keeps this analyzer useful
    outside the local MUSA tree and avoids breaking a plain TileSight import.
    """
    try:
        import tvm.tirx as tir  # type: ignore
        return tir
    except ImportError:
        try:
            from tilelang import tvm  # type: ignore
            return tvm.tirx
        except (ImportError, AttributeError):
            import tvm.tir as tir  # type: ignore
            return tir


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _integer(value: Any) -> int | None:
    try:
        return int(value.value)
    except (AttributeError, TypeError, ValueError):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None


def _product(values: Iterable[Any]) -> int | None:
    result = 1
    for value in values:
        parsed = _integer(value)
        if parsed is None:
            return None
        result *= parsed
    return result


def _dtype_bytes(dtype: Any) -> int | None:
    text = str(dtype)
    match = re.search(r"(bool|int|uint|float|bfloat)(\d+)?(?:x(\d+))?", text)
    if not match:
        return None
    bits = 1 if match.group(1) == "bool" else int(match.group(2) or 32)
    lanes = int(match.group(3) or 1)
    return math.ceil(bits * lanes / 8)


def _scope(buffer: Any) -> str:
    try:
        raw = str(buffer.scope())
    except (AttributeError, TypeError):
        raw = str(getattr(buffer, "scope", ""))
    if raw.startswith("shared"):
        return "shared"
    if raw.startswith("local") or raw.startswith("wmma"):
        return "local"
    if raw.startswith("constant"):
        return "constant"
    return "global" if not raw else raw


def _op_name(call: Any) -> str:
    op = getattr(call, "op", None)
    name = getattr(op, "name", None)
    if name:
        return str(name)
    text = str(op)
    match = re.search(r'name="([^"]+)"', text)
    return match.group(1) if match else text


def _annotations(node: Any) -> dict[str, str]:
    raw = getattr(node, "annotations", None)
    if raw is None:
        return {}
    try:
        return {str(key): str(value) for key, value in raw.items()}
    except AttributeError:
        return {}


def _iter_prim_funcs(value: Any) -> list[tuple[str, Any]]:
    tir = _tir_module()
    if isinstance(value, tir.PrimFunc):
        name = str(value.attrs.get("global_symbol", "main")) if value.attrs else "main"
        return [(name, value)]
    functions = getattr(value, "functions", None)
    if functions is None:
        raise TypeError("expected a TIR PrimFunc or IRModule-like object")
    result = []
    for global_var, function in functions.items():
        if isinstance(function, tir.PrimFunc):
            name = getattr(global_var, "name_hint", None) or str(global_var)
            result.append((str(name), function))
    if not result:
        raise ValueError("input contains no TIR PrimFunc")
    return result


def _post_order(root: Any) -> list[Any]:
    tir = _tir_module()
    nodes: list[Any] = []
    tir.stmt_functor.post_order_visit(root, nodes.append)
    return nodes


def _buffer_record(buffer: Any, origin: str) -> dict[str, Any]:
    shape = [str(value) for value in buffer.shape]
    elements = _product(buffer.shape)
    width = _dtype_bytes(buffer.dtype)
    scope = _scope(buffer)
    name = str(buffer.name)
    return {
        "buffer_id": _stable_id("buffer", name, origin, shape, str(buffer.dtype), scope),
        "name": name,
        "origin": origin,
        "shape": shape,
        "dtype": str(buffer.dtype),
        "scope": scope,
        "elements": elements,
        "allocation_bytes": elements * width if elements is not None and width else None,
        "evidence": "compiler_observed",
    }


def _collect_buffers(function: Any, nodes: list[Any]) -> list[dict[str, Any]]:
    buffers: dict[str, dict[str, Any]] = {}
    for buffer in function.buffer_map.values():
        buffers[str(buffer.name)] = _buffer_record(buffer, "parameter")
    for node in nodes:
        for attribute in ("alloc_buffers", "match_buffers"):
            for item in getattr(node, attribute, ()):
                buffer = getattr(item, "buffer", item)
                if hasattr(buffer, "shape"):
                    buffers.setdefault(str(buffer.name), _buffer_record(buffer, "allocation"))
    return list(buffers.values())


def _collect_launch(nodes: list[Any]) -> dict[str, Any]:
    axes: dict[str, int | str] = {}
    for node in nodes:
        if type(node).__name__ != "AttrStmt" or str(node.attr_key) != "thread_extent":
            continue
        axis = getattr(getattr(node.node, "thread_tag", None), "value", None)
        if not axis:
            axis = str(getattr(node.node, "thread_tag", "unknown"))
        extent = _integer(node.value)
        axes[str(axis)] = extent if extent is not None else str(node.value)
    grid = {key: value for key, value in axes.items() if key.startswith("blockIdx")}
    block = {key: value for key, value in axes.items() if key.startswith("threadIdx")}
    grid_size = _product(value for value in grid.values())
    block_size = _product(value for value in block.values())
    return {
        "grid_axes": grid,
        "thread_axes": block,
        "grid_size": grid_size,
        "threads_per_block": block_size,
        "evidence": "compiler_observed" if axes else "unavailable",
    }


def _collect_loops(nodes: list[Any], kernel_id: str) -> list[dict[str, Any]]:
    loops = []
    for ordinal, node in enumerate(item for item in nodes if type(item).__name__ == "For"):
        annotations = _annotations(node)
        loop_id = _stable_id(
            "loop", kernel_id, ordinal, str(node.loop_var), str(node.min),
            str(node.extent), annotations,
        )
        loops.append({
            "loop_id": loop_id,
            "ordinal": ordinal,
            "var": str(node.loop_var),
            "min": str(node.min),
            "extent": str(node.extent),
            "extent_value": _integer(node.extent),
            "kind": str(node.kind),
            "thread_binding": str(node.thread_binding) if node.thread_binding else None,
            "annotations": annotations,
            "evidence": "compiler_observed",
            "_node": node,
        })
    return loops


def _region(call: Any) -> dict[str, Any] | None:
    if _op_name(call) != "tl.tileop.region" or not call.args:
        return None
    load = call.args[0]
    buffer = getattr(load, "buffer", None)
    if buffer is None:
        return None
    access_type = _integer(call.args[1]) if len(call.args) > 1 else None
    extents = list(call.args[2:])
    elements = _product(extents)
    width = _dtype_bytes(buffer.dtype)
    return {
        "buffer": str(buffer.name),
        "scope": _scope(buffer),
        "dtype": str(buffer.dtype),
        "indices": [str(value) for value in getattr(load, "indices", ())],
        "extents": [str(value) for value in extents],
        "access_type": access_type,
        "logical_bytes": elements * width if elements is not None and width else None,
    }


def _tileop_access(call: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    name = _op_name(call)
    regions = [item for item in (_region(arg) for arg in call.args) if item is not None]
    if name == "tl.tileop.copy" and len(regions) >= 2:
        return [regions[0]], [regions[1]]
    if name in {"tl.tileop.gemm", "tl.tileop.gemm_sp"} and len(regions) >= 3:
        return regions[:2] + [regions[2]], [regions[2]]
    if name in {"tl.tileop.fill", "tl.tileop.clear"} and regions:
        return [], [regions[0]]
    reads = [region for region in regions if region["access_type"] in (1, 3)]
    writes = [region for region in regions if region["access_type"] in (2, 3)]
    return reads, writes


def _call_family(name: str) -> str:
    short = name.removeprefix("tl.tileop.")
    if short.startswith("gemm"):
        return "gemm"
    if short in {"copy", "copy_g2s", "copy_s2g"}:
        return "copy"
    if "reduce" in short:
        return "reduce"
    if short in {"fill", "clear"}:
        return "fill"
    return short or "unknown"


def _gemm_detail(call: Any) -> dict[str, Any] | None:
    if _call_family(_op_name(call)) != "gemm" or len(call.args) < 8:
        return None
    dims = [_integer(call.args[index]) for index in (5, 6, 7)]
    if any(value is None for value in dims):
        return {"m": str(call.args[5]), "n": str(call.args[6]), "k": str(call.args[7]), "flops": None}
    m, n, k = dims
    return {"m": m, "n": n, "k": k, "flops": 2 * m * n * k}


def _contained_loop_ids(call: Any, loops: list[dict[str, Any]]) -> list[str]:
    result = []
    for loop in loops:
        if any(node.same_as(call) for node in _post_order(loop["_node"].body) if hasattr(node, "same_as")):
            result.append(loop["loop_id"])
    return result


def _collect_tile_ops(nodes: list[Any], loops: list[dict[str, Any]], kernel_id: str) -> list[dict[str, Any]]:
    calls = [node for node in nodes if type(node).__name__ == "Call" and _op_name(node).startswith("tl.tileop.") and _op_name(node) != "tl.tileop.region"]
    operations = []
    for ordinal, call in enumerate(calls):
        name = _op_name(call)
        reads, writes = _tileop_access(call)
        raw = str(call)
        operations.append({
            "op_id": _stable_id("op", kernel_id, ordinal, name, raw),
            "ordinal": ordinal,
            "kind": name.removeprefix("tl.tileop."),
            "canonical_family": _call_family(name),
            "source_kind": "tileop_call",
            "raw_ir": raw,
            "annotations": _annotations(call),
            "reads": reads,
            "writes": writes,
            "loop_ids": _contained_loop_ids(call, loops),
            "gemm": _gemm_detail(call),
            "evidence": "compiler_observed",
            "support_level": "structured" if reads or writes or _call_family(name) == "gemm" else "structural",
        })
    return operations


def _floating_scalar_ops(expression: Any) -> int:
    """Count explicit scalar floating-point arithmetic in one expression."""
    count = 0
    binary = {"Add", "Sub", "Mul", "Div", "FloorDiv", "FloorMod", "Mod", "Min", "Max"}
    for node in _post_order(expression):
        dtype = str(getattr(node, "dtype", ""))
        if type(node).__name__ in binary and ("float" in dtype or "bfloat" in dtype):
            count += 1
        elif type(node).__name__ == "Call":
            name = _op_name(node).lower()
            if ("fma" in name or "mad" in name) and ("float" in dtype or "bfloat" in dtype):
                count += 2
    return count


def _collect_lowered_ops(nodes: list[Any], loops: list[dict[str, Any]], kernel_id: str) -> list[dict[str, Any]]:
    """Fallback when TileOp calls have already disappeared after lowering."""
    operations = []
    candidates = [
        node for node in nodes
        if type(node).__name__ in {"BufferStore", "Call"}
        and not (type(node).__name__ == "Call" and _op_name(node) == "tl.tileop.region")
    ]
    for ordinal, node in enumerate(candidates):
        reads: list[dict[str, Any]] = []
        writes: list[dict[str, Any]] = []
        if type(node).__name__ == "BufferStore":
            buffer = node.buffer
            width = _dtype_bytes(buffer.dtype)
            writes.append({
                "buffer": str(buffer.name), "scope": _scope(buffer),
                "dtype": str(buffer.dtype), "indices": [str(v) for v in node.indices],
                "extents": ["1"], "access_type": 2, "logical_bytes": width,
            })
            for child in _post_order(node.value):
                if type(child).__name__ == "BufferLoad":
                    source = child.buffer
                    reads.append({
                        "buffer": str(source.name), "scope": _scope(source),
                        "dtype": str(source.dtype), "indices": [str(v) for v in child.indices],
                        "extents": ["1"], "access_type": 1,
                        "logical_bytes": _dtype_bytes(source.dtype),
                    })
            name, family = "buffer_store", "elementwise"
            scalar_flops = _floating_scalar_ops(node.value)
        else:
            name = _op_name(node)
            family = "intrinsic" if not name.startswith("tl.tileop.") else _call_family(name)
            scalar_flops = 0
        operations.append({
            "op_id": _stable_id("op", kernel_id, ordinal, name, str(node)),
            "ordinal": ordinal,
            "kind": name,
            "canonical_family": family,
            "source_kind": "lowered_tir",
            "raw_ir": str(node),
            "annotations": _annotations(node),
            "reads": reads,
            "writes": writes,
            "loop_ids": _contained_loop_ids(node, loops),
            "gemm": None,
            "scalar_flops": scalar_flops,
            "evidence": "compiler_observed",
            "support_level": "structural",
        })
    return operations


def _build_dag(operations: list[dict[str, Any]]) -> dict[str, Any]:
    last_writer: dict[str, str] = {}
    readers: dict[str, list[str]] = defaultdict(list)
    edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for operation in operations:
        current = operation["op_id"]
        for region in operation["reads"]:
            buffer = region["buffer"]
            if buffer in last_writer:
                key = (last_writer[buffer], current, buffer, "RAW")
                edges[key] = {"from": key[0], "to": key[1], "buffer": buffer, "kind": "RAW", "confidence": "buffer_level_may_depend"}
            readers[buffer].append(current)
        for region in operation["writes"]:
            buffer = region["buffer"]
            if buffer in last_writer:
                key = (last_writer[buffer], current, buffer, "WAW")
                edges[key] = {"from": key[0], "to": key[1], "buffer": buffer, "kind": "WAW", "confidence": "buffer_level_may_depend"}
            for reader in readers[buffer]:
                if reader != current:
                    key = (reader, current, buffer, "WAR")
                    edges[key] = {"from": reader, "to": current, "buffer": buffer, "kind": "WAR", "confidence": "buffer_level_may_depend"}
            last_writer[buffer] = current
            readers[buffer] = []
    return {
        "nodes": [{"op_id": op["op_id"], "ordinal": op["ordinal"], "family": op["canonical_family"]} for op in operations],
        "edges": list(edges.values()),
        "scope": "single kernel; conservative buffer-level dependence without alias proof",
        "evidence": "semantically_derived",
    }


def _attach_dependencies(
    operations: list[dict[str, Any]], dag: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return public TileOp metadata with dependencies expressed by op_id."""
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in dag["edges"]:
        if edge["from"] not in incoming[edge["to"]]:
            incoming[edge["to"]].append(edge["from"])
    public = []
    for operation in operations:
        gemm = operation.get("gemm")
        tile_shape = None
        if isinstance(gemm, dict):
            tile_shape = {key: gemm.get(key) for key in ("m", "n", "k")}
        public.append({
            "op_id": operation["op_id"],
            "ordinal": operation["ordinal"],
            "category": operation["canonical_family"],
            "kind": operation["kind"],
            "source_kind": operation["source_kind"],
            "reads": operation["reads"],
            "writes": operation["writes"],
            "hardware_levels": {
                "read": sorted({item["scope"] for item in operation["reads"]}),
                "write": sorted({item["scope"] for item in operation["writes"]}),
            },
            "dependency_ids": incoming[operation["op_id"]],
            "loop_ids": operation["loop_ids"],
            "tile_shape": tile_shape,
            "annotations": operation["annotations"],
            "raw_ir": operation["raw_ir"],
            "evidence": operation["evidence"],
            "support_level": operation["support_level"],
        })
    return public


def _pipeline(loops: list[dict[str, Any]], operations: list[dict[str, Any]], kernel_id: str) -> dict[str, Any]:
    pipelines = []
    for loop in loops:
        annotation = loop["annotations"].get("num_stages")
        if annotation is None:
            continue
        stages = _integer(annotation)
        members = [op["op_id"] for op in operations if loop["loop_id"] in op["loop_ids"]]
        pipelines.append({
            "pipeline_id": _stable_id("pipeline", kernel_id, loop["loop_id"]),
            "loop_id": loop["loop_id"],
            "stage_count": stages if stages is not None else annotation,
            "operation_ids": members,
            "stage_assignment_status": "unavailable_without_explicit_per_operation_annotation",
            "evidence": "compiler_annotation",
        })
    return {
        "status": "explicit" if pipelines else "unavailable",
        "pipelines": pipelines,
        "limitations": [] if pipelines else ["no software-pipeline annotation is present in this IR snapshot"],
    }


def _loop_multiplier(operation: dict[str, Any], loop_by_id: dict[str, dict[str, Any]]) -> int | None:
    value = 1
    for loop_id in operation["loop_ids"]:
        extent = loop_by_id[loop_id]["extent_value"]
        if extent is None:
            return None
        value *= extent
    return value


def _workload(operations: list[dict[str, Any]], loops: list[dict[str, Any]], launch: dict[str, Any]) -> dict[str, Any]:
    loop_by_id = {loop["loop_id"]: loop for loop in loops}
    per_block = defaultdict(int)
    flops = 0
    unknown_bytes = 0
    unknown_flops = 0
    for operation in operations:
        multiplier = _loop_multiplier(operation, loop_by_id)
        if multiplier is None:
            multiplier = 1
            unknown_bytes += 1
            if operation.get("gemm"):
                unknown_flops += 1
        gemm_flops = (operation.get("gemm") or {}).get("flops")
        if isinstance(gemm_flops, int):
            flops += gemm_flops * multiplier
        scalar_flops = operation.get("scalar_flops")
        if isinstance(scalar_flops, int):
            flops += scalar_flops * multiplier
        for direction, regions in (("read", operation["reads"]), ("write", operation["writes"])):
            for region in regions:
                amount = region.get("logical_bytes")
                if isinstance(amount, int):
                    per_block[f"{region['scope']}_{direction}_bytes"] += amount * multiplier
                else:
                    unknown_bytes += 1
    grid_size = launch.get("grid_size")
    per_kernel = {key: value * grid_size if isinstance(grid_size, int) else None for key, value in per_block.items()}
    return {
        "per_block": {**dict(per_block), "flops": flops},
        "per_kernel": {**per_kernel, "flops": flops * grid_size if isinstance(grid_size, int) else None},
        "compute_kind": "tensor" if any(op["canonical_family"] == "gemm" for op in operations) else "scalar_or_vector",
        "unknown_byte_terms": unknown_bytes,
        "unknown_flop_terms": unknown_flops,
        "value_kind": "logical_static_workload",
    }


def _positive_attr(device: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(device, name, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _roofline(workload: dict[str, Any], device: Any) -> dict[str, Any]:
    kernel = workload["per_kernel"]
    if workload["compute_kind"] == "tensor":
        compute_peak = _positive_attr(device, "fp16_tensor_flops", "fp32_cuda_core_flops", "compute_flops")
        compute_resource = "tensor_core"
    else:
        compute_peak = _positive_attr(device, "fp32_cuda_core_flops", "fp16_cuda_core_flops", "compute_flops")
        compute_resource = "scalar_or_vector_core"
    compute_util = _positive_attr(device, "compute_max_util") or 1.0
    bandwidths = {
        "ddr": _positive_attr(device, "ddr_bandwidth"),
        "l2": _positive_attr(device, "l2_bandwidth"),
        "shared": _positive_attr(device, "smem_bandwidth"),
    }
    max_utils = {
        "ddr": _positive_attr(device, "ddr_max_util") or 1.0,
        "l2": _positive_attr(device, "l2_max_util") or 1.0,
        "shared": _positive_attr(device, "l1_max_util") or 1.0,
    }
    costs: dict[str, float | None] = {}
    flops = kernel.get("flops")
    costs["compute"] = flops / (compute_peak * compute_util) if isinstance(flops, int) and compute_peak else None
    byte_keys = {
        "ddr": ("global_read_bytes", "global_write_bytes"),
        "l2": ("global_read_bytes", "global_write_bytes"),
        "shared": ("shared_read_bytes", "shared_write_bytes"),
    }
    for level, keys in byte_keys.items():
        values = [kernel.get(key) for key in keys]
        traffic = sum(value for value in values if isinstance(value, int)) if any(isinstance(value, int) for value in values) else None
        bandwidth = bandwidths[level]
        costs[level] = traffic / (bandwidth * max_utils[level]) if traffic is not None and bandwidth else None
    available = {key: value for key, value in costs.items() if value is not None}
    bottleneck = max(available, key=available.get) if available else "unavailable"
    return {
        "model_kind": "generic_static_roofline",
        "compute_resource": compute_resource,
        "status": "available" if available else "unavailable",
        "service_cost_us": {key: round(value * 1e6, 6) if value is not None else None for key, value in costs.items()},
        "latency_lower_bound_us": round(max(available.values()) * 1e6, 6) if available else None,
        "bottleneck": bottleneck,
        "overlap_rule": "max(compute, DDR, L2, Shared); logical traffic, no cache-hit inference",
        "value_kind": "model_lower_bound",
    }


def _public_loops(loops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in loop.items() if key != "_node"} for loop in loops]


def _analyze_function(name: str, function: Any, device: Any, phase: str) -> dict[str, Any]:
    script = function.script() if hasattr(function, "script") else str(function)
    kernel_id = _stable_id("kernel", name, script)
    nodes = _post_order(function.body)
    loops = _collect_loops(nodes, kernel_id)
    tile_ops = _collect_tile_ops(nodes, loops, kernel_id)
    operations = tile_ops or _collect_lowered_ops(nodes, loops, kernel_id)
    launch = _collect_launch(nodes)
    dag = _build_dag(operations)
    unknown = sum(op["support_level"] != "structured" for op in operations)
    diagnostics = []
    if not tile_ops:
        diagnostics.append({
            "severity": "warning",
            "code": "HIGH_LEVEL_TILEOP_SEMANTICS_UNAVAILABLE",
            "message": "analysis fell back to lowered TIR operations",
        })
    return {
        "kernel_id": kernel_id,
        "name": name,
        "analysis_phase": phase,
        "attrs": {str(key): str(value) for key, value in (function.attrs.items() if function.attrs else [])},
        "launch": launch,
        "buffers": _collect_buffers(function, nodes),
        "loops": _public_loops(loops),
        "operations": _attach_dependencies(operations, dag),
        "dependencies": dag,
        "pipeline": _pipeline(loops, operations, kernel_id),
        "coverage": {
            "operation_count": len(operations),
            "structured_operation_count": len(operations) - unknown,
            "structural_or_unknown_operation_count": unknown,
            "tileop_semantics_available": bool(tile_ops),
            "lossless_action_claim": False,
        },
        "diagnostics": diagnostics,
    }


def analyze_tir(value: Any, device: Any, *, phase: str) -> dict[str, Any]:
    """Analyze a PrimFunc/IRModule and return a JSON-serializable report."""
    if device is None:
        raise TypeError("device architecture must not be None")
    functions = _iter_prim_funcs(value)
    kernels = [_analyze_function(name, function, device, phase) for name, function in functions]
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": str(getattr(device, "core", type(device).__name__)),
        "analysis_phase": phase,
        "kernels": kernels,
        "summary": {
            "kernel_count": len(kernels),
            "operation_count": sum(kernel["coverage"]["operation_count"] for kernel in kernels),
            "diagnostic_count": sum(len(kernel["diagnostics"]) for kernel in kernels),
        },
        "value_kind": "compiler_observed_and_semantically_derived_metadata",
    }
