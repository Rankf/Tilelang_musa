"""Family adapters for TileLang TileOp calls and lossless opaque fallback."""

from __future__ import annotations

from typing import Any

from ...adapters.tilelang.tileop_discovery import op_name
from ...core.dtypes import dtype_bytes
from ...core.evidence import COMPILER_OBSERVED, SEMANTIC_DERIVED, UNKNOWN
from ...core.ids import stable_id


FAMILY_BY_NAME = {
    "copy": "copy", "async_copy": "copy", "tma_copy": "copy",
    "transpose": "transform_copy", "im2col": "transform_copy", "c2d_im2col": "transform_copy",
    "gemm": "gemm_dense", "gemm_py": "gemm_dense", "wgmma_gemm": "gemm_dense", "tcgen05_gemm": "gemm_dense",
    "gemm_sp": "gemm_sparse", "wgmma_gemm_sp": "gemm_sparse", "tcgen05_gemm_sp": "gemm_sparse",
    "fill": "fill",
    "reduce": "reduce", "cumsum": "scan", "finalize_reducer": "reducer_finalize",
    "atomicadd": "atomic", "atomicmax": "atomic", "atomicmin": "atomic",
}
CUDA_VARIANTS = {"wgmma_gemm", "tcgen05_gemm", "wgmma_gemm_sp", "tcgen05_gemm_sp"}


def _expr(value: Any) -> str:
    return str(value)


def _integer(value: Any) -> int | None:
    try:
        return int(value.value)
    except (AttributeError, TypeError, ValueError):
        try:
            return int(str(value))
        except ValueError:
            return None


def _product(values: list[Any]) -> tuple[str, int | None]:
    expression = " * ".join(_expr(value) for value in values) if values else "1"
    total = 1
    for value in values:
        parsed = _integer(value)
        if parsed is None:
            return expression, None
        total *= parsed
    return expression, total


def _region(region: Any) -> dict[str, Any] | None:
    """Parse TileLang's standard ``tl.tileop.region`` argument."""
    if op_name(region) != "tl.tileop.region" or len(region.args) < 3:
        return None
    load = region.args[0]
    buffer = getattr(load, "buffer", None)
    if buffer is None:
        return None
    shape_args = list(region.args[2:])
    shape_expression, elements = _product(shape_args)
    width = dtype_bytes(str(buffer.dtype))
    return {
        "buffer": str(buffer.name),
        "scope": str(buffer.scope()),
        "indices": [_expr(value) for value in load.indices],
        "shape": [_expr(value) for value in shape_args],
        "elements": elements if elements is not None else shape_expression,
        "logical_bytes": elements * width if elements is not None and width is not None else None,
        "evidence_level": SEMANTIC_DERIVED,
    }


def _regions_or_unknown(call: Any, read_indices: list[int], write_indices: list[int]) -> tuple[list[dict[str, Any]] | str, list[dict[str, Any]] | str]:
    def collect(indices: list[int]):
        values = []
        for index in indices:
            if index >= len(call.args):
                return UNKNOWN
            parsed = _region(call.args[index])
            if parsed is None:
                return UNKNOWN
            values.append(parsed)
        return values
    return collect(read_indices), collect(write_indices)


def _annotations(call: Any) -> dict[str, str]:
    return {str(key): _expr(value) for key, value in call.annotations.items()}


def _base_record(call: Any, context: dict[str, Any]) -> dict[str, Any]:
    raw_name = op_name(call)
    short_name = raw_name.removeprefix("tl.tileop.")
    family = FAMILY_BY_NAME.get(short_name, "unknown")
    return {
        "op_id": stable_id("op", [context["ordinal"], raw_name, _expr(call), context["statement_path"]]),
        "ordinal": context["ordinal"],
        "kind": short_name,
        "raw_op_name": raw_name,
        "canonical_family": family,
        "raw_ir": _expr(call),
        "raw_args": [_expr(value) for value in call.args],
        "annotations": _annotations(call),
        "source_snapshot": "S00_INPUT_TILE_IR",
        "statement_path": context["statement_path"],
        "loop_context": context["loop_context"],
        "condition_context": context["condition_context"],
        "evidence_level": COMPILER_OBSERVED,
        "diagnostic_codes": [],
    }


def _with_access(record: dict[str, Any], reads: list[dict[str, Any]] | str, writes: list[dict[str, Any]] | str) -> dict[str, Any]:
    record["reads"] = reads
    record["writes"] = writes
    return record


def _dense_gemm(record: dict[str, Any], call: Any) -> dict[str, Any]:
    reads, writes = _regions_or_unknown(call, [0, 1, 2], [2])
    _with_access(record, reads, writes)
    dimensions = [_integer(call.args[index]) if index < len(call.args) else None for index in (5, 6, 7)]
    m, n, k = dimensions
    record["gemm"] = {
        "m": m if m is not None else (_expr(call.args[5]) if len(call.args) > 5 else UNKNOWN),
        "n": n if n is not None else (_expr(call.args[6]) if len(call.args) > 6 else UNKNOWN),
        "k": k if k is not None else (_expr(call.args[7]) if len(call.args) > 7 else UNKNOWN),
        "fma_count": m * n * k if None not in dimensions else None,
        "flops": 2 * m * n * k if None not in dimensions else None,
        "evidence_level": SEMANTIC_DERIVED,
    }
    record["support_level"] = "structured" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
    if record["kind"] in CUDA_VARIANTS:
        record["diagnostic_codes"].append("TARGET_VARIANT_MISMATCH")
    return record


def _sparse_gemm(record: dict[str, Any], call: Any) -> dict[str, Any]:
    reads, writes = _regions_or_unknown(call, [0, 1, 2, 3], [3])
    _with_access(record, reads, writes)
    dimensions = [_integer(call.args[index]) if index < len(call.args) else None for index in (7, 8, 9)]
    m, n, k = dimensions
    record["gemm_sparse"] = {
        "m": m if m is not None else (_expr(call.args[7]) if len(call.args) > 7 else UNKNOWN),
        "n": n if n is not None else (_expr(call.args[8]) if len(call.args) > 8 else UNKNOWN),
        "k_dense_equivalent": k if k is not None else (_expr(call.args[9]) if len(call.args) > 9 else UNKNOWN),
        "flops_dense_equivalent": 2 * m * n * k if None not in dimensions else None,
        "workload_not_applicable": "sparse effective FLOPs require an explicit sparsity semantics contract",
        "evidence_level": SEMANTIC_DERIVED,
    }
    record["support_level"] = "structural" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
    if record["kind"] in CUDA_VARIANTS:
        record["diagnostic_codes"].append("TARGET_VARIANT_MISMATCH")
    return record


def build_tileop_record(call: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Adapt a TileOp into a structured record or a lossless opaque record."""
    record = _base_record(call, context)
    family = record["canonical_family"]
    if context["classification"] == "unregistered_action":
        record.update({"support_level": "opaque", "reads": UNKNOWN, "writes": UNKNOWN})
        record["diagnostic_codes"].append("UNREGISTERED_TILEOP_CALL")
        return record
    if family in {"copy", "transform_copy"}:
        reads, writes = _regions_or_unknown(call, [0], [1])
        _with_access(record, reads, writes)
        record["support_level"] = "structured" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
        if family == "transform_copy":
            record["transform"] = {"kind": record["kind"], "parameters": record["raw_args"][2:]}
        else:
            record["copy"] = {"async": record["kind"] == "async_copy", "tma": record["kind"] == "tma_copy"}
        return record
    if family == "gemm_dense":
        return _dense_gemm(record, call)
    if family == "gemm_sparse":
        return _sparse_gemm(record, call)
    if family == "fill":
        reads, writes = _regions_or_unknown(call, [], [0])
        _with_access(record, reads, writes)
        record["fill"] = {"value": _expr(call.args[1]) if len(call.args) > 1 else UNKNOWN}
        record["support_level"] = "structured" if isinstance(writes, list) else "opaque"
        return record
    if family == "reduce":
        reads, writes = _regions_or_unknown(call, [0], [1])
        _with_access(record, reads, writes)
        record["reduction"] = {
            "kind": _expr(call.args[2]) if len(call.args) > 2 else UNKNOWN,
            "dim": _expr(call.args[3]) if len(call.args) > 3 else UNKNOWN,
            "clear": _expr(call.args[4]) if len(call.args) > 4 else UNKNOWN,
        }
        record["support_level"] = "structural" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
        return record
    if family == "scan":
        reads, writes = _regions_or_unknown(call, [0], [1])
        _with_access(record, reads, writes)
        record["scan"] = {"dim": _expr(call.args[2]) if len(call.args) > 2 else UNKNOWN, "reverse": _expr(call.args[3]) if len(call.args) > 3 else UNKNOWN}
        record["support_level"] = "structural" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
        return record
    if family == "reducer_finalize":
        reads, writes = _regions_or_unknown(call, [0], [0])
        _with_access(record, reads, writes)
        record["support_level"] = "structural" if isinstance(reads, list) and isinstance(writes, list) else "opaque"
        return record
    if family == "atomic":
        value_read = _region(call.args[0]) if call.args else None
        destination = _region(call.args[1]) if len(call.args) > 1 else None
        reads: list[dict[str, Any]] | str = [] if value_read is None else [value_read]
        if destination is None:
            _with_access(record, UNKNOWN, UNKNOWN)
            record["support_level"] = "opaque"
        else:
            # Atomic operation is read-modify-write on its destination.
            read_regions = ([*reads, destination] if isinstance(reads, list) else UNKNOWN)
            _with_access(record, read_regions, [destination])
            record["support_level"] = "structured"
        record["atomic"] = {"operation": record["kind"].removeprefix("atomic"), "value": _expr(call.args[0]) if call.args else UNKNOWN}
        return record
    record.update({"support_level": "opaque", "reads": UNKNOWN, "writes": UNKNOWN})
    record["diagnostic_codes"].append("UNSUPPORTED_TILEOP_ADAPTER")
    return record
