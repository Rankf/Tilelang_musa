"""Lossless, context-aware extraction from input TileLang Tile IR."""

from __future__ import annotations

from typing import Any

from ...adapters.tilelang.execution_context import collect_tileop_calls
from ...core.dtypes import dtype_bytes
from ...core.evidence import COMPILER_OBSERVED
from ...core.ids import stable_id
from .execution_counts import attach_execution_counts, launch_context
from .tileop_registry import build_tileop_record


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
    text = " * ".join(_expr(value) for value in values) if values else "1"
    numeric = 1
    for value in values:
        parsed = _integer(value)
        if parsed is None:
            return text, None
        numeric *= parsed
    return text, numeric


def _buffer_record(buffer: Any, origin: str) -> dict[str, Any]:
    shape = [_expr(value) for value in buffer.shape]
    _, elements = _product(list(buffer.shape))
    byte_width = dtype_bytes(str(buffer.dtype))
    return {
        "buffer_id": stable_id("buffer", [buffer.name, origin, shape, str(buffer.dtype), buffer.scope()]),
        "name": str(buffer.name),
        "origin": origin,
        "shape": shape,
        "dtype": str(buffer.dtype),
        "scope": str(buffer.scope()),
        "elements": elements if elements is not None else " * ".join(shape),
        "allocation_bytes": elements * byte_width if elements is not None and byte_width else None,
        "source_snapshot": "S00_INPUT_TILE_IR",
        "evidence_level": COMPILER_OBSERVED,
    }


def _collect_buffers_and_loops(prim_func: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import tvm.tirx as tirx

    buffers: dict[str, dict[str, Any]] = {
        str(buffer.name): _buffer_record(buffer, "parameter")
        for _, buffer in prim_func.buffer_map.items()
    }
    loops: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        kind = type(node).__name__
        if kind == "SBlock":
            for buffer in node.alloc_buffers:
                buffers.setdefault(str(buffer.name), _buffer_record(buffer, "allocation"))
        elif kind == "For":
            annotations = {str(key): _expr(value) for key, value in node.annotations.items()}
            loops.append({
                "loop_id": stable_id("loop", [str(node.loop_var), _expr(node.extent), annotations]),
                "var": str(node.loop_var),
                "min": _expr(node.min),
                "extent": _expr(node.extent),
                "extent_value": _integer(node.extent),
                "kind": str(node.kind),
                "thread_binding": _expr(node.thread_binding) if node.thread_binding else None,
                "annotations": annotations,
                "source_snapshot": "S00_INPUT_TILE_IR",
                "evidence_level": COMPILER_OBSERVED,
            })

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)
    return list(buffers.values()), loops


def extract_metadata(prim_func: Any) -> dict[str, Any]:
    """Extract all TileOp actions and preserve unsupported actions as opaque."""
    buffers, loops = _collect_buffers_and_loops(prim_func)
    tile_ops: list[dict[str, Any]] = []

    def capture(call: Any, context: dict[str, Any]) -> None:
        tile_ops.append(build_tileop_record(call, context))

    coverage = collect_tileop_calls(prim_func, capture)
    launch = launch_context(prim_func)
    attach_execution_counts(tile_ops, launch)
    coverage.update({
        "exported_action_records": len(tile_ops),
        "structured_records": sum(op["support_level"] == "structured" for op in tile_ops),
        "structural_records": sum(op["support_level"] == "structural" for op in tile_ops),
        "opaque_records": sum(op["support_level"] == "opaque" for op in tile_ops),
        "invalid_records": sum(op["support_level"] == "invalid" for op in tile_ops),
    })
    coverage["unexported_action_calls"] = coverage["observed_action_calls"] - coverage["exported_action_records"]
    coverage["lossless_export"] = coverage["unexported_action_calls"] == 0
    pipeline_loops = [loop for loop in loops if "num_stages" in loop["annotations"]]
    thread_bindings = [
        {"axis": axis, "extent": extent, "evidence_level": COMPILER_OBSERVED}
        for axis, extent in {**launch["block_axes"], **launch["thread_axes"]}.items()
    ]
    return {
        "kernel": {
            "name": str(prim_func.attrs.get("global_symbol", "unknown")),
            "attrs": {str(key): _expr(value) for key, value in prim_func.attrs.items()},
            "thread_bindings": thread_bindings,
            "launch_context": launch,
            "source_snapshot": "S00_INPUT_TILE_IR",
            "evidence_level": COMPILER_OBSERVED,
        },
        "buffers": buffers,
        "loops": loops,
        "pipeline": {
            "pipeline_loops": pipeline_loops,
            "status": "structural_only",
            "source_snapshot": "S00_INPUT_TILE_IR",
            "limitations": ["v0 records Pipelined loop annotations but does not infer prologue, steady-state, or epilogue schedules"],
            "evidence_level": COMPILER_OBSERVED,
        },
        "tileop_coverage": coverage,
        "tile_ops": tile_ops,
    }


__all__ = ["dtype_bytes", "extract_metadata"]
