"""Static TileOp execution domains and conservative symbolic multiplicities."""

from __future__ import annotations

from typing import Any

from ...core.evidence import SEMANTIC_DERIVED


def _multiply(values: list[int | str]) -> int | str:
    numeric = 1
    symbolic: list[str] = []
    for value in values:
        if isinstance(value, int):
            numeric *= value
        else:
            symbolic.append(value)
    if not symbolic:
        return numeric
    terms = ([str(numeric)] if numeric != 1 else []) + symbolic
    return " * ".join(terms) if terms else "1"


def _format(value: int | str) -> str:
    return str(value)


def launch_context(prim_func: Any) -> dict[str, Any]:
    """Get launch axes from the input Tile IR without querying a device."""
    import re

    bindings: dict[str, int | str] = {}
    for axis, extent in re.findall(r'T\.launch_thread\("([^"]+)",\s*([^\)]+)\)', prim_func.script()):
        normalized = extent.strip()
        try:
            bindings[axis] = int(normalized)
        except ValueError:
            bindings[axis] = normalized
    block_axes = [bindings.get(axis, 1) for axis in ("blockIdx.x", "blockIdx.y", "blockIdx.z")]
    thread_axes = [bindings.get(axis, 1) for axis in ("threadIdx.x", "threadIdx.y", "threadIdx.z")]
    return {
        "block_axes": {axis: bindings.get(axis, 1) for axis in ("blockIdx.x", "blockIdx.y", "blockIdx.z")},
        "thread_axes": {axis: bindings.get(axis, 1) for axis in ("threadIdx.x", "threadIdx.y", "threadIdx.z")},
        "block_count": _multiply(block_axes),
        "thread_count": _multiply(thread_axes),
    }


def attach_execution_counts(tile_ops: list[dict[str, Any]], launch: dict[str, Any]) -> None:
    """Attach per-call/per-block/per-kernel static multiplicities in place."""
    block_count = launch["block_count"]
    for op in tile_ops:
        factors: list[int | str] = []
        for loop in op["loop_context"]:
            # A TileOp is block collective.  Any explicit thread-binding loop is
            # excluded to prevent multiplying a collective operation by threads.
            if loop.get("thread_binding"):
                continue
            value = loop.get("extent_value")
            factors.append(value if isinstance(value, int) else loop["extent"])
        condition = " and ".join(op.get("condition_context", [])) or "true"
        base = _multiply(factors)
        conditional = condition != "true"
        per_block: int | str = base if not conditional else f"indicator({condition}) * ({_format(base)})"
        per_kernel: int | str = _multiply([per_block, block_count])
        op["execution"] = {
            "execution_domain": "per_invocation",
            "static_occurrences": 1,
            "serial_multiplicity_per_block_expr": _format(base),
            "serial_multiplicity_per_block_value": base if isinstance(base, int) and not conditional else None,
            "block_count_expr": _format(block_count),
            "block_count_value": block_count if isinstance(block_count, int) else None,
            "calls_per_block_expr": _format(per_block),
            "calls_per_block_value": per_block if isinstance(per_block, int) else None,
            "calls_per_kernel_expr": _format(per_kernel),
            "calls_per_kernel_value": per_kernel if isinstance(per_kernel, int) else None,
            "condition_expr": condition,
            "evidence_level": SEMANTIC_DERIVED,
        }
