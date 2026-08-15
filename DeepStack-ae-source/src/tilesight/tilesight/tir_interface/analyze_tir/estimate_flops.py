"""Compatibility FLOP estimator backed by the auditable TIR analyzer."""

from __future__ import annotations

from typing import Any

from .core import _iter_prim_funcs, _post_order, _collect_loops, _collect_tile_ops, _workload, _collect_launch, _stable_id


def estimate_flops_from_mod(mod: Any) -> float:
    """Return statically resolved TileOp FLOPs without debug output.

    This compatibility helper counts explicit structured GEMM TileOps.  For a
    complete report, including unknown terms, callers should use
    :func:`analyze_prim_func` or :func:`analyze_tir_after_vectorize`.
    """
    total = 0
    for name, function in _iter_prim_funcs(mod):
        kernel_id = _stable_id("kernel", name, function.script())
        nodes = _post_order(function.body)
        loops = _collect_loops(nodes, kernel_id)
        operations = _collect_tile_ops(nodes, loops, kernel_id)
        workload = _workload(operations, loops, _collect_launch(nodes))
        value = workload["per_kernel"].get("flops")
        if isinstance(value, int):
            total += value
    return float(total)


__all__ = ["estimate_flops_from_mod"]
