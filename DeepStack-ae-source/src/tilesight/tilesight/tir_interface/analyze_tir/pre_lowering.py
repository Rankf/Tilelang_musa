"""TileSight analysis entry for source/high-level TileLang PrimFunc."""

from __future__ import annotations

from typing import Any

from .core import analyze_tir


def analyze_prim_func(prim_func: Any, device: Any) -> dict[str, Any]:
    """Extract 5.1 TileOp/pipeline metadata before lowering loses semantics."""
    return analyze_tir(prim_func, device, phase="pre_lowering")


__all__ = ["analyze_prim_func"]
