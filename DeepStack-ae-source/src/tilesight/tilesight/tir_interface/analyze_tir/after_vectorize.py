"""TileSight analysis entry for post-vectorization/lowered TIR."""

from __future__ import annotations

from typing import Any

from .core import analyze_tir


def analyze_tir_after_vectorize(mod: Any, device: Any) -> dict[str, Any]:
    """Extract the compiler metadata required by performance-tool item 5.1.

    If TileOp calls/provenance remain, the report preserves them.  Otherwise
    the analyzer intentionally falls back to conservative lowered-TIR facts
    and reports that semantic loss in ``diagnostics``.

    The result contains TileOp categories, hardware levels, dependencies,
    pipeline structure, and stable kernel/pipeline/op indexes.  Performance
    costs and predictions intentionally belong to the separate model layer.
    """
    return analyze_tir(mod, device, phase="after_vectorize")


__all__ = ["analyze_tir_after_vectorize"]
