"""Public TIR analysis interfaces.

The historical arch/utils helpers depend on TileLang installations which
still vendor ``tilelang.tools.tilesight``.  Keep them optional so the core TIR
analyzer remains usable with current TileLang and plain TVM installations.
"""

from .analyze_tir import (
    analyze_prim_func,
    analyze_tir_after_vectorize,
    estimate_flops_from_mod,
)

try:  # Backward compatibility for older TileLang packages.
    from .arch import *  # noqa: F401,F403
    from .utils import *  # noqa: F401,F403
except ImportError:
    pass

__all__ = [
    "analyze_prim_func",
    "analyze_tir_after_vectorize",
    "estimate_flops_from_mod",
]
