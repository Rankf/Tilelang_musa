"""TileLang compiler-information collection through existing compiler APIs.

The module deliberately collects compiler facts only.  It does not import a
performance model, launch a kernel, read profiler counters, or modify the
``tilelang.compile``/``tilelang.jit``/``tilelang.lower`` interfaces.
"""

from .collector import (
    SCHEMA_VERSION,
    CompilerInfoCollector,
    collect_compiler_information,
)

__all__ = [
    "SCHEMA_VERSION",
    "CompilerInfoCollector",
    "collect_compiler_information",
]
