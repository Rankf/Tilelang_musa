"""Runtime evidence adapters and model/runtime analysis for TileSight.

Runtime observations are deliberately kept out of prediction functions.  The
APIs in this package join an immutable model result with an independently
collected runtime-only bundle for comparison and diagnostics.
"""

from .analysis import analyze_with_runtime
from .comparison import compare_with_runtime
from .diagnostics import diagnose_with_runtime
from .schema import (
    load_s5000_runtime,
    metric_value,
    select_runtime_record,
    validate_runtime_bundle,
    validate_workload_identity,
)

__all__ = [
    "analyze_with_runtime",
    "compare_with_runtime",
    "diagnose_with_runtime",
    "load_s5000_runtime",
    "metric_value",
    "select_runtime_record",
    "validate_runtime_bundle",
    "validate_workload_identity",
]
