"""Explicit provenance labels used by every exported result."""

from __future__ import annotations

COMPILER_OBSERVED = "compiler_observed"
SEMANTIC_DERIVED = "semantic_derived"
MODEL_DERIVED_UNVALIDATED = "model_derived_unvalidated"
HEURISTIC_MAPPING = "heuristic_mapping"
UNKNOWN = "unknown"
RUNTIME_DIRECT = "runtime_direct"
RUNTIME_TELEMETRY = "runtime_telemetry"
PROFILER_OBSERVED = "profiler_observed"
PROFILER_REPLAY_OBSERVED = "profiler_replay_observed"
INSTRUMENTED_RUNTIME = "instrumented_runtime"
DERIVED_CORRELATED = "derived_correlated"

ALL_LEVELS = {
    COMPILER_OBSERVED,
    SEMANTIC_DERIVED,
    MODEL_DERIVED_UNVALIDATED,
    HEURISTIC_MAPPING,
    UNKNOWN,
    RUNTIME_DIRECT,
    RUNTIME_TELEMETRY,
    PROFILER_OBSERVED,
    PROFILER_REPLAY_OBSERVED,
    INSTRUMENTED_RUNTIME,
    DERIVED_CORRELATED,
}
