"""High-level TileSight model/runtime analysis API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .comparison import compare_with_runtime
from .diagnostics import diagnose_with_runtime


SCHEMA_VERSION = "tilesight/model-runtime-analysis-v1"


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _hierarchy_view(
    model_result: Mapping[str, Any], diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    findings = {
        finding.get("metric_id"): finding
        for finding in diagnostics.get("findings", [])
        if isinstance(finding, Mapping)
    }
    bottleneck = findings.get("5.6", {})
    runtime = bottleneck.get("runtime_finding", {})
    runtime_util = (
        runtime.get("utilization_pct", {})
        if isinstance(runtime, Mapping) else {}
    )
    levels = []
    for level, model_field, runtime_field in (
        ("compute", "compute_util_pct", "compute"),
        ("shared_memory", "smem_util_pct", "shared_memory"),
        ("l1", None, "l1"),
        ("l2", "l2_util_pct", "l2"),
        ("ddr", "ddr_util_pct", "ddr"),
    ):
        levels.append({
            "level": level,
            "model_utilization_pct": (
                _number(model_result.get(model_field)) if model_field else None
            ),
            "runtime_utilization_pct": (
                _number(runtime_util.get(runtime_field))
                if isinstance(runtime_util, Mapping) else None
            ),
        })
    return {
        "metric_id": "5.13",
        "value_kind": "model_runtime_view",
        "levels": levels,
    }


def analyze_with_runtime(
    model_result: Mapping[str, Any],
    runtime: Mapping[str, Any] | str | Path,
    record_id: str,
    *,
    model_source: str | None = None,
    runtime_source: str | None = None,
    require_identity: bool = True,
) -> dict[str, Any]:
    """Return comparison and 5.6-5.10 diagnostics as one joined artifact."""
    comparison = compare_with_runtime(
        model_result,
        runtime,
        record_id,
        model_source=model_source,
        runtime_source=runtime_source,
        require_identity=require_identity,
    )
    diagnostics = diagnose_with_runtime(
        model_result,
        runtime,
        record_id,
        require_identity=require_identity,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_scope": "comparison_and_5.6_to_5.10",
        "model_source": model_source,
        "runtime_source": runtime_source or (
            str(runtime) if isinstance(runtime, (str, Path)) else None
        ),
        "runtime_record_id": record_id,
        "model_data_embedded_in_runtime": False,
        "runtime_data_embedded_in_model": False,
        "tilesight_prediction_modified": False,
        "identity": comparison["identity"],
        "comparison": comparison,
        "diagnostics": diagnostics,
        "prediction_deviation": comparison["prediction_deviation"],
        "hierarchy_load_view": _hierarchy_view(model_result, diagnostics),
    }
