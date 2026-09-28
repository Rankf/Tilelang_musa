"""Validation and identity helpers for S5000 runtime-only bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


RUNTIME_BUNDLE_SCHEMA = "s5000/runtime-measurement-bundle-v1"
SUPPORTED_TOOLS = {
    "musa_event",
    "moore_perf_compute",
    "moore_perf_system",
    "mupti_kernel_activity",
}
FORBIDDEN_MODEL_KEYS = {
    "model",
    "model_request",
    "model_result",
    "predicted_latency_us",
    "prediction",
    "tilesight",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def validate_runtime_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a runtime-only bundle without changing it."""
    if bundle.get("schema_version") != RUNTIME_BUNDLE_SCHEMA:
        raise ValueError(
            f"runtime schema must be {RUNTIME_BUNDLE_SCHEMA!r}"
        )
    if bundle.get("bundle_scope") != "runtime_only":
        raise ValueError("runtime input must have bundle_scope='runtime_only'")
    if bundle.get("model_data_included") is not False:
        raise ValueError("runtime bundle must explicitly exclude model data")
    embedded = sorted(FORBIDDEN_MODEL_KEYS.intersection(bundle))
    if embedded:
        raise ValueError(
            "runtime bundle contains forbidden model field(s): "
            + ", ".join(embedded)
        )

    case = _mapping(bundle.get("case"), "runtime case")
    missing = [
        field for field in ("case_id", "operator", "shape", "dtype", "kernel_name")
        if field not in case
    ]
    if missing:
        raise ValueError("runtime case missing: " + ", ".join(missing))

    records = bundle.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("runtime bundle requires non-empty records")
    identifiers: set[str] = set()
    for record in records:
        record = _mapping(record, "runtime record")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("runtime record needs a non-empty record_id")
        if record_id in identifiers:
            raise ValueError(f"duplicate runtime record_id: {record_id}")
        identifiers.add(record_id)
        if record.get("tool") not in SUPPORTED_TOOLS:
            raise ValueError(f"unsupported runtime tool: {record.get('tool')!r}")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"runtime record {record_id!r} requires metrics")
    return dict(bundle)


def load_s5000_runtime(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Load and validate one S5000 runtime bundle."""
    if isinstance(source, Mapping):
        bundle = dict(source)
    else:
        bundle = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_runtime_bundle(bundle)


def select_runtime_record(
    bundle: Mapping[str, Any], record_id: str,
) -> dict[str, Any]:
    """Select one unambiguous record from a validated bundle."""
    matches = [
        record for record in bundle.get("records", [])
        if record.get("record_id") == record_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"runtime record {record_id!r} resolved to {len(matches)} records"
        )
    return dict(matches[0])


def metric_value(record: Mapping[str, Any], name: str) -> float | None:
    """Return the median (preferred) or scalar value of a normalized metric."""
    metric = record.get("metrics", {}).get(name)
    if not isinstance(metric, Mapping):
        return None
    summary = metric.get("summary")
    if isinstance(summary, Mapping):
        median = summary.get("median")
        if isinstance(median, (int, float)) and not isinstance(median, bool):
            return float(median)
    value = metric.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _path(data: Mapping[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _normalized_cache(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    if lowered.startswith("cold"):
        return "cold"
    if lowered.startswith("warm"):
        return "warm"
    return lowered


def _normalized_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def validate_workload_identity(
    model_result: Mapping[str, Any], bundle: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify common model/runtime identity fields before numeric comparison.

    Missing optional binary identifiers are reported rather than invented.  A
    comparison remains eligible when the core semantic signature and every
    common launch field match.
    """
    case = _mapping(bundle.get("case"), "runtime case")
    environment = bundle.get("environment_profiles", {}).get(
        record.get("environment_profile"), {}
    )
    comparisons = {
        "operator": (model_result.get("operator"), case.get("operator")),
        "shape": (
            _normalized_shape(model_result.get("shape")),
            _normalized_shape(case.get("shape")),
        ),
        "dtype": (model_result.get("dtype"), case.get("dtype")),
        "kernel_name": (
            model_result.get("kernel_name"), case.get("kernel_name")
        ),
        "grid_size": (
            _path(model_result, "launch", "grid_size"),
            _path(case, "launch", "grid_size"),
        ),
        "threads_per_block": (
            _path(model_result, "launch", "threads_per_block"),
            _path(case, "launch", "threads_per_block"),
        ),
        "dynamic_shared_memory_bytes": (
            _path(model_result, "launch", "dynamic_shared_memory_bytes"),
            _path(case, "launch", "dynamic_shared_memory_bytes"),
        ),
        "kernel_source_sha256": (
            _path(model_result, "launch", "kernel_source_sha256"),
            _path(case, "launch", "kernel_source_sha256"),
        ),
        "cache_condition": (
            _normalized_cache(_path(model_result, "cache_behavior", "cache_condition")),
            _normalized_cache(
                environment.get("cache_condition")
                if isinstance(environment, Mapping) else None
            ),
        ),
    }
    verified: list[str] = []
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    core_missing: list[str] = []
    for field, (model_value, runtime_value) in comparisons.items():
        if model_value is None or runtime_value is None:
            missing.append(field)
            if field in {"operator", "shape", "dtype"}:
                core_missing.append(field)
            continue
        if model_value != runtime_value:
            mismatches.append({
                "field": field,
                "model_value": model_value,
                "runtime_value": runtime_value,
            })
        else:
            verified.append(field)
    eligible = not mismatches and not core_missing
    return {
        "status": (
            "mismatch" if mismatches else
            "insufficient_identity" if core_missing else
            "matched" if not missing else "matched_with_missing_optional_fields"
        ),
        "comparison_eligible": eligible,
        "verified_fields": verified,
        "missing_fields": missing,
        "mismatches": mismatches,
        "case_id": case.get("case_id"),
        "record_id": record.get("record_id"),
    }
