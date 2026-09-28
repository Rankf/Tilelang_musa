"""S5000 access-class-aware memory service model.

The upstream TileSight rule (convert every level to time and take ``max``) is
kept by the caller as the legacy steady-state roofline.  This module layers a
versioned, independently calibrated startup/steady/non-overlap/drain model on
top.  Target operator measurements are never accepted as prediction input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_CALIBRATION = Path(__file__).with_name("s5000_service_calibration_v1.json")


def classify_memory_access(
    *, ordinary_read_bytes: float, ordinary_write_bytes: float,
    tma_read_bytes: float, cache_condition: str,
) -> dict[str, Any]:
    """Classify a path from model-visible access semantics, not profiler data."""
    if cache_condition not in {"cold", "warm", "intra_kernel_reuse"}:
        raise ValueError(
            "cache_condition must be cold, warm, or intra_kernel_reuse; got "
            f"{cache_condition!r}"
        )
    normalized_cache = "cold" if cache_condition == "cold" else "warm"
    if tma_read_bytes > 0:
        access_class = "tma_read_mixed" if ordinary_write_bytes else "tma_read_only"
    elif ordinary_read_bytes <= 0 and ordinary_write_bytes > 0:
        access_class = "ordinary_cached_write_only"
    elif ordinary_write_bytes <= 0:
        access_class = "ordinary_cached_read_only"
    else:
        ratio = ordinary_read_bytes / ordinary_write_bytes
        if 0.8 <= ratio <= 1.25:
            access_class = "ordinary_cached_read_write_1to1"
        elif 1.75 <= ratio <= 2.25:
            access_class = "ordinary_cached_two_read_one_write"
        else:
            access_class = "ordinary_cached_mixed"
    return {
        "access_class": access_class,
        "cache_condition": normalized_cache,
        "ordinary_read_bytes": ordinary_read_bytes,
        "ordinary_write_bytes": ordinary_write_bytes,
        "tma_read_bytes": tma_read_bytes,
        "read_write_ratio": (
            ordinary_read_bytes / ordinary_write_bytes
            if ordinary_write_bytes else None
        ),
        "value_kind": "semantic_derived",
    }


def _load_registry() -> dict[str, Any]:
    payload = json.loads(_CALIBRATION.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "s5000/memory-service-calibration-v1":
        raise ValueError("invalid S5000 memory service calibration schema")
    return payload


def _curve(access_class: str, cache_condition: str) -> dict[str, Any] | None:
    matches = [
        item for item in _load_registry()["curves"]
        if item["access_class"] == access_class
        and item["cache_condition"] == cache_condition
    ]
    if len(matches) > 1:
        raise ValueError("duplicate S5000 memory service calibration curve")
    return matches[0] if matches else None


def _effective_bandwidth_gb_s(point: dict[str, Any]) -> float:
    return point["logical_bytes"] / point["latency_us"] / 1_000.0


def _interpolate_curve(curve: dict[str, Any], logical_bytes: float) -> dict[str, Any]:
    observations = sorted(curve["observations"], key=lambda item: item["logical_bytes"])
    if any(item.get("sample_count", 0) < 3 for item in observations):
        raise ValueError("calibration curve requires at least three samples per point")
    if len(observations) < 2:
        raise ValueError("calibration curve requires at least two workload sizes")

    if logical_bytes <= observations[0]["logical_bytes"]:
        lower, upper = observations[0], observations[1]
        position = 0.0
        extrapolated = logical_bytes < observations[0]["logical_bytes"]
    elif logical_bytes >= observations[-1]["logical_bytes"]:
        lower, upper = observations[-2], observations[-1]
        position = 1.0
        extrapolated = logical_bytes > observations[-1]["logical_bytes"]
    else:
        lower, upper = observations[0], observations[1]
        for left, right in zip(observations, observations[1:]):
            if left["logical_bytes"] <= logical_bytes <= right["logical_bytes"]:
                lower, upper = left, right
                break
        position = (
            (math.log(logical_bytes) - math.log(lower["logical_bytes"]))
            / (math.log(upper["logical_bytes"]) - math.log(lower["logical_bytes"]))
        )
        extrapolated = False

    lower_bw = _effective_bandwidth_gb_s(lower)
    upper_bw = _effective_bandwidth_gb_s(upper)
    # Bandwidth changes multiplicatively across cache/capacity regimes, so the
    # interpolation is linear in log(bytes)-log(GB/s), not a target-op fit.
    predicted_bw = math.exp(
        math.log(lower_bw) + position * (math.log(upper_bw) - math.log(lower_bw))
    )
    return {
        "effective_bandwidth_gb_s": predicted_bw,
        "bandwidth_range_gb_s": [min(lower_bw, upper_bw), max(lower_bw, upper_bw)],
        "bracket_case_ids": [lower["case_id"], upper["case_id"]],
        "bracket_log_position": position,
        "extrapolated": extrapolated,
        "sample_count_per_point_min": min(item["sample_count"] for item in observations),
        "curve_point_count": len(observations),
    }


def estimate_memory_service(
    *, service_costs_us: dict[str, float], logical_traffic_bytes: float,
    access: dict[str, Any], drain_us: float = 0.0,
) -> dict[str, Any]:
    """Return calibrated startup/steady/non-overlap/drain service components."""
    service_costs_us = {
        str(level): float(value) for level, value in service_costs_us.items()
    }
    logical_traffic_bytes = float(logical_traffic_bytes)
    # Preserved upstream core result: all memory levels overlap perfectly and
    # the largest calibrated bandwidth service cost is the memory time.
    legacy_level = max(service_costs_us, key=service_costs_us.get)
    legacy_us = service_costs_us[legacy_level]
    curve = _curve(access["access_class"], access["cache_condition"])
    if curve is None or logical_traffic_bytes <= 0:
        return {
            "status": "legacy_fallback_no_matching_access_curve",
            "model_version": "legacy_upstream_max_overlap",
            "access": access,
            "startup_us": 0.0,
            "steady_us": legacy_us,
            "non_overlap_us": 0.0,
            "drain_us": drain_us,
            "total_us": legacy_us + drain_us,
            "legacy_upstream": {
                "critical_level": legacy_level,
                "service_cost_us": legacy_us,
                "overlap_rule": "max(L1 Cache, L2, DDR, Shared service cost)",
            },
            "confidence": "low",
        }

    interpolated = _interpolate_curve(curve, logical_traffic_bytes)
    curve_total_us = (
        logical_traffic_bytes / interpolated["effective_bandwidth_gb_s"] / 1_000.0
    )
    startup_us = min(float(curve.get("startup_us", 0.0)), curve_total_us)
    drain_us = max(float(drain_us), 0.0)
    non_overlap_us = max(curve_total_us - startup_us - legacy_us - drain_us, 0.0)
    total_us = startup_us + legacy_us + non_overlap_us + drain_us
    secondary = sorted(service_costs_us.values(), reverse=True)[1]
    low_bw, high_bw = interpolated["bandwidth_range_gb_s"]
    range_us = sorted([
        max(legacy_us, logical_traffic_bytes / high_bw / 1_000.0),
        max(legacy_us, logical_traffic_bytes / low_bw / 1_000.0),
    ])
    return {
        "status": "calibrated_access_curve",
        "model_version": "s5000_access_service_v1",
        "access": access,
        "startup_us": startup_us,
        "steady_us": legacy_us,
        "non_overlap_us": non_overlap_us,
        "non_overlap_ratio_of_secondary": (
            non_overlap_us / secondary if secondary > 0 else 0.0
        ),
        "drain_us": drain_us,
        "total_us": total_us,
        "latency_range_us": range_us,
        "effective_bandwidth_gb_s": interpolated["effective_bandwidth_gb_s"],
        "calibration": {
            **interpolated,
            "schema_version": "s5000/memory-service-calibration-v1",
            "source": "P3 independent three-process microbenchmarks",
            "target_operator_measurement_used": False,
            "limitations": curve.get("limitations", []),
        },
        "legacy_upstream": {
            "critical_level": legacy_level,
            "service_cost_us": legacy_us,
            "overlap_rule": "max(L1 Cache, L2, DDR, Shared service cost)",
        },
        "confidence": "low" if interpolated["extrapolated"] else "medium",
    }
