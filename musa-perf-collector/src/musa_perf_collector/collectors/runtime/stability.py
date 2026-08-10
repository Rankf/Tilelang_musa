"""Thermal, clock, and timing stability assessment for baseline evidence."""

from __future__ import annotations

import json
from statistics import median
from typing import Any


def _snapshot_point(snapshot: dict[str, Any]) -> dict[str, float] | None:
    if int(snapshot.get("returncode", -1)) != 0 or not snapshot.get("raw_json"):
        return None
    try:
        document = json.loads(snapshot["raw_json"])
        gpu = document["GPU"][0]
        temperature = float(str(gpu["Temperature"]["GPU Current Temp"]).removesuffix("C"))
        frequency = float(str(gpu["Clocks"]["Graphics"]).removesuffix("MHz"))
        power_key = next(key for key in gpu["Power Readings"] if key.strip() == "Power Draw")
        power = float(str(gpu["Power Readings"][power_key]).removesuffix("W"))
        return {"temperature_c": temperature, "frequency_mhz": frequency, "power_w": power,
                "host_monotonic_s": float(snapshot["host_monotonic_s"])}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, StopIteration):
        return None


def assess(
    before: dict[str, Any], active: dict[str, Any], after: dict[str, Any],
    timing_samples: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    """Assess stability without inventing per-TileOp temperature or power."""
    if not bool(config.get("enabled", True)):
        return {"status": "disabled", "required": bool(config.get("required", False)), "reasons": []}
    points = []
    first = _snapshot_point(before)
    last = _snapshot_point(after)
    if first is not None:
        points.append({"source": "before_snapshot", **first})
    points.extend({"source": "active_window", **item} for item in active.get("samples", []))
    if last is not None:
        points.append({"source": "after_snapshot", **last})
    temperatures = [float(item["temperature_c"]) for item in points]
    frequencies = [float(item["frequency_mhz"]) for item in points]
    powers = [float(item["power_w"]) for item in points]
    timing = [float(item["per_launch_ms"]) for item in timing_samples]
    minimum_points = int(config.get("min_telemetry_points", 3))
    minimum_batches = int(config.get("min_timing_batches", 4))
    reasons: list[dict[str, Any]] = []
    if len(points) < minimum_points:
        reasons.append({"code": "INSUFFICIENT_TELEMETRY_POINTS", "observed": len(points), "required": minimum_points})
    if len(timing) < minimum_batches:
        reasons.append({"code": "INSUFFICIENT_TIMING_BATCHES", "observed": len(timing), "required": minimum_batches})
    thermal: dict[str, Any] = {"point_count": len(points), "points": points}
    if points:
        temperature_delta = max(temperatures) - min(temperatures)
        frequency_delta_pct = ((max(frequencies) - min(frequencies)) / median(frequencies) * 100.0
                               if median(frequencies) else float("inf"))
        thermal.update({
            "temperature_min_c": min(temperatures), "temperature_max_c": max(temperatures),
            "temperature_delta_c": temperature_delta, "frequency_min_mhz": min(frequencies),
            "frequency_max_mhz": max(frequencies), "frequency_delta_pct": frequency_delta_pct,
            "power_min_w": min(powers), "power_max_w": max(powers),
        })
        if max(temperatures) > float(config.get("max_temperature_c", 90.0)):
            reasons.append({"code": "TEMPERATURE_LIMIT_EXCEEDED", "observed": max(temperatures),
                            "allowed": float(config.get("max_temperature_c", 90.0))})
        if temperature_delta > float(config.get("max_temperature_delta_c", 5.0)):
            reasons.append({"code": "TEMPERATURE_DRIFT_EXCEEDED", "observed": temperature_delta,
                            "allowed": float(config.get("max_temperature_delta_c", 5.0))})
        if frequency_delta_pct > float(config.get("max_frequency_delta_pct", 5.0)):
            reasons.append({"code": "FREQUENCY_DRIFT_EXCEEDED", "observed": frequency_delta_pct,
                            "allowed": float(config.get("max_frequency_delta_pct", 5.0))})
    timing_result: dict[str, Any] = {"batch_count": len(timing)}
    if timing:
        split = max(1, len(timing) // 2)
        early = median(timing[:split])
        late = median(timing[-split:])
        drift_pct = abs(late - early) / early * 100.0 if early else float("inf")
        timing_result.update({"early_median_per_launch_ms": early, "late_median_per_launch_ms": late,
                              "median_drift_pct": drift_pct})
        if len(timing) >= minimum_batches and drift_pct > float(config.get("max_timing_drift_pct", 10.0)):
            reasons.append({"code": "TIMING_DRIFT_EXCEEDED", "observed": drift_pct,
                            "allowed": float(config.get("max_timing_drift_pct", 10.0))})
    insufficient = any(item["code"].startswith("INSUFFICIENT_") for item in reasons)
    unstable = any(not item["code"].startswith("INSUFFICIENT_") for item in reasons)
    status = "unstable" if unstable else "insufficient" if insufficient else "stable"
    return {"status": status, "required": bool(config.get("required", False)),
            "thermal": thermal, "timing": timing_result, "reasons": reasons,
            "semantics": "session_window_only_not_per_tileop"}
