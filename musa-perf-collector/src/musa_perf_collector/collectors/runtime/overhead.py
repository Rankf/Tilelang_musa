"""ABBA collector-overhead comparison utilities."""

from __future__ import annotations

from statistics import median
from typing import Any, Callable


def compare(baseline: Callable[[], list[dict[str, Any]]], collected: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    """Run baseline/collected/collected/baseline to limit monotonic thermal bias."""
    groups = [("baseline_a", baseline()), ("collected_b1", collected()), ("collected_b2", collected()), ("baseline_a2", baseline())]
    values = {name: [float(item["per_launch_ms"]) for item in samples] for name, samples in groups}
    base = median(values["baseline_a"] + values["baseline_a2"])
    measured = median(values["collected_b1"] + values["collected_b2"])
    absolute = measured - base
    return {"order": [name for name, _ in groups], "baseline_median_ms": base, "collected_median_ms": measured,
            "absolute_overhead_ms": absolute, "relative_overhead": absolute / base if base > 0 else None,
            "samples": values, "status": "complete"}


def kernel_only_from_activity(baseline: dict[str, Any], collected: dict[str, Any]) -> dict[str, Any]:
    """Compare profiler-observed device time without mixing it with callable time.

    Profiler data is deliberately reported as a separate domain: it is an
    aggregate short-window estimate, not a correction applied to Event timing.
    """
    def per_launch(activity: dict[str, Any]) -> float | None:
        launches = int(activity.get("launches", 0))
        # torch_musa currently exposes device kernels as PrivateUse1 rather
        # than DeviceType.MUSA.  CPU launch API events have zero device time.
        values = [float(event["device_time_us"]) for event in activity.get("events", [])
                  if event.get("device_time_us") is not None
                  and str(event.get("device_type", "")) != "DeviceType.CPU"
                  and float(event["device_time_us"]) > 0]
        return sum(values) / launches / 1000.0 if launches > 0 and values else None

    base = per_launch(baseline)
    measured = per_launch(collected)
    if base is None or measured is None:
        return {"status": "unavailable", "reason": "profiler did not expose MUSA device event time"}
    absolute = measured - base
    return {"status": "complete", "baseline_per_launch_ms": base, "instrumented_per_launch_ms": measured,
            "absolute_overhead_ms": absolute, "relative_overhead": absolute / base if base > 0 else None,
            "method": "torch_profiler_musa_activity", "warning": "Do not compare this percentage with callable Event percentage; compare absolute overhead only."}
