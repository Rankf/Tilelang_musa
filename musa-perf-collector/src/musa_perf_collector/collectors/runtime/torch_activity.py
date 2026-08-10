"""Short-window PyTorch MUSA activity collector."""

from __future__ import annotations

import time
from typing import Any, Callable

from ...core.evidence import PROFILER_OBSERVED


def collect(run: Callable[[], Any], *, launches: int, max_duration_ms: float | None = None) -> dict[str, Any]:
    import torch

    if launches <= 0:
        raise ValueError("launches must be positive")
    if max_duration_ms is not None and max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be positive")
    activity = torch.profiler.ProfilerActivity.MUSA
    collector_started = time.monotonic()
    completed_launches = 0
    with torch.profiler.profile(activities=[activity]) as profiler:
        active_started = time.monotonic()
        for _ in range(launches):
            if completed_launches and max_duration_ms is not None and (time.monotonic() - active_started) * 1000 >= max_duration_ms:
                break
            run()
            completed_launches += 1
        torch.musa.synchronize()
        duration_ms = (time.monotonic() - active_started) * 1000
    collector_wall_duration_ms = (time.monotonic() - collector_started) * 1000
    records = []
    for item in profiler.key_averages():
        device_time = getattr(item, "self_device_time_total", None)
        records.append({"name": item.key, "calls": item.count, "self_device_time_us": device_time,
                        "evidence_level": PROFILER_OBSERVED})
    events = []
    for item in profiler.events():
        events.append({"name": item.name, "device_type": str(item.device_type), "device_index": item.device_index,
                       "device_time_us": getattr(item, "device_time_total", None), "self_device_time_us": getattr(item, "self_device_time_total", None),
                       "is_async": item.is_async, "evidence_level": PROFILER_OBSERVED})
    duration_exceeded = max_duration_ms is not None and duration_ms > max_duration_ms
    truncated = completed_launches < launches or duration_exceeded
    return {"status": "partial" if truncated else "complete", "launches": completed_launches,
            "requested_launches": launches, "records": records, "events": events,
            "profile_duration_ms": duration_ms, "max_profile_duration_ms": max_duration_ms,
            "collector_wall_duration_ms": collector_wall_duration_ms,
            "duration_budget_exceeded": duration_exceeded, "truncated": truncated,
            "dropped_records": None, "dropped_records_status": "not_exposed_by_torch_profiler",
            "evidence_level": PROFILER_OBSERVED, "counter_passes": 0, "kernel_serialized": False}
