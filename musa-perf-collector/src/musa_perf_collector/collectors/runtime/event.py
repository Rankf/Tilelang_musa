"""MUSA Event timing without per-launch profiler activation."""

from __future__ import annotations

from typing import Any, Callable

from ...core.evidence import RUNTIME_DIRECT


def batch_event_samples(
    run: Callable[[], Any], *, warmup_launches: int, repeat_launches: int, batch_size: int,
    instrumented: bool = False, stream: Any | None = None,
) -> list[dict[str, Any]]:
    import torch

    if repeat_launches <= 0 or batch_size <= 0:
        raise ValueError("repeat_launches and batch_size must be positive")
    for _ in range(warmup_launches):
        run()
    torch.musa.synchronize()
    samples = []
    index = 0
    while index < repeat_launches:
        launches = min(batch_size, repeat_launches - index)
        timing_stream = stream if stream is not None else torch.musa.current_stream()
        stream_before = int(timing_stream.stream_id)
        start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
        start.record(timing_stream)
        for _ in range(launches):
            run()
        end.record(timing_stream); end.synchronize()
        stream_after = int(timing_stream.stream_id)
        if stream_after != stream_before:
            raise RuntimeError("current MUSA stream changed inside one Event timing batch")
        samples.append({"sample_index": len(samples), "launch_start": index, "launch_count": launches,
                        "elapsed_ms": start.elapsed_time(end), "per_launch_ms": start.elapsed_time(end) / launches,
                        "timing_kind": "batch_average", "evidence_level": RUNTIME_DIRECT,
                        "instrumented": instrumented, "stream_id": stream_before,
                        "completion_event_synchronized": True,
                        "synchronization_scope": "batch_end_event"})
        index += launches
    return samples
