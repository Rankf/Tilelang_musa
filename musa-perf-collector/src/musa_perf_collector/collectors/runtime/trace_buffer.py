"""Trace-buffer lifecycle bookkeeping and conservative MUSA D2H capability probing.

This module intentionally does not label a copy as asynchronous merely because a
``non_blocking`` argument exists.  A usable pinned host allocation is a hard
precondition; callers otherwise use the explicit post-window synchronous path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TraceBufferState(str, Enum):
    FREE = "FREE"
    DEVICE_WRITING = "DEVICE_WRITING"
    DEVICE_READY = "DEVICE_READY"
    D2H_COPYING = "D2H_COPYING"
    HOST_READY = "HOST_READY"


_ALLOWED = {
    TraceBufferState.FREE: {TraceBufferState.DEVICE_WRITING},
    TraceBufferState.DEVICE_WRITING: {TraceBufferState.DEVICE_READY},
    TraceBufferState.DEVICE_READY: {TraceBufferState.D2H_COPYING, TraceBufferState.HOST_READY},
    TraceBufferState.D2H_COPYING: {TraceBufferState.HOST_READY},
    TraceBufferState.HOST_READY: {TraceBufferState.FREE},
}


@dataclass
class TraceBufferSlot:
    slot_id: int
    state: TraceBufferState = TraceBufferState.FREE
    generation_id: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, target: TraceBufferState) -> None:
        if target not in _ALLOWED[self.state]:
            raise RuntimeError(f"illegal trace slot transition {self.state.value} -> {target.value}")
        self.transitions.append({"from": self.state.value, "to": target.value, "generation_id": self.generation_id})
        self.state = target

    def begin_write(self) -> None:
        self.transition(TraceBufferState.DEVICE_WRITING)
        self.generation_id += 1

    def manifest(self) -> dict[str, Any]:
        return {"slot_id": self.slot_id, "generation_id": self.generation_id,
                "state": self.state.value, "transitions": self.transitions}


def probe_async_d2h() -> dict[str, Any]:
    """Verify pinned allocation plus independent-stream/Event D2H semantics."""
    import torch
    import time

    result: dict[str, Any] = {"status": "unavailable", "policy": "sync_after_window",
                              "reason": None, "pinned_host_memory": False,
                              "probe_bytes": 16 * 1024 * 1024}
    try:
        # torch_musa patches Tensor.pin_memory(device="musa").  In this build,
        # torch.empty(pin_memory=True) does not reliably select that allocator.
        count = int(result["probe_bytes"]) // 8
        host = torch.empty((count,), dtype=torch.int64).pin_memory()
        # torch_musa's patched no-argument method already targets MUSA; passing
        # the device explicitly is deprecated in this PyTorch alpha build.
        result["pinned_host_memory"] = bool(host.is_pinned())
        if not result["pinned_host_memory"]:
            result["reason"] = "explicit Tensor.pin_memory() is not reported pinned for MUSA"
            return result
        device = torch.arange(count, device="musa", dtype=torch.int64)
        producer = torch.musa.current_stream()
        collector = torch.musa.Stream()
        ready = torch.musa.Event()
        done = torch.musa.Event()
        ready.record(producer)
        started = time.perf_counter()
        with torch.musa.stream(collector):
            collector.wait_event(ready)
            host.copy_(device, non_blocking=True)
            done.record(collector)
        result["submit_s"] = time.perf_counter() - started
        result["done_immediately"] = bool(done.query())
        done.synchronize()
        result["complete_s"] = time.perf_counter() - started
        result["data_verified"] = bool(host[0].item() == 0 and host[-1].item() == count - 1)
        if not result["data_verified"]:
            result["reason"] = "asynchronous D2H completed with incorrect data"
            return result
        if result["done_immediately"]:
            result["reason"] = "probe copy completed before asynchronous behavior could be observed"
            return result
        result.update({"status": "available", "policy": "async_after_window", "reason": None,
                       "semantics": "producer_event_to_collector_stream_nonblocking_d2h_copy_done_event"})
        return result
    except Exception as error:  # pragma: no cover - backend dependent
        result["reason"] = f"pinned host allocation failed: {error}"
        return result
