"""Preallocated online Trace buffers with non-blocking lease semantics."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass
class TraceLease:
    slot_id: int
    device_tensor: Any
    host_tensor: Any | None
    generation: int
    sentinel: int
    state: str = "DEVICE_WRITING"
    copy_done_event: Any | None = None
    start_event: Any | None = None
    end_event: Any | None = None
    ready_event: Any | None = None
    launch_identity: dict[str, Any] | None = None


class TraceBufferPool:
    """A small fail-closed buffer pool; acquisition never waits for capacity."""

    def __init__(
        self, *, trace_extent: int, capacity: int, device: int = 0,
        native_capacity: int | None = None, event_timing_every: int = 1,
    ):
        if trace_extent <= 0 or capacity <= 0:
            raise ValueError("TraceBufferPool extent and capacity must be positive")
        import torch

        self._torch = torch
        self.trace_extent = int(trace_extent)
        native_type = None
        try:
            from ._native_musa_trace import NativeMusaTracePool
            native_type = NativeMusaTracePool
        except ImportError:
            native_type = None
        allocated_capacity = (
            max(int(capacity), int(native_capacity or capacity))
            if native_type is not None else int(capacity))
        self.capacity = allocated_capacity
        self.device = int(device)
        self._lock = threading.RLock()
        self._collector_stream = torch.musa.Stream(device=device)
        self._generation = 0
        self._leases: list[TraceLease] = []
        self._async_d2h = False
        self._native = None
        self.native_error: dict[str, str] | None = None
        # Device tensors are shared with the native pool.  Python host buffers
        # and Events are fallback-only resources and are intentionally not
        # duplicated when the C/MUSART path is available.
        for slot_id in range(allocated_capacity):
            device_tensor = torch.empty(
                (trace_extent,), dtype=torch.int64, device=f"musa:{device}")
            lease = TraceLease(
                slot_id=slot_id, device_tensor=device_tensor,
                host_tensor=None, generation=0,
                sentinel=-(1 << 62) + slot_id, state="FREE")
            self._leases.append(lease)
        try:
            self._native = native_type(
                device_pointers=[lease.device_tensor.data_ptr() for lease in self._leases],
                trace_extent=trace_extent,
                device=device,
                event_timing_every=event_timing_every,
            )
        except (TypeError, RuntimeError) as error:
            self.native_error = {
                "error_type": type(error).__name__, "message": str(error)}
            self._native = None
            self._leases = self._leases[:int(capacity)]
            self.capacity = int(capacity)
            for lease in self._leases:
                try:
                    candidate = torch.empty(
                        (trace_extent,), dtype=torch.int64).pin_memory()
                    if candidate.is_pinned():
                        lease.host_tensor = candidate
                except Exception:
                    lease.host_tensor = None
                lease.start_event = torch.musa.Event(enable_timing=True)
                lease.end_event = torch.musa.Event(enable_timing=True)
                lease.copy_done_event = torch.musa.Event()
                lease.ready_event = torch.musa.Event()
            self._async_d2h = all(
                lease.host_tensor is not None for lease in self._leases)
            if not self._async_d2h:
                for lease in self._leases:
                    lease.host_tensor = None
            # Fallback initialization is startup overhead, never launch timing.
            with torch.musa.stream(self._collector_stream):
                for lease in self._leases:
                    lease.device_tensor.fill_(lease.sentinel)
            self._collector_stream.synchronize()

    @property
    def readback_policy(self) -> str:
        if self._native is not None:
            return "native_deferred_at_flush"
        return "async_after_window" if self._async_d2h else "sync_at_flush"

    @property
    def native_status(self) -> str:
        return "available" if self._native is not None else "fallback"

    def try_acquire(self, launch_identity: dict[str, Any]) -> TraceLease | None:
        if self._native is not None:
            acquired = self._native.acquire(launch_identity)
            if acquired is None:
                return None
            slot_id, sentinel, generation = acquired
            lease = self._leases[int(slot_id)]
            lease.generation = int(generation)
            lease.sentinel = int(sentinel)
            lease.launch_identity = dict(launch_identity)
            lease.state = "DEVICE_WRITING"
            return lease
        with self._lock:
            for lease in self._leases:
                if lease.state != "FREE":
                    continue
                self._generation += 1
                lease.generation = self._generation
                lease.launch_identity = dict(launch_identity)
                lease.state = "DEVICE_WRITING"
                return lease
        return None

    def record_start(self, lease: TraceLease) -> None:
        if self._native is not None:
            stream = self._torch.musa.current_stream(self.device)
            self._native.record_start(
                lease.slot_id, int(getattr(stream, "musa_stream", 0)))
            return
        lease.start_event.record(self._torch.musa.current_stream(self.device))

    def update_identity(self, lease: TraceLease, launch_identity: dict[str, Any]) -> None:
        if self._native is not None:
            self._native.update_identity(lease.slot_id, launch_identity)
            lease.launch_identity = dict(launch_identity)
            return
        with self._lock:
            if lease.state != "DEVICE_WRITING":
                raise RuntimeError("trace lease identity can only change before device launch")
            lease.launch_identity = dict(launch_identity)

    def enqueue_readback(self, lease: TraceLease) -> None:
        if self._native is not None:
            stream = self._torch.musa.current_stream(self.device)
            self._native.enqueue_readback(
                lease.slot_id, int(getattr(stream, "musa_stream", 0)))
            lease.state = "D2H_COPYING"
            return
        torch = self._torch
        producer = torch.musa.current_stream(self.device)
        lease.end_event.record(producer)
        if lease.host_tensor is None:
            lease.state = "DEVICE_READY_SYNC_PENDING"
            return
        with torch.musa.stream(self._collector_stream):
            self._collector_stream.wait_event(lease.end_event)
            lease.host_tensor.copy_(lease.device_tensor, non_blocking=True)
            lease.copy_done_event.record(self._collector_stream)
        lease.state = "D2H_COPYING"

    def abort(self, lease: TraceLease) -> None:
        if self._native is not None:
            self._native.abort(lease.slot_id)
            lease.state = "INVALID"
            return
        with self._lock:
            lease.state = "INVALID"

    def _complete(self, lease: TraceLease, *, wait: bool) -> dict[str, Any] | None:
        if lease.state == "D2H_COPYING":
            if not wait and not lease.copy_done_event.query():
                return None
            lease.copy_done_event.synchronize()
            values = [int(value) for value in lease.host_tensor.tolist()]
        elif lease.state == "DEVICE_READY_SYNC_PENDING":
            if not wait and not lease.end_event.query():
                return None
            if wait:
                lease.end_event.synchronize()
            values = [int(value) for value in lease.device_tensor.cpu().tolist()]
        else:
            return None
        elapsed_ms = float(lease.start_event.elapsed_time(lease.end_event))
        result = {
            "slot_id": lease.slot_id,
            "generation": lease.generation,
            "sentinel": lease.sentinel,
            "values": values,
            "kernel_event_ms": elapsed_ms,
            "readback_policy": self.readback_policy,
            "launch_identity": dict(lease.launch_identity or {}),
        }
        # Reset on the collector stream before the lease becomes reusable.
        lease.sentinel = -(1 << 62) + self._generation + self.capacity + lease.slot_id
        with self._torch.musa.stream(self._collector_stream):
            lease.device_tensor.fill_(lease.sentinel)
            lease.ready_event.record(self._collector_stream)
        # Do not expose the buffer until reset completion. poll() promotes it.
        lease.state = "RESETTING"
        return result

    @property
    def pending_count(self) -> int:
        if self._native is not None:
            return int(self._native.pending_count)
        with self._lock:
            return sum(
                lease.state in {
                    "DEVICE_WRITING", "DEVICE_READY_SYNC_PENDING",
                    "D2H_COPYING", "RESETTING",
                }
                for lease in self._leases
            )

    def poll(self, *, wait: bool = False) -> list[dict[str, Any]]:
        if self._native is not None:
            completed = list(self._native.poll(wait=wait))
            for record in completed:
                record["readback_policy"] = self.readback_policy
                lease = self._leases[int(record["slot_id"])]
                lease.state = "RESETTING"
            return completed
        completed: list[dict[str, Any]] = []
        with self._lock:
            for lease in self._leases:
                if lease.state == "RESETTING":
                    if wait:
                        lease.ready_event.synchronize()
                    if wait or lease.ready_event.query():
                        lease.state = "FREE"
                        lease.launch_identity = None
                    continue
                result = self._complete(lease, wait=wait)
                if result is not None:
                    completed.append(result)
            if wait:
                # A completion above may have scheduled reset work.
                for lease in self._leases:
                    if lease.state == "RESETTING":
                        lease.ready_event.synchronize()
                        lease.state = "FREE"
                        lease.launch_identity = None
        return completed

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None
