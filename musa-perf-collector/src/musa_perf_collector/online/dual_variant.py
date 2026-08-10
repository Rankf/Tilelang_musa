"""Launch dispatcher preserving the baseline JITKernel public surface."""

from __future__ import annotations

import itertools
import time
from typing import Any

try:
    from ._native_dispatch import NativePeriodicDispatcher
except ImportError:  # Source-only installs retain the reviewed Python fallback.
    NativePeriodicDispatcher = None


class OnlineDualVariantKernel:
    def __init__(self, baseline: Any, instrumented: Any, session: Any, kernel_id: str, plan: Any, pool: Any):
        self._baseline = baseline
        self._instrumented = instrumented
        self._session = session
        self._kernel_id = kernel_id
        self._plan = plan
        self._pool = pool
        self._owner_pid = session.pid
        self._launch_counter = itertools.count()
        selector = session.config.launch_selector or {"every": 64, "start": 0}
        self._periodic_every = int(selector["every"]) if "every" in selector else None
        self._periodic_start = int(selector.get("start", 0))
        self._exact_indices = (
            frozenset(int(value) for value in selector["indices"])
            if "indices" in selector else None)
        self._terminal_disabled_at: int | None = None
        self._pending_launch_records: list[dict] = []
        self._pending_instrumented_launches = 0
        self._pending_diagnostics: list[dict] = []
        self._cached_device_id: int | None = None
        self._cached_stream_id: int | None = None
        self._cached_ts = 0.0
        self.online_instrumentation = {
            "mode": "eager_dual_variant",
            "kernel_id": kernel_id,
            "plan_sha256": plan.plan_sha256,
            "baseline_kernel": baseline,
            "instrumented_kernel": instrumented,
            "readback_policy": pool.readback_policy,
            "dispatch_path": "python",
        }
        self._native_callable = None
        if self._periodic_every is not None and NativePeriodicDispatcher is not None:
            self.online_instrumentation["dispatch_path"] = "native_c_vectorcall"
            self._native_callable = NativePeriodicDispatcher(
                baseline=baseline,
                selected=self._native_selected,
                forked=self._run_forked_baseline,
                every=self._periodic_every,
                start=self._periodic_start,
                owner_pid=self._owner_pid,
                error_callback=self._native_baseline_error,
                online_instrumentation=self.online_instrumentation,
            )
            session.register_launch_summary_provider(self._native_callable)
        else:
            session.register_launch_summary_provider(self)
        session.register_dual_variant_kernel(self)

    def as_callable(self) -> Any:
        """Return the C fast-path callable when available, otherwise this fallback."""
        return self._native_callable if self._native_callable is not None else self

    @property
    def dispatch_path(self) -> str:
        return str(self.online_instrumentation["dispatch_path"])

    def _native_baseline_error(
        self, launch_index: int, reason: str, error_type: str,
    ) -> None:
        """Record the exceptional fast path; successful launches never call Python."""
        device_id, stream_id = self._cached_stream_metadata()
        self._session.record_launch(
            self._kernel_id, launch_index, device_id, stream_id, None,
            status="application_error", error_type=error_type,
            selection_reason=reason)

    def _run_forked_baseline(self, *args: Any, **kwargs: Any) -> Any:
        """Re-enter the audited session reset path only in a forked child."""
        launch_index = self._session.next_launch_index(self._kernel_id)
        device_id, stream_id = self._session.stream_metadata()
        started = time.perf_counter_ns()
        try:
            result = self._baseline(*args, **kwargs)
        except BaseException as error:
            self._pending_launch_records.append({
                "kernel_id": self._kernel_id, "launch_index": launch_index,
                "device_id": device_id, "stream_id": stream_id,
                "started_ns": started, "status": "application_error",
                "error_type": type(error).__name__,
                "selection_reason": "forked_child_instrumentation_disabled",
            })
            raise
        self._session.record_launch(
            self._kernel_id, launch_index, device_id, stream_id, started,
            status="baseline", selection_reason="forked_child_instrumentation_disabled")
        return result

    def _native_selected(
        self, launch_index: int, *args: Any, **kwargs: Any,
    ) -> tuple[Any, str | None]:
        """Run the existing audited slow path only for a C-selected launch."""
        selected, reason, _selection_ordinal = self._session.select_launch(
            self._kernel_id, launch_index, None,
            trace_extent=self._plan.trace_extent)
        if not selected:
            self._session.note_unselected_launch(reason)
            terminal_reason = reason if reason in {
                "invalid_trace_extent", "sampling_window_closed",
                "duration_budget_exhausted", "instrumented_launch_budget_exhausted",
                "trace_record_budget_exhausted", "trace_byte_budget_exhausted",
                "kernel_not_selected",
            } else None
            return self._run_fast_baseline(
                launch_index, reason, args, kwargs), terminal_reason

        self._session.poll_trace_pool(self._kernel_id, self._plan, self._pool)
        device_id, stream_id = self._cached_stream_metadata()
        if device_id is not None and device_id != self._pool.device:
            self._session.record_diagnostic(
                "ONLINE_TRACE_DEVICE_MISMATCH", kernel_id=self._kernel_id,
                launch_index=launch_index, expected_device=self._pool.device,
                actual_device=device_id)
            started = time.perf_counter_ns()
            try:
                result = self._baseline(*args, **kwargs)
            except BaseException as error:
                self._session.record_launch(
                    self._kernel_id, launch_index, device_id, stream_id, started,
                    status="application_error", error_type=type(error).__name__,
                    selection_reason="trace_device_mismatch")
                raise
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="baseline", selection_reason="trace_device_mismatch")
            return result, "trace_device_mismatch"
        identity = {
            "process_id": self._session.pid,
            "kernel_id": self._kernel_id,
            "device_id": device_id,
            "stream_id": stream_id,
            "launch_index": launch_index,
            "plan_sha256": self._plan.plan_sha256,
        }
        lease = self._pool.try_acquire(identity)
        if lease is None:
            self._session.record_diagnostic(
                "ONLINE_TRACE_BUFFER_UNAVAILABLE", kernel_id=self._kernel_id,
                launch_index=launch_index, readback_policy=self._pool.readback_policy)
            started = time.perf_counter_ns()
            try:
                result = self._baseline(*args, **kwargs)
            except BaseException as error:
                self._session.record_launch(
                    self._kernel_id, launch_index, device_id, stream_id, started,
                    status="application_error", error_type=type(error).__name__,
                    selection_reason="trace_buffer_unavailable")
                raise
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="baseline", selection_reason="trace_buffer_unavailable")
            terminal_reason = (
                "trace_buffer_capacity_exhausted"
                if self._pool.readback_policy == "native_deferred_at_flush"
                else None)
            return result, terminal_reason
        dispatch_ordinal = self._session.claim_instrumented_dispatch()
        identity.update({
            "instrumented_variant_launch_index": dispatch_ordinal,
            "instrumented_variant_cold_start": dispatch_ordinal == 0,
        })
        self._pool.update_identity(lease, identity)
        started = time.perf_counter_ns()
        try:
            self._pool.record_start(lease)
            result = self._instrumented(*args, lease.device_tensor, **kwargs)
            self._pool.enqueue_readback(lease)
        except BaseException as error:
            try:
                self._pool.abort(lease)
            except Exception as abort_error:
                self._session.record_diagnostic(
                    "ONLINE_TRACE_ABORT_FAILED", kernel_id=self._kernel_id,
                    launch_index=launch_index,
                    original_error_type=type(error).__name__,
                    abort_error_type=type(abort_error).__name__,
                    abort_message=str(abort_error))
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="instrumented_error", error_type=type(error).__name__,
                selection_reason="selected")
            raise
        self._pending_launch_records.append({
            "kernel_id": self._kernel_id, "launch_index": launch_index,
            "device_id": device_id, "stream_id": stream_id,
            "started_ns": started, "status": "instrumented",
            "selection_reason": "selected",
        })
        self._pending_instrumented_launches += 1
        return result, None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Device tensors, streams, and events created before fork are never
        # reused in the child. The inherited baseline remains transparent, but
        # this wrapper is permanently instrumentation-disabled outside its
        # creation process.
        if not self._session.instrumentation_process_matches(self._owner_pid):
            launch_index = self._session.next_launch_index(self._kernel_id)
            device_id, stream_id = self._cached_stream_metadata()
            started = time.perf_counter_ns()
            try:
                result = self._baseline(*args, **kwargs)
            except BaseException as error:
                self._session.record_launch(
                    self._kernel_id, launch_index, device_id, stream_id, started,
                    status="application_error", error_type=type(error).__name__,
                    selection_reason="forked_child_instrumentation_disabled")
                raise
            self._pending_launch_records.append({
                "kernel_id": self._kernel_id, "launch_index": launch_index,
                "device_id": device_id, "stream_id": stream_id,
                "started_ns": started, "status": "baseline",
                "selection_reason": "forked_child_instrumentation_disabled",
            })
            return result
        # Production returns the native callable directly. Keeping this
        # delegation makes direct wrapper use share the same single counter.
        if self._native_callable is not None:
            return self._native_callable(*args, **kwargs)
        if self._periodic_every is not None or self._exact_indices is not None:
            launch_index = next(self._launch_counter)
            if self._terminal_disabled_at is not None:
                return self._run_fast_baseline(
                    launch_index, self._terminal_disabled_reason, args, kwargs)
            if self._periodic_every is not None:
                candidate = (
                    launch_index >= self._periodic_start
                    and (launch_index - self._periodic_start) % self._periodic_every == 0)
            else:
                candidate = launch_index in self._exact_indices
            if not candidate:
                return self._run_fast_baseline(
                    launch_index, "launch_not_selected", args, kwargs)
            selected, reason, _selection_ordinal = self._session.select_launch(
                self._kernel_id, launch_index, None,
                trace_extent=self._plan.trace_extent)
            device_id = stream_id = None
        else:
            # Probability selection deliberately remains stream-aware and uses
            # the full path. Periodic/index selectors never enter this branch.
            launch_index = self._session.next_launch_index(self._kernel_id)
            device_id, stream_id = self._cached_stream_metadata()
            selected, reason, _selection_ordinal = self._session.select_launch(
                self._kernel_id, launch_index, stream_id,
                trace_extent=self._plan.trace_extent)
        if not selected:
            self._session.note_unselected_launch(reason)
            if reason in {
                "invalid_trace_extent", "sampling_window_closed",
                "duration_budget_exhausted", "instrumented_launch_budget_exhausted",
                "trace_record_budget_exhausted", "trace_byte_budget_exhausted",
                "kernel_not_selected",
            } and (self._periodic_every is not None or self._exact_indices is not None):
                self._terminal_disabled_at = launch_index
                self._terminal_disabled_reason = reason
            return self._run_fast_baseline(launch_index, reason, args, kwargs)

        # Metadata, trace polling, and per-launch records exist only for a
        # selected launch. This is the intentional periodic-sampling slow path.
        self._session.poll_trace_pool(self._kernel_id, self._plan, self._pool)
        device_id, stream_id = self._cached_stream_metadata()
        if device_id is not None and device_id != self._pool.device:
            self._session.record_diagnostic(
                "ONLINE_TRACE_DEVICE_MISMATCH", kernel_id=self._kernel_id,
                launch_index=launch_index, expected_device=self._pool.device,
                actual_device=device_id)
            started = time.perf_counter_ns()
            try:
                result = self._baseline(*args, **kwargs)
            except BaseException as error:
                self._session.record_launch(
                    self._kernel_id, launch_index, device_id, stream_id, started,
                    status="application_error", error_type=type(error).__name__,
                    selection_reason="trace_device_mismatch")
                raise
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="baseline", selection_reason="trace_device_mismatch")
            if self._periodic_every is not None or self._exact_indices is not None:
                self._terminal_disabled_at = launch_index
                self._terminal_disabled_reason = "trace_device_mismatch"
            return result

        identity = {
            "process_id": self._session.pid,
            "kernel_id": self._kernel_id,
            "device_id": device_id,
            "stream_id": stream_id,
            "launch_index": launch_index,
            "plan_sha256": self._plan.plan_sha256,
        }
        lease = self._pool.try_acquire(identity)
        if lease is None:
            self._session.record_diagnostic(
                "ONLINE_TRACE_BUFFER_UNAVAILABLE", kernel_id=self._kernel_id,
                launch_index=launch_index, readback_policy=self._pool.readback_policy)
            started = time.perf_counter_ns()
            try:
                result = self._baseline(*args, **kwargs)
            except BaseException as error:
                self._session.record_launch(
                    self._kernel_id, launch_index, device_id, stream_id, started,
                    status="application_error", error_type=type(error).__name__,
                    selection_reason="trace_buffer_unavailable")
                raise
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="baseline", selection_reason="trace_buffer_unavailable")
            if (
                self._pool.readback_policy == "native_deferred_at_flush"
                and (self._periodic_every is not None or self._exact_indices is not None)
            ):
                self._terminal_disabled_at = launch_index
                self._terminal_disabled_reason = "trace_buffer_capacity_exhausted"
            return result
        dispatch_ordinal = self._session.claim_instrumented_dispatch()
        identity.update({
            "instrumented_variant_launch_index": dispatch_ordinal,
            "instrumented_variant_cold_start": dispatch_ordinal == 0,
        })
        self._pool.update_identity(lease, identity)
        started = time.perf_counter_ns()
        try:
            self._pool.record_start(lease)
            result = self._instrumented(*args, lease.device_tensor, **kwargs)
            self._pool.enqueue_readback(lease)
        except BaseException as error:
            try:
                self._pool.abort(lease)
            except Exception as abort_error:
                self._session.record_diagnostic(
                    "ONLINE_TRACE_ABORT_FAILED", kernel_id=self._kernel_id,
                    launch_index=launch_index,
                    original_error_type=type(error).__name__,
                    abort_error_type=type(abort_error).__name__,
                    abort_message=str(abort_error))
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="instrumented_error", error_type=type(error).__name__,
                selection_reason="selected")
            raise
        self._pending_launch_records.append({
            "kernel_id": self._kernel_id, "launch_index": launch_index,
            "device_id": device_id, "stream_id": stream_id,
            "started_ns": started, "status": "instrumented",
            "selection_reason": "selected",
        })
        self._pending_instrumented_launches += 1
        return result

    def _run_fast_baseline(
        self, launch_index: int, reason: str,
        args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> Any:
        try:
            return self._baseline(*args, **kwargs)
        except BaseException as error:
            device_id, stream_id = self._cached_stream_metadata()
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, None,
                status="application_error", error_type=type(error).__name__,
                selection_reason=reason)
            raise

    def online_launch_summary(self) -> dict[str, Any]:
        """Return lock-free periodic/index counts for session-time aggregation."""
        total = int(self._launch_counter.__reduce__()[1][0])
        if self._periodic_every is not None:
            if total <= self._periodic_start:
                candidates = 0
            else:
                candidates = 1 + (total - 1 - self._periodic_start) // self._periodic_every
        elif self._exact_indices is not None:
            candidates = sum(index < total for index in self._exact_indices)
        else:
            return {
                "total_launch_count": 0,
                "fast_path_launch_count": 0,
                "fast_path_skip_reasons": {},
            }
        ordinary_non_candidates = total - candidates
        disabled_candidates = 0
        if self._terminal_disabled_at is not None:
            active_total = min(total, self._terminal_disabled_at + 1)
            if self._periodic_every is not None:
                if active_total <= self._periodic_start:
                    active_candidates = 0
                else:
                    active_candidates = (
                        1 + (active_total - 1 - self._periodic_start)
                        // self._periodic_every)
            else:
                active_candidates = sum(
                    index < active_total for index in self._exact_indices)
            disabled_candidates = candidates - active_candidates
        non_candidates = ordinary_non_candidates + disabled_candidates
        reasons: dict[str, int] = {}
        if ordinary_non_candidates:
            reasons["launch_not_selected"] = ordinary_non_candidates
        if disabled_candidates:
            reasons[self._terminal_disabled_reason] = disabled_candidates
        return {
            "total_launch_count": total,
            "fast_path_launch_count": non_candidates,
            "fast_path_skip_reasons": reasons,
        }

    def flush_pending(self) -> None:
        for record in self._pending_launch_records:
            self._session.record_launch(**record)
        self._pending_launch_records.clear()
        for _ in range(self._pending_instrumented_launches):
            self._session.record_instrumented_launch()
        self._pending_instrumented_launches = 0


    def _cached_stream_metadata(self):
        now = time.monotonic()
        if now - self._cached_ts > 0.1:
            self._cached_device_id, self._cached_stream_id = self._session.stream_metadata()
            self._cached_ts = now
        return self._cached_device_id, self._cached_stream_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._baseline, name)
