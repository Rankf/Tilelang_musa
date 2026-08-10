"""P6.0 online session: observe real JIT compiles and launches safely."""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
import uuid
import weakref

from .config import OnlineConfig
from ..collectors.runtime.sampling import stable_selected


_TRACE_DRAIN_TIMEOUT_S = 5.0
_MAX_NATIVE_TRACE_POOL_SLOTS = 4096
_MAX_NATIVE_TRACE_POOL_BYTES = 64 * 1024 * 1024


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_metadata() -> tuple[int | None, int | None]:
    try:
        import torch
        if hasattr(torch, "musa") and torch.musa.is_available():
            stream = torch.musa.current_stream()
            stream_id = int(getattr(stream, "musa_stream", getattr(stream, "stream_id", 0)))
            return int(torch.musa.current_device()), stream_id
    except Exception:
        pass
    return None, None


class OnlineObservedKernel:
    """Transparent launch-observation proxy around the original JITKernel."""

    def __init__(self, baseline: Any, session: "OnlineSession", kernel_id: str):
        self._baseline = baseline
        self._session = session
        self._kernel_id = kernel_id
        self.online_instrumentation = {
            "mode": "observe_only",
            "kernel_id": kernel_id,
            "config_sha256": session.config.config_sha256,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        launch_index = self._session.next_launch_index(self._kernel_id)
        device_id, stream_id = _stream_metadata()
        started = time.perf_counter_ns()
        try:
            result = self._baseline(*args, **kwargs)
        except BaseException as error:
            self._session.record_launch(
                self._kernel_id, launch_index, device_id, stream_id, started,
                status="application_error", error_type=type(error).__name__,
            )
            raise
        self._session.record_launch(
            self._kernel_id, launch_index, device_id, stream_id, started, status="observed",
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._baseline, name)


class OnlineSession:
    """Thread-safe process session installed as one TileLang compile hook."""

    def __init__(self, config: OnlineConfig):
        self.config = config
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._created_monotonic = time.monotonic()
        self._compile_records: list[dict[str, Any]] = []
        self._launch_records: list[dict[str, Any]] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._trace_records: list[dict[str, Any]] = []
        self._trace_enriched_records: list[dict[str, Any]] = []
        self._correctness_records: list[dict[str, Any]] = []
        self._tileop_context: dict[str, dict[str, dict[str, Any]]] = {}
        self._launch_indexes: dict[str, int] = {}
        self._total_launches = 0
        self._fast_path_launches = 0
        self._fast_path_skip_reasons: dict[str, int] = {}
        self._launch_summary_providers: list[weakref.ReferenceType[Any]] = []
        self._trace_pools: dict[str, tuple[Any, Any]] = {}
        self._dual_variant_kernels: list[weakref.ReferenceType[Any]] = []
        self._selection_reservations = 0
        self._reserved_trace_records = 0
        self._reserved_trace_bytes = 0
        self._instrumented_launches = 0
        self._instrumented_dispatches = 0
        self._closed = False
        self._token: str | None = None
        self._telemetry_start: dict[str, Any] | None = None
        self._telemetry_end: dict[str, Any] | None = None
        self._create_process_directory()
        if self.config.telemetry and self.config.telemetry.get("enabled"):
            from ..collectors.runtime.telemetry import snapshot
            self._telemetry_start = snapshot(int(self.config.telemetry.get("device", 0)))
        atexit.register(self.flush)

    def _create_process_directory(self) -> None:
        session_id = f"online_{self._pid}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.directory = self.config.output / session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(self.directory / "resolved_online_config.json", self.config.as_dict())
        _atomic_json(self.directory / "capability.json", {
            "schema": "online-capability-v1",
            "phase": "P6.2" if self._trace_pools else "P6.0",
            "compile_observation": True,
            "launch_observation": True,
            "automatic_instrumentation": bool(self._trace_pools),
            "reason": "P6.0 observe-only mode",
        })
        _atomic_json(self.directory / "process_manifest.json", {
            "schema": "online-process-manifest-v1",
            "session_id": self.session_id,
            "pid": self._pid,
            "created_at": _utc_now(),
            "config_sha256": self.config.config_sha256,
            "profile": "online_runtime_observation_v1",
            "correctness": self.config.correctness_policy,
        })

    def install(self) -> "OnlineSession":
        if self._token is not None:
            return self
        from tilelang.jit.hooks import register_compile_hook
        self._token = register_compile_hook(self)
        return self

    @property
    def pid(self) -> int:
        return self._pid

    def instrumentation_process_matches(self, owner_pid: int) -> bool:
        """Whether a wrapper's MUSA resources belong to the current process."""
        return os.getpid() == owner_pid == self._pid

    @staticmethod
    def stream_metadata() -> tuple[int | None, int | None]:
        return _stream_metadata()

    def uninstall(self) -> None:
        token = self._token
        if token is None:
            return
        from tilelang.jit.hooks import unregister_compile_hook
        unregister_compile_hook(token)
        self._token = None

    def _ensure_process(self) -> None:
        current = os.getpid()
        if current == self._pid:
            return
        # A forked child must never share counters or output files with its parent.
        self._pid = current
        self._compile_records = []
        self._launch_records = []
        self._diagnostics = [{"code": "PROCESS_FORK_REINITIALIZED", "at": _utc_now()}]
        self._trace_records = []
        self._trace_enriched_records = []
        self._correctness_records = []
        self._tileop_context = {}
        self._launch_indexes = {}
        self._total_launches = 0
        self._fast_path_launches = 0
        self._fast_path_skip_reasons = {}
        self._launch_summary_providers = []
        self._trace_pools = {}
        self._selection_reservations = 0
        self._reserved_trace_records = 0
        self._reserved_trace_bytes = 0
        self._instrumented_launches = 0
        self._instrumented_dispatches = 0
        self._created_monotonic = time.monotonic()
        self._closed = False
        self._create_process_directory()
        self._telemetry_start = None
        self._telemetry_end = None
        if self.config.telemetry and self.config.telemetry.get("enabled"):
            from ..collectors.runtime.telemetry import snapshot
            self._telemetry_start = snapshot(int(self.config.telemetry.get("device", 0)))

    def __call__(self, request: Any) -> Any:
        from tilelang.jit.hooks import CompileAugmentation
        with self._lock:
            self._ensure_process()
            kernel_id = request.baseline_cache_identity.replace("sha256:", "kernel_")[:31]
            attrs = getattr(request.prim_func, "attrs", None)
            symbol = str(attrs["global_symbol"]) if attrs and "global_symbol" in attrs else "unknown"
            self._compile_records.append({
                "schema": "online-compile-observation-v1",
                "at": _utc_now(),
                "pid": self._pid,
                "kernel_id": kernel_id,
                "global_symbol": symbol,
                "baseline_cache_identity": request.baseline_cache_identity,
                "normalized_out_idx": list(request.normalized_out_idx),
                "original_out_idx": list(request.original_out_idx),
                "target": str(request.target),
                "target_host": str(request.target_host) if request.target_host is not None else None,
                "execution_backend": request.execution_backend,
                "compile_origin": request.compile_origin,
            })
            self._flush_locked()

        if self.config.compile_policy == "eager_dual_variant":
            try:
                return self._dual_variant_augmentation(request, kernel_id)
            except Exception as error:
                self.record_diagnostic(
                    "ONLINE_INSTRUMENTATION_PLAN_REJECTED", kernel_id=kernel_id,
                    error_type=type(error).__name__, message=str(error))

        def wrap(baseline: Any, _internal_compile: Any) -> OnlineObservedKernel:
            self._write_baseline_manifest(kernel_id, request, baseline)
            return OnlineObservedKernel(baseline, self, kernel_id)

        return CompileAugmentation(action="observe", wrap_kernel=wrap, owner="musa_perf_collector.online")

    def _dual_variant_augmentation(self, request: Any, kernel_id: str) -> Any:
        from tilelang import env
        from tilelang.jit.execution_backend import resolve_execution_backend
        from tilelang.jit.hooks import CompileAugmentation
        from tilelang.utils.target import determine_target

        from .dual_variant import OnlineDualVariantKernel
        from .plan import (
            build_plan, instrumented_pass_configs, load_selected_tileop_context,
            prepare_instrumented_primfunc,
        )
        from .trace_pool import TraceBufferPool

        requested_target = request.target if request.target is not None else env.get_default_target()
        target = determine_target(requested_target, return_object=True)
        requested_backend = (
            request.execution_backend
            if request.execution_backend is not None
            else env.get_default_execution_backend())
        backend = resolve_execution_backend(requested_backend, target)
        if target.kind.name != "musa" or backend not in self.config.execution_backends or backend != "tvm_ffi":
            raise RuntimeError(
                f"online automatic instrumentation only supports MUSA tvm_ffi, got {target.kind.name}/{backend}")
        plan = build_plan(self.config, request, target=str(target), execution_backend=backend)
        tileop_context = load_selected_tileop_context(plan)
        instrumented_func = prepare_instrumented_primfunc(request, plan)
        kernel_directory = self.directory / "kernels" / kernel_id
        _atomic_json(kernel_directory / "instrumentation_plan.json", plan.as_dict())
        _atomic_json(kernel_directory / "baseline_manifest.json", {
            "schema": "online-baseline-manifest-v1",
            "kernel_id": kernel_id,
            "baseline_cache_identity": request.baseline_cache_identity,
            "primfunc_sha256": request.prim_func_sha256,
            "original_out_idx": list(request.original_out_idx),
            "normalized_out_idx": list(request.normalized_out_idx),
            "launch_context": plan.launch_context,
        })
        _atomic_json(kernel_directory / "compiler_lineage.json", {
            "schema": "online-compiler-lineage-v1",
            "status": "compiler_exact_lineage",
            "baseline_primfunc_sha256": plan.baseline_primfunc_sha256,
            "tilelang_revision": plan.tilelang_revision,
            "plan_sha256": plan.plan_sha256,
            "mappings": [
                {"source_op_id": op_id, "source_ordinal": ordinal,
                 "source_kind": kind, "lowered_instance_count": count,
                 "lowering_semantics": plan.lowering_semantics}
                for op_id, ordinal, kind, count in zip(
                    plan.source_op_ids, plan.source_ordinals, plan.source_kinds,
                    plan.lowered_instance_counts)
            ],
        })
        _atomic_jsonl(
            kernel_directory / "tileop_context.jsonl",
            [tileop_context[op_id] for op_id in plan.source_op_ids])
        with self._lock:
            self._tileop_context[kernel_id] = tileop_context

        def wrap(baseline: Any, internal_compile: Any) -> Any:
            try:
                self._write_baseline_manifest(
                    kernel_id, request, baseline, launch_context=plan.launch_context)
                compile_started = time.perf_counter_ns()
                instrumented = internal_compile(
                    _instrumented_plan_hash=plan.plan_sha256,
                    func=instrumented_func,
                    out_idx=None,
                    execution_backend=backend,
                    target=target,
                    target_host=request.target_host,
                    verbose=request.verbose,
                    pass_configs=instrumented_pass_configs(request, plan),
                    instruments=list(request.instruments),
                    compile_flags=list(request.compile_flags),
                )
                compile_ns = time.perf_counter_ns() - compile_started
                _atomic_json(kernel_directory / "compile_overhead.json", {
                    "schema": "online-compile-overhead-v1",
                    "instrumented_compile_ns": compile_ns,
                    "excluded_from_kernel_latency": True,
                })
                _atomic_json(kernel_directory / "instrumented_manifest.json", {
                    "schema": "online-instrumented-manifest-v1",
                    "plan_sha256": plan.plan_sha256,
                    "parameter_count": len(instrumented_func.params),
                    "trace_parameter_index": len(request.prim_func.params),
                    "cache_namespace": f"instrumented/{plan.plan_sha256}",
                    "baseline_kernel_source_sha256": self._kernel_source_sha256(baseline),
                    "instrumented_kernel_source_sha256": self._kernel_source_sha256(instrumented),
                })
                per_slot_bytes = plan.trace_extent * 8
                if per_slot_bytes > _MAX_NATIVE_TRACE_POOL_BYTES:
                    raise RuntimeError(
                        "one online trace buffer exceeds the 64 MiB safety ceiling")
                safety_capacity = min(
                    _MAX_NATIVE_TRACE_POOL_SLOTS,
                    _MAX_NATIVE_TRACE_POOL_BYTES // per_slot_bytes,
                )
                trace_budget_capacity = min(
                    self.config.budget.max_trace_records // plan.trace_extent,
                    self.config.budget.max_trace_bytes // per_slot_bytes,
                )
                native_capacity = max(1, min(
                    self.config.budget.max_instrumented_launches,
                    trace_budget_capacity,
                    self.config.launch_candidate_count,
                    safety_capacity,
                ))
                fallback_capacity = max(1, min(
                    self.config.budget.max_pending_readbacks,
                    safety_capacity,
                ))
                pool = TraceBufferPool(
                    trace_extent=plan.trace_extent,
                    capacity=fallback_capacity,
                    native_capacity=native_capacity,
                    event_timing_every=self.config.event_timing_every,
                    device=0,
                )
                with self._lock:
                    self._trace_pools[kernel_id] = (plan, pool)
                dual_variant = OnlineDualVariantKernel(
                    baseline, instrumented, self, kernel_id, plan, pool)

                implementation_phase = (
                    "P6.9" if pool.readback_policy == "native_deferred_at_flush"
                    else "P6.8" if dual_variant.dispatch_path == "native_c_vectorcall"
                    else "P6.5" if plan.lowering_semantics != "single" else "P6.4")
                _atomic_json(self.directory / "capability.json", {
                    "schema": "online-capability-v1",
                    "phase": implementation_phase,
                    "compile_observation": True,
                    "launch_observation": True,
                    "automatic_instrumentation": True,
                    "target": "musa",
                    "execution_backend": "tvm_ffi",
                    "lineage": "compiler_exact",
                    "lowering_semantics": plan.lowering_semantics,
                    "readback_policy": pool.readback_policy,
                    "dispatch_path": dual_variant.dispatch_path,
                    "event_timing_every": self.config.event_timing_every,
                    "trace_pool_capacity": pool.capacity,
                    "trace_pool_device": pool.device,
                    "native_trace_status": pool.native_status,
                    "native_trace_error": pool.native_error,
                })
                return dual_variant.as_callable()
            except Exception as error:
                self.record_diagnostic(
                    "ONLINE_INSTRUMENTED_COMPILE_FAILED", kernel_id=kernel_id,
                    error_type=type(error).__name__, message=str(error))
                return OnlineObservedKernel(baseline, self, kernel_id)

        return CompileAugmentation(
            action="dual_variant_plan", wrap_kernel=wrap,
            owner="musa_perf_collector.online")

    @staticmethod
    def _kernel_source_sha256(kernel: Any) -> str | None:
        try:
            source = kernel.get_kernel_source()
            return sha256(source.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _write_baseline_manifest(
        self, kernel_id: str, request: Any, baseline: Any,
        *, launch_context: dict[str, Any] | None = None,
    ) -> None:
        _atomic_json(self.directory / "kernels" / kernel_id / "baseline_manifest.json", {
            "schema": "online-baseline-manifest-v1",
            "kernel_id": kernel_id,
            "baseline_cache_identity": request.baseline_cache_identity,
            "primfunc_sha256": request.prim_func_sha256,
            "kernel_source_sha256": self._kernel_source_sha256(baseline),
            "original_out_idx": list(request.original_out_idx),
            "normalized_out_idx": list(request.normalized_out_idx),
            "launch_context": launch_context,
        })

    def next_launch_index(self, kernel_id: str) -> int:
        with self._lock:
            self._ensure_process()
            return self._next_launch_index_locked(kernel_id)

    def _next_launch_index_locked(self, kernel_id: str) -> int:
        value = self._launch_indexes.get(kernel_id, 0)
        self._launch_indexes[kernel_id] = value + 1
        self._total_launches += 1
        return value

    def prepare_fast_launch(
        self, kernel_id: str, *, trace_extent: int,
    ) -> tuple[int, bool, str, int | None] | None:
        """Resolve non-probabilistic selectors without per-launch metadata.

        ``None`` means that stream-aware probability selection is required.
        Every returned unselected decision is counted only in aggregate; the
        caller can invoke the baseline immediately without stream queries,
        trace polling, or allocating a launch-record dictionary.
        """
        selector = self.config.launch_selector or {"every": 64, "start": 0}
        if "probability" in selector:
            return None
        with self._lock:
            self._ensure_process()
            launch_index = self._next_launch_index_locked(kernel_id)
            selected, reason, ordinal = self._select_launch_locked(
                kernel_id, launch_index, None, trace_extent=trace_extent,
                selector=selector)
            if not selected:
                self._fast_path_launches += 1
                self._fast_path_skip_reasons[reason] = (
                    self._fast_path_skip_reasons.get(reason, 0) + 1)
            return launch_index, selected, reason, ordinal

    def note_unselected_launch(self, reason: str) -> None:
        """Aggregate an unselected stream-aware probability decision."""
        with self._lock:
            self._fast_path_launches += 1
            self._fast_path_skip_reasons[reason] = (
                self._fast_path_skip_reasons.get(reason, 0) + 1)

    def register_launch_summary_provider(self, provider: Any) -> None:
        with self._lock:
            self._launch_summary_providers.append(weakref.ref(provider))

    def register_dual_variant_kernel(self, kernel: Any) -> None:
        with self._lock:
            self._dual_variant_kernels.append(weakref.ref(kernel))

    def _aggregate_launch_counts_locked(
        self,
    ) -> tuple[int, int, dict[str, int]]:
        total = self._total_launches
        fast = self._fast_path_launches
        reasons = dict(self._fast_path_skip_reasons)
        alive: list[weakref.ReferenceType[Any]] = []
        for reference in self._launch_summary_providers:
            provider = reference()
            if provider is None:
                continue
            alive.append(reference)
            summary = provider.online_launch_summary()
            total += int(summary["total_launch_count"])
            fast += int(summary["fast_path_launch_count"])
            for reason, count in summary["fast_path_skip_reasons"].items():
                reasons[str(reason)] = reasons.get(str(reason), 0) + int(count)
        self._launch_summary_providers = alive
        return total, fast, reasons

    def record_launch(
        self, kernel_id: str, launch_index: int, device_id: int | None, stream_id: int | None,
        started_ns: int | None, *, status: str, error_type: str | None = None,
        selection_reason: str | None = None,
    ) -> None:
        with self._lock:
            self._ensure_process()
            self._launch_records.append({
                "schema": "online-launch-observation-v1",
                "at": _utc_now(),
                "pid": self._pid,
                "thread_id": threading.get_ident(),
                "kernel_id": kernel_id,
                "launch_index": launch_index,
                "device_id": device_id,
                "stream_id": stream_id,
                "status": status,
                "error_type": error_type,
                "selection_reason": selection_reason,
                "callable_wall_ns": (
                    time.perf_counter_ns() - started_ns
                    if started_ns is not None else None),
                "evidence": (
                    "host_observation_only" if started_ns is not None
                    else "host_exception_without_timing"),
            })

    def select_launch(
        self, kernel_id: str, launch_index: int, stream_id: int | None,
        *, trace_extent: int,
    ) -> tuple[bool, str, int | None]:
        with self._lock:
            return self._select_launch_locked(
                kernel_id, launch_index, stream_id, trace_extent=trace_extent,
                selector=self.config.launch_selector or {"every": 64, "start": 0})

    def _select_launch_locked(
        self, kernel_id: str, launch_index: int, stream_id: int | None,
        *, trace_extent: int, selector: dict[str, Any],
    ) -> tuple[bool, str, int | None]:
        """Selection implementation; caller must hold ``self._lock``."""
        if trace_extent <= 0:
            return False, "invalid_trace_extent", None
        if launch_index >= self.config.launch_candidate_count:
            return False, "sampling_window_closed", None
        if self._selection_reservations >= self.config.budget.max_instrumented_launches:
            return False, "instrumented_launch_budget_exhausted", None
        if self._reserved_trace_records + trace_extent > self.config.budget.max_trace_records:
            return False, "trace_record_budget_exhausted", None
        trace_bytes = trace_extent * 8
        if self._reserved_trace_bytes + trace_bytes > self.config.budget.max_trace_bytes:
            return False, "trace_byte_budget_exhausted", None
        seed: int | None = None
        if "indices" in selector:
            selected = launch_index in {int(value) for value in selector["indices"]}
        elif "every" in selector:
            start = int(selector.get("start", 0))
            selected = launch_index >= start and (launch_index - start) % int(selector["every"]) == 0
        elif "probability" in selector:
            seed = int(self.config.config_sha256.split(":", 1)[1][:16], 16)
            identity = f"{kernel_id}:{stream_id}:{launch_index}"
            selected = stable_selected(identity, seed=seed, probability=float(selector["probability"]))
        else:
            selected = False
        if not selected:
            return False, "launch_not_selected", None
        if time.monotonic() - self._created_monotonic > self.config.budget.max_total_duration_s:
            return False, "duration_budget_exhausted", None
        if seed is None:
            seed = int(self.config.config_sha256.split(":", 1)[1][:16], 16)
        kernel_selector = self.config.kernel_selector or {"probability": 1.0}
        if "probability" in kernel_selector and not stable_selected(
            kernel_id, seed=seed, probability=float(kernel_selector["probability"])):
            return False, "kernel_not_selected", None
        selection_ordinal = self._selection_reservations
        self._selection_reservations += 1
        self._reserved_trace_records += trace_extent
        self._reserved_trace_bytes += trace_bytes
        return True, "selected", selection_ordinal

    def record_instrumented_launch(self) -> None:
        with self._lock:
            self._instrumented_launches += 1

    def claim_instrumented_dispatch(self) -> int:
        """Return a gap-free ordinal for an instrumented kernel invocation."""
        with self._lock:
            ordinal = self._instrumented_dispatches
            self._instrumented_dispatches += 1
            return ordinal

    def poll_trace_pool(self, kernel_id: str, plan: Any, pool: Any, *, wait: bool = False) -> None:
        completions = pool.poll(wait=wait)
        if not completions:
            return
        with self._lock:
            for completion in completions:
                values = completion["values"]
                sentinel = completion["sentinel"]
                extent_valid = len(values) == plan.trace_extent
                all_slots_written = extent_valid and all(value != sentinel for value in values)
                for layout in plan.slot_layout:
                    slot = int(layout["slot"])
                    value = values[slot] if slot < len(values) else None
                    slot_valid = extent_valid and value != sentinel
                    interval_cycles = None
                    if layout["boundary"] == "end" and slot > 0 and slot_valid:
                        previous = plan.slot_layout[slot - 1]
                        begin_value = values[slot - 1]
                        pair_matches = (
                            previous["boundary"] == "begin"
                            and previous["op_id"] == layout["op_id"]
                            and previous["block"] == layout["block"]
                            and previous.get("lowered_instance_index")
                            == layout.get("lowered_instance_index")
                        )
                        if pair_matches and begin_value != sentinel:
                            begin = int(begin_value) & ((1 << 64) - 1)
                            end = int(value) & ((1 << 64) - 1)
                            interval_cycles = (end - begin) & ((1 << 64) - 1)
                    record = {
                        "schema": "online-tile-trace-v1",
                        **completion["launch_identity"],
                        **layout,
                        "clock_cycles": value,
                        "generation": completion["generation"],
                        "readback_policy": completion["readback_policy"],
                        "kernel_event_ms": completion["kernel_event_ms"],
                        "valid": bool(slot_valid),
                        "interval_cycles": interval_cycles,
                        "interval_semantics": (
                            "lane0_issue_boundary_envelope" if interval_cycles is not None else None),
                        "evidence_level": "compiler_exact_instrumented_issue_boundary",
                    }
                    self._trace_records.append(record)
                    context = self._tileop_context.get(kernel_id, {}).get(str(layout["op_id"]))
                    if context is not None:
                        self._trace_enriched_records.append({**record, "tileop": context})
                if not all_slots_written:
                    self._diagnostics.append({
                        "code": "ONLINE_TRACE_PARTIAL", "at": _utc_now(),
                        "kernel_id": kernel_id, "generation": completion["generation"]})

    def record_diagnostic(self, code: str, **details: Any) -> None:
        with self._lock:
            self._diagnostics.append({"code": code, "at": _utc_now(), **details})

    def report_correctness(
        self, *, status: str, kernel_id: str | None = None,
        launch_index: int | None = None, source: str = "application",
    ) -> None:
        if status not in {"pass", "fail"}:
            raise ValueError("application correctness status must be pass or fail")
        with self._lock:
            self._correctness_records.append({
                "schema": "online-application-correctness-v1",
                "at": _utc_now(), "status": status, "source": source,
                "kernel_id": kernel_id, "launch_index": launch_index,
                "evidence": "application_reported",
            })

    def _write_audit_inputs_locked(
        self, *, total_launch_count: int, fast_path_launch_count: int,
        fast_path_skip_reasons: dict[str, int],
    ) -> None:
        """Persist every non-record input required by the disk-backed audit."""
        _atomic_json(self.directory / "session_audit_inputs.json", {
            "schema": "online-session-audit-inputs-v1",
            "total_launch_count": total_launch_count,
            "fast_path_launch_count": fast_path_launch_count,
            "fast_path_skip_reasons": dict(sorted(fast_path_skip_reasons.items())),
            "instrumented_launch_count": self._instrumented_launches,
            "instrumented_dispatch_count": self._instrumented_dispatches,
            "reserved_instrumented_launches": self._selection_reservations,
            "reserved_trace_records": self._reserved_trace_records,
            "reserved_trace_bytes": self._reserved_trace_bytes,
            "kernels": {
                kernel_id: {
                    "readback_policy": pool.readback_policy,
                    "plan_path": str(
                        Path("kernels") / kernel_id / "instrumentation_plan.json"),
                }
                for kernel_id, (_plan, pool) in self._trace_pools.items()
            },
            "budget": {
                "max_instrumented_launches": self.config.budget.max_instrumented_launches,
                "max_trace_records": self.config.budget.max_trace_records,
                "max_trace_bytes": self.config.budget.max_trace_bytes,
            },
        })

    def _load_persisted_audit_inputs(self) -> dict[str, Any]:
        """Reload the full evidence bundle after atomic writes, fail-closed."""
        inputs = _read_json_object(self.directory / "session_audit_inputs.json")
        if inputs.get("schema") != "online-session-audit-inputs-v1":
            raise ValueError("session audit input schema mismatch")
        kernels = inputs.get("kernels")
        if not isinstance(kernels, dict):
            raise ValueError("session audit inputs kernels must be an object")
        expected_trace_extents: dict[str, int] = {}
        expected_trace_layouts: dict[str, list[dict[str, Any]]] = {}
        expected_plan_hashes: dict[str, str | None] = {}
        expected_readback_policies: dict[str, str] = {}
        for kernel_id, kernel_input in kernels.items():
            if not isinstance(kernel_id, str) or not kernel_id or not isinstance(kernel_input, dict):
                raise ValueError("session audit kernel input is malformed")
            relative_plan = kernel_input.get("plan_path")
            if not isinstance(relative_plan, str):
                raise ValueError("session audit plan path is malformed")
            plan_path = self.directory / relative_plan
            if plan_path.parent.parent != self.directory / "kernels":
                raise ValueError("session audit plan path escapes kernels directory")
            plan = _read_json_object(plan_path)
            if plan.get("schema") != "online-instrumentation-plan-v1":
                raise ValueError("instrumentation plan schema mismatch")
            trace_extent = _strict_nonnegative_int(
                plan.get("trace_extent"), field="instrumentation plan trace_extent")
            slot_layout = plan.get("slot_layout")
            if not isinstance(slot_layout, list):
                raise ValueError("instrumentation plan slot_layout must be a list")
            plan_sha256 = plan.get("plan_sha256")
            if not isinstance(plan_sha256, str) or not plan_sha256:
                raise ValueError("instrumentation plan plan_sha256 is malformed")
            readback_policy = kernel_input.get("readback_policy")
            if not isinstance(readback_policy, str) or not readback_policy:
                raise ValueError("session audit readback_policy is malformed")
            expected_trace_extents[kernel_id] = trace_extent
            expected_trace_layouts[kernel_id] = slot_layout
            expected_plan_hashes[kernel_id] = plan_sha256
            expected_readback_policies[kernel_id] = readback_policy
        budget = inputs.get("budget")
        if not isinstance(budget, dict):
            raise ValueError("session audit budget is malformed")
        return {
            "launches": _read_jsonl_objects(self.directory / "launch_observations.jsonl"),
            "trace_records": _read_jsonl_objects(self.directory / "trace.jsonl"),
            "enriched_records": _read_jsonl_objects(self.directory / "trace_enriched.jsonl"),
            "expected_trace_extents": expected_trace_extents,
            "expected_trace_layouts": expected_trace_layouts,
            "expected_plan_hashes": expected_plan_hashes,
            "expected_readback_policies": expected_readback_policies,
            "total_launch_count": _strict_nonnegative_int(
                inputs.get("total_launch_count"), field="total_launch_count"),
            "fast_path_launch_count": _strict_nonnegative_int(
                inputs.get("fast_path_launch_count"), field="fast_path_launch_count"),
            "fast_path_skip_reasons": inputs.get("fast_path_skip_reasons"),
            "instrumented_launch_count": _strict_nonnegative_int(
                inputs.get("instrumented_launch_count"), field="instrumented_launch_count"),
            "instrumented_dispatch_count": _strict_nonnegative_int(
                inputs.get("instrumented_dispatch_count"), field="instrumented_dispatch_count"),
            "reserved_instrumented_launches": _strict_nonnegative_int(
                inputs.get("reserved_instrumented_launches"), field="reserved_instrumented_launches"),
            "reserved_trace_records": _strict_nonnegative_int(
                inputs.get("reserved_trace_records"), field="reserved_trace_records"),
            "reserved_trace_bytes": _strict_nonnegative_int(
                inputs.get("reserved_trace_bytes"), field="reserved_trace_bytes"),
            "budget": SimpleNamespace(
                max_instrumented_launches=_strict_nonnegative_int(
                    budget.get("max_instrumented_launches"),
                    field="budget.max_instrumented_launches"),
                max_trace_records=_strict_nonnegative_int(
                    budget.get("max_trace_records"), field="budget.max_trace_records"),
                max_trace_bytes=_strict_nonnegative_int(
                    budget.get("max_trace_bytes"), field="budget.max_trace_bytes"),
            ),
        }

    def _flush_locked(self) -> None:
        from .validation import audit_session

        advanced_lowering = any(
            plan.lowering_semantics != "single" for plan, _pool in self._trace_pools.values())
        native_trace = any(
            pool.readback_policy == "native_deferred_at_flush"
            for _plan, pool in self._trace_pools.values())
        correctness_failed = any(
            item["status"] == "fail" for item in self._correctness_records)
        launch_failed = any(
            item["status"] in {"application_error", "instrumented_error"}
            for item in self._launch_records)
        drain_failed = any(
            item.get("code") == "ONLINE_TRACE_DRAIN_FAILED"
            for item in self._diagnostics)
        total_launch_count, fast_path_launch_count, fast_path_skip_reasons = (
            self._aggregate_launch_counts_locked())
        _atomic_jsonl(self.directory / "compile_observations.jsonl", self._compile_records)
        _atomic_jsonl(self.directory / "launch_observations.jsonl", self._launch_records)
        _atomic_jsonl(self.directory / "diagnostics.jsonl", self._diagnostics)
        _atomic_jsonl(self.directory / "trace.jsonl", self._trace_records)
        _atomic_jsonl(self.directory / "trace_enriched.jsonl", self._trace_enriched_records)
        _atomic_jsonl(self.directory / "correctness.jsonl", self._correctness_records)
        _atomic_json(self.directory / "telemetry.json", {
            "schema": "online-telemetry-snapshots-v1",
            "timing_scope": "outside_kernel_event_window",
            "start": self._telemetry_start,
            "end": self._telemetry_end,
        })
        for kernel_id in self._trace_pools:
            kernel_directory = self.directory / "kernels" / kernel_id
            raw = [record for record in self._trace_records if record.get("kernel_id") == kernel_id]
            enriched = [record for record in self._trace_enriched_records if record.get("kernel_id") == kernel_id]
            _atomic_jsonl(kernel_directory / "trace.jsonl", raw)
            _atomic_jsonl(kernel_directory / "trace_enriched.jsonl", enriched)
        self._write_audit_inputs_locked(
            total_launch_count=total_launch_count,
            fast_path_launch_count=fast_path_launch_count,
            fast_path_skip_reasons=fast_path_skip_reasons)
        try:
            session_audit = audit_session(**self._load_persisted_audit_inputs())
            session_audit["evidence_source"] = "persisted_session_files"
            session_audit["audit_inputs_path"] = "session_audit_inputs.json"
            session_audit["checks"].append("persisted_evidence_replay")
        except Exception as error:
            session_audit = {
                "schema": "online-session-audit-v1",
                "audit_revision": "P6.9.4",
                "checks": ["persisted_evidence_replay"],
                "status": "fail",
                "violation_count": 1,
                "violations": [{
                    "code": "PERSISTED_EVIDENCE_READ_FAILED",
                    "error_type": type(error).__name__, "message": str(error),
                }],
                "observed": {}, "reserved": {},
            }
        _atomic_json(self.directory / "session_audit.json", session_audit)
        _atomic_json(self.directory / "session_validation.json", {
            "schema": "online-session-validation-v1",
            "phase": (
                "P6.9" if native_trace else "P6.5" if advanced_lowering
                else "P6.4" if self._trace_pools else "P6.0"),
            "status": (
                "invalid" if correctness_failed or launch_failed or drain_failed
                or session_audit["status"] != "pass" else
                "complete" if not self._diagnostics else "partial"),
            "compile_observation_count": len(self._compile_records),
            "launch_observation_count": len(self._launch_records),
            "total_launch_count": total_launch_count,
            "fast_path_launch_count": fast_path_launch_count,
            "fast_path_skip_reasons": dict(sorted(fast_path_skip_reasons.items())),
            "automatic_instrumentation": bool(self._trace_pools),
            "instrumented_launch_count": self._instrumented_launches,
            "instrumented_dispatch_count": self._instrumented_dispatches,
            "reserved_instrumented_launches": self._selection_reservations,
            "reserved_trace_records": self._reserved_trace_records,
            "reserved_trace_bytes": self._reserved_trace_bytes,
            "trace_record_count": len(self._trace_records),
            "independent_audit": session_audit["status"],
            "audit_revision": session_audit["audit_revision"],
            "audit_violation_count": session_audit["violation_count"],
            "correctness": (
                "application_reported" if self._correctness_records else self.config.correctness_policy),
            "application_correctness_status": (
                "fail" if any(item["status"] == "fail" for item in self._correctness_records)
                else "pass" if self._correctness_records else "not_observed"),
            "elapsed_s": time.monotonic() - self._created_monotonic,
        })

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
        if (
            self._telemetry_end is None and self.config.telemetry
            and self.config.telemetry.get("enabled")
        ):
            from ..collectors.runtime.telemetry import snapshot
            self._telemetry_end = snapshot(int(self.config.telemetry.get("device", 0)))
        pools = []
        with self._lock:
            pools = [(kernel_id, plan, pool) for kernel_id, (plan, pool) in self._trace_pools.items()]
        alive_kernels = []
        for ref in self._dual_variant_kernels:
            kernel = ref()
            if kernel is not None:
                kernel.flush_pending()
                alive_kernels.append(ref)
        self._dual_variant_kernels = alive_kernels
        drain_deadline = time.monotonic() + _TRACE_DRAIN_TIMEOUT_S
        for kernel_id, plan, pool in pools:
            try:
                if pool.readback_policy == "native_deferred_at_flush":
                    self.poll_trace_pool(kernel_id, plan, pool, wait=True)
                    continue
                while pool.pending_count:
                    self.poll_trace_pool(kernel_id, plan, pool, wait=False)
                    if not pool.pending_count:
                        break
                    if time.monotonic() >= drain_deadline:
                        self.record_diagnostic(
                            "ONLINE_TRACE_DRAIN_TIMEOUT", kernel_id=kernel_id,
                            pending_count=pool.pending_count,
                            timeout_s=_TRACE_DRAIN_TIMEOUT_S)
                        break
                    time.sleep(0.001)
            except Exception as error:
                self.record_diagnostic(
                    "ONLINE_TRACE_DRAIN_FAILED", kernel_id=kernel_id,
                    error_type=type(error).__name__, message=str(error))
        with self._lock:
            self._ensure_process()
            self._flush_locked()

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.uninstall()
        for _plan, pool in self._trace_pools.values():
            pool.close()
