"""Bounded runtime measurement orchestration for reviewed workload adapters."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from statistics import median
import time
from typing import Any

from .adapters.tilelang.runtime_workload import RuntimeWorkload
from .collectors.runtime.capability import probe
from .collectors.runtime.correctness import verify
from .collectors.runtime.event import batch_event_samples
from .collectors.runtime.overhead import compare as compare_overhead, kernel_only_from_activity
from .collectors.runtime.offline_join import associate as associate_offline_tileops
from .collectors.runtime.sampling import (BudgetExceeded, CollectionBudget, apply_budget,
                                          identity_selection_evidence, linear_block_coordinates,
                                          selection_evidence)
from .collectors.runtime.stability import assess as assess_stability
from .collectors.runtime.stream_sync import audit as audit_stream_sync
from .collectors.runtime.telemetry import TelemetrySampler, snapshot
from .collectors.runtime.trace_buffer import TraceBufferSlot, TraceBufferState, probe_async_d2h
from .collectors.runtime.trace_validation import validate as validate_trace
from .collectors.runtime.torch_activity import collect as collect_activity
from .core.ids import sha256_text, stable_id
from .core.io import write_json, write_jsonl, write_text
from .core.runtime_validation import validate_session
from .exporters.runtime_report import render_html, render_markdown


def _load_factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("runtime factory must use package.module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"runtime factory {spec} is not callable")
    return function


def _resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(config))
    output.setdefault("mode", "light")
    output.setdefault("device", 0)
    output.setdefault("sampling_seed", 0)
    output.setdefault("baseline", {})
    output["baseline"].setdefault("warmup_launches", 10)
    output["baseline"].setdefault("repeat_launches", 100)
    output["baseline"].setdefault("event_batch_size", 10)
    output["baseline"].setdefault("cache_policy", "preserve")
    output.setdefault("standard", {})
    output["standard"].setdefault("max_profiled_launches", 10)
    output.setdefault("stability", {})
    output["stability"].setdefault("enabled", True)
    output["stability"].setdefault("required", False)
    output["stability"].setdefault("min_telemetry_points", 3)
    output["stability"].setdefault("min_timing_batches", 4)
    output["stability"].setdefault("max_temperature_c", 90.0)
    output["stability"].setdefault("max_temperature_delta_c", 5.0)
    output["stability"].setdefault("max_frequency_delta_pct", 5.0)
    output["stability"].setdefault("max_timing_drift_pct", 10.0)
    output.setdefault("budget", {})
    budget = CollectionBudget.from_config(output["budget"])
    output["budget"] = budget.__dict__
    return output


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["per_launch_ms"]) for item in samples]
    return {"sample_count": len(values), "median_per_launch_ms": median(values) if values else None,
            "min_per_launch_ms": min(values) if values else None, "max_per_launch_ms": max(values) if values else None,
            "timing_kind": "batch_average", "instrumented": False}


def _run_for_correctness(case: Any):
    value = case.run()
    if case.execution_stream is not None:
        case.execution_stream.synchronize()
    return value


def _resolve_deep_scope(resolved: dict[str, Any], case: Any, stream_id: int) -> bool:
    """Resolve case/kernel/stream gates without suppressing baseline collection."""
    deep = resolved.setdefault("deep", {})
    seed = int(resolved.get("sampling_seed", 0))
    candidates = {
        "case": [case.name],
        "kernel": [case.kernel_id],
        "stream": [str(stream_id)],
    }
    evidence: dict[str, Any] = {}
    selected = True
    for dimension, values in candidates.items():
        chosen, item = identity_selection_evidence(
            values, deep.get(f"{dimension}_selector"), seed=seed, dimension=dimension)
        evidence[dimension] = item
        selected = selected and bool(chosen)
    deep.setdefault("selection_evidence", {}).update(evidence)
    deep["scope_selected"] = selected
    return selected


def _bound_deep_config(
    resolved: dict[str, Any], diagnostics: list[dict[str, Any]], *, launch: dict[str, Any] | None = None
) -> None:
    """Apply every deep selection limit before any instrumented kernel is compiled."""
    deep = resolved.setdefault("deep", {})
    budget = resolved["budget"]
    policy = budget["on_budget_exceeded"]
    seed = int(resolved.get("sampling_seed", 0))
    legacy_launches = list(deep.get("launch_indices", [0]))
    inferred_launch_count = max((int(value) for value in legacy_launches), default=0) + 1
    launch_count = int(deep.get("launch_candidate_count", inferred_launch_count))
    if launch_count <= 0:
        raise RuntimeError("deep launch_candidate_count must be positive")
    launch_selector = deep.get("launch_selector")
    if launch_selector is not None and "launch_indices" in deep:
        raise RuntimeError("use either deep.launch_selector or deep.launch_indices, not both")
    if launch_selector is None:
        launch_selector = {"indices": legacy_launches}
    launch_indices, launch_evidence, launch_diagnostic = selection_evidence(
        range(launch_count), launch_selector, seed=seed, dimension="launch",
        budget_limit=int(budget["max_instrumented_launches"]), policy=policy)
    if not launch_indices:
        raise RuntimeError("deep launch selector selected no instrumented launches")
    deep["launch_indices"] = launch_indices
    deep["launch_candidate_count"] = launch_count
    if launch_diagnostic is not None:
        diagnostics.append(launch_diagnostic)

    raw_coordinates = deep.get("block_coordinates")
    block_selector = deep.get("block_selector")
    if raw_coordinates is not None and block_selector is not None:
        raise RuntimeError("use either deep.block_selector or deep.block_coordinates, not both")
    block_candidates = None
    if block_selector is not None:
        if launch is None:
            raise RuntimeError("deep block selector requires runtime launch metadata")
        try:
            block_candidates = linear_block_coordinates(launch.get("grid", []))
        except (TypeError, ValueError) as error:
            raise RuntimeError(str(error)) from error
        selected_linear, block_evidence, block_diagnostic = selection_evidence(
            range(len(block_candidates)), block_selector, seed=seed, dimension="block",
            budget_limit=int(budget["max_sampled_blocks_per_launch"]), policy=policy)
        if not selected_linear:
            raise RuntimeError("deep block selector selected no blocks")
        raw_coordinates = [block_candidates[index] for index in selected_linear]
        block_evidence["selected_coordinates"] = raw_coordinates
        if block_diagnostic is not None:
            diagnostics.append(block_diagnostic)
    else:
        raw_coordinates = raw_coordinates if raw_coordinates is not None else [[0, 0, 0]]
        block_evidence = None
    if not isinstance(raw_coordinates, list) or not raw_coordinates:
        raise RuntimeError("deep block_coordinates must contain at least one [x,y,z] coordinate")
    normalized_coordinates: list[list[int]] = []
    for coordinate in raw_coordinates:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3:
            raise RuntimeError("each deep block coordinate must be an [x,y,z] triple")
        normalized = [int(value) for value in coordinate]
        if any(value < 0 for value in normalized):
            raise RuntimeError("deep block coordinates must be non-negative")
        normalized_coordinates.append(normalized)
    if len({tuple(coordinate) for coordinate in normalized_coordinates}) != len(normalized_coordinates):
        raise RuntimeError("deep block coordinates must be unique")
    if launch is not None:
        raw_grid = launch.get("grid")
        if not isinstance(raw_grid, (list, tuple)) or not 1 <= len(raw_grid) <= 3:
            raise RuntimeError("deep block selection requires a static runtime launch grid")
        try:
            grid = [int(value) for value in raw_grid] + [1] * (3 - len(raw_grid))
        except (TypeError, ValueError) as error:
            raise RuntimeError("deep block selection requires integer runtime grid extents") from error
        if any(extent <= 0 for extent in grid):
            raise RuntimeError("deep block selection requires positive runtime grid extents")
        outside = [coordinate for coordinate in normalized_coordinates
                   if any(value >= grid[axis] for axis, value in enumerate(coordinate))]
        if outside:
            raise RuntimeError(f"deep block coordinates {outside} are outside runtime grid {grid}")
    deep["block_coordinates"] = normalized_coordinates
    mappings = (("block_coordinates", "max_sampled_blocks_per_launch"),
                ("tileop_families", "max_tileops_per_block"))
    for key, limit_key in mappings:
        values = list(deep.get(key, []))
        try:
            bounded, diagnostic = apply_budget(values, int(budget[limit_key]), label=key, policy=policy)
        except BudgetExceeded as error:
            raise RuntimeError(str(error)) from error
        deep[key] = bounded
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    if block_evidence is None:
        evidence_candidates = normalized_coordinates
        if launch is not None:
            evidence_candidates = linear_block_coordinates(launch["grid"])
        coordinate_to_index = {tuple(coordinate): index for index, coordinate in enumerate(evidence_candidates)}
        selected_coordinate_ids = [coordinate_to_index[tuple(coordinate)] for coordinate in deep["block_coordinates"]]
        _, block_evidence, _ = selection_evidence(
            range(len(evidence_candidates)), {"indices": selected_coordinate_ids}, seed=seed, dimension="block",
            budget_limit=int(budget["max_sampled_blocks_per_launch"]), policy=policy)
        block_evidence["selected_coordinates"] = deep["block_coordinates"]
    deep.setdefault("selection_evidence", {}).update({"launch": launch_evidence, "block": block_evidence})
    for key, limit_key in (("max_iterations_per_tileop", "max_iterations_per_tileop"),
                           ("max_trace_records", "max_trace_records"), ("max_trace_bytes", "max_trace_bytes")):
        requested = int(deep.get(key, budget[limit_key]))
        allowed = int(budget[limit_key])
        if requested > allowed:
            if policy == "fail":
                raise RuntimeError(f"{key}: requested {requested}, budget {allowed}")
            deep[key] = allowed
            diagnostics.append({"code": "BUDGET_EXCEEDED", "label": key, "requested": requested, "allowed": allowed, "action": policy})
        else:
            deep[key] = requested


def collect(factory_spec: str, *, config: dict[str, Any], output_root: Path) -> Path:
    session_started = time.monotonic()
    resolved = _resolved_config(config)
    capability = probe(int(resolved["device"]))
    if not capability.get("musa_available"):
        raise RuntimeError("MUSA is unavailable; runtime measurement is not permitted")
    factory = _load_factory(factory_spec)
    workload = factory(resolved)
    if not isinstance(workload, RuntimeWorkload):
        raise TypeError("runtime factory must return RuntimeWorkload")
    baseline_case = workload.prepare(resolved, variant="baseline")
    correctness = verify(lambda: _run_for_correctness(baseline_case), baseline_case.reference, baseline_case.compare)
    requested_mode = str(resolved["mode"])
    diagnostics: list[dict[str, Any]] = []
    total_duration_budget_s = float(resolved["budget"]["max_total_duration_s"])
    import torch
    selected_runtime_stream = (baseline_case.execution_stream if baseline_case.execution_stream is not None
                               else torch.musa.current_stream())
    runtime_stream_id = int(selected_runtime_stream.stream_id)
    deep_scope_selected = True

    def session_budget_available(stage: str) -> bool:
        used = time.monotonic() - session_started
        if used < total_duration_budget_s:
            return True
        if not any(item.get("code") == "SESSION_DURATION_BUDGET_EXHAUSTED" for item in diagnostics):
            diagnostics.append({"code": "SESSION_DURATION_BUDGET_EXHAUSTED", "stage": stage,
                                "elapsed_s": used, "budget_s": total_duration_budget_s,
                                "action": "skip_remaining_collectors"})
        return False
    if requested_mode == "deep":
        deep_scope_selected = _resolve_deep_scope(
            resolved, baseline_case, runtime_stream_id)
        if deep_scope_selected:
            _bound_deep_config(resolved, diagnostics, launch=baseline_case.launch)
        else:
            diagnostics.append({"code": "DEEP_SCOPE_NOT_SELECTED",
                                "message": "case/kernel/stream selectors excluded this deep collector",
                                "action": "baseline_and_standard_only"})
    actual_mode = "light"
    profile_duration_budget_ms = float(resolved["budget"]["max_profile_duration_ms"])
    profile_duration_used_ms = 0.0
    before = snapshot(int(resolved["device"]))
    telemetry_config = resolved.get("telemetry", {})
    sampler = None
    if bool(telemetry_config.get("enabled", True)):
        try:
            sampler = TelemetrySampler(int(resolved["device"]), int(telemetry_config.get("interval_ms", 200)))
            sampler.start()
        except Exception as error:
            diagnostics.append({"code": "TELEMETRY_SAMPLER_UNAVAILABLE", "message": str(error)})
    started = time.monotonic()
    samples = batch_event_samples(baseline_case.run, **{
        "warmup_launches": int(resolved["baseline"]["warmup_launches"]),
        "repeat_launches": int(resolved["baseline"]["repeat_launches"]),
        "batch_size": int(resolved["baseline"]["event_batch_size"]),
        "stream": baseline_case.execution_stream,
    })
    elapsed = time.monotonic() - started
    active_telemetry = sampler.stop() if sampler is not None else {"status": "disabled", "samples": []}
    if sampler is not None and not active_telemetry.get("samples"):
        diagnostics.append({"code": "TELEMETRY_ACTIVE_WINDOW_INSUFFICIENT", "message": "no periodic telemetry sample fell inside the Event window"})
    if elapsed > float(resolved["budget"]["max_total_duration_s"]):
        diagnostics.append({"code": "TOTAL_DURATION_BUDGET_EXCEEDED", "message": f"Event window took {elapsed:.6f}s"})
    after = snapshot(int(resolved["device"]))
    stability = assess_stability(before, active_telemetry, after, samples, resolved["stability"])
    if stability["required"] and stability["status"] != "stable":
        diagnostics.append({"code": "STABILITY_GATE_NOT_MET", "status": stability["status"],
                            "reasons": stability["reasons"], "action": "mark_partial"})
    standard: dict[str, Any] | None = None
    if requested_mode in {"standard", "deep"} and session_budget_available("standard"):
        maximum = min(int(resolved["standard"]["max_profiled_launches"]), int(resolved["budget"]["max_profiled_launches"]))
        try:
            standard = collect_activity(baseline_case.run, launches=maximum,
                                        max_duration_ms=max(profile_duration_budget_ms - profile_duration_used_ms, 1e-9))
            profile_duration_used_ms += float(standard.get("profile_duration_ms", 0.0))
            actual_mode = "standard"
            if standard.get("status") != "complete":
                diagnostics.append({"code": "PROFILE_DURATION_BUDGET_EXCEEDED",
                                    "message": "standard activity window exceeded or exhausted max_profile_duration_ms",
                                    "requested_launches": maximum, "completed_launches": standard.get("launches"),
                                    "profile_duration_ms": standard.get("profile_duration_ms")})
        except Exception as error:
            diagnostics.append({"code": "STANDARD_ACTIVITY_UNAVAILABLE", "message": str(error)})
        metrics = list(resolved["standard"].get("metrics", []))
        if metrics:
            counter = capability.get("collectors", {}).get("mupti", {}).get("raw_counter_adapter", {})
            diagnostics.append({"code": "MUPTI_COUNTER_ADAPTER_UNAVAILABLE", "message": "raw MUPTI counter collection is unavailable",
                                "missing_symbols": counter.get("missing_symbols", [])})
        if bool(resolved["standard"].get("pc_sampling", {}).get("enabled", False)):
            pc = capability.get("collectors", {}).get("mupti", {}).get("pc_sampling", {})
            diagnostics.append({"code": "PC_SAMPLING_UNAVAILABLE", "message": "MUPTI PC sampling is unavailable",
                                "missing_symbols": pc.get("missing_symbols", [])})
    deep: dict[str, Any] | None = None
    overhead: dict[str, Any] = {"status": "not_resolved", "reason": "light Event timing is the baseline collector; no lower-perturbation device timer is available"}
    if requested_mode == "deep" and deep_scope_selected and session_budget_available("deep"):
        # Deep execution is intentionally fail-closed until a workload supplies a separately compiled trace variant.
        try:
            instrumented_case = workload.prepare(resolved, variant="instrumented")
            trace = getattr(instrumented_case, "tile_trace", None)
            if trace is None:
                raise RuntimeError("instrumented workload did not expose a bounded tile_trace reader")
            timestamps_per_launch = int((instrumented_case.trace_metadata or {}).get("timestamps_per_sample", 0))
            trace_bytes_per_launch = int((instrumented_case.trace_metadata or {}).get(
                "device_trace_bytes", timestamps_per_launch * 8))
            if timestamps_per_launch <= 0 or trace_bytes_per_launch <= 0:
                raise RuntimeError("instrumented workload did not expose positive per-launch trace bounds")
            trace_launch_limit = min(
                int(resolved["deep"]["max_trace_records"]) // timestamps_per_launch,
                int(resolved["deep"]["max_trace_bytes"]) // trace_bytes_per_launch,
            )
            selected_launches = list(resolved["deep"]["launch_indices"])
            if len(selected_launches) > trace_launch_limit:
                try:
                    bounded_launches, trace_budget_diagnostic = apply_budget(
                        selected_launches, trace_launch_limit, label="launch_trace_capacity",
                        policy=resolved["budget"]["on_budget_exceeded"])
                except BudgetExceeded as error:
                    raise RuntimeError(str(error)) from error
                if not bounded_launches:
                    raise RuntimeError("trace record/byte budget cannot hold one selected launch")
                resolved["deep"]["launch_indices"] = bounded_launches
                launch_selection = resolved["deep"]["selection_evidence"]["launch"]
                launch_selection["selected"] = bounded_launches
                launch_selection["budget_truncated"] = True
                launch_selection.setdefault("truncations", []).append({
                    "constraint": "aggregate_trace_capacity", "records_per_launch": timestamps_per_launch,
                    "bytes_per_launch": trace_bytes_per_launch, "allowed_launches": trace_launch_limit})
                diagnostics.append(trace_budget_diagnostic)
            instrumented_budget = int(resolved["budget"]["max_instrumented_launches"])
            if instrumented_budget < 1:
                raise RuntimeError("deep trace requires at least one instrumented launch for correctness and trace")
            instrumented_usage = {"budget": instrumented_budget, "used": 0, "remaining": instrumented_budget,
                                  "by_purpose": {"correctness_and_trace": 0, "callable_overhead": 0,
                                                 "kernel_only_profiler": 0}}

            def run_instrumented(purpose: str, function):
                if instrumented_usage["used"] >= instrumented_budget:
                    raise RuntimeError(f"instrumented launch budget exhausted before {purpose}")
                # Debit before submission so a launch that raises later is
                # still accounted for rather than disappearing from evidence.
                instrumented_usage["used"] += 1
                instrumented_usage["by_purpose"][purpose] += 1
                instrumented_usage["remaining"] = instrumented_budget - instrumented_usage["used"]
                return function()

            async_probe = probe_async_d2h()
            requested_readback = str(resolved.get("deep", {}).get("readback_policy", "async_after_window"))
            use_async = (requested_readback == "async_after_window" and async_probe["status"] == "available"
                         and instrumented_case.enqueue_trace_readback is not None
                         and instrumented_case.finish_trace_readback is not None)
            if requested_readback == "async_after_window" and not use_async:
                diagnostics.append({"code": "ASYNC_D2H_UNAVAILABLE", "message": async_probe["reason"],
                                    "action": "sync_after_window"})
            actual_readback = "async_after_window" if use_async else "sync_after_window"
            selected_launches = list(resolved.get("deep", {}).get("launch_indices", [0]))
            candidate_count = int(resolved.get("deep", {}).get("launch_candidate_count", 1))
            selected_set = set(selected_launches)
            trace_values: list[dict[str, Any]] = []
            validations: list[dict[str, Any]] = []
            joins: list[dict[str, Any]] = []
            readback_runs: list[dict[str, Any]] = []
            correctness_runs: list[dict[str, Any]] = []
            # Walk the declared logical launch window. Unselected positions
            # execute the unmodified baseline callable; only selected positions
            # consume the instrumented budget and write the trace buffer.
            for launch_index in range(candidate_count):
                if time.monotonic() - session_started >= total_duration_budget_s:
                    raise RuntimeError(
                        f"session duration budget exhausted inside logical launch window at index {launch_index}")
                if launch_index not in selected_set:
                    baseline_case.run()
                    continue
                trace_slot = TraceBufferSlot(slot_id=len(readback_runs))
                trace_slot.begin_write()
                correctness_run = verify(
                    lambda: run_instrumented("correctness_and_trace", lambda: _run_for_correctness(instrumented_case)),
                    instrumented_case.reference, instrumented_case.compare)
                if correctness_run["status"] != "pass":
                    raise RuntimeError(f"instrumented correctness failed at logical launch {launch_index}")
                correctness_runs.append({"launch_index": launch_index, **correctness_run})
                if instrumented_case.current_trace_generation is not None:
                    trace_slot.generation_id = int(instrumented_case.current_trace_generation())
                trace_slot.transition(TraceBufferState.DEVICE_READY)
                readback_details: dict[str, Any] = {}
                if use_async:
                    trace_slot.transition(TraceBufferState.D2H_COPYING)
                    readback_details = instrumented_case.enqueue_trace_readback()
                    if int(readback_details.get("generation_id", -1)) != trace_slot.generation_id:
                        raise RuntimeError("trace readback generation does not match the device-writing generation")
                    one_trace = instrumented_case.finish_trace_readback()
                else:
                    one_trace = trace()
                trace_slot.transition(TraceBufferState.HOST_READY)
                trace_slot.transition(TraceBufferState.FREE)
                validation = validate_trace(instrumented_case, one_trace, resolved.get("deep", {}))
                association = associate_offline_tileops(instrumented_case, one_trace, resolved.get("deep", {}))
                trace_values.extend([{**record, "launch_index": launch_index} for record in one_trace])
                validations.append({"launch_index": launch_index, **validation})
                if association["status"] == "complete":
                    association["trace_records"] = [
                        {**record, "launch_index": launch_index} for record in association["trace_records"]]
                joins.append(association)
                readback_runs.append({"launch_index": launch_index, "details": readback_details,
                                      "slot": trace_slot.manifest()})
            if [item["launch_index"] for item in correctness_runs] != selected_launches:
                raise RuntimeError("executed instrumented launch set differs from resolved selection")
            instrumented_correctness = {"status": "pass", "selected_launches": selected_launches,
                                        "runs": correctness_runs}
            trace_validation = {"status": "pass", "selected_launches": selected_launches,
                                "samples": validations,
                                **{key: value for key, value in validations[0].items()
                                   if key not in {"status", "launch_index"}}}
            first_join = joins[0]
            if any(item["status"] != first_join["status"] for item in joins):
                raise RuntimeError("offline/runtime association status differs across selected launches")
            offline_runtime_join = dict(first_join)
            if first_join["status"] == "complete":
                offline_runtime_join["trace_records"] = [record for item in joins for record in item["trace_records"]]
                offline_runtime_join["trace_record_count"] = len(offline_runtime_join["trace_records"])
            deep = {"status": "complete", "instrumented_correctness": instrumented_correctness,
                    "lineage": instrumented_case.lineage or [], "trace": trace_values,
                    "instrumented_kernel_hash": sha256_text(instrumented_case.kernel_source),
                    "trace_metadata": instrumented_case.trace_metadata or {},
                    "trace_validation": trace_validation,
                    "offline_runtime_join": offline_runtime_join,
                    "instrumented_launch_usage": instrumented_usage,
                    "selection": resolved.get("deep", {}).get("selection_evidence", {}),
                    "readback": {"requested_policy": requested_readback, "actual_policy": actual_readback,
                                 "async_probe": async_probe, "runs": readback_runs,
                                 "details": readback_runs[0]["details"],
                                 "slot": readback_runs[0]["slot"]}}
            # ABBA invokes the instrumented callable twice. Bound each B group
            # to half of the remaining all-purpose instrumented launch budget.
            remaining = instrumented_budget - instrumented_usage["used"]
            audit_repeats = min(500, int(resolved["baseline"]["repeat_launches"]), remaining // 2)
            if audit_repeats > 0:
                audit_batch = min(int(resolved["baseline"]["event_batch_size"]), audit_repeats)
                overhead = compare_overhead(
                    lambda: batch_event_samples(baseline_case.run, warmup_launches=0, repeat_launches=audit_repeats, batch_size=audit_batch,
                                                stream=baseline_case.execution_stream),
                    lambda: batch_event_samples(
                        lambda: run_instrumented("callable_overhead", instrumented_case.run_timed or instrumented_case.run),
                        warmup_launches=0, repeat_launches=audit_repeats, batch_size=audit_batch,
                        instrumented=True, stream=instrumented_case.execution_stream),
                )
            else:
                overhead = {"status": "unavailable", "reason": "instrumented launch budget is exhausted after correctness/trace"}
            overhead["collector"] = "instrumented_tile_trace"
            overhead["instrumented"] = True
            # A separate profiler window supplies the kernel-only domain. It
            # is never included in callable Event samples or baseline reports.
            remaining = instrumented_budget - instrumented_usage["used"]
            profile_launches = min(10, int(resolved["standard"]["max_profiled_launches"]),
                                   int(resolved["budget"]["max_profiled_launches"]), remaining)
            try:
                if profile_launches <= 0:
                    raise RuntimeError("instrumented launch budget has no capacity for kernel-only profiling")
                profile_time_remaining = profile_duration_budget_ms - profile_duration_used_ms
                if profile_time_remaining <= 0:
                    raise RuntimeError("profile duration budget is exhausted before kernel-only profiling")
                baseline_activity = collect_activity(baseline_case.run, launches=profile_launches,
                                                     max_duration_ms=profile_time_remaining)
                profile_duration_used_ms += float(baseline_activity.get("profile_duration_ms", 0.0))
                matched_launches = min(profile_launches, int(baseline_activity.get("launches", 0)))
                profile_time_remaining = profile_duration_budget_ms - profile_duration_used_ms
                if matched_launches <= 0 or profile_time_remaining <= 0:
                    raise RuntimeError("profile duration budget cannot hold a matched instrumented window")
                instrumented_activity = collect_activity(
                                                         lambda: run_instrumented(
                                                             "kernel_only_profiler",
                                                             instrumented_case.run_timed or instrumented_case.run),
                                                         launches=matched_launches,
                                                         max_duration_ms=profile_time_remaining)
                actual_profiled = int(instrumented_activity.get("launches", 0))
                profile_duration_used_ms += float(instrumented_activity.get("profile_duration_ms", 0.0))
                overhead["kernel_only"] = kernel_only_from_activity(baseline_activity, instrumented_activity)
                overhead["kernel_only"]["profile_launches"] = actual_profiled
                overhead["kernel_only"]["requested_profile_launches"] = profile_launches
                overhead["kernel_only"]["profile_duration_budget_ms"] = profile_duration_budget_ms
                overhead["kernel_only"]["profile_duration_used_ms"] = profile_duration_used_ms
            except Exception as error:
                overhead["kernel_only"] = {"status": "unavailable", "reason": str(error)}
            overhead["domains"] = {
                "callable": "complete; MUSA Event ABBA; excludes trace readback",
                "kernel_only": overhead["kernel_only"]["status"],
                "readback": f"{actual_readback}; excluded from Event and profiler windows",
                "mixed_session": "not separately benchmarked in this single-window implementation",
            }
            instrumented_usage["remaining"] = instrumented_budget - instrumented_usage["used"]
            overhead["instrumented_launch_usage"] = instrumented_usage
            if instrumented_usage["used"] > instrumented_budget:
                raise RuntimeError("instrumented launch accounting exceeded its hard budget")
            actual_mode = "deep"
        except Exception as error:
            diagnostics.append({"code": "DEEP_TILE_TRACE_UNAVAILABLE", "message": str(error)})
    stream_sync = audit_stream_sync(samples, runtime_stream_id, standard, deep)
    if stream_sync["status"] != "pass":
        diagnostics.append({"code": "STREAM_SYNC_AUDIT_FAILED", "message": "stream/sync evidence is inconsistent",
                            "violations": stream_sync["violations"]})
    status = "complete" if correctness["status"] == "pass" and actual_mode == requested_mode else "partial"
    if any(item["code"] in {"TELEMETRY_ACTIVE_WINDOW_INSUFFICIENT", "TOTAL_DURATION_BUDGET_EXCEEDED", "ASYNC_D2H_UNAVAILABLE",
                             "MUPTI_COUNTER_ADAPTER_UNAVAILABLE", "PC_SAMPLING_UNAVAILABLE",
                             "PROFILE_DURATION_BUDGET_EXCEEDED", "SESSION_DURATION_BUDGET_EXHAUSTED",
                             "TELEMETRY_SAMPLER_UNAVAILABLE", "BUDGET_EXCEEDED",
                             "STABILITY_GATE_NOT_MET"} for item in diagnostics):
        status = "partial"
    if correctness["status"] != "pass":
        status = "invalid"
    if stream_sync["status"] != "pass":
        status = "invalid"
    session_elapsed = time.monotonic() - session_started
    if session_elapsed > float(resolved["budget"]["max_total_duration_s"]):
        diagnostics.append({"code": "SESSION_DURATION_BUDGET_EXCEEDED", "message": f"session took {session_elapsed:.6f}s"})
        if status == "complete":
            status = "partial"
    baseline_hash = sha256_text(baseline_case.kernel_source)
    runtime_launch = dict(baseline_case.launch)
    runtime_launch.update({"device": int(resolved["device"]), "stream_id": runtime_stream_id,
                           "launch_evidence_level": "runtime_direct"})
    session_id = stable_id("runtime", [factory_spec, resolved, baseline_hash])
    output = output_root / session_id
    manifest = {"schema_version": "musa-perf-collector/runtime-measurement-v1", "session_id": session_id,
                "workload": baseline_case.name, "requested_mode": requested_mode, "actual_mode": actual_mode,
                "status": status, "device": resolved["device"], "baseline_kernel_id": baseline_case.kernel_id,
                "baseline_kernel_hash": baseline_hash, "instrumented": deep is not None, "cache_policy": resolved["baseline"]["cache_policy"],
                "measurement_duration_s": elapsed, "session_duration_s": session_elapsed, "launch": runtime_launch}
    if deep is not None:
        manifest["instrumented_kernel_hash"] = deep["instrumented_kernel_hash"]
    summary = _summary(samples)
    runtime_validation = validate_session(manifest, correctness, summary, diagnostics, deep, resolved,
                                          stability=stability, stream_sync=stream_sync)
    if runtime_validation["status"] != "pass":
        manifest["status"] = "invalid"
        diagnostics.append({"code": "RUNTIME_SCHEMA_INVALID",
                            "message": "independent runtime bundle validation failed",
                            "violations": runtime_validation["violations"]})
    write_json(output / "manifest.json", manifest)
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "capability_probe.json", capability)
    write_json(output / "correctness.json", correctness)
    write_json(output / "overhead.json", overhead)
    write_json(output / "runtime_validation.json", runtime_validation)
    write_json(output / "stream_sync.json", stream_sync)
    if requested_mode == "deep":
        write_json(output / "deep" / "selection.json", resolved.get("deep", {}).get("selection_evidence", {}))
    # Preserve the exact generated code behind both hashes.  This is evidence
    # for review only; it is written after all timed windows and is never used
    # by a runtime workload.
    write_text(output / "baseline" / "kernel.mu", baseline_case.kernel_source)
    write_json(output / "baseline" / "summary.json", summary)
    write_jsonl(output / "baseline" / "event_samples.jsonl", samples)
    write_json(output / "telemetry" / "before.json", before)
    write_json(output / "telemetry" / "after.json", after)
    write_json(output / "telemetry" / "active.json", active_telemetry)
    write_json(output / "telemetry" / "stability.json", stability)
    if standard is not None:
        write_json(output / "standard" / "activity.json", standard)
    if deep is not None:
        instrumentation = {key: value for key, value in deep.items() if key not in {"trace", "offline_runtime_join"}}
        instrumentation["offline_runtime_join"] = {
            key: value for key, value in deep["offline_runtime_join"].items()
            if key not in {"tileops", "trace_records"}
        }
        write_json(output / "deep" / "instrumentation_manifest.json", instrumentation)
        write_json(output / "deep" / "readback.json", deep["readback"])
        write_jsonl(output / "deep" / "trace.jsonl", deep["trace"])
        write_json(output / "deep" / "offline_runtime_join.json", {
            key: value for key, value in deep["offline_runtime_join"].items()
            if key not in {"tileops", "trace_records"}
        })
        if deep["offline_runtime_join"]["status"] == "complete":
            write_jsonl(output / "deep" / "tileop_context.jsonl", deep["offline_runtime_join"]["tileops"])
            write_jsonl(output / "deep" / "trace_enriched.jsonl", deep["offline_runtime_join"]["trace_records"])
        write_text(output / "deep" / "instrumented_kernel.mu", instrumented_case.kernel_source)
    diagnostic_record = {"diagnostics": diagnostics}
    write_json(output / "diagnostics.json", diagnostic_record)
    markdown = render_markdown(manifest, correctness, summary, diagnostic_record, overhead,
                               deep["offline_runtime_join"] if deep is not None else None)
    write_text(output / "report" / "report.md", markdown)
    write_text(output / "report" / "report.html", render_html(markdown))
    return output
