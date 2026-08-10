"""Audit stream identity and synchronization evidence across collectors."""

from __future__ import annotations

from typing import Any


def audit(
    samples: list[dict[str, Any]], expected_stream_id: int,
    standard: dict[str, Any] | None, deep: dict[str, Any] | None,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    observed = sorted({int(item.get("stream_id", -1)) for item in samples})
    if observed != [expected_stream_id]:
        violations.append({"code": "BASELINE_STREAM_MISMATCH", "expected": expected_stream_id,
                           "observed": observed})
    incomplete_events = [int(item.get("sample_index", -1)) for item in samples
                         if item.get("completion_event_synchronized") is not True or
                         item.get("synchronization_scope") != "batch_end_event"]
    if incomplete_events:
        violations.append({"code": "BASELINE_EVENT_COMPLETION_MISSING", "sample_indices": incomplete_events})
    readbacks: list[dict[str, Any]] = []
    if deep is not None and deep.get("readback", {}).get("actual_policy") == "async_after_window":
        for run in deep["readback"].get("runs", []):
            details = run.get("details", {})
            evidence = {
                "launch_index": int(run.get("launch_index", -1)),
                "producer_stream_id": details.get("producer_stream_id"),
                "collector_stream_id": details.get("collector_stream_id"),
                "producer_event_recorded": details.get("producer_event_recorded"),
                "collector_waited_for_producer": details.get("collector_waited_for_producer"),
                "copy_done_event_recorded": details.get("copy_done_event_recorded"),
            }
            readbacks.append(evidence)
            if (evidence["producer_stream_id"] != expected_stream_id or
                    evidence["collector_stream_id"] in {None, expected_stream_id} or
                    evidence["producer_event_recorded"] is not True or
                    evidence["collector_waited_for_producer"] is not True or
                    evidence["copy_done_event_recorded"] is not True):
                violations.append({"code": "ASYNC_READBACK_STREAM_SYNC_INVALID", **evidence})
    profiler_sync_events = []
    if standard is not None:
        profiler_sync_events = [item for item in standard.get("events", [])
                                if any(token in str(item.get("name", "")).lower()
                                       for token in ("sync", "wait", "event"))]
    return {
        "schema": "musa-perf-collector/runtime-stream-sync-v1",
        "status": "pass" if not violations else "fail",
        "expected_baseline_stream_id": expected_stream_id,
        "observed_baseline_stream_ids": observed,
        "event_batch_count": len(samples),
        "event_completion_policy": "batch_end_event",
        "async_readbacks": readbacks,
        "profiler_sync_event_count": len(profiler_sync_events),
        "profiler_sync_events": profiler_sync_events,
        "violations": violations,
    }
