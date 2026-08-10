"""Independent fail-closed validation for runtime measurement bundles."""

from __future__ import annotations

from typing import Any


ALLOWED_STATUSES = {"complete", "partial", "failed", "invalid"}
REQUIRED_SELECTION_DIMENSIONS = {"case", "kernel", "stream", "launch", "block"}


def validate_session(
    manifest: dict[str, Any], correctness: dict[str, Any], summary: dict[str, Any],
    diagnostics: list[dict[str, Any]], deep: dict[str, Any] | None,
    resolved: dict[str, Any], *, stability: dict[str, Any] | None = None,
    stream_sync: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every schema violation; callers must mark any violation invalid."""
    violations: list[dict[str, str]] = []

    def require(condition: bool, code: str, message: str) -> None:
        if not condition:
            violations.append({"code": code, "message": message})

    status = manifest.get("status")
    require(manifest.get("schema_version") == "musa-perf-collector/runtime-measurement-v1",
            "SCHEMA_VERSION_INVALID", "runtime manifest schema_version is not supported")
    require(status in ALLOWED_STATUSES, "STATUS_INVALID", "runtime manifest status is not recognized")
    require(summary.get("instrumented") is False, "BASELINE_MARKED_INSTRUMENTED",
            "baseline summary must be explicitly uninstrumented")
    require(correctness.get("status") in {"pass", "fail"}, "CORRECTNESS_STATUS_INVALID",
            "baseline correctness status must be pass or fail")
    if correctness.get("status") != "pass":
        require(status == "invalid", "CORRECTNESS_NOT_INVALID", "failed correctness must make the session invalid")
    if status == "complete":
        require(manifest.get("requested_mode") == manifest.get("actual_mode"),
                "COMPLETE_MODE_MISMATCH", "complete session must finish its requested mode")
        require(not diagnostics, "COMPLETE_HAS_DIAGNOSTICS", "complete session must not carry degradation diagnostics")
    if bool(resolved.get("stability", {}).get("required", False)):
        require(stability is not None and stability.get("status") == "stable",
                "STABILITY_GATE_INVALID", "required thermal/timing stability evidence did not pass")
    if stream_sync is not None:
        require(stream_sync.get("status") == "pass", "STREAM_SYNC_GATE_INVALID",
                "stream/synchronization evidence did not pass")

    if deep is None:
        require(manifest.get("instrumented") is False, "INSTRUMENTED_FLAG_MISMATCH",
                "manifest claims instrumentation without deep evidence")
    else:
        require(manifest.get("instrumented") is True, "INSTRUMENTED_FLAG_MISMATCH",
                "deep evidence requires manifest.instrumented=true")
        require(deep.get("status") == "complete", "DEEP_STATUS_INVALID", "persisted deep evidence is incomplete")
        require(deep.get("instrumented_correctness", {}).get("status") == "pass",
                "INSTRUMENTED_CORRECTNESS_INVALID", "instrumented correctness did not pass")
        require(deep.get("trace_validation", {}).get("status") == "pass",
                "TRACE_VALIDATION_INVALID", "deep trace validation did not pass")
        require(manifest.get("baseline_kernel_hash") != manifest.get("instrumented_kernel_hash"),
                "KERNEL_HASH_NOT_ISOLATED", "baseline and instrumented kernel hashes must differ")
        selection = deep.get("selection", {})
        require(REQUIRED_SELECTION_DIMENSIONS.issubset(selection), "SELECTION_EVIDENCE_INCOMPLETE",
                "deep evidence lacks one or more case/kernel/stream/launch/block selections")
        expected_launches = list(resolved.get("deep", {}).get("launch_indices", []))
        observed_launches = sorted({int(item.get("launch_index", -1)) for item in deep.get("trace", [])})
        require(observed_launches == expected_launches, "TRACE_LAUNCH_SET_MISMATCH",
                "trace launch_index set differs from the resolved selection")
        require(len(deep.get("trace", [])) <= int(resolved["deep"]["max_trace_records"]),
                "TRACE_RECORD_BUDGET_EXCEEDED", "aggregate trace records exceed the resolved limit")
        usage = deep.get("instrumented_launch_usage", {})
        require(int(usage.get("used", -1)) <= int(usage.get("budget", -1)),
                "INSTRUMENTED_BUDGET_EXCEEDED", "instrumented launch usage exceeds its hard budget")
        readback_launches = [int(item.get("launch_index", -1)) for item in deep.get("readback", {}).get("runs", [])]
        require(readback_launches == expected_launches, "READBACK_LAUNCH_SET_MISMATCH",
                "trace readback generations differ from the resolved launch selection")

    return {
        "schema": "musa-perf-collector/runtime-validation-v1",
        "status": "pass" if not violations else "fail",
        "violation_count": len(violations),
        "violations": violations,
        "checked_invariants": [
            "schema_and_status", "baseline_correctness", "baseline_not_instrumented",
            "requested_actual_mode", "deep_correctness_and_trace", "kernel_hash_isolation",
            "selection_dimensions", "trace_launch_set", "trace_record_budget",
            "instrumented_launch_budget", "readback_launch_set",
            "thermal_and_timing_stability",
            "stream_and_synchronization",
        ],
    }
