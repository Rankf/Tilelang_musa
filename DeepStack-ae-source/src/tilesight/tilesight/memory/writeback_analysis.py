"""First-stage kernel-scope writeback analysis.

This module deliberately models a *single kernel execution window*.  It
separates logical output stores from dirty data that is predicted to reach DDR
before that window ends.  It is a continuous capacity approximation, not a
replacement for the later access-graph/cache-state-machine implementation.
"""

from __future__ import annotations

from typing import Dict


_CACHED_STORE_PATHS = {"ordinary_cached", "ordinary"}
_BYPASS_STORE_PATHS = {"bypass"}


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def estimate_kernel_scope_writeback(
    *,
    live_read_bytes: float,
    logical_output_write_bytes: float,
    effective_l2_capacity_bytes: float,
    store_path: str = "ordinary_cached",
    cache_condition: str = "cold",
    physical_write_amplification: float = 1.0,
    dirty_retention_capacity_bytes: float | None = None,
    work_mode: str = "single_kernel",
) -> Dict[str, object]:
    """Estimate output writeback visible inside one kernel window.

    ``live_read_bytes`` is the portion of input data still live while output
    stores become dirty.  The initial Elementwise integration uses its
    read-working-set size as a conservative first-stage approximation.  Later
    access-lifetime analysis will provide a tighter value.

    Only cached ordinary stores and explicit bypass stores are supported in
    this first-stage model.  Atomic, scatter and TMA stores require their own
    access/path semantics and must not silently reuse this approximation.
    """
    for name, value in (
        ("live_read_bytes", live_read_bytes),
        ("logical_output_write_bytes", logical_output_write_bytes),
        ("effective_l2_capacity_bytes", effective_l2_capacity_bytes),
        ("physical_write_amplification", physical_write_amplification),
    ):
        _require_nonnegative(name, value)
    if dirty_retention_capacity_bytes is not None:
        _require_nonnegative(
            "dirty_retention_capacity_bytes", dirty_retention_capacity_bytes
        )

    if cache_condition not in {"cold", "warm"}:
        raise ValueError(
            "cache_condition must be either 'cold' or 'warm', got "
            f"{cache_condition!r}"
        )
    if work_mode != "single_kernel":
        raise ValueError(
            "first-stage writeback analysis currently supports only "
            f"'single_kernel', got {work_mode!r}"
        )

    resident_pressure = live_read_bytes + logical_output_write_bytes
    if store_path in _CACHED_STORE_PATHS:
        if dirty_retention_capacity_bytes is not None:
            # Calibrated streaming-store form: completed input lines cease to
            # be live, while a bounded amount of dirty output survives the
            # kernel window.  This avoids treating the entire input tensor as
            # simultaneously resident with the output.
            overflow_bytes = max(
                0.0, logical_output_write_bytes - dirty_retention_capacity_bytes
            )
            kernel_scope_logical_evicted_bytes = overflow_bytes
            write_prediction_method = "streaming_dirty_retention_v1"
        else:
            overflow_bytes = max(0.0, resident_pressure - effective_l2_capacity_bytes)
            kernel_scope_logical_evicted_bytes = min(
                logical_output_write_bytes, overflow_bytes
            )
            write_prediction_method = "continuous_live_pressure_v1"
        deferred_writeback_bytes = (
            logical_output_write_bytes - kernel_scope_logical_evicted_bytes
        )
        confidence = "low"
        unknowns = [
            "access_order",
            "l2_address_hash",
            "replacement_policy",
            "store_allocation_policy",
        ]
    elif store_path in _BYPASS_STORE_PATHS:
        overflow_bytes = logical_output_write_bytes
        kernel_scope_logical_evicted_bytes = logical_output_write_bytes
        deferred_writeback_bytes = 0.0
        write_prediction_method = "bypass_store_in_kernel_scope"
        confidence = "medium"
        unknowns = []
    else:
        raise ValueError(
            "first-stage writeback analysis supports ordinary_cached or "
            f"bypass stores, got {store_path!r}"
        )

    kernel_scope_ddr_write_bytes = (
        kernel_scope_logical_evicted_bytes * physical_write_amplification
    )
    writeback_ratio = (
        kernel_scope_logical_evicted_bytes / logical_output_write_bytes
        if logical_output_write_bytes
        else 0.0
    )

    return {
        "cache_condition": cache_condition,
        "work_mode": work_mode,
        "store_path": store_path,
        "writeback_model": "continuous_capacity_v1",
        "write_prediction_method": write_prediction_method,
        "live_read_bytes": live_read_bytes,
        "logical_output_write_bytes": logical_output_write_bytes,
        "resident_pressure_bytes": resident_pressure,
        "effective_l2_capacity_bytes": effective_l2_capacity_bytes,
        "dirty_retention_capacity_bytes": dirty_retention_capacity_bytes,
        "overflow_bytes": overflow_bytes,
        "kernel_scope_logical_evicted_bytes": kernel_scope_logical_evicted_bytes,
        "kernel_scope_ddr_write_bytes": kernel_scope_ddr_write_bytes,
        "deferred_writeback_bytes": deferred_writeback_bytes,
        "steady_state_write_bytes_per_launch": None,
        "physical_write_amplification": physical_write_amplification,
        "writeback_ratio": writeback_ratio,
        "measurement_scope": "single_kernel",
        "value_kind": "model_estimate",
        "confidence": confidence,
        "unknowns": unknowns,
    }
