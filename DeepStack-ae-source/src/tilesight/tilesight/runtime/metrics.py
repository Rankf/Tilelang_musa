"""Canonical S5000 runtime metric semantics used by TileSight reports."""

from __future__ import annotations

from typing import Any


METRIC_SEMANTICS: dict[str, dict[str, Any]] = {
    "kernel_time_us": {
        "scope": "target_kernel_device_execution_window",
        "unit": "us",
        "comparison": "latency_us_full_logical_output_writeback",
    },
    "ddr_read_bytes": {
        "scope": "target_kernel_physical_ddr_transactions",
        "unit": "bytes",
        "comparison": "definition_pending",
    },
    "ddr_write_bytes": {
        "scope": "target_kernel_physical_ddr_transactions",
        "unit": "bytes",
        "comparison": "kernel_scope_write_bytes_definition_pending",
    },
    "ddr_total_bytes": {
        "scope": "target_kernel_physical_ddr_transactions",
        "unit": "bytes",
        "comparison": "definition_pending",
    },
    "ddr_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "peak_denominator_requires_audit",
    },
    "l1_downstream_read_bytes": {
        "scope": "target_kernel_l1_downstream_transactions",
        "unit": "bytes",
        "comparison": "definition_pending",
    },
    "l1_downstream_write_bytes": {
        "scope": "target_kernel_l1_downstream_transactions",
        "unit": "bytes",
        "comparison": "definition_pending",
    },
    "l1_sector_hit_rate_pct": {
        "scope": "target_kernel_l1_mixed_sector_requests",
        "unit": "percent",
        "comparison": "not_logical_reuse_rate",
    },
    "l1_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "peak_denominator_requires_audit",
    },
    "l2pu_sector_hit_rate_pct": {
        "scope": "target_kernel_l2pu_mixed_cacheline_requests",
        "unit": "percent",
        "comparison": "not_directly_comparable_to_logical_l2_reuse",
    },
    "l2_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "peak_denominator_requires_audit",
    },
    "llc_read_sector_hit_rate_pct": {
        "scope": "target_kernel_llc_read_sector_requests",
        "unit": "percent",
        "comparison": "direct_when_model_scope_is_read_sector",
    },
    "compute_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "runtime_bottleneck_evidence",
    },
    "smem_load_pipe_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "runtime_bottleneck_evidence",
    },
    "smem_store_pipe_util_pct": {
        "scope": "target_kernel_counter_window",
        "unit": "percent",
        "comparison": "runtime_bottleneck_evidence",
    },
    "waves_per_multiprocessor": {
        "scope": "target_launch_grid_and_residency",
        "unit": "waves/MP",
        "comparison": "directly_comparable",
    },
    "launch_grid_size": {
        "scope": "target_launch",
        "unit": "blocks",
        "comparison": "identity",
    },
    "launch_block_size": {
        "scope": "target_launch",
        "unit": "threads",
        "comparison": "identity",
    },
    "dynamic_shared_memory_bytes": {
        "scope": "target_launch",
        "unit": "bytes/block",
        "comparison": "identity",
    },
    "registers_per_thread": {
        "scope": "target_kernel_binary_resource",
        "unit": "registers/thread",
        "comparison": "directly_comparable_to_binary_resource",
    },
}


def metric_semantics(name: str) -> dict[str, Any]:
    return dict(METRIC_SEMANTICS.get(name, {
        "scope": "unknown",
        "unit": "not_specified",
        "comparison": "unreviewed",
    }))
