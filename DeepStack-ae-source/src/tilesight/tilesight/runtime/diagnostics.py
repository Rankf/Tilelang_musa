"""Joint S5000 diagnostics for indicators 5.6 through 5.10."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .comparison import compare_with_runtime
from .schema import load_s5000_runtime, metric_value, select_runtime_record


SCHEMA_VERSION = "tilesight/runtime-diagnostics-v1"


def _path(data: Mapping[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _model_bottleneck(model: Mapping[str, Any]) -> str | None:
    return (
        _path(model, "memory_hierarchy_prediction", "critical_memory_path", "bottleneck")
        or model.get("bottleneck")
        or _path(model, "prediction", "bottleneck")
    )


def _runtime_bottleneck(record: Mapping[str, Any]) -> dict[str, Any]:
    smem_load = metric_value(record, "smem_load_pipe_util_pct")
    smem_store = metric_value(record, "smem_store_pipe_util_pct")
    smem_values = [value for value in (smem_load, smem_store) if value is not None]
    values = {
        "compute": metric_value(record, "compute_util_pct"),
        "shared_memory": max(smem_values) if smem_values else None,
        "l1": metric_value(record, "l1_util_pct"),
        "l2": metric_value(record, "l2_util_pct"),
        "ddr": metric_value(record, "ddr_util_pct"),
    }
    available = {
        key: value for key, value in values.items()
        if value is not None and 0.0 <= value <= 100.0
    }
    unranked = {
        key: value for key, value in values.items()
        if value is not None and not 0.0 <= value <= 100.0
    }
    if not available:
        return {
            "status": "unavailable",
            "reason": "运行时记录缺少可比较的计算和存储层级利用率",
            "utilization_pct": values,
            "unranked_out_of_range_pct": unranked,
        }
    primary = max(available, key=available.get)
    return {
        "status": "observed",
        "critical_resource": primary,
        "utilization_pct": values,
        "shared_memory_pipe_utilization_pct": {
            "load": smem_load,
            "store": smem_store,
        },
        "unranked_out_of_range_pct": unranked,
        "confidence": "medium" if len(available) >= 3 else "low",
        "reason": (
            f"有效0至100%利用率中{primary}最高；"
            "缺失或超范围counter不参与排序"
        ),
    }


def _synchronization_finding(model: Mapping[str, Any]) -> dict[str, Any]:
    pipeline = model.get("pipeline", {})
    if not isinstance(pipeline, Mapping):
        pipeline = {}
    count = _number(pipeline.get("synchronization_count"))
    cost = _number(pipeline.get("synchronization_us"))
    declared = pipeline.get("synchronization_status")
    if count == 0:
        return {
            "status": "not_applicable",
            "synchronization_count": 0,
            "synchronization_us": cost or 0.0,
            "synchronization_bottleneck": False,
            "reason": declared or "模型中没有显式同步",
        }
    if count is None or cost is None:
        return {
            "status": "unavailable",
            "synchronization_count": count,
            "synchronization_us": cost,
            "synchronization_bottleneck": None,
            "reason": "模型未同时输出同步次数和同步代价",
        }
    kernel_latency = _number(model.get("kernel_scope_latency_us"))
    if kernel_latency is None:
        kernel_latency = _number(model.get("latency_us"))
    share = cost / kernel_latency * 100.0 if kernel_latency else None
    return {
        "status": "predicted",
        "synchronization_count": int(count),
        "synchronization_us": cost,
        "synchronization_cost_pct": share,
        "synchronization_bottleneck": (
            share is not None and share >= 50.0
        ),
        "reason": declared or "基于模型同步代价判断",
    }


def _model_load_distribution(model: Mapping[str, Any]) -> dict[str, Any]:
    grid = _number(_path(model, "launch", "grid_size"))
    mp_count = _number(_path(model, "hardware", "sm_count"))
    resident = _number(_path(model, "load_distribution", "resident_tiles_per_sm"))
    if resident is None:
        resident = _number(model.get("tiles_per_sm"))
    if grid is None or mp_count is None or resident is None:
        return {
            "status": "unavailable",
            "grid_ctas": grid,
            "mp_count": mp_count,
            "resident_ctas_per_mp": resident,
            "reason": "缺少Grid、MP数量或每MP驻留CTA数",
        }
    grid_i, mp_i, resident_i = int(grid), int(mp_count), int(resident)
    if min(grid_i, mp_i, resident_i) <= 0:
        return {"status": "unavailable", "reason": "负载分布输入必须为正数"}
    slots_per_wave = mp_i * resident_i
    wave_count = math.ceil(grid_i / slots_per_wave)
    tail_ctas = grid_i % slots_per_wave
    if tail_ctas == 0:
        tail_ctas = slots_per_wave
    quotient, remainder = divmod(grid_i, mp_i)
    maximum_ctas = quotient + (1 if remainder else 0)
    mean_ctas = grid_i / mp_i
    cta_groups = [{"cta_count": quotient, "mp_count": mp_i - remainder}]
    if remainder:
        cta_groups.append({"cta_count": quotient + 1, "mp_count": remainder})
    return {
        "status": "model_estimate",
        "grid_ctas": grid_i,
        "mp_count": mp_i,
        "resident_ctas_per_mp": resident_i,
        "waves_per_mp": grid_i / slots_per_wave,
        "wave_count": wave_count,
        "overall_slot_utilization_pct": (
            grid_i / (wave_count * slots_per_wave) * 100.0
        ),
        "idle_resident_slots": wave_count * slots_per_wave - grid_i,
        "tail_wave_utilization_pct": tail_ctas / slots_per_wave * 100.0,
        "tail_wave_idle_pct": 100.0 - tail_ctas / slots_per_wave * 100.0,
        "theoretical_ctas_per_mp": {
            "minimum": quotient,
            "maximum": maximum_ctas,
            "mean": mean_ctas,
            "spread": maximum_ctas - quotient,
            "spread_pct_of_mean": (
                (maximum_ctas - quotient) / mean_ctas * 100.0
            ),
            "distribution_groups": cta_groups,
        },
        "load_imbalance": False,
        "load_imbalance_status": "balanced_theoretical_assignment",
        "tail_wave_underutilization": tail_ctas != slots_per_wave,
        "resource_idle": wave_count * slots_per_wave != grid_i,
        "value_kind": "model_estimate",
    }


def diagnose_with_runtime(
    model_result: Mapping[str, Any],
    runtime: Mapping[str, Any] | str | Path,
    record_id: str,
    *,
    require_identity: bool = True,
) -> dict[str, Any]:
    """Produce 5.6-5.10 model/runtime findings without changing prediction."""
    bundle = load_s5000_runtime(runtime)
    record = select_runtime_record(bundle, record_id)
    comparison = compare_with_runtime(
        model_result, bundle, record_id, require_identity=require_identity
    )

    model_bottleneck = _model_bottleneck(model_result)
    runtime_bottleneck = _runtime_bottleneck(record)
    finding_56 = {
        "metric_id": "5.6",
        "analysis_type": "pipeline_bottleneck",
        "model_finding": {
            "status": "predicted" if model_bottleneck else "unavailable",
            "critical_resource": model_bottleneck,
            "synchronization": _synchronization_finding(model_result),
        },
        "runtime_finding": {
            **runtime_bottleneck,
            "synchronization": {
                "status": "unavailable",
                "reason": "当前runtime bundle没有同步等待时间",
            },
        },
        "joint_conclusion": (
            "agreement" if model_bottleneck
            and runtime_bottleneck.get("critical_resource") == model_bottleneck
            else "different_or_insufficient_runtime_evidence"
        ),
    }

    ddr = _path(model_result, "memory_hierarchy_prediction", "ddr") or {}
    logical_bytes = _number(model_result.get("logical_traffic_bytes"))
    model_total = (
        _number(ddr.get("kernel_scope_total_traffic_bytes"))
        if isinstance(ddr, Mapping) else None
    )
    runtime_read = metric_value(record, "ddr_read_bytes")
    runtime_write = metric_value(record, "ddr_write_bytes")
    runtime_total = metric_value(record, "ddr_total_bytes")
    if runtime_total is None and runtime_read is not None and runtime_write is not None:
        runtime_total = runtime_read + runtime_write
    finding_57 = {
        "metric_id": "5.7",
        "analysis_type": "cross_level_traffic",
        "model_finding": {
            "kernel_scope_ddr_bytes": model_total,
            "logical_traffic_bytes": logical_bytes,
        },
        "runtime_finding": {
            "ddr_read_bytes": runtime_read,
            "ddr_write_bytes": runtime_write,
            "ddr_total_bytes": runtime_total,
            "status": "observed" if runtime_total is not None else "unavailable",
        },
        "joint_conclusion": "definition_pending",
        "reason": (
            "逻辑流量、模型Kernel范围流量和MCU物理事务并列保留；"
            "在MCU聚合定义确认前不据差额自动判定异常。"
        ),
    }

    cache = model_result.get("cache_behavior", {})
    model_hit = _number(cache.get("l2_hit_rate_pct")) if isinstance(cache, Mapping) else None
    llc_read_hit = metric_value(record, "llc_read_sector_hit_rate_pct")
    l2pu_hit = metric_value(record, "l2pu_sector_hit_rate_pct")
    finding_58 = {
        "metric_id": "5.8",
        "analysis_type": "cache_utilization",
        "model_finding": {"logical_l2_hit_rate_pct": model_hit},
        "runtime_finding": {
            "llc_read_sector_hit_rate_pct": llc_read_hit,
            "l2pu_mixed_cacheline_hit_rate_pct": l2pu_hit,
            "status": "observed" if llc_read_hit is not None or l2pu_hit is not None
            else "unavailable",
        },
        "joint_conclusion": (
            "definition_pending_same_request_class_required"
            if llc_read_hit is not None else
            "not_directly_comparable" if l2pu_hit is not None else
            "runtime_unavailable"
        ),
    }

    pipeline = model_result.get("pipeline", {})
    stage_count = (
        _number(pipeline.get("stage_count"))
        if isinstance(pipeline, Mapping) else None
    )
    memory_per_iter = (
        _number(pipeline.get("memory_per_iter_us"))
        if isinstance(pipeline, Mapping) else None
    )
    compute_per_iter = (
        _number(pipeline.get("compute_per_iter_us"))
        if isinstance(pipeline, Mapping) else None
    )
    steady_per_iter = (
        _number(pipeline.get("steady_per_iter_us"))
        if isinstance(pipeline, Mapping) else None
    )
    theoretical_overlap = None
    if (
        stage_count is not None and stage_count >= 2
        and memory_per_iter is not None
        and compute_per_iter is not None
        and steady_per_iter is not None
    ):
        serial = memory_per_iter + compute_per_iter
        ideal = max(memory_per_iter, compute_per_iter)
        overlap_range = serial - ideal
        theoretical_overlap = (
            100.0 if overlap_range == 0 else
            max(
                0.0,
                min(
                    100.0,
                    (serial - steady_per_iter) / overlap_range * 100.0,
                ),
            )
        )
    overlap_not_applicable = stage_count is not None and stage_count < 2
    finding_59 = {
        "metric_id": "5.9",
        "analysis_type": "compute_io_overlap",
        "model_finding": {
            "value_kind": "model_estimate",
            "status": "not_applicable" if overlap_not_applicable else "predicted",
            "stage_count": stage_count,
            "memory_per_iter_us": memory_per_iter,
            "compute_per_iter_us": compute_per_iter,
            "steady_per_iter_us": steady_per_iter,
            "theoretical_overlap_efficiency_pct": theoretical_overlap,
        },
        "runtime_finding": {
            "status": "not_applicable" if overlap_not_applicable else "unavailable",
            "actual_overlap_efficiency_pct": None,
            "failed_tile_ops": [],
            "reason": (
                "单阶段Kernel没有多阶段计算–IO重叠失败"
                if overlap_not_applicable else
                "当前bundle没有Kernel内部逐阶段硬件重叠率"
            ),
        },
        "joint_conclusion": (
            "not_applicable_no_multistage_pipeline"
            if overlap_not_applicable else "model_only_not_runtime_confirmed"
        ),
    }

    model_waves = _number(model_result.get("waves_unrounded"))
    if model_waves is None:
        model_waves = _number(model_result.get("waves"))
    runtime_waves = metric_value(record, "waves_per_multiprocessor")
    wave_error = None
    if model_waves is not None and runtime_waves not in (None, 0):
        wave_error = round((model_waves - runtime_waves) / runtime_waves * 100, 6)
    finding_510 = {
        "metric_id": "5.10",
        "analysis_type": "load_distribution",
        "model_finding": _model_load_distribution(model_result),
        "runtime_finding": {
            "waves_per_multiprocessor": runtime_waves,
            "per_mp_cta_timeline": None,
            "per_mp_cta_timeline_status": "unavailable",
        },
        "joint_conclusion": (
            "waves_directly_comparable_per_mp_distribution_unavailable"
            if runtime_waves is not None else "runtime_unavailable"
        ),
        "waves_relative_error_pct": wave_error,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_scope": "tilesight_model_and_s5000_runtime",
        "runtime_data_included": True,
        "tilesight_prediction_modified": False,
        "runtime_record_id": record_id,
        "identity": comparison["identity"],
        "findings": [finding_56, finding_57, finding_58, finding_59, finding_510],
    }
