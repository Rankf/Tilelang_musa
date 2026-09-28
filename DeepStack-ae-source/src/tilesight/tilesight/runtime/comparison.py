"""Same-scope comparison of immutable TileSight and S5000 runtime results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .metrics import metric_semantics
from .schema import (
    load_s5000_runtime,
    metric_value,
    select_runtime_record,
    validate_workload_identity,
)


SCHEMA_VERSION = "tilesight/runtime-comparison-v1"


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


def _error(model_value: float | None, runtime_value: float | None) -> float | None:
    if model_value is None or runtime_value in (None, 0):
        return None
    return round((model_value - runtime_value) / runtime_value * 100.0, 6)


def _check(
    indicator: str, model_field: str | None, model_value: Any,
    runtime_metric: str | None, runtime_value: Any, unit: str,
    comparison_class: str, conclusion: str,
) -> dict[str, Any]:
    result = {
        "indicator": indicator,
        "model_field": model_field,
        "model_value": model_value,
        "runtime_metric": runtime_metric,
        "runtime_value": runtime_value,
        "unit": unit,
        "comparison_class": comparison_class,
        "conclusion": conclusion,
    }
    if comparison_class in {"directly_comparable", "derived_same_scope"}:
        relative = _error(_number(model_value), _number(runtime_value))
        if relative is not None:
            result["relative_error_pct"] = relative
    if runtime_metric:
        result["runtime_metric_semantics"] = metric_semantics(runtime_metric)
    return result


def compare_with_runtime(
    model_result: Mapping[str, Any],
    runtime: Mapping[str, Any] | str | Path,
    record_id: str,
    *,
    model_source: str | None = None,
    runtime_source: str | None = None,
    require_identity: bool = True,
    significant_error_threshold_pct: float = 10.0,
) -> dict[str, Any]:
    """Compare one model result with one runtime record without mutation."""
    bundle = load_s5000_runtime(runtime)
    record = select_runtime_record(bundle, record_id)
    if record.get("status") != "available":
        raise ValueError(f"runtime record {record_id!r} is not available")
    if record.get("correctness_status") != "pass":
        raise ValueError(
            f"runtime record {record_id!r} did not pass correctness"
        )
    identity = validate_workload_identity(model_result, bundle, record)
    if require_identity and not identity["comparison_eligible"]:
        raise ValueError(
            "model/runtime workload identity mismatch: "
            + str(identity["mismatches"] or identity["missing_fields"])
        )

    measured_latency = metric_value(record, "kernel_time_us")
    if measured_latency is None:
        raise ValueError("runtime record has no kernel_time_us")
    kernel_scope_latency = _number(model_result.get("kernel_scope_latency_us"))
    full_latency = _number(model_result.get("latency_us"))
    if kernel_scope_latency is None and full_latency is None:
        raise ValueError("TileSight result has no numeric latency")
    # Keep the upstream-compatible full logical output-writeback latency as
    # the formal prediction.  S5000's kernel-window estimate is diagnostic
    # only; selecting it here would make Exp/Log/Sqrt use a different scope
    # from the rest of Elementwise and from upstream TileSight.
    comparison_latency = full_latency if full_latency is not None else kernel_scope_latency

    checks = [_check(
        "Kernel执行时间",
        "latency_us",
        comparison_latency,
        "kernel_time_us",
        measured_latency,
        "us",
        "different_scope",
        "正式比较采用完整逻辑输出写回；实测为目标Kernel窗口，"
        "误差包含模型与写回完成边界差异。",
    )]
    if kernel_scope_latency is not None and full_latency is not None:
        checks.append(_check(
            "Kernel窗口写回诊断时延",
            "kernel_scope_latency_us",
            kernel_scope_latency,
            "kernel_time_us",
            measured_latency,
            "us",
            "diagnostic_only",
            "S5000扩展诊断字段，不替代正式的latency_us。",
        ))

    ddr = _path(model_result, "memory_hierarchy_prediction", "ddr") or {}
    model_ddr_read = _number(ddr.get("read_bytes")) if isinstance(ddr, Mapping) else None
    model_ddr_write = None
    model_ddr_total = None
    if isinstance(ddr, Mapping):
        model_ddr_write = _number(ddr.get("kernel_scope_write_bytes"))
        model_ddr_total = _number(ddr.get("kernel_scope_total_traffic_bytes"))
    runtime_read = metric_value(record, "ddr_read_bytes")
    runtime_write = metric_value(record, "ddr_write_bytes")
    runtime_total = metric_value(record, "ddr_total_bytes")
    if runtime_total is None and runtime_read is not None and runtime_write is not None:
        runtime_total = runtime_read + runtime_write
    for indicator, field, model_value, metric, runtime_value in (
        ("DDR读取字节", "memory_hierarchy_prediction.ddr.read_bytes",
         model_ddr_read, "ddr_read_bytes", runtime_read),
        ("DDR写入字节（Kernel窗口）",
         "memory_hierarchy_prediction.ddr.kernel_scope_write_bytes",
         model_ddr_write, "ddr_write_bytes", runtime_write),
        ("DDR总字节（Kernel窗口）",
         "memory_hierarchy_prediction.ddr.kernel_scope_total_traffic_bytes",
         model_ddr_total, "ddr_total_bytes", runtime_total),
    ):
        checks.append(_check(
            indicator, field, model_value, metric, runtime_value, "bytes",
            "definition_pending" if runtime_value is not None else "runtime_unavailable",
            "模型与MCU均标记为Kernel范围，但物理事务聚合定义仍需厂商规范确认。",
        ))

    model_ddr_util = _number(model_result.get("ddr_util_pct"))
    runtime_ddr_util = metric_value(record, "ddr_util_pct")
    checks.append(_check(
        "DDR利用率", "ddr_util_pct", model_ddr_util,
        "ddr_util_pct", runtime_ddr_util, "percent",
        "definition_pending" if runtime_ddr_util is not None else "runtime_unavailable",
        "双方峰值分母和采集窗口定义需审核后才能计算正式误差。",
    ))

    model_waves = _number(model_result.get("waves_unrounded"))
    if model_waves is None:
        model_waves = _number(model_result.get("waves"))
    runtime_waves = metric_value(record, "waves_per_multiprocessor")
    checks.append(_check(
        "Waves", "waves_unrounded/waves", model_waves,
        "waves_per_multiprocessor", runtime_waves, "waves/MP",
        "directly_comparable" if runtime_waves is not None else "runtime_unavailable",
        "两者均表示目标launch按MP折算的波数。",
    ))

    cache = model_result.get("cache_behavior", {})
    model_hit = _number(cache.get("l2_hit_rate_pct")) if isinstance(cache, Mapping) else None
    llc_read_hit = metric_value(record, "llc_read_sector_hit_rate_pct")
    l2pu_hit = metric_value(record, "l2pu_sector_hit_rate_pct")
    if llc_read_hit is not None:
        checks.append(_check(
            "LLC读取命中率", "cache_behavior.l2_hit_rate_pct", model_hit,
            "llc_read_sector_hit_rate_pct", llc_read_hit, "percent",
            "definition_pending",
            "运行时值为LLC读取sector命中率；只有模型也采用相同请求定义时才可计算误差。",
        ))
    else:
        checks.append(_check(
            "缓存命中率", "cache_behavior.l2_hit_rate_pct", model_hit,
            "l2pu_sector_hit_rate_pct" if l2pu_hit is not None else None,
            l2pu_hit, "percent",
            "not_directly_comparable" if l2pu_hit is not None else "runtime_unavailable",
            "L2PU指标是读写混合cacheline请求命中率，不等于模型逻辑复用率。",
        ))

    comparable = [
        check for check in checks
        if check.get("comparison_class") in {
            "directly_comparable", "derived_same_scope"
        }
        and isinstance(check.get("relative_error_pct"), (int, float))
    ]
    significant = []
    for check in comparable:
        relative_error = float(check["relative_error_pct"])
        if abs(relative_error) <= significant_error_threshold_pct:
            continue
        significant.append({
            "indicator": check["indicator"],
            "relative_error_pct": relative_error,
            "direction": (
                "model_overprediction"
                if relative_error > 0 else "model_underprediction"
            ),
            "uncaptured_hardware_behavior": True,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_scope": "model_runtime_join_without_prediction_mutation",
        "model_data_embedded_in_runtime": False,
        "runtime_data_embedded_in_model": False,
        "model_source": model_source,
        "runtime_source": runtime_source or (
            str(runtime) if isinstance(runtime, (str, Path)) else None
        ),
        "runtime_record_id": record_id,
        "identity": identity,
        "record": {
            "tool": record.get("tool"),
            "measurement_scope": record.get("measurement_scope"),
            "status": record.get("status"),
            "correctness_status": record.get("correctness_status"),
        },
        "checks": checks,
        "prediction_deviation": {
            "metric_id": "5.11",
            "significant_error_threshold_pct": significant_error_threshold_pct,
            "threshold_basis": "project_holdout_acceptance_threshold",
            "comparable_indicator_count": len(comparable),
            "significant_difference_count": len(significant),
            "status": (
                "significant_model_deviation"
                if significant else "within_threshold"
            ),
            "uncaptured_hardware_behavior": bool(significant),
            "significant_differences": significant,
        },
    }
