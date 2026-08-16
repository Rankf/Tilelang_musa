#!/usr/bin/env python3
"""S5000 算子性能分析工具

用法:
  python3 analyze_op.py --tile-metadata metadata.json --arch S5000 -o report.json
  python3 analyze_op.py relu --N 16777216 --arch S5000 -o report.json
"""
import os, sys, json, argparse, importlib, math
from pathlib import Path

os.environ.setdefault("MUSA_HOME", "/usr/local/musa")
BASE = "/root/s5000-calibration"
sys.path = [BASE, BASE+"/TileOPs/src", BASE+"/DeepStack-ae-source/src/tilesight",
            BASE+"/DeepStack-ae/src"] + sys.path

# ── S5000 架构（使用校准后的 ConfigurableStackedGpu 配置）──
from deepstack.mosaic.arch import make_s5000_config, make_custom_stacked_gpu


S5000_FP16_GEMM_MEMORY_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [1, 1, 1, 2],
    "out1": [1, 0, 1, 2],
}


def _load_tile_metadata(value):
    if isinstance(value, dict):
        metadata = value
        source = "python_object"
    else:
        path = Path(value).expanduser().resolve()
        metadata = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    if metadata.get("schema_version") != "tilesight/tile-metadata-v1":
        raise ValueError("tile metadata must use tilesight/tile-metadata-v1")
    kernels = metadata.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("tile metadata contains no kernels")
    return metadata, source


def _select_metadata_kernel(metadata, kernel_id=None):
    kernels = metadata["kernels"]
    if kernel_id is None:
        if len(kernels) != 1:
            raise ValueError(
                "tile metadata contains multiple kernels; --kernel-id is required"
            )
        return kernels[0]
    matches = [item for item in kernels if item.get("kernel_id") == kernel_id]
    if len(matches) != 1:
        raise ValueError(f"kernel_id {kernel_id!r} does not resolve uniquely")
    return matches[0]


def _one(items, description):
    if len(items) != 1:
        raise ValueError(
            f"expected exactly one {description}, found {len(items)}"
        )
    return items[0]


def _static_shape(buffer_record):
    try:
        shape = [int(value) for value in buffer_record["shape"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"buffer {buffer_record.get('name')!r} requires a static integer shape"
        ) from error
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(
            f"buffer {buffer_record.get('name')!r} must have a positive 2-D shape"
        )
    return shape


def derive_gemm_model_input(tile_metadata, kernel_id=None):
    """Derive a validated TileSight GEMM query from 5.1 metadata."""
    metadata, source = _load_tile_metadata(tile_metadata)
    kernel = _select_metadata_kernel(metadata, kernel_id)
    operations = kernel.get("operations", [])
    operation_by_id = {item.get("op_id"): item for item in operations}
    if len(operation_by_id) != len(operations) or None in operation_by_id:
        raise ValueError("tile metadata operations require unique non-null op_id values")

    gemm = _one(
        [item for item in operations if item.get("category") == "gemm"],
        "GEMM TileOp",
    )
    tile = gemm.get("tile_shape") or {}
    try:
        bm, bn, bk = (int(tile[key]) for key in ("m", "n", "k"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GEMM TileOp requires static m/n/k tile_shape") from error
    if min(bm, bn, bk) <= 0:
        raise ValueError("GEMM tile dimensions must be positive")

    shared_inputs = [
        item for item in gemm.get("reads", [])
        if item.get("scope") == "shared"
    ]
    if len(shared_inputs) < 2:
        raise ValueError("GEMM TileOp requires two ordered shared-memory inputs")

    def producer_for(shared_buffer):
        return _one([
            item for item in operations
            if item.get("category") == "copy"
            and any(write.get("buffer") == shared_buffer
                    and write.get("scope") == "shared"
                    for write in item.get("writes", []))
            and any(read.get("scope") == "global"
                    for read in item.get("reads", []))
        ], f"global-to-shared producer for {shared_buffer}")

    copy_a = producer_for(shared_inputs[0].get("buffer"))
    copy_b = producer_for(shared_inputs[1].get("buffer"))
    if copy_a["op_id"] == copy_b["op_id"]:
        raise ValueError("GEMM A and B inputs resolve to the same copy TileOp")
    read_a = _one(
        [item for item in copy_a.get("reads", []) if item.get("scope") == "global"],
        "global A read",
    )
    read_b = _one(
        [item for item in copy_b.get("reads", []) if item.get("scope") == "global"],
        "global B read",
    )

    output_copy = _one([
        item for item in operations
        if item.get("category") == "copy"
        and gemm["op_id"] in item.get("dependency_ids", [])
        and any(read.get("scope") == "local" for read in item.get("reads", []))
        and any(write.get("scope") == "global" for write in item.get("writes", []))
    ], "GEMM local-to-global output copy")
    write_c = _one(
        [item for item in output_copy.get("writes", []) if item.get("scope") == "global"],
        "global C write",
    )

    buffers = {item.get("name"): item for item in kernel.get("buffers", [])}
    try:
        a_shape = _static_shape(buffers[read_a["buffer"]])
        b_shape = _static_shape(buffers[read_b["buffer"]])
        c_shape = _static_shape(buffers[write_c["buffer"]])
    except KeyError as error:
        raise ValueError(f"operation references unknown buffer {error.args[0]!r}") from error
    M, K = a_shape
    K_b, N = b_shape
    if K != K_b or c_shape != [M, N]:
        raise ValueError(
            f"inconsistent GEMM buffer shapes A={a_shape}, B={b_shape}, C={c_shape}"
        )

    pipelines = kernel.get("pipeline", {}).get("pipelines", [])
    pipeline = _one([
        item for item in pipelines
        if gemm["op_id"] in item.get("operation_ids", [])
        and copy_a["op_id"] in item.get("operation_ids", [])
        and copy_b["op_id"] in item.get("operation_ids", [])
    ], "pipeline containing GEMM and both input copies")
    try:
        ns = int(pipeline["stage_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pipeline requires a static integer stage_count") from error
    threads = kernel.get("launch", {}).get("threads_per_block")
    if not isinstance(threads, int) or threads <= 0 or threads % 32:
        raise ValueError("threads_per_block must be a positive multiple of 32")
    nw = threads // 32

    dtype_a = buffers[read_a["buffer"]].get("dtype")
    dtype_b = buffers[read_b["buffer"]].get("dtype")
    dtype_c = buffers[write_c["buffer"]].get("dtype")
    if (dtype_a, dtype_b, dtype_c) != ("float16", "float16", "float16"):
        raise ValueError(
            "current metadata-driven GEMM path supports FP16 A/B/C only"
        )
    return {
        "model_input": {
            "M": M, "N": N, "K": K,
            "bm": bm, "bn": bn, "bk": bk,
            "nw": nw, "ns": ns,
        },
        "metadata_index": {
            "schema": "tilesight/model-input-index-v1",
            "input_source": "tile_metadata",
            "metadata_source": source,
            "kernel_id": kernel["kernel_id"],
            "pipeline_id": pipeline["pipeline_id"],
            "operation_ids": {
                "copy_a": copy_a["op_id"],
                "copy_b": copy_b["op_id"],
                "gemm": gemm["op_id"],
                "copy_c": output_copy["op_id"],
            },
            "buffer_roles": {
                "A": read_a["buffer"],
                "B": read_b["buffer"],
                "C": write_c["buffer"],
            },
            "derivation_evidence": {
                "shape": "compiler-observed global buffer shapes and TileOp dataflow",
                "tile": "GEMM TileOp tile_shape",
                "pipeline_stages": "pipeline compiler annotation",
                "warps_per_block": "threads_per_block / 32",
                "memory_path": "FP16 global-to-shared A/B and local-to-global C",
            },
        },
    }


def _round_us(seconds):
    return round(seconds * 1e6, 4)


def _resource_cost(name, utilization, total_latency, bandwidth, max_utilization):
    """Build an auditable kernel-level roofline cost for one resource.

    TileSight reports utilization against the physical peak.  Multiplying it
    by kernel latency recovers the peak-roofline service time and traffic.
    The calibrated service time additionally applies the measured sustained
    utilization ceiling used by the latency model.
    """
    peak_time = utilization * total_latency
    effective_time = peak_time / max_utilization if max_utilization > 0 else None
    traffic_bytes = peak_time * bandwidth if bandwidth > 0 else None
    return {
        "resource": name,
        "traffic_bytes": round(traffic_bytes) if traffic_bytes is not None else None,
        "peak_bandwidth_gb_s": round(bandwidth / 1e9, 3),
        "max_sustained_util_pct": round(max_utilization * 100, 3),
        "peak_roofline_cost_us": _round_us(peak_time),
        "calibrated_service_cost_us": (
            _round_us(effective_time) if effective_time is not None else None
        ),
        "value_kind": "model_estimate",
    }


def _service_cost_us(traffic_bytes, bandwidth, max_utilization):
    if not traffic_bytes or not bandwidth or not max_utilization:
        return 0.0
    return round(traffic_bytes / (bandwidth * max_utilization) * 1e6, 4)


def _predict_ordinary_l1_hit_rate(*, reuse_count, working_set_bytes_per_sm,
                                  arch):
    """Capacity/reuse approximation for ordinary Global-load traffic only."""
    if reuse_count <= 1:
        return 0.0, "single_use_stream"
    capacity = arch.l1_cache_capacity
    if capacity is None:
        return None, "l1_capacity_not_configured"
    if working_set_bytes_per_sm <= capacity:
        return (reuse_count - 1) / reuse_count, "capacity_and_reuse"
    return 0.0, "working_set_exceeds_effective_capacity"


def _memory_hierarchy_prediction(*, arch, ordinary_read_bytes,
                                 ordinary_write_bytes, tma_read_bytes,
                                 shared_read_bytes, shared_write_bytes,
                                 l2_hit_rate, ddr_write_bytes=None,
                                 minimum_useful_read_bytes=None,
                                 reuse_count=1,
                                 working_set_bytes_per_sm=0,
                                 cache_condition="cold"):
    """Predict read/write traffic using the measured S5000 path split."""
    if ordinary_read_bytes:
        l1_hit_rate, l1_method = _predict_ordinary_l1_hit_rate(
            reuse_count=reuse_count,
            working_set_bytes_per_sm=working_set_bytes_per_sm,
            arch=arch,
        )
        ordinary_l1_miss_bytes = ordinary_read_bytes * (1 - l1_hit_rate)
    else:
        l1_hit_rate, l1_method = None, "not_applicable_no_eligible_traffic"
        ordinary_l1_miss_bytes = 0

    bypass_bytes = tma_read_bytes if arch.tma_bypasses_ordinary_l1 else 0
    if tma_read_bytes and not arch.tma_bypasses_ordinary_l1:
        ordinary_l1_miss_bytes += tma_read_bytes
    l2_read_input_bytes = ordinary_l1_miss_bytes + bypass_bytes
    ddr_read_bytes = l2_read_input_bytes * (1 - l2_hit_rate)
    if ddr_write_bytes is None:
        ddr_write_bytes = ordinary_write_bytes
    l2_total_bytes = l2_read_input_bytes + ordinary_write_bytes
    ddr_total_bytes = ddr_read_bytes + ddr_write_bytes
    shared_total_bytes = shared_read_bytes + shared_write_bytes
    l1_cache_traffic = ordinary_read_bytes + ordinary_write_bytes
    if minimum_useful_read_bytes is None:
        minimum_useful_read_bytes = ordinary_read_bytes + tma_read_bytes
    service_costs = {
        "l1_cache": _service_cost_us(
            l1_cache_traffic, arch.l1_cache_bandwidth,
            arch.l1_cache_max_util,
        ),
        "l2": _service_cost_us(
            l2_total_bytes, arch.l2_bandwidth, arch.l2_max_util,
        ),
        "ddr": _service_cost_us(
            ddr_total_bytes, arch.ddr_bandwidth, arch.ddr_max_util,
        ),
        "shared_memory": _service_cost_us(
            shared_total_bytes, arch.smem_bandwidth, arch.l1_max_util,
        ),
    }
    critical_level = max(service_costs, key=service_costs.get)

    return {
        "status": "available",
        "value_kind": "model_estimate",
        "cache_condition": cache_condition,
        "minimum_useful_read_bytes": round(minimum_useful_read_bytes),
        "calibration": {
            "l1_effective_capacity_bytes_per_mp": arch.l1_cache_capacity,
            "l1_cache_line_bytes": arch.l1_cache_line_bytes,
            "l1_sector_bytes": arch.l1_cache_sector_bytes,
            "l1_peak_bandwidth_gb_s": round(arch.l1_cache_bandwidth / 1e9, 3),
            "l1_max_sustained_util_pct": round(arch.l1_cache_max_util * 100, 3),
            "tma_bypasses_ordinary_l1": arch.tma_bypasses_ordinary_l1,
            "associativity": "not_fixed; conflict onset measured but address hashing is unresolved",
        },
        "l1_cache": {
            "eligible_read_bytes": round(ordinary_read_bytes),
            "bypass_read_bytes": round(bypass_bytes),
            "bypass_value_kind": "logical_tile_request_bytes",
            "predicted_hit_rate_pct": (
                round(l1_hit_rate * 100, 3) if l1_hit_rate is not None
                else "not_applicable"
            ),
            "prediction_method": l1_method,
            "downstream_read_bytes": round(ordinary_l1_miss_bytes),
            "ordinary_write_bytes": round(ordinary_write_bytes),
            "service_cost_us": service_costs["l1_cache"],
        },
        "l2": {
            "input_read_bytes": round(l2_read_input_bytes),
            "input_read_value_kind": "logical_cache_request_bytes",
            "predicted_hit_rate_pct": round(l2_hit_rate * 100, 3),
            "write_bytes": round(ordinary_write_bytes),
            "total_traffic_bytes": round(l2_total_bytes),
            "service_cost_us": service_costs["l2"],
        },
        "ddr": {
            "read_bytes": round(ddr_read_bytes),
            "write_bytes": round(ddr_write_bytes),
            "total_traffic_bytes": round(ddr_total_bytes),
            "service_cost_us": service_costs["ddr"],
        },
        "shared_memory": {
            "global_to_shared_bytes": round(tma_read_bytes),
            "global_to_shared_value_kind": "logical_tile_transfer_bytes",
            "read_bytes": round(shared_read_bytes),
            "write_bytes": round(shared_write_bytes),
            "total_traffic_bytes": round(shared_total_bytes),
            "hit_rate": "not_applicable",
            "service_cost_us": service_costs["shared_memory"],
        },
        "critical_memory_path": {
            "bottleneck": critical_level,
            "service_cost_us": service_costs[critical_level],
            "overlap_rule": "max(L1 Cache, L2, DDR, Shared service cost)",
        },
        "formula": {
            "l2_read_input": "tma_bypass + ordinary_read * (1 - l1_hit_rate)",
            "ddr_read": "l2_read_input * (1 - l2_hit_rate)",
            "ddr_total": "ddr_read + writeback_reaching_ddr",
        },
    }


def _elementwise_l2_hit_rate(read_working_set_bytes, arch, cache_condition):
    if cache_condition == "cold":
        return 0.0, "cold_start", None
    if read_working_set_bytes <= arch.l2_capacity:
        # P9 warm elementwise validation: ReLU 1M=99.7317%, Add 1M=99.8065%
        # by downstream-vs-DDR bytes.  Use their rounded midpoint rather than
        # claiming an impossible exact 100% prediction.
        return 0.9975, "calibrated_warm_working_set_fits_l2", [99.0, 100.0]
    return 0.0, "warm_working_set_exceeds_l2_capacity", None


def _elementwise_ddr_write_bytes(read_bytes, write_bytes, arch,
                                 cache_condition):
    # The operator report covers this kernel's execution.  An input+output
    # working set that fits L2 can remain resident; later eviction/writeback is
    # outside the operator's modeled execution interval.  This write-back
    # behavior applies to both cold and warm input states.
    if cache_condition not in {"cold", "warm"}:
        raise ValueError("cache_condition must be either 'cold' or 'warm'")
    if read_bytes + write_bytes <= arch.l2_capacity:
        return (
            0,
            write_bytes,
            "dirty_output_resident_in_l2_during_kernel",
        )
    return write_bytes, 0, "writeback_reaches_ddr_in_kernel_scope"


def _transfer_anomaly_analysis(hierarchy, latency_us):
    """Structured 5.7 cross-level transfer checks from model outputs only."""
    findings = []

    def add(rule_id, path, status, severity, predicted, reference, threshold,
            reason):
        findings.append({
            "rule_id": rule_id,
            "path": path,
            "status": status,
            "severity": severity,
            "predicted": predicted,
            "model_reference": reference,
            "threshold": threshold,
            "reason": reason,
            "value_kind": "model_estimate",
        })

    l1 = hierarchy["l1_cache"]
    l2 = hierarchy["l2"]
    ddr = hierarchy["ddr"]
    minimum_useful = hierarchy["minimum_useful_read_bytes"]
    eligible = l1["eligible_read_bytes"]
    downstream = l1["downstream_read_bytes"]
    if eligible:
        ratio = downstream / eligible
        abnormal = ratio > 1.05
        add(
            "excess_l1_downstream_read",
            "ordinary_global->l1->l2",
            "anomaly" if abnormal else "normal",
            "error" if abnormal else "info",
            {"downstream_over_eligible": round(ratio, 6)},
            {"maximum_ratio": 1.0},
            {"anomaly_if_ratio_gt": 1.05},
            "L1 downstream read exceeds eligible logical read by more than transaction tolerance"
            if abnormal else "L1 downstream traffic is consistent with eligible ordinary-load traffic",
        )
    else:
        add(
            "excess_l1_downstream_read", "ordinary_global->l1->l2",
            "not_applicable", "info", None, None, None,
            "no ordinary L1-eligible read traffic",
        )

    expected_l2_input = downstream + l1["bypass_read_bytes"]
    l2_residual = l2["input_read_bytes"] - expected_l2_input
    l2_abnormal = abs(l2_residual) > max(expected_l2_input * 0.05, 64)
    add(
        "cross_level_read_conservation",
        "l1/tma->l2",
        "anomaly" if l2_abnormal else "normal",
        "error" if l2_abnormal else "info",
        {"l2_input_bytes": l2["input_read_bytes"], "residual_bytes": l2_residual},
        {"l1_downstream_plus_bypass_bytes": expected_l2_input},
        {"absolute_or_relative_tolerance": "max(64 B, 5%)"},
        "L2 input is inconsistent with L1 downstream plus bypass traffic"
        if l2_abnormal else "cross-level read traffic is conserved by the model",
    )

    ddr_ratio = ddr["read_bytes"] / l2["input_read_bytes"] if l2["input_read_bytes"] else 0
    ddr_abnormal = ddr_ratio > 1.05
    add(
        "excess_ddr_read",
        "l2->ddr",
        "anomaly" if ddr_abnormal else "normal",
        "error" if ddr_abnormal else "info",
        {"ddr_read_over_l2_input": round(ddr_ratio, 6)},
        {"maximum_ratio": 1.0},
        {"anomaly_if_ratio_gt": 1.05},
        "DDR read amplification exceeds L2 input tolerance"
        if ddr_abnormal else "DDR read does not exceed predicted L2 input",
    )

    logical_source_read = eligible + l1["bypass_read_bytes"]
    amplification = logical_source_read / minimum_useful if minimum_useful else 0
    if amplification > 64:
        status, severity = "anomaly", "error"
    elif amplification > 4:
        status, severity = "warning", "warning"
    else:
        status, severity = "normal", "info"
    add(
        "logical_transfer_amplification",
        "global/tma->cache/shared",
        status, severity,
        {
            "logical_source_read_bytes": logical_source_read,
            "minimum_useful_read_bytes": minimum_useful,
            "amplification_ratio": round(amplification, 6),
        },
        {"ideal_amplification_ratio": 1.0},
        {"warning_if_gt": 4.0, "anomaly_if_gt": 64.0},
        "Tile schedule creates high repeated cross-level read traffic"
        if status != "normal" else
        "logical cross-level read amplification is within the model threshold",
    )

    for level in ("l1_cache", "l2", "ddr", "shared_memory"):
        cost = hierarchy[level]["service_cost_us"]
        pressure = cost / latency_us if latency_us > 0 else 0
        if pressure > 1.05:
            status, severity = "anomaly", "error"
        elif pressure >= 0.85:
            status, severity = "warning", "warning"
        else:
            status, severity = "normal", "info"
        add(
            "transfer_service_pressure",
            level,
            status,
            severity,
            {"service_cost_us": cost, "pressure_pct": round(pressure * 100, 3)},
            {"report_latency_us": latency_us},
            {"warning_pct": 85.0, "anomaly_pct": 105.0},
            "transfer path is close to the calibrated sustained-service limit"
            if status == "warning" else (
                "transfer service demand exceeds report latency beyond tolerance"
                if status == "anomaly" else
                "transfer service pressure is below the warning threshold"
            ),
        )

    anomalies = [item for item in findings if item["status"] == "anomaly"]
    warnings = [item for item in findings if item["status"] == "warning"]
    return {
        "status": "available",
        "overall_status": (
            "anomaly" if anomalies else "warning" if warnings else "normal"
        ),
        "analysis_scope": "tile_pipeline_and_hardware_model",
        "input_dependency": "operator shape/tile/pipeline and S5000 profile only",
        "summary": {
            "anomaly_count": len(anomalies),
            "warning_count": len(warnings),
            "normal_count": sum(item["status"] == "normal" for item in findings),
        },
        "findings": findings,
    }


def _backfill_elementwise_hierarchy_fields(report, arch):
    """Keep legacy top-level fields consistent with the hierarchy model."""
    hierarchy = report["memory_hierarchy_prediction"]
    latency_s = report["latency_with_memory_hierarchy_us"] / 1e6
    report["l2_hit_rate_pct"] = hierarchy["l2"]["predicted_hit_rate_pct"]
    report["l2_util_pct"] = round(
        hierarchy["l2"]["total_traffic_bytes"]
        / arch.l2_bandwidth / latency_s * 100,
        3,
    ) if latency_s else 0.0
    report["ddr_util_pct"] = round(
        hierarchy["ddr"]["total_traffic_bytes"]
        / arch.ddr_bandwidth / latency_s * 100,
        3,
    ) if latency_s else 0.0
    report["smem_util_pct"] = round(
        hierarchy["shared_memory"]["total_traffic_bytes"]
        / arch.smem_bandwidth / latency_s * 100,
        3,
    ) if latency_s else 0.0
    report["memory_field_source"] = (
        "S5000 memory_hierarchy_prediction; utilization uses physical peak "
        "bandwidth and latency_with_memory_hierarchy_us"
    )


def _gemm_cost_model(r, arch):
    """Expose the 5.2 cost model already used internally by TileSight."""
    detail = r.pipeline_detail
    if detail is None:
        return {
            "status": "not_available",
            "reason": "TileSight did not return pipeline_detail",
        }

    memory_levels = {
        "ddr": _resource_cost(
            "ddr", r.ddr_util, r.total_latency, arch.ddr_bandwidth,
            arch.ddr_max_util,
        ),
        "l2": _resource_cost(
            "l2", r.l2_util, r.total_latency, arch.l2_bandwidth,
            arch.l2_max_util,
        ),
        # TileSight's smem resource represents explicit Shared Memory traffic,
        # not an L1 cache hit-rate prediction.
        "shared_memory": _resource_cost(
            "shared_memory", r.smem_util, r.total_latency,
            arch.smem_bandwidth, arch.l1_max_util,
        ),
    }
    dominant_memory_level = max(
        memory_levels,
        key=lambda key: memory_levels[key]["calibrated_service_cost_us"],
    )

    memory_us = _round_us(detail.mem_time_per_iter)
    compute_us = _round_us(detail.compute_time_per_iter)
    steady_us = _round_us(detail.steady_time_per_iter)
    if compute_us >= memory_us:
        bottleneck = "compute"
        reason = (
            f"steady-state compute cost {compute_us:.4f} us/iter is not less "
            f"than memory cost {memory_us:.4f} us/iter"
        )
    else:
        bottleneck = dominant_memory_level
        reason = (
            f"steady-state memory cost {memory_us:.4f} us/iter exceeds compute "
            f"cost {compute_us:.4f} us/iter; the largest kernel-level calibrated "
            f"memory service cost is {dominant_memory_level}"
        )

    return {
        "status": "available",
        "value_kind": "model_estimate",
        "pipeline_iteration_cost_us": {
            "memory": memory_us,
            "compute": compute_us,
            "steady_state_after_overlap": steady_us,
            "overlap_rule": "max(memory, compute) for pipeline stage >= 2",
        },
        "memory_level_cost": memory_levels,
        "critical_path": {
            "scope": "steady_state_pipeline_iteration",
            "bottleneck": bottleneck,
            "dominant_memory_level": dominant_memory_level,
            "cost_us_per_iter": steady_us,
            "reason": reason,
        },
    }


def _derive_warp_shape(bm, bn, bk, nw):
    """Map a requested warp count to a valid, near-square warp tile.

    TileSight derives warps/block as ``(bm / wm) * (bn / wn)``.  Therefore
    dividing both tile dimensions by ``nw`` would accidentally create
    ``nw**2`` warps.  Select two integer split factors whose product is nw.
    """
    if nw <= 0:
        raise ValueError("nw must be a positive integer")

    candidates = []
    for split_n in range(1, nw + 1):
        if nw % split_n:
            continue
        split_m = nw // split_n
        if bm % split_m == 0 and bn % split_n == 0:
            wm, wn = bm // split_m, bn // split_n
            # Prefer a balanced warp tile.  On ties, split M more than N,
            # matching the GEMM tile convention used by the S5000 kernels.
            aspect_error = abs(math.log(wm / wn))
            candidates.append((aspect_error, -split_m, wm, wn))

    if not candidates:
        raise ValueError(
            f"cannot map nw={nw} onto tile ({bm}, {bn}): "
            "nw must factor into divisors of bm and bn"
        )

    _, _, wm, wn = min(candidates)
    return (wm, wn, bk)


def _gemm_tile_cache_predictions(r, M, N, K, bm, bn, bk, mem_levels, arch):
    """Format cache-model operand detail without changing model results."""
    cache_detail = r.cache_detail
    if not cache_detail or not cache_detail.get("per_tensor"):
        return {
            "status": "not_available",
            "reason": "cache model did not return per-operand detail",
        }

    grid_m = math.ceil(M / bm)
    grid_n = math.ceil(N / bn)
    grid_k = math.ceil(K / bk)
    specs = (
        ("in1", "A", [bm, bk], bm * bk * mem_levels["in1"][-1]),
        ("in2", "B", [bk, bn], bk * bn * mem_levels["in2"][-1]),
    )
    operands = []
    for key, operand, tile_shape, tile_bytes in specs:
        predicted = cache_detail["per_tensor"].get(
            key, cache_detail["per_tensor"].get(operand)
        )
        if predicted is None:
            continue
        request_bytes = grid_m * grid_n * grid_k * tile_bytes
        hit_rate = predicted["l2_hit_rate"]
        hit_bytes = request_bytes * hit_rate
        miss_bytes = request_bytes - hit_bytes
        operands.append({
            "operand": operand,
            "model_tensor": key,
            "operation": "copy",
            "source_level": "global",
            "destination_level": "shared",
            "tile_shape": tile_shape,
            "tile_request_bytes_per_k_iter": round(tile_bytes),
            "k_iterations": grid_k,
            "logical_request_bytes": round(request_bytes),
            "ordinary_l1": {
                "status": (
                    "bypass" if arch.tma_bypasses_ordinary_l1 else
                    "not_separately_modeled"
                ),
                "predicted_hit_rate_pct": "not_applicable",
            },
            "l2": {
                "predicted_hit_rate_pct": round(hit_rate * 100, 6),
                "request_bytes": round(request_bytes),
                "predicted_hit_bytes": round(hit_bytes),
                "predicted_miss_bytes": round(miss_bytes),
            },
            "ddr": {"predicted_read_bytes": round(miss_bytes)},
            "shared_memory": {
                "write_bytes": round(request_bytes),
                "hit_rate": "not_applicable",
            },
            "raw_cache_observation": predicted,
        })

    total_request = sum(item["l2"]["request_bytes"] for item in operands)
    total_hit = sum(item["l2"]["predicted_hit_bytes"] for item in operands)
    total_miss = sum(item["l2"]["predicted_miss_bytes"] for item in operands)
    weighted_hit_rate = total_hit / total_request if total_request else 0.0
    return {
        "status": "available",
        "scope": "gemm_tile_operand",
        "value_kind": "model_estimate",
        "calculation_change": "none; formats existing cache-model accumulators",
        "aggregate": {
            "l2_hit_rate_pct": round(cache_detail["l2_hit_rate"] * 100, 6),
            "weighted_operand_l2_hit_rate_pct": round(
                weighted_hit_rate * 100, 6
            ),
            "l2_request_bytes": total_request,
            "predicted_l2_hit_bytes": total_hit,
            "predicted_l2_miss_bytes": total_miss,
            "predicted_ddr_read_bytes": total_miss,
            "global_to_shared_bytes": total_request,
        },
        "operands": operands,
    }


def _attach_metadata_index(report, metadata_index):
    """Attach compiler identities to an already computed GEMM report."""
    ids = metadata_index["operation_ids"]
    input_source = metadata_index.get("input_source", "tile_metadata")
    report["input_source"] = input_source
    index_key = (
        "compiler_info_index"
        if input_source == "compiler_info"
        else "metadata_index"
    )
    report[index_key] = metadata_index
    report["kernel_id"] = metadata_index["kernel_id"]
    report["pipeline_id"] = metadata_index["pipeline_id"]

    cache = report.get("tile_cache_predictions", {})
    role_by_operand = {"A": "copy_a", "B": "copy_b"}
    for item in cache.get("operands", []):
        role = role_by_operand.get(item.get("operand"))
        if role:
            item["op_id"] = ids[role]
            item["pipeline_id"] = metadata_index["pipeline_id"]

    cost = report.get("cost_model", {})
    critical = cost.get("critical_path", {})
    if critical.get("bottleneck") == "compute":
        critical_ops = [{
            "op_id": ids["gemm"],
            "category": "gemm",
            "role": "compute",
            "modeled_cost_us_per_iter": cost.get(
                "pipeline_iteration_cost_us", {}).get("compute"),
            "value_kind": "model_estimate",
        }]
    else:
        by_operand = {
            item.get("operand"): item for item in cache.get("operands", [])
        }
        critical_ops = []
        for operand, role in (("A", "copy_a"), ("B", "copy_b")):
            item = by_operand.get(operand, {})
            critical_ops.append({
                "op_id": ids[role],
                "category": "copy",
                "operand": operand,
                "role": "memory_path_member",
                "path": "global->l2->shared",
                "predicted_ddr_read_bytes": item.get("ddr", {}).get(
                    "predicted_read_bytes"),
                "value_kind": "model_estimate",
                "cost_attribution": (
                    "member of aggregate memory path; no exclusive per-op time"
                ),
            })
    critical["pipeline_id"] = metadata_index["pipeline_id"]
    critical["tile_ops"] = critical_ops
    report["tile_op_model_mapping"] = {
        "status": "complete",
        "association_mode": (
            "compiler_exact_lineage"
            if input_source == "compiler_info"
            else "compiler_metadata_structural_dataflow"
        ),
        "pipeline_id": metadata_index["pipeline_id"],
        "operations": [
            {"model_role": role, "op_id": op_id}
            for role, op_id in ids.items()
        ],
    }
    return report


def _analyze_gemm_model(M, N, K, bm, bn, bk, nw, ns, arch,
                        metadata_index, *, model_result=None,
                        mem_levels=None, warp_shape=None,
                        memory_path_kind=None):
    """Format a validated native TileSight GEMM result as a tool report."""
    if mem_levels is None:
        mem_levels = S5000_FP16_GEMM_MEMORY_LEVELS
    if warp_shape is None:
        warp_shape = _derive_warp_shape(bm, bn, bk, nw)
    if model_result is None:
        from tilesight.fused_op_pipeline_wave.matmul_pipeline_wave import (
            calculate_matmul_pipeline_wave,
        )
        model_result = calculate_matmul_pipeline_wave(
            (M, N, K), (bm, bn, bk), warp_shape, ns, arch, mem_levels,
            mma_type="wmma",
        )
    r = model_result
    flop_count = 2 * M * N * K
    report = {
        "op": "gemm", "shape": [M,N,K], "tile": [bm,bn,bk],
        "warps_per_block": nw, "warp_tile": list(warp_shape), "stage": ns,
        "memory_path_kind": (
            memory_path_kind
            or "s5000_fp16_global_shared_from_tile_metadata"
        ),
        "memory_levels": mem_levels,
        "latency_us": round(r.total_latency*1e6, 2),
        "flop_count": flop_count,
        "compute_throughput_tflops": round(flop_count/r.total_latency/1e12, 2),
        "ddr_util_pct": round(r.ddr_util*100, 1),
        "l2_util_pct": round(r.l2_util*100, 1),
        "smem_util_pct": round(r.smem_util*100, 1),
        "compute_util_pct": round(r.compute_util*100, 1),
        "l2_hit_rate_pct": round(r.l2_hit_rate*100, 1),
        "tiles_per_sm": r.tiles_per_sm,
        "waves": round(r.waves, 1),
        "pipeline": {
            "prologue_us": round(r.pipeline_detail.prologue_time*1e6, 2) if r.pipeline_detail else 0,
            "steady_per_iter_us": round(r.pipeline_detail.steady_time_per_iter*1e6, 2) if r.pipeline_detail else 0,
            "epilogue_us": round(r.pipeline_detail.epilogue_time*1e6, 2) if r.pipeline_detail else 0,
            "memory_per_iter_us": _round_us(r.pipeline_detail.mem_time_per_iter) if r.pipeline_detail else None,
            "compute_per_iter_us": _round_us(r.pipeline_detail.compute_time_per_iter) if r.pipeline_detail else None,
        }
    }
    report["cost_model"] = _gemm_cost_model(r, arch)
    grid_m = math.ceil(M / bm)
    grid_n = math.ceil(N / bn)
    grid_k = math.ceil(K / bk)
    tma_read_bytes = (
        grid_m * grid_n * grid_k * (bm * bk + bk * bn) * 2
    )
    shared_total_bytes = report["cost_model"]["memory_level_cost"][
        "shared_memory"
    ]["traffic_bytes"]
    report["memory_hierarchy_prediction"] = _memory_hierarchy_prediction(
        arch=arch,
        ordinary_read_bytes=0,
        ordinary_write_bytes=M * N * 2,
        tma_read_bytes=tma_read_bytes,
        shared_read_bytes=max(shared_total_bytes - tma_read_bytes, 0),
        shared_write_bytes=tma_read_bytes,
        l2_hit_rate=r.l2_hit_rate,
        minimum_useful_read_bytes=(M * K + K * N) * 2,
        cache_condition="intra_kernel_reuse",
    )
    report["tile_cache_predictions"] = _gemm_tile_cache_predictions(
        r, M, N, K, bm, bn, bk, mem_levels, arch
    )
    report["latency_with_memory_hierarchy_us"] = round(max(
        report["latency_us"],
        report["memory_hierarchy_prediction"]["critical_memory_path"][
            "service_cost_us"
        ],
    ), 2)
    report["transfer_anomaly_analysis"] = _transfer_anomaly_analysis(
        report["memory_hierarchy_prediction"],
        report["latency_with_memory_hierarchy_us"],
    )
    _attach_metadata_index(report, metadata_index)
    return report


def analyze_gemm_from_metadata(tile_metadata, arch=None, kernel_id=None):
    """Run the TileSight GEMM model directly from a validated 5.1 artifact."""
    if arch is None:
        arch = make_custom_stacked_gpu(make_s5000_config())
    derived = derive_gemm_model_input(tile_metadata, kernel_id=kernel_id)
    values = derived["model_input"]
    return _analyze_gemm_model(
        values["M"], values["N"], values["K"],
        values["bm"], values["bn"], values["bk"],
        values["nw"], values["ns"], arch,
        metadata_index=derived["metadata_index"],
    )


def _analyze_gemm_compiler_request(request, model_result, contract_result, arch):
    """Build the existing presentation report from one prepared GEMM request."""
    arguments = request["model"]["arguments"]
    M, N, K = arguments["op_shape"]
    bm, bn, bk = arguments["tb_shape"]
    identity = request["identity"]
    model_index = {
        "schema": "tilesight/model-input-index-v1",
        "input_source": "compiler_info",
        "compiler_info_source": request["input_source"],
        "compilation_id": identity["compilation_id"],
        "kernel_id": identity["kernel_id"],
        "pipeline_id": identity["pipeline_id"],
        "planned_pipeline_ids": identity["planned_pipeline_ids"],
        "operation_ids": identity["operation_ids"],
    }
    report = _analyze_gemm_model(
        M, N, K, bm, bn, bk,
        request["compiler_facts"]["warps_per_block"],
        arguments["stage_num"], arch,
        metadata_index=model_index,
        model_result=model_result,
        mem_levels=arguments["mem_levels"],
        warp_shape=tuple(arguments["wp_shape"]),
        memory_path_kind="s5000_fp16_global_shared_from_compiler_info",
    )
    # Keep the formal, lossless contract beside the presentation-oriented
    # report.  Compiler facts, adapter assumptions, and predictions therefore
    # remain independently auditable after the report transformation.
    report["model_request"] = request
    report["model_result"] = contract_result
    return report


def _build_elementwise_report(
    op, N, threads, elements_per_block, arch, r, cache_condition
):
    """Format TileSight output; all hierarchy quantities remain model-derived."""
    input_count = 1 if op == "relu" else 2
    read_bytes = N * 4 * input_count
    write_bytes = N * 4
    logical_traffic_bytes = read_bytes + write_bytes
    cache_detail = getattr(r, "cache_detail", None)
    if cache_detail is not None:
        expected = {
            "cache_condition": cache_condition,
            "read_working_set_bytes": read_bytes,
            "logical_output_write_bytes": write_bytes,
        }
        inconsistent = {
            field: {"tilesight": cache_detail.get(field), "report": value}
            for field, value in expected.items()
            if cache_detail.get(field) != value
        }
        if inconsistent:
            raise ValueError(
                "TileSight elementwise cache detail is inconsistent with the "
                f"report request: {inconsistent}"
            )
        predicted_l2_hit = cache_detail["l2_hit_rate"]
        l2_method = cache_detail["l2_prediction_method"]
        l2_range = cache_detail["l2_prediction_range_pct"]
        predicted_ddr_write = cache_detail["kernel_scope_ddr_write_bytes"]
        deferred_writeback = cache_detail["deferred_writeback_bytes"]
        write_method = cache_detail["ddr_write_prediction_method"]
    else:
        # Backward-compatible fallback for an older TileSight installation.
        predicted_l2_hit, l2_method, l2_range = _elementwise_l2_hit_rate(
            read_bytes, arch, cache_condition
        )
        (predicted_ddr_write, deferred_writeback,
         write_method) = _elementwise_ddr_write_bytes(
            read_bytes, write_bytes, arch, cache_condition
        )
    report = {
        "op": op, "N": N, "threads": threads,
        "elements_per_block": elements_per_block,
        "elements_per_thread": elements_per_block // threads,
        "latency_us": round(r.total_latency*1e6, 2),
        "logical_traffic_bytes": logical_traffic_bytes,
        "effective_bandwidth_gb_s": round(logical_traffic_bytes/r.total_latency/1e9, 1),
        "ddr_util_pct": round(r.ddr_util*100, 1),
        "l2_util_pct": round(r.l2_util*100, 1),
        "smem_util_pct": round(r.smem_util*100, 1),
        "compute_util_pct": round(r.compute_util*100, 1),
        "l2_hit_rate_pct": round(r.l2_hit_rate*100, 1),
        "tiles_per_sm": r.tiles_per_sm,
        "waves": round(r.waves, 1),
    }
    report["memory_hierarchy_prediction"] = _memory_hierarchy_prediction(
        arch=arch, ordinary_read_bytes=read_bytes, ordinary_write_bytes=write_bytes,
        tma_read_bytes=0, shared_read_bytes=0, shared_write_bytes=0,
        l2_hit_rate=predicted_l2_hit, ddr_write_bytes=predicted_ddr_write,
        reuse_count=1,
        working_set_bytes_per_sm=math.ceil(read_bytes / arch.sm_count),
        cache_condition=cache_condition,
    )
    report["memory_hierarchy_prediction"]["l2"]["prediction_method"] = l2_method
    if l2_range is not None:
        report["memory_hierarchy_prediction"]["l2"][
            "calibrated_prediction_range_pct"
        ] = l2_range
    report["memory_hierarchy_prediction"]["ddr"]["write_prediction_method"] = write_method
    report["memory_hierarchy_prediction"]["ddr"].update({
        "logical_output_write_bytes": round(write_bytes),
        "kernel_scope_write_bytes": round(predicted_ddr_write),
        "deferred_writeback_bytes": round(deferred_writeback),
        "measurement_scope": "modeled_kernel_execution_window",
    })
    report["latency_with_memory_hierarchy_us"] = round(max(
        report["latency_us"],
        report["memory_hierarchy_prediction"]["critical_memory_path"][
            "service_cost_us"
        ],
    ), 2)
    _backfill_elementwise_hierarchy_fields(report, arch)
    report["transfer_anomaly_analysis"] = _transfer_anomaly_analysis(
        report["memory_hierarchy_prediction"],
        report["latency_with_memory_hierarchy_us"],
    )
    return report


def _analyze_elementwise_compiler_request(
    request, model_result, contract_result, arch
):
    arguments = request["model"]["arguments"]
    model_name = request["model"]["name"]
    op = "relu" if model_name == "elementwise_n_0_pipeline_wave" else "add"
    report = _build_elementwise_report(
        op,
        arguments["op_shape"][0],
        arguments["dim_threads"][0],
        arguments["tb_shape"][0],
        arch,
        model_result,
        request["model_scenario"]["cache_condition"],
    )
    identity = request["identity"]
    report.update({
        "input_source": "compiler_info",
        "compiler_info_source": request["input_source"],
        "compilation_id": identity["compilation_id"],
        "kernel_id": identity["kernel_id"],
        "pipeline_id": None,
        "operation_ids": identity["operation_ids"],
        "memory_path_kind": "s5000_fp32_global_elementwise_from_compiler_info",
        "memory_levels": arguments["mem_levels"],
        "model_request": request,
        "model_result": contract_result,
    })
    return report


def analyze_from_compiler_info(
    compiler_info, arch=None, kernel_id=None, cache_condition="cold"
):
    """Dispatch compiler facts to GEMM, Add, or ReLU without changing TileSight."""
    from tilelang_compiler_info_adapter import (
        build_tilesight_result,
        invoke_tilesight_model,
        prepare_tilesight_request,
    )

    if arch is None:
        arch = make_custom_stacked_gpu(make_s5000_config())
    request = prepare_tilesight_request(
        compiler_info,
        kernel_id=kernel_id,
        cache_condition=cache_condition,
    )
    model_result = invoke_tilesight_model(request, arch)
    contract_result = build_tilesight_result(request, model_result, arch)
    if request["model"]["name"] == "matmul_pipeline_wave":
        return _analyze_gemm_compiler_request(
            request, model_result, contract_result, arch
        )
    return _analyze_elementwise_compiler_request(
        request, model_result, contract_result, arch
    )


def analyze_gemm_from_compiler_info(compiler_info, arch=None, kernel_id=None):
    """Backward-compatible GEMM-only compiler-info analysis entry."""
    report = analyze_from_compiler_info(
        compiler_info, arch=arch, kernel_id=kernel_id, cache_condition="cold"
    )
    if report["op"] != "gemm":
        raise ValueError("analyze_gemm_from_compiler_info requires a GEMM artifact")
    return report


def analyze_relu(N, threads, arch, cache_condition="cold"):
    from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import (
        calculate_N_0_elementwise_pipeline_wave,
    )
    mem_levels = {"in1": [1,0,0,4], "out1": [1,0,1,4]}
    elements_per_block = threads * 4
    r = calculate_N_0_elementwise_pipeline_wave(
        (N,), (elements_per_block,), (threads,), arch, mem_levels,
        cache_condition=cache_condition,
    )
    return _build_elementwise_report(
        "relu", N, threads, elements_per_block, arch, r, cache_condition
    )


def analyze_add(N, threads, arch, cache_condition="cold"):
    from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import (
        calculate_N_N_elementwise_pipeline_wave,
    )
    mem_levels = {"in1": [1,0,0,4], "in2": [1,0,0,4], "out1": [1,0,1,4]}
    elements_per_block = threads * 4
    r = calculate_N_N_elementwise_pipeline_wave(
        (N,), (elements_per_block,), (threads,), arch, mem_levels,
        cache_condition=cache_condition,
    )
    return _build_elementwise_report(
        "add", N, threads, elements_per_block, arch, r, cache_condition
    )


def _ranked(items, value_key):
    """Sort report rows descending and attach stable one-based ranks."""
    ordered = sorted(items, key=lambda item: item.get(value_key) or 0,
                     reverse=True)
    return [{"rank": rank, **item} for rank, item in enumerate(ordered, 1)]


def build_hierarchical_report(report):
    """Build the 5.5 report view from existing model output only.

    This is a presentation transform.  It does not invoke or modify a
    performance, cache, traffic, or latency calculation.
    """
    hierarchy = report.get("memory_hierarchy_prediction", {})
    utilization_fields = (
        ("compute", "compute_util_pct"),
        ("shared_memory", "smem_util_pct"),
        ("l2", "l2_util_pct"),
        ("global_memory_ddr", "ddr_util_pct"),
    )
    resource_hotspots = _ranked([
        {
            "resource": resource,
            "utilization_pct": report.get(field, 0),
            "ranking_metric": "modeled_physical_peak_utilization_pct",
            "value_kind": "model_estimate",
        }
        for resource, field in utilization_fields
    ], "utilization_pct")

    category_hotspots = []
    cost_model = report.get("cost_model", {})
    iteration_cost = cost_model.get("pipeline_iteration_cost_us", {})
    if iteration_cost:
        category_hotspots = _ranked([
            {
                "tile_op_category": "gemm",
                "cost_us_per_pipeline_iteration": iteration_cost.get("compute", 0),
                "ranking_metric": "modeled_cost_us_per_pipeline_iteration",
                "value_kind": "model_estimate",
            },
            {
                "tile_op_category": "copy",
                "cost_us_per_pipeline_iteration": iteration_cost.get("memory", 0),
                "ranking_metric": "modeled_cost_us_per_pipeline_iteration",
                "value_kind": "model_estimate",
            },
        ], "cost_us_per_pipeline_iteration")
    else:
        category_hotspots = [{
            "rank": 1,
            "tile_op_category": report.get("op", "unknown"),
            "cost_us": report.get("latency_with_memory_hierarchy_us",
                                  report.get("latency_us", 0)),
            "ranking_metric": "modeled_kernel_cost_us",
            "value_kind": "model_estimate",
        }]

    transfer_rows = []
    tile_cache = report.get("tile_cache_predictions", {})
    for item in tile_cache.get("operands", []):
        transfer_rows.append({
            "op_id": item.get("op_id"),
            "pipeline_id": item.get("pipeline_id"),
            "tile_op_category": item.get("operation", "copy"),
            "operand": item.get("operand"),
            "path": f"{item.get('source_level')}->l2->{item.get('destination_level')}",
            "tile_shape": item.get("tile_shape"),
            "logical_request_bytes": item.get("logical_request_bytes", 0),
            "predicted_ddr_read_bytes": item.get("ddr", {}).get(
                "predicted_read_bytes", 0),
            "l2_hit_rate_pct": item.get("l2", {}).get(
                "predicted_hit_rate_pct"),
            "ranking_metric": "predicted_ddr_read_bytes",
            "value_kind": "model_estimate",
        })
    transfer_hotspots = _ranked(transfer_rows, "predicted_ddr_read_bytes")

    pipeline = report.get("pipeline", {})
    timeline = []
    phase_hotspots = []
    if pipeline:
        grid_k = 1
        if report.get("op") == "gemm":
            grid_k = math.ceil(report["shape"][2] / report["tile"][2])
        phases = [
            ("prologue", pipeline.get("prologue_us", 0)),
            ("steady", (pipeline.get("steady_per_iter_us") or 0) * grid_k),
            ("epilogue", pipeline.get("epilogue_us", 0)),
        ]
        cursor = 0.0
        for sequence, (phase, duration) in enumerate(phases):
            duration = round(duration, 6)
            timeline.append({
                "sequence": sequence,
                "event": f"pipeline_{phase}",
                "start_us": round(cursor, 6),
                "duration_us": duration,
                "end_us": round(cursor + duration, 6),
                "scope": "per_tile_pipeline_model",
                "value_kind": "model_estimate",
            })
            phase_hotspots.append({
                "phase": phase,
                "duration_us": duration,
                "ranking_metric": "modeled_per_tile_phase_duration_us",
                "value_kind": "model_estimate",
            })
            cursor += duration
        phase_hotspots = _ranked(phase_hotspots, "duration_us")
    else:
        duration = report.get("latency_with_memory_hierarchy_us",
                              report.get("latency_us", 0))
        timeline = [{
            "sequence": 0,
            "event": report.get("op", "kernel"),
            "start_us": 0.0,
            "duration_us": duration,
            "end_us": duration,
            "scope": "kernel_model",
            "value_kind": "model_estimate",
        }]
        phase_hotspots = [{
            "rank": 1, "phase": report.get("op", "kernel"),
            "duration_us": duration,
            "ranking_metric": "modeled_kernel_duration_us",
            "value_kind": "model_estimate",
        }]

    levels = []
    for level, source_key in (
        ("ordinary_l1", "l1_cache"),
        ("l2", "l2"),
        ("global_memory_ddr", "ddr"),
        ("shared_memory", "shared_memory"),
    ):
        source = hierarchy.get(source_key, {})
        levels.append({
            "level": level,
            "predicted_hit_rate_pct": source.get(
                "predicted_hit_rate_pct", source.get("hit_rate", "not_applicable")
            ),
            "read_bytes": source.get(
                "input_read_bytes", source.get("read_bytes",
                source.get("eligible_read_bytes", 0))
            ),
            "write_bytes": source.get(
                "write_bytes", source.get("ordinary_write_bytes", 0)
            ),
            "total_traffic_bytes": source.get("total_traffic_bytes"),
            "service_cost_us": source.get("service_cost_us"),
            "value_kind": "model_estimate",
        })

    kernel_scale = {
        "modeled_latency_us": report.get("latency_with_memory_hierarchy_us",
                                         report.get("latency_us")),
        "waves": report.get("waves"),
    }
    block_scale = {"threads": report.get("threads")}
    if report.get("op") == "gemm":
        grid_m = math.ceil(report["shape"][0] / report["tile"][0])
        grid_n = math.ceil(report["shape"][1] / report["tile"][1])
        kernel_scale.update({
            "grid_shape_tiles": [grid_m, grid_n],
            "grid_tile_count": grid_m * grid_n,
            "inner_iterations_per_tile": math.ceil(
                report["shape"][2] / report["tile"][2]
            ),
        })
        block_scale = {
            "tile_shape": report.get("tile"),
            "warps_per_block": report.get("warps_per_block"),
            "pipeline_stages": report.get("stage"),
        }

    return {
        "schema": "tilesight/hierarchical-report-v1",
        "report_scope": "kernel",
        "value_kind": "model_estimate",
        "source": "existing_analyze_op_fields",
        "calculation_change": "none; report-layer transformation only",
        "kernel_summary": {
            "op": report.get("op"),
            "shape": report.get("shape", [report.get("N")]),
            "arch": report.get("arch", "S5000"),
            "latency_us": report.get("latency_us"),
            "latency_with_memory_hierarchy_us": report.get(
                "latency_with_memory_hierarchy_us"),
            "critical_path": cost_model.get("critical_path",
                hierarchy.get("critical_memory_path")),
        },
        "hotspots": {
            "resource_by_utilization": resource_hotspots,
            "tile_op_category_by_cost": category_hotspots,
            "tile_transfer_by_ddr_bytes": transfer_hotspots,
            "pipeline_phase_by_duration": phase_hotspots,
        },
        "timeline": {
            "scope": timeline[0]["scope"] if timeline else "not_available",
            "events": timeline,
            "interpretation": (
                "model timeline; not a runtime trace. GEMM phase offsets describe "
                "one Tile pipeline and exclude cross-SM wave scheduling"
            ),
        },
        "hardware_hierarchy": {
            "path": ["global_memory_ddr", "l2", "ordinary_l1_or_bypass",
                     "shared_memory", "register_tensor_compute"],
            "levels": levels,
            "critical_memory_path": hierarchy.get("critical_memory_path"),
        },
        "cache_behavior": tile_cache or {
            "status": "available_at_kernel_level",
            "l2_hit_rate_pct": report.get("l2_hit_rate_pct"),
        },
        "scale_units": {
            "kernel": kernel_scale,
            "sm": {
                "count": report.get("sm_count"),
                "resident_tiles_per_sm": report.get("tiles_per_sm"),
            },
            "thread_block": block_scale,
            "warp": {"warp_tile": report.get("warp_tile")},
        },
        "anomaly_summary": report.get("transfer_anomaly_analysis"),
        "limitations": [
            "No runtime trace or hardware counter is consumed by this report.",
            "Tile op IDs require the 5.1 metadata index to be supplied in a future joined report.",
            "Per-SM measured load distribution is not inferred from aggregate model data.",
        ],
    }


def render_report_summary(report, hierarchical=None):
    """Render a compact human-readable 5.5 report for terminal use."""
    view = hierarchical or build_hierarchical_report(report)
    summary = view["kernel_summary"]
    lines = [
        "S5000 TileSight 层级性能报告",
        f"算子: {summary['op']}  Shape: {summary['shape']}  架构: {summary['arch']}",
        f"预测延迟: {summary['latency_us']} us",
        "",
        "资源热点（按物理峰值利用率）",
    ]
    for item in view["hotspots"]["resource_by_utilization"]:
        lines.append(
            f"  {item['rank']}. {item['resource']}: {item['utilization_pct']}%"
        )
    lines.extend(["", "TileOp类别热点"])
    for item in view["hotspots"]["tile_op_category_by_cost"]:
        cost = item.get("cost_us_per_pipeline_iteration", item.get("cost_us"))
        lines.append(f"  {item['rank']}. {item['tile_op_category']}: {cost} us")
    critical = summary.get("critical_path") or {}
    if critical.get("tile_ops"):
        lines.extend(["", "关键路径TileOp"])
        for item in critical["tile_ops"]:
            lines.append(
                f"  {item['op_id']}: {item['category']} / {item['role']}"
            )
    transfers = view["hotspots"]["tile_transfer_by_ddr_bytes"]
    if transfers:
        lines.extend(["", "Tile搬运热点（按预测DDR读取量）"])
        for item in transfers:
            lines.append(
                f"  {item['rank']}. {item['operand']} "
                f"({item.get('op_id') or 'op_id unavailable'}) {item['path']}: "
                f"DDR={item['predicted_ddr_read_bytes']} B, "
                f"L2命中率={item['l2_hit_rate_pct']}%"
            )
    lines.extend(["", "Pipeline模型时间线"])
    for event in view["timeline"]["events"]:
        lines.append(
            f"  {event['sequence']}. {event['event']}: "
            f"{event['start_us']} -> {event['end_us']} us "
            f"(duration={event['duration_us']} us)"
        )
    lines.extend(["", "硬件层级"])
    for level in view["hardware_hierarchy"]["levels"]:
        lines.append(
            f"  {level['level']}: hit={level['predicted_hit_rate_pct']}, "
            f"read={level['read_bytes']} B, write={level['write_bytes']} B, "
            f"service={level['service_cost_us']} us"
        )
    scale = view["scale_units"]
    lines.extend([
        "", "Scale Units",
        f"  kernel: {scale['kernel']}",
        f"  sm: {scale['sm']}",
        f"  thread_block: {scale['thread_block']}",
        f"  warp: {scale['warp']}",
        "", "说明: 以上均为 model_estimate；Pipeline时间线不是运行期实测trace。",
    ])
    return "\n".join(lines) + "\n"


def build_session_report(reports, session_id="offline_session"):
    """Aggregate existing kernel reports without recomputing any kernel."""
    kernels = []
    for index, report in enumerate(reports):
        view = build_hierarchical_report(report)
        hierarchy = report.get("memory_hierarchy_prediction", {})
        kernels.append({
            "kernel_index": index,
            "kernel_id": report.get("kernel_id", f"kernel_{index}"),
            "op": report.get("op"),
            "shape": report.get("shape", [report.get("N")]),
            "latency_us": report.get("latency_with_memory_hierarchy_us",
                                     report.get("latency_us", 0)),
            "ddr_total_bytes": hierarchy.get("ddr", {}).get(
                "total_traffic_bytes", 0),
            "l2_total_bytes": hierarchy.get("l2", {}).get(
                "total_traffic_bytes", 0),
            "dominant_resource": (
                view["hotspots"]["resource_by_utilization"][0]["resource"]
                if view["hotspots"]["resource_by_utilization"] else None
            ),
            "hierarchical_report": view,
        })
    total_latency = sum(item["latency_us"] or 0 for item in kernels)
    return {
        "schema": "tilesight/session-report-v1",
        "session_id": session_id,
        "report_scope": "session",
        "value_kind": "model_estimate",
        "calculation_change": "none; aggregates existing kernel reports",
        "summary": {
            "kernel_count": len(kernels),
            "total_modeled_serial_latency_us": round(total_latency, 6),
            "aggregation_assumption": "kernels_execute_serially_in_input_order",
        },
        "kernel_hotspots_by_latency": _ranked([
            {key: value for key, value in item.items()
             if key != "hierarchical_report"}
            for item in kernels
        ], "latency_us"),
        "kernel_hotspots_by_ddr_bytes": _ranked([
            {key: value for key, value in item.items()
             if key != "hierarchical_report"}
            for item in kernels
        ], "ddr_total_bytes"),
        "timeline": [
            {
                "sequence": index,
                "kernel_id": item["kernel_id"],
                "op": item["op"],
                "start_us": round(sum(
                    previous["latency_us"] or 0 for previous in kernels[:index]
                ), 6),
                "duration_us": item["latency_us"],
                "end_us": round(sum(
                    previous["latency_us"] or 0
                    for previous in kernels[:index + 1]
                ), 6),
                "scope": "modeled_serial_session",
                "value_kind": "model_estimate",
            }
            for index, item in enumerate(kernels)
        ],
        "kernels": kernels,
        "limitations": [
            "Input order is treated as serial because no runtime session trace is supplied.",
            "This aggregation does not infer concurrency or measured inter-kernel gaps.",
        ],
    }


def render_session_summary(session):
    lines = [
        f"TileSight Session报告: {session['session_id']}",
        f"Kernel数量: {session['summary']['kernel_count']}",
        ("串行预测总延迟: "
         f"{session['summary']['total_modeled_serial_latency_us']} us"),
        "", "Kernel热点（按预测延迟）",
    ]
    for item in session["kernel_hotspots_by_latency"]:
        lines.append(
            f"  {item['rank']}. {item['kernel_id']} {item['op']} "
            f"{item['shape']}: {item['latency_us']} us, "
            f"dominant={item['dominant_resource']}"
        )
    lines.extend(["", "Session模型时间线"])
    for item in session["timeline"]:
        lines.append(
            f"  {item['sequence']}. {item['kernel_id']}: "
            f"{item['start_us']} -> {item['end_us']} us"
        )
    lines.extend([
        "", "说明: 输入顺序按串行聚合；这不是运行期实测Session trace。",
    ])
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="S5000 算子性能分析工具")
    parser.add_argument("op", nargs="?", choices=["relu", "add"],
                        help=("非GEMM算子类型；GEMM使用--compiler-info，"
                              "兼容旧输入时使用--tile-metadata"))
    parser.add_argument("-o", "--output", default="", help="输出文件 (默认打印到终端)")
    parser.add_argument(
        "--report-type", choices=["json", "hierarchical", "summary"],
        default="json",
        help=("报告类型：json=完整原始JSON；hierarchical=5.5层级JSON；"
              "summary=终端可读摘要"),
    )
    parser.add_argument("--arch", default="S5000", help="架构 (默认 S5000)")
    model_input = parser.add_mutually_exclusive_group()
    model_input.add_argument(
        "--compiler-info", default="",
        help=("tilelang/compiler-info-v1 JSON；经严格适配器生成统一"
              "ModelRequest后调用TileSight"),
    )
    model_input.add_argument(
        "--tile-metadata", default="",
        help=("兼容路径：5.1 tilesight/tile-metadata-v1 JSON；"
              "自动推导GEMM模型输入"),
    )
    parser.add_argument(
        "--kernel-id", default="",
        help="metadata含多个Kernel时选择目标kernel_id",
    )

    # Elementwise 参数
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=256, help="线程数")
    parser.add_argument(
        "--cache-condition", choices=["cold", "warm"], default="cold",
        help="元素级算子的L2初始条件；必须显式区分冷启动与预热复用",
    )

    args = parser.parse_args()
    arch = make_custom_stacked_gpu(make_s5000_config())

    if args.compiler_info:
        if args.op is not None:
            parser.error("--compiler-info cannot be combined with an op argument")
        report = analyze_from_compiler_info(
            args.compiler_info,
            arch=arch,
            kernel_id=args.kernel_id or None,
            cache_condition=args.cache_condition,
        )
    elif args.tile_metadata:
        if args.op is not None:
            parser.error("--tile-metadata cannot be combined with an op argument")
        report = analyze_gemm_from_metadata(
            args.tile_metadata, arch=arch, kernel_id=args.kernel_id or None
        )
    elif args.op == "relu":
        report = analyze_relu(args.N, args.threads, arch, args.cache_condition)
    elif args.op == "add":
        report = analyze_add(args.N, args.threads, arch, args.cache_condition)
    else:
        parser.error(
            "op is required unless --compiler-info or --tile-metadata is supplied"
        )

    # 添加元信息
    report["arch"] = args.arch
    report["core_freq_ghz"] = arch.core_freq / 1e9
    report["sm_count"] = arch.sm_count

    hierarchical = build_hierarchical_report(report)
    if args.report_type == "json":
        output = json.dumps(report, indent=2, ensure_ascii=False)
    elif args.report_type == "hierarchical":
        output = json.dumps(hierarchical, indent=2, ensure_ascii=False)
    else:
        output = render_report_summary(report, hierarchical).rstrip("\n")
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"报告已保存到: {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()
