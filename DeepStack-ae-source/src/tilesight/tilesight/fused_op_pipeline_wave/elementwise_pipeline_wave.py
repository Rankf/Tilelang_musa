"""Pipeline-aware element-wise operation performance model.

Element-wise ops 通常 stage_num=1（无内层循环），pipeline 退化为 roofline。
主要改进来自 wave head/tail + occupancy 分析。

包含 N_0, N_1, N_N 三种变体。
"""
import numpy as np
import math
import logging

from tilesight.arch import uses_dram_wave_quantization

from .resource_types import (
    PerIterationResources, PrologueEpilogueResources,
    TileResources, PipelineResult,
)
from .occupancy import compute_occupancy
from .pipeline_overlap import (
    compute_capacity_for_resource,
    compute_pipeline_tile_latency,
    compute_pipeline_tile_latency_with_occupancy,
    resources_to_times,
)
from .wave_model import compute_wave_adjusted_latency
from tilesight.memory import estimate_kernel_scope_writeback

log = logging.getLogger(__name__)


def _apply_wave_bytes_correction(io_val, arch):
    """Apply ddr_wave_bytes quantization when the architecture opts in."""
    if io_val > 0 and uses_dram_wave_quantization(arch):
        waves = io_val / arch.ddr_wave_bytes
        if waves > 0:
            return io_val * math.ceil(waves) / waves
    return io_val


def _compute_thread_overhead(thread_per_tb):
    """计算线程数不足导致的计算开销。"""
    if thread_per_tb <= 32:
        return 32 / thread_per_tb
    elif thread_per_tb <= 128:
        return 128 / thread_per_tb
    elif thread_per_tb <= 256:
        return 256 / thread_per_tb
    elif thread_per_tb <= 384:
        return 384 / thread_per_tb
    else:
        return 1


def _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate, data_bytes=4,
                  cache_detail=None, occupancy_detail=None,
                  cache_condition="cold"):
    """从 TileResources 构建 PipelineResult 的通用流程。"""
    from .wave_model import compute_wave_geometry
    total_tiles = int(np.prod(tile_res.grids))
    model_waves = compute_wave_geometry(total_tiles, tiles_per_sm, arch)['waves_float']
    # Pipeline latency
    sm_latency, pipeline_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_res, tiles_per_sm, arch, data_bytes=data_bytes, model_waves=model_waves)
    per_tile_latency, _ = compute_pipeline_tile_latency(
        tile_res, arch, data_bytes=data_bytes)

    # Wave adjustment (传入 tile_res 以便 tail wave 精确重算 active_sms 带宽)
    total_tiles = int(np.prod(tile_res.grids))
    dispatch_mode = getattr(
        arch, "elementwise_wave_dispatch_mode", "repeated_resident_batch"
    )
    memory_service_scale = 1.0
    if dispatch_mode == "continuous_cta_dispatch":
        profiles = getattr(arch, "elementwise_wave_memory_service_scales", {})
        memory_service_scale = float(profiles.get(cache_condition, 1.0))
    total_latency, wave_info = compute_wave_adjusted_latency(
        sm_latency, tiles_per_sm, total_tiles, arch,
        pipeline_detail=pipeline_detail,
        tile_res=tile_res, data_bytes=data_bytes,
        dispatch_mode=dispatch_mode,
        fixed_head_time=(
            float(getattr(arch, "elementwise_wave_fixed_head_us", 0.0)) * 1e-6
            if dispatch_mode == "continuous_cta_dispatch" else 0.0
        ),
        memory_service_scale=memory_service_scale,
    )
    dispatch_detail = wave_info.get("continuous_pipeline_detail")
    dispatch_components = wave_info.get("continuous_dispatch")

    # 将内部已建模的计算量正式带到结果层。此前这里只返回 latency，
    # PipelineResult.compute_util 因而一直保留默认值 0。
    per_tile_compute = (
        tile_res.per_iter.compute_flops * tile_res.num_iterations
        + tile_res.prologue_epilogue.epilogue_compute_flops
    )
    total_compute = per_tile_compute * total_tiles
    compute_resource = tile_res.per_iter.compute_resource
    compute_peak = compute_capacity_for_resource(
        arch, data_bytes, compute_resource, tile_res.per_iter.special_math_op,
        tile_res.per_iter.compute_profile_key,
    )
    compute_roofline_time = (
        total_compute / compute_peak if compute_peak > 0 else 0.0
    )
    compute_util = (
        compute_roofline_time / total_latency if total_latency > 0 else 0.0
    )

    return PipelineResult(
        per_tile_latency=per_tile_latency,
        total_latency=total_latency,
        compute_util=compute_util,
        compute_resource=compute_resource,
        compute_operations=total_compute,
        compute_peak_ops_per_s=compute_peak,
        compute_profile_key=tile_res.per_iter.compute_profile_key,
        occupancy_detail=occupancy_detail or {},
        l2_hit_rate=l2_hit_rate,
        smem_footprint=tile_res.smem_footprint,
        reg_footprint=tile_res.reg_footprint,
        tiles_per_sm=tiles_per_sm,
        waves=wave_info['waves_float'],
        pipeline_detail=pipeline_detail,
        dispatch_pipeline_detail=dispatch_detail,
        dispatch_components=dispatch_components,
        cache_detail=cache_detail,
    )


def _estimate_elementwise_cache(read_working_set_bytes,
                                logical_output_write_bytes, arch,
                                cache_condition, *,
                                dirty_retention_capacity_bytes=None):
    """Estimate cache residency for one standalone element-wise kernel.

    ``cache_condition`` describes input residency before the modeled kernel.
    Output stores are write-back traffic: when the complete read/output working
    set fits L2, dirty output can remain resident beyond the kernel measurement
    window.  The logical output traffic is retained separately so that callers
    do not confuse an in-kernel DDR observation with bytes eventually written
    back after eviction.
    """
    if cache_condition not in ("cold", "warm"):
        raise ValueError(
            "cache_condition must be either 'cold' or 'warm', got "
            f"{cache_condition!r}"
        )

    l2_capacity = arch.l2_capacity
    if cache_condition == "cold":
        l2_hit_rate = 0.0
        prediction_method = "cold_start"
        prediction_range_pct = None
    elif read_working_set_bytes <= l2_capacity:
        # S5000 P9 warm element-wise validation, derived from byte traffic:
        # ReLU 1M=99.7317%, Add 1M=99.8065%.  Use a conservative rounded
        # midpoint rather than claiming an exact 100% cache hit rate.
        l2_hit_rate = 0.9975
        prediction_method = "calibrated_warm_working_set_fits_l2"
        prediction_range_pct = [99.0, 100.0]
    else:
        l2_hit_rate = 0.0
        prediction_method = "warm_working_set_exceeds_l2_capacity"
        prediction_range_pct = None

    writeback = estimate_kernel_scope_writeback(
        # First-stage approximation: Elementwise integration does not yet
        # provide per-wave input last-use information.  The generic module
        # keeps this approximation explicit for replacement by phase two.
        live_read_bytes=read_working_set_bytes,
        logical_output_write_bytes=logical_output_write_bytes,
        effective_l2_capacity_bytes=l2_capacity,
        store_path="ordinary_cached",
        cache_condition=cache_condition,
        dirty_retention_capacity_bytes=dirty_retention_capacity_bytes,
        work_mode="single_kernel",
    )

    return {
        "cache_condition": cache_condition,
        "l2_hit_rate": l2_hit_rate,
        "l2_prediction_method": prediction_method,
        "l2_prediction_range_pct": prediction_range_pct,
        "read_working_set_bytes": read_working_set_bytes,
        "logical_output_write_bytes": writeback["logical_output_write_bytes"],
        "resident_working_set_bytes": writeback["resident_pressure_bytes"],
        "kernel_scope_ddr_write_bytes": writeback["kernel_scope_ddr_write_bytes"],
        "deferred_writeback_bytes": writeback["deferred_writeback_bytes"],
        "ddr_write_prediction_method": writeback["write_prediction_method"],
        # The primary TileSight performance model retains the upstream
        # full-output cost convention.  Kernel-window/deferred estimates stay
        # available as an independent diagnostic and MCU comparison scope.
        "performance_model_ddr_write_bytes": logical_output_write_bytes,
        "performance_model_write_scope": "full_logical_output_writeback_cost",
        "l2_capacity_bytes": l2_capacity,
        "writeback_analysis": writeback,
    }


# =====================================================================
# N_0 Element-wise: 单输入单输出, 形状不变
# =====================================================================
def calculate_N_0_elementwise_pipeline_wave(op_shape, tb_shape, dim_threads,
                                            arch, mem_levels, num_ops=1, *,
                                            cache_condition="cold",
                                            compute_resource="vector_alu",
                                            compute_ops_per_element=None,
                                            special_math_op=None,
                                            compute_profile_key=None,
                                            occupancy_hint=None,
                                            overlap_adapter_key=None,
                                            ddr_write_scope="full_logical"):
    """N_0 element-wise pipeline-aware model.

    Args:
        op_shape: 操作形状 (多维)
        tb_shape: thread block tile 形状
        dim_threads: 各维度线程数
        arch: 架构对象
        mem_levels: {'in1': [...], 'out1': [...]}
        num_ops: fused 操作数
        cache_condition: modeled L2 input state, ``cold`` or ``warm``

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    thread_tiles = [math.ceil(tb_dim / thread_dim)
                    for tb_dim, thread_dim in zip(tb_shape, dim_threads)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(op_shape, tb_shape)]

    in1_level = mem_levels['in1']
    out1_level = mem_levels['out1']

    # ---- IO 计算 ----
    total_elements = np.prod(op_shape)
    logical_l2_read_io = total_elements * in1_level[0] * in1_level[-1]
    logical_l2_store_io = total_elements * out1_level[-1] * out1_level[0]
    cache_detail = _estimate_elementwise_cache(
        logical_l2_read_io, logical_l2_store_io, arch, cache_condition)
    l2_hit_rate = cache_detail["l2_hit_rate"]

    l2_read_io = logical_l2_read_io
    l2_store_io = logical_l2_store_io

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)

    # Preserve TileSight's original full logical output-write cost in the
    # primary latency model.  The partial in-kernel writeback estimate is a
    # separate measurement-scope diagnostic and must not redefine latency.
    if ddr_write_scope == "full_logical":
        selected_write_io = logical_l2_store_io
    elif ddr_write_scope == "kernel_scope":
        selected_write_io = cache_detail["kernel_scope_ddr_write_bytes"]
    else:
        raise ValueError("ddr_write_scope must be full_logical or kernel_scope")
    ddr_write_io = _apply_wave_bytes_correction(selected_write_io, arch)
    ddr_io = (l2_read_io * (1 - l2_hit_rate) + ddr_write_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_elements = np.prod(tb_shape)

    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)
    per_tile_smem = (per_tile_elements * in1_level[1] * in1_level[-1]
                     + per_tile_elements * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    if compute_ops_per_element is None:
        compute_ops_per_element = 2 * num_ops
    per_tile_compute = (
        compute_ops_per_element * per_tile_elements * overheads
    )
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = (per_tile_elements
                      * max(out1_level[1] - out1_level[0], 0)
                      * out1_level[-1])
    reg_footprint = math.ceil(np.prod(thread_tiles)
                              * (in1_level[1] * in1_level[-1] / 4 + out1_level[1] / 4)
                              * REG_spill_para)

    # ---- TileResources (element-wise: 1 iteration, stage=1) ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
        compute_resource=compute_resource,
        special_math_op=special_math_op,
        compute_profile_key=compute_profile_key,
    )
    pe = PrologueEpilogueResources(
        store_ddr_io=0, store_l2_io=0, store_smem_io=0,
    )
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
        # Compatibility argument only; special math uses service profiles.
        overlap_adapter_key=None,
    )

    occupancy_detail = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch,
        occupancy_hint=occupancy_hint, return_detail=True)
    tiles_per_sm = occupancy_detail["active_cta_per_sm"]

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1], cache_detail=cache_detail,
                         occupancy_detail=occupancy_detail,
                         cache_condition=cache_condition)


# =====================================================================
# N_1 Element-wise: 两输入 (broadcast), 单输出
# =====================================================================
def calculate_N_1_elementwise_pipeline_wave(in1_shape, in2_shape, in1_tb_shape,
                                            dim_threads, arch, mem_levels,
                                            num_ops=1, *,
                                            compute_resource="vector_alu",
                                            compute_ops_per_element=None,
                                            compute_profile_key=None,
                                            occupancy_hint=None):
    """N_1 element-wise pipeline-aware model.

    Args:
        in1_shape: 输入1 形状 (大 tensor)
        in2_shape: 输入2 形状 (broadcast tensor)
        in1_tb_shape: 输入1 的 tile 形状
        dim_threads: 各维度线程数
        arch: 架构对象
        mem_levels: {'in1': [...], 'in2': [...], 'out1': [...]}
        num_ops: fused 操作数

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    in1_thread_shape = [math.ceil(tb_dim / thread_dim)
                        for tb_dim, thread_dim in zip(in1_tb_shape, dim_threads)]
    in2_tb_shape = [math.ceil(float(in2_dim / in1_dim * tb_dim))
                    for in1_dim, in2_dim, tb_dim in zip(in1_shape, in2_shape, in1_tb_shape)]
    in2_thread_shape = [math.ceil(float(in2_dim / in1_dim * tb_dim))
                        for in1_dim, in2_dim, tb_dim
                        in zip(in1_tb_shape, in2_tb_shape, in1_thread_shape)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(in1_shape, in1_tb_shape)]

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # ---- IO 计算 ----
    l2_read_io = (np.prod(in1_shape) * in1_level[0] * in1_level[-1]
                  + np.prod(grids) * np.prod(in2_tb_shape) * in2_level[0] * in2_level[-1])
    l2_store_io = np.prod(in1_shape) * out1_level[-1] * out1_level[0]
    ddr_read_io = (np.prod(in1_shape) * in1_level[0] * in1_level[-1]
                   + np.prod(in2_shape) * in2_level[0] * in2_level[-1])
    ddr_store_io = np.prod(in1_shape) * out1_level[-1] * out1_level[0]

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)
    ddr_read_io = _apply_wave_bytes_correction(ddr_read_io, arch)
    ddr_store_io = _apply_wave_bytes_correction(ddr_store_io, arch)

    l2_hit_rate = max((l2_read_io - ddr_read_io) / max(l2_read_io, 1), 0)
    ddr_io = (ddr_read_io + ddr_store_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)

    per_tile_smem = (np.prod(in1_tb_shape) * in1_level[1] * in1_level[-1]
                     + np.prod(grids) * np.prod(in2_tb_shape) * in2_level[0] * in2_level[-1] * 2 / max(total_tiles, 1)
                     + thread_per_tb * np.prod(in2_thread_shape) * in2_level[1] * in2_level[-1]
                     + np.prod(in1_tb_shape) * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    if compute_ops_per_element is None:
        compute_ops_per_element = 2 * num_ops
    per_tile_compute = (
        compute_ops_per_element * np.prod(in1_tb_shape) * overheads
    )
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = (np.prod(in2_tb_shape) * (1 - (in2_level[1] - in2_level[0])) * in2_level[-1]
                      + np.prod(in1_tb_shape) * (out1_level[1] - out1_level[0]) * out1_level[-1])
    reg_footprint = math.ceil(
        (np.prod(in1_thread_shape) * (in1_level[1] * in1_level[-1] / 4 + out1_level[-1] / 4)
         + np.prod(in2_thread_shape) * in2_level[1] * in2_level[-1] / 4)
        * REG_spill_para)

    # ---- TileResources ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
        compute_resource=compute_resource,
        compute_profile_key=compute_profile_key,
    )
    pe = PrologueEpilogueResources()
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
    )

    occupancy_detail = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch,
        occupancy_hint=occupancy_hint, return_detail=True)
    tiles_per_sm = occupancy_detail["active_cta_per_sm"]

    # Broadcast reuse/traffic above is derived from the two shapes.  This
    # path has no externally supplied cache state or cache-detail object.
    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1],
                         occupancy_detail=occupancy_detail)


# =====================================================================
# N_N Element-wise: 两输入同形状, 单输出
# =====================================================================
def calculate_N_N_elementwise_pipeline_wave(op_shape, tb_shape, dim_threads,
                                            arch, mem_levels, num_ops=1, *,
                                            cache_condition="cold",
                                            compute_resource="vector_alu",
                                            compute_ops_per_element=None,
                                            compute_profile_key=None,
                                            occupancy_hint=None):
    """N_N element-wise pipeline-aware model.

    Args:
        op_shape: 操作形状
        tb_shape: thread block tile 形状
        dim_threads: 各维度线程数
        arch: 架构对象
        mem_levels: {'in1': [...], 'in2': [...], 'out1': [...]}
        num_ops: fused 操作数
        cache_condition: modeled L2 input state, ``cold`` or ``warm``

    Returns:
        PipelineResult
    """
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1

    thread_per_tb = np.prod(dim_threads)
    thread_tiles = [math.ceil(tb_dim / thread_dim)
                    for tb_dim, thread_dim in zip(tb_shape, dim_threads)]
    grids = [math.ceil(dim / tb_dim) for dim, tb_dim in zip(op_shape, tb_shape)]

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    total_elements = np.prod(op_shape)

    # ---- IO ----
    logical_l2_read_io = (total_elements * in1_level[0] * in1_level[-1]
                          + total_elements * in2_level[0] * in2_level[-1])
    logical_l2_store_io = total_elements * out1_level[-1] * out1_level[0]
    cache_detail = _estimate_elementwise_cache(
        logical_l2_read_io, logical_l2_store_io, arch, cache_condition)
    l2_hit_rate = cache_detail["l2_hit_rate"]

    l2_read_io = logical_l2_read_io
    l2_store_io = logical_l2_store_io

    l2_read_io = _apply_wave_bytes_correction(l2_read_io, arch)
    l2_store_io = _apply_wave_bytes_correction(l2_store_io, arch)

    # Preserve TileSight's original full logical output-write cost in the
    # primary latency model.  See the N_0 path above for scope semantics.
    ddr_write_io = _apply_wave_bytes_correction(logical_l2_store_io, arch)
    ddr_io = (l2_read_io * (1 - l2_hit_rate) + ddr_write_io) * DDR_non_ideal_para

    if arch.core in ("A100", "H100", "B200"):
        l2_io = l2_read_io * l2_hit_rate + l2_read_io * (1 - l2_hit_rate) * 2 + l2_store_io * 2
    else:
        l2_io = l2_read_io + l2_store_io

    # ---- Per tile ----
    total_tiles = int(np.prod(grids))
    per_tile_ddr = ddr_io / max(total_tiles, 1)
    per_tile_l2 = l2_io / max(total_tiles, 1)

    per_tile_elements = np.prod(tb_shape)
    per_tile_smem = (per_tile_elements * in1_level[1] * in1_level[-1]
                     + per_tile_elements * in2_level[1] * in2_level[-1]
                     + per_tile_elements * out1_level[1] * out1_level[-1])

    overheads = _compute_thread_overhead(thread_per_tb)
    if compute_ops_per_element is None:
        compute_ops_per_element = 2 * num_ops
    per_tile_compute = (
        compute_ops_per_element * per_tile_elements * overheads
    )
    per_tile_smem *= overheads

    # ---- Footprint ----
    smem_footprint = (per_tile_elements
                      * max(out1_level[1] - out1_level[0], 0)
                      * out1_level[-1])
    reg_footprint = math.ceil(
        np.prod(thread_tiles)
        * (out1_level[-1] / 4 + in1_level[1] * in1_level[-1] / 4
           + in2_level[1] * in2_level[-1] / 4)
        * REG_spill_para)

    # ---- TileResources ----
    per_iter = PerIterationResources(
        ddr_io=per_tile_ddr, l2_io=per_tile_l2,
        smem_io=per_tile_smem, compute_flops=per_tile_compute,
        compute_resource=compute_resource,
        compute_profile_key=compute_profile_key,
    )
    pe = PrologueEpilogueResources()
    tile_res = TileResources(
        per_iter=per_iter, prologue_epilogue=pe,
        num_iterations=1, stage_num=1,
        smem_footprint=smem_footprint, reg_footprint=reg_footprint,
        warps_per_block=max(int(thread_per_tb / 32), 1),
        grids=tuple(grids),
    )

    occupancy_detail = compute_occupancy(
        smem_footprint, reg_footprint,
        max(int(thread_per_tb / 32), 1), arch,
        occupancy_hint=occupancy_hint, return_detail=True)
    tiles_per_sm = occupancy_detail["active_cta_per_sm"]

    return _build_result(tile_res, tiles_per_sm, arch, l2_hit_rate,
                         data_bytes=out1_level[-1], cache_detail=cache_detail,
                         occupancy_detail=occupancy_detail,
                         cache_condition=cache_condition)
