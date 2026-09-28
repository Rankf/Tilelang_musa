import math
import logging
from .resource_types import TileResources, PipelineDetail, PipelineResult

log = logging.getLogger(__name__)


def _special_math_profile(arch, operation):
    if not operation:
        return {}
    return getattr(arch, "special_math_service_profiles", {}).get(operation, {})


def compute_capacity_for_resource(arch, data_bytes, compute_resource,
                                  special_math_op=None, compute_profile_key=None):
    """返回指定计算资源的全芯片模型峰值（operations/s）。"""
    if compute_profile_key is not None:
        resolver = getattr(arch, "resolve_compute_profile", None)
        if not callable(resolver):
            raise ValueError(
                f"architecture has no dtype-aware compute profile resolver for {compute_profile_key!r}"
            )
        profile = resolver(compute_profile_key)
        if profile.get("resource") != compute_resource:
            raise ValueError(
                f"compute profile {compute_profile_key!r} uses {profile.get('resource')!r}, "
                f"not {compute_resource!r}"
            )
        return float(profile["peak_ops_per_s"])
    if compute_resource == "tensor_core":
        if data_bytes == 2:
            return arch.fp16_tensor_flops
        if data_bytes == 1:
            return getattr(
                arch, "fp8_tensor_flops",
                getattr(arch, "int8_tensor_flops", arch.fp16_tensor_flops),
            )
        if data_bytes == 4:
            return getattr(
                arch, "fp32_tensor_flops",
                getattr(arch, "fp32_cuda_core_flops", 0.0),
            )
    elif compute_resource == "vector_alu":
        attribute = {
            1: "fp8_cuda_core_flops",
            2: "fp16_cuda_core_flops",
            4: "fp32_cuda_core_flops",
            8: "fp64_cuda_core_flops",
        }.get(data_bytes)
        if attribute:
            return getattr(arch, attribute, 0.0)
    elif compute_resource == "sfu":
        rate = _special_math_profile(arch, special_math_op).get(
            "effective_service_rate_ops_per_s"
        )
        if rate is not None:
            return float(rate)
        return getattr(arch, "sfu_flops", 0.0)
    elif compute_resource == "log_sequence":
        # Deprecated alias normalized to the unified SFU Log profile.
        return compute_capacity_for_resource(arch, data_bytes, "sfu", "log")
    elif compute_resource == "int_alu":
        return getattr(arch, "int32_cuda_core_flops", 0.0)
    else:
        raise ValueError(
            "compute_resource must be tensor_core, vector_alu, sfu, log_sequence, or "
            f"int_alu; got {compute_resource!r}"
        )
    return 0.0


def resources_to_times(ddr_io, l2_io, smem_io, compute_flops, arch, data_bytes=2,
                       l1_5_io=0, compute_resource="tensor_core",
                       special_math_op=None, compute_profile_key=None):
    """将资源量转换为时间 (秒), 使用全系统带宽。

    返回值是"在全系统带宽下完成该资源量所需时间"。
    对于 per-SM 分析, 共享资源(DDR/L2)需乘以竞争 SM 数。

    Args:
        ddr_io: DDR 流量 (bytes)
        l2_io: L2 流量 (bytes)
        smem_io: SMEM 读写 (bytes)
        compute_flops: 计算量 (FLOPs)
        arch: 架构对象
        data_bytes: 数据类型字节数
        l1_5_io: L1.5 流量 (bytes), 0 if no L1.5
        compute_resource: tensor_core/vector_alu/sfu/int_alu

    Returns:
        (ddr_time, l2_time, l1_5_time, smem_time, compute_time) 各单位为秒
    """
    ddr_time = ddr_io / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0.0
    l2_time = l2_io / arch.l2_bandwidth if arch.l2_bandwidth > 0 else 0.0
    smem_time = smem_io / arch.smem_bandwidth if arch.smem_bandwidth > 0 else 0.0

    l1_5_bw = getattr(arch, 'l1_5_bandwidth', 0)
    l1_5_time = l1_5_io / l1_5_bw if l1_5_bw > 0 else 0.0

    flops_capacity = compute_capacity_for_resource(
        arch, data_bytes, compute_resource, special_math_op, compute_profile_key
    )
    if compute_flops > 0 and flops_capacity <= 0:
        raise ValueError(
            f"{compute_resource} has no calibrated {data_bytes}-byte "
            "compute capacity on this architecture"
        )

    compute_time = compute_flops / flops_capacity if flops_capacity > 0 else 0.0

    return ddr_time, l2_time, l1_5_time, smem_time, compute_time


def _per_sm_iter_times(per_iter, arch, data_bytes, sm_count, active_sms=None):
    """计算每迭代在 per-SM 带宽下的时间。

    关键区分:
    - DDR/L2: 全系统共享带宽, 由 active_sms 个 SM 平分
    - SMEM/Compute: per-SM 独立资源, 始终除以 sm_count (因为 bandwidth 定义含 sm_count)

    Args:
        per_iter: PerIterationResources
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构总 SM 数 (用于 SMEM/Compute, 因为 arch bandwidth 包含了 sm_count)
        active_sms: 实际活跃的 SM 数 (用于 DDR/L2 共享带宽分配),
                    默认 None 表示 full wave = sm_count

    Returns:
        (mem_time_per_iter, comp_time_per_iter) — 已除以 max_util
    """
    if active_sms is None:
        active_sms = sm_count

    ddr_t, l2_t, l1_5_t, smem_t, comp_t = resources_to_times(
        per_iter.ddr_io, per_iter.l2_io, per_iter.smem_io, per_iter.compute_flops,
        arch, data_bytes, l1_5_io=per_iter.l1_5_io,
        compute_resource=per_iter.compute_resource,
        special_math_op=per_iter.special_math_op,
        compute_profile_key=per_iter.compute_profile_key,
    )

    # DDR/L2: 共享带宽, 由 active_sms 个 SM 竞争
    # per-SM 时间 = 全系统时间 * active_sms
    ddr_t *= active_sms
    l2_t *= active_sms

    # L1.5: per-group 共享, bandwidth 定义是 whole chip
    # 与 SMEM 类似, 需要 × sm_count
    l1_5_t *= sm_count

    # SMEM/Compute: per-SM 资源, 但 arch.smem_bandwidth 和 flops 是全系统总和
    # per-SM 时间 = 全系统时间 * sm_count
    smem_t *= sm_count
    comp_t *= sm_count

    # 各 mem 层之间可 overlap (取 max), 再除以 max_util
    l1_5_max_util = getattr(arch, 'l1_5_max_util', arch.l1_max_util)
    mem_time = max(ddr_t / arch.ddr_max_util,
                   l2_t / arch.l2_max_util,
                   l1_5_t / l1_5_max_util,
                   smem_t / arch.l1_max_util)
    has_sustained_sfu_rate = bool(
        per_iter.compute_resource == "sfu"
        and _special_math_profile(arch, per_iter.special_math_op).get(
            "effective_service_rate_ops_per_s"
        ) is not None
    )
    profile_util = None
    if per_iter.compute_profile_key is not None:
        profile_util = arch.resolve_compute_profile(
            per_iter.compute_profile_key
        ).get("max_utilization")
    comp_time = (
        comp_t
        if per_iter.compute_resource == "log_sequence" or has_sustained_sfu_rate
        else comp_t / float(profile_util or arch.compute_max_util)
    )

    return mem_time, comp_time


def _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms=None):
    """计算 store (epilogue 写回) 在 per-SM 带宽下的时间。"""
    if active_sms is None:
        active_sms = sm_count

    store_ddr_t, store_l2_t, store_l1_5_t, store_smem_t, _ = resources_to_times(
        pe.store_ddr_io, pe.store_l2_io, pe.store_smem_io, 0,
        arch, data_bytes, l1_5_io=pe.store_l1_5_io
    )
    # DDR/L2 共享, L1.5/SMEM per-SM
    store_ddr_t *= active_sms
    store_l2_t *= active_sms
    store_l1_5_t *= sm_count
    store_smem_t *= sm_count

    l1_5_max_util = getattr(arch, 'l1_5_max_util', arch.l1_max_util)
    store_time = max(store_ddr_t / arch.ddr_max_util,
                     store_l2_t / arch.l2_max_util,
                     store_l1_5_t / l1_5_max_util,
                     store_smem_t / arch.l1_max_util)
    return store_time


def compute_pipeline_tile_latency(tile_res, arch, data_bytes=2,
                                  sm_count=None, active_sms=None):
    """计算单个 tile 的 pipeline-aware latency。

    使用 per-SM 带宽模型:
    - DDR/L2 (共享): 每个 SM 分到 系统带宽 / active_sms
    - SMEM/Compute (per-SM): 每个 SM 独立拥有 系统带宽 / sm_count

    当 total_tiles < sm_count 时, active_sms < sm_count,
    每个活跃 SM 分到更多的 DDR/L2 带宽。

    根据 stage_num 分情况：
    - stage_num=1: 无 pipeline，每迭代内 load 与 compute 串行
    - stage_num>=2: 有 pipeline，prologue + steady(overlap) + epilogue

    Args:
        tile_res: TileResources
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构 SM 总数 (默认 arch.sm_count)
        active_sms: 实际活跃 SM 数 (默认 = sm_count, 即 full wave)

    Returns:
        (per_tile_latency, PipelineDetail)
    """
    if sm_count is None:
        sm_count = arch.sm_count
    if active_sms is None:
        active_sms = sm_count

    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    stage_num = tile_res.stage_num
    num_iters = tile_res.num_iterations

    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms)
    store_time = _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms)

    if stage_num <= 1:
        # ===== Case 1: 无软件流水线 =====
        # 每迭代 load 与 compute 串行
        per_iter_time = mem_time_per_iter + comp_time_per_iter
        per_tile_latency = num_iters * per_iter_time + store_time

        detail = PipelineDetail(
            prologue_time=0.0,
            steady_time_per_iter=per_iter_time,
            epilogue_time=store_time,
            mem_time_per_iter=mem_time_per_iter,
            compute_time_per_iter=comp_time_per_iter,
        )
    else:
        # ===== Case 2: 有软件流水线 (stage_num >= 2) =====
        pipeline_depth = stage_num - 1

        prologue_time = pipeline_depth * mem_time_per_iter
        steady_iters = max(num_iters - pipeline_depth, 0)
        steady_time_per_iter = max(mem_time_per_iter, comp_time_per_iter)
        steady_time = steady_iters * steady_time_per_iter
        epilogue_time = pipeline_depth * comp_time_per_iter + store_time

        per_tile_latency = prologue_time + steady_time + epilogue_time

        detail = PipelineDetail(
            prologue_time=prologue_time,
            steady_time_per_iter=steady_time_per_iter,
            epilogue_time=epilogue_time,
            mem_time_per_iter=mem_time_per_iter,
            compute_time_per_iter=comp_time_per_iter,
        )

    log.info("tile latency: stage=%d, iters=%d, active_sms=%d, mem_t=%.3e, "
             "comp_t=%.3e, store_t=%.3e => tile_lat=%.3e",
             stage_num, num_iters, active_sms, mem_time_per_iter,
             comp_time_per_iter, store_time, per_tile_latency)

    return per_tile_latency, detail


def compute_pipeline_tile_latency_with_occupancy(tile_res, tiles_per_sm, arch,
                                                 data_bytes=2, sm_count=None,
                                                 active_sms=None, model_waves=None):
    """考虑 SM 上多 tile 交织的 pipeline latency。

    当 tiles_per_sm > 1 时，多个 tile 在同一 SM 上交替执行。
    等效 pipeline depth = stage_num * tiles_per_sm (算法中的 stage 累乘)。

    Args:
        tile_res: TileResources
        tiles_per_sm: occupancy (tiles per SM)
        arch: 架构对象
        data_bytes: 数据类型字节数
        sm_count: 架构 SM 总数 (默认 arch.sm_count)
        active_sms: 实际活跃 SM 数 (默认 = sm_count)

    Returns:
        (sm_latency, PipelineDetail)
    """
    if sm_count is None:
        sm_count = arch.sm_count
    if active_sms is None:
        active_sms = sm_count

    if tiles_per_sm <= 1:
        per_tile_lat, detail = compute_pipeline_tile_latency(
            tile_res, arch, data_bytes, sm_count=sm_count, active_sms=active_sms)
        return per_tile_lat, detail

    # tiles_per_sm > 1: 多 tile 交织
    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    stage_num = tile_res.stage_num
    num_iters = tile_res.num_iterations

    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms)
    store_time = _per_sm_store_time(pe, arch, data_bytes, sm_count, active_sms)
    total_iters = num_iters * tiles_per_sm

    # 等效 pipeline depth
    effective_stage = stage_num * tiles_per_sm
    effective_pipeline_depth = effective_stage - 1
    prologue_time = min(effective_pipeline_depth, total_iters) * mem_time_per_iter
    steady_iters = max(total_iters - effective_pipeline_depth, 0)
    steady_time_per_iter = max(mem_time_per_iter, comp_time_per_iter)
    steady_time = steady_iters * steady_time_per_iter
    drain_iters = min(effective_pipeline_depth, total_iters)
    epilogue_time = drain_iters * comp_time_per_iter + tiles_per_sm * store_time

    sm_latency = prologue_time + steady_time + epilogue_time

    # Deliberately no operation-specific CTA overlap correction is applied.
    # SFU operation profiles still provide their measured service rate through
    # ``compute_capacity_for_resource``; dispatch remains the common Wave model.

    detail = PipelineDetail(
        prologue_time=prologue_time,
        steady_time_per_iter=steady_time_per_iter,
        epilogue_time=epilogue_time,
        mem_time_per_iter=mem_time_per_iter,
        compute_time_per_iter=comp_time_per_iter,
    )

    log.info("SM latency: tiles_per_sm=%d, active_sms=%d, eff_stage=%d, "
             "total_iters=%d => sm_lat=%.3e",
             tiles_per_sm, active_sms, effective_stage, total_iters, sm_latency)

    return sm_latency, detail


def compute_continuous_dispatch_latency(tile_res, resident_tiles_per_sm,
                                        total_tiles, arch, data_bytes=2,
                                        sm_count=None,
                                        memory_service_scale=1.0):
    """Model a persistent CTA stream without draining at analytical waves.

    ``resident_tiles_per_sm`` is a concurrency limit, not a batch size.  Once
    the first resident set has filled the machine, completed CTAs are replaced
    continuously.  Consequently pipeline fill/drain is paid once for the
    kernel while the steady work and stores scale with the number of CTAs
    handled by the busiest SM.

    The result deliberately contains no dtype- or operation-specific timing
    constant.  Those costs remain in ``tile_res`` and the architecture
    resource profiles.
    """
    if sm_count is None:
        sm_count = arch.sm_count
    if memory_service_scale <= 0:
        raise ValueError("memory_service_scale must be positive")
    if total_tiles <= 0:
        return 0.0, PipelineDetail(), {
            "active_sms": 0,
            "busiest_tiles_per_sm": 0,
            "resident_tiles_per_sm": 0,
            "pipeline_fill_time": 0.0,
            "pipeline_steady_time": 0.0,
            "pipeline_drain_time": 0.0,
            "store_time": 0.0,
        }

    active_sms = min(total_tiles, sm_count)
    busiest_tiles_per_sm = math.ceil(total_tiles / active_sms)
    resident_tiles = min(
        max(int(resident_tiles_per_sm), 1), busiest_tiles_per_sm
    )

    per_iter = tile_res.per_iter
    pe = tile_res.prologue_epilogue
    mem_time_per_iter, comp_time_per_iter = _per_sm_iter_times(
        per_iter, arch, data_bytes, sm_count, active_sms
    )
    # Keep the logical full-writeback byte count unchanged.  This factor only
    # calibrates how the shared memory service time maps to wall-clock time for
    # the continuous CTA stream; the upstream max(memory, compute) rule stays
    # intact.  The default 1.0 preserves the upstream behavior.
    mem_time_per_iter *= memory_service_scale
    store_time_per_tile = _per_sm_store_time(
        pe, arch, data_bytes, sm_count, active_sms
    )

    total_iters = tile_res.num_iterations * busiest_tiles_per_sm
    effective_stage = tile_res.stage_num * resident_tiles
    pipeline_depth = max(effective_stage - 1, 0)
    fill_iters = min(pipeline_depth, total_iters)
    steady_iters = max(total_iters - pipeline_depth, 0)
    drain_iters = min(pipeline_depth, total_iters)

    fill_time = fill_iters * mem_time_per_iter
    steady_time_per_iter = max(mem_time_per_iter, comp_time_per_iter)
    steady_time = steady_iters * steady_time_per_iter
    drain_time = drain_iters * comp_time_per_iter
    store_time = busiest_tiles_per_sm * store_time_per_tile
    latency = fill_time + steady_time + drain_time + store_time

    detail = PipelineDetail(
        prologue_time=fill_time,
        steady_time_per_iter=steady_time_per_iter,
        epilogue_time=drain_time + store_time,
        mem_time_per_iter=mem_time_per_iter,
        compute_time_per_iter=comp_time_per_iter,
    )
    components = {
        "active_sms": active_sms,
        "busiest_tiles_per_sm": busiest_tiles_per_sm,
        "resident_tiles_per_sm": resident_tiles,
        "pipeline_fill_time": fill_time,
        "pipeline_steady_time": steady_time,
        "pipeline_drain_time": drain_time,
        "store_time": store_time,
        "memory_service_scale": memory_service_scale,
    }
    return latency, detail, components
