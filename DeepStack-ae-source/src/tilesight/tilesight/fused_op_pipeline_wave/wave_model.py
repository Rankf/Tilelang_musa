import math
import logging
from .resource_types import PipelineDetail

log = logging.getLogger(__name__)


def compute_wave_geometry(total_tiles, tiles_per_sm, arch,
                          mma_type="wmma", sm_count_override=None):
    """Shared upstream wave arithmetic, independent of latency estimates."""
    sm_count = sm_count_override if sm_count_override else arch.sm_count
    if mma_type == "utcmma_cta2" and sm_count_override is None:
        sm_count = sm_count // 2
    tiles_per_wave = max(sm_count * tiles_per_sm, 1)
    num_full_waves = total_tiles // tiles_per_wave
    tail_tiles = total_tiles % tiles_per_wave
    return {
        'sm_count': sm_count,
        'tiles_per_wave': tiles_per_wave,
        'num_full_waves': num_full_waves,
        'tail_tiles': tail_tiles,
        'total_waves': num_full_waves + (1 if tail_tiles > 0 else 0),
        'waves_float': total_tiles / tiles_per_wave,
    }


def compute_wave_adjusted_latency(sm_latency, tiles_per_sm, total_tiles, arch,
                                  pipeline_detail=None, mma_type="wmma",
                                  tile_res=None, data_bytes=2, sm_count_override=None,
                                  dispatch_mode="repeated_resident_batch",
                                  fixed_head_time=0.0,
                                  memory_service_scale=1.0):
    """计算考虑 wave head/tail 效应的 kernel 总延迟。

    关键改进: 当 total_tiles < sm_count 时, 活跃 SM 数 < sm_count,
    每个活跃 SM 分到更多的 DDR/L2 共享带宽, per-tile 延迟更短。

    如果提供了 tile_res, 会对 tail wave (以及 total_tiles < sm_count 的情况)
    用正确的 active_sms 重新计算 per-tile latency。

    Args:
        sm_latency: full wave 下一个 SM 的延迟 (秒), 假设所有 SM 活跃
        tiles_per_sm: occupancy
        total_tiles: 空间 tile 总数 (gridM * gridN)
        arch: 架构对象
        pipeline_detail: PipelineDetail from full-wave computation
        mma_type: MMA 类型
        tile_res: TileResources, 用于精确重算 tail wave latency (可选)
        data_bytes: 数据类型字节数
        sm_count_override: 覆盖 sm_count (如 utcmma_cta2 用 sm_count//2)

    Returns:
        (total_latency, wave_info)
    """
    geometry = compute_wave_geometry(
        total_tiles, tiles_per_sm, arch, mma_type, sm_count_override)
    sm_count = geometry['sm_count']
    tiles_per_wave = geometry['tiles_per_wave']
    num_full_waves = geometry['num_full_waves']
    tail_tiles = geometry['tail_tiles']
    total_waves = geometry['total_waves']

    if dispatch_mode not in {
            "repeated_resident_batch", "continuous_cta_dispatch"}:
        raise ValueError(f"unsupported wave dispatch mode: {dispatch_mode!r}")

    if dispatch_mode == "continuous_cta_dispatch":
        if tile_res is None:
            raise ValueError("continuous CTA dispatch requires tile_res")
        from .pipeline_overlap import compute_continuous_dispatch_latency
        total_latency, continuous_detail, components = (
            compute_continuous_dispatch_latency(
                tile_res, tiles_per_sm, total_tiles, arch,
                data_bytes=data_bytes, sm_count=sm_count,
                memory_service_scale=memory_service_scale,
            )
        )
        total_latency += fixed_head_time
        components["fixed_head_time"] = fixed_head_time
        wave_info = {
            'total_waves': total_waves,
            'num_full_waves': num_full_waves,
            'tail_tiles': tail_tiles,
            'tiles_per_wave': tiles_per_wave,
            'head_penalty': 1.0,
            'waves_float': geometry['waves_float'],
            'dispatch_mode': dispatch_mode,
            'continuous_dispatch': components,
            'continuous_pipeline_detail': continuous_detail,
        }
        return total_latency, wave_info

    # === Head wave penalty ===
    head_penalty = 1.0
    if pipeline_detail is not None and pipeline_detail.prologue_time > 0:
        prologue_ratio = pipeline_detail.prologue_time / max(sm_latency, 1e-30)
        head_penalty = 1.0 + 0.1 * prologue_ratio

    # === Full waves ===
    if num_full_waves >= 1:
        # Full wave: 所有 sm_count 个 SM 活跃, sm_latency 已正确
        head_wave_time = sm_latency * head_penalty
        middle_waves = max(num_full_waves - 1, 0)
        middle_wave_time = sm_latency
    else:
        head_wave_time = 0.0
        middle_waves = 0
        middle_wave_time = 0.0

    # === Tail wave (包括 total_tiles < sm_count 的情况) ===
    #
    # GPU 调度是 round-robin: 先每 SM 分 1 个 tile, 填满后再分第 2 个。
    # 所以 tail_tiles 个 tile 的分配:
    #   - tail_tiles <= sm_count: 每个活跃 SM 恰好 1 个 tile
    #   - tail_tiles > sm_count: 所有 SM 活跃, busiest SM 有 ceil(tail_tiles/sm_count) 个
    #
    if tail_tiles > 0:
        if tail_tiles <= sm_count:
            # 不够填满所有 SM, 每个活跃 SM 只分到 1 个 tile
            active_sms = tail_tiles
            tail_tiles_per_sm = 1
        else:
            # 多于 SM 数, 所有 SM 活跃
            active_sms = sm_count
            tail_tiles_per_sm = math.ceil(tail_tiles / sm_count)
            # 不能超过 occupancy 容量
            tail_tiles_per_sm = min(tail_tiles_per_sm, tiles_per_sm)

        if tile_res is not None:
            # 精确重算: 用正确的 active_sms 和 tail_tiles_per_sm
            from .pipeline_overlap import compute_pipeline_tile_latency_with_occupancy
            tail_sm_latency, _ = compute_pipeline_tile_latency_with_occupancy(
                tile_res, tail_tiles_per_sm, arch, data_bytes=data_bytes,
                sm_count=sm_count, active_sms=active_sms,
                model_waves=geometry['waves_float'])
        else:
            tail_sm_latency = sm_latency * tail_tiles_per_sm / max(tiles_per_sm, 1)

        if num_full_waves == 0:
            tail_wave_time = tail_sm_latency * head_penalty
        else:
            tail_wave_time = tail_sm_latency
    else:
        tail_wave_time = 0.0
        active_sms = sm_count

    # === 总延迟 ===
    total_latency = head_wave_time + middle_waves * middle_wave_time + tail_wave_time

    waves = geometry['waves_float']

    wave_info = {
        'total_waves': total_waves,
        'num_full_waves': num_full_waves,
        'tail_tiles': tail_tiles,
        'tiles_per_wave': tiles_per_wave,
        'head_penalty': head_penalty,
        'waves_float': waves,
        'dispatch_mode': dispatch_mode,
    }

    log.info("wave model: total_tiles=%d, sm_count=%d, tiles_per_wave=%d, "
             "full_waves=%d, tail=%d, active_sms=%s => total_lat=%.3e",
             total_tiles, sm_count, tiles_per_wave, num_full_waves,
             tail_tiles, active_sms, total_latency)

    return total_latency, wave_info
