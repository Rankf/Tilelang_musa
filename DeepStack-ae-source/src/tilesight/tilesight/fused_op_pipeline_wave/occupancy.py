import math
import logging

log = logging.getLogger(__name__)


def _compute_occupancy_legacy_upstream(
    smem_footprint, reg_footprint, warps_per_block, arch, mma_type="wmma"
):
    """原始远程TileSight occupancy逻辑，保留用于回退和对照。

    该实现原名为 ``compute_occupancy``。S5000扩展不删除或改写它，
    只是默认改走下面包含threads/warp和精确二进制覆盖的新路径。

    三项约束取 min:
    1. configurable_smem_capacity / smem_footprint  (shared memory 限制)
    2. register_capacity_per_sm / (reg_per_block)    (寄存器限制)
    3. max_blocks_per_sm                              (硬件上限)

    Args:
        smem_footprint: 每个 thread block 的 shared memory 用量 (bytes)
        reg_footprint:  每个 warp 的寄存器用量 (4-byte register 数)
        warps_per_block: 每个 thread block 的 warp 数
        arch: 架构对象
        mma_type: MMA 类型，utcmma_cta2 需要特殊处理

    Returns:
        tiles_per_sm: int, 至少为 1
    """
    max_blocks = getattr(arch, 'max_blocks_per_sm', 24)

    # Constraint 1: shared memory
    if smem_footprint > 0:
        smem_limit = int(arch.configurable_smem_capacity / smem_footprint)
    else:
        smem_limit = max_blocks

    # Constraint 2: registers
    # reg_footprint 是每个 warp 使用的 4-byte 寄存器数
    # register_capacity_per_sm 是 bytes
    # 每 block 寄存器用量 (bytes) = reg_footprint * 4 * 32 * warps_per_block
    #   其中 32 = threads_per_warp, 4 = bytes_per_register
    # 但在现有代码中 reg_footprint 的单位因 op 类型而异：
    #   matmul: 已除以 32 和 4（是 register count per thread 的近似）
    #   elementwise: 类似
    # 使用保守估计: reg_footprint 作为 per-warp 的 4B register 数
    if reg_footprint > 0 and warps_per_block > 0:
        reg_per_block_bytes = reg_footprint * 4 * 32 * warps_per_block
        reg_limit = int(arch.register_capacity_per_sm / reg_per_block_bytes)
    else:
        reg_limit = max_blocks

    tiles_per_sm = min(smem_limit, reg_limit, max_blocks)

    # 至少为 1（即使 footprint 超出容量，仍能 launch 一个 block）
    tiles_per_sm = max(tiles_per_sm, 1)

    # utcmma_cta2: 2-CTA cluster，每个 cluster 占 2 个 SM slot
    if mma_type == "utcmma_cta2":
        tiles_per_sm = max(tiles_per_sm // 2, 1)

    log.info("occupancy: smem_limit=%d, reg_limit=%d, max_blocks=%d => tiles_per_sm=%d",
             smem_limit, reg_limit, max_blocks, tiles_per_sm)

    return tiles_per_sm


def compute_occupancy_detail(
    smem_footprint,
    reg_footprint,
    warps_per_block,
    arch,
    mma_type="wmma",
    *,
    occupancy_hint=None,
):
    """计算驻留CTA，并保留约束来源和证据等级。

    ``occupancy_hint``可以承载编译后二进制的MUSA Occupancy API结果。
    该结果优先级最高；没有它时使用架构threads/warp上限以及原有
    Shared/register估计。完全没有S5000扩展字段时回退上游实现。
    """
    hint = occupancy_hint or {}
    exact_active = hint.get("active_cta_per_sm")
    if exact_active is not None:
        if isinstance(exact_active, bool) or int(exact_active) != exact_active:
            raise ValueError("active_cta_per_sm must be a positive integer")
        active = int(exact_active)
        if active <= 0:
            raise ValueError("active_cta_per_sm must be a positive integer")
        if mma_type == "utcmma_cta2":
            active = max(active // 2, 1)
        return {
            "active_cta_per_sm": active,
            "source": hint.get("source", "exact_binary_musa_occupancy_api"),
            "confidence": hint.get("confidence", "high"),
            "limiting_resource": hint.get("limiting_resource", "api_resolved"),
            "constraints": {
                "exact_binary_active_cta_per_sm": int(exact_active),
                "threads_per_block": int(warps_per_block * 32),
                "warps_per_block": int(warps_per_block),
                "dynamic_shared_memory_bytes": hint.get(
                    "dynamic_shared_memory_bytes"
                ),
                "registers_per_thread": hint.get("registers_per_thread"),
            },
            "legacy_upstream_active_cta_per_sm": _compute_occupancy_legacy_upstream(
                smem_footprint, reg_footprint, warps_per_block, arch, mma_type
            ),
        }

    legacy = _compute_occupancy_legacy_upstream(
        smem_footprint, reg_footprint, warps_per_block, arch, mma_type
    )
    max_threads = getattr(arch, "max_resident_threads_per_sm", None)
    max_warps = getattr(arch, "max_resident_warps_per_sm", None)
    architectural_blocks = getattr(
        arch, "max_architectural_ctas_per_sm", None
    )
    if max_threads is None and max_warps is None and architectural_blocks is None:
        return {
            "active_cta_per_sm": legacy,
            "source": "legacy_upstream_resource_estimate",
            "confidence": "low",
            "limiting_resource": "legacy_min_smem_register_block",
            "constraints": {},
            "legacy_upstream_active_cta_per_sm": legacy,
        }

    threads_per_block = int(warps_per_block * 32)
    limits = {"legacy_resource_estimate": legacy}
    if max_threads is not None:
        limits["resident_threads"] = max(int(max_threads // threads_per_block), 1)
    if max_warps is not None:
        limits["resident_warps"] = max(int(max_warps // warps_per_block), 1)
    if architectural_blocks is not None:
        limits["architectural_cta"] = int(architectural_blocks)
    active = max(min(limits.values()), 1)
    if mma_type == "utcmma_cta2":
        active = max(active // 2, 1)
    limiting = min(limits, key=limits.get)
    return {
        "active_cta_per_sm": active,
        "source": "analytical_arch_and_tilesight_resources",
        "confidence": "medium" if limiting in {"resident_threads", "resident_warps"} else "low",
        "limiting_resource": limiting,
        "constraints": {
            **limits,
            "threads_per_block": threads_per_block,
            "warps_per_block": int(warps_per_block),
            "modeled_smem_footprint_bytes": smem_footprint,
            "modeled_register_footprint": reg_footprint,
        },
        "legacy_upstream_active_cta_per_sm": legacy,
    }


def compute_occupancy(
    smem_footprint,
    reg_footprint,
    warps_per_block,
    arch,
    mma_type="wmma",
    *,
    occupancy_hint=None,
    return_detail=False,
):
    """S5000扩展occupancy入口；上游算法保留为显式legacy函数。"""
    detail = compute_occupancy_detail(
        smem_footprint,
        reg_footprint,
        warps_per_block,
        arch,
        mma_type,
        occupancy_hint=occupancy_hint,
    )
    log.info("occupancy detail: %s", detail)
    return detail if return_detail else detail["active_cta_per_sm"]
