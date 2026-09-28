"""Source-visible builders for caller-defined stacked-GPU architectures.

The paper-facing reference profiles are packaged separately from this module.
This file contains only generic equations and validation so that artifact users
can substitute their own clocks, compute resources, cache sizes, and DRAM
interface assumptions without depending on any built-in calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

from tilesight.arch.arch_base import Arch


if TYPE_CHECKING:
    from mosaic.cost.energy import ChipEnergyConfig


__all__ = [
    "ConfigurableStackedGpu",
    "DramInterfaceConfig",
    "DramTimingConfig",
    "StackedGpuConfig",
    "dram_connectivity_efficiency",
    "make_custom_stacked_gpu",
]


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _finite_number(
    value: object,
    field: str,
    *,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{field} must be {relation}")
    return result


def _fraction(value: object, field: str) -> float:
    result = _finite_number(value, field, allow_zero=True)
    if result > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def dram_connectivity_efficiency(
    total_layers: int,
    connected_layers: int,
    *,
    fully_connected_efficiency: float,
) -> float:
    """Interpolate a caller-supplied fully-connected DRAM efficiency.

    The generic model assumes that recharge can be fully hidden when no more
    than half of the physical layers are connected.  Between that point and a
    fully connected stack it linearly interpolates to
    ``fully_connected_efficiency``.  Supplying the endpoint explicitly keeps
    the function reusable without embedding a paper-specific timing ratio.
    """

    total = _positive_int(total_layers, "total_layers")
    connected = _positive_int(connected_layers, "connected_layers")
    if connected > total:
        raise ValueError("connected_layers must not exceed total_layers")
    endpoint = _fraction(
        fully_connected_efficiency,
        "fully_connected_efficiency",
    )
    half = total / 2.0
    if connected <= half:
        return 1.0
    if connected >= total:
        return endpoint
    position = (connected - half) / (total - half)
    return 1.0 + position * (endpoint - 1.0)


@dataclass(frozen=True)
class DramTimingConfig:
    """Caller-owned DRAM row service and latency timing.

    This interface deliberately exposes physical construction inputs for
    custom studies without supplying a reference default. ``row_read_cycles``
    and the fully connected efficiency are derived from the caller's row,
    sector, and recharge values. Bank service cycles and round-trip latency
    cycles use independent caller-supplied clock domains.
    """

    row_bytes: int
    sector_bytes: int
    sector_cycles: int
    recharge_cycles: int
    round_trip_latency_cycles: float
    data_rate_hz: float
    latency_clock_hz: float

    def __post_init__(self) -> None:
        row = _positive_int(self.row_bytes, "row_bytes")
        sector = _positive_int(self.sector_bytes, "sector_bytes")
        if row % sector:
            raise ValueError("row_bytes must be an integer multiple of sector_bytes")
        _positive_int(self.sector_cycles, "sector_cycles")
        _nonnegative_int(self.recharge_cycles, "recharge_cycles")
        _finite_number(
            self.round_trip_latency_cycles,
            "round_trip_latency_cycles",
            allow_zero=True,
        )
        _finite_number(self.data_rate_hz, "data_rate_hz")
        _finite_number(self.latency_clock_hz, "latency_clock_hz")

    @property
    def sectors_per_row(self) -> int:
        return self.row_bytes // self.sector_bytes

    @property
    def row_read_cycles(self) -> int:
        return self.sectors_per_row * self.sector_cycles

    @property
    def full_row_cycles(self) -> int:
        return self.row_read_cycles + self.recharge_cycles

    @property
    def fully_connected_efficiency(self) -> float:
        return self.row_read_cycles / self.full_row_cycles

    @property
    def round_trip_latency_seconds(self) -> float:
        return self.round_trip_latency_cycles / self.latency_clock_hz


@dataclass(frozen=True)
class DramInterfaceConfig:
    """Explicit parameters for one configurable stacked-DRAM interface."""

    total_layers: int
    connected_layers: int
    channels_per_connected_layer: int
    bytes_per_channel_transfer: float | None
    transfers_per_memory_clock: float | None
    capacity_per_layer_bytes: int
    transaction_bytes: int
    fully_connected_efficiency: float | None
    uncached_max_utilization: float
    l2_bandwidth_multiplier: float = 1.0
    direct_peak_bandwidth_per_connected_layer_bytes_per_s: float | None = None
    bank_timing: DramTimingConfig | None = None

    def __post_init__(self) -> None:
        total = _positive_int(self.total_layers, "total_layers")
        connected = _positive_int(self.connected_layers, "connected_layers")
        if connected > total:
            raise ValueError("connected_layers must not exceed total_layers")
        _positive_int(
            self.channels_per_connected_layer,
            "channels_per_connected_layer",
        )
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is None:
            if (
                self.bytes_per_channel_transfer is None
                or self.transfers_per_memory_clock is None
            ):
                raise ValueError(
                    "transfer composition requires bytes_per_channel_transfer "
                    "and transfers_per_memory_clock"
                )
            _finite_number(
                self.bytes_per_channel_transfer,
                "bytes_per_channel_transfer",
            )
            _finite_number(
                self.transfers_per_memory_clock,
                "transfers_per_memory_clock",
            )
        else:
            _finite_number(
                direct_peak,
                "direct_peak_bandwidth_per_connected_layer_bytes_per_s",
            )
            if (
                self.bytes_per_channel_transfer is not None
                or self.transfers_per_memory_clock is not None
            ):
                raise ValueError(
                    "direct peak bandwidth and transfer composition are "
                    "mutually exclusive"
                )
        _positive_int(self.capacity_per_layer_bytes, "capacity_per_layer_bytes")
        _positive_int(self.transaction_bytes, "transaction_bytes")
        if self.bank_timing is None:
            if self.fully_connected_efficiency is None:
                raise ValueError(
                    "supply fully_connected_efficiency or bank_timing"
                )
            _fraction(
                self.fully_connected_efficiency,
                "fully_connected_efficiency",
            )
        else:
            if not isinstance(self.bank_timing, DramTimingConfig):
                raise TypeError("bank_timing must be a DramTimingConfig")
            if self.fully_connected_efficiency is not None:
                raise ValueError(
                    "fully_connected_efficiency and bank_timing are "
                    "mutually exclusive"
                )
        _fraction(
            self.uncached_max_utilization,
            "uncached_max_utilization",
        )
        _finite_number(
            self.l2_bandwidth_multiplier,
            "l2_bandwidth_multiplier",
        )

    @property
    def resolved_fully_connected_efficiency(self) -> float:
        if self.bank_timing is not None:
            return self.bank_timing.fully_connected_efficiency
        assert self.fully_connected_efficiency is not None
        return float(self.fully_connected_efficiency)

    def peak_bandwidth_bytes_per_s(
        self,
        memory_frequency_hz: float,
        *,
        connected_layers: int | None = None,
    ) -> float:
        connected = (
            self.connected_layers
            if connected_layers is None
            else _positive_int(connected_layers, "connected_layers")
        )
        memory_frequency = _finite_number(
            memory_frequency_hz,
            "memory_frequency_hz",
        )
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is not None:
            return connected * direct_peak
        assert self.bytes_per_channel_transfer is not None
        assert self.transfers_per_memory_clock is not None
        return (
            connected
            * self.channels_per_connected_layer
            * self.bytes_per_channel_transfer
            * self.transfers_per_memory_clock
            * memory_frequency
        )


@dataclass(frozen=True)
class StackedGpuConfig:
    """Complete caller-owned input for :class:`ConfigurableStackedGpu`."""

    name: str
    sm_count: int
    core_frequency_hz: float
    memory_frequency_hz: float
    noc_frequency_hz: float
    tensor_cores_per_sm: int
    tensor_core_shape: tuple[int, int, int]
    fp32_cores_per_sm: int
    int32_cores_per_sm: int
    sfu_cores_per_sm: int
    sm_sub_partitions: int
    shared_memory_throughput_bytes_per_cycle: float
    shared_memory_capacity_bytes: int
    register_capacity_per_sm_bytes: int
    warp_schedulers_per_sm: int
    l2_capacity_bytes: int
    l1_noc_bytes_per_cycle: float
    dram: DramInterfaceConfig
    tensor_instruction_shapes: Mapping[float, tuple[int, int, int]]
    # Deprecated compatibility input.  New configurations should describe
    # every special-math operation through ``special_math_service_profiles``.
    log_sequence_ops_per_s: float | None = None
    chip_energy_config: "ChipEnergyConfig | None" = None
    ddr_max_utilization: float = 0.9
    l2_max_utilization: float = 0.9
    l1_max_utilization: float = 0.9
    compute_max_utilization: float = 0.9
    support_wgmma: bool = False
    support_utcmma: bool = False
    use_tensor_core_resource_model: bool = False
    apply_dram_wave_quantization: bool = False
    # Ordinary Global-load L1 cache.  This is deliberately separate from
    # Shared Memory: ``l1_max_utilization`` above is retained for the existing
    # Shared/SMEM resource model and must not be reused as an L1-cache value.
    l1_cache_capacity_bytes_per_sm: int | None = None
    l1_cache_line_bytes: int | None = None
    l1_cache_sector_bytes: int | None = None
    l1_cache_peak_bandwidth_bytes_per_s: float | None = None
    l1_cache_max_utilization: float | None = None
    tma_bypasses_ordinary_l1: bool = False
    # Kernel residency limits.  These are separate from max threads/block and
    # from the device property's architectural CTA ceiling.  When absent,
    # TileSight keeps the upstream legacy occupancy calculation available.
    max_resident_threads_per_sm: int | None = None
    max_resident_warps_per_sm: int | None = None
    # Deliberately use a new name.  The upstream TileSight occupancy function
    # probes ``max_blocks_per_sm`` and otherwise defaults to 24; defining that
    # old attribute here would mutate the preserved legacy result.
    max_architectural_ctas_per_sm: int | None = None
    # Optional empirical Log timing adapter. Disabled preserves upstream math.
    cta_overlap_adapter_enabled: bool = False
    cta_overlap_eta_inf: float = 0.0
    cta_overlap_startup_loss: float = 0.0
    cta_overlap_fixed_overhead_us: float = 0.0
    # Deprecated compatibility input for removed operation-specific CTA
    # adapters.  It is retained for old callers but is not used by the model.
    special_math_cta_overlap_profiles: Mapping[str, Mapping[str, object]] | None = None
    # Unified per-operation SFU service description.  A profile may provide a
    # measured sustained semantic-operation rate.  Missing rates retain the
    # ordinary SFU peak/utilisation path.
    special_math_service_profiles: Mapping[str, Mapping[str, object]] | None = None
    ordinary_fp16_unary_dirty_retention_bytes: int | None = None
    # Explicit dtype/instruction profiles.  Legacy callers leave this empty
    # and retain byte-width based behavior.
    compute_profiles: Mapping[str, Mapping[str, object]] | None = None
    enable_bf16_profiles: bool = False
    # How Elementwise converts resident CTA capacity into kernel duration.
    # The default preserves upstream's repeated analytical-wave behavior.
    elementwise_wave_dispatch_mode: str = "repeated_resident_batch"
    elementwise_wave_fixed_head_us: float = 0.0
    # Generic cache-state service slope for the opt-in continuous CTA path.
    # A value of 1.0 preserves the upstream memory-time calculation.
    elementwise_wave_memory_service_scales: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.elementwise_wave_dispatch_mode not in {
                "repeated_resident_batch", "continuous_cta_dispatch"}:
            raise ValueError(
                "elementwise_wave_dispatch_mode must be "
                "'repeated_resident_batch' or 'continuous_cta_dispatch'"
            )
        if (not isinstance(self.elementwise_wave_fixed_head_us, Real)
                or not math.isfinite(self.elementwise_wave_fixed_head_us)
                or self.elementwise_wave_fixed_head_us < 0):
            raise ValueError(
                "elementwise_wave_fixed_head_us must be finite and nonnegative"
            )
        if self.elementwise_wave_memory_service_scales is not None:
            if not isinstance(self.elementwise_wave_memory_service_scales, Mapping):
                raise TypeError(
                    "elementwise_wave_memory_service_scales must be a mapping"
                )
            for cache_condition, scale in self.elementwise_wave_memory_service_scales.items():
                if cache_condition not in {"cold", "warm"}:
                    raise ValueError(
                        "elementwise memory service scales support only cold/warm"
                    )
                if (not isinstance(scale, Real) or not math.isfinite(scale)
                        or scale <= 0):
                    raise ValueError(
                        "elementwise memory service scales must be finite and positive"
                    )
        if not isinstance(self.enable_bf16_profiles, bool):
            raise ValueError('enable_bf16_profiles must be bool')
        if self.compute_profiles is not None:
            if not isinstance(self.compute_profiles, Mapping):
                raise TypeError('compute_profiles must be a mapping')
            for key, profile in self.compute_profiles.items():
                if not isinstance(key, str) or not key or not isinstance(profile, Mapping):
                    raise ValueError('compute profile keys must be non-empty strings and values mappings')
                alias_of = profile.get('alias_of')
                if alias_of is not None:
                    if not isinstance(alias_of, str) or not alias_of:
                        raise ValueError(f'compute profile alias {key!r} requires a non-empty alias_of')
                    continue
                if profile.get('resource') not in {'tensor_core', 'vector_alu', 'sfu', 'int_alu'}:
                    raise ValueError(f'invalid compute profile resource for {key!r}')
                rate = profile.get('peak_ops_per_s')
                if not isinstance(rate, Real) or not math.isfinite(rate) or rate <= 0:
                    raise ValueError(f'compute profile {key!r} requires positive peak_ops_per_s')
                util = profile.get('max_utilization', 1.0)
                if not isinstance(util, Real) or not math.isfinite(util) or not 0 < util <= 1:
                    raise ValueError(f'compute profile {key!r} max_utilization must be in (0, 1]')
            for key, profile in self.compute_profiles.items():
                alias_of = profile.get('alias_of')
                if alias_of is not None:
                    target = self.compute_profiles.get(alias_of)
                    if target is None or target.get('alias_of') is not None:
                        raise ValueError(
                            f'compute profile alias {key!r} must reference a concrete profile'
                        )
        if not isinstance(self.cta_overlap_adapter_enabled, bool):
            raise ValueError('cta_overlap_adapter_enabled must be bool')
        if not math.isfinite(self.cta_overlap_eta_inf) or not 0 <= self.cta_overlap_eta_inf <= 1:
            raise ValueError('cta_overlap_eta_inf must be finite and in [0, 1]')
        if not math.isfinite(self.cta_overlap_startup_loss) or self.cta_overlap_startup_loss < 0:
            raise ValueError('cta_overlap_startup_loss must be finite and nonnegative')
        if not math.isfinite(self.cta_overlap_fixed_overhead_us) or self.cta_overlap_fixed_overhead_us < 0:
            raise ValueError('cta_overlap_fixed_overhead_us must be finite and nonnegative')
        if self.log_sequence_ops_per_s is not None and (
                not math.isfinite(self.log_sequence_ops_per_s) or self.log_sequence_ops_per_s <= 0):
            raise ValueError('log_sequence_ops_per_s must be finite and positive')
        if self.special_math_cta_overlap_profiles is not None:
            if not isinstance(self.special_math_cta_overlap_profiles, Mapping):
                raise TypeError('special_math_cta_overlap_profiles must be a mapping')
            for operation, profile in self.special_math_cta_overlap_profiles.items():
                if operation not in {'exp', 'sqrt'} or not isinstance(profile, Mapping):
                    raise ValueError('special-math CTA profiles support only exp and sqrt mappings')
                if not isinstance(profile.get('enabled', False), bool):
                    raise ValueError('special-math CTA profile enabled must be bool')
                eta = profile.get('eta_inf', 0.0)
                loss = profile.get('startup_loss', 0.0)
                overhead = profile.get('fixed_overhead_us', 0.0)
                if not isinstance(eta, Real) or not math.isfinite(eta) or not 0 <= eta <= 1:
                    raise ValueError('special-math CTA eta_inf must be finite and in [0, 1]')
                if not isinstance(loss, Real) or not math.isfinite(loss) or loss < 0:
                    raise ValueError('special-math CTA startup_loss must be finite and nonnegative')
                if not isinstance(overhead, Real) or not math.isfinite(overhead) or overhead < 0:
                    raise ValueError('special-math CTA fixed_overhead_us must be finite and nonnegative')
        if self.special_math_service_profiles is not None:
            if not isinstance(self.special_math_service_profiles, Mapping):
                raise TypeError('special_math_service_profiles must be a mapping')
            for operation, profile in self.special_math_service_profiles.items():
                if operation not in {'exp', 'log', 'sqrt'} or not isinstance(profile, Mapping):
                    raise ValueError(
                        'special-math service profiles support only exp, log, and sqrt mappings'
                    )
                rate = profile.get('effective_service_rate_ops_per_s')
                if rate is not None and (
                        not isinstance(rate, Real) or not math.isfinite(rate) or rate <= 0):
                    raise ValueError(
                        'special-math effective service rate must be finite and positive'
                    )
                overlap = profile.get('cta_overlap', {})
                if not isinstance(overlap, Mapping):
                    raise TypeError('special-math cta_overlap must be a mapping')
                if not isinstance(overlap.get('enabled', False), bool):
                    raise ValueError('special-math CTA profile enabled must be bool')
                eta = overlap.get('eta_inf', 0.0)
                loss = overlap.get('startup_loss', 0.0)
                overhead = overlap.get('fixed_overhead_us', 0.0)
                if not isinstance(eta, Real) or not math.isfinite(eta) or not 0 <= eta <= 1:
                    raise ValueError('special-math CTA eta_inf must be finite and in [0, 1]')
                if not isinstance(loss, Real) or not math.isfinite(loss) or loss < 0:
                    raise ValueError('special-math CTA startup_loss must be finite and nonnegative')
                if not isinstance(overhead, Real) or not math.isfinite(overhead) or overhead < 0:
                    raise ValueError('special-math CTA fixed_overhead_us must be finite and nonnegative')
        if self.ordinary_fp16_unary_dirty_retention_bytes is not None:
            _positive_int(
                self.ordinary_fp16_unary_dirty_retention_bytes,
                'ordinary_fp16_unary_dirty_retention_bytes',
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for field in (
            "sm_count",
            "tensor_cores_per_sm",
            "fp32_cores_per_sm",
            "int32_cores_per_sm",
            "sfu_cores_per_sm",
            "sm_sub_partitions",
            "shared_memory_capacity_bytes",
            "register_capacity_per_sm_bytes",
            "warp_schedulers_per_sm",
            "l2_capacity_bytes",
        ):
            _positive_int(getattr(self, field), field)
        for field in (
            "core_frequency_hz",
            "memory_frequency_hz",
            "noc_frequency_hz",
            "shared_memory_throughput_bytes_per_cycle",
            "l1_noc_bytes_per_cycle",
        ):
            _finite_number(getattr(self, field), field)
        if (
            not isinstance(self.tensor_core_shape, tuple)
            or len(self.tensor_core_shape) != 3
        ):
            raise TypeError("tensor_core_shape must be a three-integer tuple")
        for dimension in self.tensor_core_shape:
            _positive_int(dimension, "tensor_core_shape dimension")
        if not isinstance(self.tensor_instruction_shapes, Mapping):
            raise TypeError("tensor_instruction_shapes must be a mapping")
        if not self.tensor_instruction_shapes:
            raise ValueError("tensor_instruction_shapes must not be empty")
        for bytes_per_element, shape in self.tensor_instruction_shapes.items():
            _finite_number(bytes_per_element, "tensor instruction byte width")
            if not isinstance(shape, tuple) or len(shape) != 3:
                raise TypeError(
                    "every tensor instruction shape must be a three-integer tuple"
                )
            for dimension in shape:
                _positive_int(
                    dimension,
                    "tensor instruction shape dimension",
                )
        if self.chip_energy_config is not None:
            from mosaic.cost.energy import ChipEnergyConfig

            if not isinstance(self.chip_energy_config, ChipEnergyConfig):
                raise TypeError(
                    "chip_energy_config must be a ChipEnergyConfig or None"
                )
        for field in (
            "ddr_max_utilization",
            "l2_max_utilization",
            "l1_max_utilization",
            "compute_max_utilization",
        ):
            _fraction(getattr(self, field), field)
        for field in (
            "l1_cache_capacity_bytes_per_sm",
            "l1_cache_line_bytes",
            "l1_cache_sector_bytes",
            "max_resident_threads_per_sm",
            "max_resident_warps_per_sm",
            "max_architectural_ctas_per_sm",
        ):
            value = getattr(self, field)
            if value is not None:
                _positive_int(value, field)
        if self.l1_cache_peak_bandwidth_bytes_per_s is not None:
            _finite_number(
                self.l1_cache_peak_bandwidth_bytes_per_s,
                "l1_cache_peak_bandwidth_bytes_per_s",
            )
        if self.l1_cache_max_utilization is not None:
            _fraction(self.l1_cache_max_utilization, "l1_cache_max_utilization")
        if not isinstance(self.support_wgmma, bool):
            raise TypeError("support_wgmma must be a bool")
        if not isinstance(self.support_utcmma, bool):
            raise TypeError("support_utcmma must be a bool")
        if not isinstance(self.use_tensor_core_resource_model, bool):
            raise TypeError("use_tensor_core_resource_model must be a bool")
        if not isinstance(self.apply_dram_wave_quantization, bool):
            raise TypeError("apply_dram_wave_quantization must be a bool")
        if not isinstance(self.tma_bypasses_ordinary_l1, bool):
            raise TypeError("tma_bypasses_ordinary_l1 must be a bool")


class ConfigurableStackedGpu(Arch):
    """Architecture object derived only from an explicit user configuration."""

    def __init__(self, config: StackedGpuConfig):
        if not isinstance(config, StackedGpuConfig):
            raise TypeError("config must be a StackedGpuConfig")
        super().__init__()
        self.config = config
        self.core = config.name
        self.sm_count = config.sm_count
        self.core_freq = config.core_frequency_hz
        self.memory_freq = config.memory_frequency_hz
        self.noc_freq = config.noc_frequency_hz
        self.base_freq = self.core_freq
        self.max_freq = self.core_freq
        self.tensor_cores_per_sm = config.tensor_cores_per_sm
        self.tensor_core_shape = config.tensor_core_shape
        self.fp32_cores_per_sm = config.fp32_cores_per_sm
        self.int32_cores_per_sm = config.int32_cores_per_sm
        self.sfu_cores_per_sm = config.sfu_cores_per_sm
        profiles = {
            key: dict(value)
            for key, value in (config.special_math_service_profiles or {}).items()
        }
        # Normalize the measured SFU service rate.  Deprecated CTA-overlap
        # fields are intentionally not copied into runtime profiles.
        if config.log_sequence_ops_per_s is not None:
            profiles.setdefault('log', {}).setdefault(
                'effective_service_rate_ops_per_s', config.log_sequence_ops_per_s
            )
            profiles['log'].setdefault('rate_kind', 'legacy_measured_sustained_sequence')

        self.special_math_service_profiles = profiles
        # Deprecated attributes remain readable for one compatibility cycle.
        log_profile = profiles.get('log', {})
        self.log_sequence_ops_per_s = log_profile.get(
            'effective_service_rate_ops_per_s'
        )
        self.cta_overlap_adapter_enabled = False
        self.cta_overlap_eta_inf = 0.0
        self.cta_overlap_startup_loss = 0.0
        self.cta_overlap_fixed_overhead_us = 0.0
        self.special_math_cta_overlap_profiles = {}
        self.ordinary_fp16_unary_dirty_retention_bytes = (
            config.ordinary_fp16_unary_dirty_retention_bytes
        )
        self.compute_profiles = {
            key: dict(value) for key, value in (config.compute_profiles or {}).items()
        }
        self.enable_bf16_profiles = config.enable_bf16_profiles
        self.elementwise_wave_dispatch_mode = (
            config.elementwise_wave_dispatch_mode
        )
        self.elementwise_wave_fixed_head_us = (
            float(config.elementwise_wave_fixed_head_us)
        )
        self.elementwise_wave_memory_service_scales = {
            key: float(value)
            for key, value in (config.elementwise_wave_memory_service_scales or {}).items()
        }
        self.sm_sub_partitions = config.sm_sub_partitions
        self.l1_smem_throughput_per_cycle = (
            config.shared_memory_throughput_bytes_per_cycle
        )
        self.configurable_smem_capacity = config.shared_memory_capacity_bytes
        self.register_capacity_per_sm = config.register_capacity_per_sm_bytes
        self.warp_schedulers_per_sm = config.warp_schedulers_per_sm
        self.l2_capacity = config.l2_capacity_bytes
        self._l1_noc_bytes_per_cycle = config.l1_noc_bytes_per_cycle
        self._tensor_instruction_shapes = dict(config.tensor_instruction_shapes)
        self.ddr_max_util = config.ddr_max_utilization
        self.l2_max_util = config.l2_max_utilization
        self.l1_max_util = config.l1_max_utilization
        self.compute_max_util = config.compute_max_utilization
        self.support_wgmma = config.support_wgmma
        self.support_utcmma = config.support_utcmma
        self.use_tensor_core_resource_model = (
            config.use_tensor_core_resource_model
        )
        self.apply_dram_wave_quantization = (
            config.apply_dram_wave_quantization
        )
        self.l1_cache_capacity = config.l1_cache_capacity_bytes_per_sm
        self.l1_cache_line_bytes = config.l1_cache_line_bytes
        self.l1_cache_sector_bytes = config.l1_cache_sector_bytes
        self.l1_cache_bandwidth = config.l1_cache_peak_bandwidth_bytes_per_s
        self.l1_cache_max_util = config.l1_cache_max_utilization
        self.tma_bypasses_ordinary_l1 = config.tma_bypasses_ordinary_l1
        self.max_resident_threads_per_sm = config.max_resident_threads_per_sm
        self.max_resident_warps_per_sm = config.max_resident_warps_per_sm
        self.max_architectural_ctas_per_sm = (
            config.max_architectural_ctas_per_sm
        )
        self.chip_energy_config = config.chip_energy_config
        self.ddr_transaction_size = config.dram.transaction_bytes
        self.ddr_stack = config.dram.channels_per_connected_layer
        self.dram_3d_uncached_max_util = (
            config.dram.uncached_max_utilization
        )
        timing = config.dram.bank_timing
        if timing is not None:
            self.dram_row_bytes = timing.row_bytes
            self.dram_sector_bytes = timing.sector_bytes
            self.dram_sector_cycles = timing.sector_cycles
            self.dram_recharge_cycles = timing.recharge_cycles
            self.dram_round_trip_latency_cycles = (
                timing.round_trip_latency_cycles
            )
            self.dram_data_rate_hz = timing.data_rate_hz
            self.dram_latency_clock_hz = timing.latency_clock_hz
        self.update_ddr(
            config.dram.total_layers,
            config.dram.connected_layers,
        )
        self.update_derived()

    def update_ddr(
        self,
        total_layers: int,
        active_layers: int | None = None,
    ) -> None:
        total = _positive_int(total_layers, "total_layers")
        connected = (
            total
            if active_layers is None
            else _positive_int(active_layers, "active_layers")
        )
        if connected > total:
            raise ValueError("active_layers must not exceed total_layers")
        dram = self.config.dram
        self.dram_layers_per_cluster = total
        self.dram_active_layers = connected
        self.ddr_peak_bandwidth = dram.peak_bandwidth_bytes_per_s(
            self.memory_freq,
            connected_layers=connected,
        )
        efficiency = dram_connectivity_efficiency(
            total,
            connected,
            fully_connected_efficiency=(
                dram.resolved_fully_connected_efficiency
            ),
        )
        self.ddr_bandwidth = self.ddr_peak_bandwidth * efficiency
        self.ddr_capacity = total * dram.capacity_per_layer_bytes
        self.l2_bandwidth = (
            self.ddr_peak_bandwidth * dram.l2_bandwidth_multiplier
        )
        self.ddr_wave_bytes = self.ddr_transaction_size * self.ddr_stack

    def update_derived(self) -> None:
        self.layer1_noc_size = self.sm_count
        self.layer1_noc = "xbar"
        self.layer1_noc_single_direction_bw = (
            self._l1_noc_bytes_per_cycle * self.noc_freq
        )
        self.tensor_core_flops = math.prod(self.tensor_core_shape) * 2
        self.fp16_tensor_flops = (
            self.sm_count
            * self.max_freq
            * self.tensor_cores_per_sm
            * self.tensor_core_flops
        )
        self.fp32_tensor_flops = (
            self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        )
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.fp32_tensor_flops
        self.fp16_cuda_core_flops = self.fp32_tensor_flops
        self.fp8_cuda_core_flops = self.fp32_tensor_flops
        self.fp64_cuda_core_flops = self.fp32_tensor_flops * 0.5
        self.int32_cuda_core_flops = (
            self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        )
        self.sfu_flops = (
            self.sm_count * self.max_freq * self.sfu_cores_per_sm
        )
        self.smem_bandwidth = (
            self.sm_count
            * self.max_freq
            * self.l1_smem_throughput_per_cycle
        )
        self.smem_l2_bandwidth = self.smem_bandwidth
        self.l2_to_smem_bandwidth = self.smem_bandwidth * 0.5
        self.smem_to_l2_bandwidth = self.smem_bandwidth * 0.5
        self.smem_register_bandwidth = self.smem_bandwidth
        self.register_bandwidth = (
            self.sm_count
            * self.max_freq
            * self.sm_sub_partitions
            * 32
            * 4
        )

    def get_tensor_core_minimum_ptx(
        self,
        bytes: float = 2,
    ) -> tuple[int, int, int]:
        try:
            return self._tensor_instruction_shapes[bytes]
        except KeyError as exc:
            choices = ", ".join(
                str(value) for value in sorted(self._tensor_instruction_shapes)
            )
            raise ValueError(
                f"unsupported tensor element width {bytes}; choose one of: {choices}"
            ) from exc

    def resolve_compute_profile(self, key: str) -> dict[str, object]:
        if key.startswith("bf16_") and not self.enable_bf16_profiles:
            raise ValueError(f"BF16 compute profile {key!r} is disabled")
        try:
            profile = dict(self.compute_profiles[key])
        except KeyError as exc:
            raise ValueError(f"missing compute profile {key!r}") from exc
        alias_of = profile.get("alias_of")
        if alias_of is not None:
            profile = dict(self.compute_profiles[alias_of])
            profile["resolved_from_alias"] = key
            profile["canonical_key"] = alias_of
        return profile

    def get_tensor_instruction_shape(self, key: str) -> tuple[int, int, int]:
        profile = self.resolve_compute_profile(key)
        shape = profile.get("instruction_shape")
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError(f"compute profile {key!r} has no instruction_shape")
        return tuple(int(value) for value in shape)

    def set_to_spec(self) -> "ConfigurableStackedGpu":
        return self

    def set_to_microbench(self) -> "ConfigurableStackedGpu":
        return self

    def set_to_ncu(self) -> "ConfigurableStackedGpu":
        return self


def make_custom_stacked_gpu(config: StackedGpuConfig) -> ConfigurableStackedGpu:
    """Build a configurable architecture without loading reference defaults."""

    return ConfigurableStackedGpu(config)
