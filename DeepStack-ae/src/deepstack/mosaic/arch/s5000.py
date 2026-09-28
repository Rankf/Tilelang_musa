"""S5000 GPU architecture profile for DeepStack / TileSight.

Calibrated on a real MTT S5000 device (PH100 chip, PingHu architecture)
through MUSA Runtime API queries, micro-benchmarks, and TileLang-MUSA source
inspection (commit 4d666968).

This module provides the recommended ``StackedGpuConfig``-based configuration
for the S5000.  Use ``make_s5000_config()`` with ``make_custom_stacked_gpu()``
to build a fully typed architecture object:

    >>> from deepstack.mosaic.arch import make_s5000_config, make_custom_stacked_gpu
    >>> gpu = make_custom_stacked_gpu(make_s5000_config())

A convenience callable ``S5000()`` is also provided, returning the same
``ConfigurableStackedGpu`` instance.

Verified parameters:
- sm_count (musaGetDeviceProperties)
- l2_capacity (musaGetDeviceProperties, 60 MiB)
- register_capacity_per_sm (musaGetDeviceProperties, 131072 32-bit regs)
- shared_memory_capacity (musaGetDeviceProperties, 192 KiB)
- core_freq (musaGetDeviceProperties clockRate = 1.5 GHz base)
- memory_freq (musaGetDeviceProperties memoryClockRate = 20 GHz effective)
- memory_bus_width (musaGetDeviceProperties = 640 bits)
- bandwidth verified via fill: 1401 GB/s / 1600 theoretical = 87.6% efficiency
- fp32_cores_per_sm (pure FP32 FMA microbench: 22.7 TF / 23.0 TF = 98.5%)
- tensor_core_shape (TileLang-MUSA mma.h: MP31_16x8x8_F32F16F16F32)
- tensor_cores_per_sm (FP16 GEMM microbench: 153.4 TF / 184 TF = 83.3%)

P8 measured parameters (2026-08-07):
- sm_sub_partitions (verified via 16-way independent FMA peak: 22.99 TF = 99.96% of 23.0 TF)
- warp_schedulers_per_sm (same as sm_sub_partitions on S5000)
- sfu_cores_per_sm (ILP-saturated expf: 645 Gops/s, confirmed ~8 cores/SM)
- l1_smem_throughput_per_cycle (shared memory sequential read: 11.0 TB/s, 122 bytes/cycle/SM, confirmed ~128)
- ordinary Global-load L1 effective capacity (32 KiB/MP), line candidate
  (64 B), sector candidate (32 B), and MCU physical peak (7168 GB/s)
- TMA Global-to-Shared payload bypasses the MCU ordinary-L1 counter path
- ddr_max_util (fill microbench: 1401/1600 = 87.6%)
- compute_max_util (Tensor Core GEMM: 153.4/184 = 83.2%; single value used for all compute)
- int32_cores_per_sm (inconclusive: MAD 4.35 TOP/s, ADD 5.62 TOP/s; kept at NVIDIA-like 64)
- l2 peak bandwidth (MCU l2pu cache-bandwidth metric, 7168 GB/s)
- l2_max_util (P8 sustained read bandwidth, ~2450 GB/s / 7168 GB/s)
- l1_max_util (P8 shared-memory throughput)
"""

from .custom_profile import (
    ConfigurableStackedGpu,
    DramInterfaceConfig,
    StackedGpuConfig,
    make_custom_stacked_gpu,
)

__all__ = ["make_s5000_config", "S5000"]


def make_s5000_config(*, enable_log_adapter: bool = True,
                      enable_exp_sqrt_adapters: bool = True,
                      enable_bf16_profiles: bool = True,
                      enable_continuous_elementwise_wave: bool = False
                      ) -> StackedGpuConfig:
    """Return a complete ``StackedGpuConfig`` for the MTT S5000.

    All values are calibrated on a real MTT S5000 device (PH100 chip, PingHu
    architecture) via MUSA Runtime API queries, micro-benchmarks (P8), and
    TileLang-MUSA source inspection.

    The S5000 uses GDDR7 (640-bit, 20 GHz effective, 1600 GB/s theoretical),
    which is a non-stacked DRAM configuration. The ``DramInterfaceConfig`` is
    therefore built with a single layer and the measured direct peak bandwidth.
    """

    # ``enable_log_adapter`` and ``enable_exp_sqrt_adapters`` are retained as
    # source-compatible no-op arguments.  Operation-specific CTA timing
    # corrections were removed; the SFU service-rate framework remains active.
    del enable_log_adapter, enable_exp_sqrt_adapters

    # --- GDDR7 DRAM interface (non-stacked, single layer) ---
    dram = DramInterfaceConfig(
        total_layers=1,                 # not 3D-stacked; standard GDDR7
        connected_layers=1,
        channels_per_connected_layer=1,  # single interface (direct peak used)
        bytes_per_channel_transfer=None,  # use direct peak bandwidth below
        transfers_per_memory_clock=None,
        # 640-bit x 20 GHz / 8 = 1600 GB/s theoretical peak
        direct_peak_bandwidth_per_connected_layer_bytes_per_s=1600e9,
        capacity_per_layer_bytes=80 * (1024 ** 3),   # 80 GB
        transaction_bytes=128,                       # cache line size
        fully_connected_efficiency=1.0,              # single layer, always connected
        uncached_max_utilization=0.876,              # P8: fill 1401/1600
        # MCU reports a 7168 GB/s physical L2 peak.  Sustained bandwidth is
        # represented separately by l2_max_utilization below.
        l2_bandwidth_multiplier=4.48,                # 7168/1600 GB/s
        bank_timing=None,
    )

    # --- Tensor instruction shapes per element width (MP31) ---
    tensor_shapes = {
        2.0: (16, 8, 8),    # FP16
        4.0: (16, 8, 4),    # TF32
        1.0: (16, 8, 16),   # INT8 / FP8
        0.5: (16, 8, 32),   # FP4
    }

    return StackedGpuConfig(
        # Core configuration
        name="S5000",
        sm_count=60,                        # musaGetDeviceProperties
        core_frequency_hz=1.5e9,            # musaGetDeviceProperties clockRate
        memory_frequency_hz=20.0e9,         # effective data rate (GDDR7)
        noc_frequency_hz=1.5e9,             # not directly measured; = core_freq

        # Compute units
        tensor_cores_per_sm=1,              # TileLang-MUSA mma.h
        tensor_core_shape=(16, 8, 8),       # MP31_16x8x8_F32F16F16F32
        fp32_cores_per_sm=128,              # FP32 FMA microbench (22.7/23.0 = 98.5%)
        int32_cores_per_sm=64,              # NOT measured; NVIDIA-like estimate
        sfu_cores_per_sm=8,                 # CONFIRMED from ILP-saturated SFU data
        # FP16 elementwise SFU service profiles.  These are semantic-operation
        # rates fitted against the first-batch MCU kernel-time measurements
        # after enabling the common continuous-CTA dispatch path.  They are
        # not physical SFU peak rates.  BF16 uses the dtype-specific
        # ``compute_profiles`` below and is therefore unaffected by this
        # FP16 calibration.
        special_math_service_profiles={
            "log": {
                "effective_service_rate_ops_per_s": 200.0e9,
                "rate_kind": "first_batch_fp16_continuous_cta_fit",
                "value_kind": "calibrated_model_parameter",
                "sequence_kind": "fp16_to_fp32_log_mixed_sequence",
                "semantic_ops_per_element": 1,
            },
            "exp": {
                "effective_service_rate_ops_per_s": 350.0e9,
                "rate_kind": "first_batch_fp16_continuous_cta_fit",
                "value_kind": "calibrated_model_parameter",
                "sequence_kind": "fp16_to_fp32_exp_mixed_sequence",
                "semantic_ops_per_element": 1,
            },
            "sqrt": {
                "effective_service_rate_ops_per_s": 470.0e9,
                "rate_kind": "first_batch_fp16_continuous_cta_fit",
                "value_kind": "calibrated_model_parameter",
                "sequence_kind": "fp16_to_fp32_sqrt_mixed_sequence",
                "semantic_ops_per_element": 1,
            },
        },
        # No operation-specific dirty-retention correction is applied. The
        # common writeback model owns this behavior for every dtype/operator.
        ordinary_fp16_unary_dirty_retention_bytes=None,

        # SM internals
        sm_sub_partitions=4,                # P8: 16-way FMA 99.96% peak
        shared_memory_throughput_bytes_per_cycle=128.0,  # P8: 122 bytes/cycle measured
        shared_memory_capacity_bytes=192 * 1024,         # 192 KiB (API)
        register_capacity_per_sm_bytes=131072 * 4,       # 131072 regs x 4 bytes
        warp_schedulers_per_sm=4,           # P8: same as sm_sub_partitions
        l2_capacity_bytes=60 * (1024 ** 2), # 60 MiB (API)

        # NoC / interconnect
        l1_noc_bytes_per_cycle=128.0,       # layer1_noc_single_direction_bw / noc_freq

        # DRAM + memory hierarchy
        dram=dram,
        tensor_instruction_shapes=tensor_shapes,
        enable_bf16_profiles=enable_bf16_profiles,
        # Exact 720-CTA boundary probes show that resident capacity is a
        # concurrency window: CTAs are replenished continuously and do not
        # force a complete pipeline drain/refill at every analytical wave.
        elementwise_wave_dispatch_mode=(
            "continuous_cta_dispatch"
            if enable_continuous_elementwise_wave
            else "repeated_resident_batch"
        ),
        # Median shared head residual from the randomized L2-resident sweep
        # across FP16/BF16 Add, ReLU, Exp and Sqrt.  This is one kernel-level
        # term, never a per-wave or per-dtype correction.
        elementwise_wave_fixed_head_us=6.5,
        # Cold continuous-CTA memory slope calibration.  This scales service
        # time only; logical read/write bytes and the upstream max(memory,
        # compute) rule are unchanged.  Development fit=0.596, holdout
        # optimum=0.612, rounded to one generic cache-state parameter.
        elementwise_wave_memory_service_scales={"warm": 1.0, "cold": 0.60},
        compute_profiles={
            # MP31 BF16 GEMM uses FP32 accumulation.  The matched P2 BF16 and
            # FP16 schedules measured equivalent throughput; retain the
            # physical MP31 tensor rate and the existing sustained ceiling.
            "bf16_mma_bf16_bf16_f32": {
                "resource": "tensor_core",
                "peak_ops_per_s": 184.32e12,
                "max_utilization": 0.832,
                "instruction_shape": (16, 8, 8),
                "evidence": "S5000 P2 matched FP16/BF16 GEMM schedules",
            },
            # Add and compare/select share one dtype/resource profile, matching
            # the legacy FP16 vector model.  Operator differences remain in
            # compute_ops_per_element rather than creating per-op hardware
            # capacities.  The paired service probe and exact TileLang
            # operator holdout did not identify a safe replacement for this
            # shared capacity, so the generic rate is retained.
            "bf16_vector_alu": {
                "resource": "vector_alu",
                "peak_ops_per_s": 23.04e12,
                "max_utilization": 0.985,
                "evidence": (
                    "MP31 generic vector capacity retained after paired BF16 "
                    "service and exact TileLang operator holdout validation"
                ),
            },
            # Compatibility-only aliases for requests produced by the first
            # BF16 adapter revision.  New requests emit bf16_vector_alu.
            "bf16_vector_add": {"alias_of": "bf16_vector_alu"},
            "bf16_vector_compare_select": {
                "alias_of": "bf16_vector_alu",
            },
            "bf16_exp_f32_sequence": {
                "resource": "sfu", "peak_ops_per_s": 720e9,
                "max_utilization": 0.896,
                "evidence": (
                    "BF16 storage lowered through FP32 special math; paired "
                    "service/operator holdout rejected a rate replacement"
                ),
            },
            "bf16_log_f32_sequence": {
                "resource": "sfu", "peak_ops_per_s": 247.469e9,
                "max_utilization": 1.0,
                "evidence": (
                    "FP32 log sequence service retained after paired BF16 "
                    "service/operator holdout validation"
                ),
            },
            "bf16_sqrt_f32_sequence": {
                "resource": "sfu", "peak_ops_per_s": 900e9,
                "max_utilization": 1.0,
                "evidence": (
                    "S5000 paired FP16/BF16 register-service sweep plus exact "
                    "TileLang BF16 Sqrt development/holdout validation"
                ),
            },
        },

        # Utilisation ceilings (all P8 measured)
        ddr_max_utilization=0.876,          # P8: fill 1401/1600
        l2_max_utilization=0.342,           # P8: ~2450/7168 GB/s sustained
        l1_max_utilization=0.95,            # P8: smem 11.01/11.52 = 95.6%
        compute_max_utilization=0.832,      # Tensor Core GEMM 153.4/184 (single value)

        # Instruction support
        support_wgmma=False,                # NVIDIA Hopper-specific
        support_utcmma=False,               # not applicable to MUSA
        use_tensor_core_resource_model=False,
        apply_dram_wave_quantization=False,

        # Ordinary cached Global-load L1 (P9 measured).  Keep separate from
        # the 192 KiB Shared Memory and its 11.52 TB/s physical bandwidth.
        l1_cache_capacity_bytes_per_sm=32 * 1024,
        l1_cache_line_bytes=64,
        l1_cache_sector_bytes=32,
        l1_cache_peak_bandwidth_bytes_per_s=7168e9,
        # P9 ILP4 probe: 5.23 TB/s / 7.168 TB/s physical peak.
        l1_cache_max_utilization=0.73,
        tma_bypasses_ordinary_l1=True,

        # P4 exact-kernel occupancy probes show a 3072-thread / 96-warp
        # residency ceiling for low-resource kernels: 64/128/256/384/512
        # threads map to 48/24/12/8/6 active CTA per MP.  The driver-reported
        # 256 CTA value is only the architectural block ceiling.  Exact binary
        # Occupancy API results still take precedence over these limits.
        max_resident_threads_per_sm=3072,
        max_resident_warps_per_sm=96,
        max_architectural_ctas_per_sm=256,
    )


def S5000() -> ConfigurableStackedGpu:
    """Build an S5000 architecture object (convenience wrapper).

    Returns a :class:`ConfigurableStackedGpu` built from
    :func:`make_s5000_config`, equivalent to
    ``make_custom_stacked_gpu(make_s5000_config())``.
    """
    return make_custom_stacked_gpu(make_s5000_config())
