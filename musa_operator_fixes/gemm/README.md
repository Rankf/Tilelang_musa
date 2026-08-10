# GEMM family

`dense/` holds ordinary FP16/BF16 dense GEMM repairs.
`bmm/` holds batched matrix multiplication.
`attention_qk_pv/` holds the GEMM bodies of attention (QK-transpose and PV).
`moe_grouped/` holds grouped/expert GEMM bodies, excluding routing.
`advanced_fp8_persistent/` holds FP8, persistent, WGMMA/TMA-heavy variants.

Keep failures classified as CUDA/Hopper guard, GEMM lowering, layout, resource,
dependency, or runtime with the implementation that addresses them.
