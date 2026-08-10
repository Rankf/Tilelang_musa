# Dense GEMM repairs

## MP31 FP16 NN

`MusaMP31Fp16NNDenseGemmKernel` replaces TileOPs' Hopper-only dense
`GemmKernel` through the existing `GemmOp(kernel_map=...)` injection point.
It uses target-aware `T.gemm` rather than explicit Hopper `T.wgmma_gemm`.

Current functional scope:

- MUSA MP31 (`supported_archs=[31]`);
- FP16 or BF16 inputs with FP32 accumulation;
- NN, NT, TN, and TT storage layouts;
- arbitrary positive M/N/K dimensions, including M/N/K tails;
- fixed `64x64x64`, 4-warp, stage-1 schedule; and
- no autotuning or schedule ranking.

The 1024x1024x1024 correctness smoke compiled and passed once on S5000 on
2026-07-29. All 14 non-FP8 `GemmOp` manifest workloads from TileOPs were then
compiled, run once, and checked against a FP32 reference on S5000; this includes
decode/prefill NT, mid-M, K-dominant, and wide-N cases. The detailed coverage is
in `manifest_coverage_20260729.json`.

A subsequent three-process MUSA Event comparison reached 82.64%,
85.22%, and 83.87% of the paired torch_musa `torch.matmul` throughput. The
median paired ratio is 83.87%. This is a preliminary S5000 comparison, not an
NVIDIA CUDA comparison or a frozen calibration result. See
`benchmark_20260729_summary.json` for conditions and limitations.
