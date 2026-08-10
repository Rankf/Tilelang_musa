# P2 单卡 compute 校准完成审核

状态：completed with explicit pending short-kernel cases  
日期：2026-07-29  
硬件：1× MTT S5000 / MUSA capability 3.1

## 范围和口径

本阶段完成 FP16、BF16、INT8 GEMM，FP32 vector add、exp、rsqrt 和 row reduce 的最小纵向切片。每个 case 严格执行：编译 → 数值正确性 → 计时。输入和输出均预分配；正式计时用 MUSAGraph 捕获 100 个相同 kernel 后回放 30 次，另用 MUSA profiler 聚合 50 次 kernel self-device time 审核。

一键脚本：`scripts/p2_run_compute_suite.sh`；配置：`configs/p2_compute_suite.json`；runner：`benchmarks/p2_tilelang_compute_suite.py`。

三个独立进程：

- `results/raw/p2_compute/20260729T020316095211165Z`
- `results/raw/p2_compute/20260729T020444776690014Z`
- `results/raw/p2_compute/20260729T020602462471032Z`

每个目录包含 20 条 summary、600 条 Graph 原始样本、profiler 聚合、GPU 快照、配置和 harness hash；源码与 mcc 命令在对应 `logs/build/p2_compute/<run>/`。

## 三进程审核结果

| case | Graph latency | 条件有效性能 | 处置 |
|---|---:|---:|---|
| FP16 1024³ | ~23.42 µs | ~91.7 TFLOPS | eligible |
| FP16 512×512×4096 | ~43.95 µs | ~48.9 TFLOPS | eligible |
| FP16 skinny/tall | ~17–18 µs | ~59–63 TFLOPS | eligible, short-boundary |
| FP16 1024³ tile128 stage3 | ~32.69 µs | ~65.7 TFLOPS | eligible |
| BF16 1024³ | ~23.25 µs | ~92.4 TFLOPS | eligible |
| BF16 512×512×4096 | ~43.89 µs | ~48.9 TFLOPS | eligible |
| INT8 512³ / 1024³ → INT32 | ~18.82 / 30.70 µs | ~14.3 / 70 TOPS | eligible |
| vector add 16M FP32 | ~144.2 µs | ~116 GElem/s | eligible |
| exp 16M FP32 | ~100.8 µs | ~166 GElem/s | eligible |
| rsqrt 16M FP32 | ~114.0 µs | ~147 GElem/s | eligible |
| reduce 4096×256 / ×1024 | ~45.6 / 64.3 µs | ~23 / 65 GElem/s input | eligible |

精确值见 `results/processed/p2_compute/aggregate.csv` 和 `.json`。这些是指定 TileLang schedule 的条件有效性能，不是硬件峰值。

## Pending 与审核边界

FP16 256³/512³、BF16 512³ 以及 1M vector/exp/rsqrt 均已编译且正确，Graph 分布也稳定；但它们只有约 5–11 µs，Graph 固定开销相对 profiler self-device time 占比过大，部分独立进程未通过 25% 方法一致性门禁。因此标记 `measured_but_pending_short_kernel_method`，不进入参数拟合。详见 `07_p2_gemm_outlier_root_cause.md`。

P2 证明这些路径可形成可复查测量，不等于已经找到最优 tile 或峰值。Graph 用于 kernel intrinsic 校准；direct-launch 还包含 host launch/queue 行为，后续端到端模型应单建 launch-overhead 项。P2 不包含 DDR/L2/shared-memory、occupancy/wave 或 hold-out 验证。

结论：P2 compute vertical slice 已完成，可以进入 P3 memory/cache；短 kernel 参数仍需低开销计时校准，不能在 P5 直接消费。
