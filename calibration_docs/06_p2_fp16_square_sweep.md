# P2 计算路径：FP16 方阵 shape sweep

状态：superseded measurement evidence；计时口径已由 07/08 报告替代  
执行日期：2026-07-29  
范围：固定 TileLang schedule 的 `128³/256³/512³/1024³` FP16 GEMM；用于建立 shape 依赖的初步证据，不是 tile 搜索、峰值搜索或参数冻结。

## 1. 固定路径

配置见 `configs/p2_fp16_gemm_square_sweep.json`：

```text
tile = 64×64×64
threads = 128 (4 warp)
policy = default
pipeline stage = 1
dtype = FP16, accumulator = FP32
```

每个 shape 均在三个独立进程中重新编译、正确性检查、warmup，并用 200 次 kernel/event batch 采集 15 个样本。每次都要求：`correctness=pass`、p10–p90 相对跨度 ≤15%、max(sample)/median ≤2；跨进程还要求全部 3 次为 measured 且中位值相对跨度 ≤15%。

## 2. 结果

| shape | 三次状态 | 通过运行的中位延迟 (µs) | 条件有效 TFLOPS | 审核结论 |
|---|---|---:|---:|---|
| 128³ | measured / measured / measured | 12.923801, 13.591200, 14.819400 | 0.324541, 0.308604, 0.283028 | provisional；第三次 `max/median=1.990`，接近门槛 |
| 256³ | runtime_failed / measured / runtime_failed | 13.974400, 10.644400, 15.713000 | 仅中间运行 3.152308 | pending；两次严重离群 |
| 512³ | runtime_failed / measured / measured | 13.592600, 13.259000, 14.875201 | 20.245529, 18.045837 | pending；首次严重离群 |
| 1024³ | measured / measured / measured | 22.046800, 22.039399, 22.019999 | 97.405686, 97.438394, 97.524239 | provisional，跨运行中位值相对跨度 0.1216% |

`1024³` 是当前最稳定的实测 FP16 GEMM 点；其三次生成源码 hash 一致：`fe21a0a2b3bc8a0cb2a8403d1f6d8750c198dd9c2bad3d72457b7ae76a44c4d6`。这仍只是指定 schedule 的有效速率，不能表述为 S5000 的峰值。

`128³` 虽按当前规则通过，但接近 max/median 门槛；后续 P2 数据消费时应保留其边界标记。`256³` 在较早初始三次严格运行中曾全通过，但本 sweep 的两次重复失败。为避免选择性取样，它从 provisional 集合移至 pending。

## 3. 工件

三次原始目录：

- `results/raw/p2/20260729T013827745932157Z`
- `results/raw/p2/20260729T013921155738726Z`
- `results/raw/p2/20260729T014002157390184Z`

每个目录均含 4 条 summary、60 条 raw samples、JSON metadata、GPU 前后快照；对应生成源码和 mcc 命令在同 run ID 的 `logs/build/p2_tilelang_kernel/` 子目录。

## 4. 当前 P2 状态

P2 未完成。已完成的是 FP16 standard GEMM 的一个固定 schedule shape sweep 与严格稳定性分流。下一步应优先定位 256³/512³离群的可复现条件（例如更长 batch、运行间隔、GPU 状态），同时补 FP16 large-K、skinny/tall，再进入独立 BF16/INT8/vector/reduce/SFU 路径。任何 pending 样本均不能进入 P5 参数拟合。
