# S5000 stage 1/2/3 独立 Pipeline 探针

## 结论

独立 MUSA 探针已经完成，但按预先定义的严格门槛，当前多阶段公式状态为：

```text
formula_rejected
```

本实验没有修改 TileSight Pipeline 公式。拒绝原因是最低数学负载 case 的 stage 3 相对 stage 1 时延没有达到公式预测的重叠收益：

| 分组 | Split | Iterations | Stage | 预测/Stage1 与实测/Stage1 差值 |
| --- | --- | ---: | ---: | ---: |
| memory_bound | development | 16 | 3 | -16.04 percentage points |
| memory_bound | holdout | 24 | 3 | -11.04 percentage points |

stage 2、balanced 和 compute-bound case 通过该门槛。阶段周期公式在开发集和留出集也分别通过 10%/15% 误差门，但这不能覆盖上述 Kernel 级 stage 3 偏差。

## 独立变量控制

stage 1/2/3 固定了：

- Grid `240`、Block `256`；
- 动态 Shared `24576 B`、静态 Shared `4096 B`；
- Active CTA/MP `6`；
- 相同输入、输出、迭代工作量和三缓冲启动 footprint；
- SDK 5.2，经 `/usr/local/bin/musa52-run` 编译和运行。

寄存器为 stage 1 的 15 个和 stage 2/3 的 16 个，但没有改变 occupancy。

## MCU 正式结果

以下均为三次 MCU 重复中位数；所有 CV 不超过 3%。

| Case | Iterations | Stage 1 | Stage 2 | Stage 3 |
| --- | ---: | ---: | ---: | ---: |
| memory_bound development | 16 | 23.672 µs | 21.970 µs | 24.439 µs |
| memory_bound holdout | 24 | 31.454 µs | 28.969 µs | 30.583 µs |
| balanced development | 16 | 28.953 µs | 25.096 µs | 26.771 µs |
| balanced holdout | 24 | 38.994 µs | 33.523 µs | 35.110 µs |
| compute-bound development | 16 | 83.304 µs | 79.401 µs | 80.763 µs |
| compute-bound holdout | 24 | 120.122 µs | 115.003 µs | 116.401 µs |

`memory_bound` 是配置中的最低数学负载标签。预热后 DDR counter 接近零，因此准确解释是 TMA load/Shared/同步主导，而不是 cold-DDR 带宽主导。

## 当前能够和不能证明的内容

探针支持：

- 固定 occupancy 条件下 stage 1/2/3 的真实 Kernel 差异；
- 单 CTA 阶段周期对 `fill + steady + drain` 代数结构的验证；
- stage 3 在低数学负载时存在当前公式未覆盖的额外同步/调度成本。

探针不能支持：

- 把阶段时间戳解释成硬件 scheduler trace；
- 验证 `stage_num * resident_cta` 的 TileSight effective-depth 扩展；
- 在没有额外证据时直接增加 stage 3 经验常数。

正式结果：`results/profiling/s5000_pipeline_stage_probe_20260916/validation/summary.json`。
