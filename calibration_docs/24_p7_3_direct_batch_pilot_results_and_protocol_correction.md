# P7.3 direct-batch GEMM pilot：结果与协议修正

状态：pilot 完成；不能作为完整 development 数据集或 schedule-ranking 数据。
日期：2026-07-29（UTC+8）

## 执行结果

raw：`results/raw/p7_3_gemm_development_pilot/20260729T075007912153857Z/`。
processed：`results/processed/p7_3_gemm_development_pilot_v1/summary.{json,csv}`。

固定清单是 3 个 FP16 shapes × 4 个已编译 schedule × 3 个新进程，共 36/36 trial。所有 trial 均编译并通过数值正确性；GPU 后快照为 0 MiB、0%。计时路径为 graph-free MUSA Event、batch-32、每 batch 同步，不使用 graph 或 Profiler。

但预注册的单-trial `P90-P10 / median <= 5%` 门槛只让 4/12 schedule 成为 `development_measured`：

- 512³：128×64×64 stage-1（14.503 us）、64×128×64 stage-1（14.423 us）、128×128×64 stage-2（20.151 us）；
- 1024³：128×128×64 stage-2（34.162 us）。

其余 8/12 schedule 至少一个 trial 出现 sample spread，因而正确记录为 `not_development_measured`。尤其 64×64×64 的较快单 trial 不能因“看起来快”而被接纳。

## 为什么不能继续用这 4 个 schedule

这不是“仅有四个 schedule 能运行”：所有 12 个 schedule 都 compile/correct。拒绝原因是 direct batch-32 的样本稳定性，而不是 TileLang lowering 错误。

如果将仅通过的 4 个 schedule 输入 development/ranking，会让计时稳定性筛掉部分较快 schedule，造成选择偏差。这样的数据既不能训练预测器，也不能评价 schedule 排名。因此：

- 本 pilot 只保留为 `timing_protocol_diagnostic`；
- 不写入 P5/P6，不激活 P8 profile，不产生 P9；
- 不事后把 5% 门槛调宽，也不从失败 trial 挑成功 samples。

## P7.3.1 修正

P7.2 已表明 graph-free 多-unroll Event 斜率可以在部分 kernel 上稳定，而极短 case 会被明确拒绝。因此 P7.3.1 将对同一固定 12 个 schedule 使用新的、独立版本的 `graph_free_event_linear` 协议：每个 unroll 采样、拟合 total-time slope、保存 intercept/R²/residual，并对三个进程的 slope 使用预注册门槛。它不会复用或覆盖本 pilot。

若 P7.3.1 仍不能为每个 shape 得到至少 4 个稳定 schedule，P7.3 保持 incomplete；不能以这 4 个 schedule 声称完成。
