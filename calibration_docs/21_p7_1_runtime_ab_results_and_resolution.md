# P7.1 Runtime A/B：结果、修正与可执行结论

状态：完成诊断；不进入 P5/P6 性能参数拟合。
日期：2026-07-29（UTC+8）

## 结论

此前的“恢复门禁失败”已解决：它是 `mthreads-gmi` 不支持 process-monitor/query 的误门禁，**不是**已证实的设备失效。修正后的 device-level JSON + `/proc` 门禁通过后，严格复现历史同源 GEMM 的 graph-free、无 Profiler 路径稳定完成。

历史异常 kernel（FP16 `1536x1024x1536`、tile `64x64x64`、4 warp、stage 1）的 baseline source SHA 是
`9ace2f6f78aed08ced8f393797da60e524f08afc89c1ca4bd5c587f9022d33ff`。P7.1 的 A baseline 三个独立进程均生成该 SHA、编译通过、数值正确，并各完成 630 次 kernel launch（10 samples × batch 1/2/4/8/16/32），共 1,890 次，没有 `err 999`、MUDNN Fill、MUSAGraph 或 host-corruption 错误。

因此实际可采取的修复是：**校准 harness 禁用 MUSAGraph 和 torch Profiler，使用 graph-free MUSA Event batch-unroll、每个 batch 后同步、独立进程运行。** 这使该 case 可以在此计时路径下继续被诊断；它并不证明 graph/profiler 的实现已修复，也不能把根因精确归到 FillOp。

## 运行证据

主 A/B raw run：`results/raw/p7_1_runtime_ab/20260729T071325086080913Z/`。

WS/TMA lowering 耦合修正后的 C raw run：`results/raw/p7_1_runtime_ab_ws_coupling/20260729T071817312104693Z/`。
生成 MUSA 源：`logs/build/p7_1_runtime_ab/20260729T071325086080913Z/` 与
`logs/build/p7_1_runtime_ab_ws_coupling/20260729T071817312104693Z/`。

下表为 batch-32 的三个 trial 中位数的均值；仅供 runtime-path A/B，不是 S5000 通用吞吐参数。

| 变体 | 生成特征 | batch-32 均值 (us) | 相对 A | 结论 |
|---|---|---:|---:|---|
| A baseline | TMA + descriptor prefetch + WS + async barrier | 37.381 | — | 三次 pass；与历史异常 kernel 同 source SHA |
| B no descriptor prefetch | TMA + WS + async barrier，无 prefetch | 37.558 | +0.47% | 三次 pass；关闭 prefetch 没有修复一个可复现的 graph-free 故障，因为该故障未出现 |
| C no warp specialized | 无 TMA/prefetch/WS/async barrier，128 threads | 54.157 | +44.88% | 三次 pass；见下方耦合说明 |
| D sync copy | 无 TMA/prefetch/WS/async barrier，128 threads | 53.973 | +44.39% | 三次 pass |
| E sync copy + no WS | 无 TMA/prefetch/WS/async barrier，128 threads | 54.006 | +44.47% | 三次 pass；与 D 同 source SHA |

主 run 中 GPU 前后均为 0 MiB；结束快照的默认 `mthreads-gmi` 也能列出 `No running processes found`。

## 审核中发现并修正的两项问题

1. 初版 P7.1 harness 将循环变量拼写为 `kk`，导致生成 source SHA 与历史 `k` 版本不同，虽仅为文本变量名变化。已改回 `k` 并复跑；最早的预修正样本 `results/raw/p7_1_runtime_ab/20260729T071136951176623Z/` 仅保留作诊断记录，不用于本结论。
2. 初版 C 的 source contract 错误地要求“关闭 WS 但保留 TMA/prefetch”。实测生成 source SHA 为 `c608…03b5`，且特征显示 TMA、prefetch、WS、async barrier 都关闭，恰与 D 相同。这不是 detector 错判：当前 TileLang-MUSA lowering 中 `TL_DISABLE_WARP_SPECIALIZED` 会一并走同步-copy路径。config 已改为记录这个真实合同，并在独立的三 trial run 中重跑 C。

因此，现有后端无法用该 flag 单独估计“仅 WS 的边际影响”；只能估计从 A 的 TMA+WS 路径切换到同步-copy路径的组合代价。要单独分离 WS 与 TMA，需要新增一个后端支持的、能保留 TMA 的 non-WS lowering，而不是从本次数据硬拟合。

## 未解决项和边界

- 尚未重新运行 MUSAGraph/Profiler 对照；这样做会重新触发已知不可靠路径，且当前性能校准并不需要它。
- 所以“graph/profiler 路径为最高优先级嫌疑”是证据支持的归因，不是已证明的单一根因。
- 这批结果不修改 P5 v1、P5 v2 或 P6-v3。审计时重新运行 `scripts/p5_validate_model.py`（89 parameters、45 predictions）和 `scripts/p5_1_validate_transfer_model.py`（19 records）均通过；冻结 SHA 仍分别为 `9ef664…aeec0` 和 `b94296…a9be`。
