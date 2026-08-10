# P7.1–P7.3 复核审计

日期：2026-07-29（UTC+8）
结论：状态、源码快照、计数与冻结模型均一致；发现 P7.3 pilot 的 per-run environment/MUBIN provenance 缺口，已降级并将在 P7.3.1 前向修复。

## 已通过的检查

| 范围 | 检查 | 结果 |
|---|---|---|
| P7.1 | raw result 数量 | 15；12 pass，3 source-contract-failed，符合 A–E × 3 的记录 |
| P7.2 | raw result 数量 | 主 run 21，单写修正 run 3；3 compile failure 与 18 个已生成 source 对应 |
| P7.3 pilot | raw result 数量 | 36，等于 3 shapes × 4 schedules × 3 trials |
| source/config provenance | metadata SHA 对照 raw 内 snapshots | P7.1、P7.2、P7.3 三组均逐字一致 |
| source/mcc coverage | build 目录 | P7.1 15/15，P7.2 18/18（其余 3 未编译），P7.3 36/36 `.mu` 与 `.mu.cmd` |
| 状态不变量 | `measured`/`measurement_rejected` 前置 | 没有 measurement 状态缺少 compile/correctness pass 的记录 |
| P7.2 汇总 | 3 accepted、4 pending、1 historical excluded | 与 raw 重算一致；历史无效 single-write 未被当作正式 case |
| P7.3 汇总 | 4 development-measured、8 not-development-measured | 与 36 个 raw trial 重算一致 |
| 冻结模型 | P5 v1/v2 SHA | 分别仍为 `9ef664…aeec0`、`b94296…a9be` |

## 发现的缺口与处置

P7.3 pilot runner 保存了 config、harness、GPU 前后快照、`.mu` 和 mcc command，但没有单独的环境快照，且
未保存 mcc 实际产生并被 runtime 加载的 MUBIN。`.mu.cmd` 虽显示目标路径为 `.mubin`，默认临时产物在 runtime
加载后没有保留，因此不能把 pilot 的 resource/provenance 声称为 exact-binary。

这不改变 pilot 的 compile/correctness/timing 事实，但限制其用途：它继续是 protocol diagnostic，不能成为
P7.4 occupancy/resource 或 P8 profile 的 exact-binary 证据。

P7.3.1 的强制修复：

1. 通过 TileLang-MUSA 已实现的 pass-context key `musa.kernels_output_dir` 保存实际 mcc output；
2. 每 trial 保存该 MUBIN、SHA-256、`.mu`、mcc command 和 artifact path；
3. raw root 追加 `environment.txt`、`mcc_version.txt`、Python/TileLang commit 信息和 GPU before/after；
4. finalizer 只接受 MUBIN SHA 存在且与试验记录一致的 schedule。

历史 P7.1/P7.2/P7.3 raw 不会补写或伪造 MUBIN；其 provenance 按各自实际保存内容标注。
