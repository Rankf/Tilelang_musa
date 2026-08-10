# P7.1 恢复门禁结果与 Runtime A/B 静态准备

状态：blocked before GPU launch；静态 A/B 准备已完成
日期：2026-07-29（UTC+8）

> 历史记录说明：本文件描述初版门禁的当时结果。后续只读诊断确认失败项是该设备不支持的
> `mthreads-gmi` process-monitor/query；门禁已修正并完成 A/B，见文档 20 和文档 21。下文原始
> evidence 保留，不改写为成功。

## 1. 执行结论

P7.1 的只读恢复门禁未通过，因此没有启动最小 runtime smoke、最小 TileLang kernel 或大 GEMM A/B。
这符合文档 18 的停止规则：`mthreads-gmi` 必须连续三次正常返回进程表，任一次失败即停止 GPU kernel
执行。

门禁 raw run：
`results/raw/p7_1_recovery_gate/20260729T070308879919165Z/`

三次 `mthreads-gmi` 都报告：MTT S5000 为 0% GPU、0 MiB / 81920 MiB、P12、27C；但每次进程表均为：

```text
Failed to query running processes
```

OS 进程快照中没有残留 P2/P7 controller 或 Python benchmark；仅有本次恢复门禁 shell 本身。因此不能把
“0 MiB、无残留用户 benchmark”视作 driver/runtime 已恢复的证明，也不能从该现象推断物理硬件故障。

## 2. 静态 A/B 资产

已新增、但尚未运行的 P7.1 工件：

| 工件 | 作用 | SHA-256 |
|---|---|---|
| `configs/p7_1_runtime_ab.json` | 固定 shape、五个变体、源码合同、batch-unroll 和停止规则 | `dfac0db06627d4356408b2fd42d517d7e5bd8810cb0819f3588d022a494dc6ca` |
| `benchmarks/p7_1_runtime_ab.py` | versioned graph-free A/B harness；分离 compile/correctness/runtime/source-contract failure | `4c85236c843af85dd1b3b0de8b72047867e1ab5ba4714baa2d083411414eecad` |
| `scripts/p7_1_recovery_gate.sh` | 三次只读 `mthreads-gmi` 门禁与进程快照 | `ce36063e5e5cc57a9659d7c8023e2e47aeac1caa65e8e26300a750d6c9153e47` |
| `scripts/p7_1_run_runtime_ab.sh` | 强制先调用恢复门禁，再按 variant/trial 启动独立进程 | `6223d9860d8a1ba81bd0b3feb1d8aaf8601983e19c20aece2741b17508763293` |

五个变体为 A baseline、B 关闭 descriptor prefetch、C 关闭 warp specialization、D per-copy 关闭 TMA、
E 同时关闭 TMA 与 warp specialization。A/B primary protocol 不使用 MUSAGraph 或 torch Profiler；每个
batch 后同步，batch-unroll 固定为 1/2/4/8/16/32，任一 runtime failure 立即结束该进程。

TMA 对照使用 `T.copy(..., disable_tma=True)`，不依赖即将移除的全局兼容 flag。warp specialization
使用已存在的 `TL_DISABLE_WARP_SPECIALIZED`。每个 variant 在性能前必须先检查生成 `.mu`：

- TMA：`MUtensorDescriptor` 或 `tl::tma_load<`；
- descriptor prefetch：`prefetch_tma_descriptor`；
- warp specialization：producer/consumer thread 分支加 async barrier；
- 异步 barrier：`__musa_async_`。

源码合同不匹配时记录 `source_contract_failed`，不把它当作成功 A/B 样本。

## 3. 静态审核

以下检查全部通过：

1. P7.1 harness 和全部现有 benchmark/script 的 Python、shell 语法检查；
2. P7.1 config JSON 解析、五个 variant ID、三个 trial 和 batch-unroll 清单检查；
3. 281 个 calibration artifact JSON 均可解析；
4. 已保存的异常大 GEMM baseline `.mu` 被 feature detector 正确识别为：TMA=true、descriptor prefetch=true、
   warp specialization=true、async barrier=true；
5. recovery gate `gate_status.json` 及三份快照一致地记录 blocked；
6. P5 v1/v2 冻结 SHA 和验证器未变化。

门禁脚本直接执行会受新文件未设置 executable bit 影响；A/B controller 已修正为显式使用 `bash` 调用门禁，
并通过 `bash -n`。这不是 GPU/runtime 故障，也没有影响任何历史数据。

## 4. 不得执行的事项

在新的恢复门禁通过前：

- 不运行 P7.1 A/B controller；
- 不运行 P7.2 短 kernel 测量；
- 不运行 P7.3/P7.4 新 GPU 数据采集；
- 不开始会产生性能结论的 P8 profile 冻结；
- 不修改 P5 v1/v2、P6-v3 或历史 raw/processed 数据。

P7.1 的下一步只能是设备状态发生外部恢复后，重新运行只读 gate；任何 reset/reboot 仍需要用户明确授权。
