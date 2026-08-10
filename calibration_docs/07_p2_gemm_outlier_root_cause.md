# P2 GEMM 256³/512³ 离群根因审核

状态：root cause confirmed for the original measurement path  
日期：2026-07-29

## 结论

原 256³/512³ 的 58–437 µs 严重离群，主要不是 GEMM kernel 本体偶发慢数十倍，而是旧测量区间包含了 Python 提交线程停顿与返回式 kernel 的内部输出分配。短 kernel 只有约 6–10 µs；当 host 在 `start.record()` 之后、kernel/end event 入队之前被调度暂停时，MUSA event 会如实计入这段队列空洞。因此 event 数值本身没有坏，但它测到的范围不是纯 kernel self-device time。

这一结论只适用于已复现的旧 harness；不能由此断言设备永远没有执行长尾。

## 原始离群证据

旧数据保留在 `results/raw/p2/`，不删除、不改写，并标为 superseded measurement evidence：

| run | shape | sample median (µs) | max (µs) |
|---|---:|---:|---:|
| 013313... | 512³ | 16.097 | 433.843 |
| 013420... | 256³ | 13.107 | 437.377 |
| 013420... | 512³ | 19.117 | 326.647 |
| 013827... | 256³ | 13.974 | 348.144 |
| 013827... | 512³ | 13.593 | 434.335 |
| 014002... | 256³ | 15.713 | 58.423 |

GPU 快照在离群前后均显示 P0、约 34–36°C、无其他 GPU 进程，因此没有证据支持温度降频或外部 GPU 负载是主因。

## 受控诊断 case

诊断程序：`benchmarks/p2_gemm_timing_diagnostic.py`  
原始目录：`results/raw/p2_diagnostics/20260729T020009280631901Z/`

| case | queued event median / max | synchronized event median | MUSAGraph×100 median | profiler self-device |
|---|---:|---:|---:|---:|
| 256³ default s1 | 17.84 / 37.68 µs | 12.08 µs | 8.09 µs | 7.15 µs |
| 512³ default s1 | 26.08 / 24152.48 µs | 15.12 µs | 10.69 µs | 9.25 µs |
| 512³ 128² tile s3 | 29.00 / 47.16 µs | 24.60 µs | 19.92 µs | 17.94 µs |
| 1024³ default s1 | 33.46 / 47.20 µs | 31.22 µs | 23.20 µs | 20.85 µs |

512³ 的 24.152 ms event 样本对应 host submission 24.137 ms；两者同步出现，是 host 在 event 区间内停顿的直接证据。MUSAGraph 捕获成功，预分配 A/B/C 后批量回放消除了 Python 逐次提交空洞；profiler 独立给出 kernel self-device 聚合时间。

## 两个污染源与处置

1. 旧 `actual = kernel(a,b)` 路径经 TVM-FFI adapter 每次内部执行 `torch.empty`。改为 `out_idx=[]` 和显式预分配 C 后，正常延迟明显下降。
2. 旧 Python 提交会在 `start event → kernel call → end event` 的 host 入队过程中遭遇调度停顿。短 kernel 很快耗尽已排队工作，形成 queue starvation；event 因此包含空洞。

六组旧 run 不得用于拟合。正式 P2 使用显式预分配、MUSAGraph 100 kernel 展开回放，并用 profiler self-device time 交叉审核。256³/512³ 的 Graph 分布已稳定，但绝对时间仍含约 1.3–2.5 µs 的 graph 固定开销；短点继续标记 pending，不作为吞吐参数锚点。
