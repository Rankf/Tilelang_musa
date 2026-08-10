# P1 测量基础设施审核报告

> 2026-07-29 勘误：原 P1 返回式 kernel 会经 TVM-FFI adapter 内部执行 `torch.empty`，因此原文若称 output allocation excluded 并不准确。原 P1 数据仅保留为 infrastructure smoke，不得作为校准参数。P2 已改用 `out_idx=[]`、显式预分配输出，并通过 MUSAGraph 与 MUSA profiler 交叉验证；详见 `07_p2_gemm_outlier_root_cause.md` 和 `08_p2_compute_completion_audit.md`。

状态：completed  
执行日期：2026-07-29  
范围：验证计时、正确性门禁、可追溯性和原始数据契约；不输出 S5000 性能参数。

## 1. 验收结论

P1 已通过。现在可以进入 P2 计算路径校准，但 P1 的 `142 µs` 级数值只是计时稳定性 smoke，明确排除在 P2–P5 的硬件参数拟合之外。

通过依据：同一 TileLang-MUSA float32 elementwise-add 在三个独立进程中均先完成编译和数值校验，再进行 warmup 与 MUSA-event 计时；每次均产生原始样本、汇总、元数据、GPU 前后快照、生成源码和 mcc 命令。

## 2. 已实现的 P1 数据路径

入口：`scripts/p1_run_measurement_smoke.sh`。

它固定 P0 的 TileLang-MUSA commit，采集环境与本地 patch 快照，为每次运行创建唯一纳秒时间戳目录，并调用 `benchmarks/p1_tilelang_elementwise_add.py`。计时顺序为：

```text
compile → allocate/input init → correctness assert_close → warmup + synchronize
        → MUSA event around repeated compiled-kernel calls → event synchronize
```

故计时不包含编译、内存分配、输入初始化或正确性比较。每个样本是 20 次已编译 kernel 调用的平均；每个运行采 15 个独立样本，并输出 p10、median、p90 与 `(p90-p10)/median`。

配置见 `configs/p1_elementwise_add_smoke.json`：float32、`N=16,777,216`、256 threads、每 thread 8 elements、10 warmup、15 samples、20 repeats/sample。它是稳定性测试配置，不是 tile/shape sweep。

## 3. 结果与审核

| 原始运行目录 | correctness | median (µs) | p10–p90 相对跨度 | 结论 |
|---|---:|---:|---:|---|
| `results/raw/p1/20260728T092151498074293Z` | pass | 142.646003 | 0.1360% | measured |
| `results/raw/p1/20260728T092223334167960Z` | pass | 142.605996 | 0.1411% | measured |
| `results/raw/p1/20260728T092254685206801Z` | pass | 142.650008 | 0.0981% | measured |

配置的阈值是 15%；三次均低于 0.15%。三次中位值的最大差为 0.044012 µs，约为中位值的 0.031%。这说明在该具体稳定 kernel、该负载、当前软件栈和短时间窗口内，计时分布可解释；它不外推到其他 shape、dtype、算子或长时间运行。

每个 `raw_samples.csv` 均有 15 条样本；`summary.csv` 含状态、shape、launch、warmup/repeats、p10/median/p90、正确性和计时范围；`metadata.json` 保存 MUSA capability、torch/torch_musa、配置、run ID 与工件路径。三次记录的 kernel source hash、harness hash、config hash、固定 TileLang commit 和本地 patch hash 完全一致。

对应生成源码/命令目录为：

- `logs/build/p1_tilelang_kernel/20260728T092151498074293Z/`
- `logs/build/p1_tilelang_kernel/20260728T092223334167960Z/`
- `logs/build/p1_tilelang_kernel/20260728T092254685206801Z/`

三份 `.mu` 的 SHA-256 都是 `3be1a14eae7d0a6fd6a4bd216cf96b04eaa80cfd35e40e12f8dffb7799573a23`。

## 4. 审核发现与已处理修改

审核中发现最初 metadata 虽保存 kernel source hash，但依赖运行目录命名才能反查源码。已修改 runner/harness，使每条 summary 与 metadata 显式保存：

- `kernel_dump_dir`：生成 `.mu` 与 `.mu.cmd` 的绝对路径；
- `runner_log`：本次完整执行日志；
- `harness_hash`：计时程序版本；
- `config_hash`、`source_hash`、`tilelang_patch_sha256`：配置、生成源码和本地 backend 补丁的版本边界。

此后重新完成上述三个独立运行。静态审核也已通过：shell `bash -n`、Python `py_compile`、JSON 解析和 TileLang 工作树 `git diff --check`。

## 5. 已知限制

- TileLang 的 parallel-loop verifier 仍对该 kernel 报 data-race warning；本次三次 `torch.testing.assert_close` 都通过，但没有将 warning 擅自判为误报。P2 必须保留正确性门禁，并增加边界长度、不同 grid 和重复运行覆盖。
- `mthreads-gmi` 前后快照显示设备空闲、无用户 GPU process；显示 2 MiB runtime 占用。它不是硬件 counter。
- P1 仅验证 default-stream MUSA event 计时；多 stream、并发 CTA、graph capture 或跨进程干扰不在本阶段结论内。

## 6. P2 进入条件与建议

P2 可复用此 runner 的记录格式、正确性门禁和 trace 链，但需新建 compute benchmark 配置与结果目录。顺序应为：先 FP16 GEMM 单一垂直切片，逐点执行 compile → correctness → timing；只有 `status=measured` 且稳定的点才能进入有效吞吐曲线。BF16、INT8、vector/reduce/SFU 应随后分开开展，不能由本 P1 数值反推。
