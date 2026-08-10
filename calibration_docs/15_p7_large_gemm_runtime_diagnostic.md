# P7 大 FP16 GEMM runtime/计时诊断

状态：completed — 已定位到 runtime 层级，但未得到可用于校准的高 batch 计时替代方案。

日期：2026-07-29；设备：MTT S5000（MUSA capability 3.1）；驱动：3.3.5-server；SDK/mcc：4.3.5；TileLang-MUSA：`4d666968a005846314e9a84d8ff121c5dbffa9ee`。

## 范围与不可变性

诊断目标是此前 `1536×1024×1536` FP16 TileLang GEMM 的间歇性 `MUSA error: operation not permitted`，不是重新拟合模型。所有 P7 数据都标记 `diagnostic only`：P5 v1 SHA `9ef664f1e27f78bcd99cceb458750cf02319d6ae68de8573438cec69a17aeec0`、P5 v2 SHA `b94296eb50b904b0e34b80d79535887ad7932b7b08f57fe736c609eb48caa9be` 和 P6-v3 结果均未修改。

固定 kernel 条件为 FP16、FP32 accumulation，shape `(M,N,K)=(1536,1024,1536)`、`block=(64,64,64)`、4 warp、stage 1。所有成功和失败编译出的 MUSA 源均为 SHA-256 `9ace2f6f78aed08ced8f393797da60e524f08afc89c1ca4bd5c587f9022d33ff`，动态 shared memory 为 16384 bytes，launch threads 为 256。

## 实验和证据

| 试验 | 独立进程结果 | 关键观察 | 证据 |
|---|---:|---|---|
| 每次 replay 后同步的 graph-unroll 扫描 | unroll 1–50: 3/3 pass；100: 2/3 pass + 1 个启动前超时 | 成功时 unroll 50/100 的 replay 约 41.0/39.8 µs，接近 direct event 40.2 µs；启动前超时不得当作 replay 失败或性能样本 | `results/processed/p7_large_gemm_graph/summary.csv` |
| P2 式 queued replay（最后统一 synchronize） | depth 1: 3/3；2: 3/3；4: 2/3；8: 3/3；16: 2/3；30: 2/3 | depth 4、16、30 均可失败，但 8 通过，故没有单调的 graph queue-depth 阈值 | `results/processed/p7_large_gemm_graph_queue/summary.csv` |
| 原 P2 harness 的单 case 精确复现 | compile 3/3；correctness 3/3；measured 2/3 | 第 2 进程在 `graph_samples` 的最终 `torch.musa.synchronize()` 报 `operation not permitted`；由此排除“只有 mixed-case 生命周期才触发” | `results/processed/p7_p2_exact_single_gemm/summary.json` |
| 同一复现 + `MUSA_LAUNCH_BLOCKING=1` | compile 3/3；correctness 3/3；measured 2/3 | 第 3 进程错误提前定位到 `MUSAGraph.replay()`：`FillOp MUDNN failed in: Run`；MUDNN 日志给出 `Fill::Run` / `musa runtime ... err 999 = unknown error` | `results/processed/p7_p2_exact_single_gemm_launch_blocking/summary.json`、`results/raw/p7_p2_exact_single_gemm_launch_blocking/20260729T053938161898105Z/process_3/run.log` |
| 无 graph 的普通 launch，100 个 launch/批 | 0/3 pass | 三次均在首个 batch 的同步报 `unknown error`；它不是可靠的 graph-free 计时回退 | `results/processed/p7_large_gemm_direct_batch/summary.json` |
| 普通 launch batch-unroll sweep，每样本后同步 | 1–32: 各 3/3；64: 1/3；100: 3/3 | 64 的失败发生在完成第 5、2 个样本之后，但 100 又全部通过；所以不存在能由本数据支持的固定“安全 launch 深度” | `results/processed/p7_large_gemm_direct_batch_sweep/summary.csv` |
| batch-32 独立 shape 交叉检查 | `1536×768×1024`: 3/3 pass | 30-sample 事件/Profiler 差异为 1.9%–5.4%，但此 shape 的成功不能替代原大 GEMM 的确认 | `results/processed/p7_direct_batch32_crosscheck/summary.json` |
| batch-32 原大 GEMM、同一 30-sample 协议确认 | 1/3 pass | trial 1 的 30 个事件样本完成后在 Profiler 退出报 `unknown error`，随后 `double free or corruption` 并挂起；trial 2 在第 18 个样本后同步失败；仅 trial 3 通过，候选协议被拒绝 | `results/processed/p7_direct_batch32_primary_confirmation/summary.json`、`results/raw/p7_direct_batch32_primary_confirmation/20260729T055654751996952Z/controller_run_note.json` |

原始源码、mcc 命令和每进程日志都在对应 `results/raw/`、`logs/build/p7_*` 目录。受控终止后首次 `mthreads-gmi` 显示 0% GPU、0 MiB、无残留进程；随后两次检查仍显示 0%/0 MiB（P12），但进程表查询返回 `Failed to query running processes`。因此既不能把前者当作驱动已完全恢复的证明，也不能从该 CLI 现象推出物理硬件故障；在状态明确前不再启动新 kernel。

## 结论

1. 该 kernel 能稳定完成编译和单次数值正确性验证，故 P7 没有发现 TileLang-MUSA lowering 的确定性编译/数值错误。
2. 同一 source SHA 在成功与失败的全新 Python 进程中共存；失败随 graph replay 或连续普通 launch 出现，且 batch/depth 结果非单调。因此不能将异常归因于 shape、tile、mcc 产物或某个固定 queue-depth。
3. `MUSA_LAUNCH_BLOCKING=1` 使一次错误在 `MUSAGraph.replay()` 的 MUDNN Fill 操作处暴露，但异步设备错误可在后续 API 才被报告；因此该证据既不能证明 Fill 是起因，也不能证明 kernel 本身无关。无 graph 的连续 launch 同样失败，故障域应表述为该 kernel/lowering 与连续执行、driver/runtime、graph/profiler 路径之间的交互，而不是仅限 graph。
4. graph-free batch-32 曾在短 sweep 和独立 shape 交叉检查中通过，但同一原大 GEMM 的同一 30-sample 协议仅 1/3 通过，且包含 Profiler 退出错误与 host 内存破坏症状。因此它不是可冻结的回退协议。
5. 不能从成功的 36–45 µs 诊断值拟合 S5000 GEMM 吞吐。它们受计时 unroll、首批开销和失败选择影响，且没有三次可靠的统一计时协议。P5 v2/P6-v3 的窄域结论保持不变。

## 静态根因定位与待证假设

与 P2 中三进程稳定通过的 FP16 1024³、64×64×64、4 warp、stage 1 生成源码逐行比较后，两者
使用相同的 `MUtensorDescriptor` TMA load、`__musa_async_*` barrier 和 producer/consumer warp
specialization；可见 `.mu` 差异只有 K-loop 从 16 次增加到 24 次。host launch grid 则从
256 CTA 增加到 384 CTA。两者均为 256 threads 和 16 KiB dynamic shared。这个对照说明异常并非
由切换到完全不同的 codegen 模板造成，但不能区分长 barrier 协议、descriptor 参数路径或 runtime
连续 launch 问题。

当前待证假设按排查优先级排列如下：

1. MP31 TME/TMA、异步 barrier 与 warp specialization 在更长循环和重复 launch 下的 lowering/compiler/runtime 交互；
2. TileLang-MUSA TMA barrier protocol 或 kernel 退出前 async operation 的 drain 不完整；
3. 64-byte、64-byte-aligned `MUtensorDescriptor` 经 `grid_constant` 按值传参时的 encode、打包或生命周期问题；
4. MUSA driver/runtime 对连续 launch、graph replay 或 profiler 收尾的独立问题；
5. MUDNN Fill 目前只能视为错误暴露点；物理硬件故障也没有充分证据。

这些是可由 A/B 实验证伪的候选，不是已经确认的根因。设备状态恢复后应在全新进程中依次对比：
baseline、仅关闭 descriptor prefetch、关闭 warp specialization、禁用 TMA lowering 改用同步
global/shared copy、同时关闭 TMA 与 specialization。每条路径先单次正确性，再做受控重复；至少
3 个新进程，失败即结束该进程，初轮不启用 MUSAGraph 或 Profiler。`TL_ENABLE_MUSA_TMA_PREFETCH`
只控制 descriptor prefetch，不能被误写为“关闭 TMA”。

## 审核结果与恢复条件

P7 新增文件已审核：配置 JSON 可解析；Python 通过 `py_compile`；shell runner 通过 `bash -n`；所有汇总器在完整 raw run 上实际运行。首次 graph-free 汇总器暴露出 CSV 额外字段错误，已在同一轮通过 `extrasaction="ignore"` 修复并重新运行；修复不改变任何 raw 数据。batch-32 原 shape trial 1 在写出 runtime_failed 结果后发生 host 侧 `double free or corruption` 并持续 CPU 活跃，控制器记录状态后只终止该诊断 run；该控制动作和结果均保留在 raw 目录。新的 direct-batch runners 已加入每个诊断子进程 120 秒 TERM/KILL 超时与无 `result.json` 的事实 marker，避免未来无限挂起。

当前不应开始 P5.2 拟合。恢复大 GEMM 校准需先满足以下任一条件：

1. MUSA/torch_musa 侧给出并验证此 replay/Fill runtime 错误的修复，随后用冻结配置至少 3 个全新进程重跑；或
2. 新的计时协议在同一 kernel 上完成编译、正确性和至少 3/3 进程稳定性验证，并对不少于一个独立 GEMM shape 做交叉验证；或
3. 厂商支持定位为已知限制并提供受支持的安全计时 API/调用方式。

若条件满足，必须新建隔离的 P5.2 development 集，冻结后再创建全新的 P6-v4 hold-out；不得回填 P5 v2 或重用 P6-v3 hold-out。

## 可复现实验入口

- `scripts/p7_run_large_gemm_graph_diagnostic.sh`
- `scripts/p7_run_large_gemm_graph_queue_diagnostic.sh`
- `scripts/p7_run_p2_exact_single_gemm.sh`
- `scripts/p7_run_p2_exact_single_gemm_launch_blocking.sh`
- `scripts/p7_run_large_gemm_direct_batch_diagnostic.sh`
- `scripts/p7_run_large_gemm_direct_batch_sweep.sh`

其中含 `MUSA_LAUNCH_BLOCKING=1` 的入口只用于定位，绝不可将其时间值写入性能参数表。
