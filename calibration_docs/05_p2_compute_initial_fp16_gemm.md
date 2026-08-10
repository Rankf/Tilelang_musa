# P2 计算路径：初始 FP16 GEMM 垂直切片

状态：superseded measurement evidence；计时口径已由 07/08 报告替代  
执行日期：2026-07-29  
范围：TileLang-MUSA、MP31、FP16 输入/FP32 累加 GEMM 的第一小批配置；不是完整 P2，也不是 S5000 峰值报告。

## 1. 结论

初始 FP16 GEMM 路径已由真实 TileLang lowering、MP31 `mcc` 编译和 S5000 数值运行验证。四个 case 都完成编译且 `correctness=pass`。但计时审核发现少量严重离群 batch；因此“编译成功/数值正确”与“可用于校准”被严格分开。

按当前 admission policy：每次运行必须通过 p10–p90 与 max/median 门禁、连续 3 次独立进程均为 `measured`，并且跨运行中位值相对跨度不超过 15%。目前只有下列 case 暂具备初始入表资格：

```text
M=N=K=256, tile=64×64×64, 4 warp (128 threads), default policy, stage=1
```

它三次有效 TFLOPS 分别为 2.245766、2.509606、2.544393；中位值为 2.509606 TFLOPS，跨运行中位值相对跨度约 13.12%。这是此配置、此软件版本和当前运行窗口的条件化有效速率，不是理论峰值，也不能用于其他 tile/shape。

## 2. 实现与计时范围

- runner：`scripts/p2_run_fp16_gemm_initial.sh`
- benchmark：`benchmarks/p2_tilelang_fp16_gemm.py`
- 配置：`configs/p2_fp16_gemm_initial.json`

GEMM 实现直接取自仓库 MP31 standard GEMM 的结构：shared-memory A/B tile、FP32 fragment accumulation、`T.Pipelined` 和 `T.gemm`，并启用仓库已有的 MUSA TMA-prefetch pass config。每个点顺序严格为：

```text
compile → torch reference + assert_close → warmup + synchronize
        → MUSA event around 200 compiled-kernel calls × 15 samples
```

不计入：编译、输入分配、随机初始化、torch reference、数值比较。每个结果保存 raw CSV、summary CSV、metadata、GPU 前后快照、生成 `.mu`、对应 `.mu.cmd` 与 runner log。

## 3. 生成代码证据

初始 case 的生成 MUSA 源码不是手写 CUDA 映射。它包含下列可见文本特征：

- `tl::gemm_ss<...>`；
- `tl::tma_load<...>` 与 `tl::prefetch_tma_descriptor(...)`；
- `__musa_async_wait`、`__musa_async_arrive`、`__musa_async_bar_record`；
- `extern __shared__`。

例如 [首个严格运行的源码](/root/s5000-calibration/logs/build/p2_tilelang_kernel/20260729T013313482287356Z/tilelang_musa_kernel_0000.mu) 与 [其 mcc 命令](/root/s5000-calibration/logs/build/p2_tilelang_kernel/20260729T013313482287356Z/tilelang_musa_kernel_0000.mu.cmd) 明确目标为 `mp_31`。这些只记录实际 codegen 路径；本报告不由文本反推矩阵指令 fragment 或硬件细节。

## 4. 严格运行结果

严格配置把 event batch 由 20 增至 200，并新增 `max(sample)/median <= 2` 门禁。此前两批较短 batch 的原始失败证据保留在 `results/raw/p2/20260729T012916615386641Z` 和 `results/raw/p2/20260729T013123668878020Z`，未删除也未覆写。

| case | 严格运行状态（3 次） | 审核结论 |
|---|---|---|
| `256³, 64×64×64, w4, default, s1` | measured / measured / measured | 暂时 eligible |
| `256³, 64×64×64, w8, FullRow, s3` | measured / runtime_failed / measured | 一次严重离群，pending |
| `512³, 64×64×64, w4, default, s1` | measured / runtime_failed / measured | 一次严重离群，pending |
| `512³, 128×128×64, w4, default, s3` | runtime_failed / measured / measured | 一次严重离群，pending |

严格运行原始目录：

- `results/raw/p2/20260729T013313482287356Z`
- `results/raw/p2/20260729T013420813026040Z`
- `results/raw/p2/20260729T013502859906428Z`

每个严格运行的每个 case 均保留 15 条原始样本。被拒绝的 case 不是编译失败或数值失败；其 `summary.csv` 中有 `stability_gate=fail`、`max_latency_us` 和 `max_over_median`，可直接定位异常程度。

## 5. 审核修改

初始审核发现只用 `(p90-p10)/median` 会漏掉 15 个样本中单个严重异常。已作以下受控修改：

1. event batch 从 20 次 kernel 扩至 200 次，减少偶发事件对短 kernel 样本的影响；
2. 除 p10–p90 门槛外，新增 `max(sample)/median <= 2`；
3. 未通过任一门禁时 status 为 `runtime_failed`、`stability_gate=fail`，并清空 `effective_tflops`；
4. 生成代码特征由空 token 列表改为记录实际的 `gemm_ss`、TMA 和 MUSA async/barrier 文本特征。

shell/Python/JSON 静态检查以及 TileLang 工作树 `git diff --check` 已通过。

## 6. 后续 P2 工作

P2 尚未完成。下一批仍先做 FP16：补齐 square/large-K 的 shape curve，再覆盖 skinny/tall；每点沿用上述门禁。之后才独立开展 BF16、INT8、vector/elementwise、reduce、exp、rsqrt。任何不稳定点都留在 measurement 问题表，不进入 P5 参数化。
