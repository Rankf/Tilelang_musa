# P6：冻结模型的 Hold-out 验证与 TileSight 接入评审

**状态：completed — 结论为证据不足，当前不可接入。**  
**测量日期：2026-07-29（UTC+8）**  
**原始 run：** `results/raw/p6_holdout/20260729T043144812058724Z/`

## 1. 目的与不可变约束

P6 不再收集用于拟合 P5 的数据，而是在 P5 参数冻结后，用从未进入 P5
完整条件表的 kernel/schedule 条件检验模型能否预测。它遵守
`02_s5000_tilesight_calibration_review_v2.md` 中的 P6 定义：每点先编译、再数值
正确性、后计时，并报告实测、预测、误差和误差归因。

预注册范围在 `configs/p6_holdout_scope.json`。预检发生在任何 hold-out kernel
编译或执行之前，冻结 P5 模型为：

```text
results/processed/p5_model/s5000_single_gpu_model_v1.json
SHA-256 9ef664f1e27f78bcd99cceb458750cf02319d6ae68de8573438cec69a17aeec0
```

完成测量后重新计算的 SHA-256 相同；`p5_model_changed_after_preflight=false`。P6
没有修改 P5 模型、参数注册表或速率。每个类别分别在三个独立 Python 进程中运行。

## 2. 预注册 hold-out 与预检

| 案例 | 未见条件 | P5 预检结果 |
|---|---|---|
| `holdout_fp16_768x1280x1024` | FP16 GEMM, M/N/K=768/1280/1024, 64x64x64 tile, 4 warp, stage 1 | `unknown` |
| `holdout_reduce_4096x512` | FP32 reduce-sum, M/K=4096/512, 128 threads | `unknown` |
| `holdout_fused_exp_add_16m` | FP32 `exp(A)+B`, N=16,777,216, 256 threads, vector 8 | `unknown` |
| `holdout_stride3_16m` | FP32 strided copy, N=4,194,304, stride 3, warm, 256 threads, vector 8 | `unknown` |

`scripts/p6_preflight.py` 在 `preflight.json` 中记录四项均为
`no exact P5 calibrated condition; cross-condition transfer is forbidden`。这是刻意的
模型边界，不是将留出实测反向拟合进 P5 的理由。

## 3. 编译、正确性与实测结果

执行命令：

```bash
bash scripts/p6_run_all_holdout.sh
python3 scripts/p6_finalize_holdout.py
```

所有六个独立进程的 `summary.csv` 均存在，四个案例在三次进程中均为
`status=measured`、`correctness=pass`；跨进程相对跨度均小于 P6 的 15% 稳定性门槛。

| 案例 | 实测中位延迟 (us) | 三进程跨度 | profiler self-device 中位 (us) | 测量资格 | P5 预测 / 误差 |
|---|---:|---:|---:|---|---|
| FP16 GEMM | 19.747201 | 2.0084% | 19.296820 | pass | unknown / unknown |
| reduce-sum | 53.283801 | 0.3784% | 49.533680 | pass | unknown / unknown |
| fused exp+add | 143.870196 | 0.1228% | 141.664960 | pass | unknown / unknown |
| stride-3 copy | 26.684768 | 0.1014% | 27.940830 | pass | unknown / unknown |

因此没有计算 MAPE、带符号误差或“预测为零”的伪误差：P5 对这些完整条件没有数值预测，
而不是预测值为零。fused `exp(A)+B` 的编译与正确性通过，是 TileLang-MUSA 非 GEMM
融合路径可工作的后端证据，但不能单独证明成本模型具有泛化能力。

原始样本、每进程 profiler 事件、编译/运行日志与配置均保留在上述 raw run；生成的 MUSA
源码和 mcc 命令在 `logs/build/p6_holdout/20260729T043144812058724Z/`。汇总产物为：

- `results/processed/p6_holdout/holdout_results.csv`
- `results/processed/p6_holdout/holdout_results.json`

测量前 `mthreads-gmi` 能确认设备、驱动与显存空闲，但其进程查询显示
`Failed to query running processes`，故不将其作为“无并发进程”的证据；测量后报告为
`No running processes found`。性能稳定性结论仅依赖三次独立进程的实测数据。

## 4. 误差归因与接入决定

所有四个样本的编译、数值和计时门槛通过，故没有 backend/codegen、correctness 或测量
不稳定失败。阻塞点是 P5 的设计：它只接受完整条件的 exact lookup，并明确禁止跨
shape、tile、op 或 layout 的速率迁移。因此四个真正留出条件全部是
`model_coverage_unknown`。

**P6 最终决定：`evidence_insufficient_p5_has_no_holdout_coverage`。当前不能将该 P5
模型接入 TileSight/TileLang cost model，亦不能声称只在某个泛化域可接入。**

这不是 S5000 性能参数失效的结论；P2–P4 的精确条件测量仍有效。它仅说明尚未建立并验证
“从已校准条件迁移到新 TileLang schedule”的预测规则。

## 5. 下一步（新的开发/校准阶段，而非修改本次 P6）

1. 另建 development/calibration 分区，针对 GEMM shape/tile、reduce、fused SFU 与
   stride/layout 设计跨条件 transfer rule；规则应基于资源查询和额外测量，而非 P6 数据。
2. 对相应 TileLang kernel 收集可审计的生成代码、block/grid、寄存器/SMEM/occupancy 证据，
   将 rate、wave 与布局折减写成有适用域的模型项。
3. 冻结新模型与新的 hold-out 集，再重新执行 P6；只有预检能在测量前产生数值预测，才可计算
   MAPE、符号误差和接入域。

