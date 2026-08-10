# S5000 Calibration 文档索引

本目录只保存方案、审核、实验协议、阶段报告和结论，不保存 benchmark 源码或原始测量数据。

## 当前有效文档

| 编号 | 文档 | 状态 | 用途 |
|---|---|---|---|
| 01 | `01_s5000_tilesight_calibration_plan.md` | 需求基线 v1 | 保存原始目标、数据契约、范围和交付物。 |
| 02 | `02_s5000_tilesight_calibration_review_v2.md` | 当前实施方案 | 记录审核结论、7 阶段执行方案和阶段门槛。 |
| 03 | `03_p0_backend_environment_audit.md` | completed | 保存 P0 原生 MUSA、TileLang-MUSA 编译/运行、MP31 兼容修补与数值验证证据。 |
| 04 | `04_p1_measurement_infrastructure_audit.md` | completed | 保存 P1 计时、正确性门禁、原始数据契约、三次稳定性 smoke 与审核结论。 |
| 05 | `05_p2_compute_initial_fp16_gemm.md` | superseded | 保存旧 P2 初始 FP16 GEMM 数据，仅作测量污染证据。 |
| 06 | `06_p2_fp16_square_sweep.md` | superseded | 保存旧固定 schedule sweep，仅作离群与旧计时口径证据。 |
| 07 | `07_p2_gemm_outlier_root_cause.md` | completed | 保存 256³/512³ 离群根因、原始数据与三种计时路径交叉证据。 |
| 08 | `08_p2_compute_completion_audit.md` | completed | 保存 FP16/BF16/INT8/vector/reduce/exp/rsqrt 三进程 P2 完成审核。 |
| 09 | `09_p3_memory_cache_shared_audit.md` | completed with limitations | 保存 DDR/cache/shared-memory 证据、未知项与 P4 进入条件。 |
| 10 | `10_p4_occupancy_wave_audit.md` | completed（含 P4.1/P4.2） | 保存 MUSA occupancy API、受控寄存器压力、原生 CTA/wave、TileLang fixed-grid/pipeline 资源和 compute-wave 尾波审计。 |
| 11 | `11_p5_parameter_model_audit.md` | completed for exact-condition development | 保存版本化 P2–P4 参数注册表、完整条件模型、资源/尾波边界与 P6 限制。 |
| 12 | `12_p6_holdout_integration_review.md` | completed — evidence insufficient | 保存冻结 P5 后的四类 hold-out 编译/正确性/性能证据、覆盖度判断与 TileSight 接入决定。 |
| 13 | `13_p5_1_transfer_development_protocol.md` | completed for bounded P5 v2 development | 保存 P5.1 数据隔离、transfer 规则、Graph runtime-failure 处置、v2 冻结和审核。 |
| 14 | `14_p6_v3_holdout_integration_review.md` | completed — guarded experimental narrow-domain integration | 保存全新 P6-v3 的事前预测、实测误差、接入边界与旧 P6 只读回顾。 |
| 15 | `15_p7_large_gemm_runtime_diagnostic.md` | completed — runtime blocker retained | 保存大 FP16 GEMM 的 graph/普通 launch 失败复现、定位证据、不可拟合结论与 P5.2 恢复条件。 |
| 16 | `16_p0_p7_correctness_and_root_cause_review.md` | completed static audit — dynamic A/B pending | 汇总 P0–P7 有效性、确定勘误、P7 静态根因排序和安全复验矩阵。 |
| 17 | `17_s5000_measured_data_human_readable.md` | reviewed human-readable snapshot | 汇总 S5000 实测 compute、memory/cache、occupancy/wave、hold-out 数据及明确 unknown。 |
| 18 | `18_p7_to_p9_execution_plan_and_review.md` | reviewed execution plan | 定义 runtime A/B、计时修复、GEMM 数据、cache/resource/tail、P8 adapter 与 P9 排名验证。 |
| 19 | `19_p7_1_recovery_gate_and_ab_preparation.md` | blocked before GPU launch | 保存恢复门禁失败证据、静态 A/B 资产、审计和恢复条件。 |

## 后续文档编号约定

- `03_...`：P0 后端与环境审计报告；
- `04_...`：P1 测量基础设施协议与验证报告；
- `05_...` 起：计算、memory/cache、occupancy/wave、模型与 hold-out 报告；
- 重要结论必须链接对应源码、配置、日志和原始数据，不能只写汇总数字。

## 文档状态规则

- `draft`：仍可能改变实验设计；
- `approved`：可以进入对应实施阶段；
- `completed`：阶段验收门槛全部通过；
- `blocked`：保留失败证据和恢复条件；
- `superseded`：被更新文档替代，但不得删除。
