# P5 S5000 单卡参数注册与初版模型审核

状态：completed for exact-condition calibration/development；未进行 hold-out 验证  
日期：2026-07-29  
模型版本：`s5000-single-gpu-v1`

## 结论

P5 已完成 P2–P4 参数冻结、原始 run 追溯、完整条件键选择与可调用的精确条件预测器。它是单卡校准域模型：只有条件键完全匹配时才返回有效率和预测；未见 shape、tile、working set、layout 或 source 则返回 `unknown`。因此没有把 calibration lookup 的重建误差当作泛化精度，也没有使用全局残差修正。

该状态符合 P5 的“参数化与初版模型”门槛，可进入 P6；但不构成接入 TileSight 的结论。

## 输入冻结与追溯

模型 manifest 记录全部输入文件的 SHA256，并额外记录 P5 scope SHA、builder SHA、生成时间、Python 版本和平台。参数注册表有 89 行；每行都有 phase、状态、完整适用条件、evidence 与 `raw_runs_json`。三项 driver 查询参数使用对应 runtime-query 日志作为追溯路径，其余参数都链接到三进程 raw run 或受控 resource-query run。

P3 不再只导入摘要参数：global、fixed-grid cache、read、shared 以及 pointer-chase 五组 aggregate 均被冻结。每个 memory rate 的选择键保存完整 case JSON，包含工作集、cache condition、stride、offset、threads、vector 或 shared access pattern；模型不再按单一 `op` 混合不同 cache/workset 条件。

## 模型形式

GEMM 条件有效率项：

`effective_rate_us = 2*M*N*K / exact_condition_TFLOP_per_s / 1e6`

非 GEMM 条件有效率项：

`effective_rate_us = work_items / exact_condition_Gitems_per_s / 1e3`

memory 条件有效率项：

`effective_rate_us = logical_payload_bytes / exact_condition_logical_GB_per_s / 1e3`

其中非 GEMM 的单位换算已校正；例如 `vector_add_16m` 为 116.326 Gitems/s，不是 0.116 Gitems/s。

总时间结构仍为 `prologue + steady + tail + epilogue`。当前 P2/P3 数据只提供条件有效时间，因此 `prologue_us`、`epilogue_us` 和大部分 kernel family 的 `tail_wave_us` 保持 unknown，不把条件有效率伪装成已分离的硬件 steady 项。

P4.2 compute-wave 是唯一可使用 tail 区间的 family：256 threads、2048 dependent FP32 iterations、12 CTA/MP、720 CTA capacity 的 +1 CTA tail 为 `0.392–2.790 µs`，中位 `1.199 µs`。模型为三个测得边界输出该 interval component；copy 和 GEMM 不继承它。

## 资源分解与预测输出

`calibration_development_predictions_v1.csv` 有 48 条记录：P2 GEMM、vector/SFU/reduce、P3 各条件 memory，以及 P4.2 tail interval。45 条数值条件 lookup 可由同一行 effective rate 反算，重建 MAPE 为 0；另外 3 条是 interval tail component，不以单点误差表示。

对资源的处理遵循 source-match 规则。例如 FP16 1024³、64×64×64、stage 1 的 source SHA 与 P4 pipeline resource 查询匹配，输出 10 CTA/MP、600 CTA 条件容量、grid=256 CTA、full waves=0、tail CTA=256。其他 TileLang source 的 resident CTA、wave 与约束来源保持 unknown，绝不借用这一值。

`scripts/p5_predict_condition.py` 是 P5 的条件预测入口：精确条件匹配返回对应预测和证据；任一字段不同则返回 `unknown`。这正是 P6 前的保护边界。

## 已知 unknown 与 P6 要求

- physical cache tier 与工作集映射；
- scheduler 到 MP 的实际 CTA 映射；
- 跨 kernel family 的统一 tail-wave 常数；
- GEMM prologue/epilogue 的独立分解；
- 未测 TileLang source 的 resource/occupancy；
- 任意跨 shape/tile/layout 的有效率迁移规则。

P6 必须预注册不在 P5 精确条件表中的 GEMM、reduce、layout/stride、fused 或 attention case。它应通过预测器得到 `unknown` 或在明确新增的、独立冻结的条件规则下得到预测，并逐点报告实测、预测、误差符号、绝对百分比误差与资源项。P6 不得引用 P5 的 0% lookup reconstruction 作为验证准确率。

## 交付物

- scope：[configs/p5_scope.json](/root/s5000-calibration/configs/p5_scope.json)
- builder：[scripts/p5_build_model.py](/root/s5000-calibration/scripts/p5_build_model.py)
- predictor：[scripts/p5_predict_condition.py](/root/s5000-calibration/scripts/p5_predict_condition.py)
- validator：[scripts/p5_validate_model.py](/root/s5000-calibration/scripts/p5_validate_model.py)
- 参数：[s5000_parameter_registry_v1.csv](/root/s5000-calibration/results/processed/p5_model/s5000_parameter_registry_v1.csv)
- 模型：[s5000_single_gpu_model_v1.json](/root/s5000-calibration/results/processed/p5_model/s5000_single_gpu_model_v1.json)
- 条件预测：[calibration_development_predictions_v1.csv](/root/s5000-calibration/results/processed/p5_model/calibration_development_predictions_v1.csv)
