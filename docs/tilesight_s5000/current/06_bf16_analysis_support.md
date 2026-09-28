# S5000 TileSight BF16 分析支持

更新日期：2026-09-19

## 状态

BF16 已作为显式计算精度接入 TileSight，不再通过“2 字节等同 FP16”推导计算能力。现有 FP16/FP32 调用未指定 `compute_profile_key` 时继续执行原有字节宽度路径。

当前实现覆盖：

- BF16 GEMM，BF16 A/B/C、FP32 accumulator；正式 compiler-info 适配首批限定已有 stage 1 路径。
- BF16 Add、ReLU 的 compiler-info 和直接分析入口；二者共用通用
  `bf16_vector_alu` 资源 Profile，算子差异由操作量表达。
- BF16 Exp、Log、Sqrt 的直接分析入口，以及 compiler-info 表达式分类和模型分发。
- dtype-aware Tensor/Vector/SFU profile、指令形状、利用率和结果精度元数据。
- `enable_bf16_profiles` 回退开关；关闭后 BF16 fail closed，不回退 FP16。

## 兼容边界

- `compute_profile_key=None` 保持既有 TileSight 计算路径。
- BF16 Vector 与 FP16 一样采用通用资源模型；旧的
  `bf16_vector_add`/`bf16_vector_compare_select` 仅作为兼容别名解析，
  新结果统一输出 `bf16_vector_alu`。
- `mem_levels` 仍只描述存储流量宽度，计算 dtype 由独立 profile 描述。
- 所有特殊数学都不再使用精度/算子专用 CTA overlap 或 dirty-retention 参数；
  统一保留 SFU 服务率 Profile。
- Pipeline stage 公式没有修改；此前 `formula_rejected` 结论保持不变。

## 证据等级

BF16 GEMM profile 使用现有 S5000 P2 匹配 schedule 数据作为首批证据。2026-09-19
已在 SDK 5.2 用户态环境中完成 FP16/BF16 配对服务探针，以及 20 个真实
TileLang Add/ReLU/Exp/Log/Sqrt 开发集和留出集 case。全部 case 通过正确性检查。

Vector 仍采用通用 `bf16_vector_alu`：真实流式 Add/ReLU 没有显示需要独立的
BF16 倍率。Exp/Log 未出现同时改善开发集与留出集的候选，因此保持不变。
Sqrt 的配对服务率和真实算子验证共同支持将
`bf16_sqrt_f32_sequence` 的有效服务率由 `645.12 Gop/s` 校准为
`900 Gop/s`；该改动只影响 BF16 Sqrt。统一采用完整逻辑输出写回口径后，
开发集和留出集平均绝对误差由 `78.89%` 降至 `62.22%`。此前
`27.49% → 10.83%` 是 Kernel 窗口诊断口径，不再作为正式结果。原始样本和
审核报告分别位于：

- `results/profiling/s5000_bf16_service_calibration_20260919/`
- `results/profiling/s5000_bf16_operator_validation_20260919/`

本轮使用 MUSA Event，并未把它描述为 MCU 指令计数证据。BF16 特殊数学不再
依赖 FP16 CTA overlap/dirty-retention；SFU 只使用统一服务率 Profile。

## 通用 Wave 边界验证（2026-09-20）

已用独立原生探针对 FP16/BF16 × Add/ReLU/Exp/Sqrt 执行精确 CTA 边界扫描。
所有 Kernel 的 Occupancy API 结果均为 `12 CTA/MP`，对应 S5000 的
`720 CTA/wave`；在 1/2/4/8/12 waves 的每个边界均测量 `−1/0/+1 CTA`，
每点包含 30 个 Event 样本和3个独立进程。

- 1/2/4/8 waves 的 `固定项 + 稳态斜率 × waves` 拟合 R² 全部超过 0.998。
- 8-wave 实测仅为 `8 × 第一波` 的约 28%～44%，不支持每个分析 Wave
  完整重新填充和排空。
- 边界 `+1 CTA` 没有出现接近完整第一波的阶跃；FP16/BF16 同算子斜率接近。
- 12-wave 的额外拐点与超出 60 MiB L2 的缓存/DDR 区间相关，必须与 CTA
  连续调度分开建模。

因此本轮确认问题具有通用性。边界证据报告在
`results/profiling/s5000_wave_boundary_validation_20260920/report.md`；后续已经
实现 opt-in 连续调度公式，其 A/B 结论见下节。冷缓存资源模型通过前，不替换
通用 Elementwise 默认路径。

随后将验证扩展到 FP16/BF16 × Copy/Add/Mul/ReLU/Exp/Log/Sqrt，并密集扫描
0.25～16 Wave。主实验每点 30 个 Event 样本、3 个独立进程，756 个进程采用
随机顺序，避免把设备时段变化误判为精度差异。按输入+输出工作集分段后：

- Copy/ReLU/Add/Mul/Exp/Sqrt 在 L2 内区间的仿射拟合 R² 均不低于 0.9978。
- 这 6 个稳定算子的 BF16/FP16 稳态斜率比为 0.985～1.013，且落在独立进程
  斜率波动范围内；没有证据支持 BF16 独立 Wave 公式。
- 不同算子的绝对斜率明显不同，说明只能共用连续调度结构，不能共用一个固定
  斜率；服务成本仍必须来自 Vector/SFU、IO 字节和 dtype Profile。
- Binary 工作集约在 7.1 Wave、Unary 约在 10.7 Wave 跨过 60 MiB L2；跨界后
  FP16/BF16 斜率同步增大，属于缓存/写回区间，不属于 BF16 调度差异。
- Log 在独立进程下非单调且高方差；增加到 100 次预热、100 次采样、5 个进程
  后仍未消除。其 12/14/16 Wave 大负载区 FP16/BF16 基本重合，但 Log 暂不进入
  Wave 参数拟合，后续需要固定频率或持续驻留进程验证。

扩展报告位于
`results/profiling/s5000_wave_shape_sweep_randomized_20260920/report.md`；Log 加强
原始数据位于 `results/profiling/s5000_wave_log_stabilized_20260920/`。结论是：
通用 Wave 结构修改是必要项，但不是充分项；缓存/写回转折和资源服务率仍需
独立保留与校准。

## Elementwise 默认延迟恢复为 TileSight 核心结果（2026-09-20）

对照官方 `tile-ai/DeepStack` 后确认，`memory_hierarchy_prediction` 和后置
`max(Pipeline/Wave, critical memory path)` 是本项目 S5000 报告层扩展，不是
上游 TileSight Kernel 延迟公式。TileSight 核心已经把 DDR/L2/SMEM 服务时间
纳入 tile、occupancy、pipeline 和 wave 计算。

因此 Elementwise 默认 `latency_us` 已恢复为 `PipelineResult.total_latency`；
`latency_with_memory_hierarchy_us`、L1/L2/DDR 流量、写回和异常诊断仍保留，但
明确标记为实验性报告诊断，不再覆盖默认延迟。分层报告和 Session 汇总也遵循
显式 `latency_source`。GEMM 路径本轮未改变。

使用既有 20 个 FP16/BF16 开发/留出 case 回放：

- 只有 4 个 ReLU case 的默认预测发生变化；Add 与 Exp/Log/Sqrt 不变。
- ReLU MAPE 从 35.707% 降至 31.061%。
- 全部 20 case MAPE 从 `41.350%` 降至 `40.421%`。
- BF16 10 case MAPE 从 `38.669%` 降至 `37.707%`。

这说明移除外层延迟覆盖没有破坏既有精度，且小幅改善误差；ReLU 剩余约
22%～40% 的高估仍属于后续通用 Wave/连续 CTA 调度问题。审核报告位于
`results/profiling/s5000_elementwise_latency_selection_audit_20260920/report.md`。

## 通用连续 Wave 公式试验（2026-09-20）

已增加精度无关的 `continuous_cta_dispatch` 路径。新公式把
`resident CTA/MP` 解释为并发窗口，而不是必须逐批排空的批大小：

```text
T = T_fixed_head
  + T_fill_once
  + T_steady(busiest CTA count)
  + T_drain_once
  + T_store(all CTA on busiest MP)
```

其中最忙 MP 的 CTA 数为 `ceil(total_cta / active_mp)`，所以越过
`k × 720 CTA` 的 `+1 CTA` 只增加一个 CTA 的尾部工作，不再增加完整 Wave。
资源时间仍来自现有 Vector/SFU、缓存流量和 dtype Profile；公式中没有 BF16
独立系数。跨 FP16/BF16 Add/ReLU/Exp/Sqrt 的 L2 驻留拟合给出一次性共同首波
项 `6.5 µs`。此前按 Exp/Log/Sqrt、FP16 形状单独拟合的 CTA overlap 修正已移除；
SFU 仍通过统一的 `special_math_service_profiles` 提供服务率参数。

A/B 结果显示：

- L2 驻留随机 Shape Sweep 的 100 个 case：MAPE 从 `44.199%` 降至
  `6.326%`；FP16 为 `44.547% → 7.213%`，BF16 为
  `43.851% → 5.438%`。
- 统一完整逻辑写回并移除旧算子专用修正后，既有冷缓存 20 个算子 case：
  MAPE 从 `41.350%` 降至 `33.976%`。
- Exp/Log/Sqrt 改善，但 Add/ReLU 分别回退 `12.473/21.977 pp`。这表明连续
  调度结构成立，但当前 Add/ReLU 冷缓存 DDR/L2 服务成本尚未与新结构解耦。

因此连续调度实现当前保持 opt-in：
`make_s5000_config(enable_continuous_elementwise_wave=True)`。默认仍使用已验证的
旧路径，不破坏 FP16/BF16 现有结果。审核报告位于
`results/profiling/s5000_continuous_wave_audit_20260920/report.md`。

## 冷缓存连续 CTA 内存服务斜率校准（2026-09-20）

在完整逻辑输出写回字节数保持不变、`max(memory, compute)` 保持不变的前提下，
对 continuous CTA 的 cold 内存稳态服务时间增加统一缓存状态参数：

```text
warm: 1.00
cold: 0.60
```

该参数只缩放连续 CTA 的内存服务斜率，不影响固定首波项、计算时间、写回字节数、
SFU profile 或默认 repeated-resident-batch 路径。0.60 来自现有冷缓存开发/留出
数据的通用拟合：开发集最佳约 0.596，留出集最佳约 0.612。

重新回放结果：

- L2 驻留 100 个 case：连续 Wave MAPE 保持 `6.326%`；
- 冷缓存 20 个 case：`41.350% → 8.170%`；
- FP16：`44.993% → 7.896%`；BF16：`37.707% → 8.444%`；
- Add：`22.784% → 12.311%`；ReLU：`31.061% → 2.005%`；
- Exp/Log/Sqrt：分别为 `14.649%`、`8.758%`、`3.126%`。

本轮只在 continuous CTA opt-in 路径启用该参数，S5000 默认路径保持不变。完整
结果位于 `results/profiling/s5000_continuous_wave_audit_20260920_memory_slope/`。

## 追加 Elementwise 候选算子验证（2026-09-20）

既有 Shape Sweep 候选集合为 `Copy / ReLU / Add / Mul / Exp / Log / Sqrt`；
其中 Log 保留独立稳定性限制。本轮新增了 Copy 和 Mul 的通用 Vector 分析入口，
复用 FP16/BF16、warm、完整逻辑写回和 continuous CTA 口径。

- Copy：L2 内稳态 MAPE 为 FP16 `11.716%`、BF16 `14.618%`；跨 L2 后为
  `15.532%`、`15.093%`。
- Mul：L2 内稳态 MAPE 为 FP16 `2.443%`、BF16 `2.734%`；跨 L2 后为
  `1.828%`、`1.718%`。

Mul 与现有通用 Wave 公式较好吻合；Copy 在跨 L2 转折区仍有较大残差，暂不
增加 Copy 专用常数，继续归入缓存/服务斜率验证。详细结果位于
`results/profiling/s5000_extended_elementwise_wave_audit_20260920/`。

## 验证

- FP16/BF16 服务探针覆盖 2 种 dtype、6 种序列、5 档 ILP、3 档迭代数，
  另以不同迭代数和 blocks/MP 完成独立留出。
- 20 个真实 TileLang operator-level case（开发/留出 × 5 算子 × 2 dtype）
  全部通过正确性检查。
- BF16 Add/ReLU 静态模型统一生成 `bf16_vector_alu`；Exp、Log、Sqrt
  仍按不同特殊数学序列生成各自的服务 profile key。
- Python AST 检查和 `git diff --check` 通过。

## P2 实测对比

已增加 `scripts/compare_bf16_tilesight_to_p2.py`，按相同 shape、tile、warp
和 stage 将 TileSight BF16 GEMM 输出与 P2 三进程实测聚合对齐。生成物位于
`results/profiling/tilesight_bf16_p2_comparison_20260917/`。

该对比属于拟合内回放：BF16 compute profile 本身引用了 P2 匹配 schedule 作为
证据，不能当作独立留出准确率。`bf16_square_512` 仍是短核参考项，不进入参数
有效性结论；下一步需要用未参与 profile 建立的新 shape 做真实硬件留出验证。
