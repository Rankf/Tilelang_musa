# Elementwise 算子分批追踪

更新日期：2026-09-20

原始的《TileSight_Elementwise算子分批测试方案.md》当前工作区中缺失。本页根据
TileOPs Elementwise Manifest 重新建立可审计的批次追踪表。下面的“第二批”和“第三批”
是恢复后的执行分组，不把它们表述为原始文档的逐字恢复。

## 选择范围

候选全集来自：

- `TileOPs/src/tileops/manifest/elementwise_unary_math.yaml`
- `TileOPs/src/tileops/manifest/elementwise_unary_activation.yaml`
- `TileOPs/src/tileops/manifest/elementwise_binary.yaml`
- `TileOPs/src/tileops/manifest/elementwise_multi_input.yaml`
- `TileOPs/src/tileops/manifest/elementwise_fused_gated.yaml`
- `TileOPs/src/tileops/manifest/elementwise_generative.yaml`

Copy 不在上述官方 Manifest 中，保留为自定义内存探针；它不能作为官方 Elementwise
批次算子计数。每个批次仍需分别记录 Manifest workload、编译签名、运行期数据和
TileSight 比较结果。

## 批次状态

| 批次 | 算子/算子族 | 选择目的 | 当前状态 |
|---|---|---|---|
| 第一批（历史） | ReLU、Add、Exp、Log、Sqrt | 基础 Vector、双输入 Vector、SFU | 已有模型和 S5000 运行数据；Log 仍有非单调稳定性限制 |
| 第二批 A | Mul、Sub、Abs、Neg、Maximum、Minimum | 扩展通用 Vector unary/binary，验证统一 Wave 结构 | Mul、Sub、Abs 已有 FP16/BF16 warm Shape Sweep；Neg/Maximum/Minimum 待补同签名实测 |
| 第二批 B | Rsqrt、Expm1、Sigmoid、Tanh、Erf | 扩展 SFU/复合特殊数学，验证 per-op SFU profile | Expm1 有独立算子工作，但尚未并入本轮 TileSight Wave 误差验收；其余待测 |
| 第三批 A | Div、Pow、FloorDivide、Lerp、PReLU | 验证除法、广播和输入访问模式 | 候选，尚无本轮统一 Wave 实测 |
| 第三批 B | GELU、SiLU、HardSwish、HardSigmoid、Mish、Softplus、Clamp | 验证复合激活和多指令 Vector/SFU 混合 | 候选，需先固定 lowering 和输入范围 |
| 第三批 C | Where、SiLUAndMul、GELUAndMul、GELUTanhAndMul、Alibi、Sinusoidal | 验证多输入、融合和生成型 Elementwise | 候选，需单独定义流量、广播和融合签名 |
| 第三批 D | Eq、Ne、Gt、Lt、Ge、Le、LogicalAnd、LogicalOr、BitwiseAnd、BitwiseOr、BitwiseXor | 验证比较、谓词和整数/布尔流量 | 候选，需补充输出 dtype 与混合类型流量定义 |
| 第三批 E | Sign、Floor、Ceil、Round、Trunc、LogicalNot、BitwiseNot、Isnan、Isinf、Isfinite、NanToNum、MaskedFill、LerpTensor | 覆盖低算术量、谓词、掩码和多输入变体 | 候选，暂不进入首轮校准 |

## 当前优先测试集

本轮先完成了第二批 A 的 `Sub / Abs`，并对已测 `Mul` 保留后续 cold 和 L2 边界
任务。Sub/Abs 可以复用现有 Vector Wave 结构，不引入新的公式分支；测试重点是：

1. FP16/BF16 同一 Shape、同一 Block/线程配置；
2. warm/cold 两种缓存状态；
3. L2 容量边界前后 `-1 / 0 / +1 CTA`；
4. 完整逻辑输出写回字节和 Kernel 窗口物理流量分别记录；
5. 统一比较固定项、稳态 Wave 斜率和跨 L2 转折。

第二批 B 的特殊数学算子在 Vector 集合完成后进行，避免把 SFU lowering 差异和通用
Wave 调度误差混在一起。第三批先完成静态 Manifest、编译签名和流量定义，再安排实机
采集。

## 证据状态定义

- `模型`：TileSight 能生成结构化预测，但没有同签名硬件数据。
- `warm sweep`：已有 S5000 Event Shape Sweep，可用于稳态 Wave 结构审核。
- `cold pending`：缺少对应算子的冷缓存和 L2 边界验证。
- `validated`：模型、编译签名、运行期窗口和统计口径一致，误差结论可用于校准。

当前第二批的 `Mul` 只有 `warm sweep` 状态，不能直接写成 `validated`；Copy 的结果
继续放在自定义探针报告中。

## Sub/Abs 首轮实测结果（2026-09-20）

S5000 warm Event Shape Sweep 已完成：FP16/BF16、0.25～16 Wave、每点 30 个
样本、3 次独立重复，共 216 条记录。结果文件：

`results/profiling/s5000_second_batch_sub_abs_20260920/`

| 算子 | 精度 | L2 内连续 Wave MAPE | L2 外连续 Wave MAPE | 过渡区连续 Wave MAPE |
|---|---|---:|---:|---:|
| Sub | FP16 | 2.327% | 1.719% | 12.573% |
| Sub | BF16 | 3.690% | 1.832% | 12.285% |
| Abs | FP16 | 8.684% | 15.528% | 20.441% |
| Abs | BF16 | 12.680% | 15.582% | 23.827% |

Sub 与 Add/Mul 的误差形态一致，支持继续共用 Vector Wave 结构。Abs 在 warm
Shape Sweep 上的误差明显更高，尤其是 L2 过渡区；先归因于 unary 内存服务和缓存
转折差异，不增加 Abs 专用常数。两种精度没有显示出独立 Wave 公式证据。
