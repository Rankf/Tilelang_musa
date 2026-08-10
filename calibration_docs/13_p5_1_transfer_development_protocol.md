# P5.1：跨条件 Transfer 规则开发与 P6-v2 预注册协议

**状态：completed for bounded P5 v2 development；P6-v3 接入结论见文档 14。**

## 目标

P5 v1 的完整条件查表是可信校准注册表，但不能预测未见的 shape、tile 或 stride。本阶段只在
独立 development 集中开发有限、可解释且有适用域的跨条件规则；它不修改 P5 v1，也不读取
原 P6 的 raw/processed 结果。随后使用本文件签署前已经预注册的 P6-v2 集作唯一的新增接入
验证。

## 数据隔离

- development compute：`configs/p5_1_transfer_development_compute.json`；
- development layout：`configs/p5_1_transfer_development_layout.json`；
- P6-v2 hold-out：`configs/p6_v2_compute_holdout.json` 和
  `configs/p6_v2_layout_holdout.json`，登记于 `configs/p6_v2_holdout_manifest.json`；
- 原 P6 四例是已观察的历史结果，只能在 P5 v2 冻结后作回顾性诊断，不能进入拟合或修改域。

P5.1 builder 的输入清单必须显式列出，且 validator 必须确认其中没有 `p6_holdout` 或
`p6_v2_holdout` 路径。每一个新 development case 仍按编译、数值正确性、稳定性计时的顺序，
在三个独立进程中运行。

在首批 development 执行期间、P5 v2 拟合和任何 P6-v2 执行之前，对特征域作静态审阅发现
P6-v2 的 GEMM/reduce 派生工作量会略超出首批 development 最大值。因此增加
`p5_1_transfer_development_extension_compute.json`：1536x1024x1536 FP16 GEMM 和
8192x1024 reduce。两者均不等于任何 P6/P6-v2 条件；该扩展不修改公式、hold-out 配置或
判定规则，仅避免将 P6-v2 当成外推测试。

该 extension 的 100-kernel MUSAGraph 回放出现间歇性 `operation not permitted` reset：单次
执行和数值正确性通过，但失败进程不产生性能参数，且会使同进程后续 case 无效。保留全部
失败和偶发成功日志，不将其混合为三进程 aggregate。替代配置
`p5_1_transfer_development_extension_graph20_compute.json` 使用相同 kernel、输入与 profiler
交叉检查，但固定为 20-kernel graph replay；它必须重新取得三次完整成功才可作为 P5 v2 输入。

实际复测表明 graph-20 仍在同一 Graph replay 步骤失败，故两个 extension 配置均保留为
runtime-failure 证据，**不进入 P5 v2**。原 P6-v2 的 GEMM/reduce 条件因此处于未经验证的
外推域，保持预注册但不执行。为避免以失败数据延展域，另在 P5 v2 拟合前预注册 P6-v3：
`p6_v3_*` 的 GEMM/reduce 派生特征均落在首批稳定 development 范围内，仍与所有 development
case 和旧 P6 case 不同。P6-v3 是本轮唯一的接入判定集。

初次 development 残差还显示：4M fused 的短规模点不能代表 8M 以上的线性吞吐，且只用
stride 1/4/8 对 8M 的中间 stride 做留一插值误差过大。因此在 P6-v3 执行前补充
`p5_1_transfer_refinement_compute.json`（12M、20M fused）和
`p5_1_transfer_refinement_layout.json`（8M、stride 2/3/5/7）。这只加密 development 采样：
不测 24M fused 或 stride6，不改公式、P6-v3 配置或模型接入判定规则。

构建时 FP32 fused 的 v2 适用域固定为 `N >= 8M`：4M 是已知的短规模 regime，留在原始
development 数据和报告中，但不用于 steady fused 线性项；这与 P2 对短 kernel 固定开销未
分离的既有限制一致。P6-v3 的 24M 在该域内。

## 结果与冻结

首批 development 和 refinement 的所有纳入 case 都取得三次独立、编译/正确性/稳定性通过的
测量。P5.1 extension 的 1536x1024x1536 GEMM 与 8192x1024 reduce 虽均能编译且单次正确，
但在 MUSAGraph replay 反复出现 `operation not permitted` runtime reset；graph-100 和 graph-20
两种口径均不能形成三次成功测量。因此其 raw/log 保留为 backend/measurement failure 证据，
不参与任何模型参数。

冻结产物：

- 模型：`results/processed/p5_1_transfer_model/s5000_single_gpu_model_v2.json`；
- SHA-256：`b94296eb50b904b0e34b80d79535887ad7932b7b08f57fe736c609eb48caa9be`；
- development aggregate：`development_aggregate.csv` / `.json`；
- 逐点 development 残差：`development_predictions.csv`；
- provenance、隔离和公式审核：`scripts/p5_1_validate_transfer_model.py`（通过 19 条线性
  公式重算，且 source manifest 不含 P6/P6-v2/P6-v3 路径）。

模型只包含本文预先声明的非负、条件化经验规则：FP16 GEMM 的 operation/grid 特征项、FP32
reduce 的 input/CTA 特征项、8M–32M FP32 fused steady 项，以及相同 selector 的 log2(stride)
有效带宽插值。它不把这些项解释为 S5000 的寄存器、cache 或 matrix fragment 规格。

## 预先固定的规则候选与域

不根据 hold-out 结果挑选公式。以下公式在测量前固定；若系数不满足物理单调性或开发拟合
审计失败，则对应 family 保持 `unknown`，而不是替换为更有利的规则。

| family | 固定规则 | 允许域 |
|---|---|---|
| FP16 GEMM | `latency_us = b0 + b1*ops_giga + b2*ops_giga/grid_cta`，三个系数均须非负 | default policy、64x64x64、4 warp、stage 1、M/N/K 均为 64 的倍数，且 `ops_giga`、`grid_cta` 落在 development 范围内 |
| FP32 reduce-sum | `latency_us = b0 + b1*input_million + b2*(M/4096)`，三个系数均须非负 | 128 threads，M/K 及两个派生特征均落在 development 范围内 |
| FP32 fused `exp(A)+B` | `latency_us = b0 + b1*N_million`，两系数均须非负 | 256 threads、vector 8，N 落在 development 范围内 |
| FP32 warm strided copy | 在相同 N/threads/vector/cache 条件下，对相邻实测 stride 的 `log2(stride)` 进行线性有效带宽插值 | 仅插值、不外推；两端都必须是稳定且正确的 development 条件 |

这些是经验性 S5000/当前 TileLang-MUSA 版本条件模型，不主张 register、cache 或矩阵
fragment 的未测硬件细节。GEMM 的 `grid_cta` 为源代码可见的 launch grid；不把它解释为已知的
MP residency 或 wave 数。

## 通过与接入规则

P5 v2 只在所有训练点符合 measurement gate、模型 SHA 已冻结、并且生成逐点开发残差时发布。
P6-v2 预检必须在测量前对全部四点输出数值预测；否则最终为 coverage insufficiency。P6-v2
不采用预设的 MAPE 通过线：接入决定基于四点实测误差是否小、方向是否稳定、是否存在明确的
shape/layout 系统偏差及模型适用域是否足够窄且可解释。
