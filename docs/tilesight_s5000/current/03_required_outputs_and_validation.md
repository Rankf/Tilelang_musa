# S5000 TileSight 指标要求输出补充与实验验证

更新日期：2026-09-16

## 范围

本轮只补技术指标原文明示的输出，不增加统一 CLI、自动校准、任意算子覆盖、Kernel 哈希门禁或其他自定义验收条件。

## 已补输出

### 5.6 Tile Pipeline 瓶颈

- 输出模型关键资源和运行时已采集资源利用率。
- 增加同步次数、同步代价和同步瓶颈判断。
- 第一批单阶段 Elementwise 明确输出：无显式同步，`not_applicable`，而不是缺失或推测同步瓶颈。
- 运行时接入 MCU MP throughput、Shared load pipe 和 Shared store pipe 三个指标。

### 5.9 计算–IO 重叠失败

- `stage_count < 2` 的单阶段 Elementwise 明确输出 `not_applicable_no_multistage_pipeline`。
- 多阶段 Pipeline 保留理论重叠率；没有阶段时间线时，实际 overlap 和失败 TileOp 仍输出 `unavailable`。
- 不再把单阶段 Elementwise 误报成“缺少重叠实测”。

### 5.10 负载不均与资源闲置

利用已有 Grid、MP 数量和 resident CTA/MP 输出：

- Waves/MP 和总 wave 数；
- 整体 resident slot 利用率；
- 尾波利用率与尾波空闲率；
- 空闲 resident slot 数；
- 理论每 MP CTA 数的最小值、最大值、均值和分布组；
- 理论分配是否均衡、是否存在尾波资源闲置。

这些字段均为 `model_estimate`。当前 MCU Waves 仍是 launch 级指标，不冒充逐 MP 实测分布。

### 5.11 模型预测偏差

- 对同口径可比较项输出显著偏差判断。
- 沿用项目现有独立留出验收门限：绝对相对误差超过 10% 时，标记 `significant_model_deviation` 和 `uncaptured_hardware_behavior=true`。
- 同时输出高估/低估方向和命中的指标；不同口径字段不进入判断。

### 5.13 层级负载视图

- 输出 compute、Shared、L1、L2、DDR 五层的模型利用率与运行时利用率并列表。
- 没有采集到的层级保持 `null`，不推测填充。

## S5000 实验验证

使用 MCU 1.3.0 在 S5000 上执行两个独立 Kernel 探针：

| 探针 | MP throughput | Shared load pipe | Shared store pipe | 结论 |
| --- | ---: | ---: | ---: | --- |
| FMA compute probe | 75.470414% | 0.000134% | 0.000050% | 计算 counter 可用，Shared 接近零 |
| Shared reduction probe | 9.415293% | 0.262336% | 0.259809% | Shared load/store counter 均返回非零 |
| ReLU FP32 正式探针 | 22.949745% | 0.028987% | 0.010870% | 完整指标组可在同一目标 Kernel 采集 |

三个探针正确性均为 pass。ReLU 同次采集还获得 L1 32.235533%、DDR 75.461385% 和 Waves 11.377778。L2PU throughput 返回 166.396563%，超出普通利用率的 0～100% 范围，因此原值保留，但不参与瓶颈排序。原始 `.mcu-rep` 和 SHA-256 清单位于：

`results/profiling/tilesight_required_output_probe_20260916/`

该实验证明更新后的完整 MCU 指标组可以用于 Kernel 聚合层面的 5.6 运行时瓶颈证据；超范围 counter 会被排除。实验不证明逐 Tile 同步等待或真实计算–IO overlap。

## 现有 Exp 4096² 输出复核

- 5.6：无显式同步，`synchronization_bottleneck=false`。
- 5.9：单阶段 Pipeline，结论为不适用。
- 5.10：12 waves；整体 slot 利用率 94.8148%；尾波利用率 37.7778%；448 个尾波 resident slot 空闲；理论最优分配为 28 个 MP 各 136 CTA、32 个 MP 各 137 CTA，因此 CTA 分配本身均衡，但存在尾波闲置。
- 5.11：Kernel 时延和 Waves 两个同口径项均未超过 10%，状态为 `within_threshold`。
- 5.13：旧归档运行时记录只有 DDR 利用率，因此其他运行时层级保持空值；后续按更新后的采集配置重采后可填充 MP、Shared、L1 和 L2。

## 仍无法由当前数据满足的原文要求

- 5.4：按 `pipeline_id/op_id` 归集的 Tile 维度运行期实测代价。
- 5.6：存在显式同步的 Kernel，其运行时同步等待时间。
- 5.9：多阶段 Pipeline 的实际 overlap rate、失败 TileOp 和调度原因。
- 5.10/5.13：CTA/SM/cluster 的真实运行时负载分布。

以上需要阶段或执行单元级 trace。Kernel 总时间、聚合 counter 和 launch-level Waves 无法可靠推导，因此本轮没有添加伪实测输出。
