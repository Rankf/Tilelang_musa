# S5000 TileSight 当前状态与交接

更新日期：2026-09-16
工作目录：`/root/iie/ruankefan/Tilelang_musa`
基线提交：`a914e61 Archive updated TileOPs cache sweep evidence`

## 1. 目标与不可破坏的边界

当前目标是在 MTT S5000 上完成 TileLang/TileOPs 到 TileSight 的性能分析链路，按 5.1～5.15 输出编译期信息、模型预测、运行期采集、异常诊断和模型–实测比较。当前重点是第一批 Elementwise：Add、ReLU、Exp、Log、Sqrt。

继续工作时必须遵守：

- 不更新 GPU 驱动；MUSA SDK 5.2 命令继续通过 `/usr/local/bin/musa52-run` 执行。
- 不提交、不推送、不回退现有脏工作树。
- TileSight 模型、运行期实测、比较报告分开保存。
- `model_estimate`、`runtime_measured`、`derived`、`unavailable` 必须显式区分。
- 只有统计范围、Kernel 签名、Shape、dtype 和输入条件一致时才计算正式误差。
- 保留上游通用公式；S5000 经验适配必须可关闭、限定适用范围并标注参数来源。

### 新聊天直接接续点

不要重新实现 runtime API，也不要重新拟合 Log。当前已完成：

1. 5.1～5.10 均已有输出入口；其中 5.4、5.6、5.9、5.10 尚缺阶段/执行单元级硬件证据，不能把模型估算冒充实测。
2. Log 已从 `log_sequence` 迁移到统一 SFU per-op service profile，归档预测变化为 `0.0 µs`，开发集和回归留出集均通过。
3. stage 1/2/3 独立 MUSA 探针已完成，但多阶段公式状态是 `formula_rejected`；TileSight 原公式未修改。
4. 下一项公式工作应先隔离 stage 3 的 barrier/scheduler 增量，再独立验证 `stage_num * resident_cta` effective-depth。没有证据前不增加 stage 3 经验常数。

新聊天建议直接使用以下任务描述：

```text
继续 /root/iie/ruankefan/Tilelang_musa 的 S5000 TileSight 工作。
先读 docs/tilesight_s5000/README.md、current/01_current_status_and_handoff.md
和 current/05_pipeline_stage_validation.md。不要重做 Log 迁移，不改原 TileSight
Pipeline 公式。设计并运行受控 MUSA 探针，隔离 stage 3 的 barrier/scheduler
额外成本，并单独验证 resident CTA effective-depth；根据开发集与留出集证据决定
公式是否可修改。
```

## 2. 当前已打通的链路

### 2.1 编译信息与统一输入

- `tilelang_compiler_info_adapter.py` 将 TileLang 编译信息转成 TileSight 输入。
- 输入已覆盖算子、Shape、dtype、Grid/Block、线程规模、寄存器、共享内存和访存语义。
- `analyze_op.py` 负责建模与分层输出。

### 2.2 运行期采集与结果包

- Event、MCU、MSYS 可独立采集并建立统一结果包。
- 结果包保留 case 签名、统计范围、正确性、样本统计和原始证据哈希。
- 主要入口：
  - `scripts/collect_s5000_runtime_measurements.py`
  - `scripts/build_runtime_measurement_bundle.py`
  - `scripts/compare_tilesight_model_to_runtime.py`

### 2.3 诊断与比较

- `tools/tilesight_diagnostics/` 独立读取已保存的 TileSight JSON，执行 5.6～5.10 规则判断，不改写模型计算。
- 正式比较统一使用上游兼容的完整逻辑输出写回 `latency_us`；
  `kernel_scope_latency_us` 仅作为 S5000 诊断字段保留。
- `tilesight.runtime` 已提供正式运行时输入 API：校验 runtime-only bundle 和 workload 身份后，统一生成同口径比较及 5.6～5.10 model + runtime 联合诊断；预测对象保持不变。
- `scripts/analyze_tilesight_with_runtime.py` 是对应命令行入口；实现和验证见 [运行时接口实现与验证](02_runtime_interface.md)。
- 模型完整写回时延与 Kernel 窗口时延已在特殊数学路径中分开输出；旧 Add/ReLU 归档仍有单一 `latency_us` 历史文件，不能回溯性改写为实测。

## 3. 主要模型修正

### 3.1 Occupancy 与 Waves

- 256 线程 Elementwise 已从统一的 24 CTA/MP 修正为资源约束下的 12 CTA/MP。
- TileSight 预测约 `11.4 waves/MP`，MCU `launch__waves_per_multiprocessor` 约 `11.378`，基本一致。

### 3.2 stage=1 Pipeline 问题

上游基础公式使用：

```text
effective_pipeline_depth = stage_num * tiles_per_sm - 1
```

当 `stage_num=1`、`tiles_per_sm=12` 时，公式将 12 个 resident CTA 的并发过度解释成需要 11 层 fill/drain 的 Pipeline，导致 Exp/Sqrt 大 Shape 时延高估。

当前处理方式：

- 保留上游公式及其 legacy 分解。
- 已移除 Exp/Sqrt/Log 的算子专用 CTA 重叠修正；SFU 统一服务率 Profile 仍保留，
  所有精度和算子回到通用 Wave 结构。
- 使用 TileSight 原生 waves，memory/compute 服务时间及 MCU 数据进行有限范围的经验校准。
- `stage_num=1` 不等于整个 Kernel 串行；单 CTA 没有多级软件 Pipeline，但多个 CTA 仍可被硬件交错调度。

### 3.3 Log 路径

- Log 已使用统一 SFU 路径的 `special_math_op=log` 服务 profile；历史独立序列候选 `247.469e9 ops/s` 不是通用硬件 SFU 峰值。当前第一批 FP16 冷启动 Continuous CTA 交付使用 `200e9 ops/s` 的有限范围校准值，BF16 保留独立的 `247.469e9 ops/s` Profile。
- 旧 `log_sequence` 字段只保留兼容读取；迁移没有改变预测数值。
- stage 1/2/3 独立 SDK 5.2 探针已完成。低数学负载 stage 3 的 Kernel 相对时延门失败，当前公式保持不变并标记为 `formula_rejected`，等待同步/调度增量实验。

## 4. 第一批算子当前结果

### 4.1 Exp/Sqrt 修正后结果（历史 Kernel 窗口诊断）

| 算子 | Shape | 原 Kernel 口径 | 修正后 | MCU 实测 | 修正后误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp | 4096² | 53.39 µs | 48.80 µs | 51.214 µs | -4.71% |
| Exp | 16384² | 1154.14 µs | 813.18 µs | 796.129 µs | +2.14% |
| Sqrt | 4096² | 53.39 µs | 46.25 µs | 43.827 µs | +5.53% |
| Sqrt | 16384² | 1154.14 µs | 815.88 µs | 786.430 µs | +3.74% |

上述表格是历史 Kernel 窗口诊断结果；当前正式结果统一采用完整逻辑输出写回
`latency_us`。独立留出数据仍保留在
`results/profiling/tilesight_exp_sqrt_adapter_20260911/adapter_validation.json`，
不能直接作为新的完整写回误差门禁。

### 4.2 Log（历史 Kernel 窗口诊断）

| Shape | Kernel 窗口预测 | 完整逻辑写回模型 | MCU Kernel 实测 |
| --- | ---: | ---: | ---: |
| 4096² | 90.27 µs | 94.10 µs | 93.281 µs |
| 16384² | 1235.73 µs | 1239.91 µs | 1241.169 µs |

Log 新鲜留出 Shape 验证已通过，但结果只支持当前配置范围，不能扩大为所有 Log lowering 的通用结论。

### 4.3 Add/ReLU 写回

- 实机 L2/LLC 属性为 60 MiB，persisting 上限为 50 MiB。
- 普通 Store 及 `__stwb/__stcg/__stcs/__stwt(float4)` 均未证明能在目标 Kernel 结束前完成全部输出到 DDR 的物理写回。
- Add/ReLU 的后续驱逐对照实验支持“部分输出相关写入在后续 L2 驱逐路径中被观察到”，但不是逐字节的 Kernel 结束时 L2 驻留账本。
- 公开 SDK 5.2 和 mp_31 后端中尚未找到对普通 Global Store 具有“指定范围 clean/writeback 到本地 DDR + 完成等待”明确语义的可用闭环。

## 5. 必须使用的统计口径

| 字段/概念 | 含义 | 用途 |
| --- | --- | --- |
| `kernel_scope_latency_us` | 目标 Kernel 设备执行窗口 | 诊断项，不作为正式 TileSight 误差选择 |
| `kernel_scope_write_bytes` | 目标 Kernel 窗口内 MCU 物理 DDR 写入 | 只与同类 counter 比较 |
| `logical_output_write_bytes` | 由 Shape 和算子语义得到的逻辑输出 | 模型输入，不是硬件 counter |
| `deferred_writeback_bytes` | 模型值或对照组扣除后的候选量 | 诊断，必须标注证据类型 |
| `full_writeback_cost_us` / `latency_us` | TileSight 完整逻辑写回成本 | 正式预测口径；与 Event 比较时注明窗口差异 |
| `forced_writeback_interval_us` | 目标 Kernel 加人为驱逐的完整区间 | 只用于诊断，不是普通算子时延 |

## 6. 5.1～5.15 严格状态

| 指标 | 状态 | 当前证据与主要缺口 |
| --- | --- | --- |
| 5.1 编译期 Tile 元数据 | 已覆盖首批路径 | Grid/Block、tile、pipeline、资源与 ID 可输出；更复杂 lowering 仍需扩展验证 |
| 5.2 Tile 性能建模 | 已实现 | 延迟、带宽、计算/内存利用率和 waves 已输出 |
| 5.3 缓存行为预测 | 部分完成 | 冷/热、分层流量和基础命中率已有；复杂冲突、通用写回预测尚未闭环 |
| 5.4 运行期 trace | 基础采集已有，正式口径未完全闭环 | Event/MCU/MSYS 结果包可用；MUPTI Kernel Activity `device_busy` 的逐次 correlation 仍需完成 |
| 5.5 离线分析与报告 | 已实现 | JSON、Markdown、终端摘要和结果清单均有现有入口 |
| 5.6 Pipeline 瓶颈 | 联合 API 已实现 | 已增加同步判断及 MP/Shared 运行时 counter；显式同步的实测等待仍依赖阶段 trace |
| 5.7 跨层搬运异常 | 联合 API 已实现 | 逻辑、预测和实测流量分层；MCU 事务定义未确认前不自动判异常 |
| 5.8 缓存利用异常 | 联合 API 已实现、证据部分完成 | LLC 读取与 L2PU 混合口径分开；内部 counter 跨版本规则尚需固化 |
| 5.9 计算–IO 重叠 | 单阶段判定已闭环 | 单阶段 Elementwise 输出不适用；多阶段实际 overlap 和失败 TileOp 仍依赖阶段 trace |
| 5.10 负载不均 | 模型分布与 Waves 比较已实现 | 已输出理论每 MP CTA 分布、尾波闲置；仍缺逐 MP 实测时间分布 |
| 5.11 模型预测偏差 | 已实现 | 同口径误差超过 10% 时标记显著偏差和未捕捉硬件行为 |
| 5.12 性能指标获取 | 已覆盖当前粒度 | 编译信息、模型和运行期指标已分层保存 |
| 5.13 层级感知视图 | 模型/运行时并列表已实现 | 缺失层级保持空值；仍缺 CTA/SM/cluster 实测负载视图 |
| 5.14 结果导出 | 已实现 | 结构化 JSON、表格/报告、版本与 artifact manifest 可用 |
| 5.15 工作模式 | 已可分开执行 | 建模、采集、比较由不同入口组合，不强制统一 CLI |

## 7. 已否定的表述

后续报告不得继续使用：

- “`__stwt(float4)` 保证 Kernel 退出前完整写回 DDR”。
- “MCU 漏掉了 32 MiB 写入”。
- “Kernel 窗口未观察到的全部差额就是 Kernel 结束时精确留在 L2 的数据”。
- 将约 2 ms 的 256 MiB 驱逐 Kernel 时间算到 Add/ReLU 普通执行时延。
- 将 Log `247.469 Gops/s` 称为通用 S5000 Log 硬件峰值。
- 将 `stage_num=1` 说成“整个 Kernel 无并行、全串行”。

## 8. 下一阶段优先级

1. 完成 MUPTI Kernel Activity `device_busy` 与单次 launch 的 external/callback correlation，固化正式基线。
2. 统一所有旧路径的双时延 Schema；正式结果选择完整逻辑写回口径，Kernel 窗口值仅作诊断。
3. 已完成 Log 统一 SFU profile 迁移，保留兼容和回退开关。
4. 已增加真实 `stage_num>1` 独立探针；下一步隔离 stage 3 同步/调度增量，并验证 resident CTA effective-depth。
5. 完善 5.8～5.10 的硬件证据：稳定跨版本缓存 counter、实测 overlap、逐 MP 负载时间线；联合 API 已完成，不再另建重复入口。
6. 扩展算子、Shape、dtype 和线程配置，每个校准参数都保留独立留出集。
7. 在模型口径稳定后再开展真实 Kernel 优化，并用 MCU/MUPTI 证明实测加速。

## 9. 验证与工作树

2026-09-16 已完成的定向回归包括：

```text
39 passed（Log/Wave 适配定向测试）
40 passed, 1 skipped（分析、runtime API、Pipeline 探针及导入回归）
```

覆盖 Log/Exp/Sqrt 适配、S5000 服务与 Occupancy、同口径比较、诊断、运行期结果包以及 stage 1/2/3 探针；唯一 skip 是既有环境条件跳过，不是源码断言失败。真实 Exp 4096² API/CLI 验证通过。MP throughput 与 Shared load/store counter 的 S5000 实验见 [指标要求输出补充与实验验证](03_required_outputs_and_validation.md)。

工作树在本轮整理前已有 82 项状态记录（20 项已跟踪修改、62 项未跟踪），包含用户既有工作与当前阶段产物，本轮没有回退或覆盖。

## 10. 本轮清理记录

2026-09-16 仅清理可重建缓存：

- 44 个 `__pycache__` / `.pytest_cache` 等缓存目录。
- 370 个 `.pyc/.pyo` 等缓存或编辑中间文件。
- 清理前尺寸约 4,729,811 bytes（4.51 MiB）。

未删除：

- `results/` 中的 MCU 原始报告、JSON、命令、生成 Kernel 和哈希。
- `build/`、`logs/`、源码、测试、探针与历史文档。
- 任何 Git 已跟踪或未跟踪的有效开发文件。

## 11. 关键证据入口

- 第一批模型摘要：`results/profiling/tilesight_exp_sqrt_adapter_20260911/summary.json`
- Exp/Sqrt 校准与留出验收：`results/profiling/tilesight_exp_sqrt_adapter_20260911/adapter_validation.json`
- Log 校准摘要：`results/profiling/tilesight_log_adapter_enabled_20260909/summary.json`
- Log 独立留出：`results/profiling/log_shape_fresh_holdout_20260909/frozen_validation/report.md`
- Log 缓存口径：`docs/S5000_Log缓存命中率口径核验_20260910.md`
- Add/ReLU 后驱逐：`results/profiling/elementwise_deferred_writeback_probe_20260907/report.md`
- Store 策略：`results/profiling/elementwise_store_policy_probe_20260907/report.md`
- cache-clean 编译审核：`results/profiling/cache_clean_compile_audit_20260909/report.md`
- 历史主交接：`docs/archive/S5000_TileSight_新对话交接_20260908.md`
