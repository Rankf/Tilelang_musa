# 性能分析工具技术指标 vs 实际实现对照表

## 3.1 功能指标

| 指标项 | 招标技术要求 | 实际实现 | 实现状态 |
|:-------|:------------|:---------|:--------:|
| 工具功能定义 | 满足附件一《性能分析工具/调试工具功能定义》 | 以 tile 为基本视角，覆盖编译期元数据提取、Tile 性能建模、缓存预测、运行期 trace 采集、离线分析报告，支持建模/采集/交叉校准三种模式 | ✅ 已实现 |

## 五、附件一：性能分析工具功能定义

| 序号 | 指标项 | 招标技术要求 | 当前实现 | 关键证据 | 实现状态 |
|:----:|:-------|:------------|:---------|:---------|:--------:|
| 5.1 | 编译期 tile 元数据与 pipeline 提取 | 提取 tile 操作类别、硬件层级、数据依赖关系，构建 tile pipeline 结构 | TileLang 编译生成 `.mu` 源码，含 `__launch_bounds__`、`T.Pipelined` 多 stage 配置，`mcc -resource-usage` 输出寄存器/共享内存用量 | `benchmarks/p2_tilelang_fp16_gemm.py`、`benchmarks/p4_run_tilelang_pipeline.sh` | ✅ 已实现 |
| 5.2 | Tile 性能建模 | 基于 tile pipeline 和目标硬件能力描述，估算 tile 操作代价、跨层级搬运代价及关键路径 | DeepStack 核心引擎，`s5000.py` 含 18 个硬件参数，`analyze_op.py` 可查询任意 shape/tile 的延迟、利用率、瓶颈预测 | `analyze_op.py`、`s5000.py`、`P6 hold-out MAPE=4.39%` | ✅ 已实现 |
| 5.3 | 缓存行为预测 | 基于 tile pipeline 访问模式与缓存层级结构，预测 tile 维度缓存命中率与搬运行为 | 已输出普通L1 eligible/downstream、TMA bypass、L2、DDR和Shared分层流量、命中率与代价；warm小工作集采用99.75%校准值及范围 | `memory_hierarchy_prediction`、P9 L1/TME报告 | ⚠️ 基本实现，复杂reuse-distance待完善 |
| 5.4 | 运行期 tile-aware trace 采集 | 运行期将硬件计数器与运行时事件按 tile pipeline 结构归集 | MUSA event 计时、MUSAGraph replay、profiler self-device time、P7 Runtime A/B 测试全部可用，三进程独立计时门禁 | `benchmarks/p2_tilelang_fp16_gemm.py`、`scripts/p7_1_run_runtime_ab.sh` | ✅ 已实现 |
| 5.5 | 离线分析与报告 | 按 tile op 类别、硬件层级、scale unit、kernel 维度生成热点排序、时间线与层级感知视图 | 25 份 `calibration_docs` 文档覆盖 P0~P8 全阶段，`analyze_op.py` 输出 JSON 报告，CSV 聚合数据，含跨进程跨度、正确性门禁 | `calibration_docs/`、`analyze_op.py`、`results/processed/` | ✅ 已实现 |
| 5.6 | Tile pipeline 瓶颈 | 通过 tile 维度代价模型与实测对照，定位 pipeline 中计算/搬运关键路径，判断计算/带宽/同步瓶颈 | DeepStack 预测标注 `compute_util_pct` 和 `ddr_util_pct`，`analyze_op.py` 输出含瓶颈类型。P7 大 GEMM 诊断确认计算瓶颈 | `analyze_op.py` 输出 `compute_util_pct`、`ddr_util_pct`、P7.3 pilot 验证 | ✅ 已实现 |
| 5.7 | 跨层级搬运异常 | 通过 tile 维度搬运量与代价聚合，发现跨层级流量超预期、搬运效率偏低 | `transfer_anomaly_analysis` 仅基于Tile pipeline和硬件Profile，结构化输出层级守恒、搬运放大和service pressure；区分normal/warning/anomaly/not_applicable | `analyze_op.py`、`tests/test_analyze_op.py` | ✅ 已实现，不依赖实测输入 |
| 5.8 | 缓存利用异常 | 通过缓存行为预测与实测对照，发现缓存命中率偏低、流量分布不均、跨层级搬运超预期 | 大小工作集带宽对比可反推缓存命中，P3 8/16 MiB vs 64/128 MiB 有明确带宽差异。但 L2 工作集边界标注为 "unknown" | `s5000_memory_parameters_provisional.json` 中 `candidate_effective_cache_workset_boundary: unknown` | ⚠️ 部分实现 |
| 5.9 | 计算-IO 重叠失败 | 评估计算阶段与数据搬运阶段在 pipeline 上的实际重叠程度，定位 overlap 失败或不足的 tile 操作 | P4 pipeline stage 扫描（1~4 stage）找到最佳 stage=2，`PipelineDetail` 输出 `mem_time_per_iter` 和 `compute_time_per_iter` 可判断重叠 | `aggregate_tilelang_pipeline.csv`、`PipelineDetail` 包含 `mem_time` 和 `compute_time` | ⚠️ 部分实现 |
| 5.10 | 负载不均 | 通过 scale unit 维度代价聚合，发现 CTA、cluster 或更粗层级负载不均与资源闲置 | P4 wave 扫描精确测量了 CTA 边界尾波开销：720→721 CTA 时延 +2.57%，1440→1441 时延 +3.87%，compute-dominant kernel 测得 0.392~2.790 µs 尾波区间 | `aggregate_tilelang_wave.csv`、`aggregate_tilelang_compute_wave.csv` | ✅ 已实现 |
| 5.11 | 模型预测偏差 | 当 cost model 预测与实测出现显著差异时，标记模型未捕捉的硬件行为，辅助模型校准与版本演进 | P6 hold-out 验证 MAPE=4.39%，大 GEMM 1.0% 误差（已确认），小 GEMM 22~26% 误差（已标记需校准）。P8 完成 7 项硬件参数校准 | `06_p6_v3_holdout_integration_review.md`、`P8` 校准文档、`s5000.py` 更新历史 | ✅ 已实现 |
| 5.12 | 性能指标获取 | 支持 tile、pipeline、硬件层级聚合、缓存层级、Kernel 与 session 级元数据等层级的性能指标获取 | P2~P8 数据覆盖每个层级：tile 级（pipeline stage）、硬件级（occupancy）、缓存级（工作集扫描）、kernel 级（GEMM/elementwise）、session 级（三进程聚合） | `results/processed/` 下各层级聚合数据 | ✅ 已实现 |
| 5.13 | 层级感知报告视图 | 支持跨层级搬运视图、缓存行为视图、层级负载分布视图等 | 架构图、跨层级搬运视图、缓存行为视图在 `calibration_docs` 和项目总结 `S5000_DeepStack_Project_Summary.md` 中均有体现 | `S5000_DeepStack_Project_Summary.md` 架构图、各阶段 audit 文档 | ✅ 已实现 |
| 5.14 | 结果导出 | 支持结果导出为可被通用性能可视化工具消费的格式 | 所有 benchmark 输出 CSV，`analyze_op.py` 输出 JSON，`calibration_docs` 输出 Markdown | `results/raw/`、`results/processed/` 下 CSV/JSON 文件 | ✅ 已实现 |
| 5.15 | 工作模式 | 支持仅建模、仅采集、交叉校准三种工作模式的自由组合 | 三种模式都已验证：仅建模（DeepStack 预测）、仅采集（benchmark 运行）、交叉校准（P6 hold-out、P8 参数测量校准） | P6 hold-out、P8 参数校准 | ✅ 已实现 |

## 统计

```
总计 15 项指标
├── ✅ 已实现    11 项（5.1,5.2,5.4,5.5,5.6,5.10,5.11,5.12,5.13,5.14,5.15）
├── ⚠️ 部分实现   4 项（5.3,5.7,5.8,5.9）
└── ❌ 未实现      0 项
```

## 4 项部分实现指标的缺口说明

| 指标 | 已做 | 缺口 |
|:-----|:-----|:------|
| **5.3 缓存行为预测** | 已增加32 KiB/MP普通L1、TMA bypass、Shared语义及分层流量预测 | 复杂reuse-distance、地址哈希和任意Tile访问模式仍待完善 |
| **5.7 跨层级搬运异常** | 已有结构化模型规则和异常/告警输出，仅依赖Tile pipeline与硬件Profile | 当前规则覆盖流量守恒、逻辑搬运放大和传输压力；后续可继续扩展算子特定规则，但不影响本项基础响应 |
| **5.8 缓存利用异常** | 大小工作集带宽对比可反推命中率 | 无独立的缓存命中率预测模型，需人工分析 |
| **5.9 计算-IO 重叠失败** | `PipelineDetail` 含 `mem_time` 和 `compute_time`，可判断重叠 | 仅对特定 GEMM 做了 stage 扫描，未通用化 |
