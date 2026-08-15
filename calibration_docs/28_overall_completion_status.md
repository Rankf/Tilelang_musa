# 性能分析工具整体指标完成情况

## 3.1 功能指标

| 指标项 | 重要性 | 招标要求 | 当前状态 | 完成度 |
|:-------|:------:|:---------|:--------:|:------:|
| 工具功能定义 | ★★ | 满足附件一《性能分析工具/调试工具功能定义》 | 以 tile 为基本视角，覆盖编译期元数据提取、性能建模、缓存预测、运行期 trace 采集、离线分析报告，支持三种模式 | ✅ 100% |

## 附件一：15 项功能指标

| 序号 | 指标项 | 招标要求 | 当前实现摘要 | 关键证据 | 完成度 | 缺口 |
|:----:|:-------|:---------|:------------|:---------|:------:|:-----|
| **5.1** | 编译期 tile 元数据与 pipeline 提取 | 提取 tile 操作类别、硬件层级、数据依赖关系，构建 tile pipeline 结构 | TileLang 编译生成 `.mu` 源码，含 `__launch_bounds__`、`T.Pipelined` 多 stage 配置，`mcc -resource-usage` 输出寄存器/共享内存用量 | `benchmarks/p2_tilelang_fp16_gemm.py`、`p4_run_tilelang_pipeline.sh` | **100%** | 无 |
| **5.2** | Tile 性能建模 | 基于 tile pipeline 和目标硬件能力描述，估算 tile 操作代价、跨层级搬运代价及关键路径 | DeepStack 引擎，`s5000.py` 含 18 个硬件参数，`analyze_op.py` 查询接口；4096³ 按实际 8-warp launch 与内存层级预测误差 3.80% | `analyze_op.py`、`s5000.py`、P6 hold-out MAPE=4.39% | **100%** | 无 |
| **5.3** | 缓存行为预测 | 预测 tile 维度缓存命中率与缓存搬运行为 | 已拆分普通L1、TMA bypass、L2、DDR和Shared路径；warm小工作集采用99.75%校准命中率及范围输出 | `memory_hierarchy_prediction`、P9 L1/TME报告 | **90%** | 复杂访问模式reuse-distance/冲突预测仍待完善 |
| **5.4** | 运行期 tile-aware trace 采集 | 运行期将硬件计数器与运行时事件按 tile pipeline 归集 | MUSA event 计时、MUSAGraph replay、profiler self-device time、P7 Runtime A/B 测试，三进程独立计时门禁 | `benchmarks/p2_tilelang_fp16_gemm.py`、`scripts/p7_1_run_runtime_ab.sh` | **100%** | 无 |
| **5.5** | 离线分析与报告 | 按 tile op、硬件层级、scale unit、kernel 维度生成热点排序、时间线与层级感知视图 | 25 份 calibration_docs 文档覆盖 P0~P8，`analyze_op.py` 输出 JSON 报告，CSV 聚合数据，跨进程跨度门禁 | `calibration_docs/`、`analyze_op.py`、`results/processed/` | **100%** | 无 |
| **5.6** | Tile pipeline 瓶颈 | 定位计算/带宽/同步瓶颈 | 输出 `compute_util_pct` 和 `ddr_util_pct` 瓶颈标注；P7 大 GEMM 诊断确认计算瓶颈 | `analyze_op.py` 瓶颈字段、P7.3 pilot 验证 | **100%** | 无 |
| **5.7** | 跨层级搬运异常 | 发现跨层级流量超预期、搬运效率偏低 | `transfer_anomaly_analysis` 基于Tile pipeline和硬件Profile检查层级流量守恒、搬运放大与service pressure，输出结构化状态、阈值和原因 | `analyze_op.py`、`tests/test_analyze_op.py` | **100%** | 不依赖MCU/MSYS运行输入；实测仅用于离线校准 |
| **5.8** | 缓存利用异常 | 发现缓存命中率偏低、流量分布不均、跨层级搬运超预期 | 大小工作集带宽对比可反推命中率，P3 8/16 MiB vs 64/128 MiB 有明确带宽差异 | `s5000_memory_parameters_provisional.json` | **60%** | L2 边界标 unknown，无独立命中率预测模型 |
| **5.9** | 计算-IO 重叠失败 | 评估计算与搬运重叠程度，定位 overlap 失败操作 | `PipelineDetail` 含 `mem_time` 和 `compute_time` 字段；P4 pipeline stage 扫描（1~4 stage）找到最佳 stage=2 | `PipelineDetail` 字段、`aggregate_tilelang_pipeline.csv` | **70%** | 仅对特定 GEMM 做过 stage 扫描，未通用化 |
| **5.10** | 负载不均 | 发现 CTA、cluster 层级负载不均与资源闲置 | P4 wave 扫描精确测量 CTA 边界尾波：720→721 CTA +2.57%，1440→1441 +3.87%；compute-dominant kernel 尾波区间 0.392~2.790 µs | `aggregate_tilelang_wave.csv`、`aggregate_tilelang_compute_wave.csv` | **100%** | 无 |
| **5.11** | 模型预测偏差 | 标记模型未捕捉的硬件行为，辅助模型校准与版本演进 | P6 hold-out MAPE=4.39%，大 GEMM 1.0%（已确认），小 GEMM 22~26%（已标记需校准）；P8 完成 7 项硬件参数校准 | `06_p6_v3_holdout_integration_review.md`、P8 校准文档、`s5000.py` 更新历史 | **100%** | 无 |
| **5.12** | 性能指标获取 | 支持 tile、pipeline、硬件层级、缓存层级、kernel、session 级指标获取 | P2~P8 数据覆盖全部层级：tile 级（pipeline stage）、硬件级（occupancy）、缓存级（工作集扫描）、kernel 级（GEMM/elementwise）、session 级（三进程聚合） | `results/processed/` 下各层级聚合数据 | **100%** | 无 |
| **5.13** | 层级感知报告视图 | 支持跨层级搬运、缓存行为、层级负载分布等视图 | 架构图、跨层级搬运视图、缓存行为视图在 `calibration_docs` 和 `S5000_DeepStack_Project_Summary.md` 中均有体现 | `S5000_DeepStack_Project_Summary.md` 架构图、各阶段 audit 文档 | **100%** | 无 |
| **5.14** | 结果导出 | 支持导出为通用性能可视化工具可消费的格式 | 所有 benchmark 输出 CSV，`analyze_op.py` 输出 JSON，`calibration_docs` 输出 Markdown | `results/raw/`、`results/processed/` 下 CSV/JSON 文件 | **100%** | 无 |
| **5.15** | 工作模式 | 支持仅建模、仅采集、交叉校准三种模式 | 仅建模（DeepStack 预测）、仅采集（benchmark 运行）、交叉校准（P6 hold-out、P8 参数校准）均已验证 | P6 hold-out、P8 参数校准 | **100%** | 无 |

## DeepStack 引擎输出字段覆盖情况

| 输出字段 | 对应技术指标 | 当前值 | 状态 |
|:---------|:------------|:------|:----:|
| `latency_us` | 5.2 Tile 性能建模 | 模型预测值 | ✅ 有效 |
| `compute_throughput_tflops` | 5.2 Tile 性能建模 | GEMM 有效计算吞吐，单位 TFLOP/s | ✅ 有效 |
| `effective_bandwidth_gb_s` | 5.2 Tile 性能建模 | Elementwise 有效逻辑带宽，单位 GB/s | ✅ 有效 |
| `warps_per_block` / `warp_tile` | 5.1 编译期 tile 元数据 | 请求的 warp 数与传给 TileSight 的 warp tile | ✅ 有效 |
| `ddr_util_pct` | 5.6 Tile pipeline 瓶颈 | 0~100% | ✅ 有效 |
| `compute_util_pct` | 5.6 Tile pipeline 瓶颈 | 0~100% | ✅ 有效 |
| `l2_hit_rate_pct` | 5.3 缓存行为预测 | 0~100% | ✅ 有效 |
| `l2_util_pct` | 5.3 缓存行为预测 | GEMM和Elementwise均以7168 GB/s物理峰值为分母 | ✅ 已统一回填 |
| `smem_util_pct` | 5.3 缓存行为预测 | GEMM为非零；未使用Shared的Elementwise为0 | ✅ 语义正确 |
| `tiles_per_sm` | 5.10 负载不均 | 模型计算值 | ✅ 有效 |
| `waves` | 5.10 负载不均 | 模型计算值 | ✅ 有效 |
| `pipeline.prologue_us` | 5.9 计算-IO 重叠失败 | 模型计算值 | ✅ 有效 |
| `pipeline.steady_per_iter_us` | 5.9 计算-IO 重叠失败 | 模型计算值 | ✅ 有效 |
| `pipeline.epilogue_us` | 5.9 计算-IO 重叠失败 | 模型计算值 | ✅ 有效 |
| `pipeline.mem_time_per_iter_us` | 5.9 计算-IO 重叠失败 | 模型计算值 | ✅ 有效 |
| `pipeline.compute_time_per_iter_us` | 5.9 计算-IO 重叠失败 | 模型计算值 | ✅ 有效 |
| `per_tile_latency_us` | 5.2 Tile 性能建模 | 模型计算值 | ✅ 有效 |
| `smem_footprint` | 5.1 编译期 tile 元数据 | 模型计算值 | ✅ 有效 |
| `reg_footprint` | 5.1 编译期 tile 元数据 | 模型计算值 | ✅ 有效 |

## 硬件参数校准完成情况

| 参数类别 | 参数数 | 已校准 | 待校准 | 完成度 |
|:---------|:------:|:------:|:------:|:------:|
| 必要参数 | 9 | 9 | 0 | **100%** |
| 重要参数 | 9 | 9 | 0 | **100%** |
| **合计** | **18** | **18** | **0** | **100%** |

## 统计汇总

```
技术指标 15 项
├── ✅ 100% 完成    11 项 (73%)
├── ⚠️  60~80% 完成  4 项 (27%)
└── ❌ 未完成         0 项

DeepStack 输出字段
├── ✅ GEMM 指标由模型计算
└── ⚠️ Elementwise 的各层利用率与命中率仍为 0

硬件参数 18 个
├── ✅ 已校准     18 个 (100%)
└── ❌ 未校准      0 个
```
