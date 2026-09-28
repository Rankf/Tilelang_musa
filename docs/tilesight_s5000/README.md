# S5000 TileSight 当前统一说明

更新日期：2026-09-28
工作目录：`/root/iie/ruankefan/Tilelang_musa`

## 1. 当前目标

本项目的核心不是继续增加展示字段，而是完善 TileSight 在 MTT S5000 上的性能公式和硬件参数，并使用独立运行时实测验证：

```text
TileLang/TileOPs 编译信息
          ↓
TileSight + S5000 参数/公式
          ↓
不可变模型结果
          ├──────────────┐
          ↓              ↓
模型诊断 5.6～5.10    S5000 runtime-only 采集
          └──────┬───────┘
                 ↓
        同口径比较、5.11 偏差、5.13 层级视图
```

运行时数据只用于独立比较和验证，不回填或暗中修改预测结果。需要校准的参数必须记录适用算子、dtype、线程配置、来源、开关和留出验证。

## 2. 当前目录结构

```text
DeepStack-ae-source/src/tilesight/tilesight/
  fused_op_pipeline_wave/       TileSight Pipeline、Occupancy、Wave 模型
  memory/                       S5000 分层内存与写回模型
  runtime/                      S5000 runtime-only 输入、比较和联合诊断 API

configs/
  runtime/                      Event/MCU/MSYS 运行时采集配置

scripts/
  collect_s5000_runtime_measurements.py
  build_runtime_input_from_collection.py
  build_runtime_measurement_bundle.py
  analyze_tilesight_with_runtime.py
  compare_tilesight_model_to_runtime.py
  ...                           公式校准、Shape sweep 和能力探针

tests/
  test_tilesight_runtime_api.py
  test_build_runtime_input_from_collection.py
  test_build_runtime_measurement_bundle.py
  test_tilesight_diagnostics.py
  ...                           模型、采集和诊断回归

results/profiling/              实测、模型、比较报告和原始 profiler 证据
evidence/legacy/                从仓库根目录迁入的历史版本化证据

docs/tilesight_s5000/
  README.md                     本文，当前唯一首要入口
  current/
    00_project_map_20260928.md  当前目录结构、交付入口和清理边界
    01_current_status_and_handoff.md
    02_runtime_interface.md
    03_required_outputs_and_validation.md
    04_log_sfu_service_migration.md
    05_pipeline_stage_validation.md
    06_bf16_analysis_support.md
    07_elementwise_batch_tracking.md
    08_model_input_boundary_audit.md
    07_elementwise_batch_tracking.md
```

`docs/` 根目录的其他 S5000 文档是历史审核、专项实验或旧交接材料。需要追溯时通过 `docs/S5000_TileSight_文档索引.md` 查找，不应优先于本文和 `current/` 下的文档。

## 3. 已完成的公式与参数工作

### Occupancy 和 Waves

- 256 线程 Elementwise 的 resident CTA/MP 已从错误的 24 修正为二进制 Occupancy API 验证的 12。
- 模型约 11.37778 waves/MP，与 MCU 11.377778 基本一致。

### 单阶段 Pipeline

- 保留上游公式和 legacy 分解。
- 修正 `stage_num=1` 被错误解释为多层 fill/drain 的问题。
- 区分 CTA 内软件 Pipeline 与 CTA 间硬件交错执行。
- Exp/Sqrt 使用范围受控、可关闭的 S5000 独立 CTA overlap profile。

### 特殊数学

- Exp/Sqrt 的 4096²、16384² 验证误差已控制在约 ±6% 内。
- Log 已迁移到统一 SFU 服务 profile。历史 SDK 5.2 独立序列候选为 `247.469 Gops/s`；当前第一批 FP16、冷启动、Continuous CTA 交付域使用 `200 Gops/s` 的有限范围校准值。BF16 仍使用独立的 `bf16_log_f32_sequence=247.469 Gops/s` Profile。
- Log 参数不是通用 SFU 峰值；旧 `log_sequence` 仅保留兼容读取，迁移前后预测完全等价。

### BF16

- 已增加显式 dtype-aware compute profile；BF16 不再因同为 2 字节而借用 FP16 计算身份。
- 已覆盖 BF16 GEMM、Add、ReLU、Exp、Log、Sqrt 的分析入口，见 [BF16 分析支持](current/06_bf16_analysis_support.md)。
- 旧 FP16/FP32 调用保留原路径；BF16 profile 可关闭，缺失或关闭时禁止静默回退 FP16。
- 已完成配对服务探针与 20 个真实 TileLang operator-level 开发/留出 case；仅 BF16 Sqrt 通过参数晋级门，将有效服务率校准为 `900 Gop/s`，其余 Profile 保持不变。
- BF16 特殊数学仍不复用 FP16 overlap/dirty-retention 校准；新鲜 MCU 留出是扩大该部分正式适用范围前的剩余证据门。

### 内存与写回

- 已输出 L1/L2/DDR 分层流量、服务时间和关键路径。
- 已分开 `kernel_scope_latency_us` 与完整逻辑写回 `latency_us`。
- Add/ReLU 的延迟写回已按逻辑写入、Kernel 范围物理写入和后续驱逐证据分层，不能把未观测差额直接称为精确 L2 驻留量。

## 4. 当前运行时接口

Python API：

```python
from tilesight.runtime import analyze_with_runtime

result = analyze_with_runtime(
    model_result,
    "runtime_measurement_bundle.json",
    "mcu_idle_cold_kernel_scope",
)
```

CLI：

```bash
python3 scripts/analyze_tilesight_with_runtime.py \
  --model MODEL.json \
  --runtime runtime_measurement_bundle.json \
  --runtime-record mcu_idle_cold_kernel_scope \
  --output analysis.json
```

联合结果输出：

- workload 身份与统计口径校验；
- Kernel 时间和 Waves 的同口径实测误差；
- DDR、缓存指标的口径分类；
- 5.6～5.10 联合诊断；
- 5.11 显著模型偏差标记；
- 5.13 compute/Shared/L1/L2/DDR 层级并列表。

不同范围或定义不一致的字段只并列展示，不计算正式误差。

## 5. 当前实测结论

Exp FP16 4096² cold：

| 指标 | 模型 | S5000 实测 | 相对误差 |
| --- | ---: | ---: | ---: |
| Kernel 时间 | 48.800 µs | 51.214286 µs | -4.714086% |
| Waves/MP | 11.377777778 | 11.377778 | -0.000002% |

5.10 模型分布：12 waves，整体 slot 利用率 94.8148%，尾波利用率 37.7778%，448 个尾波 resident slot 空闲；理论 CTA 分配本身均衡。

MCU counter 实验：

- FMA 探针 MP throughput：75.470414%。
- Shared reduction 的 load/store pipe 均返回非零。
- ReLU 正式探针可在一次目标 Kernel 记录中同时获得 MP、Shared、L1、L2、DDR 和 Waves。
- ReLU 的 L2PU throughput 返回 166.396563%，原值保留但因超出 0～100% 不参与瓶颈排序。

原始报告和哈希：`results/profiling/tilesight_required_output_probe_20260916/manifest.json`。

## 6. 指标状态与真实缺口

5.1～5.10 都有对应输出入口，但“有字段”不等于“原文要求已完全闭环”：

| 指标 | 当前输出 | 状态 |
| --- | --- | --- |
| 5.1 | Tile、pipeline、launch、资源和稳定 ID 等编译元数据 | 当前首批路径已覆盖 |
| 5.2 | 时延、带宽、计算/内存利用率、waves | 已实现 |
| 5.3 | 冷/热缓存、分层流量、基础命中率 | 已输出；复杂冲突和通用写回仍有限 |
| 5.4 | Event/MCU/MSYS Kernel 级运行期结果包 | 有基础输出；尚非按 `pipeline_id/op_id` 归集的 Tile trace |
| 5.5 | JSON、Markdown、终端摘要、artifact 清单 | 已实现 |
| 5.6 | 模型关键资源、同步判断、MP/Shared 运行时 counter | 已输出；显式同步等待时间仍缺阶段 trace |
| 5.7 | 逻辑/预测/实测分层流量与搬运异常判断 | 联合 API 已实现 |
| 5.8 | cache 状态、LLC read 与 L2PU 混合口径分类 | 联合 API 已实现；内部 counter 的跨版本稳定性待固化 |
| 5.9 | 理论 overlap；单阶段明确为不适用 | 已输出；多阶段实际 overlap、失败 TileOp 和调度原因仍缺失 |
| 5.10 | 理论逐 MP CTA 分布、waves、slot/尾波利用率及 MCU Waves 对照 | 已输出；逐 MP 实测时间分布仍缺失 |

因此，若按“是否能生成指标字段”回答，5.1～5.10 都有输出；若按完整验收回答，5.4、5.6、5.9、5.10 仍有运行时细粒度缺口。

仍无法从当前 Kernel 聚合 counter 可靠得到：

- 5.4：按 `pipeline_id/op_id` 归集的 Tile 维度运行期代价；
- 5.6：显式同步 Kernel 的运行时同步等待时间；
- 5.9：多阶段 Pipeline 的实际 overlap rate、失败 TileOp 和调度原因；
- 5.10/5.13：CTA/SM/cluster 的真实运行时负载分布。

这些需要阶段或执行单元级 trace，不能由 Kernel 总时间、聚合利用率或 launch-level Waves 推导。

## 7. 下一步公式工作

1. Log 已迁移到统一 SFU 服务 profile，迁移前后数值等价，见 [Log SFU 服务迁移](current/04_log_sfu_service_migration.md)。
2. stage 1/2/3 独立 MUSA 探针已完成；当前公式因低数学负载 stage 3 的相对时延门失败，暂不修改模型，见 [Pipeline stage 验证](current/05_pipeline_stage_validation.md)。
3. 下一步应隔离 stage 3 的 barrier/scheduler 增量，并验证 resident CTA effective-depth；在此之前不增加经验常数。
4. 完善 Add/ReLU 缓存驻留、延迟写回和 Kernel 范围 DDR 写入公式。
5. 使用已验证的 MP、Shared、L1、L2、DDR counter 审核各层关键路径判断。

## 8. 运行与维护边界

- 不更新 GPU 驱动；SDK 5.2 命令使用 `/usr/local/bin/musa52-run`。
- 不把运行时值写回模型结果。
- 不用不同统计范围的数据计算正式误差。
- 不删除 `results/profiling` 下的原始报告、manifest 或哈希证据。
- 不把 profiler 默认生成的 `report*.mcu-rep` 留在仓库根目录。
- 测试使用 `PYTHONDONTWRITEBYTECODE=1`，避免生成缓存干扰文件。

## 9. 当前文档

- [当前结构与工作状态](current/00_project_map_20260928.md)
- [当前状态与交接](current/01_current_status_and_handoff.md)
- [运行时接口实现与验证](current/02_runtime_interface.md)
- [指标输出补充与实验验证](current/03_required_outputs_and_validation.md)
- [Log 统一 SFU 服务公式迁移](current/04_log_sfu_service_migration.md)
- [stage 1/2/3 独立 Pipeline 验证](current/05_pipeline_stage_validation.md)
- [BF16 分析支持](current/06_bf16_analysis_support.md)
- [Elementwise 批次跟踪](current/07_elementwise_batch_tracking.md)
- [模型输入边界审计](current/08_model_input_boundary_audit.md)
- [历史文档索引](../S5000_TileSight_文档索引.md)

## 10. 本轮文件整理记录

- 将 3 份当前文档从 `docs/` 根目录迁入 `docs/tilesight_s5000/current/`，统一使用编号和英文文件名。
- 将 8 份 runtime collection 配置迁入 `configs/runtime/`，并更新调用脚本与使用文档中的路径。
- 将已纳入版本历史的根目录 `report1.mcu-rep` 迁入 `evidence/legacy/`，保留原文件和 SHA-256。
- 删除未跟踪、无引用的根目录 `report2.mcu-rep`；它是 profiler 默认命名的重复中间报告，无法证明对应 case。
- 将无引用且文件名已明确标记失败的 `gemm_4096_msys_sdk52_failed.msys-rep` 移入系统回收站；它不属于正式基线，可恢复。
- 删除 `/tmp/tilesight_review.diff` 和 `/tmp/tilesight_s5000_operator_path_probe_audit` 两个 2026-09-02 遗留临时文件。
- 确认工作区没有 `__pycache__`、`.pytest_cache`、`.pyc/.pyo`、编辑器 swap、backup 或临时文件。
- 未删除 `results/profiling/` 中的模型、正式比较、manifest、MCU/MSYS 原始报告或探针证据。
