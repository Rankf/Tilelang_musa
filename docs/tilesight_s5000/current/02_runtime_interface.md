# S5000 TileSight 运行时接口实现与验证

更新日期：2026-09-16

## 1. 结论

可行，并已实现。TileSight 现在可以读取独立的 S5000 runtime-only bundle，先校验算子、Shape、dtype、launch 和缓存条件，再输出模型–实测对照以及 5.6～5.10 联合诊断。

这里的“输入运行时信息”不是用实测值回填或修改预测公式，而是把两份不可变结果在分析层连接：

```text
TileSight model JSON ─┐
                     ├─ 身份/口径校验 ─ 对照表 ─ 5.6～5.10 联合诊断
S5000 runtime bundle ─┘
```

这样可以避免数据泄漏，也不会把一次采集结果误当成通用模型参数。现有 `tools/tilesight_diagnostics/` 仍是纯模型诊断入口；新接口是并行的 model + runtime 分析入口。

## 2. 实现入口

- Python API：`tilesight.runtime.analyze_with_runtime`
- 运行时 Schema 与身份校验：`tilesight/runtime/schema.py`
- 指标口径表：`tilesight/runtime/metrics.py`
- 同口径比较：`tilesight/runtime/comparison.py`
- 5.6～5.10 联合诊断：`tilesight/runtime/diagnostics.py`
- 命令行入口：`scripts/analyze_tilesight_with_runtime.py`
- 采集转换：`scripts/build_runtime_input_from_collection.py`

命令示例：

```bash
python3 scripts/analyze_tilesight_with_runtime.py \
  --model results/profiling/tilesight_exp_sqrt_adapter_20260911/tilesight_exp_fp16_4096x4096_cold_model.json \
  --runtime results/profiling/runtime_tileops_exp_fp16_20260906/collection_cold_skip1_physical1_v2/runtime_measurement_bundle.json \
  --runtime-record mcu_idle_cold_kernel_scope \
  --output /tmp/tilesight_exp_runtime_analysis.json
```

API 示例：

```python
from tilesight.runtime import analyze_with_runtime

result = analyze_with_runtime(
    model_result,
    "runtime_measurement_bundle.json",
    "mcu_idle_cold_kernel_scope",
)
```

## 3. 输入与准入规则

运行时输入必须满足：

- `schema_version` 为 `s5000/runtime-measurement-bundle-v1`；
- `bundle_scope` 为 `runtime_only`，且 `model_data_included` 明确为 `false`；
- 不含 `model`、`prediction`、`tilesight` 等模型字段；
- 选中的记录状态为 `available`，正确性为 `pass`；
- 模型和运行时的 operator、shape、dtype 必须齐全且一致；
- 双方同时存在的 Grid、Block、动态共享内存、Kernel 哈希、缓存条件必须一致；
- 可选身份字段缺失会显式报告，但不会伪造。

若身份不匹配，正式数值比较会直接拒绝执行。

## 4. 已接入的 S5000 指标

采集转换现在覆盖：

- Kernel 时间；
- Grid、Block、动态/静态共享内存、每线程寄存器；
- Waves/MP；
- DDR 读、写、总字节和利用率；
- L1 downstream 读写字节、命中率和利用率；
- L2PU 混合请求命中率和利用率；
- LLC 读取命中/未命中 sector，并按同一样本派生读取命中率。

尚未稳定接入的指标不会推测填值。尤其是计算管线利用率和 shared-memory 利用率，需要先确认 S5000 MCU 原生 counter 名称及定义。

## 5. 5.6～5.10 的处理

| 指标 | 当前联合输出 | 边界 |
| --- | --- | --- |
| 5.6 Pipeline 瓶颈 | 模型瓶颈与已采集各层利用率并列，指出运行时最高资源 | 缺失资源不参与排序；字段少时置信度为低 |
| 5.7 跨层搬运异常 | 逻辑流量、模型 Kernel 范围 DDR、MCU 物理 DDR 并列 | MCU 物理事务定义确认前标记 `definition_pending` |
| 5.8 缓存利用异常 | 优先使用 LLC 读取 sector 命中率；否则保留 L2PU 混合命中率 | L2PU 混合命中率不与模型逻辑 L2 复用率直接比较 |
| 5.9 计算–IO 重叠 | 保留 TileSight stage 时间模型 | 当前 runtime bundle 无 Kernel 内逐阶段硬件重叠率，明确标为 unavailable |
| 5.10 负载不均 | 模型与 MCU Waves/MP 可直接比较 | 逐 MP CTA 时间线仍不可用，不据 Waves 推断真实时间分布 |

## 6. 真实数据验证

使用现有 Exp FP16 4096×4096 cold 模型和 S5000 MCU 结果包验证：

| 项目 | 模型 | 实测 | 结果 |
| --- | ---: | ---: | --- |
| 完整逻辑写回模型时延 | 48.800 µs | 51.214285714 µs | 口径不同，仅作范围比较 |
| Kernel 窗口诊断时延 | 48.800 µs | 51.214285714 µs | 诊断字段，不作为正式 TileSight 误差 |
| Waves/MP | 11.377777778 | 11.377778 | -0.000002%，同口径 |
| 缓存命中率 | 0% 逻辑 L2 预测 | 25.090533% L2PU 混合命中 | 不直接比较 |

身份校验通过的字段为 operator、shape、dtype、Grid、Block、动态共享内存和缓存条件。模型归档未提供 Kernel 名称和源码哈希，因此状态为 `matched_with_missing_optional_fields`，而不是伪造完全匹配。

验证还覆盖：

- API 不修改传入的模型或运行时对象；
- 5.6～5.10 五项均生成结构化结果；
- Shape 不一致时拒绝比较；
- mixed bundle、携带模型数据的 bundle、不可用记录和正确性失败记录均被拒绝；
- 扩展 MCU 指标转换及 LLC 读取命中率派生正确；
- 命令行入口可生成 `tilesight/model-runtime-analysis-v1` JSON。

当前系统 Python 没有安装 pytest，本轮使用标准库 `unittest` 执行新增的 9 项接口与转换定向测试，全部通过；相关运行时链路共执行 28 项回归，全部通过，同时完成真实归档的 API 与 CLI 端到端验证。

## 7. 尚未完成的硬件证据

- 5.9 的硬件实测计算–IO overlap；
- 5.10 的逐 MP CTA 时间分布；
- 部分 MCU counter 的公开稳定语义；
- MUPTI Kernel Activity 与单次 launch correlation。

最后一项用于把 MUPTI 活动记录严格归属到某一次 launch，并不影响当前 MCU runtime bundle 接入和联合分析已经可用。
