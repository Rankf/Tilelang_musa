# 29. S5000 参数在 DeepStack 模型中的实际使用情况审计

状态：completed
日期：2026-08-08
对象：DeepStack 分析引擎（`DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/`）

## 一、审计目的

确认 S5000 校准的 18 个参数（9 必须 + 9 重要）在 DeepStack 模型核心计算代码中是否被实际使用，区分"直接使用"、"间接使用"和"完全未使用"。

## 二、审计方法

对模型核心计算目录执行引用扫描：

```
DIR = DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/
grep -rn "arch.<param>" $DIR
```

## 三、审计结果

### 3.1 必须参数（9个）

| 参数 | 模型使用方式 | 引用数 | 使用状态 |
|:-----|:------------|:------:|:--------:|
| sm_count | 直接使用（occupancy、wave、active SM 计算）| 17 | ✅ 已使用 |
| l2_capacity_bytes | 直接使用（缓存层级分析）| 5 | ✅ 已使用 |
| shared_memory_capacity_bytes | 直接使用（occupancy 计算）| 1 | ✅ 已使用 |
| register_capacity_per_sm_bytes | 直接使用（occupancy 计算）| 1 | ✅ 已使用 |
| core_frequency_hz | 间接使用（计算所有 flops）| - | ✅ 已使用 |
| fp32_cores_per_sm | 间接使用（→ fp32_cuda_core_flops）| - | ✅ 已使用 |
| tensor_cores_per_sm | 间接使用（→ fp16_tensor_flops）| - | ✅ 已使用 |
| tensor_core_shape | 间接使用（→ fp16_tensor_flops）| - | ✅ 已使用 |
| **memory_frequency_hz** | **无（ddr_bandwidth 硬编码 1600e9）** | **0** | ⚠️ **未使用** |

### 3.2 重要参数（9个）

| 参数 | 模型使用方式 | 引用数 | 使用状态 |
|:-----|:------------|:------:|:--------:|
| ddr_max_utilization | 直接使用（roofline 利用率）| 2 | ✅ 已使用 |
| compute_max_utilization | 直接使用（计算利用率）| 1 | ✅ 已使用 |
| l2_max_utilization | 直接使用（L2 利用率）| 2 | ✅ 已使用 |
| l1_max_utilization | 直接使用（L1 利用率）| 4 | ✅ 已使用 |
| shared_memory_throughput_bytes_per_cycle | 间接使用（→ smem_bandwidth）| - | ✅ 已使用 |
| sfu_cores_per_sm | 间接使用（→ sfu_flops）| - | ✅ 已使用 |
| **sm_sub_partitions** | **无（→ register_bandwidth，未用）** | **0** | ❌ **未使用** |
| **warp_schedulers_per_sm** | **无** | **0** | ❌ **未使用** |
| **int32_cores_per_sm** | **无（→ int32_cuda_core_flops，未用）** | **0** | ❌ **未使用** |

## 四、汇总

```
18 个参数
├── ✅ 被模型使用     14 个
│   ├── 直接使用（模型核心计算引用）  11 个
│   └── 间接使用（→ 衍生属性被引用）   3 个
│
└── ❌ 未被模型使用     4 个
    ├── memory_frequency_hz        （必须参数）
    ├── sm_sub_partitions          （重要参数）
    ├── warp_schedulers_per_sm     （重要参数）
    └── int32_cores_per_sm         （重要参数）
```

## 五、关键发现

### 5.1 `memory_frequency_hz` 被标记"必须"但未被使用

模型里 `ddr_bandwidth = 1600e9` 是**硬编码**的，不是从 `memory_frequency_hz` 计算的。该参数虽归入"必须"，但实际模型未读取。

### 5.2 3 个重要参数确实未被使用

`sm_sub_partitions`、`warp_schedulers_per_sm`、`int32_cores_per_sm` 这三个**完全不影响当前模型预测**。

- `sm_sub_partitions` 仅用于计算 `register_bandwidth`（模型未引用）
- `warp_schedulers_per_sm` 未用于任何衍生计算
- `int32_cores_per_sm` 仅用于计算 `int32_cuda_core_flops`（模型未引用）

### 5.3 与 26 号文档的关联

这解释了 26 号文档中"int32_cores_per_sm 未实测"的影响评估：
- 模型未使用 `int32_cores_per_sm` → 对当前预测**无影响**
- 之前担心的"模型高估 INT32 性能 2.6x"**不成立**（模型根本不读该值）

## 六、影响与建议

### 6.1 对当前模型的影响

未被使用的 4 个参数**对当前模型预测无影响**，无需紧急修正。

### 6.2 未来扩展风险

若未来扩展模型功能，这些参数将变得重要：

| 参数 | 未来使用场景 |
|:-----|:------------|
| memory_frequency_hz | 若改为从 memory_freq 计算 ddr_bandwidth |
| sm_sub_partitions | 多核/子分区级调度建模 |
| warp_schedulers_per_sm | 指令级调度建模 |
| int32_cores_per_sm | INT32 算子分析支持 |

### 6.3 建议

1. 保持这些参数的正确值（已校准），即使当前未使用
2. 记录本审计结果，避免误以为"参数错误导致模型不准"
3. 若未来扩展功能，应先确认这些参数的可靠性

## 七、相关文件

- 模型目录：`DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/`
- 架构配置：`DeepStack-ae/src/deepstack/mosaic/arch/s5000.py`
- 审计脚本：`/tmp/check_params.sh`
- 关联文档：`26_int32_core_uncertainty_analysis.md`