# S5000 算子性能分析框架 — 项目总结

## 一、整体定位

整个体系的核心引擎是 **DeepStack**，它是一个面向 GPU 架构的性能分析工具。所有信息采集工作（S5000 硬件校准、算子级静态分析、运行期 trace 采集）都是为 DeepStack 提供校准数据，帮助它更准确地预测 S5000 上各类算子的性能。

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                       校准数据输入层                                │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │ s5000-calibration   │  │ musa-perf-collector  │                  │
│  │ 硬件实测校准         │  │ 算子级分析工具       │                  │
│  │ (P0~P7 八阶段)       │  │ (CPU离线 + 运行期)   │                  │
│  └──────────┬──────────┘  └──────────┬──────────┘                  │
│             └──────────┬─────────────┘                              │
│                        │                                            │
│              ┌─────────┴─────────┐                                  │
│              │ 硬件规格参数(已补) │                                  │
│              │ (频率/Core规格等)  │                                  │
│              └─────────┬─────────┘                                  │
└────────────────────────┼────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        核心引擎：DeepStack                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ 架构描述      │  │ 性能模型      │  │ 设计空间探索(DSE)        │  │
│  │ s5000.py     │  │ OpPerfStats  │  │ 并行策略/NoC/DRAM       │  │
│  │ 参数注入     │  │ ModelPerf    │  │ (暂未使用)               │  │
│  │              │  │ Stats        │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          输出层                                     │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ 性能预测      │  │ 瓶颈分析      │  │ 优化建议                  │  │
│  │ • 延迟(µs)   │  │ • 计算瓶颈    │  │ • tile 大小调整          │  │
│  │ • TFLOP/s   │  │ • 访存瓶颈    │  │ • pipeline stage 调整   │  │
│  │ • GB/s      │  │ • Occupancy  │  │ • warp 数调整            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────────────┐
            │           使用场景                    │
            │                                      │
            │  场景A：手工算子优化                   │
            │  开发者写 TileLang kernel →           │
            │  DeepStack 分析 → 迭代调优            │
            │                                      │
            │  场景B：Agent 算子评估                 │
            │  Agent 生成 kernel →                 │
            │  DeepStack 自动评估 → 反馈迭代         │
            └──────────────────────────────────────┘
```

---

## 二、硬件参数获取（全量）

### 2.1 必须参数（无这些参数，性能分析无法运行）

| 参数 | 说明 | 状态 | 值 | 获取方式 | 置信度 |
|:-----|:-----|:----:|:---|:---------|:------:|
| **sm_count** | SM 数量 | ✅ 已有 | 60 | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **core_frequency_hz** | 核心频率 | ✅ 已有 | 1,500,000,000 (1.5 GHz) | musaGetDeviceProperties clockRate | ⭐⭐⭐⭐⭐ |
| **memory_frequency_hz** | 显存频率（有效数据率） | ✅ 已有 | 20,000,000,000 (20 GHz) | musaGetDeviceProperties memoryClockRate | ⭐⭐⭐⭐⭐ |
| **tensor_cores_per_sm** | 每 SM Tensor Core 数量 | ✅ 已有 | 1 | FP16 GEMM 实测 153.4 TFLOP/s 反推（83.3% 效率） | ⭐⭐⭐⭐ |
| **tensor_core_shape** | Tensor Core 形状 (M,N,K) | ✅ 已有 | (16, 8, 8) | TileLang-MUSA mma.h 源码确认（MP31_16x8x8_F32F16F16F32） | ⭐⭐⭐⭐⭐ |
| **fp32_cores_per_sm** | 每 SM FP32 Core 数量 | ✅ 已有 | 128 | FP32 FMA 实测 22.7 TFLOP/s，对理论 23.0 = 98.5% 效率 | ⭐⭐⭐⭐⭐ |
| **l2_capacity_bytes** | L2 缓存大小 | ✅ 已有 | 62,914,560 (60 MiB) | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **shared_memory_capacity_bytes** | 每 SM 共享内存 | ✅ 已有 | 196,608 (192 KiB) | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **register_capacity_per_sm_bytes** | 每 SM 寄存器总数 | ✅ 已有 | 131,072 (32-bit regs) | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |

**9 个必须参数全部到位。**

### 2.2 重要但可用默认值（缺了也能跑，精度会下降）

| 参数 | 当前值 | 可尝试方法获取 | 状态 |
|:-----|:------|:------------|:----:|
| **ddr_max_utilization** | 0.876（实测 fill 1401/1600=87.6%） | 从实测数据反推 | ✅ 已更新 |
| **compute_max_utilization** | 0.832（实测 GEMM 153.4/184=83.2%） | 从实测数据反推 | ✅ 已更新 |
| l2_max_utilization | 0.9（默认） | 从实测 99/理论峰值反推 | ❌ 默认 |
| l1_max_utilization | 0.9（默认） | 从实测 99/理论峰值反推 | ❌ 默认 |
| sm_sub_partitions | 4（默认） | 写不同 warp 数量 kernel 测性能拐点 | ❌ 默认 |
| warp_schedulers_per_sm | 4（默认） | 与 sm_sub_partitions 强相关，可间接推断 | ❌ 默认 |
| shared_memory_throughput_bytes_per_cycle | 128（默认） | 从实测 shared memory 带宽反推 | ❌ 默认 |
| sfu_cores_per_sm | 16（默认） | 写纯 SFU 指令 kernel 测吞吐反推 | ❌ 默认 |
| int32_cores_per_sm | 64（默认） | 写纯 INT32 计算 kernel 测吞吐反推 | ❌ 默认 |

### 2.3 特定场景才需要

| 参数 | 使用场景 | 状态 |
|:-----|:---------|:----:|
| **DramInterfaceConfig**（全部） | 仅设计空间探索（DSE）/ 3D DRAM 堆叠分析 | ❌ S5000 非 3D 堆叠，可能不需要 |
| **ChipEnergyConfig**（全部） | 仅能耗分析（pJ/bit, pJ/op） | ❌ 需硬件测量，当前不支持 |
| **noc_frequency_hz** | 仅 NoC 瓶颈分析 | ❌ 芯片内部参数，需厂商 |
| **l1_noc_bytes_per_cycle** | 仅 NoC 瓶颈分析 | ❌ 芯片内部参数，需厂商 |
| **tensor_instruction_shapes** | 仅 FP8/INT8 混合精度 | ❌ 需 MUSA 内部文档 |
| **support_wgmma / utcmma** | 仅 Hopper 特性分析 | ❌ MUSA 不支持，已设 False |

---

## 三、硬件参数交叉验证

### 3.1 显存带宽验证

| 项目 | 值 |
|:-----|:----|
| 显存位宽 | 640 bits |
| 显存频率（有效） | 20 GHz |
| 理论峰值带宽 | 640 × 20GHz ÷ 8 = **1,600 GB/s** |
| 实测 fill 带宽 | **1,401 GB/s**（vectorized float4, 1024 MiB） |
| 实测 copy 带宽 | **1,349 GB/s**（vectorized float4, 1024 MiB） |
| 利用率 | 1,401 / 1,600 = **87.6%** ✅ 正常 |

### 3.2 FP32 算力验证

| 项目 | 值 |
|:-----|:----|
| 每 SM FP32 Core | 128 |
| SM 数量 | 60 |
| 核心频率 | 1.5 GHz |
| 理论峰值 | 60 × 128 × 1.5G × 2(FMA) = **23.0 TFLOP/s** |
| 实测 FP32 FMA | **22.7 TFLOP/s** |
| 利用率 | 22.7 / 23.0 = **98.5%** ✅ 计算密集 kernel 效率极高 |

### 3.3 Tensor Core 算力验证

| 项目 | 值 |
|:-----|:----|
| Tensor Core 形状 | (16, 8, 8) → 每次 2048 FLOPs |
| 每 SM Tensor Core | 1 |
| 核心频率 | 1.5 GHz |
| 理论峰值 | 60 × 1 × 1.5G × 2048 = **184 TFLOP/s** |
| 实测 FP16 GEMM（4096×4096×8192） | **153.4 TFLOP/s** |
| 利用率 | 153.4 / 184 = **83.3%** ✅ GEMM 典型效率 |

---

## 四、校准项目数据（s5000-calibration）

### 4.1 已完成 8 阶段校准

| 阶段 | 内容 | 状态 |
|:-----|:-----|:----:|
| P0 | 后端环境审计（MUSA 3.3.5, SDK 4.3.5, MP31） | ✅ 完成 |
| P1 | 测量基础设施稳定性验证 | ✅ 完成 |
| P2 | 计算性能（FP16/BF16/INT8 GEMM, Vector, Reduce, Exp） | ✅ 完成 |
| P3 | 内存性能（Global memory, Cache, Shared memory） | ✅ 完成 |
| P4 | Occupancy 与 Wave 分析（资源约束、寄存器压力、尾波） | ✅ 完成 |
| P5 | 参数模型构建（条件查询校准） | ✅ 完成 |
| P6 | Hold-out 验证 | ✅ 完成 |
| P7 | 大 GEMM 运行时诊断、Runtime A/B | ✅ 完成 |

### 4.2 关键实测数据

**FP16 GEMM 实测算力（最高配置）**

| 配置 | Shape (M×N×K) | Tile | Stage | 延迟 (µs) | 算力 | 跨进程跨度 |
|:-----|:-------------:|:----:|:-----:|:---------:|:----:|:---------:|
| 🏆 最优 | 4096×4096×8192 | 128×128×64 | 2 | 2,190 | **153.4 TFLOP/s** | — |
| 大 shape | 4096×4096×4096 | 128×128×64 | 2 | 993.5 | 139.1 TFLOP/s | — |
| 之前校准 | 1024×1024×1024 | 64×64×64 | 2 | 21.67 | 99.1 TFLOP/s | 0.38% |
| 之前校准 | 1024×1024×1024 | 64×64×64 | 1 | 23.42 | 91.7 TFLOP/s | 0.51% |
| 之前校准 | BF16 1024³ | 64×64×64 | 1 | 23.25 | 92.4 TFLOP/s | 0.31% |
| 之前校准 | INT8 1024³ | 128×128×64 | 1 | 30.70 | 70.0 TFLOP/s | 0.11% |

**Global Memory 逻辑带宽**

| 模式 | 逻辑 Payload | 延迟 (µs) | 逻辑带宽 | 说明 |
|:-----|:-----------:|:---------:|:--------:|:-----|
| Fill（纯写） | 256 MiB | 191.3 | **1,403 GB/s** | 写入逻辑带宽 |
| Triad（A+B→C） | 768 MiB | 574.3 | **1,402 GB/s** | 两读一写混合 |
| Copy（读+写） | 1 GiB | 808.1 | **1,329 GB/s** | 大工作集双向 |
| Read Sum（读密集） | 512 MiB 输入 | 599.2 | 910 GB/s | 读密集下界 |

**Occupancy 数据**

| Threads/CTA | 低寄存器 (4 regs) | 中等寄存器 (24 regs) | 高寄存器 (53 regs) |
|:-----------:|:-----------------:|:-------------------:|:-----------------:|
| 64 | 48 CTA/MP | 48 CTA/MP | 34 CTA/MP |
| 128 | 24 CTA/MP | 24 CTA/MP | 17 CTA/MP |
| **256** | **12 CTA/MP** | **12 CTA/MP** | **8 CTA/MP** |
| 512 | 6 CTA/MP | 6 CTA/MP | 4 CTA/MP |

---

## 五、DeepStack 预测验证结果

### 5.1 GEMM 预测（使用 S5000 架构配置）

| Case | 配置 | S5000 实测 | DeepStack 预测 | 误差 | 瓶颈 |
|:-----|:-----|:---------:|:------------:|:----:|:----:|
| FP16 4096×4096×4096 | 128×128×64, stage2, warp4 | 993.5 µs | 1,003.8 µs | **1.0%** | Compute |
| BF16 128×7168×2048 | 64×64×64, stage1, warp4 | 59.7 µs | 55.3 µs | **7.3%** | Compute |
| FP16 1024×1024×1024 | 64×64×64, stage1, warp4 | 28.1 µs | 35.1 µs | 22.6% | Compute |
| FP16 1024×1024×1024 | 128×128×64, stage2, warp4 | 40.5 µs | 29.8 µs | 26.4% | Compute |

**结论**：大 GEMM 预测精度高（1%），小 GEMM 因 prologue 模型使用 NVIDIA 经验值未校准，误差偏大。

### 5.2 Elementwise 实测（S5000 真实运行）

| 算子 | N | 策略 | 延迟 | 带宽 | 吞吐 |
|:-----|:--:|:----:|:----:|:----:|:----:|
| Relu | 16M | register_copy, 256t, 4npt | 107.6 µs | 1,247 GB/s | 155.9 Gitems/s |
| Exp | 16M | register_copy, 256t, 4npt | 109.8 µs | 1,222 GB/s | 152.8 Gitems/s |
| Sigmoid | 16M | register_copy, 256t, 4npt | 111.4 µs | 1,204 GB/s | 150.5 Gitems/s |

---

## 六、TileOPs 算子适配情况

### 6.1 已适配的算子

| 算子簇 | 具体算子 | 适配方式 | 文件路径 | 状态 |
|:-------|:---------|:---------|:---------|:----:|
| **GEMM dense** | FP16/BF16, NN/NT/TN/TT | 独立 kernel 实现 | `musa_operator_fixes/gemm/dense/mp31_fp16_nn.py` | ✅ 可通过 `GemmOp(kernel_map=...)` 调用 |
| **Elementwise unary** | Relu, Exp, Sigmoid 等 28 个 kernel 类 | 运行时补丁（supported_archs + validation） | `musa_operator_fixes/elementwise_sfu/mp31_elementwise.py` | ✅ 引入即生效 |

### 6.2 适配方式说明

**GEMM 适配**：`MusaMP31DenseGemmKernel` 是独立实现的 kernel，替换 TileOPs 默认的 Hopper-only `GemmKernel`。使用 `T.gemm`（MUSA 兼容）而非 `T.wgmma_gemm`（NVIDIA 专属）。

**Elementwise 适配**：通过运行时补丁，只改了 2 个限制条件：
1. 给 28 个 kernel 类的 `supported_archs` 加上 `31`（MUSA MP31）
2. 修改 `UnaryOp._validate_input` 的 `is_cuda` 检查，改为 `is_cuda || is_musa`

### 6.3 未适配的算子簇

| 算子簇 | 依赖 Hopper 特性 | 适配难度 | 说明 |
|:-------|:---------------:|:--------:|:-----|
| **BMM (batched GEMM)** | 是（wgmma） | 中 | 类似 GEMM 适配方式 |
| **Attention** | 是（TMA/wgmma） | 高 | 复杂 kernel |
| **Reduction** | 否 | 低 | 类似 elementwise，补丁 supported_archs 即可 |
| **Dropout** | 否 | 低 | 同上 |
| **RoPE** | 否 | 低 | 同上 |
| **Norm** | 否 | 低 | 同上 |
| **Convolution** | 是（可能含 Hopper 特性） | 中 | 需检查 |
| **Grouped GEMM** | 是（wgmma） | 高 | 复杂 kernel |

---

## 七、工具链

### 7.1 命令行分析工具

文件：`/root/s5000-calibration/analyze_op.py`

GEMM只接受5.1编译期metadata；Relu/Add保留直接参数入口：
```bash
python3 analyze_op.py --tile-metadata \
  docs/tilesight_5_1_gemm_4096_output.json -o report.json
python3 analyze_op.py relu --N 16777216 -o report.json
python3 analyze_op.py add --N 16777216 -o report.json
```

输出 JSON 格式报告：
```json
{
  "op": "gemm",
  "shape": [4096, 4096, 4096],
  "latency_us": 1003.8,
  "ddr_util_pct": 12.3,
  "compute_util_pct": 83.2,
  "tiles_per_sm": 6,
  "waves": 14.2,
  "pipeline": {
    "prologue_us": 0.5,
    "steady_per_iter_us": 1.2,
    "epilogue_us": 0.3
  }
}
```

### 7.2 S5000 架构配置文件

文件：`DeepStack-ae/src/deepstack/mosaic/arch/s5000.py`

已注册到 `arch/__init__.py`，可通过 `from mosaic.arch import S5000` 直接使用。

---

## 八、待办事项

| 优先级 | 事项 | 说明 |
|:------:|:-----|:------|
| 🟢 P0 | BinaryOp/FusedGatedOp 的 is_cuda 验证补丁 | elementwise 的 Add/Sub/Mul 等算子需要在 forward 里也改 is_cuda 检查 |
| 🟢 P0 | DeepStack elementwise 模型校准 | 当前 DDR/Compute 利用率算 0%，模型对 elementwise 不支持 |
| 🟡 P1 | 短 kernel prologue 校准 | 用 P7.2 的 graph-free slope 数据校准 prologue 常数 |
| 🟡 P1 | Reduction/Dropout/RoPE 适配 | 补丁 supported_archs 即可 |
| 🔵 P2 | Moore Perf Tools 安装 | 需 MUSA SDK 5.1.0+，需登录开发者账号下载 |
| 🔵 P2 | 更多算子性能数据采集 | 用 analyze_op.py 批量扫描，建立 S5000 性能基线 |

---

## 九、整体结论

**当前状态：全链路已跑通。**

```
S5000 硬件参数 ✅ → S5000 架构配置 ✅ → DeepStack 分析引擎 ✅
  → GEMM 大 shape 预测 1% 误差 ✅ → elementwise 算子适配 ✅
  → 命令行分析工具 ✅ → JSON 报告输出 ✅
```

**可以做到**：输入算子 + shape + tile 参数，输出 JSON 格式的性能预测报告（延迟、利用率、瓶颈、pipeline 分解）。

**核心指标**：FP16 GEMM 大 shape 预测误差 1.0%，elementwise 适配 28 个 kernel 类，全链路已通。

**下一步**：完善短 kernel 精度，适配更多算子簇，安装 Moore Perf Tools 做硬件级验证。
