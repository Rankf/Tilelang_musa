# S5000 算子性能分析框架

基于 **DeepStack** 引擎，为 **MTT S5000** GPU 构建算子级性能分析工具。
服务手工算子优化与 Agent 算子评估两个场景。

> 当前 TileSight/S5000 公式、运行时接口、实测结果和后续工作统一入口：
> [docs/tilesight_s5000/README.md](docs/tilesight_s5000/README.md)。历史完成率和旧交接结论应以该文档为准进行校正。
>
> 目录导航、第一批交付包入口和缓存清理边界见
> [当前结构与工作状态](docs/tilesight_s5000/current/00_project_map_20260928.md)。

---

## 一、项目结构

```
s5000-calibration/
├── calibration_docs/          # 校准文档（P0~P7 共 30 篇）
├── scripts/                   # 校准脚本（64 个）
│   ├── p0_*.sh                # 环境审计
│   ├── p1_*.sh                # 测量基础设施
│   ├── p2_*.sh/.py            # 计算性能测试
│   ├── p3_*.sh/.py            # 内存性能测试
│   ├── p4_*.sh/.py            # Occupancy/Wave 测试
│   ├── p5_*.py                # 参数模型构建
│   ├── p6_*.py                # Hold-out 验证
│   └── p7_*.sh/.py            # 运行时诊断
├── benchmarks/                # 基准测试代码（24 个）
│   ├── p1_tilelang_*.py       # TileLang 测试
│   ├── p2_tilelang_*.py       # 计算测试
│   ├── p3_musa_*.mu           # 原生 MUSA 探针
│   ├── p3_tilelang_*.py       # 全局内存测试
│   └── p4_musa_*.mu/.cpp      # Occupancy 探针
├── configs/                   # 校准配置
│   └── runtime/               # Event/MCU/MSYS runtime-only 采集配置
├── docs/tilesight_s5000/      # 当前统一说明与 2026-09-16 交接文档
├── evidence/legacy/           # 已版本化的历史 profiler 原始证据
├── musa_operator_fixes/       # MUSA 算子适配（自定义）
│   ├── gemm/dense/            # GEMM 适配
│   │   └── mp31_fp16_nn.py    # MusaMP31DenseGemmKernel
│   └── elementwise_sfu/       # Elementwise 适配
│       └── mp31_elementwise.py # 28 个 kernel 类补丁
├── analyze_op.py              # 命令行性能分析工具
├── DeepStack-ae/              # 第三方：DeepStack（含 s5000.py 架构注入）
├── DeepStack-ae-source/       # 第三方：DeepStack 源码（含 tilesight）
├── TileOPs/                   # 第三方：TileOPs 算子库
├── tilelang-musa/             # 第三方：TileLang-MUSA 编译器
├── results/                   # [不上传] 测试结果数据
├── logs/                      # [不上传] 日志
└── .venv-tilelang-musa/       # [不上传] Python 虚拟环境

musa-perf-collector/           # 独立项目：Tile 级运行时 trace 采集
├── src/musa_perf_collector/   # 源码（61 个 py）
├── configs/                   # 配置（19 个）
├── docs/                      # 设计文档（6 篇）
├── tests/                     # 测试（23 个）
├── results/                   # [不上传] 运行结果
└── build/                     # [不上传] 构建产物
```

---

## 二、核心目标

**性能分析工具输出**：
- 算子延迟预测（µs）
- 有效算力（TFLOP/s）、有效带宽（GB/s）
- 瓶颈分析（计算/访存/Occupancy/Tail 效应）
- 优化建议（tile 大小、pipeline stage、warp 数）

**两个使用场景**：
- **场景 A**：帮助手工算子优化
- **场景 B**：帮助 Agent 生成的算子评估

---

## 三、S5000 硬件参数（已全部获取）

### 3.1 必须参数（9 个，全部到位）

| 参数 | 值 | 获取方式 | 置信度 |
|:-----|:---|:---------|:------:|
| **sm_count** | 60 | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **core_frequency_hz** | 1.5 GHz | musaGetDeviceProperties clockRate | ⭐⭐⭐⭐⭐ |
| **memory_frequency_hz** | 20 GHz（有效） | musaGetDeviceProperties memoryClockRate | ⭐⭐⭐⭐⭐ |
| **tensor_cores_per_sm** | 1 | FP16 GEMM 实测反推（83.3% 效率） | ⭐⭐⭐⭐ |
| **tensor_core_shape** | (16, 8, 8) | TileLang-MUSA mma.h 源码确认 | ⭐⭐⭐⭐⭐ |
| **fp32_cores_per_sm** | 128 | FP32 实测反推（98.5% 效率） | ⭐⭐⭐⭐⭐ |
| **l2_capacity_bytes** | 60 MiB | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **shared_memory_capacity_bytes** | 192 KiB | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |
| **register_capacity_per_sm_bytes** | 131,072 | musaGetDeviceProperties | ⭐⭐⭐⭐⭐ |

### 3.2 重要但可用默认值

| 参数 | 当前值 | 状态 |
|:-----|:------|:----:|
| **ddr_max_utilization** | 0.876（实测反推） | ✅ 已更新 |
| **compute_max_utilization** | 0.832（实测反推） | ✅ 已更新 |
| l2_max_utilization | 0.9（默认） | ❌ 默认 |
| l1_max_utilization | 0.9（默认） | ❌ 默认 |
| sm_sub_partitions | 4（默认） | ❌ 默认 |
| warp_schedulers_per_sm | 4（默认） | ❌ 默认 |
| shared_memory_throughput_bytes_per_cycle | 128（默认） | ❌ 默认 |
| sfu_cores_per_sm | 16（默认） | ❌ 默认 |
| int32_cores_per_sm | 64（默认） | ❌ 默认 |

---

## 四、跨校验（硬件参数可靠性）

| 验证链 | 理论值 | 实测值 | 效率 | 判定 |
|:-------|:------:|:------:|:----:|:----:|
| 显存带宽 | 640bit×20GHz÷8 = 1,600 GB/s | fill 1,401 GB/s | 87.6% | ✅ 正常 |
| FP32 算力 | 60×128×1.5G×2 = 23.0 TFLOP/s | 22.7 TFLOP/s | 98.5% | ✅ 正常 |
| Tensor Core 算力 | 60×1×(16×8×8×2)×1.5G = 184 TFLOP/s | 153.4 TFLOP/s | 83.3% | ✅ 正常 |

---

## 五、DeepStack 预测验证

| Case | S5000 实测 | DeepStack 预测 | 误差 |
|:-----|:---------:|:------------:|:----:|
| FP16 GEMM 4096³ (tile128 stage2) | 993.5 µs | 1,003.8 µs | **1.0%** |
| BF16 GEMM 128×7168×2048 | 59.7 µs | 55.3 µs | **7.3%** |
| FP16 GEMM 1024³ | 28.1 µs | 35.1 µs | 22.6% |

---

## 六、TileOPs 算子适配

| 算子簇 | 状态 | 适配方式 |
|:-------|:----:|:---------|
| **GEMM dense** | ✅ | MusaMP31DenseGemmKernel（独立实现） |
| **Elementwise unary**（Relu/Exp/Sigmoid 等 28 个） | ✅ | supported_archs + validation 补丁 |

---

## 七、使用方式

### 7.1 命令行性能分析

```bash
cd /root/s5000-calibration
# GEMM：模型输入由5.1编译期metadata自动推导
python3 analyze_op.py \
  --tile-metadata docs/tilesight_5_1_gemm_4096_output.json \
  -o report.json
# Relu
PYTHONPATH=... python3 analyze_op.py relu --N 16777216 -o report.json
# Add
PYTHONPATH=... python3 analyze_op.py add --N 16777216 -o report.json
```

### 7.2 环境

- 连接：SSH 到 S5000（Docker 容器，GPU 透传）
- MUSA SDK：4.3.5
- Python：3.10（.venv-tilelang-musa）
- TileLang-MUSA：commit 4d666968

---

## 八、环境限制

- **Docker 容器**：不是物理机，无法升级 GPU 驱动
- **Moore Perf Tools 未安装**：需 MUSA SDK 5.1.0+，当前 4.3.5 不支持
- **MCU 硬件计数**：暂无替代方案，用 MUSA Event + DeepStack 分析

---

## 九、Git 管理说明

**应提交**：scripts/, benchmarks/, configs/, calibration_docs/, musa_operator_fixes/, analyze_op.py, s5000.py，以及 TileLang-MUSA、DeepStack/TileSight 和 musa-perf-collector 的项目源码快照与本机自定义修改。

**默认忽略**：results/, logs/, .venv-tilelang-musa/, __pycache__/, build/, core_*.mudmp。在更换或重置 S5000 服务器前，已审核且无法从源码重建的校准证据可作为快照显式提交。

**上游源码快照**：DeepStack-ae/、DeepStack-ae-source/、TileOPs/ 和 tilelang-musa/ 由顶层仓库跟踪，以确保租赁服务器重置后可完整恢复。子目录中的 `.git` 元数据和本地镜像 DeepStack-git/ 不提交。

### S4000 的 TileLang-MUSA SDK 兼容补丁

[0.1.12+musa.5 的修改说明、应用步骤和真机验证](patches/tilelang-musa/0.1.12-musa.5/README.md)
