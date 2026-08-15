# musa-perf-collector

TileLang-MUSA 运行时性能采集工具。在 TileLang 编译期插入 clock64 探针，运行时采集每个 TileOp 的起止时间，用于细粒度算子性能分析。

---

## 作用

```text
TileLang 源码
    ↓ 编译期插桩（pass_capture.py）
    ↓ 插入 clock64 读指令
S5000 上运行
    ↓ 采集 clock64 起止时间
原始 trace（trace.jsonl）
    ↓ 关联离线元数据
Enriched trace（trace_enriched.jsonl）
```

与顶层 S5000 性能分析工具（硬件实测 + DeepStack/TileSight 预测）互补，提供的是 **Tile 级细粒度时间分解**，而非端到端性能预测。

---

## 目录结构

```text
musa-perf-collector/
├── src/musa_perf_collector/       # 核心源码
│   ├── adapters/tilelang/         # TileLang 适配层与 compiler-info API
│   ├── collectors/compiler/       # 编译器信息采集
│   ├── collectors/runtime/        # 运行时信息采集
│   ├── core/                      # 基础工具
│   ├── online/                    # 在线 JIT 自动插桩
│   ├── exporters/                 # 结果导出
│   └── examples/                  # 使用示例
├── configs/                       # 采集与 workload 配置
├── docs/                          # 设计与 API 文档
├── tests/                         # 单元和集成测试
├── results/                       # 已审核的离线/在线采集证据快照
├── pyproject.toml
└── setup.py
```

---

## 核心能力

| 能力 | 说明 |
|:-----|:-----|
| CPU 离线分析 | 无损 TileOp 发现、FLOPs/逻辑字节/DAG/Reuse distance 分析 |
| S5000 运行时采集 | MUSA Event 计时、正确性验证、clock64 Tile 级 trace |
| 在线 JIT 插桩 | 编译 hook、baseline/instrumented 双变体、自动探针 |
| 编译信息 API | 通过公开 PassInstrument 接口采集结构化 compiler facts |
| 低扰动 | 默认 every=64 采样、原生 C fast dispatch、D2H 异步 |

---

## 运行命令

```bash
# 离线分析
PYTHONPATH=src python3 -m musa_perf_collector.offline --config ...

# 编译信息采集
PYTHONPATH=src python3 -m musa_perf_collector offline analyze \
  --factory package.module:function --config config.json --output results/offline

# 在线采集
MUSA_PERF_COLLECTOR_ONLINE_CONFIG=configs/runtime/online_gemm_reduce_trace.json \
  PYTHONPATH=src python3 -m musa_perf_collector.examples.online_gemm_reduce_app

# 运行测试
PYTHONPATH=src python3 -m pytest tests/unit tests/integration
```

---

## 输出结果

每次在线运行输出一个 session 目录，主要包含：

| 文件 | 说明 |
|:-----|:-----|
| `session_validation.json` | 会话状态（complete/partial/invalid） |
| `session_audit.json` | 独立审计报告 |
| `trace.jsonl` | 原始 clock64 trace |
| `trace_enriched.jsonl` | 关联离线元数据后的 trace |
| `correctness.jsonl` | 正确性验证结果 |
| `capability.json` | 启用的采集能力 |
| `diagnostics.jsonl` | 异常记录 |

---

## 依赖与恢复

- Python ≥ 3.10
- MUSA SDK（原始校准环境为 4.3.5）
- 顶层仓库中的 `tilelang-musa/`
- `torch` / `torch-musa`

`build/`、Python `.so`、egg-info 和缓存不入库，在新服务器上通过安装/编译重建。已审核的 `results/` 快照用于保留无法从源码重建的 S5000 实测证据。
