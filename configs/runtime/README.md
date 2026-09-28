# S5000 runtime collection configurations

本目录只保存 Event、Moore Perf Compute 和 Moore Perf System 的 runtime-only 采集配置。

- `runtime_collection_add_*`、`runtime_collection_relu_*`：基础 Elementwise。
- `runtime_collection_tileops_{exp,log,sqrt}_*`：TileOPs 特殊数学算子与 Shape。
- `s5000_pipeline_stage_probe.json`：固定资源 footprint 的 stage 1/2/3
  独立 MUSA Pipeline 开发集、留出集和验收门。
- 配置不得包含 TileSight 模型结果或预测值。
- MCU 指标包含 Kernel 时间、launch、Waves、L1/L2PU/DDR，以及 5.6 使用的 MP 和 Shared load/store pipe counter。

采集入口：

```bash
python3 scripts/collect_s5000_runtime_measurements.py \
  --spec configs/runtime/<config>.json \
  --output results/profiling/<collection>
```

采集前先使用 `--dry-run` 检查命令、设备和输出目录。
