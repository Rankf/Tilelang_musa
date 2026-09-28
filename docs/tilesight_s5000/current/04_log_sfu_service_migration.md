# Log 统一 SFU 服务公式迁移

## 结论

Log 已从临时 `log_sequence` 计算资源迁移到统一 SFU 服务入口：

```text
compute_resource = sfu
special_math_op = log
effective_service_rate = 247.469 Gop/s
```

`247.469 Gop/s` 是包含辅助指令的实测持续 Log 序列服务率，不是物理 SFU 峰值。迁移没有重新拟合参数，也没有修改 Pipeline、Wave、Occupancy 或缓存公式。

统一服务公式为：

```text
R_effective(op) = per-op measured service rate, if present
                  otherwise sfu_flops * compute_max_util
T_compute = semantic_operations / R_effective(op)
```

旧 `log_sequence_ops_per_s`、Log CTA overlap 字段及 `compute_resource=log_sequence` 保留兼容读取；S5000 默认配置只写 `special_math_service_profiles`。

## 等价与留出验证

所有归档形状的迁移前后预测差值均为 `0.0 µs`（门槛 `0.011 µs`）。

| Shape | 分组 | 预测 µs | MCU µs | 误差 |
| ---: | --- | ---: | ---: | ---: |
| 6144² | development | 189.780 | 189.587 | +0.102% |
| 12288² | development | 704.960 | 705.192 | -0.033% |
| 24576² | development | 2767.460 | 2767.416 | +0.002% |
| 5120² | regression holdout | 137.200 | 136.239 | +0.705% |
| 10240² | regression holdout | 495.170 | 496.493 | -0.266% |

已有留出结果此前已经被观察，因此本次属于迁移回归证据，不称为新的盲测。

正式审计：`results/profiling/log_sfu_service_migration_20260916/audit.json`。
