# TileLang Add/ReLU：TileSight 与 MCU 实测对比

日期：2026-08-14  
设备：MTT S5000  
数据入口：真实 `workloads/elementwise_tilelang.py` PrimFunc；未使用历史 C++ probe。

## 测试口径

- TileSight：统一 compiler-info 接口，显式 `cache_condition=cold`。当前原生 N_0/N_N 模型只支持 cold/no-reuse，warm 请求会 fail-closed。
- MCU cold：输入生成后对 256 MiB 未初始化 scratch 做只读 sweep，驱逐 60 MiB L2；`-s 0 -c 1` 采第一轮目标 Kernel。
- MCU warm：连续发射 6 次目标 Kernel；`-s 5 -c 1` 采第六轮。
- MUSA Event：20 次预热后，采 100 个单 Kernel 样本。
- MCU 1.3.0 在当前容器只暴露 `/dev/mtgpu.4` 时无法枚举逻辑设备 0。采集期间临时建立 `/dev/mtgpu.0 -> /dev/mtgpu.4`，每次命令结束立即移除；未改驱动或服务。

## 身份与正确性

| 算子 | N | 编译事实 grid×block | MCU grid×block | 每线程元素 | Kernel 源码 SHA256 | 数值 |
|---|---:|---:|---:|---:|---|---|
| Add | 16,384 | 8×256 | 8×256 | 8 | `9dc987000271bbd0f41f410f7698ef4e06438ed5863229fcf73f23c6c26d2155` | pass |
| ReLU | 16,384 | 16×256 | 16×256 | 4 | `1c2942a9a983d0707a11e11227380cf97ba72ce778933b777dba26120849e28c` | pass |

两组 TileSight `compiler_validation` 均为 `all_consistent`；cold/warm MCU 的 6 个 replay pass 中，运行摘要逐字一致。

## 延迟

| 算子 | TileSight 原生 cold | 报告层 cold（含内存层级约束） | MCU cold | 实际/报告层 | 报告层低估 | MCU warm | Event warm 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Add | 0.1164 µs | 0.1400 µs | 7.1560 µs | 51.11× | 98.04% | 6.5857 µs | 7.0400 µs |
| ReLU | 0.0581 µs | 0.0900 µs | 6.8011 µs | 75.57× | 98.68% | 6.0640 µs | 6.7200 µs |

结论：这两个短 Kernel 的总延迟主要不是带宽服务时间。当前 TileSight 输出的是亚微秒级计算/内存服务成本，没有覆盖约 6–7 µs 的固定启动、调度和极小 grid 欠填充开销。MCU 的 `launch__waves_per_multiprocessor` 只有 Add 0.011111、ReLU 0.022222；因此报告中的“DDR 瓶颈”只能解释模型内部的资源临界路径，不能解释实测总延迟。

warm MCU 相对 Event 中位数分别低 6.45% 和 9.76%，绝对差 0.45–0.66 µs；两种计时方法趋势一致，但不能视作完全相同的计时口径。

## 流量

### L1 下行：静态逻辑流量基本准确

| 算子 | TileSight/编译语义读 | MCU cold L1 下行读 | 偏差 | TileSight/编译语义写 | MCU cold L1 下行写 | 偏差 |
|---|---:|---:|---:|---:|---:|---:|
| Add | 131,072 B | 134,912 B | +2.93% | 65,536 B | 65,536 B | 0.00% |
| ReLU | 65,536 B | 69,376 B | +5.86% | 65,536 B | 65,536 B | 0.00% |

这部分验证了编译信息适合给 TileSight提供 shape、tile、launch、操作数角色和逻辑读写量。两组读取都多出固定 3,840 B，属于模型尚未表达的物理事务开销；不能反向覆盖编译器事实。

### DDR：读流量接近，写回时域假设不成立

| 算子 | TileSight cold DDR 读/写/总量 | MCU cold DDR 读/写/总量 | 读偏差 | 写偏差 | 总量偏差 |
|---|---:|---:|---:|---:|---:|
| Add | 131,072 / 65,536 / 196,608 B | 148,864 / 2,080 / 150,944 B | +13.57% | -96.83% | -23.23% |
| ReLU | 65,536 / 65,536 / 131,072 B | 83,328 / 2,080 / 85,408 B | +27.15% | -96.83% | -34.84% |

冷态读流量的绝对额外量两组都是 17,792 B，提示存在固定物理事务/背景开销，尚不能仅凭这两点确定具体来源。更明确的问题是写流量：TileSight 的 `writeback_reaches_ddr_in_kernel_scope` 假设把全部输出写入计入当前 Kernel；MCU 表明当前 Kernel 结束时大部分脏输出仍留在 L2，后续驱逐才会写回。因此逻辑写 65,536 B 是正确的编译/语义事实，但不应直接等同于 kernel-scope DDR write。

warm MCU 的物理 DDR 总量为 Add 30,848 B、ReLU 4,800 B，不能与 TileSight cold 预测计算误差。统一接口拒绝 warm 是合理的，后续若要支持 warm，应增加独立场景模型，而不是改写编译器事实或把 profiler 数值灌入 TileSight 原生参数。

MCU 的 `l2pu__sector_hit_rate.pct` 包含混合 sector 类别和写分配，其分母不等同于 TileSight 的“输入逻辑读命中率”，本报告不将两者直接计算误差。两组 `l2pu__tme_read_l2_byte.avg=0`，与普通 elementwise 无 TMA 路径一致。

## 审核与排除项

- 正式 cold 采用只读驱逐。早期 `add_n16384_cold_real.mcu-rep` 使用写入式驱逐，DDR write 被脏 scratch 延迟回写污染，已明确排除。
- MCU 修复前生成的 49,152-byte 空报告均显示 `No kernels were profiled`，不作为实测。
- 正式报告无 `==ERROR==`，数值检查均 pass；临时 `/dev/mtgpu.0` 已确认不存在。
- adapter 回归测试：9 项通过；运行入口 `py_compile` 和 `git diff --check` 通过。

完整机器可读数值见 `comparison.json`，原始 MCU 报告和控制台保存在本目录。
