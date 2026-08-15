# S5000 GEMM 4096³ DeepStack 预测 vs 硬件实测

日期：2026-08-11  
设备：MTT S5000（mp_31，60 MP/SM）  
Kernel：`gemm_4096_kernel`  
配置：FP16，M=N=K=4096，tile=128×128×64，2 stages

## 1. 本轮优先级 1、2 修正

1. `--nw` 不再同时除以 M、N 两个 tile 维度。TileSight 内部现在严格得到用户请求的 warp 数：`nw=4 -> warp_tile=(64,64,64) -> 4 warp`，`nw=8 -> (32,64,64) -> 8 warp`。
2. L2 参数拆成物理峰值和持续上限：峰值 `7168 GB/s`，持续利用率上限 `0.342`，有效持续带宽约 `2451 GB/s`。旧口径为 `2448 GB/s × 0.95 = 2326 GB/s`，把持续实测值误当成了物理峰值。

MCU 报告该次真实 launch 为 256 threads，即 8 warp。所以下面的硬件验收使用 `nw=8`；`nw=4` 仍是合法的手动模型输入，但不代表该次真实 launch。

## 2. 延迟与计算吞吐

硬件对齐模型使用 8 warp，并使用真实 kernel 的 A/B FP16 读入、FP16 C 写回内存层级。内存层级修正只用于交叉验证，尚未改变通用 wrapper 的默认输入假设。

| 来源 | 延迟 (us) | 吞吐 (TFLOP/s) | 相对 MSYS 延迟误差 |
| --- | ---: | ---: | ---: |
| DeepStack，8 warp + 实际内存层级 + 新 L2 参数 | 994.941 | 138.138 | -3.80% |
| 普通运行（4 次独立进程平均） | 1040.850 | 132.05 | +0.64% |
| MSYS kernel 平均 | 1034.277 | 132.884 | 基准 |
| MSYS kernel 中位数 | 1053.802 | 130.42 | +1.89% |
| MCU Duration | 约 1080 | 约 127.26 | +4.42% |

旧 L2 参数下，同一 8-warp 模型为 1024.197 us。改成 7168 GB/s 峰值、0.342 持续上限后为 994.941 us，说明参数口径拆分不仅修正利用率分母，也使有效持续带宽从 2326 GB/s 调整到约 2451 GB/s。

## 3. 重点指标对比

| 指标 | DeepStack 修正后 | MSYS/MCU 实测 | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 延迟 | 994.941 us | 1034.277 us | -3.80% | 仍是当前主要预测误差，但已在 4% 内 |
| 计算吞吐 | 138.138 TFLOP/s | 132.884 TFLOP/s | +3.95% | 与延迟误差互为倒数关系 |
| L2 峰值带宽利用率 | 30.58% | 29.58% | +1.00 百分点 | 峰值分母统一后可直接对比，已接近实测 |
| L2 命中率 | 约 92.82% | 92.21% | +0.61 百分点 | 误差小，不继续修正 |

DDR、SMEM 和 MCU 通用 Compute(MP) 指标暂不在本轮调整：DDR 还受 wrapper 内存层级输入影响；MCU 没有与 DeepStack SMEM 总流量相同的口径；MCU 的 3.78% 通用 MP 指标也不代表该 GEMM 的算法 Tensor Core 利用率。

## 4. Launch 配置核对

| 项目 | DeepStack 修正后输入 | MCU 实测 |
| --- | ---: | ---: |
| Grid | 32×32 = 1024 CTA | 1024 |
| Block size | `nw=8`，模型内部严格为 8 warp | 256 threads，即 8 warp |
| Warp tile | 32×64×64 | MCU 不直接报告 |
| Dynamic shared memory/block | 实际内存层级模型为 64 KiB | 64 KiB |
| SM/MP count | 60 | 60 |

源码仍写有 `NW=4`、`threads=NW*32=128`，而最终 MCU launch 为 256 threads。当前做法是验收时显式传入 `--nw 8`。长期应从 final-device/launch metadata 自动读取 block size，避免手工输入与最终 kernel 不一致。

## 5. 证据边界

- Moore Perf Compute/System 官方匹配 SDK/Driver 5.2.0，当前宿主机驱动为 `3.3.5-server`；本次有效采集保留应用 SDK 4.3.5 运行环境，由已安装 profiler 外部采集。
- MCU 自定义 section 与默认 section 存在部分冲突。本报告采用默认 section 的 L2 throughput 29.58% 和 L2 hit 92.21%，不采用冲突的自定义 throughput 百分比。
- MCU replay 期间应用自身打印的 20–33 ms 包含 profiler 开销，不作为 kernel 基线。

## 6. 结论

1. 优先级 1 已完成：wrapper 的 `nw` 与 TileSight 内部 warp 数现在一致；本次真实 kernel 用 `nw=8` 对齐 MCU。
2. 优先级 2 已完成：L2 峰值和持续利用率已分开表达，L2 利用率由错误的约 87% 降到 30.58%，与 MCU 29.58% 相差 1.00 个百分点。
3. 修正后硬件对齐模型的延迟误差为 -3.80%，计算吞吐误差为 +3.95%。误差较小的 L2 命中率不再调整。

## 7. 证据文件

- DeepStack wrapper 报告：`results/reports/gemm_4096.json`
- 普通运行：`baseline.log`、`baseline_trial_1.log` 至 `baseline_trial_3.log`
- MSYS 报告：`gemm_4096_msys_sdk435.msys-rep`
- MSYS kernel 汇总：`msys_kernel_summary.txt`
- MCU 默认报告：`gemm_4096_mcu.mcu-rep`
- MCU 默认捕获日志：`mcu_capture.log`
- MCU 详细捕获日志：`mcu_detailed_capture.log`
