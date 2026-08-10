# P4 单卡 occupancy / wave 校准审核

状态：completed（P4.1 资源与精确 wave 补测已并入）  
日期：2026-07-29  
对象：1× MTT S5000，MUSA 4.3.5，TileLang-MUSA `4d666968a005846314e9a84d8ff121c5dbffa9ee`

## 结论

P4 完成了两条严格分离的证据链：

1. 原生 MUSA probe 对**精确原生 kernel**调用 MUSA occupancy API，得到了本卡 API 计算的 active CTA/MP、function resource 属性以及由该容量导出的 wave/tail 行为。
2. TileLang-MUSA 对实际生成的 fixed-grid FP32 copy kernel 完成 block 与 wave 位置扫描；每一 case 均通过编译、数值正确性、MUSAGraph 计时和 MUSA profiler self-device time 交叉门禁，并独立执行三次。P4.1 又从正式保留的 `.mu` 和原 mcc 命令重编译 MUBIN，通过 Driver API 查询该生成 kernel 的资源与 occupancy。

两条证据不得合并为“TileLang kernel 有 12 CTA/MP”。原生 `s0` kernel 的 12 CTA/MP 只对其 256-thread、4 register/thread、12 B static-shared 的 xorshift kernel 有效。P4.1 表明普通 TileLang fixed-grid copy 为 56 registers/thread、256 threads、零 static/dynamic shared，occupancy API 给出 8 CTA/MP；60 MP 对应条件容量 480 CTA。

资源结论有明确 provenance 限制：正式计时 dump 保留了 `.mu` 与 `.cmd`，没有保留当时执行的原始 `.mubin`。Driver API 查询的是从完全相同 `.mu` 和 mcc 命令重编译的 MUBIN；源码 SHA256 与新的正式 pipeline run 一致，但不能声称 queried binary 与当时执行 binary 逐字节相同。

## SDK/驱动资源与 occupancy API

原生 probe 的 `musaGetDeviceProperties` 报告：60 MP、196,608 B shared memory/MP、131,072 个 32-bit registers/MP、架构上限 256 CTA/MP、warp=32。这些是 SDK/驱动查询值；其中“架构 CTA 上限”并不等于任何 kernel 的实际 residency。

MUSA 4.3.5 头文件提供 `musaFuncGetAttributes` 和 `musaOccupancyMaxActiveBlocksPerMultiprocessor*`。实际链接审计显示 `libmusart.so` 导出的是 `musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`，而不是无 flags 拼写；P4 因此显式调用导出的 WithFlags ABI，flags=`musaOccupancyDefault`。这个差异及初始链接失败都保留在 P4 构建日志/源码修订中。

| 原生 kernel | static shared（function report） | block threads | API active CTA/MP |
|---|---:|---:|---:|
| s0 | 12 B | 64 / 128 / 256 / 384 / 512 | 48 / 24 / 12 / 8 / 6 |
| s8k | 8,204 B | 64 / 128 / 256 / 384 / 512 | 23 / 23 / 12 / 8 / 6 |
| s16k | 16,396 B | 64 / 128 / 256 / 384 / 512 | 11 / 11 / 11 / 8 / 6 |
| s32k | 32,780 B | 64 / 128 / 256 / 384 / 512 | 5 / 5 / 5 / 5 / 5 |

function report 对四种原生 kernel 均为 4 registers/thread、maxThreadsPerBlock=1024。表中的 12/32,780 B 包含编译器/ABI 可见的少量额外 shared 开销，因此不能只用源代码数组大小替代 runtime resource 属性。

## 原生 wave 证据

`s0`、256 threads 的 API capacity 是 `60 MP × 12 CTA/MP = 720 CTA`。每个 CTA 运行 512 次依赖 uint32 recurrence，输出与 CPU reference 精确比较；MUSA event 三进程中位数为：

| grid CTA | 719 | 720 | 721 | 1439 | 1440 | 1441 |
|---|---:|---:|---:|---:|---:|---:|
| event time（µs） | 134.08 | 134.00 | 137.44 | 247.68 | 247.28 | 256.84 |

在 720→721 处为约 +2.57%，在 1440→1441 处为约 +3.87%。这验证了此原生 kernel 的 API-derived wave 边界附近有可重复 tail 开销；不说明硬件调度映射、也不可以外推到 TileLang/GEMM/attention。

## TileLang-MUSA 实测行为

TileLang case 为 FP32 fixed-grid copy、预分配 A/C、每 CTA 128 次 copy iteration。生成的 MUSA source 明确有对应 `__launch_bounds__(threads, 1)`，mcc 命令明确为 `--cuda-gpu-arch=mp_31`。P4.1 对重编译 MUBIN 的 Driver API 查询结果为：64/128/256/384/512 threads 时均为 56 registers/thread、零 local/static/dynamic shared，对应 34/17/8/5/4 active CTA/MP。短 case `tl_wave_1` 生成的是另一个 39-register kernel，对应 11 CTA/MP，不能与普通 fixed-grid kernel 合并。

固定总 logical payload 754,974,720 B 的三进程中位数：

| threads | CTA grid | latency（µs） | logical GB/s |
|---:|---:|---:|---:|
| 64 | 11,520 | 639.31 | 1180.91 |
| 128 | 5,760 | 632.54 | 1193.57 |
| 256 | 2,880 | 627.46 | 1203.23 |
| 384 | 1,920 | 632.09 | 1194.41 |
| 512 | 1,440 | 628.38 | 1201.46 |

这是当前 TileLang lowering/schedule 下的条件有效带宽，不是 S5000 DDR 峰值；与 P3 vector=8 streaming copy 不是同一 kernel，不能混为单个“带宽规格”。256 threads 是本组的最高中位 logical GB/s，作为 P4 固定-grid copy 条件参数记录。

按 8 CTA/MP × 60 MP 得到的精确位置重新扫描，三进程中位数如下：

| grid CTA | 479 | 480 | 481 | 959 | 960 | 961 |
|---|---:|---:|---:|---:|---:|---:|
| latency（µs） | 99.543 | 99.906 | 100.429 | 203.045 | 206.579 | 204.270 |

480→481 为 +0.523%，960→961 为 -1.118%。所有 case 都通过正确性与稳定性门禁，但两个边界没有一致的正向 tail step，因此这两个百分比标为 `empirical_inconclusive`，不进入尾波开销拟合。occupancy API 容量仍可作为该重编译 kernel 的条件资源参数；它不提供 CTA 到 MP/scheduler 的实际映射。

`tl_wave_1` 虽正确且 graph/profiler 一致，但为约 6.72 µs 的短 kernel，固定开销比例高，未进入最终性能拟合参数。

### P4.2 compute-dominant TileLang 尾波补测

copy kernel 的内存流量会掩盖调度边界，因此新增 TileLang `fixed_grid_compute`：每线程一次输入读取、一次输出写入，中间执行 2048 次依赖 FP32 recurrence。生成源码在所有 grid 下 SHA256 相同。重编译 MUBIN 的三次资源查询一致：256 threads、3 registers/thread、零 static/dynamic shared、零 local/spill、12 active CTA/MP；60 MP 对应 720 CTA API capacity。

围绕 1×、2×、3× capacity 测得三个完整通过进程的中位数：

| grid CTA | 719 | 720 | 721 | 1439 | 1440 | 1441 | 2159 | 2160 | 2161 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latency（µs） | 33.714 | 33.997 | 34.389 | 60.840 | 60.723 | 63.513 | 89.030 | 89.024 | 90.223 |

边界后增加 1 CTA 的增量分别为 +0.392、+2.790、+1.199 µs，即 +1.152%、+4.594%、+1.347%。三个边界方向一致为正，因此原先“TileLang 无任何可拟合 tail 参数”的缺口已缩小为一个 **kernel-specific 条件范围**：`0.392–2.790 µs`，中位 `1.199 µs`。它可以在 P5 中作为该 compute-wave kernel family 的区间参数或不确定性先验，但不能作为所有 TileLang kernel 的统一固定百分比；copy family 仍保持 inconclusive。

原三进程中的 process 2 在 721 CTA 出现一次 direct-event 预估污染，graph/profiler ratio 20.99% 超过 20% 门槛，按协议整进程拒绝并完整执行 process 4。聚合明确选择 process 1/3/4，失败 process 2 与原因保留在 JSON，不做单点静默替换。

## TileLang pipeline-stage 曲线

补充使用同一 FP16 GEMM（1024³，BM=BN=BK=64，4 warp、generated launch=256 threads）扫描 `T.Pipelined` 的 1–4 stage。加入 `-resource-usage` 后重新完成三个独立进程，全部 compile/correct/measured；当前配置哈希对应的中位数如下：

| stages | latency（µs） | effective TFLOP/s |
|---:|---:|---:|
| 1 | 22.981 | 93.444 |
| 2 | 21.670 | 99.098 |
| 3 | 26.104 | 82.266 |
| 4 | 26.909 | 79.805 |

在这个特定 TileLang GEMM/schedule 下，stage 2 是本扫描的最佳点；stage 3/4 退化。降低后的 TIR 对 stage 1–4 分别声明 16/32/48/64 KiB dynamic shared。重编译 MUBIN 的 Driver API 均报告 45 registers/thread、零 static shared/local memory，对应 10/6/4/3 active CTA/MP。mcc `-resource-usage` 的 Temp register 同样为 45，且 local、spill load/store 均为 0；mcc 输出里的 `Shared` register class 随 stage 为 63/65/67/69，不能解释为 shared-memory 字节。该曲线只适用于此 kernel family 和 schedule。

## 受控寄存器压力曲线

原生 register-pressure kernel 独立执行三次并做精确数值检查。所有配置 local memory=0，mcc spill load/store=0；因此曲线体现的是无 spill 条件下寄存器增长与 occupancy API 结果的关系：

| live values | runtime registers/thread | active CTA/MP（64 / 128 / 256 / 512 threads） |
|---:|---:|---:|
| 4 | 6 | 48 / 24 / 12 / 6 |
| 16 | 24 | 48 / 24 / 12 / 6 |
| 32 | 53 | 34 / 17 / 8 / 4 |
| 64 | 115 | 16 / 8 / 4 / 2 |

这不是通用“寄存器阈值公式”，但可作为 S5000 上 exact-kernel occupancy 校准和后续模型单调性检查的实测锚点。

## 已保留的失败与修复证据

- 原生 mcc 未显式指定 arch 时，默认落到 `mp_21`，32 KiB static shared 超过该目标 28 KiB limit；可复核失败日志为 `logs/build/p4_musa_occupancy_default_arch_failure.log`。该失败不能用于 S5000。正式 runner 显式使用 `--cuda-gpu-arch=mp_31`，与 TileLang-MUSA mcc 命令一致。
- 初次 occupancy 调用使用无 flags 名称，链接发现该符号未被当前 `libmusart.so` 导出；当前库导出审计见 `logs/build/p4_musa_occupancy_abi_symbol_audit.log`。正式代码使用已导出的 WithFlags ABI。
- 初次 shared probe 仅初始化 shared 数组前 `blockDim.x` 个元素，精确 CPU check 在 index 16 失败。修复为分段初始化整个数组后，三次正式 run 全部 correctness pass。

这些失败目录/构建日志保留，不进入聚合或参数表。

## 可追溯交付物

- scope：`configs/p4_scope.json`、`configs/p4_occupancy_wave.json`、`configs/p4_tilelang_wave.json`
- 原生 benchmark/runner：`benchmarks/p4_musa_occupancy_wave.mu`、`scripts/p4_run_occupancy_wave.sh`
- TileLang runner：`benchmarks/p3_tilelang_global_memory.py`（P4 config 复用）、`scripts/p4_run_tilelang_wave.sh`
- pipeline runner：`configs/p4_tilelang_pipeline.json`、`scripts/p4_run_tilelang_pipeline.sh`、`scripts/p4_finalize_pipeline.py`
- P4.1 resource/register/exact-wave：`benchmarks/p4_musa_mubin_resource_probe.cpp`、`benchmarks/p4_musa_register_pressure.mu`、`scripts/p4_run_tilelang_resource_probe.sh`、`scripts/p4_run_tilelang_wave_resource_probe.sh`、`scripts/p4_run_register_pressure.sh`、`scripts/p4_run_tilelang_exact_wave.sh`
- P4.2 compute-wave：`configs/p4_tilelang_compute_wave.json`、`scripts/p4_run_tilelang_compute_wave.sh`、`scripts/p4_run_tilelang_compute_wave_resource_probe.sh`、`scripts/p4_finalize_compute_wave.py`
- 原生 raw：`results/raw/p4_occupancy_wave/20260729T025118233217353Z/`
- TileLang raw：`results/raw/p4_tilelang_wave/20260729T025312594585659Z/`
- generated MUSA、mcc command：`logs/build/p4_tilelang_wave/20260729T025312594585659Z/`
- 聚合与参数：`results/processed/p4_occupancy_wave/aggregate_native.csv`、`aggregate_tilelang.csv`、`s5000_occupancy_wave_parameters_provisional.csv/json`
- pipeline 聚合：`results/processed/p4_occupancy_wave/aggregate_tilelang_pipeline.csv/json`
- P4.1 聚合：`aggregate_tilelang_pipeline_resources.csv`、`aggregate_tilelang_wave_resources.csv`、`aggregate_register_pressure.csv`、`aggregate_tilelang_exact_wave.csv`、`p4_1_manifest.json`
- P4.2 聚合：`aggregate_tilelang_compute_wave.csv/json`
- 聚合脚本：`scripts/p4_finalize_occupancy_wave.py`、`scripts/p4_finalize_extensions.py`

## P5 前置结论

P4 已提供 device resource、原生 exact-kernel occupancy、受控 register-pressure 曲线、TileLang fixed-grid 与 pipeline-stage 的重编译 MUBIN 资源曲线，以及 copy/compute 两类 wave timing。它足以进入 P5 的条件 cost-model/hold-out 验证，但不能把 8/12 CTA/MP 或 pipeline 的 10/6/4/3 外推到任意 TileLang kernel；每个新 kernel 仍应按实际 registers、dynamic shared、threads 查询。compute-dominant family 已得到 `0.392–2.790 µs` 的条件 tail 区间，copy family 仍为 inconclusive。仍未知的是 scheduler-to-MP 映射、queried recompiled MUBIN 与正式执行 binary 的逐字节同一性，以及跨 kernel family 的统一 tail 常数。
