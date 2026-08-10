# MTT S5000 单卡真实测量数据汇总

状态：reviewed human-readable snapshot  
测量日期：2026-07-28 至 2026-07-29（UTC+8）  
审核日期：2026-07-29  

## 1. 这份文档是什么

本文只汇总本机 MTT S5000 上已经实际运行得到的数据，以及 MUSA Runtime/Driver API 实际返回的
设备属性。它不是官方规格表，不包含 NVIDIA/AMD 参数，也不把 P5/P6 的模型预测值冒充硬件实测值。

表中术语：

- **API 报告**：由当前 MUSA 3.3.5-server / SDK 4.3.5 runtime 或 driver API 返回；
- **条件实测**：只对表中 shape、dtype、tile、线程、工作集、源码和计时方法成立；
- **逻辑带宽**：按 benchmark 读写的逻辑 payload 字节数除以时间计算，不等于物理总线峰值；
- **有效吞吐**：按算子逻辑工作量除以实测时间计算，不等于芯片理论峰值；
- **unknown**：证据不足，明确不猜测。

正式 compute/memory 数据通常取三个独立进程中位数，并要求编译、正确性与稳定性门禁通过。
P4 资源查询注明“重编译 MUBIN”的项目不是对当时执行 binary 的逐字节查询。

## 2. 测试环境

| 项目 | 本机记录 |
|---|---|
| GPU | 1× MTT S5000，81920 MiB |
| 驱动 | MUSA 3.3.5-server |
| SDK / mcc | `/usr/local/musa-4.3.5` / mcc 4.3.5 / clang 14.0.0 |
| runtime capability | `(3, 1)`，即 MP31 目标 |
| Python | 3.10.12 |
| PyTorch / torch_musa | `2.7.1a0+gite2d141d` / `2.7.1+5ee0a64` |
| TileLang-MUSA | commit `4d666968a005846314e9a84d8ff121c5dbffa9ee`，含本地 MP31 NaN 兼容补丁 |

## 3. Runtime/Driver API 报告的设备属性

| 属性 | API 返回值 | 应如何理解 |
|---|---:|---|
| MP 数量 | 60 MP | 本机 device 0 的 runtime 属性 |
| warp size | 32 threads | 本机 runtime 属性；不是从 CUDA 类推 |
| shared memory / MP | 196608 B（192 KiB） | 设备资源属性 |
| 32-bit registers / MP | 131072 | 设备资源属性 |
| max blocks / MP | 256 CTA | 架构接口上限，不是某个 kernel 的实际 resident CTA |
| max threads / block | 1024 | resource probe 返回值 |
| L2 cache size | 62914560 B（60 MiB） | runtime 报告容量，不等于任意 kernel 可使用的有效容量 |
| max persisting L2 | 52428800 B（50 MiB） | persisting cache 接口上限，不是物理总 L2 |

## 4. Compute：通过正式门禁的条件实测

下表 latency 是三进程 graph-replay 中位数；GEMM 的 TOPS 按 `2*M*N*K/time` 计算，其他算子的
Gitems/s 按逻辑元素数除以时间计算。GEMM 均为 4 warp、default policy。

### 4.1 GEMM

| dtype | shape M×N×K | tile / stage | latency (µs) | 有效吞吐 (TOPS) | 跨进程跨度 |
|---|---|---|---:|---:|---:|
| FP16 | 1024×1024×1024 | 64×64×64 / 1 | 23.424 | 91.680 | 0.505% |
| FP16 | 512×512×4096 | 64×64×64 / 1 | 43.952 | 48.860 | 0.174% |
| FP16 | 128×4096×1024 | 64×64×64 / 1 | 17.840 | 60.188 | 2.732% |
| FP16 | 4096×128×1024 | 64×64×64 / 1 | 17.089 | 62.834 | 0.982% |
| FP16 | 1024×1024×1024 | 128×128×64 / 3 | 32.694 | 65.685 | 0.292% |
| BF16 | 1024×1024×1024 | 64×64×64 / 1 | 23.252 | 92.359 | 0.305% |
| BF16 | 512×512×4096 | 64×64×64 / 1 | 43.886 | 48.933 | 0.153% |
| INT8→INT32 | 512×512×512 | 128×128×64 / 1 | 18.824 | 14.260 | 0.719% |
| INT8→INT32 | 1024×1024×1024 | 128×128×64 / 1 | 30.699 | 69.952 | 0.111% |

这里的 INT8 TOPS 也按一次乘加计 2 ops，只是条件有效率，不代表芯片 INT8 理论峰值。

### 4.2 Vector、SFU 与 Reduce

| 算子 | 条件 | latency (µs) | 有效处理率 (Gitems/s) | 跨进程跨度 |
|---|---|---:|---:|---:|
| FP32 vector add | N=16,777,216；256 threads；vector=8 | 144.225 | 116.326 | 0.205% |
| FP32 exp | N=16,777,216；256 threads；vector=8 | 100.815 | 166.416 | 0.365% |
| FP32 rsqrt | N=16,777,216；256 threads；vector=8 | 113.974 | 147.203 | 0.209% |
| FP32 reduce-sum | M=4096，K=256；128 threads | 45.611 | 22.990 | 0.535% |
| FP32 reduce-sum | M=4096，K=1024；128 threads | 64.282 | 65.248 | 0.278% |

Reduce 的 Gitems/s 表示输入元素处理率，不是 FLOP/s。

## 5. FP16 GEMM pipeline stage 实测

这是另一次独立的 P4 正式 stage sweep：shape 1024³、tile 64³、4 warp、256 generated threads。
它和 P2 表中相同 source 的运行批次不同，因此保留各自实测值，不取更有利的一个替换另一个。

| stage | latency (µs) | 有效 TFLOP/s | dynamic shared | registers/thread | API active CTA/MP |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.981 | 93.444 | 16 KiB | 45 | 10 |
| 2 | 21.670 | 99.098 | 32 KiB | 45 | 6 |
| 3 | 26.104 | 82.266 | 48 KiB | 45 | 4 |
| 4 | 26.909 | 79.805 | 64 KiB | 45 | 3 |

本 schedule 下 stage 2 最快；不能据此断言 S5000 的所有 GEMM 都应使用 stage 2。资源项来自保留
`.mu` 和 mcc 命令重编译出的 MUBIN，source SHA 与正式 run 匹配，但 binary 不是逐字节保留的原执行文件。

## 6. Global memory：条件逻辑带宽

### 6.1 大工作集路径

| 路径 | benchmark 条件 | latency (µs) | 逻辑带宽 (GB/s) | 解释 |
|---|---|---:|---:|---|
| copy | 每个输入/输出 512 MiB；逻辑 payload 1 GiB | 808.072 | 1328.770 | 大工作集读+写 |
| fill | 256 MiB 输出 | 191.315 | 1403.105 | 写入逻辑带宽 |
| triad A+B→C | 每个数组 256 MiB；逻辑 payload 768 MiB | 574.280 | 1402.289 | 两读一写混合 |
| read-sum | 512 MiB 输入；64 loads/output | 599.183 | 910.005 | read-dominated 逻辑下界 |

这些值是特定 TileLang kernel 的逻辑有效带宽，不能直接写成“S5000 DDR 理论带宽”。尤其 fill 的
write allocation、cache、合并与实际总线事务没有硬件 counter 证明。

### 6.2 Warm stride 扫描

逻辑输出均为 4,194,304 个 FP32 元素，逻辑 payload 32 MiB；输入占用随 stride 增加。

| stride | latency (µs) | 逻辑带宽 (GB/s) | 相对 stride-1 |
|---:|---:|---:|---:|
| 1 | 13.519 | 2481.934 | 100.0% |
| 2 | 20.867 | 1608.043 | 64.8% |
| 4 | 38.650 | 868.159 | 35.0% |
| 8 | 103.746 | 323.427 | 13.0% |
| 16 | 197.051 | 170.283 | 6.9% |

相同 64 MiB copy 中，offset 0 为 102.083 µs / 1314.796 GB/s；偏移 1 个 FP32 元素后为
125.846 µs / 1066.527 GB/s。该结果证明此实现对该 misalignment 条件敏感，但不能反推出 cache line
或内存控制器映射。

## 7. Cache 工作集与 pointer chase

### 7.1 Fixed-grid copy 工作集扫描

逻辑 payload 包含一次读和一次写，因此为 case 工作集标签的 2 倍。

| case 工作集 | warm latency (µs) | post-256MiB-flush latency (µs) | warm 逻辑 GB/s | 状态 |
|---:|---:|---:|---:|---|
| 8 MiB | 7.683 | 7.954 | 2183.629 | 可用 |
| 16 MiB | 13.485 | 14.159 | 2488.348 | 可用 |
| 32 MiB | 94.840 | 130.340 | 不采用 | 三进程严重不稳定 |
| 64 MiB | 102.929 | 103.603 | 1303.981 | 可用 |
| 128 MiB | 209.103 | 210.443 | 1283.750 | 可用 |
| 256 MiB | 418.038 | 420.899 | 1284.264 | 可用 |
| 512 MiB | 849.564 | 858.192 | 1263.874 | 可用 |

8/16 MiB 的逻辑带宽高于大工作集路径，符合 cache/reuse 可能参与的表现，但现有曲线无法给出可重复、
单一的 cache cliff；32 MiB 点又因三进程不稳定被拒绝。因此 **实测有效 L2 工作集边界仍为 unknown**。

### 7.2 Native dependent pointer chase

| 工作集 | 每次依赖 load (ns) |
|---:|---:|
| 4 KiB | 237.879 |
| 16 KiB | 252.300 |
| 64 KiB | 246.429 |
| 256 KiB | 254.546 |
| 1 MiB | 254.561 |
| 4 MiB | 253.132 |
| 16 MiB | 252.540 |
| 64 MiB | 252.388 |

这组数据三进程非常稳定，但没有随工作集出现可解释的阶梯，因此不能拿它宣称测出了 L1/L2/DDR
访问延迟。它更可能受当前访问构造、编译和计时路径的共同影响，结论仅为“未识别出容量拐点”。

## 8. Shared-memory 条件测试

FP32、16-element shared tile、4096 CTA、128 threads、128 repeated reads 的顺序访问，在扣除
`repeats=0` baseline 后得到 **44686.268 GB/s** 条件有效带宽。原始未扣 baseline 的
`shared_seq_r128` 是 121.203 µs / 35436.231 GB/s。

stride-risk 2/4/8 的原始逻辑带宽分别为 37629.765、37762.727、39322.255 GB/s；它们没有表现出
按 stride 单调恶化。stride 15 的三进程跨度 21.73%，被拒绝。上述数据只能比较这一 synthetic
pattern，不能推断 S5000 shared-memory bank 数、bank 宽或冲突映射。

## 9. Occupancy 与资源约束

### 9.1 Native 低寄存器 kernel

资源为 4 registers/thread、12 B static shared；API 返回：

| threads/CTA | 64 | 128 | 256 | 384 | 512 |
|---:|---:|---:|---:|---:|---:|
| active CTA/MP | 48 | 24 | 12 | 8 | 6 |

增加 static shared 后，256-thread kernel 的 active CTA/MP 为：8 KiB→12、16 KiB→11、32 KiB→5。
实际函数 static shared 比 requested 值多 12 B。

### 9.2 受控寄存器压力

所有点 local memory=0、无 spill；下表为 API exact-kernel 查询：

| live values | registers/thread | active CTA/MP：64 / 128 / 256 / 512 threads |
|---:|---:|---|
| 4 | 6 | 48 / 24 / 12 / 6 |
| 16 | 24 | 48 / 24 / 12 / 6 |
| 32 | 53 | 34 / 17 / 8 / 4 |
| 64 | 115 | 16 / 8 / 4 / 2 |

### 9.3 TileLang fixed-grid copy

重编译 MUBIN 均为 56 registers/thread、零 static/dynamic shared：

| threads/CTA | 64 | 128 | 256 | 384 | 512 |
|---:|---:|---:|---:|---:|---:|
| API active CTA/MP | 34 | 17 | 8 | 5 | 4 |
| 实测逻辑 GB/s | 1180.914 | 1193.569 | 1203.227 | 1194.406 | 1201.459 |

这些 occupancy 数值都绑定对应 binary/source，不能把其中任意一个写成“S5000 通用 active CTA/MP”。

## 10. Wave 与尾波

### 10.1 Native compute kernel

256 threads、12 active CTA/MP、60 MP，对应 API 条件容量 720 CTA。实测 720→721 CTA 从
134.000 增至 137.440 µs（+2.567%）。这是一个 native exact-kernel 条件点，不是统一尾波百分比。

### 10.2 TileLang copy kernel

256 threads、8 active CTA/MP，对应 480 CTA 条件容量：480→481 为 +0.523%，960→961 为
−1.118%。两个边界方向不一致，因此 copy tail 保持 `empirical_inconclusive`，不拟合固定百分比。

### 10.3 TileLang compute-dominant kernel

256 threads、2048 次依赖 FP32 recurrence、12 active CTA/MP、720 CTA 条件容量：

| 边界 | capacity latency (µs) | +1 CTA latency (µs) | 增量 |
|---|---:|---:|---:|
| 720→721 | 33.997 | 34.389 | +0.392 µs / +1.152% |
| 1440→1441 | 60.723 | 63.513 | +2.790 µs / +4.594% |
| 2160→2161 | 89.024 | 90.223 | +1.199 µs / +1.347% |

三个方向一致，但差异较大；目前只允许记录为该 kernel family 的 **0.392–2.790 µs 条件区间**，
中位 1.199 µs。它不能外推给 GEMM、copy 或其他 kernel。

## 11. P6-v3：真实 Hold-out 实测与模型误差

下面“实测”列仍是 S5000 三进程真实数据；“预测”和 APE 只用于验证 P5 v2，不是硬件参数。

| case | P5 v2 事前预测 (µs) | S5000 实测 (µs) | APE | 三进程跨度 |
|---|---:|---:|---:|---:|
| FP16 GEMM 896×1152×1024 | 21.343 | 24.770 | 13.837% | 0.550% |
| FP32 reduce 6144×512 | 72.650 | 75.316 | 3.540% | 0.157% |
| FP32 fused exp+add，24M | 214.512 | 214.439 | 0.034% | 0.114% |
| warm stride-6 copy，8M | 161.258 | 161.015 | 0.151% | 0.174% |

四点 MAPE（mean APE）为 4.3907%，MdAPE 为 1.8456%，最大 APE 为 13.8375%。每个 family 只有
一个独立 hold-out，因此这些数字不能代表任意新算子的全局预测精度，也没有验证 schedule ranking。

## 12. 已真实运行但不得进入参数拟合的数据

| 数据 | 实际观察 | 不采用原因 |
|---|---|---|
| FP16 GEMM 256³ | 8.101 µs；4.142 TOPS | 短 kernel 固定开销占比高；正式状态非 3/3 measured |
| FP16 GEMM 512³ | 10.985 µs；24.436 TOPS | 同上 |
| BF16 GEMM 512³ | 11.094 µs；24.196 TOPS | 仅 1/3 measured |
| FP32 vector/exp/rsqrt 1M | 8.419 / 6.842 / 11.448 µs | 短 kernel 计时门禁未通过 |
| global copy 1 MiB | 2.946 µs | 一次进程严重离群，跨度约 1499.7% |
| fixed-grid copy 32 MiB | warm 94.840 µs | 三进程严重不稳定，跨度约 141.5% |
| P7 FP16 GEMM 1536×1024×1536 | 单次正确；成功诊断约 36–45 µs | 高频重复协议仅 1/3 稳定，含 runtime/profiler/host corruption 症状 |

这些值是真实观察，不等于可用于模型的稳定硬件参数。尤其 P7 成功 trial 的约 129.57 TOPS 存在
失败选择偏差，绝不能写入 S5000 吞吐参数表。

## 13. 当前仍为 unknown

- 基于微基准识别的有效 L2/cache 工作集边界；
- 任意 TileLang kernel 的统一 resident CTA；
- scheduler 到 MP 的实际 CTA 映射；
- 跨 kernel family 的统一 tail-wave 常数；
- GEMM prologue、steady、tail、epilogue 的独立时间分解；
- 任意新 shape/tile/stage/warp 的性能；
- attention、softmax、layer norm 等未校准 family；
- 多卡 collective、NoC 和通信计算 overlap；
- P7 大 GEMM 高频重复异常的唯一根因。

## 14. 数据来源和不可变性

主要机器可读来源：

- P2 compute：`results/processed/p2_compute/aggregate.json`
- P3 memory：`results/processed/p3_memory/aggregate_*.csv`、`s5000_memory_parameters_provisional.json`
- P4 occupancy/wave：`results/processed/p4_occupancy_wave/aggregate_*.csv`、参数 JSON
- P6-v3：`results/processed/p6_v3_holdout/holdout_results.json`
- 环境与 driver 属性：`logs/env/`、`logs/run/p3_musa_device_properties_stdout.log`

已知追溯限制：三次正式 P2 raw 保存了当时 harness SHA `107c5ede...`，但当前工作区没有对应 SHA
的 Python 源码快照；配置、生成 `.mu`、source SHA、样本和日志仍在。故 P2 数值证据可复核，
但不能声称当时的 Python harness 可逐字节复原。该限制不影响本文从既有 aggregate 重算公式，
但后续正式 run 必须在 raw 目录同步保存 harness 快照。

冻结模型未被本文修改：P5 v1 SHA
`9ef664f1e27f78bcd99cceb458750cf02319d6ae68de8573438cec69a17aeec0`；P5 v2 SHA
`b94296eb50b904b0e34b80d79535887ad7932b7b08f57fe736c609eb48caa9be`。本文是已有证据的人工可读
快照，不改变 raw、processed、模型或适用域。

## 15. 审核结论

本文已按以下规则复核：

1. compute TOPS/Gitems/s 从 shape 与 latency 重新计算；
2. memory GB/s 从聚合文件读取并核对 logical payload / latency；
3. MiB/KiB 使用二进制换算，GB/s 使用聚合文件的十进制单位；
4. P6 APE、MAPE、MdAPE 从预测与实测重新计算；
5. P4 wave 增量由相邻 CTA latency 重新计算；
6. 所有标为有效的正式 compute 条件均通过 correctness，且 `p2_parameter_eligible=true`；
7. API 属性、条件实测、预测、失败诊断和 unknown 没有混写；
8. 未修改任何历史 raw、processed 或冻结模型。

审核结果：表中数字与当前机器可读聚合一致；所有外推边界和不确定项均显式保留。
