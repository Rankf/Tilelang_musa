# P3 单卡 memory/cache/shared-memory 校准审核

状态：completed with explicit unknown cache parameters  
日期：2026-07-29  
对象：1× MTT S5000，MUSA 4.3.5，TileLang-MUSA `4d666968a005846314e9a84d8ff121c5dbffa9ee`

## 结论

P3 建立了可复跑的 global memory、stride、对齐、固定-grid 工作集、shared-memory 与 native dependent pointer-chase 数据。所有正式 TileLang case 遵循编译 → 数值正确性 → 计时，使用预分配 buffer、MUSAGraph 批量回放及 profiler self-device cross-check；每个正式集合独立执行三次。

可以进入 P4。P3 后补的 MUSA Runtime API 直接查询报告 L2 容量为 62,914,560 B（60 MiB）。但基于微基准的 cache working-set boundary 仍为 `unknown`：容量字段与特定访问模式下何时命中/驱逐、可供某个 kernel 使用的有效容量不是同一参数，后者没有足够稳定的实测证据。

## 计时与 payload 口径

- copy：每 FP32 元素按 1 读 + 1 写，即 8 B logical payload。
- fill：按 1 写，即 4 B logical payload。
- triad：A+B→C，按 2 读 + 1 写，即 12 B logical payload。
- read-sum：64 个输入读对应 1 个输出写；报告读和输出的逻辑 payload。

这些数值不是 cache line、write-allocate 或其它物理 DDR transaction 的测量，不能作为公开硬件规格。Graph 用于去除 Python launch starvation；profiler self-device time 是独立审核口径。

## global-memory 曲线（跨三进程中位数）

| 路径 | 工作集 | 逻辑有效带宽 | 结论 |
|---|---:|---:|---|
| copy（读写） | 64 / 256 / 512 MiB payload | 1319.7 / 1333.3 / 1328.8 GB/s | 稳定的 streaming copy 条件率 |
| fill（写） | 256 MiB | 1403.1 GB/s | 稳定的写路径条件率 |
| triad（两读一写） | 256 MiB/operand | 1402.3 GB/s | 稳定的混合路径条件率 |
| read-sum（读主导） | 64 / 256 / 512 MiB input | 817.9 / 891.5 / 910.0 GB/s | 读主导、但包含 64 次 FP32 加法，故为条件下界 |
| stride copy | stride 1 / 2 / 4 / 8 / 16 | 2481.9 / 1608.0 / 868.2 / 323.4 / 170.5 GB/s | 访问间隔导致持续退化 |
| 64 MiB copy pointer offset | 0 / 1 FP32 element | 1313.4 / 1065.5 GB/s | 该特定未对齐偏移有约 19% 退化 |

完整 CSV：`results/processed/p3_memory/aggregate_global_initial.csv`、`aggregate_read_initial.csv`。

## cache 工作集与延迟证据

固定 grid=1024 CTA 的 copy 工作集扫描覆盖 8–512 MiB。8/16 MiB 为约 2.18/2.49 TB/s logical payload；64 MiB 以上约 1.26–1.30 TB/s。看似在 16–64 MiB 存在变化，但 32 MiB 三进程延迟为 24.98、94.84、159.16 µs，未通过稳定性门禁。因此它不能作为 cache 拐点或拟合参数。

native MUSA pointer-chase（单线程、每 load 地址依赖前一 load，随机排列环）在 4 KiB–64 MiB 的跨进程中位数为约 238–255 ns/load，没有出现可重复 latency cliff。256 MiB/1 GiB flush 后的 copy/read-sum 也未给出独立、稳定的 cache cliff。

因此本阶段证据等级为：有效带宽曲线 L1；MUSA Runtime API 报告的 L2 容量 L1（驱动/SDK 查询，60 MiB）；基于微基准的有效工作集边界 L0/unknown。不得由容量字段推断 cache bank、cache 映射，或将 60 MiB 当作任意 kernel 的可用 L2 容量。

## P3 后补：SDK/驱动报告的 L2 容量

2026-07-29，使用与 P0 相同的 `.mu`/`mcc` 路径调用 `musaGetDeviceProperties`，返回：`l2CacheSize=62,914,560 B (60 MiB)`；同时返回 `persistingL2CacheMaxSize=52,428,800 B (50 MiB)`。MUSA 4.3.5 头文件 `musa_runtime_api.h` 对前者的定义是 “L2 cache size in bytes”，对后者的定义是 “L2 cache's maximum persisting lines size in bytes”。

这是一条本卡、当前驱动的 SDK 查询证据，不是通过 P3 流式带宽或 pointer-chase 反推的结果；`persistingL2CacheMaxSize` 也不等同于物理总 L2 容量。属性查询程序、构建日志、运行日志和 SHA-256 分别为：`benchmarks/p3_musa_device_properties.mu`、`logs/build/p3_musa_device_properties_mcc.log`、`logs/run/p3_musa_device_properties_stdout.log`、`logs/build/p3_musa_device_properties_sha256.txt`。SDK 样例的 C++ 编译兼容性失败日志也保留为 `logs/build/p3_musa_sdk_device_query_mcc_retry.log`；最终数值未依赖该失败样例。

## shared-memory 证据与限制

shared kernel 采用 global seed load、重复 shared read、最终 global store，并用 repeat=0 基线相减。顺序 128 次重复访问的增量条件率约 44.7 TB/s；本实验中的 stride 2/4/8 risk pattern 为约 48.2/48.4/51.0 TB/s，未出现可证明的冲突退化。

该结果仅是该 TileLang kernel 的 `conditional_synthetic` 指标：编译器在 generated path 报 parallel-loop data-race warning，尽管所有三次输出均在预定义 FP32 repeated-accumulation 容差内正确。因此不能由这些数据判断 shared bank 数、bank width 或冲突规则。stride15 case 跨运行不稳定，已排除。

## 原始数据、脚本与参数表

- global/stride/alignment：`configs/p3_memory_global_initial.json`，`scripts/p3_run_global_memory.sh`
- fixed-grid cache：`configs/p3_memory_fixed_grid_cache.json`
- read dominated：`configs/p3_memory_read_initial.json`
- shared：`configs/p3_shared_memory_initial.json`
- native pointer chase：`benchmarks/p3_musa_pointer_chase.mu`，`scripts/p3_run_pointer_chase.sh`
- 汇总与可追溯参数：`results/processed/p3_memory/aggregate_*.csv`、`s5000_memory_parameters_provisional.csv/json`

shared 的两次初始 builder compile failure、一次严格容差下的数值失败，以及 read-sum 的 schema failure 均保留于 `results/raw/p3_global_memory/`；它们不是正式参数数据。P3 runner 随后修复，并重做三个独立成功进程。

## P4 进入条件

P4 应使用本阶段的 large-working-set 条件有效带宽，不应使用短/不稳定 32 MiB fixed-grid 点，也不应填入虚构的 L2 容量。下一步扫描 block/线程、SMEM、pipeline stage、小/满/尾 grid，并优先寻找 MUSA occupancy API 或 mcc 资源证据。
