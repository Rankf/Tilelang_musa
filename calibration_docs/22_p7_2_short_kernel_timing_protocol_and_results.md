# P7.2：短 kernel graph-free Event 计时协议与结果

状态：部分通过；只冻结三个满足预注册门槛的 graph-free slope，其他 case 保持 pending。
日期：2026-07-29（UTC+8）

## 协议与范围

P7.1 已给出实测证据：MUSAGraph/Profiler 高重复路径存在间歇 runtime/host 异常。因此 P7.2 采用独立的 `graph_free_event_linear_v1`，不创建 MUSAGraph，也不进入 torch Profiler。每个 trial 是新进程，固定顺序为 compile → 单次同步 → correctness → 3 warmup → Event batch；每个 batch 都同步。

每个 unroll `1/2/4/8/16/32/64` 采 30 个 total-time 样本。在观察数据前已固定只用 `8/16/32/64` 的 batch 均值拟合：`total_us = intercept_us + unroll * slope_us_per_launch`。单 trial 要求 slope > 0、intercept >= 0、R² >= 0.95、mean relative residual <= 10%。一个 case 只有三个 trial 都通过，且 slope 跨进程 span 不超过 5%，才标为 `accepted_graph_free_event_slope`。

这不是 universal event-overhead 扣减模型；不能用来挽救被拒绝的 2–15 us kernel。

主 raw：`results/raw/p7_2_short_kernel_timing/20260729T072549889687244Z/`。

修正单写 raw：`results/raw/p7_2_short_kernel_timing_single_write_fix/20260729T073242262819328Z/`。

汇总：`results/processed/p7_2_short_kernel_timing_v1/summary.{json,csv}`。

## 正式结果

| case | slope 均值 (us/launch) | span | 状态 | 结论 |
|---|---:|---:|---|---|
| FP16 GEMM 512³ | 9.822 | 2.54% | accepted | 可作为此 graph-free protocol 的有效 slope 数据 |
| FP16 GEMM 1024³ | 21.761 | 0.38% | accepted | 中等时长边界对照，可用 |
| rsqrt 1M | 9.582 | 0.84% | accepted | 可用，但只适用于本 source/layout/shape 条件 |
| FP16 GEMM 256³ | 8.182 | 7.51% | pending | 三 trial 本身通过拟合，但跨进程不一致 |
| vector-add 1M | 7.751 | 7.10% | pending | 同上，不能以均值冒充稳定参数 |
| exp 1M | — | — | pending | 两个 trial 的 fitted intercept 为负，拒绝 |
| single-write（一 warp） | — | — | pending | 三个 trial 的 fitted intercept 均为负，最短 kernel 不能可靠分离固定开销 |

`single_write_1` 的三份初始 compile failure 仍保留在主 raw。原因是 `N=1` 不满足 32-thread 版本的整除前置条件；它在最终汇总中显式标为 `excluded_pre_revision_configuration_error`，不作为第二个正式 case，也没有被删除。修正后的 `single_write_one_warp` 使用 `N=32, threads=32, vector=1`，正确性通过，但仍因负截距被拒绝。

## 审核结论和后续影响

- 高 R² 不是充分条件：exp 与单写的 R² 均很高，仍因负 intercept 被拒绝。这避免了把 Event 分辨率/固定开销伪装成极短 kernel 性能。
- 当前可用于 P7.3 的是“graph-free、条件受限的测量路径”，不是所有短 kernel 的万能 timer。P7.3 的 development GEMM 必须单独遵守其三进程稳定性门槛；不以 256³ 的均值作为参数。
- graph/profiler device-time 交叉检查仍为 `deferred_runtime_path_unreliable`，不是 pass。只有 runtime owner 修复并且另建协议版本后，才可重新尝试。
- P5 v1/v2、P6-v3 没有被读取为拟合输入或修改；本阶段为 diagnostic，不产生 P8 profile 参数。

## 实现审核

新增的 config、harness、runner、finalizer 均通过 JSON/Python/shell 语法检查。finalizer 只接受 config 当前声明的 case；任何被修订配置遗留的 raw case 都会保留并显式 excluded，避免悄悄丢弃失败证据。
