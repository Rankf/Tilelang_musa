# Runtime Tile Trace 低扰动优化实施方案（审核稿）

状态：**v0.4 已实施 V2/V3/V4/V5、compiler-exact provenance、周期抽样、原生 C dispatch、MUSART trace pool 与 P6.9.2 审计，并在 S5000 验证**  
版本：runtime-trace-overhead-optimization-v0.4  
日期：2026-07-30  
工具根目录：`/root/musa-perf-collector`  
目标设备：MTT S5000，MUSA `mp_31`

当前完整交付状态、保守开销口径和最新 P6.9.2 S5000 证据见：
[`00_current_implementation_and_acceptance_status.md`](./00_current_implementation_and_acceptance_status.md)。

本文承接 `03_runtime_tile_collection_plan.md`，优化已经跑通的 GEMM+reduce_sum 深度 trace 原型；
相同受控 adapter 已在 GEMM+reduce_max、cumsum、源循环迭代 copy 和显式多 block copy 上完成 S5000
验证。compiler exact 自动插桩已用于这些审核 adapter；任意 workload、动态 grid 和全 grid 自动插桩
仍不在本版本能力声明内。

## 实施审计（2026-07-29）

已在 MTT S5000 上完成实际编译、正确性和运行采集。实现位于
`src/musa_perf_collector/examples/runtime_gemm_reduce.py`、
`collectors/runtime/trace_buffer.py` 和 `runtime.py`：trace 是 16-byte 的预分配 MUSA 输入
buffer，业务输出 ABI 不变；生成 MUSA C 只有两处 `clock64()`，并在 kernel 尾部将局部
timestamps 写入 `Trace[0:2]`。计时 callable 不包含 buffer 分配、`.cpu()`、`.tolist()` 或同步。

最新 GEMM+reduce_sum 正式产物为 `results/runtime/runtime_6c65dc699b8d092d`，并已在
GEMM+reduce_max（`runtime_f8567a1ad96a9872`）和 cumsum/scan（`runtime_b134492c7b7dc099`）复用相同
低扰动 ABI。各次采集均通过 baseline 与 instrumented correctness；异步回读状态机完成
`FREE → DEVICE_WRITING → DEVICE_READY → D2H_COPYING → HOST_READY → FREE`，并校验 device-writing
与 readback generation 一致。
后续诊断确认 `torch.empty(pin_memory=True)` 没有稳定选择 MUSA pinned allocator，而
`torch.empty(...).pin_memory()` 能得到 MUSA 可识别的 pinned host buffer。工具现使用 16 MiB
probe buffer 验证 producer Event、collector stream wait、non-blocking D2H、copy-done Event 和数据
正确性；只有全部通过才启用 `async_after_window`，否则仍自动降级为 `sync_after_window`。
正式 trace 仅复制 16 bytes，回读和 CPU 解析均在 Event 性能窗口之外。

最新正式运行的 ABBA callable 绝对增量约 0.592 μs（约 3.9%，受短 kernel 固定开销放大）；
10-launch profiler 窗口的 kernel-only 增量约 0.140 μs。两种百分比不可直接比较，
应以绝对增量结合重复窗口判断。V1→V4 的逐项
消融矩阵尚未重建，故不能将全部差异归因到某一项优化。

### 0.1 P6.7–P6.9.2 后续实现与当前口径（2026-07-30）

上述 3.9% 是早期受控 adapter 的 callable 消融结果，不是当前在线 collector 的保守端到端结论。后续已
实现 every=64 周期抽样、C vectorcall fast dispatch 和原生 MUSART trace pool：selected Event、pinned
host、collector stream wait、deferred D2H、slot/generation/reset 均下沉至 C 扩展。当前极短 kernel 的
every-64 端到端保守结果为 8.43%（4099 次真实 launch、65 次 instrumented、4034 次 fast path、260 条
Tile trace）；另一轮 7.08% 视为跨轮波动，不作为优化收益。当前没有达到 2%，不能用 3.9% 的旧口径替代。

P6.9.2 再增加独立 session audit：计划 hash、launch identity、fast-path 总账、slot/generation、uint64
interval、Event 一致性和 raw/enriched 字段都必须闭环。新版真实 S5000 结果
`online_99687_1785396484061581425_698e4790` 为应用正确性 pass、session complete、audit
`P6.9.2/pass/0 violation`、4 条 raw/enriched trace 对齐。完整细节见总文档与第 05 份在线方案。

## 1. 审核结论摘要

方案可执行，建议批准。当前约 43.8% 是 instrumented callable 总开销，不是四次 `clock64()`
的纯指令成本。复核测得：

| 口径 | baseline | instrumented | 增量 |
|---|---:|---:|---:|
| MUSA Event callable | 约 12.46 μs | 约 17.58 μs | 约 5.12 μs / 41% |
| profiler kernel activity | 约 6.47 μs | 约 7.09 μs | 约 0.62 μs / 9.6% |

ABBA 正式产物中 callable 相对开销为 43.8%，各组波动显著小于差值，因此不是统计噪声。
主要问题是原型将 trace 声明为第二个 `T.empty` 输出，包装器每次 launch 都准备额外输出；同时
四个时间戳在四个边界立即写全局内存，`trace_reader()` 最后使用同步 `.cpu()` 读取。

优化采用以下顺序：

1. trace 改为预分配输入 buffer，移除每 launch 的 trace 输出分配；
2. baseline 和 instrumented 保持相同正常输出策略和调用参数生命周期；
3. 每次深度实验只测一个区间，只保留两个时间戳；
4. 时间戳先保存在寄存器/线程局部标量，kernel 尾部统一写 trace；
5. trace 读取移出测量区间，通过 Event 和独立 stream 异步复制到 pinned host buffer；
6. 使用单 buffer 窗口后回收作为默认，只有连续采集才启用双缓冲；
7. 分别报告 kernel-only、callable、异步搬运和整个 session 的开销。

## 2. 目标、非目标和不变量

### 2.1 目标

- 消除 instrumented 与 baseline 之间由额外输出 Tensor 分配造成的非 kernel 差异；
- 将被采样 launch 的边界记录从 4 个降为 2 个；
- 避免在每个 TileOp 边界进行全局内存写入；
- 不在每个 instrumented launch 后同步或读取 trace；
- 用可复现的消融实验量化每项优化，而不是只比较优化前后两个最终数字；
- 保持 baseline 与 instrumented 正确性、代码哈希、证据等级和结果目录隔离。

### 2.2 非目标

- 本阶段不实现任意 TileOp 的自动跨 pass 插桩；
- 不同时启用 MUPTI counter、PC sampling 和 Tile trace；
- 不将优化后的 instrumented 时间修正为 baseline；
- 不承诺所有短 kernel 都达到固定百分比，因为固定绝对开销会被不同基线放大；
- 不为了获得更低数字删除正确性、overflow 或 buffer 生命周期检查。

### 2.3 强制不变量

```text
baseline timing window:
  不包含编译、输入构造、正确性、trace 清理、D2H 或 CPU 解析

instrumented timing window:
  不包含 trace buffer 分配、host pinned buffer 分配、D2H 或 CPU 解析

trace buffer:
  不得在上一轮 D2H 完成前被下一轮覆盖

report:
  instrumented latency 永远不能写入 baseline_latency
```

## 3. 当前问题与优化映射

| 当前实现 | 影响 | 优化 |
|---|---|---|
| `trace = T.empty((4,), int64)` 并作为返回值 | 每次 callable 多一个输出分配和包装 | trace 改为预分配输入参数 |
| baseline 返回 1 个输出，instrumented 返回 2 个 | wrapper 路径不等价 | 两种变体都只返回正常输出 |
| 4 次 `clock64` | 固定指令和分支成本 | 每个实验只测一个区间，降为 2 次 |
| 每次时间戳后立即 `trace[i] = ...` | 边界处全局写、顺序约束 | 局部保存，尾部一次性写回 |
| `last_trace.cpu().tolist()` | 同步 D2H 和 CPU 解析 | Event 后异步复制，窗口结束后解析 |
| 只有 callable ABBA 开销 | 无法区分 kernel 和包装器 | 同时采集 Event callable 与 activity kernel-only |

## 4. 目标运行路径

### 4.1 默认单窗口路径

```text
session prepare（计时外）
  ├── 编译 baseline / instrumented variant
  ├── 预分配 inputs、正常 outputs（若 adapter 支持）
  ├── 预分配 device trace buffer
  └── 预分配 pinned host trace buffer

未插桩 baseline ABBA

选定 instrumented launch
  ├── selected lane 读取 begin clock64
  ├── 执行目标 TileOp 区间
  ├── selected lane 读取 end clock64
  └── kernel 尾部写 2 个 int64

业务 stream 记录 trace_ready_event
collector stream wait_event(trace_ready_event)
collector stream 异步 D2H 到 pinned host buffer
collector stream 记录 copy_done_event

采集窗口结束
  ├── 等待 copy_done_event
  ├── CPU 解析 trace
  └── 生成报告
```

默认在采样窗口结束后启动 D2H，避免复制流量与后续被测 kernel 重叠。只有明确需要在线消费 trace
时，才允许在采样窗口中使用独立 collector stream，并单独测量其对业务流的影响。

### 4.2 连续采集双缓冲路径

```text
device_trace[0]  ← window N 写入
device_trace[1]  → window N-1 异步回收

host_trace[0/1] 必须是 pinned memory
每个 slot 各有 trace_ready_event 和 copy_done_event
```

slot 状态机：

```text
FREE
  → DEVICE_WRITING
  → DEVICE_READY
  → D2H_COPYING
  → HOST_READY
  → FREE
```

任何非法迁移、generation ID 不匹配或复用前 `copy_done_event.query()==false` 都必须使 session
降为 `invalid`，不能继续覆盖。

## 5. P0：冻结测量契约和变体矩阵

先保留相同输入、stream、warmup、repeat、cache policy 和时钟环境，建立以下变体：

| 变体 | trace 参数 | 时间戳 | 写回 | 读取方式 | 目的 |
|---|---|---:|---|---|---|
| V0 baseline | 无 | 0 | 无 | 无 | 原始基线 |
| V1 current | 自动输出 | 4 | 每边界写 | 同步 | 复现 43.8% |
| V2 prealloc | 预分配输入 | 4 | 每边界写 | 窗口后同步 | 隔离输出分配收益 |
| V3 two-point | 预分配输入 | 2 | 每边界写 | 窗口后同步 | 隔离减少时间戳收益 |
| V4 tail-flush | 预分配输入 | 2 | 尾部统一写 | 窗口后同步 | 隔离全局写位置收益 |
| V5 async-read | 预分配输入 | 2 | 尾部统一写 | Event + 异步 D2H | 最终目标 |

所有变体必须使用独立 kernel hash。每次 ABBA 比较只改变一项，禁止 V1 直接与 V5 比较后把全部
收益归因给某一个优化。

## 6. P1：预分配 trace buffer 和输出路径对齐

### 6.1 Kernel ABI

instrumented 目标接口从：

```text
kernel(A, B) -> (out, trace)
```

改为：

```text
kernel(A, B, trace_device) -> out
```

`trace_device` 在 prepare 阶段创建一次，后续重复传入。baseline 仍为：

```text
kernel(A, B) -> out
```

两种变体都只返回正常业务输出。若 TileLang adapter 支持显式 out 参数，则进一步改成：

```text
baseline(A, B, out)
instrumented(A, B, out, trace_device)
```

本阶段不强行修改 TileLang 的正常输出 ABI；关键门槛是 baseline 与 instrumented 不再因 trace 多出
一个自动输出。报告保存 `normal_output_allocation_policy` 和 `trace_allocation_policy=preallocated`。

### 6.2 Buffer 内容和生命周期

trace device buffer 至少包含：

```text
generation_id
valid_record_count
overflow
begin_cycles
end_cycles
```

元数据可放独立 int64 header。计时前由 host 在独立准备阶段设置 generation 和 sentinel；不得在每次
instrumented launch 前清零整个大 buffer。只重置固定 header，且重置操作不进入 instrumented Event。

### 6.3 P1 验收

- 生成 kernel 只有一个正常返回输出；
- trace 出现在输入参数中，不出现在 `out_idx` 自动输出列表；
- timed `run()` 内没有 `torch.empty`、`.cpu()`、`.tolist()` 或显式 synchronize；
- baseline 与 instrumented 正常输出分配策略一致；
- correctness 通过，trace generation 和 sentinel 正确更新；
- V2 callable 开销必须显著低于 V1，否则先调查 wrapper/allocator，不进入 P2。

## 7. P2：单区间、两个时间戳

配置从多个边界改为明确区间：

```json
{
  "selected_interval": {
    "begin_op_id": "op_ffad8deb2bc0af05",
    "end_op_id": "op_9a3c3be09d912b62",
    "semantic_name": "gemm_to_reduce_boundary"
  },
  "timestamps_per_sample": 2
}
```

一次 instrumented variant 只接受一个 begin/end 区间。需要分析 copy、gemm、reduce 三段时，分别
编译或缓存三个独立 variant，在三个短实验中采集；不能在一次 kernel 中恢复到全量边界记录。

选择器仍限制：

- 一个 workload case；
- 一个 kernel；
- 一个或少量确定性 launch；
- 默认一个 block；
- block collective 默认 `threadIdx.x==0`；
- warp-specialized kernel 改为每个被选 warp/warpgroup 一个代表 lane。

P2 验收：生成 MUSA C 中目标路径只出现两次 `clock64()`；trace 只有 begin/end 两种记录；未选
block/lane 不写 trace；不同区间 variant 具有不同 instrumented hash 和 interval ID。

## 8. P3：寄存器保存和尾部统一写回

目标伪代码：

```cpp
uint64_t trace_begin;
uint64_t trace_end;

if (selected_block && selected_lane) {
  trace_begin = clock64();
}

target_interval();

if (selected_block && selected_lane) {
  trace_end = clock64();
}

// 不增加仅用于测量的 barrier

if (selected_block && selected_lane) {
  trace_device[slot + 0] = trace_begin;
  trace_device[slot + 1] = trace_end;
}
```

TileLang 中优先使用线程局部标量或小型 local buffer。lowering 后必须检查最终 MUSA C/编译资源：

- begin/end 没有在边界处提前写 global；
- 没有因为 local buffer 产生 local-memory spill；
- 没有新增 barrier、atomic 或全线程写；
- 只有 selected lane 执行两次最终 global store；
- 记录点没有被编译器消除、合并或移动出语义边界。

如果无法证明时间戳仍在寄存器中，应保留 V3 的边界写版本并标记 `tail_flush_unavailable`，不能把
疑似 spill 的 V4 宣称为优化成功。

## 9. P4：窗口后异步回收

### 9.1 能力探测

新增 probe：

- `torch.empty(..., pin_memory=True)` 是否可用于 MUSA D2H；
- MUSA Tensor 到 pinned CPU Tensor 的 `copy_(..., non_blocking=True)` 是否真正异步；
- custom collector stream、`wait_event`、Event query 和 elapsed 行为；
- D2H 错误和设备异常是否能被 copy_done_event 正确暴露。

如果 pinned/async D2H 不可用，允许窗口结束后同步读取作为降级路径。该读取仍不得进入 kernel 或
callable timing，但 session 报告写 `trace_readback=sync_after_window`。

### 9.2 Event 顺序

```text
work_stream:
  instrumented_kernel(..., device_trace)
  trace_ready_event.record(work_stream)

collector_stream:
  collector_stream.wait_event(trace_ready_event)
  host_trace.copy_(device_trace, non_blocking=True)
  copy_done_event.record(collector_stream)

CPU:
  继续执行不依赖 trace 的工作
  在解析前等待 copy_done_event
```

禁止：

- instrumented launch 后立即 `torch.musa.synchronize()`；
- Event timing 内调用 `.cpu()`；
- 未检查 copy_done 就复用 device/host slot；
- 在 benchmark 测量窗口内启动大规模 D2H，却不测量其干扰。

### 9.3 异步回收证据

产物记录：

```text
trace_ready_event_id
copy_done_event_id
work_stream_id
collector_stream_id
d2h_bytes
d2h_enqueue_host_time_us
d2h_completion_time_us
readback_policy
generation_id
slot_id
```

D2H 时间和 session 影响单独报告，不并入 kernel-only 或 instrumented callable。

## 10. P5：抽样和整体扰动控制

### 10.1 默认策略

```text
baseline session: 只运行未插桩 kernel
deep session:     只运行少量 instrumented launch
```

不在同一 baseline 统计序列中混合 baseline 与 instrumented 时延。若需要在真实长流程中抽样，使用
确定性 launch selector，并把整个流程标为 `mixed_sampling_session`，其时延不能作为纯 baseline。

默认上限：

- 一个 case；
- 一个 interval；
- 一个 block；
- 一个代表 lane；
- 一个 instrumented launch；
- 两个时间戳；
- 40 bytes 左右的有效 trace payload，加固定 header；
- 采集窗口结束后一次 D2H。

### 10.2 整体影响

抽样只能降低整个 workload 的总扰动，不能消除被采样 launch 本身的插桩开销。因此报告同时输出：

```text
instrumented_launch_kernel_overhead
instrumented_launch_callable_overhead
mixed_session_overhead
sampling_fraction
instrumented_launch_count
```

禁止用 `单次插桩开销 × 抽样比例` 替代 mixed session 实测；缓存、频率和 variant 切换可能非线性。

## 11. 开销测量和验收门槛

### 11.1 两个主口径

1. `kernel_only_overhead`：来自 MUSA activity/MUPTI kernel duration；
2. `callable_overhead`：来自相同 batch Event 方法，包含包装和提交差异。

附加口径：

- `readback_overhead`：D2H enqueue/completion 和 session 增量；
- `mixed_session_overhead`：真实抽样流程相对纯 baseline；
- `trace_parse_cpu_time`：窗口后 CPU 解析时间。

每个口径继续使用 ABBA：

```text
baseline A → variant B → variant B → baseline A
```

保存所有原始样本、median、p95、标准差和绝对/相对差值。差值小于分辨率或不确定性时标为
`overhead_not_resolved`，不输出虚假精度。

### 11.2 阶段门槛

以下为代表性 GEMM+reduce_sum 原型的工程验收目标，不是所有 kernel 的承诺：

| 门槛 | 验收要求 |
|---|---|
| P1 分配消除 | V2 callable 绝对增量明显低于 V1，且 timed path 零 trace 分配 |
| P2 两点记录 | 生成代码恰好 2 次 `clock64`，正确性和 trace 单调性通过 |
| P3 尾部写回 | 无 spill/新增 barrier；kernel-only 开销不高于 V3 |
| P4 异步回收 | D2H 不进入两个主时延窗口；无 buffer 覆盖或 generation 错配 |
| 最终 kernel-only | 目标相对开销不高于 5%，或绝对增量不高于 0.5 μs |
| 最终 callable | callable 绝对增量与 kernel-only 绝对增量之差不高于 0.5 μs，或差值不可分辨 |
| mixed session | 默认抽样策略实测相对开销目标不高于 2% |

若未达到最终目标，不得通过删除正确性/生命周期检查来“优化”。应依次减少 sampled block、采样 launch、
时间戳区间，或接受该 kernel 不适合在线插桩，转为离线分析/PC sampling 候选。

kernel-only 和 callable 的基线分母不同，因此不得直接要求两者“相对百分比相等”；验收比较绝对增量，
相对百分比只在各自口径内部报告。

## 12. 数据结构调整

`PreparedRuntimeCase` 拆分热路径和回收路径：

```text
run_timed()                 # 只提交 kernel，不分配/读取 trace
record_trace_ready()        # 在 work stream 记录 Event
enqueue_trace_readback()    # collector stream 异步 D2H
wait_trace_readback()       # 只在解析前等待
parse_trace()               # CPU-only
```

新增产物：

```text
deep/
  buffer_manifest.json
  interval_manifest.json
  readback.json
  trace.jsonl
overhead/
  v0_current.json
  v2_prealloc.json
  v3_two_point.json
  v4_tail_flush.json
  v5_async_read.json
  comparison.json
```

每个 overhead 文件包含 baseline/instrumented hash、变体唯一差异、kernel-only/callable/readback 口径、
采样范围、代码生成审核和 correctness 状态。

## 13. 实施模块

```text
src/musa_perf_collector/
  adapters/tilelang/
    runtime_workload.py          # 拆分 timed/readback 生命周期
    trace_variant.py             # interval-specific variant
  collectors/runtime/
    trace_buffer.py              # 单/双缓冲和状态机
    tile_trace.py                # Event、D2H、解析
    overhead.py                  # 多口径 ABBA 与消融比较
    capability.py                # pinned/async D2H probe
  examples/
    runtime_gemm_reduce.py       # V0～V5 原型
  exporters/
    runtime_report.py            # 分口径报告
```

## 14. 实施顺序

### P0：复现和测量契约

冻结 V0/V1 哈希和 43.8% 产物，新增 kernel-only activity 对照。  
退出条件：复现差值，报告同时显示 callable 与 kernel-only，不能只显示 43.8%。

### P1：预分配 trace

修改 ABI、prepare 和 timed run，移除 trace 自动输出。  
退出条件：正确性通过、timed path 无分配/读取、V2 消融结果通过。

### P2：两个时间戳

引入 interval config 和单区间 variant。  
退出条件：最终 MUSA C 恰好两个时间戳，选区和 lineage 一致。

### P3：尾部写回

局部保存 begin/end，尾部统一写。  
退出条件：源码/资源审核无 spill、barrier 和全线程写，V4 不劣于 V3。

### P4：异步回收

实现 pinned host buffer、collector stream、Event 和单/双 buffer 生命周期。  
退出条件：D2H 不进入时延窗口，故障注入不能覆盖未回收 slot。

### P5：抽样和完整验收

运行所有变体消融、完整测试、设备回归和报告审核。  
退出条件：满足第 11.2 节，或诚实输出未达目标与实际最优变体；只有通过 correctness 和生命周期门禁的
结果可标为 `runtime_measurement_v1 complete`。

## 15. 测试矩阵

### 15.1 无设备单元测试

- buffer 状态机所有合法和非法迁移；
- generation/slot/overflow；
- deterministic launch/block/interval selector；
- 预算在 variant 编译前生效；
- overhead 口径禁止混合；
- readback 不得出现在 timed callback；
- baseline/instrumented hash 和 output policy 隔离。

### 15.2 S5000 集成测试

- trace 作为输入 buffer 能被 kernel 正确写入；
- V0～V5 correctness；
- 两次 `clock64` 与 begin/end 单调；
- 未选 block/lane 不写；
- 单 buffer 窗口后回收；
- 双 buffer 连续窗口无覆盖；
- pinned async D2H 或明确的同步降级；
- work/collector stream Event 顺序；
- callable/kernel-only/readback/mixed 四种开销；
- 不同 interval variant 的 op_id 和代码哈希对应。

### 15.3 故障注入

- copy_done 前复用 slot；
- generation ID 错配；
- pinned memory 不可用；
- async copy 失败；
- trace overflow；
- local timestamp spill；
- 编译器移动/删除记录点；
- instrumented correctness 失败；
- requested interval 无 exact lineage。

## 16. 风险与控制

1. **预分配 trace 参数增加寄存器/ABI 成本**：保留 kernel-only 消融；即使无写也比较参数成本。
2. **局部时间戳发生 spill**：检查生成代码和编译资源，失败则回退 V3。
3. **无 barrier 时边界语义有限**：记录命名保持 `issue_boundary/region_interval`，不称独占硬件时间。
4. **异步 D2H 与业务争用**：默认窗口后复制；在线复制需单独 mixed session 审核。
5. **双缓冲覆盖**：强制状态机、generation ID 和 Event query。
6. **不同 block 时钟不可直接对齐**：默认只计算同 lane 的 begin/end 差值。
7. **动态频率**：保留 cycles；未经校准不换算精确 ns。
8. **variant 切换污染 cache**：baseline 独立测量，mixed session 另测，不用比例估算替代。
9. **抽样失去代表性**：保存选择算法、seed 和候选/选中集合；必要时多轮选择不同 block。
10. **为降开销牺牲证据完整性**：任何 correctness、overflow、生命周期或 lineage 失败均 invalid。

## 17. 方案审核记录

### 17.1 根因对应审核：通过

每项优化都对应已观测根因，并通过 V0～V5 一次只改变一项的消融设计验证，不会把包装器收益误归因
给 `clock64()` 或尾部写回。

### 17.2 通用性审核：通过，有限制

预分配 buffer、两点区间、尾部写回、Event 异步回收、抽样和分口径报告适用于多数 GPU kernel。
warp-specialized、多 block 不均衡和动态循环需要调整代表 lane、block 和 interval 策略，因此先在
GEMM+reduce 原型证明，再抽象为通用 TileOp 插桩接口。

### 17.3 异步读取审核：通过

trace 不需要实时传回 CPU。设备端必须在语义边界实时取得时间戳，但 D2H 和解析可以在窗口结束后
异步完成。方案用 trace_ready/copy_done Event 和单/双 buffer 状态机保证顺序，避免逐 launch 同步。

### 17.4 扰动和证据审核：通过

kernel-only、callable、readback 和 mixed session 四个口径独立；D2H 不进入主时延窗口；抽样只减少
整体扰动，不掩盖单次插桩开销；instrumented 数据继续与 baseline 分离。

### 17.5 前置能力审核：有降级路径

MUSA Event 和 custom stream 已确认可用。pinned host memory 与非阻塞 MUSA D2H 必须在 P4 capability
probe 中实测；若不可用，使用窗口后同步回收，仍能完成 P0～P3 并保持基线时延无污染。

### 17.6 审核决定

建议批准并严格按 P0→P5 实施。P1 是首要阻断项：在 trace 自动输出分配消除前，不继续扩展更多
TileOp。P4 异步回收不能先于 buffer 生命周期状态机。每一阶段只有在 correctness、代码生成审核、
消融开销和证据隔离同时通过后，才进入下一阶段。
