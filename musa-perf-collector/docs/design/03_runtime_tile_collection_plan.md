# TileLang-MUSA 运行期真实性能与 Tile 细粒度采集实施方案（审核稿）

状态：**v1.3.4：受控 compiler-exact Tile trace、在线 JIT 自动适配、默认 every-64、原生 C dispatch 与 MUSART trace pool、累计预算、P6.9.4 落盘证据重放与多实例独立审计、稳定性及 stream/sync gate 已验证；原始 MUPTI counter 仍受库能力阻塞**  
版本：runtime-collection-v1.3.4  
日期：2026-07-30  
工具根目录：`/root/musa-perf-collector`  
目标设备：MTT S5000，MUSA `mp_31`

当前完整交付状态、统一复现命令和最新 P6.9.4 S5000 证据见：
[`00_current_implementation_and_acceptance_status.md`](./00_current_implementation_and_acceptance_status.md)。

本文承接 `01_offline_compiler_analysis_plan.md` 和
`02_tileop_coverage_execution_completeness_plan.md`。离线工具负责回答能够从编译产物证明的事实；
本方案只采集无法离线得到的真实运行行为，并强制把未插桩基线、profiler 观测和插桩后的
Tile trace 分开保存、分开定级、分开报告。

## 0. 实施记录（2026-07-29）

已实现并在 S5000 上验证：`runtime probe`、受控 workload adapter、未插桩 MUSA Event 批量时延、
正确性、launch/stream ID、前后快照和活跃窗口遥测、短窗口 `torch.profiler` MUSA activity、
多维深度选择与硬预算、ABBA 插桩开销对照，以及 GEMM+reduce_sum、GEMM+reduce_max 和 cumsum/scan
的单 block `clock64` trace。单 TileOp 模式输出 2 条成对边界记录；GEMM+reduce 多 TileOp 模式可选择
exact ordinal `[3,4]`，输出 GEMM 与 reduce 各自 begin/end 共 4 条记录。各模式均保存独立的
baseline/instrumented 代码哈希和 compiler exact lineage；正确性均已通过。trace 使用预分配 device
输入 buffer，避免额外输出 Tensor 分配；生成 MUSA C 的 `clock64()`/Trace 写入数量严格等于记录数，
且不引入测量专用 barrier。

已在当前 S5000 环境验证异步回读：显式 `Tensor.pin_memory()` 获得 MUSA 可识别 pinned host buffer；
16 MiB capability probe 依次验证 producer Event、collector stream `wait_event`、`non_blocking` D2H、
copy-done Event 和数据正确性。验证通过后，16-byte trace 使用独立 collector stream 异步 D2H，且
回读/解析不进入 baseline、instrumented Event 或 profiler 的性能窗口。验证失败时，工具明确降级为
`sync_after_window`，不会宣称异步能力可用。

当前不把 `torch.profiler` 汇总记录伪装成原始 MUPTI counter。当前 S5000 上的 `libmupti.so.1.0.5`
缺少 `muptiDeviceGetNumMetrics`、`muptiDeviceGetNumEventDomains`、`muptiEventGroupCreate` 和
`muptiEventGroupReadEvent`，也缺少 PC sampling 数据读取/配置函数。工具通过动态符号检查确认该事实：
若配置请求 raw metric 或 PC sampling，分别输出 `MUPTI_COUNTER_ADAPTER_UNAVAILABLE`、
`PC_SAMPLING_UNAVAILABLE`，session 为 `partial` 并保留 activity 结果。MUPTI 单独 event/metric adapter、
PC sampling 和面向任意 TileLang workload 的自动 adapter 接入仍是后续工作，不计入本版本
`complete` 的能力声明；已审核 workload 的跨 pass compiler-exact 插桩本身已经完成。

离线阶段现额外导出 `metadata/lineage.json` 的结构候选：例如单一 cumsum 可关联到
`S40_TILE_OP_LOWERED` 中唯一的 `tl::CumSum1D` 调用，单一 GEMM 可关联到 `T.tl_gemm`/`_gemm_sqmma`
结构锚点。这些记录明确标为 `structural_candidate`，即使唯一也不能作为 runtime 自动插桩的 exact
provenance；只有编译器将 source `op_id` 显式保留到 lowering 后，才可提升为 `compiler_exact_lineage`。

## 1. 审核结论摘要

方案可实施。当前 S5000 环境已经具备 MUSA Event、MUSA Stream、`torch.profiler` 的 MUSA
activity、MUPTI 1.0.5 和 `mthreads-gmi` 遥测基础，适合先实现轻量和标准模式。Tile 细粒度
观测也可实现，但需要新增 TileLang 编译插桩和稳定 TileOp lineage，只允许作为深度模式，
不得替代或修正未插桩基线。

审核批准以下核心边界：

1. 正常 benchmark 永远先运行未插桩基线；正确性检查、编译、warmup 和遥测启动时间均不计入时延。
2. 高开销采集必须支持按 case、kernel、launch、block、TileOp 和循环迭代抽样，并有时间、记录数和字节数硬上限。
3. 默认轻量模式不在 kernel 内插桩，不逐 launch 启停 profiler，不逐 tile 打印或 dump 中间张量。
4. 标准模式只在短采集窗口启用 activity 和经过白名单审核的少量 counter；需要多次 replay 的指标不得伪装成单次原始执行。
5. 深度模式只运行独立的 instrumented kernel；只记录选定 block 的 lane 0 时间戳和有限 TileOp 边界，不增加仅用于测量的 barrier。
6. 所有数据记录采集模式、选择器、采集开销、代码哈希、证据等级和是否插桩。
7. 如果采集能力、权限、计数器 pass 数、缓冲区或开销预算不满足，工具必须拒绝、降级或停止，不能静默扩大采集范围。

## 2. 目标与非目标

### 2.1 目标

1. 采集 MUSA Event 的未插桩 kernel 时延分布和正确性结果。
2. 记录实际设备、进程、stream、launch、同步、温度、功耗和频率环境。
3. 在短窗口内采集 kernel/memcpy/sync activity 和少量可用 MUPTI 汇总计数器。
4. 对已定位问题的代表性 kernel，按抽样规则采集 block/TileOp/pipeline 阶段的时间戳 trace。
5. 将运行期记录与离线 `kernel_id`、`op_id`、循环和 pipeline 元数据关联。
6. 量化采集器自身开销，并阻止插桩数据被误报为原始性能。

### 2.2 非目标

1. 不默认采集每个 launch、每个 block、每个 TileOp 或每次循环迭代。
2. 不打印 tile 级运行日志，不全量复制或保存中间 tensor。
3. 不承诺无扰动地获得每个 TileOp 独占的硬件执行时间；async copy、GEMM 和 pipeline 可能重叠。
4. 不把单次温度/功耗采样归因给微秒级 kernel 或单个 tile。
5. 不把 profiler replay、counter multiplex 或插桩后的时间反推成未经测量的基线时间。
6. 不使用 CUDA WGMMA/TCGEN 数据校准 MUSA TileOp。

## 3. 已确认环境与实施前探测

### 3.1 当前已确认能力

| 能力 | 当前环境 | 结论 |
|---|---|---|
| 设备 | MTT S5000，1 卡，`mp_31` | 可运行已验证 TileLang-MUSA kernel |
| Driver | `3.3.5-server` | 记录进每次 session manifest |
| MUSA SDK/MCC | `4.3.5` | 可编译运行期扩展和 instrumented kernel |
| MUSA Event | `torch.musa.Event(enable_timing=True)` | 可做默认时延基线 |
| Stream | current/default/custom stream、event wait/record | 可记录 stream/sync 行为 |
| Async trace D2H | 显式 MUSA pinned buffer、独立 collector stream、Event 完成和数据校验 | 已在 S5000 验证；仅 capability probe 通过后启用 |
| PyTorch profiler | `ProfilerActivity.MUSA` | 可做标准 activity 汇总 |
| MUPTI | 1.0.5，头文件与库存在；运行库仅导出部分 PC enable/disable 符号 | raw counter 与完整 PC sampling 当前不可用，动态符号 probe 后显式降级 |
| 遥测 | `mthreads-gmi --metrics-monitor` | 已实测 Temp/Power/Freq 周期采样 |

TileLang 当前 `do_bench` 已支持 MUSA Event 和 MUSA profiler activity，但默认会分配并清零
256 MB cache flush buffer。新工具必须显式记录 `cache_policy`，默认不得把 cache flush 混入
用户原始 benchmark 语义。

### 3.2 每台机器必须先执行的 capability probe

计划提供：

```text
musa-perf-collector runtime probe --device 0
```

探测只做短 smoke test，输出：

- 设备名、UUID、arch、driver、runtime、MCC、torch/torch_musa、TileLang 和 MUPTI 版本；
- Event 创建、record、elapsed、跨 stream wait 和 timer 分辨率测试；
- `torch.profiler` 的 MUSA kernel activity、stream ID 和同步记录可见性；
- MUPTI activity kind、event、metric、counter 权限、每个 metric 所需 pass 数；
- PC sampling 是否在当前 S5000/driver 上真正可用，而不只检查头文件；
- `mthreads-gmi` 的字段、最小可用采样间隔和解析格式；
- 能力状态 `available/unavailable/permission_denied/unsupported/error` 及原始错误。

未通过 probe 的能力不能出现在对应采集模式的 `enabled_collectors` 中。

异步 D2H 是深度采集的有状态 probe：它在 `runtime collect --mode deep` 的 prepare 阶段执行，而不放进
只读的 `runtime probe` 命令。原因是它会临时分配 16 MiB device/host buffer 并实际提交一次 D2H；其结果
写入 session 的 `deep/readback.json`，作为本次深度采集是否可启用 `async_after_window` 的证据。

## 4. 三种运行模式与基线隔离

### 4.1 `light`：默认轻量

默认包含：

- 未插桩 MUSA Event 批量计时；
- 独立于计时窗口的正确性检查；
- kernel/代码/输入配置哈希及编译期 launch 参数；
- 实际 device、stream ID 和测量窗口；
- 测量前、中、后的 Temp/Power/Freq 时间序列；
- warmup、repeat、cache policy、输入重用策略和统计量；
- 轻量模式自身的开销审核结果。

默认 Event 使用“批量包围”而不是给每次 launch 创建一对 Event：

```text
start_event
  launch × batch_size
end_event
elapsed / batch_size
```

需要分布时，只对确定性选择的少量 launch 使用逐 launch Event。两种口径分别标为
`batch_average` 和 `sampled_per_launch`，禁止混合。

### 4.2 `standard`：标准采集

在 light 的独立基线完成后，另开短窗口采集：

- kernel、memcpy、memset、stream 和 synchronization activity；
- runtime/driver correlation ID；
- 实际 grid/block/shared-memory/stream 等 launch 记录；
- 经过设备 probe 验证的少量 MUPTI event/metric；
- counter pass/replay/multiplex 信息和 dropped-record 计数。

标准模式不在每次 launch 启停 profiler。profiler 只包围连续的短窗口，并使用
`max_profiled_launches`、`max_profile_duration_ms` 和 `max_counter_passes` 限制范围。

默认 counter 策略：

- 只允许白名单 metric；
- 默认最多 4 个 metric；
- 默认只接受单 pass；
- 需要多 pass 时，只有显式 `allow_replay=true` 才执行；
- replay 结果标为 `profiler_replay_observed`，不标为 baseline；
- profiler 导致 kernel 序列化时必须记录 `kernel_serialized=true`。

### 4.3 `deep`：Tile 细粒度深度采集

深度模式使用独立编译的 instrumented kernel，只用于已定位的问题和少量代表性 case。
它由三类选择器共同限制：

1. host 侧只启动被选中的 case/kernel/launch；
2. device 侧只允许被选中的 block 写 trace；
3. 编译期只在被选中的 TileOp/stage/loop 边界插入时间戳。

深度模式默认不启用 PC sampling 和 TileOp 插桩的组合；两者同时启用需要显式配置并重新
测量叠加开销。

深度 trace 的回读策略为 capability-gated：默认请求 `async_after_window`，但只有异步 D2H probe
和 workload adapter 的预分配 host/device buffer 接口均通过时才实际启用。否则使用
`sync_after_window` 并产生 `ASYNC_D2H_UNAVAILABLE` 诊断。两种回读都必须在性能计时窗口结束后进行。

## 5. 强制抽样与预算系统

### 5.1 抽样维度

所有高开销 collector 必须接受以下选择器：

| 维度 | 选择方式 | 默认 |
|---|---|---|
| workload case | exact ID、配置哈希、确定性 hash 抽样 | 仅显式 case |
| kernel | `kernel_id`、exact name；regex 只能用于发现后冻结列表 | 仅显式 kernel |
| launch | 起点、数量、每 K 次、确定性概率抽样 | 在线 deep 默认每 64 次，另受最多 10 次预算限制 |
| stream | stream ID 白名单 | 当前运行 stream |
| block | 显式坐标、线性 ID、确定性 hash 抽样 | `(0,0,0)` |
| TileOp | `op_id`、family、pipeline stage | 最多 3 个 op |
| loop iteration | first/last/显式区间/每 K 次 | 每 op 最多 4 次 |
| PC sample | period 和最大样本数 | 默认关闭 |

概率抽样必须使用持久化 seed 和稳定哈希，报告保存“候选集合、选中集合、seed 和算法版本”，
保证可重现。不得使用运行时无记录的随机选择。

2026-07-30 实施状态：case、kernel、stream、launch 与静态 grid block 已接入执行编排。命名维度支持
exact `values`、`every + start` 和确定性 `probability`；launch/block 支持 exact `indices`、`every + start`
和确定性 `probability`，优先级为 exact → every → probability。概率算法为
`sha256_threshold_v1`，即对 `seed:candidate_id` 做 SHA-256 后与概率阈值比较。工具输出候选数量、
候选集合 hash、预算前/后的选中集合、seed、算法版本与截断原因。逻辑 launch 窗口中未选位置运行
未修改 baseline callable，只有选中位置运行 instrumented callable；每次选中 launch 均在 trace buffer
被覆盖前完成窗口后回读并写入 `launch_index`。block 候选按 x 最快的线性顺序从静态 grid 展开，
选中坐标直接传入编译器 `tl.provenance_trace_blocks`。动态 grid 仍 fail-closed。

在线自动插桩路径的周期参数为 `launch_selector: {"every": K, "start": S}`，`K` 必须为正整数，默认
`K=64,S=0`；所以默认命中 `0,64,128,...`，用户可显式改为 32、128、256 等。周期/exact 未命中 launch
使用 wrapper 本地单调 counter 后直接调用 baseline，不进入 session 锁、stream/device 查询、trace pool
poll 或逐 launch JSON；仅 instrumented、buffer unavailable 和错误等慢路径落盘，session 结束时聚合总数、
快路径数和原因。概率选择仍保留完整、stream-aware 路径，不能误称具有同样的快路径成本。

真实 S5000 验证结果为 `results/runtime/runtime_006acacdc3281376`：seed `20260730` 下，8 个候选
launch 以概率 0.25 选中 `[3,6]`，4 个 block 以概率 0.5 选中线性 ID `[0,3]`；输出 8 条带
`launch_index` 的 trace。两个 instrumented launch 均分别通过 correctness、固定 slot、compiler-exact
lineage、block 和 begin/end 校验，并完成独立 stream/Event 异步 D2H。instrumented 预算使用 `2/2`，
diagnostics 为空，session 状态为 `complete`。该结果证明选择器已控制真实执行，不能外推为任意 workload
已自动适配。

### 5.2 硬预算

每个采集 session 都必须设置：

```text
max_total_duration_s
max_profile_duration_ms
max_profiled_launches
max_instrumented_launches
max_sampled_blocks_per_launch
max_tileops_per_block
max_iterations_per_tileop
max_trace_records
max_trace_bytes
max_counter_passes
max_dropped_records
```

达到预算后的行为由 `on_budget_exceeded` 决定，只允许：

- `stop`：停止本 collector，保留已完成记录并标记 truncated；
- `degrade`：按固定降级链减少范围；
- `fail`：整个 session 失败，不输出貌似完整的数据。

禁止自动扩大缓冲区、增加 replay 或延长采集时间。

### 5.3 固定降级链

```text
deep: tileop+iteration trace
  → selected TileOp begin/end only
  → selected block kernel-span only
  → standard activity window
  → light Event baseline

standard: counters+activity
  → single-pass counters+activity
  → activity only
  → light Event baseline
```

每次降级必须生成诊断码，最终报告展示“请求模式”和“实际模式”。

## 6. Tile 插桩设计

### 6.1 稳定标识和编译变体

离线 `S00_INPUT_TILE_IR` 为每个动作型 TileOp 提供 `semantic_op_id`。新增编译 pass 在 lowering
前把该 ID 和 pipeline/loop 上下文写成稳定注解，并输出跨 pass lineage。无法证明 exact lineage
的 op 不允许自动插桩；用户显式允许启发式映射时，记录必须降为 `heuristic_mapping`。

当前实现的 structural candidate 只用于定位和人工审核，且 `eligible_for_exact_runtime_instrumentation=false`。
runtime 的 `require_exact_lineage=true` 默认值不会接受它；这保证了未修改 compiler provenance 的任意
workload 不会被错误自动插桩。

### 6.1 已实施的 compiler exact lineage（2026-07-29）

TileLang-MUSA 现加入 `tl.ProvenanceAnnotation` pass。它在 pipeline 展开前只给**顶层**
`Evaluate(TileOp Call)` 写入连续的 `tl_provenance_ordinal`；不会给 region、access 等内部 Call 编号。
`LowerTileOp` 将同一 ordinal 物化为包裹 lowered statement 的
`T.attr(0, "tl.provenance_ordinal", N)`，因此 source TileOp 与 lowered device IR 有可核验的一对多
关系（pipeline clone 时可有多个实例，但 ordinal 不变）。

离线工具仅当同时满足以下条件时才写出 `compiler_exact_lineage` 并允许后续自动插桩：

1. 源 TileOp 的 `ordinal` 完整且唯一；
2. S30 中的 Call annotation 覆盖全部源 ordinal；
3. S40 与 S60 都覆盖同一集合，且没有未知 ordinal。

否则输出 `compiler_provenance_incomplete` 并拒绝自动插桩；没有该编译器 pass 的旧产物仍仅为
`structural_candidates_only`。已在 CPU-only pipelined GEMM（5 个 TileOp）和 cumsum（3 个 TileOp）
上验证完整链路，且在 S5000 上回归确认既有 cumsum、GEMM+reduce_sum 和 GEMM+reduce_max 的 baseline、
instrumented 正确性与 trace 校验未受该 metadata pass 影响。

### 6.2 已实施的受控单/多 TileOp 成对边界 trace（2026-07-29）

在 `LowerTileOp` 后，`tl.ProvenanceTraceInjection` 可由显式 pass config 启用。单 TileOp 使用
`tl.provenance_trace_ordinal`；多 TileOp 使用严格递增且无重复的 ordinal 列表：

```text
tl.provenance_trace_enable=true
tl.provenance_trace_ordinal=<exact source ordinal>
tl.provenance_trace_buffer="Trace"
tl.provenance_trace_ordinals=[3,4]
```

它只接受已有的一维 `int64` 输入 buffer，容量必须至少为 `2 * selected_tileops`；不分配 trace Tensor，不增加 kernel 输出，
不插入 barrier，也不使用结构锚点猜测。pass 在选中 ordinal 的 lowered AttrStmt 前、后各写一次
`clock64()`，且仅由 `threadIdx.x == 0` 写入 `Trace[0]`/`Trace[1]`。选中 ordinal 缺失或重复、Trace ABI
不存在/类型不符、没有 `threadIdx.x` 时编译直接失败。

cumsum 及 GEMM+reduce_sum/max S5000 workloads 已移除所有手写时间戳和 Trace 写回，只保留预分配
`Trace` 输入 ABI；cumsum 生成的 MUSA C 审计为恰好 2 次 `clock64()`、2 次 Trace 写入，baseline 为 0 次
`clock64()`，二者均为 2 个已有同步点。GEMM+reduce 以 reduce 的 exact ordinal 为边界，两个点分别表示
GEMM→reduce 与 reduce→后续 copy 的程序边界，sum/max 正确性与 trace 验证均在 S5000 通过。因此该 capability
现在可标记为 `compiler_exact_lineage`，而不是原来的 `manual_exact_workload_contract`。多 TileOp 实测
`[3,4]` 输出 4 条严格单调记录，32-byte Trace 经独立 stream/Event 异步回读，正确性与 lineage 校验通过，
callable 插桩绝对开销约 `0.000638 ms`（该次运行约 `5.02%`）。当前约束为**受控少量 block、每个 source ordinal
每个 selected block 一对边界时间戳**；源循环迭代和多 block 固定槽位扩展见下节。任意 block 自动选择和全 grid
trace 尚未实施，不能宣称全量通用 trace。

### 6.3 已实施的离线 `op_id` 自动选择（2026-07-29）

runtime 深度配置可提供离线分析目录（或其中的 `metadata/lineage.json`）和显式 `trace_op_ids`。工具只接受
`compiler_exact_lineage` artifact，并按其中的 `source_op_id → source_ordinal` 映射生成编译器插桩配置；
不支持按 family 名称自动匹配，避免重复 GEMM/copy 时产生歧义。对 GEMM+reduce，runtime 还会比较离线与运行期的
`M/N/K/threads/reduction`；不一致直接拒绝。

已验证完整链路：先以 `build_gemm_reduce_prim_func` 生成离线 artifact，再以
`gemm_reduce_sum_auto_lineage_trace.json` 指向 artifact 并选择 GEMM/reduce 的两个 `op_id`。runtime 结果会在
`deep/instrumentation_manifest.json` 的 `selection_evidence` 保存 artifact 路径、离线 TileLang revision、输入 IR
hash、所选 `op_id` 与 ordinal，Trace lineage 同时保存相同的 `source_op_id`。

每个 kernel 至少有两个不可混淆的变体：

```text
baseline_kernel_hash       # 无插桩，基线和 correctness
instrumented_kernel_hash   # 带 trace ABI，只用于 deep
```

两种变体使用不同 artifact ID。报告不得用 instrumented 值覆盖 baseline 字段。

### 6.4 已实施的精确源循环迭代抽样（2026-07-30）

编译器的 provenance pass 现在为每个输入 Tile IR 的 lexical `For` 以 preorder 赋予
`tl_provenance_loop_ordinal`。该编号与离线 `execution_tree.json` 的
`loop_context[].loop_id = lexical_loop/<n>` 使用相同编号规则。深度配置可同时指定：

```text
tl.provenance_trace_loop_ordinal=<exact lexical source loop ordinal>
tl.provenance_trace_iteration=<zero-based relative source iteration>
```

两项必须同时存在；编译器只接受非 thread-binding 的 source serial loop。它在 selected TileOp 的两次
`clock64()` 写入上附加 `threadIdx.x == 0 && loop_var == loop_min + iteration` 条件；TileOp 本体仍执行，
未选迭代不写 Trace。没有携带 source-loop provenance 的编译后内部循环不参与匹配；选择的 loop 不包围
选择的 TileOp、迭代超出可知 extent、或 selector 不完整时均 fail-closed。

离线 exact-lineage artifact 的每个 mapping 现在包含 `source_loop_context`。若使用
`lineage_artifact + trace_op_ids`，runtime 在编译前验证所选 source loop 确实包围每个所选 `op_id`；因此不
会用 family 名或 lowering 后的循环猜测。没有离线 artifact 的受控 adapter 也必须显式声明自己的 exact
source ordinal/loop contract。

S5000 已用 `runtime_looped_copy` 验证：一个 `T.copy` 位于 4 次 source serial loop 内，选择
`loop=0, iteration=2` 后，生成 MUSA C 恰有 2 次 `clock64()`，两次均受 `threadIdx.x == 0 && __1 == 2`
守卫；正确性、两个单调 trace record、16-byte 异步 D2H、lineage 和 iteration selector 校验均通过。该次
完整采集的 callable ABBA 中位绝对开销为约 `0.000353 ms`（约 `3.42%`）；kernel-only profiler 估计为
约 `0.000624 ms`（约 `27.5%`，微小 kernel 的百分比易放大）。它只证明该受控 copy workload 的采集行为，
不外推为任意 TileOp 的通用低开销结论。

为便于审核，每次 runtime session 额外保存 `baseline/kernel.mu` 与
`deep/instrumented_kernel.mu`，分别对应 manifest 中的 baseline/instrumented code hash；写文件发生在所有
计时窗口之后，不参与性能数据。

### 6.5 已实施的受控多 block 固定槽位 trace（2026-07-30）

编译器新增 `tl.provenance_trace_blocks`，其值为扁平的精确 block 坐标三元组；runtime 对用户配置的
`block_coordinates=[[x,y,z], ...]` 先执行预算、非负、去重和 adapter 支持范围检查，再转换为该编译配置。
默认仍为一个 block `(0,0,0)`。每个 selected TileOp 的固定槽位布局为：

```text
slot = (tileop_index × selected_block_count + block_index) × 2 + boundary
boundary: begin=0, end=1
```

因此不存在 atomic append 或 block 间槽位竞争。写入条件为所有已绑定 thread 轴均为零，且精确匹配
`blockIdx.x/y/z`；未选 block 不写 Trace。Trace ABI 的容量必须至少为
`selected_tileops × selected_blocks × 2 × int64`，不足、坐标重复、不可用维度坐标非零、或 selected block
没有产生完整 begin/end pair 时均 fail-closed。

S5000 已用 4-block copy workload 验证，选择 `(0,0,0)` 和 `(3,0,0)`：生成 MUSA C 的 4 次 `clock64()`
分别写入 `Trace[0/1]` 与 `Trace[2/3]`；业务输出、32-byte 异步 D2H、固定 slot/lineage 映射和两组 block
内 begin/end 单调性均通过。该次 callable ABBA 中位绝对开销约 `0.000467 ms`；kernel-only profiler 估计约
`0.000460 ms`。它仅代表该两个 selected block 的受控 copy 实验，不将这个微小 kernel 的相对百分比外推。

不同 block 可能驻留在不同 SM，当前没有 S5000 全局同步时钟域的证据。因此多 block trace 只解释每个 block
自身的 `end - begin` 边界区间，`trace_pair_monotonic=true`；工具明确写入
`cross_block_clock_order=not_compared`，不会以不同 block 的 `clock64()` 绝对值推断全局先后顺序。

当前限制是**静态 grid、少量 block、经过 adapter 审核的 workload**。显式坐标以及按线性 ID 的
exact/every/确定性概率选择已经实施，但任意 TileLang workload 的自动 adapter 尚未完成，也不应据此
宣称动态 grid 或全 grid trace。

### 6.6 离线 TileOp 与 Runtime slot 自动关联方案（2026-07-30 审核通过）

关联只接受 `lineage_artifact + trace_op_ids` 的 compiler-exact 路径，不允许按 family、名字或生成代码
结构猜测。证据链为：

```text
offline tile_ops.jsonl.op_id
  ↔ metadata/lineage.json.source_op_id + source_ordinal
  ↔ runtime lineage.source_op_id + source_ordinal + trace_slot
  ↔ trace.jsonl.trace_slot
```

实施规则：

1. artifact manifest 必须为完整 CPU 离线分析，lineage 必须允许 exact runtime instrumentation；
2. `op_id` 和 ordinal 在 tile metadata、compiler lineage、runtime lineage 三处必须分别唯一且完全一致；
3. runtime family、block、begin/end 和 slot 继续通过 deep trace complete gate；关联层不得覆盖原始 trace；
4. manifest 的输入配置、TileLang revision 和 input IR hash 在已有 selection evidence 中存在时必须相符；
5. 任一冲突均 fail-closed，使请求了离线关联的 deep session 降为 partial，而不是输出可能错误的 Tile 信息；
6. 没有请求 lineage artifact 的受控 adapter 仍可输出 runtime exact trace，但关联状态为 `not_requested`，
   不伪造离线 tile shape、读写、FLOPs 或循环信息。

新增证据文件：

- `deep/offline_runtime_join.json`：artifact provenance、匹配数量和完整性状态；
- `deep/tileop_context.jsonl`：每个选中 TileOp 的 raw Tile IR、family、shape/读写、循环、执行次数、
  FLOPs/逻辑字节、DAG 入边/出边和 pipeline 上下文；
- `deep/trace_enriched.jsonl`：原始时间戳记录加对应 TileOp 上下文；原始 `trace.jsonl` 保持不变。

审核决定：方案可执行。它只解决“这个 slot 对应哪个已证明的源 TileOp 以及该 TileOp 的离线属性”，
不会把离线 FLOPs/字节伪装成运行时硬件计数，也不会把 issue-boundary cycles 改称独占执行时间。

实施结果：已完成并在 S5000 验证。正式结果 `runtime_2d1f3de643574d2c` 将 GEMM 的 slot 0/1 精确关联到
`op_ba6a0cf9b64ded14`、source ordinal 3，将 reduce_sum 的 slot 2/3 精确关联到
`op_712a09f7641bd83c`、source ordinal 4；输出包含原始 Tile IR、M/N/K、buffer shape/scope、执行次数、
FLOPs/逻辑字节和 DAG 依赖。去除新增 `tileop` 字段后，`trace_enriched.jsonl` 与原始 `trace.jsonl`
逐字段相同，证明关联层没有改写运行数据。当前 63 个 CPU 单元/故障注入测试通过；完整集成矩阵包含
10 个 S5000 runtime 测试和 4 个离线集成测试。

### 6.2 最小 trace ABI

每条记录采用固定大小二进制结构，建议字段为：

```text
schema_version
run_id
kernel_id_hash
block_x, block_y, block_z
op_id_hash
pipeline_stage
loop_iteration
record_kind        # begin/end/instant/overflow
clock_cycles
flags
```

device 侧规则：

- 只有选中 block 的 lane 0 写记录；
- 使用预分配、固定 slot，不在热路径进行字符串格式化；
- 优先由静态 slot 公式定位，避免全局 atomic；
- 记录 `clock64()` 原始 cycles；
- 缓冲区不足只写 overflow 标志，不越界、不重试；
- 不插入仅用于采集的 barrier；如果必须等待才能定义某个语义点，该记录应标为
  `measurement_requires_sync=true`，并视为更高扰动实验。

已验证的 GEMM+reduce_sum / GEMM+reduce_max 和 cumsum/scan 原型采用更小的 ABI：业务输入加 `Trace`
输入参数并仅返回业务输出，其中 `Trace` 是预分配的 2×int64 device buffer，而非自动返回输出；仅记录
一个选定区间的 begin/end 边界。两种 reduction 使用同一 `reduce` family，但保持不同 semantic op ID
和记录名称；例如 `reduce_max_issue_boundary` 不得写成泛化的 `reduce_issue_boundary`。cumsum 则使用
独立的 `scan` family 和 `cumsum_begin/end_issue_boundary` 语义。
buffer 生命周期强制为：

```text
FREE → DEVICE_WRITING → DEVICE_READY → D2H_COPYING → HOST_READY → FREE
```

每个 slot 必须保存 generation ID；device-writing generation 与 readback generation 不一致即为
`invalid`，不得复用或覆盖该 buffer。

频率动态变化时，cycles 不直接换算成精确纳秒。只有 timer 语义和频率换算经过校准后，才输出
派生时间；否则保留 cycles 并附同期频率序列。

### 6.3 Tile 时间语义

允许的观测名称必须描述边界事实，例如：

- `op_issue_interval_cycles`；
- `wait_interval_cycles`；
- `pipeline_region_cycles`；
- `sampled_block_kernel_span_cycles`。

不得把重叠流水中的 begin/end 差值直接命名为 `gemm_hardware_execution_time` 或
`async_copy_exclusive_time`。精确硬件执行归因若无额外证据，状态为 unknown。

## 7. 运行输入与正确性契约

运行期不直接执行任意 benchmark 脚本。每个 workload 提供受控 adapter：

```text
build_kernel(config, variant) -> compiled callable
build_inputs(config, seed) -> tensors
run(kernel, inputs, stream) -> outputs
reference(inputs) -> expected outputs
compare(outputs, expected, tolerance) -> correctness result
```

要求：

- 编译与输入构造发生在计时窗口外；
- seed、dtype、shape、layout、容差和 reference hash 必须记录；
- correctness 至少对 baseline 执行；deep 模式还必须验证 instrumented 变体；
- 默认只保存误差摘要、失败位置数量和有限样例，不保存完整 tensor；
- correctness 失败时禁止输出可用于校准的性能状态 `valid`。

## 8. 遥测与时间对齐

`mthreads-gmi` 作为独立后台采样器运行，建议默认 200 ms 间隔。记录 monotonic host 时间和原始
设备时间字段，并保存：

- sampler 启动/退出状态；
- 实际采样间隔分布和漏样数；
- benchmark window 开始/结束时间；
- 温度、功耗、频率、thermal slowdown；
- 采样器 CPU 时间和输出字节数。

遥测报告采用窗口统计：pre/active/post 的 min/median/max/样本数。只有 active window 足够长且
样本数达到门槛时才输出 active 平均功耗。单个短 kernel 不做功耗归因。

## 9. 采集开销测量与验收门槛

### 9.1 对照设计

同一 config、输入驻留策略、stream 和 cache policy 下执行交错对照，避免热状态单向漂移：

```text
baseline A → collected B → collected B → baseline A
```

至少重复若干轮，分别保存原始样本、median、p95 和不确定性。开销定义为：

```text
relative_overhead = (collected_median - baseline_median) / baseline_median
absolute_overhead = collected_median - baseline_median
```

Event 分辨率或样本波动大于差值时，结果标为 `overhead_not_resolved`，不能强行给出精确百分比。

### 9.2 默认验收预算

以下是工具验收门槛，不是对所有 kernel 的性能承诺：

| 模式 | 默认目标 | 超限处理 |
|---|---:|---|
| light batch Event | median 相对开销不高于 2%，或无法分辨 | 降低逐 launch Event 比例/增大 batch，再测 |
| standard activity | median 相对开销不高于 5% | 缩短窗口、减少 activity/counter |
| standard counter replay | 不作为原始性能 | 记录 pass/replay，单独报告 |
| deep instrumented | 不设“等同基线”目标 | 只报告实际开销并缩小抽样范围 |

如果 light 在某个极短 kernel 上仍超过预算，基线可改为单对 Event 包围多个 launch；这只提供批量
平均值。若用户要求逐 launch 分布，只抽样少数 launch 并明确其额外开销。

## 10. 证据等级和禁止合并规则

在现有离线 evidence 标签之外，运行期拟新增：

| evidence level | 含义 |
|---|---|
| `runtime_direct` | 未插桩 Event、正确性、实际 launch/activity 直接观测 |
| `runtime_telemetry` | mthreads-gmi 窗口遥测 |
| `profiler_observed` | profiler/MUPTI 单窗口 activity 或单 pass counter |
| `profiler_replay_observed` | 多 pass/replay/multiplex counter |
| `instrumented_runtime` | instrumented kernel trace |
| `derived_correlated` | 离线 op 与运行期 PC/trace 的派生关联 |

强制规则：

1. `instrumented_runtime` 不得写入 `baseline_latency`。
2. `profiler_replay_observed` 不得写入原始单次执行字段。
3. `runtime_telemetry` 不得直接升级为单 tile 功耗。
4. `derived_correlated` 必须保存参与关联的 source ID、target ID、算法和置信度。
5. 报告中的比较只比较同一代码哈希、输入、cache policy 和 stream 条件；否则标为不可直接比较。

## 11. 配置契约示例

```json
{
  "mode": "deep",
  "device": 0,
  "sampling_seed": 20260729,
  "workload": "gemm_reduce_sum_fp16",
  "case_ids": ["M128-N128-K64"],
  "baseline": {
    "warmup_launches": 20,
    "repeat_launches": 200,
    "event_mode": "batch_average",
    "event_batch_size": 20,
    "cache_policy": "preserve"
  },
  "telemetry": {
    "enabled": true,
    "interval_ms": 200,
    "fields": ["Temp", "Power", "Freq"]
  },
  "stability": {
    "enabled": true,
    "required": true,
    "min_telemetry_points": 3,
    "min_timing_batches": 4,
    "max_temperature_c": 90,
    "max_temperature_delta_c": 5,
    "max_frequency_delta_pct": 5,
    "max_timing_drift_pct": 20
  },
  "standard": {
    "activity_kinds": ["kernel", "memcpy", "synchronization"],
    "metrics": [],
    "max_profiled_launches": 10,
    "max_profile_duration_ms": 500,
    "max_counter_passes": 1,
    "allow_replay": false
  },
  "deep": {
    "case_selector": {"values": ["gemm_reduce_sum_fp16"]},
    "kernel_selector": {"probability": 1.0},
    "stream_selector": {"probability": 1.0},
    "kernel_ids": ["kernel_e2e36e6e22c15d1b"],
    "launch_candidate_count": 100,
    "launch_selector": {"probability": 0.02},
    "block_selector": {"every": 16, "start": 0},
    "tileop_families": ["gemm_dense", "reduce"],
    "max_tileops_per_block": 3,
    "max_iterations_per_tileop": 4,
    "max_trace_records": 128,
    "max_trace_bytes": 8192,
    "trace_variant": "preallocated_tail_flush",
    "timestamps_per_sample": 2,
    "readback_policy": "async_after_window",
    "measurement_barrier_policy": "forbid"
  },
  "budget": {
    "max_total_duration_s": 30,
    "max_instrumented_launches": 2,
    "max_sampled_blocks_per_launch": 1,
    "max_dropped_records": 0,
    "on_budget_exceeded": "degrade"
  }
}
```

配置加载后生成完全展开且不可变的 `resolved_config.json`；报告以 resolved config 为准，避免默认值
变化导致不可复现。

## 12. 产物目录和数据契约

```text
results/runtime/<session_id>/
  manifest.json
  resolved_config.json
  capability_probe.json
  correctness.json
  overhead.json
  runtime_validation.json
  stream_sync.json
  diagnostics.json
  baseline/
    event_samples.jsonl
    summary.json
  telemetry/
    before.json
    active.json
    after.json
    stability.json
  standard/
    activities.jsonl
    counters.jsonl
    dropped_records.json
  deep/
    instrumentation_manifest.json
    selection.json
    readback.json
    trace.jsonl
    offline_runtime_join.json
    tileop_context.jsonl
    trace_enriched.jsonl
  report/
    report.md
    report.html
```

`manifest.json` 至少包含：

- session/workload/case/kernel/baseline/instrumented ID；
- 源码、IR、MUSA C、binary、配置和输入签名哈希；
- 请求模式、实际模式、enabled/disabled collector；
- device/driver/runtime/compiler/profiler 版本；
- 是否插桩、是否 replay、是否序列化；
- 开始/结束时间、异常退出和数据完整性状态；
- 证据等级和与 offline analysis 的关联 ID。

## 13. 完整性状态机

运行期状态独立于 `offline_static_v1`：

```text
profile = runtime_measurement_v1
status  = complete | partial | failed | invalid
```

### `complete`

- capability probe 与实际 collector 一致；
- baseline correctness 通过；
- 未插桩 baseline 样本、配置和哈希完整；
- 请求的 collector 已完成且无静默丢记录；
- overhead 已测量或明确 `not_resolved`；
- 所有记录有 mode/evidence/source；
- deep 模式的 instrumented correctness 通过，trace 未越界，lineage 满足策略。
- 若请求异步回读，capability probe、buffer generation 和 copy-done Event 均通过；否则已明确降级。

### `partial`

- collector 按规则降级；
- trace 达预算被截断但有显式 overflow/truncated；
- 遥测样本不足，或部分非必需 metric 不可用；
- dropped records 在允许范围内且报告可识别缺口。

### `failed`

- 无法建立 Event baseline；
- collector 初始化、解析、文件写入或设备调用失败；
- 请求 `on_budget_exceeded=fail` 且预算被触发。

### `invalid`

- correctness 失败；
- baseline 和 instrumented/hash/config 混淆；
- trace 越界或记录损坏；
- 数据被错误标为未插桩；
- 开销或 replay 信息缺失，导致性能语义不可信。

`invalid` 数据可以保留用于诊断，但不能进入性能模型校准集。

## 14. 模块实施结构

```text
src/musa_perf_collector/
  runtime.py                         # session orchestration
  adapters/tilelang/
    runtime_workload.py              # 受控 workload adapter
    instrumentation.py               # variant 构建和 trace ABI
    tileop_lineage.py                # op_id 跨 lowering 映射
  collectors/runtime/
    capability.py                    # 环境/权限/API probe
    event.py                         # batch/sample Event
    correctness.py
    launch.py                        # launch/stream 元数据
    telemetry.py                     # mthreads-gmi 子进程
    torch_activity.py
    mupti_activity.py                # C/C++ 扩展 adapter
    mupti_counters.py
    pc_sampling.py
    tile_trace.py
    trace_buffer.py                   # slot lifecycle、generation 与 async D2H probe
    overhead.py
    sampling.py                      # 确定性选择器与硬预算
  core/
    runtime_schema.py
    runtime_validation.py
  exporters/
    runtime_report.py
```

CLI 计划：

```text
musa-perf-collector runtime probe
musa-perf-collector runtime collect --mode light --config ...
musa-perf-collector runtime collect --mode standard --config ...
musa-perf-collector runtime collect --mode deep --config ...
musa-perf-collector runtime audit-overhead --session ...
```

## 15. 实施顺序与验收

当前阶段审核状态：

| 阶段 | 状态 | 审核结论 |
|---|---|---|
| P0 capability/契约 | 已完成（受控范围） | probe、能力降级和证据分层已运行 |
| P1 light | 已完成（受控 adapter） | Event、正确性、launch/stream、遥测与基线隔离已验证 |
| P2 standard | 部分完成 | torch MUSA activity 已完成；raw MUPTI counter/PC sampling 受当前库能力阻塞 |
| P3 deep | 已完成（受控 adapter） | exact TileOp、iteration、多 block、异步回读和离线富化已验证 |
| P4 选择器/预算 | 已完成（受控 adapter） | case/kernel/stream/launch/block 的 exact/every/确定性概率选择和聚合 trace 预算已闭环 |
| P5 独立审核 | 已完成（受控 adapter） | schema、热稳定性、current/default/custom stream、Event/readback sync gate 与故障注入均通过；77 项为该阶段历史验收数，当前完整回归见总文档的 124 项 |
| P6.0–P6.4 在线 JIT | 已完成（受控 MVP） | 真实应用无需 workload/factory；自动 dual variant、exact slot、异步回读、遥测和应用正确性已通过 S5000 |
| P6.5 多实例 lowering | 已完成（受控 artifact） | single/envelope/instance pass 与计划通过 CPU；真实 warp-specialized GEMM 的两个 final-device instance 已通过 S5000 exact-slot、正确性和审计验收 |
| P6.6 在线可靠性审核 | 已完成 | selector fail-closed、fork 资源隔离、partial slot、累计预算、真实 dispatch 序号、独立 session audit 和有界 Event drain 已通过新一轮 S5000 验证 |
| P6.7 周期抽样快路径 | 已完成 | every/start 可配置、默认 every=64；未命中直接 baseline、Event 预创建复用、稀疏记录聚合已通过 CPU 与 S5000 审核 |
| P6.8 原生 C dispatch | 已完成（selected 慢路径仍待继续下沉） | C vectorcall fast path 为约 209 ns；S5000 every-64 整体从 28.64% 降至 19.50%，正确性/trace/audit 通过，但未达到整体 2% |
| P6.9 原生 C/MUSART trace pool | 已完成 | selected Event、pinned buffer、stream wait、deferred D2H/reset 和生命周期已下沉；S5000 every-64 整体为 8.43%，完整 Event/Tile trace、正确性和审计均通过 |
| P6.9.2 证据链/打包二次复核 | 已完成 | audit 增补 plan/launch/clock/Event/fast-reason/raw-enriched 闭环并消除 layout 顺序绕过；CPU wheel 不再携带失败构建遗留的 MUSA `.so` |
| P6.9.3 多实例证据链复核 | 已完成（受控 artifact） | warp partition leader 修正；审核 instance 索引连续性、每实例 begin/end 唯一性及 pair validity |
| P6.9.4 落盘重放/类型与 partition 契约复核 | 已完成（受控 artifact） | audit 重读 JSONL、plan 和审计输入；严格 JSON index 类型、畸形 layout fail-closed；partition extent 求和校验；124 项 collector 回归与真实 S5000 session 通过 |

因此下一实施顺序是在 compiler-exact artifact 基础上减少 workload adapter 的手工代码。厂商 raw counter
仅在 MUPTI API/运行库真正具备所需符号后继续，不用模拟数据替代。

在线通用化的已审核实施方案见 `05_online_jit_auto_instrumentation_plan.md`。该路径通过 TileLang
compiler/JIT hook 观察真实应用的 compile/launch，不要求应用为每个算子提供 `RuntimeWorkload`；现有
factory/adapter 仅保留为 managed benchmark 回归夹具。

在线正式结果为
`/root/musa-perf-collector/results/online/online_61741_1785386859701854215_7557c178`：真实应用正确性
`application_reported/pass`，三个 launch 中仅索引 1 为 instrumented，GEMM/reduce 四个 exact slot
全部有效，pinned-host 异步回读成功；独立在线审计为 `pass`、0 violation，累计预约/实耗为 1 次
instrumented dispatch、4 records、32 bytes；温度/功耗/频率只在 session 起止采样且不归因到单 TileOp。实现、
命令、数据语义与审核明细见 `05_online_jit_auto_instrumentation_plan.md` 第 18 节。

P6.7 周期功能验收结果为
`/root/musa-perf-collector/results/online/online_65978_1785388149826760723_a7df459c`：every-64 的 130 次
launch 只插桩 `[0,64,128]`，其余 127 次走快路径，12 条 trace 全部有效，correctness/validation/audit
均通过。开销验收结果为
`/root/musa-perf-collector/results/online/online_67396_1785388406715139178_0c8eb119`：4099 次 launch 中
65 次插桩、4034 次快路径；约 12.7 us 的极短 kernel 上，direct median `0.01268764 ms`、online
every-64 median `0.01632159 ms`，绝对增加 3.634 us、相对增加 28.64%。该数值如实代表在线整体平均
扰动；默认 64 已显著减少慢路径频率，但未达到极短 kernel 的 2% 门限，进一步降低需要增大 every 或把
Python selector/dispatch 下沉 C++。

P6.8 已将 periodic counter、every/start、PID gate 和 baseline/instrumented 路由下沉至 CPython C
vectorcall。功能结果 `online_70158_1785389681874641627_a0b5725c` 明确记录
`dispatch_path=native_c_vectorcall` 并通过 correctness/trace/audit；同规模 ABBA 结果
`online_70664_1785389784719659421_96f45da9` 为 direct `0.01281293 ms`、native every-64
`0.01531133 ms`，绝对增加 2.498 us、相对 19.50%。未抽中 C 路径微基准增加约 209 ns，但 selected
Event/trace/readback 仍在 Python 慢路径，因此总体不能标为 2%达标。

P6.9 继续将 selected 慢路径的 MUSA Event、pinned host、producer/collector stream 依赖、slot/generation、
sentinel reset 和 D2H 下沉到直接调用 MUSART 的 C extension；运行窗口只提交 Event 与 instrumented kernel，
D2H/解码/reset 在 flush 完成。最终结果
`online_77513_1785391962244474833_b7bf17bf` 为 direct `0.01274352 ms`、online every-64
`0.01381824 ms`，绝对增加 `1.07473 us`、相对 `8.43%`。4099 launch 中 65 次 instrumented、4034 次
native fast path、260 条 Tile trace 和 65 次 MUSA Event 时延全部有效，correctness/validation/audit 通过。
Event 时延稀疏到每 8 个 selected 一次的对照为 8.42%，没有可辨别收益，所以默认仍为每个 selected 都计时；
`event_timing_every` 仅保留为用户显式证据密度选项。当前结果仍未达到极短 kernel 的整体 2%，不得修正或
外推成已达标。

P6.9.1 独立复核发现并修复四类证据链缺口：abort 可能遮蔽原始 MUSART 异常、drain 异常可能在 audit
落盘前中断、online audit 只检查 slot 数量未逐字段对照、capability 仍显示 P6.4。修正后 audit 强制核对
launch 总账、generation、op_id/ordinal/source_kind/boundary/block/lowering 和 raw/enriched TileOp 关联；
drain 失败强制 invalid，buffer 耗尽 terminal 降级并标 partial，native/fallback 与资源上限显式披露。
trace pool device 写入 capability；current device 不匹配时禁止跨设备插桩并 terminal baseline 降级。当前自动
预分配池的批准范围明确为 device 0，其他 device 不再隐式尝试。
native extension 构建路径改由 `MUSA_HOME` 解析，不再绑定单一 4.3.5 安装目录。
最新 S5000 结果 `online_86437_1785394084578490092_f7905c7a` 为 complete/0 violation；同规模 ABBA
`online_85742_1785393837693764404_bda46c63` 测得 7.08%，但正常 launch 路径未因本次 flush-only 审计修改，
因此按轮间波动处理，继续保留 8.43% 为较保守披露值，不宣称 2%达标。

P6.9.2 二次复核把独立审计扩展到 plan hash、PID/device/stream、fast-path reason 总账、trace
schema/evidence/readback、clock 类型与 uint64 interval 推导、同 launch generation/Event、variant 序号和
raw/enriched 全关键字段，产物显式记录 `audit_revision=P6.9.2`。区间配对采用两遍扫描，逆序 layout 不能再
跳过 interval 校验。打包侧把 MUSART 扩展设为可选，并在构建前删除其精确旧输出：无 MUSA 的 CPU wheel
只含 `_native_dispatch`，正常 MUSA wheel 同时含 `_native_dispatch` 和 `_native_musa_trace`。本轮完整
collector 测试为 117 项通过（103 unit、14 integration）。此前完整运行曾出现一次
GEMM+reduce_max `partial`，随后专项和完整重跑均未复现；该事实保留为未定位波动，不能声称已证明根因。
新版真实 S5000 结果 `online_99687_1785396484061581425_698e4790` 为应用正确性通过、session
`complete`、audit `P6.9.2/pass/0 violation`，4 条 raw/enriched trace 一一对应；早于新版代码生成的
`online_90618_1785395500473711501_bfc673c7` 不作为 P6.9.2 证据。

P6.9.3 使用真实应用
`src/musa_perf_collector/examples/online_multi_instance_gemm_app.py` 生成离线 artifact
历史 artifact `results/offline/analysis_280aa44a0298baf0` 已随结果清理；当前工作树已由同一应用重新生成
`results/offline/analysis_e9455c90b580aef2`，source ordinal 0 仍对应两个 final-device instance。首次 S5000
运行虽通过应用正确性，但 producer slot 保持 sentinel；根因是原注入器固定以全局 `threadIdx.x==0` 写入，
而 warp-specialized producer 只在上分区执行。注入器现从 `kWarpSpecializationScope` 读取 producer/consumer
partition，分别使用其首线程，并对异常 scope fail-closed。独立 audit 同时新增 instance 索引从 0 连续、
每实例恰有 begin/end、pair validity 一致性检查及对应故障注入。最终证据
P6.9.4 修复审核中发现的三类缺口：JSON 合法但类型畸形的 layout 曾可能作为 unhashable pair key 使 audit
异常；generation/launch index 曾被 `int()` 宽松转换；audit 曾使用内存列表而非重新打开证据文件。现在每轮
flush 落盘后重读 trace/raw-enriched/launch JSONL、`instrumentation_plan.json` 和 `session_audit_inputs.json`；
任一读取、schema、路径或类型错误使 audit fail/validation invalid。C++ injector 同时要求
`threadIdx.x extent == producer_extent + consumer_extent`。最终
`online_128980_1785403516203950746_b3bc0e1e` 为应用正确性 pass、session complete、
`P6.9.4/pass/0 violation`、`evidence_source=persisted_session_files`；instance 0/1 的四条 raw/enriched trace
全部 valid，区间分别为 1032/353 cycles。
这些区间仍仅是各 partition leader 的 issue-boundary envelope。

2026-07-30 已新增 `runtime_validation.json` 独立审计。落盘前再次检查 schema/status、baseline
correctness 与未插桩标记、requested/actual mode、deep correctness/trace validation、两种 kernel hash
隔离、五维选择证据、trace/readback launch 集合、聚合记录预算和 instrumented launch 用量。任一冲突
都会保留采集产物并强制把 manifest 标为 `invalid`，不能继续维持 `complete`。真实概率抽样结果
`runtime_006acacdc3281376` 的该审计为 `pass`、零 violation。

P5 后续真实验证结果为 `runtime_b8972d85dd67a763`。稳定性 gate 汇总 before/active/after 三个遥测点
和 20 个未插桩 Event batch：温度保持 35°C、graphics clock 保持 1750 MHz、前后半段时延中位数
漂移 12.75%（本次显式门限 20%），因此为 `stable`；该结论只属于 session 窗口，不归因到单个 TileOp。
stream/sync gate 确认 baseline 全部运行在 stream 0，instrumented producer 也在 stream 0，两次异步
回读均由独立 stream 3 执行，且 producer Event、collector wait 和 copy-done Event 完整；profiler 同时
观察到 2 个 `musaDeviceSynchronize`。另一个 S5000 集成用例已验证 custom 非零业务 stream 与独立
collector stream，default/current 路径也通过。任一 stream、wait、Event 或 required stability 冲突均
进入 `invalid`/`partial` gate，不再仅作为报告字段。

### P0：契约与 capability probe

实现 runtime schema、resolved config、能力探测和失败分类。  
验收：在当前 S5000 上输出可审核的 Event、profiler、MUPTI、遥测能力矩阵；不运行长 benchmark。

### P1：light 基线

实现 workload adapter、correctness、batch/sampled Event、遥测、统计和 overhead ABBA 对照。  
验收：使用已经通过的 GEMM、GEMM+reduce_sum、GEMM+reduce_max、transpose-B GEMM；正确性通过，
编译/warmup/correctness 不进入 Event 区间，light 开销满足或明确无法分辨/降级。

### P2：standard activity

先接入 `torch.profiler` MUSA activity，再根据 probe 结果实现最小 MUPTI activity/counter adapter。  
验收：kernel、stream、sync、correlation、launch 参数完整；counter pass/replay 和 dropped records 可审计。

### P3：deep Tile trace 原型与低扰动回读

先对一个已验证 GEMM 和一个 GEMM+reduce 用例实现稳定 op_id、固定 trace buffer；随后扩展到经过单独
S5000 回归验证的少量显式 block 坐标。  
验收：baseline 与 instrumented correctness 都通过；选择器有效；未选 block 不写；overflow 安全；
无测量专用 barrier；报告只使用 cycles 和边界语义；trace 使用预分配输入 buffer、2 个 timestamp 和
尾部写回；经 capability probe 通过时使用独立 stream/Event 异步 D2H，否则显式降级。当前已在
`gemm_dense + reduce(sum/max)` 和 `scan(cumsum)` 上完成真实 S5000 验证；其他 TileOp family 不得
因共享 ABI 而自动标记为已验证。

### P4：抽样、预算和自动降级

实现所有选择器、确定性 hash 抽样、硬预算、降级链和超限诊断。  
验收：人为触发每个预算，工具都按 resolved config 停止/降级/失败，没有范围扩大或静默截断。

### P5：独立审核与校准准入

执行 schema、故障注入、设备回归、开销、热稳定性和报告措辞审核。  
验收：只有 `runtime_measurement_v1 complete` 且 correctness 通过的未插桩 baseline 可进入模型校准；
profiler/deep 数据只能作为解释证据或单独特征源。

## 16. 测试矩阵

### 16.1 无设备单元测试

- 确定性 case/launch/block/op/iteration 抽样；
- 预算边界、overflow、stop/degrade/fail；
- capability 状态和降级链；
- Event batch 与 sampled-per-launch 统计口径；
- telemetry 时间对齐和漏样；
- evidence 禁止合并；
- baseline/instrumented hash 隔离；
- complete/partial/failed/invalid 状态机。

### 16.2 S5000 集成测试

- Event 空操作/短 kernel 分辨率和 batch 计时；
- 已验证四类 kernel 的 correctness 与 baseline；
- current/default/custom stream 和 event wait；
- profiler activity 的 kernel/stream/sync/correlation；
- mthreads-gmi 在 idle/active 窗口的连续性；
- MUPTI event/metric 枚举、单 pass 和权限失败；
- 单/多 block Tile trace、未选 block 零写入、固定槽位隔离、trace overflow；
- pinned host、独立 collector stream、Event 排序、non-blocking D2H、copy-done Event、generation
  一致性和 capability 失败后的同步降级；
- baseline 与 light/standard/deep 的 ABBA 开销对照。

### 16.3 必须失败或降级的测试

- 请求不存在的 metric；
- 请求 metric 超过 pass 预算；
- 请求所有 block/所有 TileOp 但超出 trace 预算；
- lineage 只有 heuristic 而策略要求 exact；
- instrumented correctness 失败；
- dropped activity/trace records 超限；
- 把 instrumented latency 写入 baseline 字段；
- telemetry 活跃窗口样本不足却输出 tile 功耗。

## 17. 风险与控制

1. **Event 改变极短 kernel 的 host/stream 开销**：默认 batch 包围，逐 launch 只抽样，并测 ABBA 开销。
2. **cache flush 改变 workload**：默认 `preserve`；选择 flush 必须显式记录并使用相同 baseline 对照。
3. **MUPTI 计数器要求 replay/序列化**：默认单 pass，replay 独立定级，绝不作为原始性能。
4. **Tile 插桩破坏 pipeline**：只选少量边界、单 lane 写、无新增 barrier；仍以 instrumented 证据报告。
5. **动态频率导致 cycles 换算失真**：保留 cycles 和频率序列，未校准不输出精确 ns。
6. **遥测分辨率低于 kernel 时长**：只解释重复运行窗口，不归因单 launch/tile。
7. **跨 pass op 映射不稳定**：默认只插桩 exact lineage；heuristic 必须显式授权并降级。
8. **trace 数据过大**：多维抽样、固定缓冲区、硬上限和 overflow 标志。
9. **collector 异常改变 benchmark**：每个 collector 可独立关闭；异常 session 不进入校准集。
10. **深度结果被误当基线**：不同 kernel hash、目录、schema 字段、证据等级和报告区域强制隔离。

## 18. 方案审核记录

### 18.1 可执行性审核：通过

当前环境的 Event、Stream、torch profiler、MUPTI 库/头文件、MCC 和 mthreads-gmi 足以启动 P0/P1；
P2 的具体 counter 以及后续可选的 PC sampling 必须以 capability probe 的真实返回为准，不预设全部可用。

### 18.2 扰动控制审核：通过，附强制条件

方案已经为高开销能力提供 case/kernel/launch/block/TileOp/iteration 六级选择器、记录/字节/时间/pass
硬预算、固定降级链和 ABBA 开销对照。deep 不设“等同基线”承诺，避免用抽样掩盖插桩扰动。

### 18.3 数据语义审核：通过

baseline、activity、counter replay、instrumented trace、telemetry 和派生关联使用不同 evidence level；
禁止将重叠 pipeline 区间误称为独占硬件时间，禁止把遥测归到单 tile，禁止用 instrumented 值修正基线。

### 18.4 与现有工具衔接审核：有前置项

现有离线 TileOp、执行上下文、kernel/op ID 和 complete gate 已复用。跨 pass compiler exact lineage、
instrumented variant hash、源循环 ordinal 和显式多 block 固定槽位均已实现并通过 S5000 回归，不再是
deferred 前置项。当前前置约束转为：只允许实际 grid 内的静态 block 坐标；动态 grid 或无法证明 extent
时拒绝自动多 block 插桩；所有 correctness、trace、ABBA 和 profiler 使用的 instrumented launch 必须共同
计入 `max_instrumented_launches`，不能只限制用户选择器。

### 18.5 审核决定

P0～P5 的受控 workload 能力已批准。`complete` 只适用于通过 exact lineage、
正确性、固定槽位、回读生命周期和实际预算审核的本次 session，不代表任意 TileLang workload 的通用
自动插桩。任何阶段都不得因为“想获得更细数据”而自动扩大采样范围或把插桩性能纳入原始性能字段。

### 18.6 2026-07-30 独立复核与修复准入

独立复核确认既有受控 trace 数据真实、代码哈希匹配且 S5000 回归通过，同时发现以下 complete gate
缺口，修复前不得把 Runtime 总体状态标为完成：

1. `max_instrumented_launches` 必须覆盖 instrumented correctness、trace、ABBA 和 kernel-only profiler，
   并在结果中记录预算、实际消耗和各用途明细；预算不足时跳过可选开销实验，不得静默超限。
2. trace validator 必须逐槽核对 lineage/record 的 `begin/end`、source ordinal、block 和 slot，不能只依赖
   clock 单调性与记录名后缀。
3. deep 编译前必须用 workload 的实际静态 grid 验证每个显式 block 坐标；extent 不可证明时 fail-closed。
4. `max_profile_duration_ms`、`max_total_duration_s` 和 dropped-record 能力必须分别记录“已执行、降级或当前
   API 不可观测”，不能因字段存在就宣称硬预算已闭环。
5. 上述故障注入测试、41 个 CPU 单元测试及完整 S5000 集成测试通过后，方可将版本状态提升（41 项为该次
   phase gate 的历史计数；当前完整回归计数见总文档）。

修复结果：已完成。instrumented correctness 与 trace 复用同一 launch；ABBA 与 kernel-only profiler
只消耗剩余 launch 预算，预算不足时明确输出 `unavailable`；每次调用在提交前扣账并按用途导出
`instrumented_launch_usage`。profiler 同时记录 active-window duration 与 collector wall duration，前者受
共享 `max_profile_duration_ms` 限制，dropped-record 状态明确为当前 torch profiler API 不可观测。
block 坐标已与实际静态 grid 做编译前校验；trace slot、begin/end、ordinal、block 与 lineage 必须逐项
一致。该阶段最终 41 个 CPU 单元/故障注入测试和 12 个 S5000 集成测试全部通过；当前 P6.9.4 的完整
collector 回归已增至 124 项并在总文档记录。这里的完成结论仅覆盖已审核 adapter 与显式静态坐标，
不扩大为任意 TileLang workload 或全 grid 自动 trace。
