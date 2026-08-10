# TileLang-MUSA 在线 JIT 自动插桩实施方案（审核稿）

状态：**v0.6.4，P6.0–P6.9 已在受控范围通过；P6.9.4 完成真实 warp-specialized 多实例 S5000 验收、落盘证据重放与严格契约复核**  
版本：online-jit-auto-instrumentation-v0.6.4  
日期：2026-07-30  
采集工具：`/root/musa-perf-collector`  
编译器：`/root/s5000-calibration/tilelang-musa`  
目标设备：MTT S5000，MUSA `mp_31`

当前完整交付状态、统一复现命令和最新 P6.9.4 S5000 证据见：
[`00_current_implementation_and_acceptance_status.md`](./00_current_implementation_and_acceptance_status.md)。

## 1. 目标与审核结论

目标是在真实 TileLang 应用运行时，通过 compiler/JIT hook 自动生成 baseline 与 instrumented variant，
自动完成 Trace ABI、抽样、回读、slot 解码和证据落盘。应用继续提供真实输入并正常调用原 kernel；在线
路径不要求为每个算子编写 `RuntimeWorkload` 或 factory。

审核结论：**技术可行，建议实施，但必须分层交付。** 现有 compiler-exact ordinal、block/iteration
选择、固定 slot、异步 D2H、预算和 complete gate 已覆盖核心基础。受控 MVP 可以去除绝大多数手写
instrumented adapter；以下能力不能在第一阶段自动声称：任意 execution backend、动态 grid、一个源
TileOp 对应多个 lowering clone、带副作用 kernel 的 shadow correctness，以及无需任何业务语义的正确性
证明。

现有 `RuntimeWorkload` 保留为回归/验收夹具，不再作为在线采集的入口。两种模式严格区分：

```text
managed benchmark: collector 创建输入、reference 和调用，使用 runtime collect --factory
online observe:     应用创建真实输入并调用，collector 只 hook compile/launch
```

## 2. 当前代码锚点与可复用能力

### 2.1 TileLang-MUSA

- `tilelang/jit/__init__.py::compile`：已经解析函数 attrs、`tilelang_out_idx`、pass config，随后进入 cache；
  这是在线编译 hook 的主入口。
- `tilelang/jit/kernel.py::JITKernel`：持有 PrimFunc、CompiledArtifact、adapter 和最终 callable。
- `tilelang/cache/kernel_cache.py`：cache key 已包含 TIR、out_idx、target、backend、pass config、编译 flags
  和本地库内容 stamp，可扩展 instrumented plan hash。
- `tilelang/jit/adapter/tvm_ffi.py`：严格校验输入数量，按 `result_idx` 自动创建业务输出；instrumented
  wrapper 可以在用户参数后追加预分配 Trace tensor。
- `ProvenanceAnnotation`：在 pipeline/lowering 前为源 TileOp 和源 loop 分配 lexical ordinal。
- `LowerTileOp`：将 provenance 物化到 lowered statement。
- `ProvenanceTraceInjection`：按 exact ordinal、block、iteration 写固定 begin/end slot，拒绝结构猜测。

### 2.2 musa-perf-collector

- exact `op_id → ordinal` artifact 解析；
- case/kernel/stream/launch/block 选择器和 `sha256_threshold_v1`；
- TraceBuffer generation 状态机、pinned host、独立 stream/Event 异步 D2H；
- trace/lineage/offline join 校验；
- Event、遥测、稳定性、stream/sync、预算和 runtime schema gate。

## 3. 不变量

1. 未选 launch 必须调用原 baseline JITKernel，不能调用 instrumented 后丢弃 trace。
2. baseline PrimFunc、kernel source、cache key 和 callable ABI 不得因启用 collector 而改变。
3. instrumented variant 使用独立 cache key、代码哈希和结果目录。
4. Trace 只能作为 instrumented kernel 的额外非输出参数；用户可见输入和业务输出保持不变。
5. 不按名称、family 或生成 MUSA C 猜测 lineage；只接受 compiler-exact `op_id/ordinal`。
6. 在线默认不双运行、不复制业务输入、不 dump 中间 Tensor。
7. 在线 trace 仍是 lane-0 issue boundary，不得称为 TileOp 独占硬件执行时间。
8. buffer 不可用、计划不匹配或预算耗尽时执行 baseline 并记录 skip；不能阻塞后扩大采集范围。
9. 采集异常不得改变应用返回值；无法安全降级的编译错误在 instrumented variant 上 fail-closed，baseline
   仍可用。
10. 业务正确性默认标为 `application_owned/not_observed`，不能沿用 managed benchmark 的 `pass`。
11. `every/indices` 未选 launch 必须走无锁本地快路径：只更新进程内单调计数并直接调用 baseline，不查询
    stream/device、不轮询 trace pool、不创建 Event，也不生成逐 launch JSON；总量在 session flush 时聚合。

## 4. 总体架构

```text
真实应用
  │ @tilelang.jit / kernel.compile()
  ▼
TileLang compile()（out_idx 已解析）
  ├── 原 PrimFunc ──────────────→ baseline cache → baseline JITKernel
  └── online compile hook
        ├── 校验配置、target/backend、lineage artifact、IR hash
        ├── 冻结 instrumentation_plan.json
        ├── AttachProvenanceTraceBuffer
        ├── ProvenanceTraceInjection
        └── instrumented cache → instrumented JITKernel
                                      │
                         OnlineDualVariantKernel
                                      │
             未选 launch ─────────────┼────→ baseline(*user_args)
             选中 launch ─────────────└────→ instrumented(*user_args, trace)
                                                       │
                                                Event 后异步 D2H
                                                       │
                                 trace + compiler slot manifest + runtime evidence
```

TileLang 只提供稳定的通用 hook 和 TIR 变换，不反向依赖 `musa_perf_collector`。采集器通过公开注册 API
安装 hook。禁止把 collector import 硬编码进 TileLang，也不以 monkey patch 作为正式实现。

## 5. 配置与启动契约

应用无需声明 workload。启用方式：

```bash
MUSA_PERF_COLLECTOR_ONLINE_CONFIG=/abs/path/online.json \
PYTHONPATH=/root/musa-perf-collector/src \
/root/s5000-calibration/.venv-tilelang-musa/bin/python real_application.py
```

应用入口只需一次注册，可由 `sitecustomize` 或显式 import 完成：

```python
import musa_perf_collector.online.auto_install
```

首版配置：

```json
{
  "schema": "musa-perf-collector/online-config-v1",
  "enabled": true,
  "target": "musa",
  "execution_backends": ["tvm_ffi"],
  "artifact_policy": "compiler_exact_only",
  "lineage_artifact": "/abs/path/offline/compiler_artifact",
  "kernel_selector": {"probability": 1.0},
  "launch_candidate_count": 10000,
  "launch_selector": {"every": 64, "start": 0},
  "block_selector": {"indices": [0]},
  "trace_op_ids": ["op_xxx", "op_yyy"],
  "lowering_semantics": "single",
  "correctness_policy": "application_owned",
  "compile_policy": "eager_dual_variant",
  "buffer_unavailable_policy": "run_baseline_and_record_skip",
  "budget": {
    "max_instrumented_launches": 10,
    "max_trace_records": 128,
    "max_trace_bytes": 8192,
    "max_pending_readbacks": 2,
    "max_total_duration_s": 30
  },
  "output": "/abs/path/results/online"
}
```

配置必须在第一次 TileLang compile 前冻结并计算 SHA-256。运行中修改文件不改变当前进程策略。

`launch_selector.every` 是正整数参数，默认 `64`；`start` 默认 `0`。因此默认选中 launch
`0, 64, 128, ...`。例如改为 `{"every": 32, "start": 0}` 后选中 `0, 32, 64, ...`；如需避开
首个 launch，可同时设置非零 `start`。配置缺失 `launch_selector` 时也使用 `every=64,start=0`，非法、零值、
负值或混用多个选择方式均 fail-closed。

## 6. 编译 hook 设计

### 6.1 新增通用 hook API

TileLang 新增：

```text
tilelang/jit/hooks.py
  register_compile_hook(hook)
  unregister_compile_hook(token)
  get_compile_hooks()
```

hook 输入 `CompileRequest`：

```text
prim_func
normalized_out_idx
target / target_host
execution_backend
pass_configs / compile_flags
baseline_cache_identity
compile_origin = user_baseline | collector_instrumented_internal
```

hook 返回 `CompileAugmentation`，只允许：

- `none`：保持现有编译行为；
- `dual_variant_plan`：请求额外 instrumented variant 和 callable wrapper；
- `reject_instrumentation`：记录原因但继续 baseline。

hook 不得修改传入的 baseline PrimFunc；所有变换操作 clone。
`collector_instrumented_internal` 必须绕过在线 hook，防止双 variant 编译递归。recursion guard 使用显式
compile context/token，不使用进程全局布尔值，以兼容并发编译。

### 6.2 out_idx 归一化

这是首个强制前置项。eager kernel 常将输出保存为 `tilelang_out_idx=[-1]`。如果直接在参数末尾追加 Trace，
`-1` 会错误指向 Trace。实现必须在追加参数前根据 baseline 参数数将所有负索引转换为稳定正索引：

```text
baseline params: [A, B, Out], out_idx=[-1] → normalized_out_idx=[2]
instrumented:    [A, B, Out, Trace], result_idx 仍为 [2]
```

重复、越界、Trace 被列为业务输出均 fail-closed。多输出、无输出和 scalar 参数必须单测。

### 6.3 自动 Trace ABI pass

新增 C++ TIR pass：

```text
src/transform/attach_provenance_trace_buffer.cc
tilelang.transform.AttachProvenanceTraceBuffer(name, extent)
```

职责：

1. clone PrimFunc；
2. 确认不存在同名参数/Buffer；
3. 在 params 末尾追加 `handle`，在 `buffer_map` 追加一维、静态 extent、`int64` global Buffer；
4. 写入 `tl.provenance_trace_abi_version=1`、extent、参数索引等 attrs；
5. 不改变 body、原 params 顺序、原 attrs 和业务 output index。

该 pass 在 Python `compile()` hook 中先执行，使 JIT adapter 看到的 PrimFunc 与 artifact params 一致；不能
只在 `tilelang.lower()` 内部临时追加，否则 TVMFFI adapter 的 `func_or_mod` 与 `artifact.params` 会失配。

### 6.4 instrumentation plan

采集器新增：

```text
src/musa_perf_collector/online/plan.py
```

输出冻结的 `instrumentation_plan.json`：

```json
{
  "schema": "online-instrumentation-plan-v1",
  "baseline_primfunc_sha256": "...",
  "tilelang_revision": "...",
  "target": "musa -arch=mp_31",
  "execution_backend": "tvm_ffi",
  "source_op_ids": ["op_xxx"],
  "source_ordinals": [3],
  "selected_blocks": [[0, 0, 0]],
  "trace_extent": 2,
  "trace_bytes": 16,
  "normalized_out_idx": [2],
  "slot_layout": [
    {"slot": 0, "op_id": "op_xxx", "ordinal": 3, "block": [0, 0, 0], "boundary": "begin"},
    {"slot": 1, "op_id": "op_xxx", "ordinal": 3, "block": [0, 0, 0], "boundary": "end"}
  ],
  "plan_sha256": "..."
}
```

plan hash 加入 instrumented cache key。baseline cache key 不含 online config，确保开关 collector 不会使
原 kernel cache miss。磁盘 cache 必须使用 `baseline/` 与 `instrumented/<plan_hash>/` 隔离命名空间。

### 6.5 lineage 来源

分两步实施：

- MVP：配置提供现有 `lineage_artifact + trace_op_ids`，严格复用当前 exact 校验。
- 后续：compile hook 使用 pass instrument 导出当前 PrimFunc 的 source ordinal、loop context、lowering
  match count 和输入 IR hash，生成在线 compiler manifest；离线工具消费同一 manifest。

不得在 MVP 中以 family/name 自动替代 `op_id`。family 可用于发现候选，真正编译前必须冻结 exact ID。

## 7. 双 variant 与 JIT cache

新增 collector 侧包装器：

```text
src/musa_perf_collector/online/dual_variant.py
  OnlineDualVariantKernel
```

包装器对用户保持原 `JITKernel` callable 行为和返回值：

```python
def __call__(*user_args):
    decision = selector.next(kernel_id, stream_id)
    if not decision.selected:
        return baseline(*user_args)
    lease = trace_pool.try_acquire(...)
    if lease is None:
        record_skip("trace_buffer_unavailable")
        return baseline(*user_args)
    output = instrumented(*user_args, lease.device_tensor)
    enqueue_readback_after_producer_event(lease)
    return output
```

包装器同时代理 baseline 的 `artifact`、`adapter`、`prim_func`、`get_kernel_source()`、target、config 等
公开属性；instrumented artifact 只能通过显式 `online_instrumentation` 属性访问。不能因包装后破坏现有
profiler、debug dump、cache database 或调用方的属性假设。MVP 验收先覆盖仓库内实际使用的公开属性，
未验证的反射/序列化路径明确列为 unsupported。

第一阶段只支持 MUSA `tvm_ffi` backend 和显式 `kernel.compile()`/lazy JITKernel。通过后再接 eager
`JITImpl.__call__`、Cython/NVRTC。遇到未支持 backend 时只返回 baseline，并保存 capability 记录。

编译策略：

- `eager_dual_variant`：首次 compile 同步编译两份；首调用延迟增加但首次选中 launch 不临时编译。
- `baseline_only_until_ready`：后续可选；后台编译完成前全部 baseline。首版不实现后台线程，避免编译器
  线程安全和进程退出问题。

编译耗时单独记录，不能进入 kernel latency。

## 8. 在线 launch、buffer 与并发状态机

### 8.1 launch 身份

每个进程内使用：

```text
(process_id, kernel_id, device_id, stream_id, monotonically_increasing_launch_index)
```

选择算法沿用 `sha256_threshold_v1`。多 Python 线程下 launch index 更新必须加锁；不能依赖未记录的全局
random。fork 后检测 PID 变化并重新初始化 stream/Event/buffer，不能复用父进程 MUSA 资源。
超过 `launch_candidate_count` 后，该 kernel 本 session 的后续 launch 全部 baseline，并记录 sampling window
closed；不能循环重置计数后再次扩大样本。

### 8.2 TraceBufferPool

新增：

```text
src/musa_perf_collector/online/trace_pool.py
```

pool key：`device + trace_extent + dtype`。每个 lease 状态：

```text
FREE → DEVICE_WRITING → DEVICE_READY → D2H_COPYING → HOST_READY → DECODED → FREE
```

每个 lease 保存 generation、producer stream/Event、collector stream、copy-done Event、launch identity。
同一 buffer 未到 `DECODED` 不得复用。达到 `max_pending_readbacks` 时按策略运行 baseline 并记录 dropped
selection；默认不等待空闲 buffer，以免采集器反向阻塞应用。

### 8.3 trace 初始化和 validity

被选 launch 前在计时外将 Trace 填充为 sentinel。回读后每个计划 slot 必须被覆盖且 begin/end 成对；未写
slot 说明条件路径未执行或 mapping 错误，记录为 `not_executed/partial`，不能将旧 generation 数据当新 trace。
后续可改为显式 validity bitmap；MVP 先使用每 generation 唯一 sentinel 并做碰撞规避。

sentinel 初始化在 buffer 回收/预备阶段通过 collector stream 完成，并以 ready Event 证明 lease 可用；不能
在 instrumented Event 起点后执行 `fill_`。若 ready Event 尚未完成，默认跳过该次采集并运行 baseline。
instrumented 调用前后都在当时的真实 application stream 上记录 producer Event，不能假设永远是 stream 0。

### 8.4 shutdown

注册显式 `collector.flush()` 与受限 `atexit`：

- 只等待已有 copy-done Event，带总超时；
- 原子写 manifest/JSONL；
- 未完成 generation 标为 dropped/incomplete；
- 不在退出阶段启动新 profiler 或新编译。

## 9. 在线正确性和证据状态

在线 schema 独立定义为：

```text
profile = online_runtime_observation_v1
status  = complete | partial | invalid
correctness = application_owned | application_reported | shadow_verified | not_observed
```

默认 `application_owned`：采集器不重复执行，只证明 ABI、lineage、trace、预算和回读完整，不能写
`correctness=pass`。

`shadow_verified` 仅适用于应用明确声明 pure/idempotent 的 kernel，并限制到极少 launch；禁止自动用于
inplace、随机数、atomic、通信、状态更新或外部副作用 kernel。shadow 结果永远不进入原始性能窗口。

应用可调用 `report_correctness(kernel_id, launch_id, pass/fail)` 提交业务自有验证；采集器保存来源，不推导。

## 10. 一对多 lowering 与动态控制流

injector 的默认 `single` 语义仍强制一个 source ordinal 在 lowered IR 中恰好匹配一次。
`ProvenanceAnnotation` 允许 pipeline clone 保留 ordinal；只有配置显式选择 `source_envelope` 或
`lowered_instance` 时才接受一对多，避免静默改变证据含义。

实施顺序：

1. MVP 保持 `selected_matches == 1`，不放宽 complete gate。
2. compiler manifest 导出每个 source ordinal 的 lowering match count。
3. 新增两种显式语义：
   - `source_envelope`：所有实例的外包围边界；
   - `lowered_instance`：额外稳定 instance index，逐实例 slot。
4. 未显式选择语义时，一对多继续拒绝。

动态 grid、动态 loop extent、条件不执行分别标为 unsupported、需要 runtime bound、not_executed；不能用
静态默认值猜测。

## 11. 产物契约

```text
results/online/<process_session>/
  process_manifest.json
  capability.json
  resolved_online_config.json
  diagnostics.jsonl
  kernels/<kernel_id>/
    baseline_manifest.json
    baseline_kernel.mu
    instrumentation_plan.json
    compiler_lineage.json
    instrumented_kernel.mu
    compile_overhead.json
    launch_selection.json
    trace.jsonl
    trace_enriched.jsonl
    stream_sync.json
    overhead.json
  session_validation.json
```

每条 trace 至少保存 process/kernel/launch/device/stream、plan hash、baseline/instrumented hash、slot、op_id、
ordinal、block、boundary、cycles、generation、readback policy 和 evidence level。

## 12. 实施阶段与验收

### P6.0：hook 契约和只观察模式

改动：

- TileLang `tilelang/jit/hooks.py`；
- collector `online/config.py`、`online/session.py`、`online/auto_install.py`；
- compile/launch 只记录 metadata，不编译 instrumented。

验收：启用/关闭 hook 时 baseline PrimFunc hash、cache key 和输出一致；Event latency 做 A/B 并报告
hook dispatch 的实际差值，不要求数值逐位相同；异常 hook 不影响 baseline；多进程/PID 和原子落盘通过
故障测试。

### P6.1：自动 Trace ABI 和双编译

改动：

- `attach_provenance_trace_buffer.cc` 及 Python FFI 注册；
- out_idx 正索引归一化；
- `online/plan.py`；
- instrumented cache namespace。

验收：单输出、多输出、无输出、scalar/dynamic shape 的参数数量、输出索引和 dtype 完全一致；baseline
hash/cache 不变；Trace 不出现在用户返回值。

### P6.2：在线 dispatcher 与 buffer pool

改动：

- `dual_variant.py`、`trace_pool.py`、`launch_dispatch.py`、`decoder.py`；
- generation、async D2H、shutdown flush。

验收：未选 launch 只执行 baseline；选中 launch 只执行一次 instrumented；buffer 饱和时 baseline 降级；
无 use-after-free、跨 generation 污染或应用阻塞。

### P6.3：compiler manifest 与 exact join

改动：编译器导出 op/loop ordinal、lowering match count、IR hash；collector 自动生成 online artifact。

验收：在线 op_id/ordinal/slot 与现有离线 artifact 完全一致；故意修改 revision、IR、shape、pass config
均 fail-closed。

### P6.4：真实应用验证

不通过 `runtime collect --factory` 启动，而是使用原始 TileLang 示例/测试程序并只 import hook。

验收顺序：

1. GEMM+reduce_sum 单 TileOp；
2. GEMM+reduce_sum 多 TileOp；
3. cumsum/scan；
4. loop iteration；
5. multi-block；
6. current/default/custom stream；
7. lazy 与 eager JIT。

每项比较 managed adapter 和 online hook 的 kernel hash、slot、cycles 结构与业务输出；online 不能复用
managed adapter 的 trace 构造代码。

### P6.5：一对多 lowering

实现 source envelope/instance manifest，单独 S5000 验证 software pipeline。此阶段通过前，不把复杂
pipeline kernel 标为在线自动插桩 complete。

### P6.6：在线可靠性审核

实现 selector fail-closed、fork 资源隔离、partial slot、跨 launch 累计预算、独立 session audit 和有界
Event drain。任何计数、slot、预算或生命周期矛盾都必须使 session 失效，不能继续标 complete。

### P6.7：可配置周期抽样与未命中快路径

`launch_selector` 支持 `every + start`，默认 `every=64,start=0`，同时保留 exact `indices` 和确定性
`probability`。周期/exact 模式在 wrapper 内使用单调 counter 先判定候选；未命中直接 baseline，候选才进入
session 锁、预算、metadata、Event、buffer 和记录慢路径。Event 在 pool 初始化时预创建并跨采样复用；
落盘只保存 instrumented、buffer unavailable 和错误等有证据价值的稀疏 launch，完整总数及快路径原因在
flush 时聚合。

验收：`every=64` 的 130 次真实 launch 必须只命中 `[0,64,128]`，未命中不得查询 metadata、poll pool
或生成 launch record；总 launch、fast-path、instrumented、trace 和预算计数必须由独立 audit 交叉一致。

### P6.8：原生 C periodic dispatch

将 periodic counter、`every/start` 判断、PID gate 和 baseline/instrumented callable 路由实现为 CPython C
extension 的 vectorcall callable。正常未命中路径为 `Python application → C vectorcall → baseline TVMFFI`，
不再执行额外 Python wrapper 字节码；只有选中 launch 进入既有 Python Event/trace/readback 慢路径。fork
子进程必须回到 Python 的 session reset + baseline 安全路径，不能复用父进程 MUSA 资源。扩展不可用时保留
Python fallback，并在 capability 中以 `dispatch_path` 披露实际路径。

构建命令：

```bash
cd /root/musa-perf-collector
/root/s5000-calibration/.venv-tilelang-musa/bin/python setup.py build_ext --inplace
```

wheel 安装会自动构建扩展，无需单独执行上述命令。验收必须分别报告 native fast-path 微基准和包含每 64 次
一次深度采集的 S5000 整体 ABBA；前者不能替代后者。

### P6.9：原生 C/MUSART selected trace pool

将 selected 慢路径中的 MUSA Event、pinned host buffer、producer/collector stream 依赖、D2H、sentinel
reset、slot/generation 状态机和完成回收下沉到 CPython C extension，并直接调用 MUSART。运行窗口内只记录
start/end Event 和执行 instrumented kernel；D2H、解码及 reset 延迟到 session flush，避免每次 selected
launch 都进入 Python tensor copy 路径。池容量由 instrumented launch、trace record、trace byte 和候选窗口
预算共同限制；容量耗尽继续 fail-closed 到 baseline，不阻塞业务。

`event_timing_every` 允许独立稀疏 MUSA Event 时延，Tile clock trace 仍按 launch selector 全量保留。
但只有 ABBA 证明收益时才可将它作为默认优化；当前 S5000 上 1 与 8 无可辨别差异，因此默认值为 1，所有
被选 launch 均保留 Event 时延。fork 子进程不得销毁父进程 MUSART 资源；部分初始化失败、显式 close 和
reset Event 失败必须逐项清理或明确报错。

## 13. 测试矩阵

CPU/故障注入：

- hook 注册、重复安装、卸载、异常隔离；
- out_idx 负数/多输出/越界；
- Trace 参数重名、错误 dtype/extent；
- baseline/instrumented cache key 隔离；
- artifact IR/revision/config 冲突；
- selector 重现、预算截断、buffer 饱和；
- generation 错配、缺 slot、重复 slot、退出超时；
- PID/fork、两线程 launch index；
- correctness evidence 禁止升级。

S5000：

- hook disabled 与 enabled-but-unselected baseline A/B；
- selected instrumented correctness（仅回归用例）；
- current/default/custom stream 与独立 readback stream；
- pinned async 和强制 sync 降级；
- 1/2 个 TileOp、1/2 个 block、1/2 个 sampled launch；
- lazy/eager、单/多输出、动态 scalar；
- compile overhead、未选 dispatch overhead、selected kernel/callable overhead；
- 热稳定性和 profiler 分离。

## 14. 扰动预算

在线 hook 即使未选 launch 也有一次 dispatch 判断，因此必须单独报告：

```text
hook_disabled baseline
hook_enabled_unselected
hook_enabled_selected instrumented
readback/decoder
compile overhead
```

MVP 验收目标不是零开销，而是：

- 未选 dispatch 绝对增量和百分比可测、稳定、低于配置门限；
- 不在每次 launch 创建 Tensor/Event/stream；
- selector 使用预计算 plan 和单调 counter，不读取/解析配置文件；
- 周期/exact 未选路径不获取 stream/device、不轮询 pool、不逐 launch 落盘；
- selected 开销单独报告，不写入 baseline；
- buffer 饱和默认丢采集而不是阻塞业务。

若 Python dispatch 对极短 kernel 仍明显，后续将 selector/dispatch 下沉到 TVMFFI/C++ wrapper；MVP 不用
隐藏或修正该开销。

## 15. 风险审核

| 风险 | 审核决定 |
|---|---|
| 自动追加参数导致 `out_idx=-1` 指向 Trace | 实施前必须正索引归一化，未完成不得编译 instrumented |
| adapter PrimFunc 与 artifact params 不一致 | Trace pass 必须在进入 JITKernel/lower 前作用于 cloned PrimFunc |
| instrumented 污染 baseline cache | plan hash 独立 namespace，baseline key 不含 collector config |
| 首次选中时临时编译改变行为 | MVP 使用 eager dual compile，编译耗时单列 |
| 一对多 lowering | MVP 继续拒绝；有显式 envelope/instance 语义后才开放 |
| 条件 op 未写 slot | sentinel/validity 检测并标 not_executed，不复用旧值 |
| buffer pool 满导致应用卡顿 | 默认运行 baseline 并记 skip，不等待 |
| 在线无法证明业务正确性 | 使用 application_owned 等证据，不伪称 pass |
| shadow 重复执行有副作用 | 默认关闭，仅 pure/idempotent 白名单显式启用 |
| hook 异常影响应用 | 编译 instrumented 失败时 baseline 可用；运行状态机错误 fail-closed 并停用该 kernel collector |
| 多线程/fork 复用设备资源 | 加锁 launch identity；PID 变化后重建 pool/stream/Event |
| Python dispatch 扰动短 kernel | A/B 报告，必要时下沉 C++，不修正原始数据 |

## 16. 独立审核结论

### 16.1 可实施性：通过，附阶段限制

JIT 编译入口、cache、TVMFFI 参数适配和现有 provenance pass 均有明确扩展点。自动 ABI 和双 variant
无需改写用户 Python kernel 源码，受控 MVP 可执行。

### 16.2 数据真实性：通过

应用提供真实参数和真实 launch；未选 launch 使用原 baseline；选中 launch 的 instrumented trace 单独
定级。方案不创建模拟 counter、不把插桩性能当 baseline，也不把 telemetry 归因到单 TileOp。

### 16.3 正确性语义：通过，必须使用独立在线 schema

在线模式缺少业务 reference 不是错误，但必须标 `application_owned/not_observed`。只有应用报告或安全
shadow 才能提升 evidence，不能沿用 managed benchmark 的 correctness gate 文案。

### 16.4 主要前置项

P6.1 开始前必须先完成并测试：out_idx 正索引归一化、自动 Trace ABI、instrumented cache identity。
这三项任一缺失都会造成 ABI 或缓存语义错误，不能用手写 wrapper 临时绕过。

### 16.5 批准范围

批准按 P6.0 → P6.1 → P6.2 → P6.3 → P6.4 → P6.5 → P6.6 → P6.7 → P6.8 → P6.9 实施。首个 complete 声明只允许：MUSA
`tvm_ffi`、静态 grid、exact lineage、single-lowered-match、受控 trace 预算。其他 backend、动态 grid、
一对多 pipeline 必须分别通过验收后扩展能力矩阵。2026-07-30 已把一个真实静态 warp-specialized
producer/consumer artifact 加入能力矩阵；这不自动批准其他 pipeline 形态。

## 17. 修改控制、备份与回滚

TileLang-MUSA 当前是 dirty worktree，实施时必须遵守：

1. P6.0 开始前已记录 `git status --short`、当前 revision 和 native library hash，并曾建立阶段备份；这些
   旧备份后续已按用户明确授权删除，当前不得再把 `backups/` 描述为可用恢复源。
2. 不使用 `git reset --hard`、`git checkout --` 或清理未跟踪文件；现有 codegen、reduce、provenance 等
   修改均视为需要保留的工作。
3. 每个 phase 单独形成修改清单，执行 `git diff --check`、Python compileall、CPU 测试和对应 S5000
   测试后才能进入下一 phase。
4. hook registry 默认空、online config 默认 disabled。关闭 hook 后代码路径必须退化为当前原始
   `compile → cached → JITKernel`，不创建 collector thread、Tensor、Event 或结果目录。
5. 回滚优先通过 feature flag 停用；若必须恢复文件，先从当前 dirty worktree 和已知修改清单生成逐文件
   逆向 diff 并取得确认，不能假定旧备份存在，也不能影响用户原有 dirty 修改。
6. native pass 新增后重新构建 TileLang-MUSA，记录构建命令、库 hash 和 S5000 capability；旧库未包含
   Attach pass 时在线 capability 显式为 unavailable，不能退回 Python AST 改写。

实施完成度只按 P6 phase 和 capability matrix 报告；“方案已批准”不等于“在线自动插桩已实现”。

## 18. 2026-07-30 实施与审核结果

### 18.1 已实现

- P6.0：TileLang 通用 compile hook registry、context-local recursion guard、observe-only 会话、fork/线程身份、
  原子 JSON/JSONL 落盘；未注册 hook 时直接进入原 `cached()` 路径，不计算额外 IR hash。
- P6.1：原生 `AttachProvenanceTraceBuffer`、负 `out_idx` 在 instrumented clone 上正索引化、baseline 与
  `instrumented/<plan_sha256>/` cache 隔离、eager dual compile。
- P6.2：确定性 launch 抽样、预分配 TraceBufferPool、generation/sentinel、池满 baseline 降级、真实 stream
  Event、pinned-host 异步 D2H 与同步 flush 降级。
- P6.3：IR hash、TileLang revision、`op_id→ordinal`、lowered/final instance count、静态 grid/block 的 exact
  join；输出 compiler lineage、slot plan、原始和富化 trace。
- P6.4：独立真实 TileLang 应用验证，不经过 `runtime collect --factory`，应用提供真实输入并自行报告正确性。
- P6.5：`single`、`source_envelope`、`lowered_instance` 三种显式 lowering 语义；默认 single 继续拒绝一对多。
- P6.6：配置与选择器严格 fail-closed；fork 后禁用父进程 MUSA trace 资源；partial slot 独立有效性；
  instrumented dispatch 与选择预约分离；跨 launch 累计 record/byte 预算；新增独立 `session_audit.json`，
  launch 错误、计数/slot/预算矛盾会把 session 强制标为 `invalid`；Event drain 有 5 秒总上限并显式记录
  timeout；`close()` 后 atexit flush 为 no-op。
- P6.7：`launch_selector.every` 可配置为 `32/64/...` 且默认 64；周期/exact selector 使用 wrapper 本地
  counter 快路径，未选 launch 不再经过 session 锁、metadata、pool poll 或逐条 JSON；start/end/copy-ready
  Event 在 TraceBufferPool 初始化时预创建并复用，session 关闭时用 provider 聚合稀疏 launch 统计。
- P6.8：新增 `online/native_dispatch.c` CPython C vectorcall dispatcher；periodic 模式自动返回 native callable，
  C 层执行原子 counter、every/start、PID gate 和 callable 路由；fork 回到安全 Python 路径，扩展缺失时
  保留 Python fallback；capability 明确输出 `dispatch_path=native_c_vectorcall|python`。
- P6.9：新增 `online/native_musa_trace.c`，原生管理 pinned host、MUSA Event、collector stream、slot/generation、
  sentinel、deferred D2H/reset 和 fork/close 生命周期；native 路径不再重复创建 Python Event/pinned tensor；
  capability 输出 `readback_policy=native_deferred_at_flush`。初始化中途失败按已创建资源逐项释放，close 后
  调用明确拒绝；native capacity 同时受 launch/record/byte/candidate 四项预算约束。
- 默认轻量证据：compile/launch、MUSA Event、stream/device、应用正确性、起止温度/功耗/频率快照；遥测
  在 Event 窗口之外执行。

### 18.2 真实 S5000 结果

正式结果：

```text
/root/musa-perf-collector/results/online/online_61741_1785386859701854215_7557c178
```

- 应用正确性：`application_reported/pass`；
- launch：0 和 2 为 baseline，1 为唯一 instrumented launch；
- trace：GEMM/reduce 各一对 begin/end，共 4 条，全部 valid；
- 本次 interval：GEMM 644 cycles，reduce 6278 cycles；语义为 lane-0 issue-boundary envelope，不能称为
  TileOp 独占硬件执行时间；
- instrumented MUSA Event：0.226 ms，并标记 `instrumented_variant_cold_start=true`；
- D2H：`async_after_window`；
- launch context：compiler-observed grid `1×1×1`、block `256×1×1`；
- 遥测：起止 35°C、图形频率 1750 MHz、功耗约 394 W，明确不在 kernel Event 窗口内。
- observe-only 与 dual-variant 的 baseline source SHA-256 均为
  `d120c5f952cc86fa24f920b995bc151eb0673da4cbec1b7efe1aa833ce9243ee`；instrumented source hash 为
  `fc1a39d7684ea69fc3e07324fb5286888bcba30332059995adb4a7da65d77aea`，证明 baseline 未被插桩覆盖。
- P6.6 独立审计：`session_audit.json status=pass`、0 violation；预约与实耗均为 1 次 instrumented
  dispatch、4 个 trace records、32 bytes。

周期抽样功能验收结果：

```text
/root/musa-perf-collector/results/online/online_65978_1785388149826760723_a7df459c
```

- 配置 `every=64,start=0`，真实执行 130 次，仅 launch `[0,64,128]` 为 instrumented；
- 127 次走未命中快路径，仅落盘 3 条 instrumented launch observation，产生 12 条有效 trace；
- 应用正确性 `pass`，session validation/独立 audit 均 `pass`，0 violation；
- audit 交叉计数为 total 130、fast path 127、instrumented 3、trace 12，证明稀疏记录没有丢失总量语义。

周期抽样开销验收结果：

```text
/root/musa-perf-collector/results/online/online_67396_1785388406715139178_0c8eb119
```

- 8 个 512-launch ABBA 批次，总计 4099 次真实 launch，其中 instrumented 65、fast path 4034、trace 260；
  correctness、validation 和 audit 均通过；
- direct baseline median `0.01268764 ms`，online every-64 median `0.01632159 ms`；平均绝对增加
  `0.00363396 ms`（3.634 us），相对增加 `28.64%`；
- 该 kernel 仅约 12.7 us，约 1 us 是每次 Python wrapper 固定成本，其余为每 64 次一次的 Event、metadata、
  buffer/readback 和稀疏记录慢路径摊销。该结果是当前在线整体平均开销，不能用仅比较两个 kernel variant
  得到的约 5% 数字替代；更长 kernel 的相对比例会下降，但必须逐 workload 实测；若需继续降低可显式改为
  `every=128/256/1024`，或将 selector/dispatch 下沉 TVMFFI/C++。

P6.8 原生 C dispatch 功能结果：

```text
/root/musa-perf-collector/results/online/online_70158_1785389681874641627_a0b5725c
```

- capability 为 `dispatch_path=native_c_vectorcall`；3 次真实 launch 中 1 次 instrumented、2 次 native
  fast path；GEMM/reduce 4 条 trace 全部有效；correctness、validation、独立 audit 均通过。

P6.8 原生 C dispatch ABBA 结果：

```text
/root/musa-perf-collector/results/online/online_70664_1785389784719659421_96f45da9
```

- 与 P6.7 相同的 4099 launch：65 次 instrumented、4034 次 native fast path、260 条 trace，correctness、
  validation 和 audit 全部通过；
- direct median `0.01281293 ms`，native online every-64 median `0.01531133 ms`，绝对增加
  `0.00249840 ms`（2.498 us），整体相对开销 `19.50%`；相比 Python every-64 的 28.64% 已下降，但没有
  达到整体 2%；
- 独立纯调用微基准中 native fast path 相对 direct 增加约 `209 ns`，低于 12.7 us kernel 的 2%绝对预算
  `254 ns`。剩余整体开销来自 65 次 selected Python Event/trace/readback 慢路径的摊销，因此不能声称完整
  every-64 collector 已达到 2%。

P6.9 原生 MUSA trace pool 分阶段 ABBA 结果：

```text
逐 selected 异步 D2H：/root/musa-perf-collector/results/online/online_73327_1785390797544449831_2eb5a66b
窗口后 deferred D2H：/root/musa-perf-collector/results/online/online_75586_1785391300353777578_4eb056b2
最终 Event 全采样：  /root/musa-perf-collector/results/online/online_77513_1785391962244474833_b7bf17bf
```

- 原生 Event/pinned pool 但逐 selected D2H 时，direct `12.24254 us`、online `13.81859 us`，增加
  `1.57605 us`，相对 `12.87%`；
- D2H 延迟到 flush 后，direct `12.30926 us`、online `13.46586 us`，增加 `1.15660 us`，相对 `9.40%`；
- 最终默认 `event_timing_every=1`，direct `12.74352 us`、online `13.81824 us`，增加 `1.07473 us`，
  相对 `8.43%`。4099 次真实 launch 中 65 次 instrumented、4034 次 native fast path、260 条 Tile trace，
  65 次 MUSA Event 时延全部保留，correctness、validation、audit 均通过且 0 violation；
- `event_timing_every=8` 的对照为 `8.42%`，与全 Event 的 `8.43%` 无可辨别差异，因此不将稀疏 Event
  冒充有效降耗，也不把默认改成 8。它只作为用户显式选择的证据密度参数；
- deferred D2H 首版 `online_74802_1785391166848330490_7a3b474f` 因 flush 只做 nonblocking poll 而被独立
  audit 正确标为 invalid；修复为 flush 执行一次 wait drain 后，`online_75207_1785391243901179411_aa216ab2`
  通过。该失败产物保留作故障证据，不计入有效性能结果。

P6.9.1 独立复核修正：

- 修复 `enqueue_readback` 已进入 INVALID 后再次 abort 会遮蔽原始 MUSART 错误；abort 现在对 INVALID 幂等，
  Python 异常路径即使 abort 自身失败也保留原始 instrumented 异常并单独记录诊断；
- native drain 错误不再中断 flush 且丢失审计文件，而是落盘 `ONLINE_TRACE_DRAIN_FAILED` 并强制 session
  `invalid`；buffer 永久耗尽在 deferred 模式下只记录一次 skip，随后 C dispatcher 进入 terminal baseline，
  session 标 `partial`；
- 独立 audit 新增 total=fast+recorded 闭合、generation 所有权、逐 slot 的 op_id/ordinal/source_kind/boundary/
  block/lowering 对照，以及 raw/enriched 字段和 enriched TileOp op_id 对照；不再只核对 slot 数量；
- capability/validation 真实标记 `phase=P6.9`，并披露 Event 抽样间隔、pool capacity、native/fallback 状态和
  fallback 错误；配置拒绝 NaN/Infinity 时间预算，pool 同时受 4096 slot 和 64 MiB 原生内存安全上限保护；
- capability 披露当前预分配 trace device；selected launch 的 current device 不匹配时禁止传入跨设备 trace
  tensor，记录一次 baseline skip 后 terminal 降级并标 partial。当前自动池仍只覆盖配置内建的 device 0；
- native extension 构建不再写死 MUSA 4.3.5 路径，按 `MUSA_HOME`（默认 `/usr/local/musa`）解析 include
  及 `lib/lib64`；隔离 wheel 安装后两个扩展可导入且包内/安装哈希一致；
- 修正后的最终功能结果为 `online_86437_1785394084578490092_f7905c7a`：device 0、3 launch、1 instrumented、4 条 exact
  trace、correctness complete、加强 audit 0 violation；同规模 ABBA 复核
  `online_85742_1785393837693764404_bda46c63` 为 direct `12.10754 us`、online `12.96426 us`、相对
  `7.08%`。本次下降不能归因于 flush-only 审计修改，视为跨轮测量波动；仍保留此前 8.43% 作为较保守结果，
  不宣称达到 2%。

P6.9.2 二次独立复核修正：

- audit 继续核对 plan hash、PID/device/stream、fast-path reason 总账、trace schema/evidence/readback、
  `clock_cycles` 类型、由 begin/end `clock64` 按 uint64 推导的 interval、同 launch generation/Event 时延、
  instrumented variant 序号/cold-start，以及 raw/enriched 全关键字段；落盘显式记录
  `audit_revision=P6.9.2`；
- interval 配对改为先建立完整 begin map、再检查全部 end，不再依赖 instrumentation plan 的序列化顺序；
  “end 在 begin 前 + 篡改 interval”的故障注入已确认无法绕过；
- `_native_musa_trace` 改为可选扩展，使无 MUSA 主机仍可安装 CPU 离线分析；每次可选扩展构建前只清除
  自身精确输出，防止失败后把 `build/lib` 中陈旧 `.so` 打进 CPU wheel。`MUSA_HOME=/nonexistent/musa`
  的故障构建只含 `_native_dispatch`，正常 MUSA 4.3.5 wheel 同时含两个扩展；
- 完整 collector unit/integration 本轮为 117 项通过（103 unit、14 integration）。此前一次完整运行中
  `GEMM+reduce_max` 曾产出 `partial`，其临时目录已删除，随后单测及本轮完整运行均通过；因此保留为一次
  未复现的设备/采集波动事实，不把它改写成从未发生，也不据此虚构根因。
- 新版真实 S5000 证据为
  `/root/musa-perf-collector/results/online/online_99687_1785396484061581425_698e4790`：应用正确性通过，
  session `complete`，audit `P6.9.2/pass/0 violation`，4 条 raw trace 与 4 条 enriched trace 完整对应。
  较早的 `online_90618_1785395500473711501_bfc673c7` 生成于该 revision 落地前，不作为 P6.9.2 证据。

P6.9.3 多实例 lowering 与独立复核：

- 新增真实应用 `src/musa_perf_collector/examples/online_multi_instance_gemm_app.py`。历史离线 artifact
  `results/offline/analysis_280aa44a0298baf0` 已随结果清理；当前同一应用重新生成的
  `results/offline/analysis_e9455c90b580aef2` 中 `op_bfdf2ff536ef6911` 的 lowered/final-device instance
  count 均为 2，在线配置显式选择 `lowered_instance`；
- 首次真实运行应用正确性通过但 session 为 partial：producer instance 的 begin/end 均保持 sentinel，
  consumer instance 有效。根因是注入器固定使用全局 `threadIdx.x==0`，而 producer 只在
  `threadIdx.x>=128` 的 warp partition 执行；
- `ProvenanceTraceInjection` 现识别 `kWarpSpecializationScope`，producer/consumer 分别以分区首线程写入，
  scope 缺少双分支、partition 数量或 extent 异常时 fail-closed。新增 CPU IR 测试确认两个分区分别使用
  leader 128 和 0；
- audit revision 升至 P6.9.3，新增 lowered-instance 索引必须从 0 连续、每个实例必须恰有 begin/end、
  begin/end validity 必须一致，并以非连续索引、重复/缺失 boundary 和 validity 不一致做故障注入；
- 最终 S5000 证据
  `/root/musa-perf-collector/results/online/online_117319_1785400242605507617_62be9e46`：正确性 pass、
  session complete、audit `P6.9.3/pass/0 violation`，3 launch 中 1 次 instrumented、2 次 fast path，
  4 条 raw/enriched trace 全部有效；两个 instance 的 issue-boundary interval 为 1036 和 353 cycles。该证据
  仅保留为 P6.9.3 历史记录；当前验收以随后 P6.9.4 的落盘重放证据为准。
- 加固过程中的 `online_111692_1785399742660951316_6d2e6e8f` 因首版条件检查未接受编译器规范化的
  `consumer_extent <= threadIdx.x` 等价形式而 fail-closed 为 partial；该检查现只接受 `threadIdx.x >= extent`
  及其等价 LE 形式，拒绝反向/无关条件。该 partial session 不计作完成证据。

P6.9.4 落盘证据重放与严格契约复核：

- 独立审核不再读取 flush 内存列表：先写入 `launch_observations.jsonl`、`trace.jsonl`、
  `trace_enriched.jsonl`、每 kernel `instrumentation_plan.json` 与 `session_audit_inputs.json`，随后重新
  解析这些持久化文件。读取、JSON/schema、plan 路径或预算类型异常均记录为
  `PERSISTED_EVIDENCE_READ_FAILED`，audit fail，session invalid；audit 产物显式记录
  `evidence_source=persisted_session_files` 和 audit-input 路径；
- audit 不再对 `op_id=[]`、`boundary=[]`、`lowering_semantics={}`、`lowered_instance_index={}` 等合法 JSON
  畸形值抛 `TypeError`；字段先规范化再参与 pair key。launch index、generation、slot 只接受非 bool 的 JSON
  integer，`True` 或字符串数字均 fail-closed；
- `kWarpSpecializationScope` 除双分支、阈值和 partition leader 外，新增
  `threadIdx.x extent == producer_extent + consumer_extent` 检查；不匹配 scope 直接拒绝插桩；
- 最终 S5000 证据
  `/root/musa-perf-collector/results/online/online_128980_1785403516203950746_b3bc0e1e`：正确性 pass、
  session complete、audit `P6.9.4/pass/0 violation`、3 launch 中 1 次 instrumented、2 次 fast path，
  4 条 raw/enriched trace 全部有效；两个 instance interval 为 1032 和 353 cycles。

### 18.3 审核与测试

- collector unit：109 项通过（新增 unhashable layout、严格 generation/launch-index 类型、落盘证据读取失败和
  partition extent 不匹配的故障注入）；
- collector 全量 unit/integration：124 项通过（109 unit、15 integration）；integration 中仅有 53 条
  torch_musa pinned API 的既有 deprecation warning；
- 原生池专项 S5000：custom stream、pool full、`event_timing_every=1/2`、fork child 禁用、close 后拒绝均通过；
- 本轮 TileLang 多实例/warp-partition 注入专项：6 项通过；此前 provenance/JIT/cache 专项结果作为历史
  验收保留，不替代本轮计数；
- TileLang `git diff --check` 和 native CMake/Ninja 重建通过；P6.9.2 的 `compileall` 与 wheel 双扩展打包
  审核结论未被本轮 Python/C++逻辑修改扩大解释。

### 18.4 当前批准能力边界

可标 complete：MUSA `tvm_ffi`、静态 grid、compiler-exact artifact、受控 launch/block/TileOp 预算、
single-match 自动 trace，以及已验收 artifact 上显式选择的 two-instance `lowered_instance` trace；真实应用
不需要 RuntimeWorkload。

多实例 `source_envelope/lowered_instance` 的 pass 与 slot 计划已经实现；真实 warp-specialized GEMM 的
two-instance `lowered_instance` 已完成端到端 S5000 验收。capability 仍按具体 kernel/plan 报告，不能由
一个静态 producer/consumer artifact 泛化宣称所有 software-pipeline kernel 已完成。动态 grid、动态
instance count 和其他 execution backend 仍 fail-closed。

P6.6 后重新生成真实两级 software-pipeline GEMM artifact
`/root/musa-perf-collector/results/offline/analysis_906cab1f648e13da`；其 fill、两次 copy、GEMM 和末尾 copy
在 S40/S60 的 compiler provenance instance count 均为 1。该样本因此只能继续验证 single 语义，不能被
人为改写成一对多验收样本。该旧样本仍只证明 single 语义；P6.5 的新增完成结论来自独立的
`analysis_280aa44a0298baf0`（已清理的历史结果）对应的真实 two-instance artifact；当前可再生 artifact 为
`analysis_e9455c90b580aef2`。
