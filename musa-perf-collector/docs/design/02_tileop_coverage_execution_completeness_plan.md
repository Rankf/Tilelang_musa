# TileOp 全覆盖、执行次数与完整性完善方案（审核稿）

状态：**v0.3 已实施并完成 CPU-only 审核；作为 P6.9.2 Tile trace 富化来源已复核**  
版本：tileop-coverage-execution-v0.3  
日期：2026-07-30  
工具根目录：`/root/musa-perf-collector`  
参考 TileLang-MUSA commit：`4d666968a005846314e9a84d8ff121c5dbffa9ee`

本文是 `01_offline_compiler_analysis_plan.md` 的完整性补充方案；本方案的 TileOp 覆盖、执行域和 complete gate 定义优先于原方案对应章节。

当前完整交付状态、S5000 运行期证据和在线 trace 的 `op_id` 对照见：
[`00_current_implementation_and_acceptance_status.md`](./00_current_implementation_and_acceptance_status.md)。

## 0. 实施与审核记录（2026-07-29）

已实施：基于 `TLOpBuilder` 的无损发现、`region` constructor 分类、unknown/未注册 TileOp 的 opaque fallback、当前 20 种动作型 TileOp 的 family registry、循环/条件上下文、per-invocation/per-block/per-kernel 聚合、bounded static reuse 展开、complete gate、schema/DAG cycle 校验及新产物文件。

已验证：真实 TileLang pipelined GEMM CPU-only lowering、真实 cumsum/reduce/atomicadd IR metadata fixture，以及全部 20 种动作型 TileOp 的 adapter 契约。回归测试不得执行设备代码；最新审计产物的 `device_execution=false`、`device_compile=false`。

边界仍然有效：后续新增 TileOp 会自动导出 opaque record 并使状态为 partial；CUDA 专有显式变体若出现在 MUSA 输入中会标为 `TARGET_VARIANT_MISMATCH`；所有 hardware/runtime 字段仍为 expected unknown。

**MUSA 适配确认顺序**：适配测试优先选择能够在 `target=musa` 下完整 lowering 的 kernel；在设备运行阶段获得明确授权且设备可用时，再以这些相同 kernel 做运行确认。WGMMA/TCGEN 等 CUDA 专有变体只验证无损保留和 mismatch 诊断，不得计入 MUSA 算子适配通过率。

## 1. 审核结论

现有工具的 CPU-only lowering、IR/MUSA C 导出和 `copy + gemm + fill` 示例链路可以保留，但不能继续用当前条件输出 `status=complete`。进入真正的 offline `complete` 前，必须同时补齐以下四项：

1. 所有 TileOp 都被发现并导出，未知算子不得静默消失；
2. 按算子族建立结构化适配器，并明确每种算子的支持等级；
3. 为每个算子保存词法循环上下文，按一致执行域计算调用次数和工作量；
4. 使用可机读的完整性规则生成 `complete/partial/failed`，不能只根据两个 IR 快照判断。

本方案仍为纯 CPU 编译期分析：不运行 MUSA kernel、不调用 Event/MUPTI、不查询 S5000 设备参数，也不输出未经校准的绝对时延。

## 2. 当前问题基线

### 2.1 TileOp 覆盖不足

当前采集器硬编码白名单：

```text
copy, gemm, fill, clear, reduce, atomic
```

当前 TileLang-MUSA 实际注册的动作型 TileOp 共 20 种，另有一个非动作型参数构造器 `region`：

```text
copy, async_copy, tma_copy,
transpose, im2col, c2d_im2col,
gemm, gemm_py, gemm_sp,
wgmma_gemm, tcgen05_gemm,
wgmma_gemm_sp, tcgen05_gemm_sp,
fill,
reduce, cumsum, finalize_reducer,
atomicadd, atomicmax, atomicmin,
region  # 参数构造器，不是执行动作
```

其中 `atomic` 和 `clear` 不是当前实际注册名：`T.clear` 最终表现为 `fill`，原子操作实际名称为 `atomicadd/atomicmax/atomicmin`。因此当前版本只有 `copy/gemm/fill` 完成结构化分析，`reduce` 仅保留 unknown，其他算子会被静默遗漏。

### 2.2 指标执行域混合

当前 pipelined GEMM 中：

- `fill` 在 K 循环外，每 block 执行 1 次；
- 两个 `copy` 和一个 `gemm` 位于 `kk extent=4` 循环内，每 block 执行 4 次；
- 输出 `copy` 在循环后，每 block 执行 1 次；
- grid 为 `2 × 2`，共有 4 个 block。

现有实现把每个静态 TileOp 各累加一次，却命名为 `per_representative_tile_iteration`。这会把 prologue、loop body 和 epilogue 混入同一个分母，导致 arithmetic intensity 口径不成立。

### 2.3 完整性判断失真

当前只要存在 `S00_INPUT_TILE_IR` 和 `S60_FINAL_DEVICE_IR` 就标为 `complete`。即使有动作型 TileOp 未导出、循环次数未展开或结构化字段未知，状态仍然是 complete。该条件必须替换为多门槛验证。

## 3. 必须满足的设计不变量

实施后必须始终满足：

```text
observed_action_tileop_calls
  = exported_structured_records
  + exported_opaque_records

unexported_action_tileop_calls = 0
```

并满足：

1. `region` 等构造器可忽略，但必须计入 `ignored_constructor_calls` 并记录原因；
2. 新增或未知 TileOp 自动导出为 `opaque`，保存名称、IR、参数摘要、注解、位置与循环上下文；
3. `opaque` 动作型 TileOp 会使分析降级为 `partial`，但不会使导出失败；
4. 每个工作量数字必须带 `execution_domain`，禁止跨域相加；
5. 符号循环可以输出符号乘数，不因缺少数值绑定而伪造数值；
6. 预期未知的硬件字段（真实 cache hit、有效带宽、绝对时延）不阻止 offline 静态分析达到 complete；
7. 任何真实采集字段仍保持 `unknown/not_collected`，不能被静态结果替代。

## 4. 总体实现结构

```text
S00 Tile IR
   │
   ├── TileOp discovery（TLOpBuilder 属性，不使用动作白名单）
   │      ├── constructor: region
   │      └── action: 所有其他 TileOp
   │
   ├── Context-aware visitor
   │      ├── lexical statement path
   │      ├── loop stack / condition stack
   │      └── block/thread launch context
   │
   ├── Family adapter registry
   │      ├── copy / transform
   │      ├── dense / sparse GEMM
   │      ├── reduce / scan
   │      ├── atomic
   │      └── fill / unknown fallback
   │
   ├── Execution-domain aggregation
   │      ├── per invocation
   │      ├── per block
   │      └── per kernel launch
   │
   └── Completeness validator
          ├── coverage
          ├── semantic fields
          ├── execution context
          ├── artifact/schema
          └── complete / partial / failed
```

计划新增或拆分文件：

```text
src/musa_perf_collector/adapters/tilelang/
  tileop_discovery.py       # TLOpBuilder 检测、构造器分类
  execution_context.py      # 带 loop/condition stack 的 visitor

src/musa_perf_collector/collectors/compiler/
  tileop_registry.py        # canonical family、alias、support level
  tileops/
    base.py                 # 通用 opaque record 与 region 解析
    copy.py                 # copy/async/tma/transpose/im2col
    gemm.py                 # dense/sparse/显式变体
    reduction.py            # reduce/cumsum/finalize
    atomic.py               # add/max/min
    fill.py
  execution_counts.py       # 符号/数值调用次数
  workload.py               # 分执行域聚合

src/musa_perf_collector/core/
  validation.py             # 完整性门槛与诊断码
  schema.py                 # 轻量数据契约校验
```

现有 `metadata.py` 和 `analysis.py` 在迁移完成后只保留兼容入口，不继续扩大单文件分支。

## 5. TileOp 通用发现机制

### 5.1 发现规则

使用 TileLang 已有判定方式：

```python
is_tile_op = call.op.get_attr("TLOpBuilder") is not None
```

该规则与 TileLang 编译器识别 TileOperator 的机制一致，优于维护 `{"copy", "gemm", ...}` 白名单。为防止异常 IR，还增加第二条兜底规则：名称以 `tl.tileop.` 开头但没有 `TLOpBuilder` 时，也导出记录并产生 `UNREGISTERED_TILEOP_CALL` 诊断。

### 5.2 分类规则

每个发现的 Call 必须被分类为：

- `constructor`：当前仅 `region`，不是 DAG 节点；
- `action`：具有计算、搬运、同步或内存副作用的 TileOp；
- `unknown_kind`：无法确定类别，仍按 action 保守处理并降级 partial。

遍历结束时输出计数：

```json
{
  "observed_tileop_calls": 15,
  "observed_action_calls": 5,
  "ignored_constructor_calls": 10,
  "exported_action_records": 5,
  "unexported_action_calls": 0
}
```

这里的数字按具体 IR 生成，不硬编码为示例值。

### 5.3 未知算子最低记录契约

即使没有适配器，也必须保存：

```json
{
  "op_id": "...",
  "raw_op_name": "tl.tileop.future_op",
  "canonical_family": "unknown",
  "support_level": "opaque",
  "source_snapshot": "S00_INPUT_TILE_IR",
  "statement_path": "root/body/3/for.body/1",
  "raw_ir": "...",
  "raw_args": ["..."],
  "annotations": {},
  "loop_context": [],
  "reads": "unknown",
  "writes": "unknown",
  "workload": "unknown",
  "diagnostic_codes": ["UNSUPPORTED_TILEOP_ADAPTER"]
}
```

不得为 unknown op 自动填 `reads=[]`、`writes=[]`，因为空集合意味着“已证明无访问”，而不是“尚未解析”。

## 6. 算子族适配方案

### 6.1 支持等级

每个导出记录使用以下等级：

| 等级 | 含义 | 是否允许 overall complete |
|---|---|---|
| `structured` | 已解析 region、语义字段和适用的工作量 | 是 |
| `structural` | 已解析 region/依赖，但该算子没有或暂不计算 FLOPs | 是，需声明 `workload_not_applicable` |
| `opaque` | 仅保证不丢失，关键语义未知 | 否，overall 为 partial |
| `invalid` | IR/参数不满足契约 | 否，overall 为 failed 或 partial |

### 6.2 Canonical family 与 alias

| Family | TileOp 名称 | v0.2 必需字段 |
|---|---|---|
| `copy` | `copy`, `async_copy`, `tma_copy` | src/dst region、端点字节、scope、async/TMA/cache 注解 |
| `transform_copy` | `transpose`, `im2col`, `c2d_im2col` | src/dst region、变换参数、端点字节；不声称物理流量 |
| `gemm_dense` | `gemm`, `gemm_py` | M/N/K、transpose、A/B/C region、FMA/FLOPs、变体标记 |
| `gemm_sparse` | `gemm_sp` | 稠密等价尺寸、稀疏元数据 region、FLOPs 口径与稀疏假设 |
| `cuda_variant` | `wgmma_gemm`, `tcgen05_gemm`, `wgmma_gemm_sp`, `tcgen05_gemm_sp` | 无损保存、target mismatch；不作为 MUSA 适配通过项 |
| `fill` | `fill` | destination region、value、写字节 |
| `reduce` | `reduce` | reduction type、dim、src/dst region、combine count 表达式 |
| `scan` | `cumsum` | dim、reverse、src/dst region、元素数 |
| `reducer_finalize` | `finalize_reducer` | reducer buffer、destination、finalize 语义 |
| `atomic` | `atomicadd`, `atomicmax`, `atomicmin` | operation、destination、value、读改写字节、annotations |

`wgmma/tcgen05` 属于特定后端变体。若它们出现在 MUSA 输入 IR 中，工具仍保存其事实，但增加 `TARGET_VARIANT_MISMATCH`，不得把 CUDA 变体解释为 S5000 指令。

### 6.3 访问 region 提取优先级

1. 解析标准 `tl.tileop.region` 参数；
2. 使用 family adapter 的已知参数契约；
3. 解析可证明的 BufferLoad/access pointer；
4. 无法证明时输出 `unknown`，绝不猜测读写方向。

第一版继续保持外部工具实现，不依赖新增 TileLang FFI。若某个算子的 access region 只能由 C++ `TileOperator::GetAccessRegions()` 获得，再单独提出最小只读 FFI 变更，不在本方案中隐式修改 TileLang-MUSA。

### 6.4 Elementwise 边界

普通加减乘除、cast、exp、sqrt 等通常是普通 TIR 表达式，不属于 `tl.tileop.*`。它们不应伪装成 TileOp 适配完成。v0.2 可增加独立 `expression_ops` 计数器；如果暂不实现，报告必须明确：

```text
tileop_coverage=complete
expression_op_coverage=not_implemented
```

两类覆盖率分开计算。

## 7. 循环上下文与执行次数

### 7.1 遍历实现

用 `tvm.tirx.PyStmtExprVisitor` 替换无父上下文的 `post_order_visit`。visitor 进入/退出节点时维护：

- `statement_path`；
- `loop_stack`：loop var、min、extent、kind、annotations；
- `condition_stack`：if/predicate 表达式；
- `block_launch`：blockIdx.x/y/z extent；
- `thread_launch`：threadIdx.x/y/z extent。

每个 TileOp 在 `visit_call_` 中复制当前上下文，因此 fill、循环体 GEMM 和最终输出 copy 会获得不同 multiplicity。

### 7.2 执行次数语义

每条 action record 输出：

```json
{
  "static_occurrences": 1,
  "serial_multiplicity_per_block_expr": "4",
  "serial_multiplicity_per_block_value": 4,
  "block_count_expr": "2 * 2 * 1",
  "block_count_value": 4,
  "calls_per_block_expr": "4",
  "calls_per_kernel_expr": "16",
  "condition_expr": "true"
}
```

规则：

1. serial/pipelined 循环 extent 乘入每 block 调用次数；
2. blockIdx extent 只乘入 per-kernel，不乘入 per-block；
3. threadIdx extent不乘入高层 TileOp 次数，因为 TileOp 表示整个 tile/block 的集合语义；
4. 普通逐线程 TIR 表达式由独立 expression analyzer 处理；
5. symbolic extent 保留乘法表达式，存在配置 binding 时同时给出数值；
6. 条件执行保存 `indicator(predicate)` 或上下界，未知分支概率不能用 0.5 等假设；
7. pipeline stage 只改变重叠/调度，不改变语义调用次数，除非 IR 明确出现额外 prologue/epilogue 操作。

### 7.3 一致执行域指标

禁止继续使用含混的 `per_representative_tile_iteration`。统一输出：

```text
per_invocation
per_block
per_kernel_launch
```

每个域分别包含：

- FLOPs/FMA/SFU/reduction operations；
- logical read/write bytes，按 scope 分组；
- scope-to-scope endpoint bytes；
- arithmetic intensity。

Arithmetic intensity 必须明确分母：

```text
AI_all_logical = FLOPs / all_scope_logical_bytes
AI_global_logical = FLOPs / (global_read_bytes + global_write_bytes)
```

二者都是静态逻辑量，不是实测总线流量。

### 7.4 现有 pipelined GEMM 的验收真值

配置：`M=N=K=128`、`block_M=block_N=64`、`block_K=32`、grid=`2×2`。

| 指标 | per block | per kernel launch |
|---|---:|---:|
| FLOPs | 1,048,576 | 4,194,304 |
| logical read bytes（全部 scope） | 147,456 | 589,824 |
| logical write bytes（全部 scope） | 122,880 | 491,520 |
| logical total bytes | 270,336 | 1,081,344 |
| global logical bytes | 40,960 | 163,840 |
| AI all logical | 3.8787878788 | 3.8787878788 |
| AI global logical | 25.6 | 25.6 |

同时保留单次 GEMM 调用 `262,144 FLOPs`。上述数值用于回归测试，不能替换真实设备带宽或时延。

## 8. DAG 与复用分析调整

1. DAG 节点继续对应静态 action op，但附带 loop context 和 multiplicity；
2. 单迭代 data edge 与 loop-carried edge 分开保存；
3. buffer 名匹配只能标记 `may_depend_buffer_match`；region 可证明相交后才升级置信度；
4. opaque op 的 reads/writes 为 unknown 时，生成 `unknown_effect_barrier`，不得允许 DAG 越过该节点推导无依赖；
5. reuse distance 必须在同一执行域中展开或符号化，不能把静态 op ordinal 当作完整动态执行序列；
6. v0.2 若尚未完成循环展开 reuse，应输出 `reuse_status=partial`，不输出误导性容量曲线。

## 9. 完整性诊断和状态机

### 9.1 诊断级别

```text
error    阻止产物可信生成
blocker  产物可导出，但不得 complete
warning  不影响本离线 profile 的核心完整性
info     预期限制或 provenance
```

建议诊断码：

```text
MISSING_REQUIRED_SNAPSHOT
EMPTY_MUSA_SOURCE
UNREGISTERED_TILEOP_CALL
UNEXPORTED_TILEOP_CALL
UNSUPPORTED_TILEOP_ADAPTER
UNKNOWN_ACCESS_EFFECT
UNKNOWN_EXECUTION_MULTIPLICITY
MIXED_EXECUTION_DOMAIN
TARGET_VARIANT_MISMATCH
DAG_INVALID_ENDPOINT
DAG_CYCLE_IN_SINGLE_ITERATION
SCHEMA_VALIDATION_FAILED
EXPECTED_HARDWARE_FIELD_UNKNOWN
```

### 9.2 状态定义

| 状态 | 定义 |
|---|---|
| `failed` | lowering/导出失败、schema 无法成立或 IR 无法解析 |
| `partial` | 产物可审核，但存在 blocker，例如 opaque action op、未知访问、混合执行域或不完整关键快照 |
| `complete` | 当前 offline profile 的所有必需门槛通过；不表示存在运行期实测数据 |

### 9.3 complete 必需门槛

只有以下条件全部为真才允许 complete：

1. 必需快照 `S00/S10/S20/S30/S40/S50/S60` 全部存在，MUSA C 非空且 hash 一致；
2. `unexported_action_tileop_calls == 0`；
3. 所有 action TileOp 为 `structured` 或合法 `structural`，不存在 `opaque/invalid`；
4. 每个 action op 都有可用的数值或符号 execution multiplicity；
5. 所有聚合指标明确属于单一 execution domain；
6. DAG node/edge 引用合法，单迭代 DAG 无环；
7. JSON/JSONL 数据契约校验通过；
8. 没有 `error` 或 `blocker` 诊断。

以下内容是 offline profile 的预期 unknown，不阻止 complete：

- 物理寄存器数；
- S5000 真实 cache hit；
- S5000 有效带宽；
- occupancy；
- 绝对执行时延；
- Event/MUPTI/profiler 数据。

### 9.4 diagnostics.json 目标结构

```json
{
  "profile": "offline_static_v1",
  "status": "partial",
  "complete_gate": {
    "artifact_capture": true,
    "tileop_lossless_export": true,
    "tileop_semantic_coverage": false,
    "execution_domain_consistency": true,
    "dag_valid": true,
    "schema_valid": true
  },
  "coverage": {
    "observed_action_calls": 6,
    "exported_action_records": 6,
    "structured_records": 5,
    "opaque_records": 1,
    "unexported_action_calls": 0
  },
  "diagnostics": [
    {
      "severity": "blocker",
      "code": "UNSUPPORTED_TILEOP_ADAPTER",
      "op_id": "...",
      "message": "future_op was retained as opaque"
    }
  ],
  "expected_unknown_fields": [
    "S5000_cache_hit_rate",
    "absolute_latency_prediction"
  ]
}
```

## 10. 数据产物调整

新增：

```text
metadata/
  tileop_coverage.json
  execution_tree.json
  diagnostics.jsonl

metrics/
  workload_per_invocation.jsonl
  workload_per_block.json
  workload_per_kernel.json
```

保留 `metadata/tile_ops.jsonl`，但 schema 升级并增加：

- `raw_op_name`、`canonical_family`、`support_level`；
- `statement_path`、`loop_context`、`condition_context`；
- `calls_per_block_expr/value`；
- `calls_per_kernel_expr/value`；
- `adapter_name`、`adapter_version`；
- `diagnostic_codes`。

旧的 `metrics/workload_demand.json` 保留一个版本周期，只作为指向三个新执行域文件的兼容索引，不再保存混合口径数字。

## 11. 测试与验收矩阵

### 11.1 单元测试

| 测试 | 验收点 |
|---|---|
| TileOp 通用发现 | 依据 `TLOpBuilder` 发现，未来 unknown op 不丢失 |
| constructor 分类 | `region` 不成为 DAG action，但有忽略计数 |
| copy family | sync/async/TMA region、scope、annotations |
| transform copy | transpose/im2col 参数和端点字节 |
| GEMM family | dense/sparse/variant 的尺寸与 FLOPs 口径 |
| reduce/scan/finalize | region、dim、combine/scan 表达式 |
| atomic family | add/max/min 的 read-modify-write 语义 |
| nested loop | 正确生成乘法表达式和数值调用次数 |
| symbolic loop | 保留表达式，不伪造数值 |
| conditional op | 保存 predicate/indicator，不假定分支概率 |
| completeness gates | 任一 blocker 都不能输出 complete |

### 11.2 CPU-only 集成测试

1. 现有 pipelined GEMM：首个 MUSA lowering 确认用例，核对第 7.4 节全部真值；
2. Copy/Fill kernel：MUSA lowering 确认不同 scope 和循环外操作；
3. Reduce/Cumsum kernel：先确认 MUSA lowering，再核对结构化 region 和 operation count；
4. Atomic kernel：先确认 MUSA lowering，再核对 read-modify-write 与 partial/complete；
5. 含 unsupported TileOp 的 fixture：必须导出 opaque record、`unexported=0`、overall partial；
6. symbolic shape fixture：输出符号乘数并通过 schema；
7. 全部集成测试继续使用 `enable_device_compile=False`，不得分配或执行 MUSA tensor。

对于 MUSA 不支持的显式 CUDA variant，只做 metadata-only fixture，不要求也不允许其作为 MUSA lowering/运行通过证据。

### 11.3 回归失败条件

出现以下任一情况即验收失败：

- IR 中存在动作型 TileOp，但 `tile_ops.jsonl` 没有对应记录；
- unknown op 被输出为 `reads=[]/writes=[]`；
- 循环内外操作被标为相同调用次数；
- arithmetic intensity 的 FLOPs 和 bytes 来自不同 execution domain；
- 存在 opaque/未知访问而 diagnostics 仍为 complete；
- 报告把逻辑字节、容量假设或参数化上界描述为 S5000 实测值；
- 测试或 CLI 触发设备编译、tensor 分配或 kernel launch。

## 12. 实施顺序

### P0：契约先行（完成）

冻结新版 tile op、execution context、coverage 和 diagnostics schema；先写失败测试。  
验收：现有版本在新测试下能准确暴露漏算与混算问题。

### P1：无损发现与 opaque fallback（完成）

实现 `TLOpBuilder` discovery、constructor 分类、coverage invariant 和 unknown record。  
验收：所有动作型 TileOp 都有输出，`unexported=0`；unknown 导致 partial。

### P2：算子族适配（完成）

按 copy/transform/GEMM/reduction/atomic/fill 顺序实现 family adapter。  
验收：当前 20 种动作型 TileOp 均有明确 structured/structural 结果或可审核的 target mismatch；MUSA 适配通过率只统计可在 MUSA lowering 的 generic/target-compatible 算子。

### P3：执行上下文与工作量（完成）

实现 context-aware visitor、符号 multiplicity 和三个执行域的聚合。  
验收：pipelined GEMM 与第 7.4 节手算完全一致。

### P4：DAG/reuse 与完整性状态机（完成）

让 DAG、reuse 和报告消费执行上下文；建立 complete gates。  
验收：人为注入每种 blocker 时都稳定降级为 partial/failed。

### P5：独立审核（完成）

逐个算子 fixture、CPU-only 集成测试、产物 schema 审核和措辞审核。  
验收：全部测试通过后，才允许把 `offline_static_v1` 标为 complete。

## 13. 风险与控制

1. **TileLang 新增算子**：通用 discovery 自动捕获；没有 adapter 时 partial，不丢数据。
2. **后端专有变体误解**：保留 raw name 和 target applicability，不把 WGMMA/TCGEN 当作 MUSA 指令。
3. **符号/条件执行难以数值化**：输出表达式或上下界，不猜测概率。
4. **高层 TileOp 是 block collective**：threadIdx 不乘入 TileOp 次数，避免重复 128 倍。
5. **复用分析需要动态顺序**：未完成循环感知前降级，不沿用静态 ordinal 容量曲线。
6. **编译器 access region 信息只在 C++ 可见**：先解析标准 region 参数；确需 FFI 时另提只读接口审核。
7. **complete 被误读为真实测量完整**：状态始终带 `profile=offline_static_v1`，报告明确 runtime data 未采集。

## 14. 审核检查表

- [ ] 同意用 `TLOpBuilder` 属性发现全部 TileOp，取消动作白名单。
- [ ] 同意 unknown 动作型 TileOp 必须保留为 opaque，并使 overall 降级为 partial。
- [ ] 同意 `region` 只作为 constructor 计数，不进入 action DAG。
- [ ] 同意按 per-invocation/per-block/per-kernel 三个执行域输出指标。
- [ ] 同意 threadIdx 不乘入高层 TileOp 调用次数。
- [ ] 同意 current 20 种动作型 TileOp 按 family adapter 覆盖，后端变体带 applicability。
- [ ] 同意完整性采用多门槛状态机，不能只检查首尾快照。
- [ ] 同意循环感知 reuse 未完成时降级为 partial，不输出旧容量曲线。
- [ ] 同意 expected hardware unknown 不阻止 offline profile complete。
- [ ] 同意实施阶段继续禁止设备编译和 kernel 执行。

## 15. 建议审核决定

建议批准本方案，并按 P0→P5 顺序实施。P1 和 P3 是阻断项：在“动作 TileOp 无损导出”和“执行域一致聚合”通过前，现有结果最多标记为 `partial`。只有第 9.3 节全部 complete gates 通过，才允许输出：

```text
profile = offline_static_v1
status  = complete
```

该 complete 只表示编译期离线分析契约完整，不表示已经获得 S5000 运行期真实性能数据。
