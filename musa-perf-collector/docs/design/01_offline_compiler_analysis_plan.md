# TileLang-MUSA 编译期/离线性能分析实施方案（审核稿）

状态：**v0.3 offline_static_v1 已实施；已纳入 P6.9.2 全链路交付审核**  
版本：offline-compiler-analysis-v0.3  
日期：2026-07-30  
工具根目录：`/root/musa-perf-collector`  
参考源码：`/root/s5000-calibration/tilelang-musa`

当前完整交付状态、S5000 证据、打包审核和能力边界见：
[`00_current_implementation_and_acceptance_status.md`](./00_current_implementation_and_acceptance_status.md)。

后续完整性审核与实施门槛见：[`02_tileop_coverage_execution_completeness_plan.md`](./02_tileop_coverage_execution_completeness_plan.md)。该补充方案取代本文中“仅凭首尾快照即可 complete”的旧判断。

## 1. 审核结论摘要

本方案可在 CPU 上实现，不执行 MUSA kernel，也不依赖 S5000 在线占用。第一版采用 TileLang-MUSA 已有的 Python JIT/Lowering 接口和 TVM `PassInstrument`，在独立工具仓中完成 IR 快照、tile/pipeline 元数据提取、静态工作量分析、访问/复用分析、MUSA C 导出和离线报告生成。

第一版不修改 DeepStack，不建立 S5000 架构 Profile，也不把 NVIDIA/AMD 参数作为回退。没有经过真实设备校准时，不输出绝对执行时间；只输出程序静态事实、带来源的派生量，以及带显式变量的性能上界公式。

已按“数据契约 → IR 捕获 → 元数据提取 → 静态分析 → 报告 → 验证”实现核心链路，并按补充方案完成 TileOp 覆盖、循环执行域与 complete gate，实现了 pipelined GEMM 的 CPU-only 集成验证。Attention、MoE 与运行期采集仍不在本版范围。

## 1.1 v0 实施/审核记录（2026-07-29）

已实现：显式 `musa/mp_31` pure-codegen、PassInstrument 快照、输入 Tile IR 的 Copy/GEMM/Fill 元数据、buffer/loop/pipeline 注解、保守 DAG、FLOPs/逻辑字节/假设容量扫描/参数化上界、`.mu`、JSON/JSONL、Markdown/HTML 报告与 CPU-only 集成测试。

已审查并修正：TileLang `region` 参数构造器误计为可执行 op、向量 dtype 字节宽度解析、float32 fragment→float16 global 转换 copy 的单一搬运字节误报、同一 op 内 accumulator 的零距离复用、以及非确定性 analysis ID。DAG 的 buffer-name 依赖现明确标作 `may_depend_buffer_match`，不再声称精确 region 依赖。

明确延期：MCC compile-only、带 compiler provenance 的 exact 跨 pass op lineage、pipeline 的 prologue/steady/epilogue 推导、Reduce/Atomic/elementwise/SFU 的结构化计数、复杂 layout 结构化解析，以及本方案以外的 runtime Event/trace/MUPTI 采集。当前已导出可审核的 lowered structural candidates，但它们不升级为 exact lineage。

## 2. 目标与非目标

### 2.1 第一阶段目标

1. 从 TileLang `PrimFunc` 或 `IRModule` 离线构建编译会话。
2. 捕获关键 lowering 阶段的 Tile IR/TIR 快照。
3. 提取 tile op、buffer、pipeline、layout、循环、线程绑定、launch 结构和依赖 DAG。
4. 计算 FLOPs、逻辑读写字节、内存层级搬运、静态 footprint、复用距离和容量扫描下的潜在缓存流量。
5. 导出生成的 MUSA C、目标配置、pass 配置、编译参数和源码/产物哈希。
6. 生成机器可读 JSON/JSONL 与供人工审核的 Markdown/HTML 静态报告。
7. 保证所有结论具有来源和证据等级，无法确定的字段明确输出 `unknown` 或符号表达式。

### 2.2 本阶段不做

1. 不运行 kernel，不分配 MUSA tensor，不调用 `torch.musa.Event`。
2. 不采集 MUPTI、`torch.profiler` 或 `mthreads-gmi` 数据。
3. 不推断 S5000 峰值算力、有效带宽、缓存容量、寄存器容量或 occupancy 参数。
4. 不给出未经真实数据校准的绝对时延、TFLOPS 利用率或缓存命中率。
5. 不承诺把每个 pass 前后的 op 做成百分之百精确的跨 pass 一一映射；无法证明的映射必须带置信度。
6. 不在第一版修改 TileLang-MUSA 编译器源码；若外部插件无法获得必要字段，再单独提交最小上游改动提案审核。

## 3. 已确认的 TileLang-MUSA 接口基础

当前 TileLang-MUSA 已具备以下可复用接口：

1. `TorchFunction`/JIT 接口接受 `pass_instruments`，并在 `PassContext` 中运行。
2. TVM 提供 `PassInstrument.run_before_pass`、`run_after_pass` 和 `DumpIR`。
3. 当前 lowering 顺序包含：
   - `PipelinePlanning`
   - `InjectSoftwarePipeline`
   - `LayoutInference`
   - `LowerTileOp`
   - target-specific optimization
   - `SplitHostDevice`
   - `LowerDeviceKernelLaunch`
4. `tilelang.engine.lower.lower_to_host_device_ir(...)` 可得到 host/device IR。
5. `tilelang.lower(..., enable_device_compile=False)` 通过 `device_codegen_without_compile(...)` 生成 MUSA C，不需要执行设备 kernel。
6. `CompiledArtifact` 包含 `host_mod`、`device_mod`、参数和 `kernel_source`。
7. 若需要 CPU 编译产物，TileLang-MUSA 支持 `musa.kernels_output_dir`、`TILELANG_DUMP_KERNEL_SOURCE` 和 MCC compile flags；该模式只编译，不运行 GPU。

因此第一版优先采用外部 `PassInstrument + offline lower`，避免侵入 TileLang-MUSA 主线。

## 4. 总体架构

```text
PrimFunc / IRModule factory
        │
        ▼
Offline Session（显式 target=musa, arch=mp_31）
        │
        ├── Pass Capture ──► 多阶段 IR 快照
        │
        ├── Metadata Extractor
        │      ├── tile ops / buffers / loops
        │      ├── pipeline / layout / bindings
        │      └── dependency DAG
        │
        ├── Static Analyzer
        │      ├── FLOPs / logical bytes
        │      ├── scope-to-scope transfers
        │      ├── footprint / reuse distance
        │      └── parameterized upper bounds
        │
        ├── MUSA Codegen ──► .mu + target/flags/hash
        │
        └── Exporter ──────► JSON/JSONL + Markdown/HTML
```

模块放置规划：

```text
src/adapters/tilelang/
  offline_session.py        # 输入契约、显式 target、lower 调用
  pass_capture.py           # PassInstrument 与关键快照选择
  codegen_capture.py        # MUSA C、target、flags、hash

src/collectors/compiler/
  tile_ops.py               # Copy/Gemm/Reduce/Atomic 等
  buffers.py                # shape/dtype/scope/region/allocation
  pipeline.py               # stage、prologue/steady/epilogue 信息
  layout.py                 # layout、swizzle、向量化与线程映射
  dependencies.py           # data/control/sync/loop-carried DAG
  workload.py               # FLOPs 与逻辑字节
  reuse.py                  # 访问序列、复用距离、容量扫描
  bounds.py                 # 参数化性能上界表达式

src/core/
  ids.py                    # analysis/kernel/snapshot/op/buffer ID
  evidence.py               # 证据等级与来源
  diagnostics.py            # unsupported/unknown/error

src/exporters/
  json_exporter.py
  report_exporter.py
```

以上为实施后的目标文件，不在本审核稿阶段创建代码文件。

## 5. 输入契约

### 5.1 推荐输入

每个离线 workload 提供纯构建函数：

```text
build_prim_func(config) -> PrimFunc | IRModule
```

约束：

1. 只构建 TileLang IR，不创建 MUSA tensor。
2. 不调用 JIT kernel，不执行 correctness 或 benchmark。
3. shape、dtype、tile 参数和 pass config 必须来自显式 JSON/YAML 配置。
4. 动态维度保留为符号表达式；需要数值分析时由配置提供绑定值。
5. target 必须显式指定为 MUSA，S5000 当前编译目标写为 `mp_31`；禁止 `target=auto`，避免隐式设备查询。

### 5.2 不直接接受完整 benchmark 脚本

完整 benchmark 通常混合了 IR 构建、设备分配、正确性和计时，无法证明其离线安全。第一版不直接执行任意 benchmark 文件，而是为目标 kernel 增加轻量的 `PrimFunc factory` wrapper。这样可以从接口上保证“分析”和“运行”分离。

## 6. 编译阶段与快照设计

### 6.1 必须保存的快照

| 快照 | 获取时点 | 主要用途 |
|---|---|---|
| `S00_INPUT_TILE_IR` | lowering 前 | 高层 tile op、源码语义、原始 buffer |
| `S10_PIPELINE_PLANNED` | `PipelinePlanning` 后 | stage 规划、async copy 选择候选 |
| `S20_PIPELINE_INJECTED` | `InjectSoftwarePipeline` 后 | prologue/steady/epilogue、buffer versioning、同步 |
| `S30_LAYOUT_INFERRED` | `LayoutInference` 后 | fragment/shared layout、线程映射 |
| `S40_TILE_OP_LOWERED` | `LowerTileOp` 后 | 高层 op 到低层循环/指令的展开 |
| `S50_HOST_DEVICE_SPLIT` | `SplitHostDevice` 后 | kernel 边界、device function、launch 参数 |
| `S60_FINAL_DEVICE_IR` | device codegen 前 | 最终线程绑定、存储和调用结构 |
| `S70_MUSA_C` | codegen 后 | 生成的 MUSA C 和代码哈希 |

### 6.2 Pass 名称兼容策略

不同 TileLang commit 的 pass 名可能变化。工具先记录本次实际出现的完整 pass 序列，再通过“精确名称映射表 + 必备阶段校验”选择快照。若关键阶段缺失，不静默回退，分析状态应为 `incomplete_snapshot_set`，报告列出缺失 pass 和当前 commit。

### 6.3 跨 pass lineage

不同 pass 可能复制、合并或删除节点，因此 op ID 分两层：

1. `snapshot_op_id`：由快照、结构遍历路径、op 类型和 structural hash 决定，在单快照内稳定。
2. `semantic_op_id`：由源码 span、op 类型、输入输出 region、shape/dtype 和语义签名匹配得到，用于跨快照关联。

跨 pass 映射输出 `exact`、`high_confidence`、`heuristic` 或 `unmapped`。只有存在稳定源码/IR 标记时才标记 `exact`，禁止把启发式关联伪装成确定事实。

## 7. 元数据提取定义

### 7.1 Kernel 元数据

- global symbol、参数列表、target、arch；
- shape/dtype 配置与符号绑定；
- grid/block/thread binding；
- pass config、TileLang commit、工具 schema 版本；
- input IR、final IR、MUSA C 的 SHA-256；
- 是否进行了 MCC compile-only；
- 明确的 `device_execution=false`。

### 7.2 Tile op 元数据

v0 已覆盖：

- `T.copy`：源/目标 buffer region、scope、元素数和端点逻辑字节；cache/async 注解结构化解析为延期项；
- `T.gemm`：M/N/K、operand region/scope 和 FMA/FLOPs；
- `T.fill`/`T.clear`：目标 region；

以下属于后续覆盖项，当前保留为 `unknown` 或 deferred，而不猜测：
- `T.reduce`：归约类型、维度、输入输出规模和并行方式；
- `T.atomic`：操作类型、目标 scope、访问规模；
- elementwise/SFU：算术、比较、cast、exp、sqrt、div 等表达式计数；
- pipeline/sync action：stage、barrier、wait、commit、依赖关系。

每条 v0 元数据记录均记录 `source_snapshot=S00_INPUT_TILE_IR` 和 `evidence_level`；细粒度跨快照字段 lineage 为延期项。

### 7.3 Buffer 与访问元数据

- buffer 名称、shape、dtype、scope；
- allocation bytes（数值或符号表达式）；
- 读/写 region、索引表达式；向量宽度、对齐为延期项；
- global/shared/local/fragment 等层级间搬运方向；
- allocation footprint 与按 op 计的 logical bytes；unique footprint、双缓冲/多版本关系为延期项。

### 7.4 Pipeline 元数据

- loop variable、extent、`num_stages`；
- prologue、steady、epilogue 的结构为延期项；
- stage 内 op 顺序与 stage 间依赖；
- async copy、commit、wait、barrier；
- 可重叠候选与必须串行的边；
- tail iteration/wave 只记录结构需求，不在无硬件参数时计算真实 tail wave 时延。

### 7.5 Layout 与线程绑定

- block/thread binding 和循环 extent；
- fragment/shared layout、swizzle/vectorize/unroll、坐标映射和 layout structural hash 均为延期项。

## 8. 依赖 DAG 构建

DAG 节点为 tile op 或 lowering 后的重要 action，边分为：

1. `data`：producer 写 region 与 consumer 读 region 相交；
2. `control`：IR 顺序或条件控制要求串行；
3. `sync`：barrier/wait/commit 明确建立的先后关系；
4. `loop_carried`：跨迭代 buffer 或标量依赖；
5. `possible_overlap`：无数据/同步冲突且 pipeline 允许重叠。

Region 相交无法证明时标为 `may_depend`，不能直接作为确定依赖。DAG 必须通过无环检查；loop-carried edge 作为单独关系保存，不混入单迭代 DAG 导致伪环。

## 9. 静态计算规则

### 9.1 FLOPs

- GEMM 默认按一次 FMA = 2 FLOPs，`2*M*N*K`，同时保存 FMA count，避免口径争议；
- elementwise 按 IR 算术节点数乘迭代域；
- exp/sqrt/div 等另存 `sfu_ops`，不强行等价为普通 FLOPs；
- reduce 保存输入元素数、combine 次数和归约类型；
- symbolic shape 输出可求值表达式和绑定后的数值，两者同时保存。

### 9.2 逻辑读写字节

每个访问至少保存：

- `unique_bytes`：本 tile 覆盖的唯一数据 footprint；
- `issued_logical_bytes`：按循环和调用次数展开的逻辑访问量；
- `src_scope -> dst_scope`；
- read/write/atomic；
- 是否可能被 cache 或片上复用吸收。

逻辑字节不是硬件总线实测字节，证据等级固定为 `semantic_derived`。

### 9.3 静态存储 footprint

输出 global 参数 footprint、shared allocation、local/fragment 元素规模、pipeline 多版本放大系数。寄存器真实分配量无法仅由高层 IR准确得到时输出 `unknown`，不得把 local 元素数直接称为物理寄存器数量。

## 10. 复用距离与潜在缓存流量

### 10.1 复用键

访问按 `(buffer_id, tile_coordinate, region_shape, dtype)` 构建 reuse key。执行顺序由 loop order、thread/block order 和可见 swizzle 推导；无法确定全局执行顺序时，只计算线程块内或符号化访问序列。

### 10.2 输出内容

- reuse count；
- first-use/next-use distance；
- 以 distinct tile bytes 表示的 reuse distance；
- 按 tensor、pipeline stage、scope 分组的距离分布；
- compulsory traffic；
- 对一组**假设有效容量**的 potential traffic 曲线。

### 10.3 真实性边界

在没有 S5000 cache 容量、组相联、替换策略和真实执行顺序校准时，不输出“S5000 L2 命中率”。容量扫描结果命名为：

```text
potential_cache_traffic_under_capacity_assumption
```

报告必须展示容量是自变量而非已知硬件参数，例如 `traffic(C_l2)`，并标为 `model_derived_unvalidated`。

## 11. 性能上界候选

本阶段只输出参数化上界，不输出绝对预测时延：

```text
T_compute >= FLOPs / P_compute_effective
T_global  >= global_bytes / BW_global_effective
T_shared  >= shared_bytes / BW_shared_effective
T_stage   >= max(T_compute, T_global, T_shared)   # 仅对允许重叠的资源
T_stage   >= sum(serial_components)               # 必须串行部分
```

输出包含：

- workload demand vector；
- critical-path 静态候选；
- arithmetic intensity；
- 不同带宽/吞吐参数下的敏感性公式；
- 所有未知硬件变量列表。

只有后续真实采集得到并冻结 S5000 有效参数后，才能在交叉校准阶段生成绝对时间候选。

## 12. MUSA C 与编译参数导出

### 12.1 默认 pure-codegen 模式

- 调用 `tilelang.lower(..., target=显式MUSA目标, enable_device_compile=False)`；
- 保存 `kernel_source` 为 `.mu`；
- 保存 target、arch、pass config、compile flags、TileLang/MCC 版本；
- 不调用 MCC，不产生设备二进制；
- 保证 `device_execution=false`。

### 12.2 可选 compile-only 模式

用于审核实际 MCC 命令和静态二进制产物：

- 允许 MCC 在 CPU 上编译 `.mu`；
- 保存 `.cmd`、`.mubin`、可选 `.s` 和哈希；
- 仍不加载或运行 kernel；
- compile-only 必须通过显式 CLI 开关启用，不作为默认行为。

## 13. 输出目录与数据契约

```text
results/processed/offline/<analysis_id>/
  manifest.json
  diagnostics.json
  ir/
    S00_INPUT_TILE_IR.tir
    S10_PIPELINE_PLANNED.tir
    S20_PIPELINE_INJECTED.tir
    S30_LAYOUT_INFERRED.tir
    S40_TILE_OP_LOWERED.tir
    S50_HOST_DEVICE_SPLIT.tir
    S60_FINAL_DEVICE_IR.tir
  metadata/
    kernel.json
    tile_ops.jsonl
    buffers.jsonl
    pipeline.json
    dag.json
    lineage.json
  metrics/
    workload_demand.json
    reuse_distance.json
    upper_bound_formula.json
  generated/
    kernel.mu
    compile_config.json
    kernel.cmd            # 仅 compile-only
    kernel.mubin          # 仅 compile-only
  report/
    report.md
    report.html
```

`manifest.json`、`diagnostics.json` 和记录级 JSON 包含所需的 schema/provenance/evidence 字段；JSONL 中每条 record 均包含 `evidence_level`。v0 尚未把顶级 manifest 重复写入每一个派生文件，避免制造互相不一致的元数据副本。

完整目标契约包括：

- `schema_version`；
- `analysis_id`、`kernel_id`；
- `tilelang_commit`、工具 commit；
- 输入配置和文件 SHA-256；
- 字段级或记录级 `evidence_level`；
- `unknown_fields` 与诊断信息。

## 14. 证据等级

| 等级 | 含义 | 示例 |
|---|---|---|
| `compiler_observed` | 直接来自某阶段 IR/编译产物 | buffer scope、thread binding、MUSA C |
| `semantic_derived` | 按明确规则由 IR 推导 | GEMM FLOPs、逻辑字节 |
| `model_derived_unvalidated` | 依赖假设、尚未硬件校准 | 容量扫描流量、重叠上界公式 |
| `heuristic_mapping` | 跨 pass 启发式关联 | LowerTileOp 前后 lineage |
| `unknown` | 当前证据不足 | 真实寄存器数、S5000 cache hit |

报告禁止将后四类内容描述为设备直接实测。

## 15. 命令行目标形态

v0 已实现接口（target 固定为显式 MUSA，无 `target=auto`）：

```bash
python -m musa_perf_collector offline analyze \
  --factory musa_perf_collector.examples.pipelined_gemm:build_prim_func \
  --config /root/musa-perf-collector/configs/workloads/pipelined_gemm.json \
  --arch mp_31 \
  --output results/processed/offline
```

可选 CPU compile-only（**延期，v0 CLI 尚未提供此开关**）：

```bash
python -m musa_perf_collector offline analyze \
  --factory workloads.tilelang.gemm.fp16_nn:build_prim_func \
  --config configs/workloads/gemm_fp16_nn.json \
  --target musa --arch mp_31 \
  --compile-only \
  --output results/processed/offline
```

命令运行成功不代表所有扩展分析均已实现。v0 CLI 的 `diagnostics.json` 区分：

- `complete`：v0 必需快照和核心字段齐全；
- `partial`：可输出，但存在 unsupported/unknown；
- `failed`：输入、lowering 或 schema 校验失败。

## 16. 实施阶段与验收门槛

### P0：数据契约冻结（核心完成）

工作：定义 manifest、tile op、buffer、pipeline、DAG、metrics schema。  
验收：样例 JSON 由集成测试中的契约断言覆盖；证据等级和 unknown 规则审核通过。独立 JSON Schema 文件为后续增强。  
预计：1–2 人日。

### P1：离线会话与 IR 捕获（完成）

工作：实现 factory 输入、显式 MUSA target、PassInstrument、关键快照和 pure codegen。  
验收：无 GPU kernel 执行；GEMM 能导出全部必需快照和 MUSA C；记录完整 pass 序列。  
预计：2–3 人日。

### P2：元数据与 DAG（核心完成；扩展项延期）

工作：提取 tile ops、buffers、pipeline、layout、thread binding、依赖和 lineage。  
验收：对 elementwise、GEMM、pipelined GEMM 人工逐项核对；DAG 无伪环；不确定依赖有明确标记。  
预计：4–5 人日。

### P3：静态工作量与复用分析（核心完成；扩展项延期）

工作：FLOPs、逻辑字节、footprint、reuse distance、容量扫描和参数化上界。  
验收：手算小例与工具一致；symbolic shape 保留表达式；报告不出现伪造 S5000 参数。  
预计：4–5 人日。

### P4：报告与导出（完成）

工作：JSON/JSONL、Markdown/HTML、诊断摘要、hash/provenance。  
验收：同输入同 commit 输出确定；报告可追溯到具体 IR 快照和 op。  
预计：2–3 人日。

### P5：独立验证（pipelined GEMM CPU-only 完成）

工作：回归测试、三类 kernel、CPU-only 环境测试、跨 commit 兼容诊断。  
验收：在无可用 MUSA 设备的环境仍可 pure-codegen；执行日志无 kernel launch；产物契约断言通过。  
预计：2–3 人日。

总计预计：15–21 人日。复杂 Attention/MoE 不计入第一版门槛。

## 17. 第一版验证矩阵

| 用例 | 验证重点 | 必须通过 |
|---|---|---|
| Elementwise add | 基础循环、读写字节、线程绑定 | 延期 |
| FP16 GEMM NN | Gemm FLOPs、A/B/C scope、MUSA C | 由 pipelined GEMM 覆盖核心部分 |
| Global→Shared copy | scope 搬运、逻辑字节、layout | 由 pipelined GEMM 覆盖；layout 结构化延期 |
| Pipelined GEMM | stage、prologue/steady/epilogue、async/sync | Pipeline 注解和核心 DAG 已通过；时序结构延期 |
| Reduce | reduce 维度、combine count、DAG | 延期 |
| Symbolic shape | 表达式保存、绑定求值、诊断 | 是 |
| FlashAttention 子集 | 多 stage DAG 与 SFU 分类 | 扩展项 |

## 18. 风险与应对

1. **Pass 名变化**：记录完整 pass 序列，版本化名称映射；缺失时 partial/fail，不静默猜测。
2. **高层 op 在早期被重写**：保存 input、planning、injected、layout 四个高层快照，不能只分析 final TIR。
3. **跨 pass ID 不稳定**：采用 snapshot ID + semantic lineage + confidence，不承诺虚假的 exact mapping。
4. **动态 shape 无法静态求值**：保留符号表达式，另存 bindings；未绑定不强制数值化。
5. **缓存流量容易被误读**：输出容量扫描和假设，不输出 S5000 命中率。
6. **编译脚本意外访问设备**：禁止 `target=auto` 和 benchmark 入口；pure factory；在验证中检查无 kernel launch/设备分配。
7. **MUSA C 与 IR 对应困难**：第一版按 kernel 关联；细到代码行的 source map 作为后续增强，不阻塞基础交付。

## 19. 审核检查表（原实施前检查，保留供后续扩展复核）

请在进入实现前确认以下决策：

- [ ] 同意第一版完全独立于 DeepStack 和 S5000 架构 Profile。
- [ ] 同意默认 pure-codegen，不调用 MCC；compile-only 作为显式可选模式。
- [ ] 同意第一版输入为 `PrimFunc/IRModule factory`，不直接执行完整 benchmark。
- [ ] 同意缓存分析只输出容量扫描，不声称真实 cache hit rate。
- [ ] 同意无硬件参数时只输出参数化性能上界，不输出绝对时延。
- [ ] 同意跨 pass lineage 带置信度，不强求不可靠的一一映射。
- [ ] 同意第一版验收范围为 elementwise、GEMM、copy、pipeline、reduce、symbolic shape。

## 20. 建议审核决定

建议按以下条件批准实施：

1. 先审核并冻结第 5、6、7、9、10、13、14 节的数据语义。
2. P0 schema 通过单独审核后才进入 P1 编码。
3. P1 先证明 CPU-only、无 kernel launch，再进入静态模型开发。
4. 任何需要修改 TileLang-MUSA 主仓的需求，必须另开变更说明，不在外部工具中隐式打补丁。
5. 运行期 Event、trace、MUPTI 继续留在后续阶段，不与本方案同时实现。
