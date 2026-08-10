# P7.1–P9：S5000 数据补全、TileSight 接入与独立验证方案

状态：执行中；P7.1 已完成，P7.2 部分通过，P7.3 direct-batch pilot 已完成但需 P7.3.1（详细结果见文档 21–24）
日期：2026-07-29（UTC+8）
范围：单卡 MTT S5000 / TileLang-MUSA；不含多卡 collective 和 NoC 校准

## 1. 目标与路线

目标不是尽快得到一个“看起来合理”的 S5000 性能模型，而是依次闭合 runtime 稳定性、计时可信度、
跨 shape/schedule 数据、memory/resource/wave 数据、TileSight 适配和真正独立验证。

正式顺序：

```text
P7.1 设备恢复门禁与 runtime A/B
    ↓
P7.2 短 kernel 低固定开销计时协议
    ↓
P7.3 GEMM shape/schedule development 数据集
    ↓
P7.4 cache/occupancy/tail 数据补全
    ↓
P8 S5000 TileSight guarded adapter
    ↓
P9 全新 hold-out 与 schedule-ranking 验证
```

P8 的目录骨架可以提前静态设计，但参数激活、预测结论和版本冻结必须等 P7.1–P7.4 完成；P9 case
必须在 P8 模型冻结后才允许编译和测量。

## 2. 全阶段不可违反的规则

1. 每个 case 依次执行：编译 → 单次运行 → 数值正确性 → 稳定性 → 性能；前一步失败即停止。
2. 每个正式数据点至少三个全新进程；不得从失败进程中挑成功 sample 拼成一次“通过”。
3. development、hold-out、diagnostic 三类数据目录和 config 必须隔离。
4. P6/P7 已经观察过的 case 不得作为 P9 独立 hold-out。
5. 每个 raw run 保存 config、harness 源码快照及 SHA、TileLang commit/local patch SHA、生成 `.mu`、
   mcc 命令、MUBIN（若查询资源）、日志、样本、环境与 GPU 快照。
6. 失败状态至少区分 `compile_failed`、`correctness_failed`、`runtime_failed`、
   `measurement_rejected`，并记录 `failure_stage`。
7. MUSA Event 与 torch Profiler 只能称为两条计时接口交叉检查，不能称为两个独立测量系统。
8. NVIDIA/AMD 数值、NCU 指标和 DeepStack 既有 arch 参数不得作为 S5000 缺失值回退。
9. 任何证据不足的参数返回 `unknown`；不得用 hold-out 反向拟合后仍称该批数据为 hold-out。
10. 当前设备状态未恢复前不启动新 kernel；重启、reset 或其他设备恢复动作需用户明确授权。

## 3. P7.1：设备恢复门禁与 runtime A/B

### 3.1 目标

定位 `1536×1024×1536` FP16 GEMM 高频重复运行异常属于哪条路径，并得到一个可以进入后续性能
测量的稳定执行协议。本阶段首先是正确性/runtime 诊断，不拟合吞吐。

### 3.2 恢复前门禁

只做只读检查：

- 无残留 P2/P7 Python/controller 进程；
- `mthreads-gmi` 能连续三次正常查询设备和进程表；
- 设备内存/利用率回到空闲状态；
- 全新进程中的最小 MUSA runtime smoke 通过；
- 全新进程中的最小 TileLang elementwise kernel 编译、正确性、单次同步通过。

任一项失败则停止；不继续大 GEMM，不自动 reset/reboot。

### 3.3 A/B 变体

固定数学问题、dtype、shape、tile、输出和随机种子，只改变待隔离的生成路径：

| ID | TMA/TME | warp specialization | descriptor prefetch | 用途 |
|---|---|---|---|---|
| A | 保留 | 保留 | 开 | 当前 baseline |
| B | 保留 | 保留 | 关 | 只隔离 descriptor prefetch |
| C | 保留 | 关 | 与 A 一致 | 隔离 producer/consumer specialization |
| D | 关；同步 global→shared copy | 与 A 一致或最接近的可实现形式 | 不适用 | 隔离 TMA/barrier/descriptor |
| E | 关 | 关 | 不适用 | 最简同步对照 |

可实施性限制：当前确认存在的 pass flag `TL_ENABLE_MUSA_TMA_PREFETCH` 只控制 descriptor prefetch，
不等于关闭 TMA。TileLang-MUSA 已提供 `TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"`，
但源码提示 TMA pipeline 可能依赖 warp specialization 保证正确性，因此 C 必须先过 lowered TIR、
生成 `.mu` 和数值审核。对 D/E，当前推荐 `T.copy(..., disable_tma=True)` 逐 copy 禁用 TMA；仓库中
仍有废弃兼容 flag `TL_DISABLE_TMA_LOWER`，但不作为新协议依赖。若生成源码仍包含
`MUtensorDescriptor`、TME load 或 producer/consumer 分支，就不能把该 case 标成对应路径已关闭。

### 3.4 每个变体的执行协议

1. 新进程编译，保存 `.mu`、mcc 命令和 binary；
2. 运行一次并同步，做独立 reference correctness；
3. warmup 采用很小且固定的次数；
4. graph-free、无 Profiler，按 batch-unroll 1/2/4/8/16/32 逐级执行；每批同步；
5. 每一级保存完成批数，首次 runtime error 立即结束该进程；
6. 三个全新进程全部完成后，才允许单独增加 MUSAGraph 对照；
7. Profiler 放在最后的独立进程，不与主要 Event timing 共享进程。

### 3.5 判定矩阵

- D/E 3/3 稳定而 A/B/C 失败：优先进入 TMA/barrier/descriptor 路径；
- B 稳定而 A 失败：descriptor prefetch 高优先级，但仍需重复确认；
- C 稳定而 A/B 失败：specialization/barrier protocol 高优先级；
- A–E 全失败：优先检查 adapter、argument packing、driver/runtime 或测试协议；
- A–E 全稳定：设备/进程状态或先前异步错误污染优先，仍不能声称问题自然消失。

### 3.6 通过门槛与交付物

至少一个数学等价路径满足：三个新进程正确性全通过、受控重复 3/3 完成、无 allocator/profiler
收尾错误；再在一个独立大 GEMM shape 上交叉验证 3/3。输出最小复现、A/B 表、生成源码 diff、失败
stage 和 vendor issue 材料。任何成功延迟在计时协议冻结前仍标记 diagnostic。

## 4. P7.2：短 kernel 计时协议修复

### 4.1 目标

为约 2–15 µs 的 kernel 分离 host/runtime/Event/graph 固定开销，决定 FP16 256³、512³及 1M
elementwise/SFU 是否能形成稳定参数。

### 4.2 校准对象

- 空 graph replay；
- 最小 no-op/单写 kernel；
- FP16 GEMM 256³、512³；
- vector-add、exp、rsqrt 1M；
- 一个 20–50 µs 中等 kernel 作为长短边界对照。

### 4.3 候选计时方法

| 方法 | 扫描 |
|---|---|
| graph-free MUSA Event | batch-unroll 1/2/4/8/16/32/64 |
| MUSAGraph Event | graph-unroll 1/2/4/8/16/32/64/100 |
| torch Profiler | 独立进程，只做 device-time 交叉检查 |
| host wall time | 仅检查提交开销，不作为 kernel latency |

对每个方法拟合 `batch_time = fixed_cost + unroll * kernel_time`，报告截距、斜率、残差和不同 unroll
下的稳定性。只有斜率在独立进程一致，且空载校正不会产生负值或强依赖选取区间，才可冻结。

### 4.4 建议验收门槛

- 三进程 correctness 3/3；
- 每进程不少于 30 个 timing samples；
- 预注册的有效 unroll 区间内，斜率跨进程跨度 ≤5%；
- Event 斜率与独立 Profiler device time 差异 ≤10%；
- 固定开销校正后结果不随 unroll 呈系统漂移；
- 任一门槛失败则该 kernel 继续 `pending_short_kernel_method`。

门槛必须在观察新正式数据前冻结；若 SDK/runtime 限制使其不现实，只能新建协议版本，不能事后放宽
当前版本。

### 4.5 必须修复的 harness 问题

新建 versioned harness，不覆盖当前 P2 文件：

- `timing_method` 动态记录真实 `graph_unroll/batch_unroll`；
- 分离 correctness、runtime、measurement gate 状态；
- graph capture、replay、sync、Profiler enter/exit 分别记录 failure stage；
- 每个 raw run 内保存 harness 文件本体，而不只保存 SHA；
- 不使用已缺失源码快照的旧 P2 harness 作为逐字节复现基线。

## 5. P7.3：GEMM shape/schedule development 数据集

### 5.1 目标

建立可用于 TileSight adapter 的跨 shape、tile、warp、pipeline 特征数据，同时保留完全独立的 P9
验证空间。此阶段是 development，不报告泛化精度。

### 5.2 数据分层

在任何新 kernel 编译前，固定三个清单：

- `development`: 本阶段允许建模的数据；
- `diagnostic`: 编译/runtime 问题缩小数据，不建模；
- `reserved_p9_space`: 只登记选择规则和范围，不公开具体最终 case；P8 冻结后由固定种子生成。

### 5.3 Shape 轴

每种已稳定 dtype 至少覆盖：

- square：512、768、1024、1536、2048 中不属于 P9 的点；
- skinny/tall：M 或 N 较小，另一维较大；
- small-K、large-K；
- 对 tile 整除与非整除边界；
- grid 小于 1 wave、接近 1 wave、多完整 wave、带 tail。

FP16 为第一优先；BF16、INT8 只有在各自稳定协议通过后加入。已观察的 P2/P5/P6/P7 shape 可以作为
development 或 retrospective，不能成为 P9 hold-out。

### 5.4 Schedule 轴

同一 shape 至少扫描 4 个实际可编译候选，覆盖：

- `block_M/block_N/block_K`；
- 线程/warp 数；
- pipeline stage；
- warp policy/layout；
- TMA/prefetch/specialization（仅对可验证等价变体）；
- dynamic shared 与 registers/resource 差异。

必须保留失败候选。不能只保存最快 kernel，否则无法训练或验证 schedule ranking。

### 5.5 每行数据契约

至少包含：case/schedule ID、development/diagnostic 标签、shape、dtype/accum、tile、warp/thread、stage、
layout/policy、grid CTA、source/MUBIN SHA、TMA/barrier/specialization 特征、registers、static/dynamic
shared、API active CTA/MP、correctness、failure stage、每进程 samples、latency 分位数、有效吞吐、
SDK/compiler/date、raw/log 路径。

### 5.6 验收

- 每个进入 development 的点 3/3 compile/correct/measured；
- 同一 shape 至少 4 个有效 schedule，或明确记录为什么无法达到；
- 失败率、失败 stage 和不支持组合单独报告；
- 不在本阶段拟合或查看 P9 数据；
- 数据集冻结为版本化 manifest 和 SHA。

## 6. P7.4：cache、occupancy 与 tail 补测

### 6.1 Cache

围绕 runtime 报告的 60 MiB L2 做更密集但不预设结论的扫描：

- 4–32 MiB 粗扫；32–96 MiB 密扫；96–160 MiB 粗扫；
- warm、cold、明确 flush 后分开；
- 顺序 copy、只读、dependent pointer chase、随机/置换访问；
- 单/多 CTA 并发、不同 stride/alignment/vector；
- 若 MUSA profiler 提供经验证的 L2 counter，保存 counter；否则只报告延迟/带宽曲线。

只有至少两次独立 run 在相同区间出现可重复拐点，并且 control pattern 能解释，才输出
`effective_cache_boundary` 区间；否则保持 `unknown`。60 MiB 始终单独标记 runtime-reported。

### 6.2 Occupancy

对 P7.3 每个代表性 source 保存实际 MUBIN，再查询：threads、registers、static/dynamic shared、local/
spill、active CTA/MP。查询 binary 必须就是正式执行 binary；若只能重编译，必须降级 provenance，
不能声称逐字节一致。用吞吐曲线检查 API residency 与实际饱和点是否相符。

### 6.3 Tail wave

对完全相同 source，在条件 wave capacity `W=MP*active_CTA_per_MP` 周围扫描：

- `W−8 ... W+8`；
- `2W−8 ... 2W+8`；
- 需要时增加 `3W`；
- 每点至少 30 个样本和三个独立进程；
- 分别使用 memory-light、compute-dominant 和代表性 GEMM。

拟合绝对时间结构 `full_waves * wave_time + tail_cost(tail_CTA)`，不先假设固定百分比。copy、compute、
GEMM 分 family 保存；没有一致趋势则继续 unknown。

### 6.4 验收

- cache：输出可复现边界区间或带证据的 unknown；
- occupancy：每个 P8 激活的 schedule/source 有 exact-binary 或明确降级 provenance；
- tail：至少一个实际目标 family 获得可复验曲线，而不仅是 synthetic kernel；
- 不把 synthetic shared bandwidth、逻辑 GB/s 或 driver capacity 改写成物理峰值。

## 7. P8：S5000 TileSight guarded adapter

### 7.1 目标

在 DeepStack `ae` 的 source-visible TileSight 组件上新增 S5000 条件 profile 与适配层。初版必须是
guarded/experimental predictor，不是通用 oracle。

### 7.2 代码边界

保持 `/root/s5000-calibration/DeepStack-ae-source` 作为固定 commit `d8ffb49...` 的干净参考树；P8
执行时从该提交建立新的专用 integration worktree/branch，不在参考树或当前状态异常的
`DeepStack-git` 目录直接开发。建议在专用工作树新增而不改写既有 NVIDIA/AMD 参数：

```text
src/tilesight/tilesight/arch/s5000.py
src/tilesight/tilesight/s5000/profile_schema.py
src/tilesight/tilesight/s5000/profile_loader.py
src/tilesight/tilesight/s5000/kernel_features.py
src/tilesight/tilesight/s5000/predictor_adapter.py
tests/.../test_s5000_*.py
```

profile 不能只用单个全局 `fp16_flops` 或 `ddr_bandwidth` 覆盖所有条件；应保存：

- value、units、status；
- dtype/family/shape/tile/layout/source 适用域；
- measurement method、SDK/compiler/date；
- raw evidence；
- uncertainty/hold-out evidence；
- out-of-domain 行为。

### 7.3 接口行为

输入为版本化 kernel condition/features；输出至少包含：

- `predicted` 或 `unknown`；
- latency 及组成项（仅在证据存在时）；
- 匹配的参数和来源；
- 资源/occupancy/wave 信息；
- 适用域和警告；
- model/profile SHA。

任何 dtype、family、shape、tile、stage、layout、source/resource 条件越界都返回 `unknown`，不得回退到
A100/H100/AMD 数值或全局平均率。

### 7.4 依赖隔离

当前 DeepStack 顶层 import 会触发 `plotly` 依赖，NoC 参考组件又包含 CPython 3.11 专有 binary；S5000
单卡 adapter 的测试应采用局部 import，不依赖 NCU、plotly 或 proprietary NoC `.so`。TIR 自动分析
仍不完整时，初版接受显式 config + 生成源码 feature extractor，并把未识别特征返回 unknown。

### 7.5 P8 测试

- schema/units/status/provenance 单元测试；
- 已有 P5 v2/P6-v3 只作 regression，不重新宣称独立验证；
- 所有域外条件返回 unknown；
- 不加载 NVIDIA/AMD profile；
- profile/source/model SHA 可重复；
- adapter 预测公式与冻结 development 数据一致；
- import 不需要 NoC binary 或 NCU。

P8 完成时冻结 adapter 源码、profile、development manifest、选择规则和 SHA，然后才生成 P9 case。

## 8. P9：全新 hold-out 与 schedule-ranking 验证

### 8.1 目标

验证 P8 S5000 adapter 对真正未观察 shape/schedule 的点预测和排序能力。P9 不修模型；失败只报告
误差并决定是否回到新的 development 版本。

### 8.2 Hold-out 设计

建议第一版至少：

- FP16 GEMM：5 个新 shape，每个 4–8 个 schedule；
- BF16/INT8：仅对 P7 已建立稳定 transfer 的 dtype，各至少 3 个 shape、每个 ≥4 schedule；
- reduce、fused、stride：各至少 5 个新条件；
- cache/wave/resource：各保留未参与 P7.4 拟合的条件。

若某 family 没有足够 development 覆盖，P8 应在 preflight 返回 unknown；unknown 是正确保护行为，
不得为计算 MAPE 而临时扩大适用域。

### 8.3 防泄漏流程

1. P8 冻结并记录 SHA；
2. 依据预注册规则和固定随机种子生成具体 P9 config；
3. preflight 为全部 case 生成预测/unknown，保存并签名；
4. 再允许编译和测量；
5. finalize 复核 P8 SHA 未变化；
6. P9 数据永不回写当前 P8；若需改进，建立 P8.1 development 与全新 P9.1 hold-out。

### 8.4 指标

点预测逐 case 报告：signed error、APE、MAPE、MdAPE、P90 APE、max APE，并按 dtype/family/shape
分组；不只报告一个平均数。

每个多 schedule shape 报告：

- Spearman ρ 和 Kendall τ；
- 预测 top-1 是否为实测 top-1；
- top-k recall；
- top-1 regret：`(chosen_latency / true_best_latency - 1)`；
- 编译/正确性/runtime failure 的排序处置；
- 预测 unknown 覆盖率。

### 8.5 建议预注册验收等级

这些是工程门槛，不是论文或官方标准，必须在 P9 测量前冻结：

| 等级 | 点预测 | 排序 | 允许结论 |
|---|---|---|---|
| A | MdAPE ≤10%，P90 APE ≤20% | median Kendall τ ≥0.7，median top-1 regret ≤5% | 可在已验证域实验性参与 schedule 筛选 |
| B | MdAPE ≤15%，P90 APE ≤30% | median Kendall τ ≥0.5，median top-1 regret ≤10% | 仅作辅助提示，必须实测最终候选 |
| C | 超出 B 或稳定性不足 | 排序不稳定 | 不接入优化决策；保留证据并回到新 development |

任何 family 单独失败都不能由其他 family 的好结果平均掩盖。即使达到 A，也不能称为通用 S5000
oracle，只能在 P9 明确覆盖域内使用。

## 9. 阶段交付物

| 阶段 | 必需交付物 |
|---|---|
| P7.1 | runtime A/B runner、源码 diff、raw/log、失败矩阵、稳定路径或 blocker |
| P7.2 | versioned timing harness、固定开销拟合、协议版本、短 kernel 接纳/拒绝表 |
| P7.3 | development/diagnostic/reserved manifest、完整 GEMM CSV/JSON、失败统计 |
| P7.4 | cache 曲线、exact-binary resource 表、tail 曲线、明确 unknown |
| P8 | S5000 profile/schema/adapter、单元测试、冻结 SHA、适用域文档 |
| P9 | preflight、全新 raw、逐点误差、ranking/regret、最终接入决定 |

一键入口应按阶段独立，默认只运行该阶段已获授权的内容；任何 runner 都不得自动 reset/reboot GPU。

## 10. 方案审核

### 10.1 顺序审核

顺序正确：P7 runtime 异常会污染长 kernel 数据，短 kernel 固定开销会污染小 shape 数据；两者必须在
扩大 GEMM 集前解决。cache/occupancy/tail 依赖稳定 kernel 和实际 binary，故放在 GEMM 数据后；P8
依赖冻结参数，P9 必须在 P8 冻结后执行。

### 10.2 对现有证据的保护

- P5 v1/v2 和 P6-v3 保持不可变；
- P6-v3 只能作 P8 regression，不能重复当独立验证；
- P7 成功 trial 的 36–45 µs/约 129.57 TOPS 不进入 development；
- P2 短 kernel 和 P3 不稳定 cache 点不因新方案而追溯性改写；
- runtime-reported L2 60 MiB 与实测 effective boundary 始终分离。

### 10.3 已修正的可实施性风险

1. “关闭 TMA”不能通过关闭 descriptor prefetch 冒充；必须检查生成 `.mu`。
2. warp specialization 有已确认的 disable pass，但 TMA pipeline 可能依赖它；TMA 本身未找到统一
   disable pass。计划要求生成源码与正确性复核，并允许数学等价同步-copy 对照。
3. 当前 P2 harness metadata 硬编码且原 run 源码快照缺失，新协议必须 versioned 并随 raw 保存源码。
4. 资源查询优先保留执行 MUBIN，避免再次只有重编译 binary provenance。
5. P9 明确加入 schedule-ranking 和 regret，不再用每 family 一个点的低 APE 代替排序验证。
6. DeepStack 的 NCU、plotly、CPython 3.11 NoC 依赖不应阻塞 S5000 单卡 adapter，采用局部依赖隔离。
7. P8 使用新的专用 integration worktree，保留 `DeepStack-ae-source` 干净参考树，不使用当前大量
   tracked files 显示删除的 `DeepStack-git` 工作目录。

### 10.4 剩余风险

- MUSA runtime/driver 异常可能使 P7.1 长期 blocked；
- 同步 copy 对照与 TMA kernel 的 schedule 可能无法完全同构，归因需保守；
- 缺少公开硬件 counter 时，cache boundary 可能继续 unknown；
- 自动 TIR feature extraction 不完整，初版 adapter 可能依赖显式 condition；
- A/B 与大规模 schedule 扫描成本较高，必须先 FP16 再扩 dtype；
- P9 样本量仍不足以证明未见 family 的泛化。

### 10.5 最终审核决定

方案通过，但附带两个启动条件：

1. P7.1 动态实验必须等待设备查询恢复，并由用户授权任何必要的 reset/reboot；
2. 在执行前先冻结 P7.1 config、A/B 生成源码判定规则和 P7.2 计时门槛。

满足条件后按 P7.1 → P7.2 → P7.3 → P7.4 → P8 → P9 顺序执行。若任一阶段未过门槛，保留
失败证据并停止依赖它的下游阶段，不用猜测参数继续推进。
