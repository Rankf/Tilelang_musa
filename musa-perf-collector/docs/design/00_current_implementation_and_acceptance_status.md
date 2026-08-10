# TileLang-MUSA 真实性能采集工具：当前完整实施与验收状态

状态：**v1.2，已实施并经 S5000、CPU 离线、打包、多实例 lowering 与落盘证据重放复核**  
日期：2026-07-30  
工具根目录：`/root/musa-perf-collector`  
编译器工作树：`/root/s5000-calibration/tilelang-musa`  
目标设备：MTT S5000，MUSA `mp_31`

本文是当前交付状态的单一入口。它汇总已经真实实现和验证的能力、可复现命令、证据目录、数据语义、
审计结果与未完成边界；不以计划项代替已验收功能。

分项设计和实现细节分别见：

- [01 离线编译期分析](./01_offline_compiler_analysis_plan.md)
- [02 TileOp 覆盖与执行完整性](./02_tileop_coverage_execution_completeness_plan.md)
- [03 运行期真实性能与 Tile 细粒度采集](./03_runtime_tile_collection_plan.md)
- [04 低扰动 trace 优化](./04_runtime_trace_overhead_optimization_plan.md)
- [05 在线 JIT 自动插桩](./05_online_jit_auto_instrumentation_plan.md)

## 1. 目标与当前结论

工具目标是采集 TileLang-MUSA 的真实、可审计数据，而不是建立 S5000 性能预测或校准模型。

当前完整链路如下：

```text
TileLang IR / compiler artifact
  ├─ CPU 离线：TileOp、循环/执行域、FLOPs、逻辑字节、buffer、DAG、MUSA C、provenance
  └─ S5000 运行期：真实 launch/Event/stream/正确性/遥测
                         └─ 受控抽样的 compiler-exact Tile begin/end clock64 trace
                                  └─ raw trace + offline enriched trace + 独立 audit
```

已实现的 `complete` 仅针对一次满足 lineage、预算、正确性、回读生命周期和独立审计的 session；它不表示
任意 TileLang 程序、任意动态 grid 或所有硬件 profiler 能力都已支持。

## 2. 已交付能力

| 层级 | 已真实实现 | 关键输出/证据等级 |
|---|---|---|
| CPU 离线 | 无损 TileOp discovery、unknown opaque fallback、20 种已注册动作型 TileOp family、循环/条件上下文、per-invocation/per-block/per-kernel 执行域、FLOPs/逻辑字节/buffer/DAG/reuse、MUSA C 与 compiler metadata | 静态编译事实及显式假设；不输出真实时延 |
| 受控 runtime adapter | baseline MUSA Event、正确性、launch 参数、stream/sync、会话前后温度/功耗/频率、短窗口 MUSA activity、深度 Tile trace | baseline、activity 和 instrumented 证据分离 |
| compiler-exact trace | compiler provenance、固定 slot、TileOp/ordinal/block/iteration、`clock64` begin/end、uint64 interval、raw/enriched join | `compiler_exact_instrumented_issue_boundary`；区间不是 TileOp 独占硬件时长 |
| 在线 JIT | compile hook、baseline/instrumented 双 variant、独立 cache namespace、应用自有输入/正确性、静态 grid + 受控 launch/block/TileOp 选择 | 应用无需 `RuntimeWorkload`/factory |
| 低扰动 | 默认 every=64 周期抽样、原生 C fast dispatch、原生 MUSART Event/pinned host/collector stream/deferred D2H trace pool、记录/字节/launch 预算 | 未选中 launch 走 baseline；D2H/解析不进入主计时窗口 |
| 审计与交付 | `session_audit.json`、`session_validation.json`、`session_audit_inputs.json`、故障注入、MUSA/CPU wheel 构建检查 | `audit_revision=P6.9.4`、落盘重放、fail-closed |

## 3. 运行期采集的数据与语义

默认轻量采集：MUSA Event 时延、应用正确性、launch/grid/block/stream 信息、会话前后设备温度/功耗/频率。

标准采集：受控少量 kernel 的 stream/sync 信息和当前运行库可提供的 MUSA activity 汇总。当前
`libmupti.so.1.0.5` 缺少 raw metric/event group 和 PC sampling 所需符号；工具遇到相关请求会记录
`unavailable/partial`，不会生成模拟 counter。

深度采集：仅对被选择的 launch 和 block 插入 compiler-exact slot trace。每条记录至少包含：

- `kernel_id`、`launch_index`、`generation`、`slot`；
- `op_id`、`ordinal`、TileOp family/source kind、begin/end boundary；
- 静态 block、lowered instance 和 lowering semantics；
- `clock_cycles`、`interval_cycles`、同 launch 的 MUSA Event 时延；
- process/device/stream、plan hash、readback policy、evidence level；
- enriched TileOp 离线 metadata。

`clock64` 是由选定 lane 在插桩边界读取的 cycle envelope。它可证明该边界间隔，不能称为一个 TileOp 在
GPU 上完全独占的执行时间；不同 block 的时钟也不直接相减。插桩结果与未插桩 baseline 永远分开报告。

## 4. 当前 S5000 真实证据

当前 P6.9.4 多实例在线证据目录（2026-07-30 从当前工作树重新生成的 compiler-exact artifact 运行）：

```text
/root/musa-perf-collector/results/online/online_128980_1785403516203950746_b3bc0e1e
```

该运行由真实 warp-specialized TileLang GEMM 应用触发，不经过 managed `runtime collect --factory` 路径。
source ordinal 0 在 final-device IR 中稳定 lowering 为 producer/consumer 两个实例：

| 项目 | 结果 |
|---|---|
| 应用正确性 | `application_assert_close / pass` |
| session | `complete` |
| capability phase | `P6.9` |
| 独立审计 | `P6.9.4 / pass / 0 violation`，`evidence_source=persisted_session_files` |
| trace | 4 条 raw、4 条 enriched，一一对应 |
| diagnostics | 0 条 |
| lowered instance 0 | begin/end 均 valid，1032 cycles |
| lowered instance 1 | begin/end 均 valid，353 cycles |

上述两个 interval 都是对应 warp partition 首线程读取的 issue-boundary envelope，不是 TileOp 独占硬件执行时间。

保守的 every-64 端到端开销披露仍为约 **8.43%**（约 12.7 us 的极短 kernel、4099 launch、65 次
instrumented、260 条 Tile trace）。有一轮同规模结果为 7.08%，但本轮修改不是正常 launch 路径优化，
因此不将其宣称为性能提升，也不声称已达到 2%。

## 5. 复现命令

### 5.1 CPU 离线/全部采集器回归

```bash
cd /root/musa-perf-collector
PYTHONPATH=src:/root/s5000-calibration/tilelang-musa \
  /root/s5000-calibration/.venv-tilelang-musa/bin/python \
  -m pytest -q tests/unit tests/integration
```

最新结果：124 passed（109 unit、15 integration）。运行中有 53 个 `torch_musa` pinned-memory
API deprecation warning；它们不改变测试结果。

### 5.2 在线真实应用采集

```bash
cd /root/musa-perf-collector
MUSA_PERF_COLLECTOR_ONLINE_CONFIG=/root/musa-perf-collector/configs/runtime/online_gemm_reduce_trace.json \
PYTHONPATH=/root/musa-perf-collector/src:/root/s5000-calibration/tilelang-musa \
  /root/s5000-calibration/.venv-tilelang-musa/bin/python \
  -m musa_perf_collector.examples.online_gemm_reduce_app
```

命令会打印本次 `online_session` 目录。重点查看：

```text
session_validation.json   # complete/partial/invalid 与审计 revision
session_audit.json        # 独立账目和全部 violation
trace.jsonl               # 原始设备端 trace
trace_enriched.jsonl      # 关联离线 TileOp metadata 的 trace
correctness.jsonl         # 应用声明的正确性结果
capability.json           # 本次实际启用/降级能力
diagnostics.jsonl         # 不能忽略的异常与降级
```

### 5.3 打包检查

正常 MUSA wheel 必须同时包含 `_native_dispatch` 与 `_native_musa_trace`，且隔离安装后均可 import。
以 `MUSA_HOME=/nonexistent/musa` 故意构建时，CPU wheel 必须只包含 `_native_dispatch`；本轮已实测通过，
防止可选 MUSART 扩展失败后的陈旧 `.so` 被错误打包。

## 6. P6.9.4 落盘证据重放审核闭环

flush 先原子写入 trace/raw-enriched/launch JSONL、每 kernel plan 和 `session_audit_inputs.json`，再重新打开
这些文件构建 audit 输入；读取、解析、schema 或路径校验失败会生成 `PERSISTED_EVIDENCE_READ_FAILED` 并强制
session `invalid`。审计器不信任 session 的建议状态，并检查：

1. total launch = fast path + recorded launch，以及 fast-path 原因总账；
2. instrumented dispatch/launch、预约资源、record/byte 预算；
3. expected slot layout、plan hash、PID/device/stream 与 launch observation；
4. generation 的单一所有权、有效 clock 类型、同 launch Event 一致性；
5. begin/end 由 uint64 `clock64` 精确推导 interval，且 interval 语义正确；
6. instrumented variant 序号连续、cold-start 标识一致；
7. raw trace 与 enriched trace 的关键字段和 TileOp `op_id` 完整对应；
8. 每个 TileOp/block/lowered-instance 恰有一个 begin/end，实例索引从 0 连续且同 pair validity 一致。
9. 关键索引只接受非 bool 的 JSON integer；畸形 layout 字段只产生 violation，不能使审计器异常退出。

审计已修复“layout 中 end 排在 begin 前时可能跳过 interval 校验”的顺序依赖，并以篡改 interval 的逆序
layout 故障注入验证不能绕过。任一冲突都使 audit fail；drain 失败使 session invalid，资源耗尽会进入
terminal baseline/partial，而不会伪装为完整 trace。

## 7. 当前边界与后续工作

| 项目 | 当前状态 |
|---|---|
| device | 自动预分配 trace pool 当前批准为 device 0；当前 device 不匹配时 fail-closed/terminal baseline |
| execution backend/grid | 仅 `tvm_ffi` 与可证明的静态 grid；动态 grid 和未知 backend 拒绝自动深度插桩 |
| 多实例 lowering | 已用真实 warp-specialized GEMM artifact 完成 `lowered_instance` S5000 端到端验收；该结论只覆盖已记录的静态 producer/consumer lowering，其他动态或不同形态 pipeline 仍须逐 artifact fail-closed 验收 |
| 原始硬件 counter/PC sampling | 被当前 MUPTI 运行库符号能力阻塞；保留 activity，不模拟数据 |
| 扰动 | 极短 kernel 保守 8.43%，未达 2%；需要进一步下沉 selected 慢路径或使用更稀疏采样才能继续优化 |
| 历史波动 | 之前一轮完整测试曾有一次 GEMM+reduce_max `partial`，临时目录已删除，后续专项与两次完整重跑均未复现；不虚构根因 |

## 8. 修改控制与审核结论

TileLang-MUSA 仍是 dirty worktree，包含 provenance pass、trace injector、JIT hook、cache namespace 和
MUSA lowering/template 相关改动。它们被保留，未执行 reset、checkout 或清理未跟踪文件。本轮检查
`git diff --check` 通过；本轮新增的 TileLang 多实例/warp-partition 注入专项为 6 passed，collector 完整回归为
124 passed。既有更广 TileLang 专项的历史验收数不用于替代本轮结果。

Collector 目录本身不是 Git 仓库，因此采用源码、wheel、隔离安装文件哈希一致性和完整测试审计，而不把
不存在的 Git 状态伪装为变更证明。正常 MUSA wheel、无 MUSA CPU wheel、Python compileall 与 S5000
在线真实运行均已完成验证。

结论：工具已经可以在批准范围内采集真实运行数据和 compiler-exact Tile 细粒度证据，且产物带有可复核的
完整性、开销、模式和证据等级。上述边界以 capability/session 为准，不能扩大解释为通用 GPU profiler。
