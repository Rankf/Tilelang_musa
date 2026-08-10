# P0–P7 正确性复核与 P7 根因审计

状态：completed static audit；动态根因 A/B 因设备状态不明确而暂停  
日期：2026-07-29（UTC+8）

## 1. 审核结论

P0–P5 的主线没有偏离“先真实后端与实测参数、再建模型”的目标；P6-v3 确实完成了冻结模型后的
四个事前点预测与独立实测，但它只支持受保护的窄域实验性点预测，不支持通用 TileSight oracle
或 schedule ranking。P7 证明目标大 GEMM 可以编译、单次运行且数值正确，但高频连续执行的计时
协议不稳定；这不等于“TileLang-MUSA 算子不能运行”，也没有证据证明 S5000 物理故障。

本次发现并修正了三个确定性表述/实现错误：

1. P6-v3 把四点 APE 的中位数写成“MAPE 中位数”。正确值是 MAPE/mean APE `4.3907%`、
   MdAPE `1.8456%`、max APE `13.8375%`；历史测量值不变。
2. P2 harness 的 `timing_method` 固定写成“100 captured kernels”，无法正确描述 graph-unroll=20
   等配置；历史 raw 保持原字段，并在此勘误。当前文件不就地修改，后续必须以新版本文件读取
   `graph_unroll` 后再启用。
3. P2 harness 曾把稳定性门槛拒绝与真正 runtime exception 都写成 `runtime_failed`。新版本必须区分
   `measurement_rejected`、`correctness_failed`、`runtime_failed`，并增加 `failure_stage`；旧 raw 的
   状态不能据此追溯性改写。

此外，文档 14 的“可以接入”边界原先容易被理解得过宽，现收紧为 guarded/experimental
narrow-domain predictor。

另发现一项既有 provenance 缺口：三次正式 P2 raw 的 `metadata.json` 均记录 harness SHA
`107c5ede319630f1cab2d335efd4c3731bd03c584e55744b8b21679f37b8edfc`，但本轮开始时及审核恢复后的
`benchmarks/p2_tilelang_compute_suite.py` SHA 是
`b49e8a346387284cf21bc97f2b97513a3008c4b3908f10b1321f1ef2a7c7b626`，工作区没有找到前一 SHA
对应的源码快照。raw 配置、生成 `.mu`、source SHA、样本与日志仍在，但“运行时 Python harness
逐字节复现”不能声称完整。后续每次正式 run 必须把 harness 快照复制到该 raw run 内。

## 2. 各阶段复核

| 阶段 | 仍然有效的证据 | 限制或勘误 | 审核状态 |
|---|---|---|---|
| P0 | 原生 MUSA runtime、mcc、TileLang-MUSA 最小 kernel 编译/运行/数值正确；commit 与本地 patch 可追溯 | 只证明最小后端路径，不证明性能参数；可选 CUDA 子模块未初始化，文档已明确 | 有效 |
| P1 | 三次 event 计时 infrastructure smoke、环境/配置/hash 记录 | 原返回式 adapter 含 output allocation；已在文档 04 勘误，数据不得拟合 | 有效作基础设施 smoke |
| P2 | 显式预分配输出；FP16/BF16/INT8、vector/reduce/SFU 的编译→正确性→计时顺序与三进程原始数据 | 256³/512³ 是短 kernel 计时口径待解决，不进入参数；Event 与 Profiler 共享同一 runtime/device，只是两接口交叉检查；FP GEMM reference 是设备侧 PyTorch matmul，不是 CPU 独立参考；正式 run 的原 harness 快照缺失 | 条件有效，源码复现有缺口 |
| P3 | DDR/stride/shared 的条件有效带宽曲线；runtime API 报告 L2 `62,914,560 B (60 MiB)` | 60 MiB 是驱动属性，不是微基准证明的可用 cache boundary；后者仍为 unknown | 条件有效 |
| P4 | exact-kernel occupancy/resource 查询、native CTA/wave 与受控压力曲线 | 正式计时 binary 未保存；资源查询来自相同 `.mu`/mcc 重编译 MUBIN。8/10/12 CTA/MP 等只适用于对应 source/条件，不得外推 | 条件有效 |
| P5 v1 | 完整条件键、provenance、unknown 防护 | exact lookup 的 0% reconstruction 不是泛化验证 | 有效作注册表/精确查表 |
| P5 v2 | 在隔离 development set 上冻结的有限 transfer 规则；SHA 未被 P6/P7 改写 | 仅覆盖声明的 family/shape/tile/layout 域，不覆盖新 family 或任意 schedule | 有效作窄域 predictor |
| P6 原版 | 四个独立 hold-out 均正确且实测稳定 | P5 v1 对四例返回 unknown，故不能计算预测误差；结论“证据不足”正确 | 有效的失败验收 |
| P6-v3 | 冻结 P5 v2 后的四个事前预测与独立实测，时间顺序及 SHA 隔离成立 | 每 family 仅一个 hold-out；MAPE/MdAPE 已勘误；不能证明 schedule ranking、置信区间或广泛泛化 | 有效但范围很窄 |
| P7 | 大 GEMM 编译与单次数值正确性 3/3；graph 和 graph-free 高频连续执行均可复现 runtime 异常 | 成功计时存在选择偏差且协议仅 1/3 稳定，所有 P7 延迟都不能拟合 | 诊断完成，动态根因未完成 |

## 3. P7 静态定位

目标是 FP16/FP32 accumulation GEMM `(1536,1024,1536)`，tile `64×64×64`、4 warp、stage 1；
生成 source SHA 是 `9ace2f6f78aed08ced8f393797da60e524f08afc89c1ca4bd5c587f9022d33ff`。
它使用 MP31 路径：`MUtensorDescriptor`、TMA load、`__musa_async_*` barrier 和 128 producer /
128 consumer threads。K-loop 为 24 次，launch 为 384 CTA、256 threads、16 KiB dynamic shared。

作为静态对照，P2 中三进程稳定通过的 FP16 1024³同 tile/schedule source SHA 为
`83863ac033d66e13721a9bb826e46e1959efc8618f553480d137b4040c4fe1a6`。逐行 diff 只有两个
K-loop 上界由 16 变成 24；它使用同一 TMA/barrier/specialization 模板，launch grid 为 256 CTA。
因此已知触发差异收敛到更长的 async/barrier 生命周期、更大 grid 或两者与重复执行的交互，
而不是不同 dtype 或完全不同 codegen 模板。

`MUSA_LAUNCH_BLOCKING=1` 的一次异常在 `MUSAGraph.replay()` 内 MUDNN Fill 暴露，底层为
runtime `err 999/unknown error`。由于无 graph 连续 launch 也失败，且异步错误可能在后续 API
暴露，不能把 Fill 或 MUSAGraph 单独认定为根因。Profiler 退出错误与 `double free or corruption`
是严重的 host/runtime 后果，但现有证据也不能判定其最初破坏点。

根因候选按下一轮隔离价值排序：

1. MP31 TME/TMA + async barrier + warp specialization 在 24-loop 重复 kernel 下的 lowering/compiler/runtime 交互；
2. TileLang-MUSA barrier protocol 或 kernel 退出前 async drain 不完整；
3. 64-byte aligned descriptor 经 `grid_constant` 按值传递时的 encode/packing/lifetime 路径；
4. MUSA driver/runtime 的连续 launch、graph replay 或 profiler 独立缺陷；
5. MUDNN Fill 只是暴露点；物理硬件故障目前证据不足。

这份排序不是根因定案。尤其不能从“同 source SHA 有时成功”排除未定义行为或 runtime 状态交互，
也不能从 CLI 查询进程失败推导硬件损坏。

静态代码中尚未发现可以直接定案的常量错误：每轮 A/B tile 各为 8192 B，生成代码先
`__musa_async_add_trans(1, 8192)`，再由 `mbarrier_arrive_expect_tx(1, 8192)` 增加另一笔并 arrive，
表面上与 16384 B 总事务量一致。SDK 把 `MUtensorDescriptor` 定义为 8 个 qword（64 B）并要求
64 B 对齐；lowering 通过 `tvm_stack_alloca("tvm_ffi_any", 16)` 预留注释所称 128 B，再由 runtime
逐次调用 `muTensorDescriptorEncode`，kernel 端以 `grid_constant` 按值接收。容量看起来足够，但
实际栈对齐、FFI object lifetime 与 launch packing 仍需动态/内存工具验证，不能仅凭源码认定安全。

## 4. 数据与统计边界

- P2/P3 的 MUSA Event 和 torch Profiler 是两条计时接口，但共享 MUSA runtime、driver 和设备，
  应称交叉检查而非完全独立测量系统。
- FP16/BF16 GEMM reference 由 MUSA 上的 PyTorch FP32 matmul 计算，再转换到目标 dtype；它是不同
  实现路径的数值参考，不是 CPU reference。INT8 才使用 CPU int32 matmul reference。
- P4 resource 数值必须携带 source SHA 与查询 binary provenance；不得作为架构全局常数。
- P6-v3 四点 APE 为 `0.0340728794%`、`13.8374712115%`、`3.5402127688%`、
  `0.1509863151%`。样本太少且 family 不同，汇总 MAPE 只作描述性统计。
- P7 batch-32 原 shape 完整确认仅 1/3 通过；成功的 36–45 µs 和 129.57 TOPS 属诊断数据，
  不能进入 S5000 参数表、P5.2 或反向拟合。

## 5. 安全动态复验协议

当前 `mthreads-gmi` 虽显示 0%/0 MiB/P12，但连续两次无法查询 running processes，且无残留
P7/P2 Python 进程。静态审核期间不启动新 kernel；不把“没有用户态进程”视为 driver 已恢复。

设备恢复并经用户授权后，先以独立进程运行最小 runtime smoke，再对同一 GEMM执行以下矩阵：

1. baseline：TMA + warp specialization + descriptor prefetch；
2. 仅关闭 `TL_ENABLE_MUSA_TMA_PREFETCH`；
3. 保留 TMA、关闭 warp specialization；
4. 禁用 TMA lowering，改用同步 global→shared copy；
5. 同时禁用 TMA 与 warp specialization。

每个变体先编译、单次正确性，再受控重复；至少三个全新进程，首次失败立即结束该进程，初轮不使用
MUSAGraph/Profiler。若 4/5 稳定而 1/2/3 失败，故障域进入 TMA/barrier；若全部失败，再优先检查
descriptor/grid-constant adapter 与 driver/runtime。仅关闭 prefetch 不等于关闭 TMA。

在获得稳定 3/3 协议及独立 shape 交叉验证前，不开始 P5.2 拟合；修复后应新建 development set，
冻结新模型后再建立全新的 P6-v4 hold-out，不能回填 P5 v2 或复用 P6-v3 hold-out。

## 6. 不可变性与追溯

冻结文件复核值：P5 v1 SHA
`9ef664f1e27f78bcd99cceb458750cf02319d6ae68de8573438cec69a17aeec0`；P5 v2 SHA
`b94296eb50b904b0e34b80d79535887ad7932b7b08f57fe736c609eb48caa9be`。本次修正文档表述并记录
未来 versioned harness 的修复要求，没有修改当前 P2 harness、历史 raw、P5 v1/v2 或 P6-v3 测量结果。
