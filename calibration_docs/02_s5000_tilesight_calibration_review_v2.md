# S5000 TileSight 风格校准方案审核与实施修订 v2

状态：approved for P0  
日期：2026-07-28  
审核对象：`01_s5000_tilesight_calibration_plan.md`

## 1. 审核结论

总体技术路线成立，可以实施。正式执行从原计划的 6 个阶段调整为 7 个阶段。核心修改是把“测量基础设施是否可信”设为独立验收门槛，并区分以下三类能力：

1. S5000 在受控原生 MUSA/mcc kernel 下表现出的有效硬件能力；
2. mcc 对具体 kernel 的代码生成能力；
3. TileLang-MUSA lowering、schedule 与 codegen 路径实际达到的能力。

后续参数表必须标注参数来自哪一层。TileLang-MUSA kernel 的低性能不能自动解释为 S5000 硬件上限；原生 MUSA kernel 的高性能也不能自动视为 TileLang 当前可达性能。

## 2. 必须执行的修订

### 2.1 增加独立测量验证阶段

在开展大规模 sweep 前，先验证 MUSA event 的同步和计时口径、空 kernel/launch 开销、warm-up、统计分布、编译缓存及温度/频率漂移。未通过该阶段的数据只能标记为探索数据。

### 2.2 关键基准采用双路径

在接口和工程成本允许时，关键路径同时提供：

- 原生 MUSA/mcc 基准，用于建立当前编译器下的有效上界；
- TileLang-MUSA 基准，用于校准真实 TileLang 生成路径。

双路径必须使用相同的 shape、dtype、输入分布、正确性规则和计时统计口径。无法实现原生对照时，要将该限制写入参数 provenance。

### 2.3 提前冻结数据划分

在批量测量前建立三个配置集合：

- `calibration`：允许用于参数估计；
- `development`：允许用于调试实现和误差归因；
- `holdout`：模型定型前不得用于参数选择。

hold-out 文件建立后必须记录 hash。改变 hold-out 需要新版本和理由，不能静默覆盖。

### 2.4 第一轮采用垂直切片

第一轮只要求打通：FP16 GEMM、FP16 copy/FMA、DDR read/write、block/grid/wave sweep 和一个 hold-out GEMM。该切片用于验证日志、CSV、正确性、生成代码追踪、参数提取和预测流程。通过后再扩展 BF16、INT8、reduce、SFU 与 attention。

### 2.5 Cache 和 occupancy 使用证据等级

Cache 结论至少区分：

- L1：仅有 streaming bandwidth 曲线；
- L2：bandwidth 与工作集扫描出现可重复拐点；
- L3：再获得 dependent-latency、counter 或其他独立证据。

在 L3 前只称为“候选有效 cache 工作集拐点”，不称为物理 L2 容量。

Occupancy/驻留 CTA 证据优先级：

1. MUSA occupancy API；
2. mcc 资源报告或 launch metadata；
3. 可审计生成代码；
4. 受控资源 sweep 的吞吐台阶。

只有第 4 类证据时，参数名必须体现 `empirical`，并保留不确定范围。

## 3. 正式实施阶段

### P0：后端与环境审计

回答“运行的代码是什么”。定位 TileLang-MUSA 仓库、commit、MUSA backend、lowering/codegen、生成源码、mcc 命令、runtime 启动路径，并保存系统、GPU、驱动、SDK、Python、PyTorch 与 torch_musa 快照。

**通过门槛**：最小原生 MUSA runtime 和最小 TileLang-MUSA kernel 均正确运行；能够从 TileLang 输入追踪到生成代码、编译命令和实际启动。若无法追踪，停止性能校准，先补足可观测性。

### P1：测量基础设施验证

回答“测量值是否可信”。建立统一 runner、MUSA event 计时、同步、warm-up、repeat、median/p10/p90、正确性接口、run ID、源码/config hash、GPU 状态前后快照与原始数据 schema。

**通过门槛**：稳定 kernel 的重复分布可解释；确认计时不包含编译；单条记录能定位到代码、配置、环境和日志。

### P2：计算路径校准

先完成 FP16 垂直切片，再按 BF16、INT8、vector/elementwise、reduce、exp/rsqrt 扩展。扫描 shape、tile、线程/block 和 pipeline stage；保存关键生成代码特征。关键路径尽可能提供原生 MUSA 与 TileLang-MUSA 对照。

**通过门槛**：只有 `correctness=pass` 且稳定的样本进入吞吐参数；每类参数说明 dtype、shape/tile 适用范围及代码生成路径。

### P3：DDR、cache 与 shared-memory 校准

执行连续读、写、混合、工作集、stride、对齐、layout 和 CTA 并发扫描；加入 dependent pointer chasing 或等价的延迟实验；shared-memory 测顺序和冲突风险模式。

**通过门槛**：输出有效带宽曲线、延迟曲线、候选拐点、重复性和证据等级；不推断未经证实的 cache/bank 微架构。

### P4：Occupancy、pipeline 与 wave 校准

扫描 block size、可控制的 register 压力、SMEM 用量、pipeline stage、小 grid、满 wave、多 wave 和 tail wave。优先使用 MUSA API 与 mcc 报告，经验曲线作为补充。

**通过门槛**：输出 resident CTA/约束来源、证据等级、full/tail wave 开销和适用配置；未知约束明确标为 unknown。

### P5：S5000 参数化与初版模型

把 P2–P4 数据整理为版本化 JSON/CSV。初版采用条件化有效速率、resident CTA、prologue/steady/epilogue 和 full/tail wave 分解，不使用不可解释的全局误差修正。

**通过门槛**：每个参数可追溯到原始 run；模型能对 calibration/development 集生成预测和资源分解；缺少证据的参数输出区间或 unknown。

### P6：Hold-out 验证与接入评审

在参数冻结后运行预先定义的 GEMM、reduce、layout/stride、attention 子核或 fused kernel hold-out。报告逐点实测、预测、误差符号、绝对百分比误差和误差归因。

**最终结论只能为以下之一**：

1. 可以接入 TileSight/TileLang cost model；
2. 可以接入，但只覆盖已验证的 kernel/schedule 域；
3. 证据不足，列出必须补测或修复的项目。

## 4. 执行门控

```text
P0 后端可追踪
  ↓
P1 计时与数据可信
  ↓
P2/P3/P4 分资源校准
  ↓
P5 参数冻结
  ↓
P6 只读 hold-out 验证
  ↓
cost model 接入决策
```

阶段之间不能因为“结果看起来合理”而跳过。编译失败进入 backend 问题表；数值失败进入 correctness 问题表；计时不稳定进入 measurement 问题表；三者均不能作为性能参数。

## 5. P0 实施边界

下一阶段仅允许：只读检查现有安装和源码、记录环境、运行已有最小 MUSA 示例、建立最小 TileLang-MUSA 正确性例程及保存编译证据。P0 不开展大规模 shape sweep，不输出 S5000 峰值，不推断寄存器、矩阵 fragment、cache 映射或 occupancy 常数。

P0 完成后应新增 `03_p0_backend_environment_audit.md`，逐项给出通过、失败、证据路径和进入 P1 的决定。
