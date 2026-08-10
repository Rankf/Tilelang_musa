# MTT S5000 / MUSA TileSight 风格单卡性能校准计划

状态：规划基线 v1（尚未开始测量；实施阶段划分以 `02_s5000_tilesight_calibration_review_v2.md` 为准）  
日期：2026-07-28  
范围：1× MTT S5000；MUSA 3.3.5-server 驱动；`/usr/local/musa-4.3.5` SDK；本地 TileLang-MUSA。  
非范围：多卡 collective、网络、分布式 LLM 调度、3D-DRAM bank 模型、功耗/热模型、算子自动优化。

## 1. 目标与产出定义

建立一套由实际 MUSA 编译与 S5000 执行结果支撑的、可复现的单卡校准数据集。它用于为 TileLang-MUSA 的 tile 计划提供 TileSight 风格的白盒代价模型输入；它不是通过公开规格或 CUDA 类比得到的“理论 GPU 参数表”。

最终交付物必须包含：

1. 环境快照、命令行、生成代码和编译日志；
2. 一键运行的微基准脚本与原始 CSV；
3. 可追溯的 S5000 参数 JSON/CSV；
4. 没有参与校准的 hold-out 算子的预测/实测对照；
5. 误差归因、已知不确定项，以及是否足以接入 cost model 的结论。

每一项参数必须携带：测试日期、git commit、SDK/驱动/编译器版本、命令、dtype、shape、tile、launch 配置、重复次数与统计口径。

## 2. 方案审核

### 2.1 已确认且合理的设计

| 决策 | 审核结论 | 原因 |
|---|---|---|
| 先单卡、后多卡 | 通过 | 可避免将本地 kernel 误差与通信误差混在一起。 |
| 编译失败→正确性→性能 | 通过，设为硬门槛 | 性能数据仅对数值正确且稳定的 kernel 有意义。 |
| 先读 TileLang-MUSA 与生成 MUSA 代码 | 通过，最高优先级 | TileLang schedule 是否真的映射为预期 tile/pipeline，必须由 lowering 和 mcc 日志证实。 |
| 以微基准校准而不是公开规格 | 通过 | S5000 的执行粒度、cache 与矩阵路径不能由 CUDA 假设替代。 |
| 保留 hold-out 验证 | 通过，不能删减 | 没有 hold-out 的拟合不能证明模型具有预测力。 |

### 2.2 需要补全的控制项

| 项目 | 要求 | 不满足时的处理 |
|---|---|---|
| 时钟/温度/功耗状态 | 每次基准前后记录 `mthreads-gmi`；尽量固定空闲系统状态。 | 标记样本不可比较；不并入参数拟合。 |
| 计时方法 | 使用 MUSA event 计时；记录 warm-up、repeat、median、p10/p90。 | 禁止用 Python wall-clock 作为主性能数据。 |
| 正确性 oracle | GEMM 用高精度 CPU/PyTorch 参考；近似算子明确 atol/rtol。 | 失败样本仅进入 codegen/correctness 问题表。 |
| 编译缓存 | 每次记录 cache 命中状态与生成文件 hash。 | 缓存不透明时，单独重编译一个代表样本。 |
| 实验隔离 | 每次仅变更一个自变量；配置通过 JSON 固化。 | 多项同时变化的数据不能用于归因。 |
| 随机性 | 固定 seed、输入分布和布局。 | 方差异常时增加重复并保存所有原始轮次。 |

### 2.3 论文的正确使用边界

TileSight 论文可提供模型组织方式：资源向量、tile DAG、prologue/steady/epilogue、驻留 tile 与 tail wave、cache 工作集扫描。DeepStack 可提供后续系统级扩展方法。两篇论文中的 NVIDIA/AMD 吞吐、cache cliff、warp、tensor-core、SMEM、occupancy 和通信参数均不可直接用于 S5000。

DeepStack `ae` 的提交 `d8ffb49` 包含 source-visible 的 `src/tilesight` 组件，可作为 Python 模型结构、参数接口和验证方法的实现参考。它仍不是可直接迁移到 S5000 的性能 oracle：部分参考 NoC topology/profile 依赖专有 Linux CPython 3.11 二进制，TileLang/TIR 自动分析接口有未实现部分，分布式计算通信重叠也包含显式 `overlap_ratio` 近似。NVIDIA/AMD 的校准数值和微架构假设不得直接用于 S5000。

## 3. 实验目录与数据契约

后续实现时在工作区根目录创建如下目录（本计划文件不创建它们，以避免把规划与测量混淆）：

```text
benchmarks/       # 每个 benchmark 的 TileLang、MUSA 或 Python 驱动
scripts/          # 一键执行、环境快照、汇总与验证入口
results/raw/      # 不可改写的逐次测量 CSV/JSONL
results/derived/  # 聚合后的曲线、参数表和模型输入
logs/env/         # 环境快照
logs/build/       # TileLang lowering、MUSA 源码、mcc 命令与日志
logs/run/         # 标准输出、stderr、mthreads-gmi 快照
configs/           # 可版本化的 benchmark sweep 定义
```

原始测量 CSV 的最低字段：`run_id,timestamp,benchmark,status,dtype,shape,tile,threads,block,stages,smem_bytes,register_proxy,grid,warmup,repeats,latency_us_median,latency_us_p10,latency_us_p90,throughput,correctness,musa_sdk,mcc_version,tilelang_commit,source_hash,command`。

`status` 只能为 `compile_failed`、`runtime_failed`、`incorrect`、`measured`；只有 `measured` 可输入参数拟合。

## 4. 分阶段执行计划

### P0：环境与可追溯性基线

**工作**

- 记录 GPU 型号/数量、驱动、runtime、SDK、`mcc --version`、Python、PyTorch、torch_musa、TileLang-MUSA 路径和 commit。
- 保存 `mthreads-gmi` 的测试前后快照。
- 运行现有最小 MUSA runtime 例程，保存编译命令、输出和退出状态。
- 编译并运行最小 TileLang-MUSA elementwise kernel；保存 TileLang IR（若可得）、生成 `.mu/.cu` 或等价代码、mcc 命令、编译日志与数值结果。

**验收门槛**

- 能从单一脚本生成完整环境快照。
- 最小 MUSA 与 TileLang kernel 均通过数值正确性检查。
- 能定位每个 kernel 的生成代码和编译日志；否则先调查 TileLang-MUSA backend，停止性能阶段。

### P1：计算路径微基准

**GEMM**

- FP16、BF16、INT8 分开测量，分别建立矩阵路径的有效吞吐曲线。
- 分层扫描：先方阵与大 K 找稳定上界；再扫 skinny/tall、不同 M/N/K、边界尺寸；最后扫 tile、block/线程、pipeline stage。
- 每个点保留实际 tile schedule 与生成代码中的矩阵/向量指令证据；未知指令语义只记录文本，不做推断。

**非 GEMM**

- elementwise（copy、FMA、类型转换）、reduce（不同规约长度与轴）、exp、rsqrt 单独测量。
- 先测纯路径，再测典型 fusion；不可把 fused kernel 的时间反推为单元吞吐。

**验收门槛**

- 每个 dtype 至少有一条正确、重复稳定的 GEMM 曲线。
- 向量、规约、SFU 至少各有一个独立、正确的最小基准。
- 若 TileLang 未生成预期的硬件路径，记录为 lowering 问题，不把结果拟合成硬件参数。

### P2：DDR、cache 与 shared-memory 路径

**DDR/L2**

- 连续读、连续写、读写混合分别扫描；工作集从远小于可能 cache 到明显大于 cache。
- 每条曲线扫描对齐、stride、布局、CTA 并发度与 grid 大小。
- 参数只报告“有效带宽/工作集拐点”；在没有硬件 counter 或可复现实验支持前，不把拐点命名为物理 L2 容量。

**shared memory**

- 建立顺序访问、不同 stride、转置/冲突风险模式的带宽曲线。
- 仅通过实测退化描述冲突风险；不假设 S5000 与 CUDA bank width、bank 数或 warp 规则相同。

**验收门槛**

- 每类曲线有至少三次独立运行，方差被记录。
- 给出稳态有效带宽、候选工作集拐点及其置信范围/不确定性。

### P3：驻留、occupancy 与 wave

**工作**

- 扫 block/线程数、shared-memory 分配和受控寄存器压力（以生成代码或编译器报告为证）。
- 优先尝试 MUSA 公开 occupancy API；若不可得，以吞吐台阶、并发 CTA sweep 和生成代码报告建立经验上界。
- 对每个代表 kernel 测小 grid、整 wave、多个满 wave 与尾 wave，分开记录尾 wave 时间。

**验收门槛**

- 不把“理论 occupancy”当作事实：每一上限需 API、编译器报告或可重复吞吐台阶证据之一。
- 形成 `resident_cta_limit`、约束来源（register/SMEM/block/unknown）和 tail-wave 开销范围。

### P4：参数化与初版单卡模型

将 P1–P3 的可用数据写入版本化 JSON/CSV：矩阵、向量、SFU、DDR、候选 cache tier、SMEM、驻留 CTA、wave/pipeline 参数。模型先采用分段、可解释的有效速率与边界开销；不在第一版引入不可解释的全局回归修正。

最小预测形式为：tile 资源时间向量 + 经实测约束的 resident CTA + prologue/steady/epilogue + full/tail wave 分解。cache 项先采用工作集和访问布局条件下的经验选择；在 cache 行为证据不足时显式输出区间而非单点。

### P5：hold-out 验证与接入决策

**留出集合**

- 未参与 GEMM 拟合的形状与 tile；
- reduce；
- 一个 attention 子核或可证明的 fused kernel；
- 若 TileLang-MUSA 支持，包含至少一个不同 layout/stride 的样本。

**报告**

- 每点实测、预测、绝对百分比误差、误差符号、运行证据链接；
- 按全形状偏差、大工作集偏差、小 grid/tile 偏差、codegen/correctness 问题归类；
- 不设“必须达到”的 MAPE 目标。是否接入取决于误差是否稳定、可解释，及是否能覆盖目标 TileLang schedule 空间。

## 5. 停止条件与风险处理

| 发现 | 当前阶段的结论 | 后续动作 |
|---|---|---|
| TileLang kernel 编译失败 | backend/codegen 未就绪 | 固定最小复现、保存日志，优先修复 lowering。 |
| 数值错误 | schedule 或代码生成不可信 | 缩小到最小错误 shape，禁止采信性能。 |
| 同点性能高方差 | 系统状态或计时不稳定 | 检查时钟/温度/同步/重复次数，暂不拟合。 |
| 带宽曲线无清晰拐点 | cache 容量不可识别 | 仅保留有效带宽曲线与不确定项。 |
| 无 occupancy API/编译器数据 | 上限来源未知 | 用经验台阶标记为经验值，不虚构 register 限制。 |
| hold-out 系统性失配 | 初版模型不足 | 回溯对应资源/布局/wave 假设，增加定向实验；不强行拟合。 |

## 6. 执行顺序与近期里程碑

1. 完成 P0，并确认 TileLang-MUSA 生成路径可审计；
2. 建立 FP16 GEMM、copy/read 与 block-size sweep 的最小垂直切片；
3. 验证该切片的 CSV、日志、正确性和可重复执行格式；
4. 扩展 P1–P3；
5. 参数化后再进行 P5 的 hold-out 验证；
6. 仅在 P5 的误差结构可解释时，开始 TileSight/TileLang cost model 接入评审。

## 7. 当前决策

本文件保留原始 P0–P5 规划作为需求基线。经审核后，正式实施调整为 7 个阶段，并将测量基础设施从环境审计中独立出来；详细修改和进入条件见 `02_s5000_tilesight_calibration_review_v2.md`。
