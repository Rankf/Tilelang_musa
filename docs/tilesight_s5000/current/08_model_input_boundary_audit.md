# TileSight 输入与派生结果边界审核

审核日期：2026-09-21。

静态分析的基础输入为算子语义、dtype/存储宽度、Shape、Tile 形状、线程配置、
内存层级访问描述与目标架构；冷/热缓存是分析场景条件。
架构 Profile 包含预先配置或校准的服务能力，不要求每次预测重新采集性能数据。

## 模型负责的计算

Elementwise 核心从 Shape/Tile 推导 Grid、Tile 数与每线程 Tile；从访问语义、
存储宽度推导逻辑流量和计算量；估算 Shared Memory、寄存器 footprint，进而
计算 Occupancy、Wave、资源服务时间与预测延迟。当前单阶段 Elementwise 内部
设定 stage=1；不能表述成支持任意外部 stage 输入。

`occupancy_hint` 是可选的二进制驻留信息，没有 hint 时执行模型估算。
当前并非所有 compiler-info 寄存器/Shared 字段都会替换 footprint；其中部分
只是证据元数据，不能宣称所有编译资源均已精确接入。

## 编译和运行时接口

compiler-info 适配器在编译产物提供 Grid 时检查它与 Shape/Tile 推导值的一致性；
若产物缺少 Grid，则直接采用模型推导值，并在 `unavailable` 标记缺少该观测。
传给核心的仍是 Shape、Tile、线程和访问描述。直接静态分析入口不要求
compiler-info、Event、MCU 或 MSYS。
不要为了允许纯模型调用而删除编译产物身份/launch 校验。

运行时 bundle 提供独立实测证据，用于比较与诊断，不是静态预测必填输入。
完整逻辑输出写回保持主预测口径。连续 CTA 与 cold=0.60 仍为可选实验配置，
0.60 是经验时间缩放值，不能解释为已测得的硬件重叠率或重复成本比例。

## 本次修正

- 修复 N_1 广播 Elementwise 返回时引用未定义 `cache_detail/cache_condition`
  的错误，恢复仅凭两个 Shape 和 Tile 配置执行分析的路径；保留原广播流量估算。
- compiler-info 不再为 FP16 Exp/Log/Sqrt 注入旧 `overlap_adapter_key`。
- N_0 接受该旧参数以兼容调用，但不再借其启用算子专用 dirty-retention。
  SFU 服务 Profile 保留。
- 增加 FP16/BF16/FP32、两种 Wave 模式下无运行数据的静态预测回归，以及广播
  路径和旧修正参数无效化回归。

本次没有更换吞吐参数、重拟合 Wave 或运行新的 GPU 性能实验。
