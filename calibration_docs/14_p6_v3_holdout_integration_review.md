# P6-v3：冻结 P5 v2 后的独立 Hold-out 与接入评审

**状态：completed — 仅允许 guarded/experimental 窄域点预测接入，并保留显式误差。**  
**日期：2026-07-29（UTC+8）**  
**raw run：** `results/raw/p6_v3_holdout/20260729T050619917012177Z/`

## 1. 验证完整性

P6-v3 在 P5 v2 建模后、任何 P6-v3 kernel 编译前预注册。冻结 scope 为
`configs/p6_v3_holdout_scope.json`；其模型 SHA-256 是
`b94296eb50b904b0e34b80d79535887ad7932b7b08f57fe736c609eb48caa9be`。

`scripts/p6_v3_preflight.py` 在测量前为四个 case 全部生成数值预测；完成后重算的模型 SHA
相同，`p5_model_changed_after_preflight=false`。每个 compute/layout 类别均在三个独立进程中
按编译、正确性、计时顺序执行。P6-v3 结果没有回写 P5 v2。

## 2. 预注册案例与结果

| case | P5 v2 预测 (us) | S5000 实测中位 (us) | 符号误差 (us) | APE | 三进程跨度 |
|---|---:|---:|---:|---:|---:|
| FP16 GEMM 896x1152x1024, 64x64x64, stage 1 | 21.342630 | 24.770199 | -3.427569 | 13.8375% | 0.550% |
| reduce-sum 6144x512, 128 threads | 72.649653 | 75.316000 | -2.666347 | 3.5402% | 0.157% |
| fused `exp(A)+B`, 24M | 214.511876 | 214.438810 | +0.073066 | 0.0341% | 0.114% |
| warm stride-6 copy, 8M | 161.258316 | 161.015205 | +0.243111 | 0.1510% | 0.174% |

四例均为 `status=measured`、`correctness=pass`，通过 15% 跨进程稳定性门槛。四点的
**MAPE（mean APE）为 4.3907%**，**MdAPE（median APE）为 1.8456%**，最大 APE 为
**13.8375%**；不以这些四点统计量当作全局精度。FP16 GEMM 的 13.84% 低估是本轮最大的独立误差，
必须保留，不能通过 P6 反拟合一个修正系数。

完整机器可读结果：

- `results/processed/p6_v3_holdout/holdout_results.csv`
- `results/processed/p6_v3_holdout/holdout_results.json`

生成 MUSA 源码、mcc 命令和运行日志位于 `logs/build/p6_v3_holdout/20260729T050619917012177Z/`；
raw 样本、profiler 事件、配置副本与 GPU 快照均在上述 raw run。

## 3. 旧 P6 的只读回顾

旧 P6 四例在 P5 v2 开发前已被观察，因此不属于独立接入证据。模型冻结后仅做只读 backtest，
结果保存于 `results/processed/p6_original_retrospective_v2/`：GEMM 6.4409%、reduce 7.6541%、
fused 0.5993%、stride 6.9942%。这些数值支持新规则没有完全失真，但没有参与公式、参数、
适用域或本节接入决定。

## 4. 结论与允许接入域

依据 P6-v3，只允许接入一个 **guarded/experimental 的窄域点预测器**，且仅覆盖已验证的
TileLang-MUSA/S5000 单卡 family 与条件范围。四个 family 各只有一个独立 hold-out，尚未验证
未见 family、同一 shape 下多 schedule/tile 的排序能力、置信区间或广泛跨 shape 泛化；因此不能
称为通用性能 oracle，也不能称为已验证的 schedule-ranking cost model。接入实现必须：

- 只调用 P5 v2 已声明 family、dtype、tile/thread、pipeline、shape/工作量范围和 cache selector；
  任何域外条件返回 `unknown`，不得回退到 CUDA/NVIDIA 数值或全局平均速率；
- 将本报告的实测误差作为观测证据而非置信区间。特别是 FP16 GEMM 在一个新 shape 上低估
  13.84%，调用方不能把该预测当精确排序或峰值性能；
- 不支持未验证 tile/stage/warp、BF16/INT8 跨形状 transfer、attention/其他 fused family、cache
  容量推断、未见 occupancy/wave 或多卡通信；
- 保持 P5 v2 SHA、输入 config SHA、生成源码 SHA、测量日期和运行目录可追溯。

P6-v2 保持为未执行状态：其大 GEMM/reduce 特征需要的 development extension 发生可复现的
MUSAGraph runtime reset，未能取得三次可靠参数；这不是可以用推测补齐的 S5000 性能数据。

## 5. 后续建议

1. 为 FP16 GEMM 增加独立 hold-out shape/tile，优先解释当前 13.84% 低估是否来自 grid/CTA
   特征不足或 codegen/resource 差异；不得直接用本点校正模型。
2. 对 stride 建立更多 selector（N、cache condition、vector、alignment）和局部留出测试；当前
   只验证了 8M warm、stride6 的插值点。
3. 将发生 `operation not permitted` 的大 GEMM Graph replay 缩小为最小复现，分别检查 graph
   capture、replay 和 profiler 路径；在稳定前不得扩展该工作量域。
4. 只有每个新增 schedule 域都有冻结模型后的独立 hold-out，才扩大 TileSight/TileLang cost
   model 的可用范围。
