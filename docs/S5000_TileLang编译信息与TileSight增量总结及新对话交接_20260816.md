# S5000 TileLang编译信息与TileSight增量总结及新对话交接

日期：2026-08-16

工作目录：`/root/s5000-calibration`

范围：TileLang-MUSA编译期信息采集、性能工具侧统一适配、TileSight静态建模、GEMM/Add/ReLU验证、TileOPs manifest锚点与MCU真实对比

## 1. 这轮工作的总体结论

本轮已经把“编译器应该向性能分析工具提供什么”从概念讨论推进为一条真实、可审核的静态建模链路：

```text
TileLang-MUSA PrimFunc
  -> tilelang.tools.compiler_info
  -> tilelang/compiler-info-v1
  -> 性能工具侧严格适配器
  -> tilesight/model-request-v1
  -> TileSight原生GEMM / N_0 / N_N模型
  -> tilesight/model-result-v1
  -> analyze_op正式性能报告
```

当前已经真实打通并验证：

- S5000 FP16 GEMM；
- S5000严格FP32 Add；
- S5000严格FP32 ReLU；
- 编译事实、适配器推导、模型假设、模型预测、运行期实测分层保存；
- TileLang-MUSA内部API采集编译信息；
- 性能工具侧统一ModelRequest/ModelResult；
- TileSight继续自行预测延迟、缓存、流量、occupancy、waves和瓶颈；
- MCU和MUSA Event只用于验证，不作为TileSight日常预测输入。

当前可以认为：

> 静态编译期信息的总体边界、首批Schema、采集API、性能工具适配和GEMM/Add/ReLU验证已经基本完成。剩余重点不是重新设计接口，而是补足最终资源事实、逐TileOp stage、更复杂Elementwise访问模式，以及修正TileSight内部Elementwise缓存/写回模型。

同时必须明确：

> Add/ReLU的warm L2模型目前仍是临时容量规则，已经被更大范围MCU扫描否定。当前“接口和信息分层基本完成”不等于“Elementwise缓存模型已经准确”。

## 2. 用户约束和本轮严格边界

后续工作必须继续遵守：

1. 不使用`musa-perf-collector`目录或其Schema作为本链路组成部分。
2. 编译信息采集模块必须位于`tilelang-musa`内部。
3. 不修改或扩展`tilelang.compile`、`tilelang.jit`、`tilelang.lower`的用户调用接口。
4. 不把MCU实测命中率、延迟或流量写入CompilerInfo或ModelRequest。
5. 不用MCU值覆盖TileSight模型输出。
6. 不把容量扫描的20个点硬编码成查表模型。
7. 不擅自修改TileOPs的S5000架构白名单。
8. 编译器事实和TileSight预测必须分开标识。
9. MCU原始sector命中率和模型的输入读字节命中率不能混用。
10. 工作区存在其他未提交修改；不能把所有dirty diff都归因于本轮工作。

## 3. TileLang-MUSA内部编译信息采集增量

### 3.1 新增内部模块

实现位置：

```text
/root/s5000-calibration/tilelang-musa/tilelang/tools/compiler_info/
  __init__.py
  collector.py
```

正式API：

```python
from tilelang.tools.compiler_info import collect_compiler_information

information = collect_compiler_information(
    prim_func,
    target={"kind": "musa", "arch": "mp_31"},
)
```

返回JSON可序列化字典，Schema为：

```text
tilelang/compiler-info-v1
```

该API只执行编译lowering和事实采集：

- 不启动Kernel；
- 不读取Profiler；
- 不调用TileSight；
- 不计算缓存命中率；
- 不计算性能延迟；
- 不修改原有编译接口签名。

### 3.2 当前采集内容

当前CompilerInfo主要包括：

| 类别 | 采集内容 | 性质 |
|---|---|---|
| 编译身份 | compilation ID、kernel ID、目标target、编译/源码hash | 编译器事实 |
| Kernel Buffer | shape、dtype、scope、输入输出 | 编译器事实 |
| Loop | 循环结构、extent、loop identity | 编译器事实 |
| TileOp | operation ID、类别、读写Buffer、Tile参数 | 编译器事实 |
| 依赖 | 保守数据依赖边 | 静态语义推导 |
| Pipeline | pipeline ID、源循环、请求stage、规划结果和调用注解 | 编译器事实 |
| Layout | 编译期layout注解和关联block | 编译器事实 |
| Lowering | TileOp降低后的调用种类 | 编译器事实 |
| Launch | grid、threads/block、warp size、warps/block、动态Shared | 编译器事实 |
| Workload | 静态FLOPs和逻辑访问字节 | 静态语义推导 |
| Lineage | Source TileOp/Elementwise到最终实现的精确对应 | 编译器事实 |
| Coverage | 各采集阶段是否成功、是否具有exact lineage | 编译器事实 |

### 3.3 Add/ReLU的独立Elementwise血缘

Add/ReLU不是TileOp GEMM。采集器新增了独立的Elementwise路径：

```text
source.kernels[].elementwise_ops
elementwise_lineage
```

当前能够识别：

- Add：`C[i] = A[i] + B[i]`；
- ReLU：`B[i] = max(A[i], 0)`；
- 输入数量和输出数量；
- Buffer shape和dtype；
- 表达式类别；
- 每Block元素数；
- 每线程元素数；
- 连续同索引访问证明；
- 源表达式到最终global store的精确血缘。

`same_index_contiguous`不是看到一维索引后直接猜测，而是在绑定索引替换后，验证block/loop的仿射系数和完整连续覆盖。

若访问是stride、非连续、无法证明或存在复杂表达式，则标记为unsupported，不伪造连续访问。

### 3.4 编译器明确不提供的内容

CompilerInfo当前明确将以下内容标为不可用或非编译事实：

- 真实L1/L2命中率；
- DDR物理流量；
- Kernel运行延迟；
- 各硬件层级利用率；
- 运行期pipeline stage时间；
- scale unit运行期负载；
- 最终MCC寄存器数量；
- spill/local memory真实字节；
- MCU硬件计数器。

## 4. 性能工具侧统一适配增量

### 4.1 新增严格适配器

实现位置：

```text
/root/s5000-calibration/tilelang_compiler_info_adapter.py
```

统一Schema：

```text
输入：tilelang/compiler-info-v1
请求：tilesight/model-request-v1
结果：tilesight/model-result-v1
```

主要API：

```python
prepare_tilesight_request(...)
invoke_tilesight_model(...)
build_tilesight_result(...)
run_tilesight_request(...)
analyze_compiler_info(...)
```

### 4.2 证据分类

ModelRequest和ModelResult保留以下证据边界：

| 分类 | 示例 |
|---|---|
| `compiler_observed` | shape、Tile、stage、grid、线程、warp、动态Shared、表达式、ID、lineage |
| `semantic_derived` | GEMM FLOPs、Add/ReLU表达式类别、连续访问、每线程元素数 |
| `adapter_derived` | operand角色、TileSight `mem_levels`、模型分发、mma_type |
| `model_assumption` | warp tile、raster、随机种子、cache condition |
| `model_predicted` | 延迟、缓存命中、物理流量、利用率、occupancy、waves、资源footprint |
| `unavailable` | 编译器真实寄存器、spill、运行期计数器 |

### 4.3 fail-closed门禁

适配器不会对不完整信息静默猜测。以下情况会拒绝：

- Schema不匹配；
- target不是MUSA `mp_31`；
- Kernel选择不唯一；
- GEMM/Elementwise结构不唯一；
- dtype不受支持；
- accumulator dtype不受支持；
- lineage不完整；
- grid、Tile、线程关系不一致；
- broadcast；
- fusion；
- cast；
- predicate；
- 尾块；
- 非连续Elementwise访问；
- 未显式给Elementwise cache condition。

### 4.4 编译器/模型一致性检查

GEMM会检查：

- grid size；
- FLOPs；
- warps/block；
- dynamic shared memory bytes。

Add/ReLU会检查：

- grid size；
- threads/block；
- warp size和warps/block；
- 每Block元素数；
- 每线程元素数；
- 最终模型结构与编译事实一致。

不一致会抛出`ModelConsistencyError`，不会只记录warning后继续。

## 5. `analyze_op`正式入口增量

实现位置：

```text
/root/s5000-calibration/analyze_op.py
```

新增正式入口：

```bash
python3 analyze_op.py \
  --compiler-info compiler-info.json \
  --cache-condition cold \
  --report-type json
```

当前支持报告形式：

- 完整JSON；
- hierarchical分层JSON；
- summary终端摘要；
- session级聚合。

旧`--tile-metadata`只作为兼容GEMM路径保留，不冒充正式CompilerInfo链路。

当前报告主要输出：

- 预测延迟；
- 吞吐或有效逻辑带宽；
- DDR/L2/Shared/Compute利用率；
- L2预测命中率；
- tiles per SM；
- waves；
- pipeline detail；
- 分层内存流量；
- 各层service cost；
- critical memory path；
- transfer anomaly analysis；
- 编译器/模型ID映射；
- 原始ModelRequest/ModelResult合同。

## 6. GEMM路径完成情况

### 6.1 正式支持范围

当前正式GEMM范围：

- S5000；
- FP16 A/B/C；
- FP32 accumulator；
- A/B Global→Shared；
- GEMM读取Shared；
- C由local accumulator写回Global；
- 静态shape；
- 唯一GEMM和完整exact lineage。

### 6.2 4096³真实编译事实

真实样例：

```text
M=N=K                     4096
Tile                       128×128×64
Pipeline stage             2
Grid                       1024
Threads/block              256
Warps/block                8
模型warp tile              32×64×64（模型假设）
动态Shared                 65536 B
```

CompilerInfo和TileSight四项一致性门禁全部通过。

### 6.3 4096³模型与真实结果

| 指标 | TileSight | 实测 | 差异 |
|---|---:|---:|---:|
| 延迟 | 994.941 μs | MUSA Event 977.720 μs | +1.761% |
| 吞吐 | 138.138 TFLOP/s | 140.571 TFLOP/s | -1.731% |
| L2利用率/吞吐率 | 30.582% | MCU 29.65% | +0.932点 |
| L2命中率 | 92.819% | MCU 90.87% | +1.949点 |
| DDR利用率/吞吐率 | 9.687% | MCU 7.68% | +2.007点 |

GEMM L2 reuse-distance模型继续由TileSight内部计算，编译器只提供shape、Tile、stage、访问和调度事实。

### 6.4 GEMM输出Shared语义修复

真实MUSA源码表明：

```text
FP32 local fragment -> FP16 global C
```

因此正式映射为：

```text
out1 = [1, 0, 1, 2]
```

TileSight原公式可能对Global直写输出产生负Shared分配，错误抵消A/B双缓冲。本轮相关修复将输出Shared分配限制为非负，最终CompilerInfo和TileSight都得到64 KiB动态Shared。

### 6.5 GEMM逐操作数缓存输出

TileSight原reuse-distance计算新增只读detail返回，不改变原聚合公式：

- A、B分别的预测命中请求；
- A、B分别的L2命中率；
- 每operand L2请求/命中/未命中字节；
- 每operand DDR预测读取；
- 聚合命中率保持原计算。

## 7. Add/ReLU Elementwise路径增量

### 7.1 算子分类

| 算子 | TileOPs分类 | TileSight模型 |
|---|---|---|
| Add | binary elementwise | `elementwise_n_n_pipeline_wave` |
| ReLU | unary activation elementwise | `elementwise_n_0_pipeline_wave` |
| GEMM | matmul/GEMM，不是Elementwise | `matmul_pipeline_wave` |

### 7.2 当前严格支持范围

Add：

- FP32；
- 一维静态shape；
- 两输入同shape；
- 单输出；
- 无broadcast/fusion/cast/predicate/tail；
- 连续同索引访问；
- `N`整除每Block元素数。

ReLU：

- FP32；
- 一维静态shape；
- 严格`max(input, 0)`；
- 单输入单输出；
- 其余约束同Add。

### 7.3 真实Tile形状来自编译器

正式CompilerInfo链路得到：

| 算子 | Threads | 每线程元素 | 每Block元素 |
|---|---:|---:|---:|
| Add | 256 | 8 | 2048 |
| ReLU | 256 | 4 | 1024 |

这避免了性能工具先验硬编码Tile。

但旧直接参数入口仍存在一个未修复问题：

```text
analyze_op.py add --N ...
```

该路径仍使用`threads * 4=1024`，与真实Add编译Kernel的2048不一致。正式`--compiler-info`路径正确，本轮所有正式比较均使用编译事实2048。

### 7.4 Global输出映射修正

Add/ReLU standalone Kernel会写Global输出，因此适配器正式映射为：

```text
Add out1  = [1, 0, 1, 4]
ReLU out1 = [1, 0, 1, 4]
```

此前`[0,0,1,4]`表示寄存器中间值，不代表standalone Global输出。该修改发生在性能工具侧`mem_levels`映射，不是修改TileSight的算子表达式逻辑。

## 8. Elementwise缓存模型当前实现和边界

### 8.1 当前临时实现

TileSight当前对N_0/N_N新增：

- 显式`cache_condition=cold|warm`；
- `cache_detail`；
- logical output write；
- kernel-scope DDR write；
- deferred writeback；
- L2 prediction method；
- cold/warm报告语义。

当前L2规则仍是：

```text
cold                         -> 0%
warm且逻辑读取量 <= 60 MiB   -> 99.75%
warm且逻辑读取量 > 60 MiB    -> 0%
```

当前写回规则是：

```text
read + output <= L2 -> 输出作为dirty line留在L2，延后写回
read + output >  L2 -> 全部逻辑输出在当前Kernel窗口写回DDR
```

这些规则已经使报告不再把logical output、kernel-scope write和deferred writeback混成一个字段，但它们仍不是完善的cache模拟。

### 8.2 TileSight已有reuse-distance能力的准确含义

TileSight内部已有reuse-distance/cache模拟，但现有成熟调用路径主要面向GEMM：

- M/N/K；
- A/B Tile；
- grid遍历；
- row/column panel；
- A/B矩阵复用；
- 8-way、cache line、L2容量和SM数量。

Add/ReLU目前没有接入GEMM reuse-distance模型，也不能直接原样复用，因为Elementwise需要描述：

- 一输入或两输入线性流；
- 输入流交错次序；
- output write-allocate；
- dirty output residency；
- 跨Kernel warm初始状态；
- 顺序/反向扫描；
- 可能的冲突和替换行为。

正确方向是扩展TileSight内部Elementwise访问序列/cache模型，而不是让编译器输入命中率。

### 8.3 当前Elementwise报告仍有的输出缺口

- Add当前只有聚合L2结果，没有分别给`in1`、`in2`输出命中/未命中字节；
- `compute_util_pct=0`尚未用对应MCU compute counter验证，不能解释为真实计算单元利用率为0；
- Shared为0符合当前standalone Add/ReLU没有Shared路径，但这不代表所有Elementwise都没有Shared；
- 预测与MCU实测尚未自动合并到正式报告；
- 缓存模型没有输出访问序列、reuse distance或替换过程；
- `critical_memory_path`是模型内部资源路径判断，准确性受命中率和DDR写回时域影响。

## 9. 首批Add/ReLU真实验证

### 9.1 16K小Kernel验证

建立了真实TileLang-MUSA运行入口：

```text
/root/s5000-calibration/workloads/elementwise_tilelang.py
/root/s5000-calibration/benchmarks/compiler_info_elementwise_runtime.py
/root/s5000-calibration/scripts/run_compiler_info_elementwise_mcu_app.sh
```

两类场景：

- cold：输入生成后对256 MiB scratch进行只读L2驱逐，采第一轮目标Kernel；
- warm：连续执行6次，用MCU `-s 5 -c 1`采第六轮。

16K测试确认：

- Add grid=8、block=256、每线程8元素；
- ReLU grid=16、block=256、每线程4元素；
- Kernel源码SHA与编译信息对象一致；
- correctness pass；
- CompilerInfo与TileSight为`all_consistent`。

短Kernel暴露了约6–7 μs固定启动/调度成本，TileSight此前的亚微秒模型不能覆盖小grid固定开销。

### 9.2 逻辑流量事实

16K cold实测中：

| 算子 | 模型逻辑读 | MCU L1下行读 | 差异 | 模型逻辑写 | MCU L1下行写 |
|---|---:|---:|---:|---:|---:|
| Add | 131072 B | 134912 B | +2.93% | 65536 B | 65536 B |
| ReLU | 65536 B | 69376 B | +5.86% | 65536 B | 65536 B |

说明shape、输入数量、元素宽度和逻辑读写量适合作为编译器事实；固定物理事务开销不能反向写进CompilerInfo。

## 10. TileOPs manifest workload扩展验证

### 10.1 manifest锚点

通过官方接口：

```python
from tileops.manifest import load_workloads
```

选择：

- `AddFwdOp`首个workload：`[2048,4096]`，Add使用manifest原始FP32 shape/dtype；
- `ReluFwdOp`首个workload：`[2048,4096]`，manifest只有FP16/BF16，当前适配器仅支持FP32，因此只保留shape并归一为FP32。

TileOPs原生Elementwise Kernel声明只支持SM80/86/89/90，S5000架构31会被拒绝。没有修改白名单，而是使用语义等价的S5000 TileLang-MUSA Kernel。

因此：

- Add是manifest原始shape/dtype的精确锚点；
- ReLU是manifest shape锚点，不是原始dtype精确覆盖；
- Add broadcast workload尚不支持；
- ReLU BF16 decode workload尚不支持。

### 10.2 容量扫描口径

warm口径：

```text
连续执行6次
MCU -s 5 -c 1
采第6次正式Kernel
```

可比L2输入读字节命中率：

```text
max(0, 1 - ddr__read_bytes.sum / l1__downstream_read_bytes.sum)
```

完成：

- 24次有效MCU运行；
- 20个容量点；
- 全部correctness pass；
- Add和ReLU均覆盖小工作集、容量转折区和大工作集。

### 10.3 容量扫描关键结果

| 算子 | 读MiB | 写MiB | 总MiB | MCU字节命中率 |
|---|---:|---:|---:|---:|
| ReLU | 4 | 4 | 8 | 99.8323% |
| ReLU | 16 | 16 | 32 | 99.9756% |
| ReLU | 28 | 28 | 56 | 97.2878% |
| ReLU | 30 | 30 | 60 | 93.1714% |
| ReLU | 32 | 32 | 64 | 87.3271% |
| ReLU | 36 | 36 | 72 | 71.6490% |
| ReLU | 40 | 40 | 80 | 48.3580% |
| ReLU | 48 | 48 | 96 | 6.7746% |
| ReLU | 60 | 60 | 120 | 0.0405% |
| ReLU | 64 | 64 | 128 | 0% |
| Add | 8 | 4 | 12 | 99.9894% |
| Add | 16 | 8 | 24 | 99.9977% |
| Add | 32 | 16 | 48 | 95.8935% |
| Add | 40 | 20 | 60 | 91.3057% |
| Add | 42 | 21 | 63 | 84.8077% |
| Add | 48 | 24 | 72 | 63.2848% |
| Add | 52 | 26 | 78 | 44.0441% |
| Add | 56 | 28 | 84 | 23.6736% |
| Add | 60 | 30 | 90 | 4.0146% |
| Add | 64 | 32 | 96 | 0.8910% |

### 10.4 容量扫描结论

1. `read<=60 MiB -> 99.75%`不成立。
2. 命中率不是在64 MiB附近突然从100%跳到0，而是连续下降。
3. `read+dirty output`比只看read更合理。
4. 相同总字节下ReLU和Add仍有差异，说明不能只拟合一条容量曲线。
5. 输入流数量、交错顺序、write-allocate和dirty residency必须进入模型。
6. 20个点足以否定临时二段常数，但不足以拟合稳定的新模型。

## 11. 2026-08-16 manifest 8M fresh复测

本轮再次对`N=2048×4096=8388608`执行正式CompilerInfo→TileSight，并重新进行MUSA Event与MCU cold/warm采集。

### 11.1 正式编译信息门禁

| 算子 | 模型 | 每Block元素 | cold/warm | 验证 |
|---|---|---:|---|---|
| Add | N_N | 2048 | 均通过 | `all_consistent` |
| ReLU | N_0 | 1024 | 均通过 | `all_consistent` |

### 11.2 延迟和L2读命中

| 算子/场景 | TileSight延迟 | MCU延迟 | 延迟偏差 | 模型L2命中 | MCU派生命中 | 命中偏差 |
|---|---:|---:|---:|---:|---:|---:|
| Add warm | 75.860 μs | 77.009 μs | -1.49% | 0% | 0.897% | -0.897点 |
| ReLU warm | 28.770 μs | 28.196 μs | +2.04% | 99.75% | 89.101% | +10.649点 |
| Add cold | 75.860 μs | 58.742 μs | +29.14% | 0% | 0.114% | -0.114点 |
| ReLU cold | 49.690 μs | 32.434 μs | +53.20% | 0% | 0% | 0点 |

MUSA Event warm中位数：

| 算子 | TileSight | Event中位数 | 偏差 |
|---|---:|---:|---:|
| Add | 75.860 μs | 77.680 μs | -2.34% |
| ReLU | 28.770 μs | 28.800 μs | -0.10% |

### 11.3 分层流量

| 算子/场景 | L2总流量误差 | DDR读模型/实测 | DDR写模型/实测 | DDR总量误差 |
|---|---:|---:|---:|---:|
| Add warm | -0.33% | 67.109/66.578 MB | 33.554/33.474 MB | +0.61% |
| ReLU warm | -0.01% | 0.084/3.657 MB | 33.554/7.741 MB | +195.11% |
| Add cold | -0.41% | 67.109/67.129 MB | 33.554/11.982 MB | +27.24% |
| ReLU cold | -0.01% | 33.554/33.571 MB | 33.554/5.400 MB | +72.20% |

### 11.4 fresh复测结论

1. L1→L2逻辑流量非常准确，四组L2总流量误差不超过0.41%。
2. Add warm的延迟、DDR流量和缓存结论都接近实测。
3. ReLU warm延迟虽然接近，但L2命中高估10.65点、DDR总量高估195%，不能因延迟吻合就判模型正确。
4. cold输入DDR读准确，但输出写回量被明显高估。
5. `writeback_reaches_ddr_in_kernel_scope`把全部输出写回算进当前Kernel窗口，实际很多dirty output在Kernel结束时仍驻留L2。
6. cold延迟高估29%～53%的主要原因之一是DDR写回时域错误。

### 11.5 fresh原始证据状态

2026-08-16的8M fresh复测通过MCU实际执行，临时`.mcu-rep`生成在`/tmp`目录，未作为新的正式结果目录写入工作区。本节保留了本轮审核数字；可长期复核的warm原始报告和容量扫描仍位于：

```text
results/profiling/tileops_manifest_cache_sweep_20260814/
```

若下一轮要把8M cold作为正式校准集，应重新执行并写入带日期的新结果目录，同时保存命令、console、`.mcu-rep`和机器可读汇总，不能只引用本交接表格。

## 12. 命中率口径审核

MCU的：

```text
l2pu__sector_hit_rate.pct
```

是混合sector统计，可能包含读、写分配和不同请求类型。它的分母不等于TileSight当前报告的“输入逻辑读命中率”。

例如8M cold中，MCU原始sector hit仍约25%～34%，但根据DDR read/L1 downstream read推导的输入读字节命中接近0。这不是矛盾，而是统计口径不同。

后续报告必须同时保留：

- MCU原始counter名称和值；
- 模型字段定义；
- 可比派生公式；
- 是否可以计算误差。

不能把`l2pu__sector_hit_rate.pct`直接覆盖`l2_hit_rate_pct`。

## 13. 当前静态编译信息完成度

### 13.1 已基本完成

| 信息 | GEMM | Add/ReLU |
|---|---|---|
| Kernel/operation/pipeline ID | 已有 | 已有Elementwise lineage |
| target | 已有 | 已有 |
| Buffer shape/dtype/scope | 已有 | 已有 |
| 输入输出角色 | 已有 | 已有 |
| grid/block/warp | 已有 | 已有 |
| Tile或每Block元素 | 已有 | 已有 |
| 每线程元素 | 可推导 | 已有 |
| 逻辑读写字节 | 已有 | 已有 |
| 表达式/操作类型 | GEMM TileOp | Add/ReLU已识别 |
| 连续访问证明 | GEMM结构路径 | 严格一维已证明 |
| 整体pipeline stage | 已有 | stage=1模型语义 |
| 动态Shared | 已有 | 0/模型一致 |
| source→final lineage | 已有 | 已有 |

### 13.2 仍缺少

1. 最终MCC/backend真实寄存器数。
2. spill和local memory真实字节。
3. 任意TileOp的逐操作stage，不只是pipeline整体stage。
4. 统一的operand访问顺序和交错描述。
5. raster/swizzle的完全规范化表达。
6. broadcast语义和复用维度。
7. 非连续、stride、predicate、tail和cast的正式模型映射。
8. FP16/BF16 Elementwise路径。
9. 多Elementwise fusion和任意表达式operation count。
10. output write policy及跨Kernel初始cache state的统一模型输入。

其中1～3主要需要编译器继续补事实；4～10需要编译器提供访问事实、TileSight负责硬件预测，不能混成一类。

## 14. 当前报告能满足的指标增量

结合技术方案中性能分析工具指标，本轮增量主要完善：

| 指标方向 | 当前增量 | 状态 |
|---|---|---|
| 5.1 编译期信息 | CompilerInfo Schema、ID、Tile、pipeline、launch、lineage | 首批路径基本完成 |
| 5.2 Tile性能建模 | CompilerInfo驱动GEMM/Add/ReLU原生模型 | 支持范围内完成 |
| 5.3 缓存行为预测 | GEMM逐A/B；Elementwise cold/warm和分层流量 | GEMM较完整，Elementwise模型待修正 |
| 5.5 多粒度报告 | JSON/hierarchical/summary/session | 已接入 |
| 5.6 pipeline瓶颈 | 计算/DDR/L2/Shared critical path | 模型侧已有，Elementwise准确性受cache模型影响 |
| 5.7 跨层搬运异常 | 分层守恒、放大、service pressure | 已接入GEMM/Add/ReLU |
| 5.8 缓存利用异常 | 模型命中、MCU验证和偏差识别 | 已形成验证闭环，尚未自动接入实测 |
| 5.12 指标获取 | 编译事实+模型指标+外部实测分层 | 当前首批路径可用 |
| 5.13 层级感知视图 | L1/L2/DDR/Shared流量和代价 | 模型侧已有 |

不能宣称已完成：

- 任意TileOp覆盖；
- 5.4 Tile-aware runtime trace；
- 任意Elementwise cache模拟；
- 逐stage硬件计数器；
- 实测自动进入正式报告；
- FP16/BF16 Add/ReLU正式支持。

## 15. 测试与审核结果

### 15.1 正式联合环境

使用TileLang-MUSA虚拟环境、正确`LD_LIBRARY_PATH/PYTHONPATH`：

```text
tilelang-musa/testing/python/tools/test_compiler_info.py
tests/test_tilelang_compiler_info_adapter.py
```

结果：

```text
16 passed
```

覆盖：

- Collector是PassInstrument；
- 不改变编译API；
- JSON可序列化；
- Pipeline采集；
- 已有lower artifact finalize；
- Add/ReLU exact lineage；
- stride访问不误判连续；
- ModelRequest证据分类；
- fail-closed；
- GEMM/Add/ReLU原生模型分发；
- cold/warm场景；
- 真实TileLang artifacts；
- 正式CLI保持真实Elementwise Tile。

### 15.2 性能工具独立回归

使用正确DeepStack路径：

```text
tests.test_tilelang_compiler_info_adapter
tests.test_analyze_op
```

结果：

```text
26 passed, 2 skipped
```

两个skip来自该Python进程未加载TileLang-MUSA编译环境；真实TileLang测试已经在上面的联合环境通过，不能把skip解释为功能失败。

默认系统Python若未设置DeepStack路径，会出现`ModuleNotFoundError: deepstack`。这是运行环境问题，不是测试逻辑失败；新对话中应使用项目规定的`PYTHONPATH`。

## 16. 关键文件和证据

### 16.1 TileLang-MUSA

- `tilelang-musa/tilelang/tools/compiler_info/__init__.py`
- `tilelang-musa/tilelang/tools/compiler_info/collector.py`
- `tilelang-musa/docs/compiler_internals/compiler_information.md`
- `tilelang-musa/testing/python/tools/test_compiler_info.py`

### 16.2 性能工具适配和入口

- `tilelang_compiler_info_adapter.py`
- `analyze_op.py`
- `tests/test_tilelang_compiler_info_adapter.py`
- `tests/test_analyze_op.py`

### 16.3 TileSight

- `DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/elementwise_pipeline_wave.py`
- `DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/matmul_pipeline_wave.py`
- `DeepStack-ae-source/src/tilesight/tilesight/fused_op_pipeline_wave/resource_types.py`
- `DeepStack-ae-source/src/tilesight/tilesight/tile_cache/cache_model.py`
- `DeepStack-ae-source/src/tilesight/tilesight/util/L2_hit_rate_flow_sim_reuse_distance.py`

### 16.4 Workload和运行入口

- `workloads/elementwise_tilelang.py`
- `benchmarks/compiler_info_elementwise_runtime.py`
- `scripts/run_compiler_info_elementwise_mcu_app.sh`
- `scripts/analyze_tileops_manifest_cache_sweep.py`

### 16.5 结果

- `results/profiling/compiler_info_tilesight_runtime_20260814/`
- `results/profiling/compiler_info_elementwise_mcu_20260814/`
- `results/profiling/tileops_manifest_cache_sweep_20260814/`
- `results/profiling/tileops_manifest_cache_sweep_20260814/report.md`
- `results/profiling/tileops_manifest_cache_sweep_20260814/sweep_summary.json`

### 16.6 已有说明文档

- `docs/TileLang编译信息到TileSight统一接口说明.md`
- `docs/TileLang编译信息_TileSight_真实4096GEMM审核_20260814.md`
- `docs/S5000_性能分析工具本周完善总结与新对话交接_20260813.md`

注意：`TileLang编译信息到TileSight统一接口说明.md`中“Add/ReLU只支持cold”的部分已经过时。当前接口接受cold/warm，但warm模型仍不准确。本文件为最新交接依据。

## 17. 下一步建议顺序

### 优先级1：Elementwise cache模型

在TileSight内部扩展Elementwise访问序列/cache模型：

- ReLU单输入读流；
- Add双输入交错读流；
- output write-allocate；
- dirty output residency；
- cold/warm初始状态；
- L2容量、cache line、associativity；
- 顺序和反向遍历。

输入应来自编译器访问事实和S5000硬件参数，MCU只用于校准/验证。

### 优先级2：补辨识实验

建议新增：

- 同尺寸cold/warm重复；
- 只读、不写输出对照；
- write-bypass或不同输出策略对照；
- 第二轮反向扫描；
- 分离一输入、两输入和输出写分配；
- 地址偏移/冲突对照。

### 优先级3：修正正式入口一致性

- 修正或限制旧直接Add入口的`threads*4`硬编码；
- 正式分析优先使用CompilerInfo；
- 不允许旧入口伪装成真实编译Tile。

### 优先级4：扩展Elementwise支持范围

- FP16/BF16；
- Add broadcast；
- decode小shape；
- predicate和tail；
- cast；
- 简单fusion；
- 非连续/stride。

### 优先级5：补编译器最终资源事实

- MCC/backend寄存器数；
- spill/local memory；
- 逐TileOp stage；
- 更完整的raster/swizzle和访问顺序。

### 优先级6：运行期闭环

- Event/MCU结果按kernel identity归档；
- 预测与实测自动对照；
- 保留原始counter和派生口径；
- 不把实测反灌CompilerInfo；
- 继续推进真正的5.4 Tile-aware trace。

## 18. 新对话禁止误判的事项

1. 不要把GEMM归类为Elementwise。
2. 不要说Add/ReLU没有L2行为；它们虽通常没有单Kernel内时间复用，但有L2路径、跨Kernel warm和输出dirty residency。
3. 不要说TileSight没有reuse-distance；它已有GEMM专用模型，缺的是Elementwise适配。
4. 不要把MCU命中率作为TileSight输入。
5. 不要把99.75%写成已校准的通用warm结论。
6. 不要把64 MiB附近描述为命中率突然归零。
7. 不要只看read working set，必须考虑output dirty data。
8. 不要把ReLU warm延迟吻合解释为DDR流量正确。
9. 不要把logical output write等同于kernel-scope DDR write。
10. 不要用原始`l2pu__sector_hit_rate.pct`直接计算模型输入读命中误差。
11. 不要修改TileOPs架构白名单来伪造原生S5000覆盖。
12. 不要使用`musa-perf-collector`。

## 19. 新对话推荐启动提示

可在新对话中直接使用：

```text
请先完整阅读：
/root/s5000-calibration/docs/S5000_TileLang编译信息与TileSight增量总结及新对话交接_20260816.md

当前TileLang-MUSA内部compiler-info API、性能工具统一ModelRequest/ModelResult、
GEMM/Add/ReLU正式链路已经打通。GEMM 4096³模型与实测基本一致；Add/ReLU
编译事实和L1→L2逻辑流量准确，但Elementwise warm L2仍使用已被20点MCU容量扫描
否定的99.75%/0%临时规则，DDR输出写回时域也不准确。不要使用musa-perf-collector，
不要修改TileLang公共接口，不要把MCU实测值输入TileSight，不要修改TileOPs架构白名单。
下一步先按文档中的优先级继续分析或实现，并边做边审核。
```

## 20. 最终交接判断

当前阶段可概括为：

```text
编译信息需求边界            基本明确
TileLang-MUSA采集API         已实现
统一ModelRequest/Result      已实现
GEMM正式链路                已验证
严格FP32 Add/ReLU链路        已验证
编译事实与模型一致性门禁      已实现
TileOPs manifest shape锚点    已测试
MCU cold/warm真实对比         已完成
GEMM cache模型               可用
Elementwise cache模型        未完成，临时规则已被否定
Elementwise DDR写回时域       未完成
复杂Elementwise覆盖           未完成
运行期Tile-aware trace        未完成
```

因此新对话不需要重新讨论“编译器信息是否应该给TileSight”，应直接基于已经明确的事实边界继续处理Elementwise cache模型、复杂算子扩展或剩余指标审核。
