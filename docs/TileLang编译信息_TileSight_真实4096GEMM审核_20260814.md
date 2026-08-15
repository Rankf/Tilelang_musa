# TileLang编译信息、TileSight与真实4096³ GEMM审核

日期：2026-08-14  
设备：MTT S5000，Driver 3.3.5-server，MUSA SDK 4.3.5  
算子：FP16 GEMM，M=N=K=4096，Tile=128×128×64，Pipeline stage=2

## 1. 本次实际执行链路

本次不是读取历史模型结果。实际执行了：

```text
同一次TileLang编译
  ├─ CompilerInfoCollector -> tilelang/compiler-info-v1
  └─ 可执行main_kernel -> S5000真实运行

compiler-info
  -> tilesight/model-request-v1
  -> TileSight原生模型
  -> tilesight/model-result-v1
  -> analyze_op报告

真实Kernel
  ├─ 全矩阵torch.testing.assert_close正确性检查
  ├─ 3轮×40次MUSA Event计时
  └─ Moore Perf Compute默认section硬件计数器
```

编译器记录的Kernel源码SHA256与实际运行Kernel源码SHA256完全一致：

```text
f20c6e4c77c0adfcfce0603cf405113e0b9f50ab0c12e955747c75509151fad7
```

因此模型输入和实测对象属于同一份生成源码，不是仅靠Shape推定为同一Kernel。

## 2. 性能分析工具输出

| 字段 | 输出 |
|---|---:|
| Shape | 4096×4096×4096 |
| Tile | 128×128×64 |
| 最终线程数 | 256 |
| 最终warp数 | 8 |
| 模型warp tile | 32×64×64（明确标为模型假设） |
| Pipeline stage | 2 |
| Grid CTA | 1024 |
| 编译器动态Shared | 65536 B |
| TileSight动态Shared | 65536 B |
| 预测延迟 | 994.941 μs |
| 预测吞吐 | 138.138 TFLOP/s |
| Compute利用率 | 74.945% |
| Shared利用率 | 74.945% |
| L2利用率 | 30.582% |
| DDR利用率 | 9.687% |
| L2命中率 | 92.819% |
| resident tiles/SM | 3 |
| 模型waves | 5.689 |

Grid、FLOPs、warps/block和动态Shared四项编译器/模型一致性检查全部通过。

## 3. MUSA Event真实结果

正确性：全矩阵比对通过。

| 轮次 | 样本数 | 中位数 | P10 | P90 | P90-P10/median |
|---|---:|---:|---:|---:|---:|
| 1 | 40 | 1009.300 μs | 970.280 μs | 1042.668 μs | 7.172% |
| 2 | 40 | 976.740 μs | 962.516 μs | 995.112 μs | 3.337% |
| 3 | 40 | 977.720 μs | 963.400 μs | 1001.900 μs | 3.938% |

三轮中位数的中位数为977.720 μs，对应140.571 TFLOP/s。三轮中位数跨度为
3.330%，说明当前机器状态仍存在频率/调度波动。另两次独立运行得到1012.00 μs和
1012.16 μs，模型相对这些实测点的绝对误差也均小于1.8%。

使用三轮中位数比较：

| 指标 | 模型 | MUSA Event | 模型相对实测 |
|---|---:|---:|---:|
| 延迟 | 994.941 μs | 977.720 μs | +1.761% |
| 吞吐 | 138.138 TFLOP/s | 140.571 TFLOP/s | -1.731% |

不能据此认为模型误差已经固定为1.761%；当前证据支持的是本次多个独立窗口中模型
延迟均落在实测波动附近，单点绝对误差约1.4%～1.9%。

## 4. Fresh MCU硬件计数器

本次重新执行Moore Perf Compute 1.3.0默认section，得到：

MCU runner中的符号名是`gemm_4096_kernel`，正式链路中的符号名是`main_kernel`；
将唯一的符号名差异归一化后，两份生成源码逐字节相同，归一化SHA256仍为
`f20c6e4c...51fad7`。

| MCU字段 | 实测 |
|---|---:|
| Duration（显示精度） | 1.08 ms |
| L2 Cache Throughput | 29.65% |
| DDR Throughput | 7.68% |
| L2 Cache Hit Rate | 90.87% |
| Block Size | 256 |
| Grid Size | 1024 |
| Dynamic Shared/block | 65536 B |
| Waves Per MP | 1.42 |

可比较项：

| 指标 | 模型 | MCU | 差值 | 审核 |
|---|---:|---:|---:|---|
| L2吞吐率 | 30.582% | 29.65% | +0.932百分点 | 基本一致 |
| L2命中率 | 92.819% | 90.87% | +1.949百分点 | 接近但存在偏差 |
| DDR吞吐率 | 9.687% | 7.68% | +2.007百分点 | 仍需核对流量和分母 |

MCU的Compute(MP)、Shared pipe和Waves Per MP与TileSight对应字段定义不同，不能直接
计算模型误差。MCU replay期间应用自身的墙钟时间也被Profiler显著扰动，不能替代
MUSA Event基线。

## 5. 审核发现并修复的问题

生成的MUSA源码显示：

```text
float c_local[128];
...
C[...] = (half_t)c_local[i];
```

C由FP32 fragment直接写到FP16 Global，没有输出Shared分配。因此正式映射应为：

```text
out1 = [1, 0, 1, 2]
```

此前为了让Shared footprint从0恢复为65536 B而改成`[1,1,1,2]`，虽然数值通过，
但内存路径语义不准确。根因是TileSight原公式对Global直写输出计算出负的Shared项，
抵消了A/B双缓冲。

本次修复为：

1. 适配器恢复`out1=[1,0,1,2]`；
2. TileSight输出Shared footprint使用
   `max(out1_level[1] - out1_level[0], 0)`，禁止负分配；
3. 重新验证编译器和模型Shared均为65536 B；
4. 重新运行适配器及原有`analyze_op`回归。

这是语义修复，不是为了通过门禁而放宽一致性检查。

## 6. 仍然有效的边界

- 编译器事实、适配器推导、模型假设、模型预测、运行期实测继续分开保存。
- FP32 accumulator是编译器事实；TileSight寄存器footprint仍不是编译器寄存器数。
- MCU本次报告的`Registers Per Thread=0`不可作为真实寄存器占用。
- Torch MUSA Profiler因MUPTI/MT-Perf无法连接KM/HW而没有事件输出，已标记
  `unavailable`，没有用0值冒充实测。
- Event/MCU仍是Kernel级数据，不是按`kernel_id/pipeline_id/op_id`归集的Tile-aware
  运行期trace，因此不能据此宣称指标5.4已经满足。
- 本链路未使用`musa-perf-collector`。

## 7. 产物

- `results/profiling/compiler_info_tilesight_runtime_20260814/compiler_info.json`
- `results/profiling/compiler_info_tilesight_runtime_20260814/model_report.json`
- `results/profiling/compiler_info_tilesight_runtime_20260814/runtime_event.json`
- `results/profiling/compiler_info_tilesight_runtime_20260814/comparison.json`
- `results/profiling/compiler_info_tilesight_runtime_20260814/gemm_4096_fresh.mcu-rep`
