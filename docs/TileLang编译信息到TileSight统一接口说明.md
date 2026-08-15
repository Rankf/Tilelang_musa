# TileLang编译信息到TileSight统一接口说明

## 1. 当前正式链路

```text
tilelang/compiler-info-v1
  -> 性能工具侧严格适配器
  -> tilesight/model-request-v1
  -> TileSight原生matmul / N_0 elementwise / N_N elementwise模型
  -> tilesight/model-result-v1
  -> analyze_op性能报告
```

实现位置：

- TileLang编译信息采集：`tilelang-musa/tilelang/tools/compiler_info/`
- 性能工具侧适配器：`tilelang_compiler_info_adapter.py`
- 性能分析入口：`analyze_op.py --compiler-info`

这条链路不依赖`musa-perf-collector`，不修改`tilelang.compile`、
`tilelang.jit`、`tilelang.lower`接口，也不启动Kernel或读取Profiler。

## 2. 采集和调用

TileLang侧通过已有编译API得到JSON可序列化信息：

```python
import json
from tilelang.tools.compiler_info import collect_compiler_information

information = collect_compiler_information(
    prim_func,
    target={"kind": "musa", "arch": "mp_31"},
)
with open("compiler-info.json", "w", encoding="utf-8") as output:
    json.dump(information, output, indent=2, ensure_ascii=False)
```

性能工具通过统一入口分析：

```bash
python analyze_op.py \
  --compiler-info compiler-info.json \
  --report-type json \
  -o report.json
```

Python接口：

```python
from tilelang_compiler_info_adapter import (
    prepare_tilesight_request,
    run_tilesight_request,
)

# GEMM
request = prepare_tilesight_request("compiler-info.json")

# Add/ReLU：当前原生模型只支持显式冷缓存场景
request = prepare_tilesight_request(
    "compiler-info.json", cache_condition="cold"
)
result = run_tilesight_request(request, s5000_arch)
```

旧的`--tile-metadata`入口继续保留为兼容路径，但不会冒充正式的
`compiler-info -> ModelRequest`链路。

## 3. 证据边界

统一请求和结果使用以下分类，报告中原样保留：

| 分类 | 当前内容 |
|---|---|
| `compiler_observed` | Shape、表达式/访问、Pipeline stage、Grid、线程/warp、动态Shared、FP32 accumulator、ID和lineage |
| `semantic_derived` | GEMM FLOPs；Add/ReLU表达式分类、连续访问证明、每Block/线程元素数 |
| `adapter_derived` | 操作数角色、TileSight `mem_levels`、模型分发、`mma_type` |
| `model_assumption` | GEMM warp tile/raster/cache随机种子；Elementwise显式冷缓存场景 |
| `model_predicted` | 延迟、利用率、L2命中率、Pipeline时间、驻留Tile、wave和模型资源量 |
| `unavailable` | 编译器真实寄存器数、spill、混合精度精确寄存器footprint、运行期实测 |

适配器不会为未知值静默填数。不完整lineage、非MUSA `mp_31`、不支持的dtype、
结构不唯一或静态事实不一致都会直接拒绝。Elementwise还会拒绝broadcast、fusion、
cast、predicate、尾块、非连续访问和非冷缓存场景。

## 4. 一致性门禁

TileSight调用后，以下值必须与编译器事实一致：

- Grid size；
- FLOPs；
- warps per block；
- dynamic shared memory bytes。

Add/ReLU调用后还会核对Grid、线程数、warp数和每Block元素数；TileSight继续自行
计算延迟、利用率、缓存/物理流量、资源footprint、occupancy和waves，这些值不会
提前写入ModelRequest。

任一不一致都会抛出`ModelConsistencyError`，不会降级为警告。真实4096³
FP16 GEMM当前验证值为：

```text
op_shape                    [4096, 4096, 4096]
tb_shape                    [128, 128, 64]
wp_shape                    [32, 64, 64]（模型假设）
stage_num                   2
grid_size                   1024
threads_per_block           256
warps_per_block             8
dynamic shared（编译器）    65536 B
dynamic shared（TileSight） 65536 B
```

TileSight cache遍历内部使用NumPy随机排列。适配器在锁内保存调用者RNG状态、使用
固定种子运行并恢复状态，使相同Request可复现，同时不改变TileSight原生函数签名。

## 5. 当前能力和限制

当前正式支持以下三类已验证路径：

- S5000 FP16 GEMM：A/B由Global进入Shared，
GEMM使用Shared输入和FP32 fragment accumulator，C写回Global，且编译产物具有
完整静态Shape、唯一GEMM、唯一源Pipeline和精确TileOp lineage；
- S5000 FP32 Add：一维、无broadcast/fusion/cast/predicate、连续同索引访问、
  `N`整除每Block元素数，并具有精确elementwise lineage；
- S5000 FP32 ReLU：约束同Add，表达式严格为`max(input, 0)`。

Add/ReLU的`tb_shape`来自编译器证明的真实Block覆盖量，不再硬编码为
`threads * 4`。例如当前真实样例Add为`256 * 8 = 2048`，ReLU为
`256 * 4 = 1024`。

`mem_levels.out1`使用`[1, 0, 1, 2]`表示FP32 fragment直接写入FP16全局输出，
与生成的MUSA源码一致。TileSight原footprint公式会让该路径产生负的输出Shared项，
错误抵消A/B双缓冲；现已将输出Shared分配限制为非负，最终仍与编译器的64 KiB一致。
TileSight原生模型只有一个输出元素位宽，不能同时精确表达
FP32 accumulator与FP16外部输出，因此当前寄存器footprint只作为模型预测，不作为
编译器事实，也不参与编译器一致性验收。

运行期延迟、硬件计数器、实测缓存命中率和实测spill仍需后续运行时采集链路提供，
不能从本接口推断为实测值。
