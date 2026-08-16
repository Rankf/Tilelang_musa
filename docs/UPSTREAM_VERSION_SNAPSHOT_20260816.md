# TileLang-MUSA 与 TileOPs 上游升级回溯记录

记录时间：2026-08-16 12:14:07 +08:00

工作目录：`/root/s5000-calibration`

## 升级前基线

| 仓库 | 远端 | 升级前本地提交 | 当时最新远端提交 | 状态 |
|---|---|---|---|---|
| 主仓库 | `https://github.com/Rankf/Tilelang_musa.git` | `bbe10145d2c8f81d569ed5aa3811c1efd7e03432` | `bbe10145d2c8f81d569ed5aa3811c1efd7e03432` | `main`，工作区另有用户结果和文档改动 |
| TileLang-MUSA | `https://github.com/tile-ai/tilelang-musa.git` | `4d666968a005846314e9a84d8ff121c5dbffa9ee` | `475bf79063776fc268d01bb8cabde24a92d753f8` | detached、存在21个文件的本地CompilerInfo等开发改动 |
| TileOPs | `https://github.com/tile-ai/TileOPs.git` | `fed27f684a42b0a4a87bc7beae909c5402e21100` | `6fcdaf8fd35fa0aec614f4179ce3bb8a7c7fbf66` | `main`，干净，落后61个提交 |

TileLang-MUSA 升级前版本描述为 `4d66696-dirty`；TileOPs 为 `fed27f6`。

## 可回溯Git引用

### TileLang-MUSA

- 原始基线提交：`4d666968a005846314e9a84d8ff121c5dbffa9ee`
- 本地改动封存分支：`codex/s5000-pre-upstream-20260816`
- 本地改动封存提交：`2d3f939`（`WIP: preserve S5000 compiler information work before upstream update`）
- 该提交完整保存升级前21个修改/新增文件，共3156行新增、25行删除。

回看升级前完整状态：

```bash
git --git-dir=/root/s5000-calibration/tilelang-musa/.git \
    --work-tree=/root/s5000-calibration/tilelang-musa \
    switch codex/s5000-pre-upstream-20260816
```

### TileOPs

- 原始基线提交：`fed27f684a42b0a4a87bc7beae909c5402e21100`
- 备份分支：`codex/pre-upstream-20260816`

回看升级前版本：

```bash
git --git-dir=/root/s5000-calibration/TileOPs/.git \
    --work-tree=/root/s5000-calibration/TileOPs \
    switch codex/pre-upstream-20260816
```

## 网络与获取说明

容器原Docker DNS `127.0.0.11`出现间歇性解析失败。升级前已将当前容器的`/etc/resolv.conf`调整为：

```text
nameserver 223.5.5.5
nameserver 119.29.29.29
nameserver 1.1.1.1
options timeout:2 attempts:3 rotate ndots:0
```

GitHub主页、API和Git远端均已验证可连接。TileLang-MUSA原仓库是partial clone，普通fetch还会触发缺失blob和子模块下载，因此升级采用完整上游工作树获取后再移植本地改动。

升级收尾时普通`git ls-remote`仍会在GitHub Git端点间歇性超时，因此又通过GitHub API直接核验`main`：2026-08-16核验结果仍分别为TileLang-MUSA `475bf79063776fc268d01bb8cabde24a92d753f8`、TileOPs `6fcdaf8fd35fa0aec614f4179ce3bb8a7c7fbf66`，与本次采用的上游提交一致。

## 升级后版本

| 仓库/组件 | 升级后状态 | 说明 |
|---|---|---|
| TileLang-MUSA上游 | `475bf79063776fc268d01bb8cabde24a92d753f8` | 官方标签`v0.1.12+musa.1` |
| TileLang-MUSA本地迁移分支 | `codex/s5000-upstream-20260816`，HEAD `e30232c9b2a95329af4362c8e941432001042875` | 上游之上保留CompilerInfo/Provenance，并完成新版TIR兼容 |
| TileLang-MUSA已安装版本 | `0.1.12+musa.1.gite30232c9` | `.venv-tilelang-musa` editable安装，完整重建成功 |
| 嵌套TVM | 上游基线`ab686f3e21435b146bc303538bca079dec07e00b`，补丁`391bc332c94175c40a0b0a889bf276819c08d57b` | 分支`codex/tilelang-0.1.12-dtype-compat-20260816` |
| TileOPs | `6fcdaf8fd35fa0aec614f4179ce3bb8a7c7fbf66` | `main`与`origin/main`一致，工作区干净 |

TileLang-MUSA在官方最新提交之上的三个本地迁移提交：

```text
e7da9e7  WIP: preserve S5000 compiler information work before upstream update
3f5176b  Adapt S5000 compiler info to TileLang 0.1.12 dependencies
e30232c  fix: adapt compiler info collector to new TIR APIs
```

其中`e7da9e7`是从封存提交移植出的功能提交，不代表遗失了上游代码；当前分支相对`origin/main`为`ahead 3`。

## 更新内容与影响分析

### TileLang-MUSA：0.1.10到0.1.12

上游跨度为471个文件、39903行新增、11713行删除。主要变化如下：

1. 后端与编译流水线重构
   - 引入backend registry，host/device codegen和target normalize/detect改为后端分发；
   - `tilelang/engine/phase.py`被删除，PassPipeline迁入`tilelang/backend/pass_pipeline/`并由CUDA、MUSA、ROCm、CPU各自组装；
   - backend-specific transform进入独立命名空间，新增按后端materialize kernel launch。
2. 编译、调试和调度能力
   - 新增tile scheduler、pass diff和pass visualizer；
   - software pipeline、async copy lowering、layout inference、向量化回退诊断有较大调整；
   - 新增scan算子及相关后端实现。
3. JIT、缓存和运行时
   - target入口迁至`tilelang.backend.target`；
   - JIT diagnostics、NVCC timeout、lazy kernel lookup和扩展缓存更新；
   - CPU/LLVM backend及缓存目录行为重构。
4. 数据类型和MUSA修复
   - 新增`float4_e2m1_unpacked`、MUSA `fp8_e8_t`等类型；
   - 修复MUSA barrier thread extent、pipeline适配、copy binding、向量拓宽以及runtime对直接MUSA依赖等问题；
   - 增加MUSA lazy-loading stubs。

### S5000本地功能如何迁移

- CompilerInfo/Provenance公共使用方式保持不变，没有扩大或修改TileLang公共编译接口；
- Provenance pass从已删除的`engine/phase.py`迁至新backend pass pipeline，并接入MUSA和CUDA pipeline；
- kernel cache合并时同时保留上游锁粒度优化和本地instrumented plan hash隔离；
- `lower_tile_op.cc`同时保留新版LowerArgs与本地Provenance lineage；
- Collector适配`determine_target`新位置、绑定For形式的kernel launch以及TIR表达式禁止隐式布尔判断；
- 没有恢复上游已删除且不再执行的`engine/phase.py`。

### 官方组合中的TVM dtype缺项

官方0.1.12源码固定使用code 131的`float4_e2m1_unpacked`，但其固定TVM提交未注册该dtype，也没有对应C++判断方法，官方源码与固定TVM直接组合会构建失败。本次在嵌套TVM增加最小兼容补丁：

- `include/tvm/runtime/data_type.h`：补齐类型判断；
- `src/target/datatype/registry.cc`：补齐dtype注册；
- TVM补丁提交：`391bc33`；
- 父TileLang提交`3f5176b`记录了该submodule指针。

这属于上游发布组合缺项的构建兼容补丁，不是S5000性能模型修改。

### TileOPs：61个上游提交

上游跨度为573个文件、29032行新增、22041行删除。主要变化如下：

1. Python包从仓库根目录的`tileops/`迁到标准`src/tileops/`布局，并增加发布内容校验；
2. 新增backend registry/dispatch，第三方target可以注册kernel；
3. kernel选择从per-op谓词改为候选注册表，并在forward阶段选择；dtype也改为forward时读取；
4. benchmark timing改为交错比较，CUPTI采集、correlation-id归因、运行环境记录和nightly coverage报告大幅重构；
5. 新增或优化W4A16、attention/paged GQA、FFT、adaptive pooling、reduction、GroupNorm/InstanceNorm、Gated DeltaNet等算子；
6. workload/input构造进一步移入workloads层，manifest仍通过`tileops.manifest.load_workloads(op_name)`读取。

对本项目的直接兼容影响是Python搜索路径必须从`TileOPs`改成`TileOPs/src`。已更新：

- `analyze_op.py`；
- `export_tile_metadata.py`；
- `scripts/analyze_tileops_manifest_cache_sweep.py`。

没有修改TileOPs的S5000架构白名单，也没有把MCU实测结果输入TileSight。

### TileSight随新版TIR的兼容

新版TVM用绑定For节点表达kernel launch，且`thread_binding`对象禁止Python隐式布尔判断。为保持4096 GEMM metadata链路可用，`DeepStack-ae-source/src/tilesight/tilesight/tir_interface/analyze_tir/core.py`做了两项最小兼容：

- `thread_binding is not None`显式判断；
- launch采集同时识别绑定For节点。
- 静态FLOPs计算不再把新版TVM的`blockIdx/threadIdx`绑定For当作普通重复循环；block轴由`grid_size`统一计入，TileOp本身已表示整个线程块的工作量。

该修改只恢复编译事实读取，不改变TileSight算子模型、缓存模型或预测参数。

## 升级后验证

TileOPs manifest验证：

```text
AddFwdOp   2 workloads，首锚点 [2048, 4096]
ReluFwdOp  2 workloads，首锚点 [2048, 4096]
```

- `analyze_op`在新版`src/`布局下导入成功；
- 容量扫描分析验证24个MCU runs、覆盖20个case；
- `.venv-tilelang-musa`最终联合回归：`40 passed`；覆盖TileLang CompilerInfo、Provenance、性能工具适配器、`analyze_op`及新版TIR分析，也包含4096³ GEMM metadata链路；
- 系统Python独立性能工具口径：`24 passed, 2 skipped`；两个skip是系统进程未加载TileLang-MUSA编译环境；
- TileOPs不依赖GPU的manifest、backend、发布结构和benchmark基础组：`190 passed`；另有2个CUDA专属benchmark用例在torch_musa环境误判CUDA可用后读取NVIDIA专属`L2_cache_size`字段而失败，未修改上游快照；
- 完整editable重建成功，共编译561个单元；旧安装`0.1.10+musa.1.git4d666968`已由`0.1.12+musa.1.gite30232c9`替换。

## 回退方法

回退前先确认对应嵌套仓库工作区干净。不要在主仓库执行递归reset，也不要覆盖主仓库中原有结果文件改动。

TileLang-MUSA回看升级前完整本地状态：

```bash
git -C /root/s5000-calibration/tilelang-musa \
    switch codex/s5000-pre-upstream-20260816
```

返回本次升级后的迁移分支：

```bash
git -C /root/s5000-calibration/tilelang-musa \
    switch codex/s5000-upstream-20260816
```

TileOPs回看升级前版本与返回最新版本：

```bash
git -C /root/s5000-calibration/TileOPs switch codex/pre-upstream-20260816
git -C /root/s5000-calibration/TileOPs switch main
```

TVM补丁分支可由以下引用定位：

```text
codex/tilelang-0.1.12-dtype-compat-20260816 -> 391bc332c94175c40a0b0a889bf276819c08d57b
```

回退源码后需要在`.venv-tilelang-musa`中重新进行对应版本的editable构建/安装，不能仅切换Git引用后继续使用当前二进制。
