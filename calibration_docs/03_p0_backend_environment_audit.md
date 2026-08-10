# P0 后端与环境审计报告

状态：completed  
执行日期：2026-07-28  
范围：环境、源码、编译与运行路径审计；不包含性能测量或微架构参数推断。

## 1. 验收结论

P0 已通过，可以进入 P1 测量基础设施阶段。

已用本机证据闭合两条路径：

1. 原生 MUSA：`mcc` 编译并显式链接 `libmusart`，S5000 runtime 返回 1 个设备；
2. TileLang-MUSA：受控源码在固定提交上完成 editable build，真实生成 MP31 MUSA 源码和 `mcc` 命令，在 S5000 上运行 float32 elementwise-add 并通过数值检查。

这只证明后端路径可追踪且最小 kernel 可编译、可运行、数值正确；不构成任何吞吐、cache、occupancy、寄存器或矩阵指令结论。

## 2. 环境快照

| 项目 | 实测值 | 证据 |
|---|---|---|
| GPU | 1× MTT S5000，81920 MiB | `logs/env/mthreads-gmi_before.txt` |
| 驱动 | MUSA 3.3.5-server | `logs/env/mthreads-gmi_before.txt` |
| SDK/compiler | `/usr/local/musa-4.3.5`；mcc 4.3.5；clang 14.0.0 | `logs/env/mcc_version.txt` |
| Python | 3.10.12 | `logs/env/python_version.txt` |
| PyTorch | `2.7.1a0+gite2d141d` | `logs/env/python_musa_packages.txt` |
| torch_musa | `2.7.1+5ee0a64` | `logs/env/python_musa_packages.txt` |
| runtime capability | `(3, 1)` | `logs/env/tilelang_import.txt` |
| TileLang-MUSA | `0.1.10+musa.1.git4d666968` | `logs/env/tilelang_import.txt` |

环境可由 `scripts/p0_capture_environment.sh` 重新采集。

## 3. 原生 MUSA runtime 路径

输入为 `musa_smoke.mu`。仅执行 `mcc -O2` 时曾因未链接 runtime 报 `undefined reference to musaGetDeviceCount`；修正命令是：

```bash
mcc -O2 /root/s5000-calibration/musa_smoke.mu \
  -L/usr/local/musa-4.3.5/lib -lmusart \
  -o /root/s5000-calibration/results/derived/p0_musa_smoke
```

运行结果为 `musaGetDeviceCount=0 devices=1`，退出状态为 0。构建、hash、stdout/stderr 和运行前后 GPU 状态分别保存在 `logs/build/p0_musa_smoke_*` 与 `logs/run/p0_musa_smoke_*`。

## 4. TileLang-MUSA 受控源码与构建

- 来源：`https://github.com/tile-ai/tilelang-musa.git`
- 本地路径：`/root/s5000-calibration/tilelang-musa`
- 固定提交：`4d666968a005846314e9a84d8ff121c5dbffa9ee`（`Prepare public`）
- Python 包：`tilelang_musa`，源码要求 Python `>=3.10`
- MUSA backend：`src/backend/musa/`
- 测试：`testing/musa/`，其中存在面向 capability 3.1 的路径

构建环境为 `/root/s5000-calibration/.venv-tilelang-musa`，使用 system site packages 复用已验证的 torch/torch_musa。首次 editable build 因默认并行度过高导致 `cc1plus` 被 OOM kill；这不是 MUSA backend 编译错误。限制并行度为 4 后成功：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 USE_MUSA=ON \
MUSA_HOME=/usr/local/musa-4.3.5 \
/root/s5000-calibration/.venv-tilelang-musa/bin/python -m pip install \
  -e /root/s5000-calibration/tilelang-musa -v --no-build-isolation
```

失败与成功日志分别为 `logs/build/p0_tilelang_editable_build.log` 和 `logs/build/p0_tilelang_editable_build_parallel4.log`。

所需 MUSA/TVM 子模块已经固定并初始化：MUTLASS `d31e365...`、tvm_musa `ab686f3...`、TVM-FFI 及其 dlpack/libbacktrace。TVM 树中 CUTLASS、flash-attn 等可选 CUDA 子模块未初始化；不能表述为“所有递归子模块完整”。精确状态见 `logs/env/tilelang_musa_submodules.txt`。

## 5. 首个 TileLang kernel 与兼容修补

仓库自带的 `testing/musa/common/test_add_global.py` 首次编译真实 MP31 kernel 时失败。生成命令已证明实际目标为：

```text
mcc --cuda-device-only --cuda-gpu-arch=mp_31 ... -D__MUSA_ARCH_LIST__=310 ...
```

失败不是 elementwise-add 本体，而是公共模板头无条件定义 FP16 NaN-aware min/max。MUSA 4.3.5 的 FP16 `__hmax_nan/__hmin_nan` 在 `__MUSA_ARCH__ >= 800` 才可见，MP31 为 310。进一步定向测试还发现 BF16 NaN intrinsic 虽可见，但 MP31 编译器无法满足其输出寄存器约束。

当前工作树做了局部、语义保持的兼容修补：

- `src/tl_templates/musa/fop.h`：MP31 对 FP16/BF16 采用 `__hisnan` 检查；任一输入为 NaN 时传播该输入，否则调用普通 `__hmax/__hmin`；架构值 ≥800 保留 SDK 原生 NaN intrinsic；
- `src/tl_templates/musa/reduce.h`：reduction functor 使用同样策略；
- `src/backend/musa/codegen/codegen_musa.cc`：scalar FP16/BF16 NaN min/max 统一调用上述语义 helper，避免 codegen 绕过 fallback。
- `testing/python/language/test_tilelang_language_reduce_maxmin_nan.py`：源码检查改为验证语义 helper，而不是强制要求 MP31 不可用的 SDK intrinsic。

失败日志被保留为 `logs/run/p0_tilelang_fp16_nan_fallback.log`、`logs/run/p0_tilelang_fp16_nan_fallback_after_codegen_fix.log` 和首次完整测试文件结果 `logs/run/p0_tilelang_nan_test_file.log`。修补后，FP16/BF16 的 max/min runtime NaN 行为定向测试结果为 `2 passed`，见 `logs/run/p0_tilelang_nan_fallback_final.log`；完整相关测试结果见 `logs/run/p0_tilelang_nan_test_file_final.log`；后端重编译日志见 `logs/build/p0_tilelang_rebuild_mp31_nan_fallback*.log`。

这些是基于固定 SDK/目标的本地未提交改动，P1 数据必须同时记录基线提交和工作树 diff，不能只记录上游 commit。当前完整 diff 与 hash 已保存为 `logs/env/tilelang_musa_local.patch` 和 `logs/env/tilelang_musa_local_patch_sha256.txt`。

## 6. 最小 TileLang-MUSA 验收结果

一键命令：

```bash
/root/s5000-calibration/scripts/p0_run_tilelang_smoke.sh
```

最终证据：

- 生成源码：`logs/build/p0_tilelang_kernel/tilelang_musa_kernel_0022.mu`
- 生成编译/汇编命令：`logs/build/p0_tilelang_kernel/tilelang_musa_kernel_0022.mu.cmd`
- 运行日志：`logs/run/p0_tilelang_add_global_repro.log`
- 数值结果：`pass!`

生成 kernel 是 `__launch_bounds__(256, 1)` 的 float32 vector add，编译目标明确为 `mp_31`。运行时使用 torch_musa 分配输入并用 `torch.testing.assert_close` 校验结果。

TileLang 静态检查同时报告一条可能 data race 的 warning；实际生成索引和本次数值检查均通过，但 P0 不将该 warning 擅自归类为误报。P1 harness 应保留 warning 并增加跨 shape、边界长度和重复运行的正确性检查。

## 7. DeepStack TileSight 参考边界

干净参考树 `DeepStack-ae-source` 固定在 `d8ffb49ae62584d3812ad6944ea6a737e8e64052`，`src/tilesight` 含 181 个跟踪文件，包括 `arch`、`single_op`、`fused_op_pipeline_wave`、`tile_cache`、`distributed` 与 `welder_info_extract`。

该源码可作为模型结构、参数接口和验证方法参考，不是 S5000 oracle：其中有专有 CPython 3.11 NoC profile 二进制、未实现的 TIR 接口和显式 `overlap_ratio` 近似。其 NVIDIA/AMD 数值不得直接用于 S5000。

## 8. P0 门槛与 P1 决定

| 门槛 | 结果 |
|---|---|
| 环境和版本可复现 | 通过 |
| 原生 MUSA 编译/runtime | 通过 |
| TileLang-MUSA 来源与 backend 可追踪 | 通过 |
| 生成 MUSA 源码及 mcc 命令可保存 | 通过 |
| 最小 kernel 数值正确 | 通过 |
| 已知编译问题先修复并留失败证据 | 通过 |
| 性能或微架构参数推断 | 未进行，符合 P0 边界 |

决定：进入 P1，但 P1 只能建设计时、同步、warmup、正确性门禁和 CSV/JSON schema；尚不能直接开始大规模性能扫描。
