# 在 S4000 上使用 TileLang-MUSA 0.1.12+musa.5

本补丁对应上游 `tile-ai/tilelang-musa` 提交 `a5fc8648825de0ec7015bccb4aa4ce618ba48f54`。仓库顶层的 `tilelang-musa/` 是带有 S5000 和 CompilerInfo 修改的另一份快照。两者在转换模板和归约实现上有差异，请在指定的上游工作树应用补丁。

## 主要修改

- 用目标架构的 mcc 编译最小用例，生成供模板使用的 SDK 能力配置。SDK 缺少 E8M0 类型时，普通算子可以编译，显式依赖该类型的算子仍需相应 SDK 支持。
- 在 SDK 缺少 FP16 带 NaN 最值函数时，标量和归约共用后备实现，覆盖 NaN、正负零和无穷值。
- PyTorch 将 CUDA 命名空间映射到 MUSA 时，JIT 适配器避免调用 CUDA 私有接口。Cython 适配器按 MUSA 目标读取调用方当前流，并保留显式流参数。

## 应用和复测

从本目录开始，在装有 MUSA 工具链的 Linux 主机上执行。示例使用 S4000 的目标架构 mp_22。切换 SDK 或目标架构后应重新生成配置，并在打包或运行 JIT 时使用含该配置的模板目录。

```bash
patch_dir="$PWD"
git clone https://github.com/tile-ai/tilelang-musa.git tilelang-sdk-check
cd tilelang-sdk-check
git checkout --detach a5fc8648825de0ec7015bccb4aa4ce618ba48f54
git submodule update --init 3rdparty/mutlass
git apply --check "$patch_dir/s4000-sdk-compat.patch"
git apply "$patch_dir/s4000-sdk-compat.patch"

export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:${LD_LIBRARY_PATH:-}"
python3 maint/scripts/probe_musa_headers.py --arch mp_22 \
    --output src/tl_templates/musa/sdk_compat_config.h
mcc -std=c++17 -mtgpu --cuda-gpu-arch=mp_22 -Isrc -I3rdparty/mutlass/include \
    "$patch_dir/half_edge.mu" -L"$MUSA_HOME/lib" -lmusart -o half-edge
./half-edge
```

## 验证范围

S4000、SDK 4.3.6、驱动 3.3.5-server。提交前在干净基线上重新应用补丁，源码树与实测修复提交 `f47c254` 一致。重新生成了 SDK 配置并在真机运行 half-edge，28 项位级检查全部通过。本次编译使用已安装 TileLang 包中的 MUTLASS 头文件。

同一修复此前通过 256 元素 FP32 加法、FP16/BF16 矩阵乘，以及非默认流上的 debugger 采集。SDK 5.1.0 和 S5000 未在本次验证范围内。
