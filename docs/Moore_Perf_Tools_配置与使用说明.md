# Moore Perf Tools 配置与使用说明

本文记录当前 S5000 容器中 Moore Perf Compute、Moore Perf System 和
MUSA Compute Sanitizer 的实际安装状态、环境配置与常用命令。

## 1. 当前环境

| 项目 | 当前值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，x86_64 |
| GPU | MTT S5000，MUSA 架构 `mp_31` |
| 容器可见 GPU | 1 张，80 GiB 显存 |
| Linux Driver | `3.3.5-server` |
| 项目默认 MUSA SDK | `4.3.5` |
| 隔离 MUSA SDK | `5.2.0`，路径 `/usr/local/musa-5.2.0/current` |
| Moore Perf Compute | `1.3.0-85bf41325` |
| Moore Perf System | `1.8.0-3260d06f3` |
| MUSA Compute Sanitizer | `1.1.1` |

三个工具均已在当前 S5000 和驱动环境中完成实际运行验证。需要注意：
Moore Perf Compute 1.3.0、Moore Perf System 1.8.0 和 MUSA Compute
Sanitizer 1.1.1 的官方匹配环境是 MUSA SDK/Driver 5.2.0。当前驱动版本
不在官方版本矩阵内，本文记录的是本机实测兼容结果，不代表官方支持承诺。

## 2. SDK 隔离配置

项目已有脚本大量使用 MUSA SDK 4.3.5，因此没有修改全局链接：

```text
/usr/local/musa -> /usr/local/musa-4.3.5
```

为避免 4.3.5 与 5.2.0 的编译器、头文件和运行库混用，已经配置全局命令
`musa52-run`。该命令只为紧随其后的单条命令设置 SDK 5.2 环境，不会改变
当前 shell 或项目默认环境。

检查配置：

```bash
musa52-run mcc --version
```

预期包含：

```text
mcc version 5.2.0
```

编译和运行 MUSA 5.2 程序时都应使用同一个入口：

```bash
musa52-run mcc -O2 --cuda-gpu-arch=mp_31 program.mu -lmusart -o program
musa52-run ./program
```

不要只替换 `PATH` 而继续使用 4.3.5 的 `LD_LIBRARY_PATH`，否则可能同时加载
两套 SDK 的组件。

## 3. Moore Perf Compute

Moore Perf Compute 用于分析单个 GPU kernel 的计算吞吐、内存吞吐、缓存命中率、
启动配置和 Roofline 数据。

检查安装：

```bash
mcu --version
mcu --list-sections
mcu --list-metrics
```

采集全部 kernel：

```bash
musa52-run mcu -o report.mcu-rep ./program [程序参数]
```

只采集名称中包含 `kernel_name` 的 kernel：

```bash
musa52-run mcu -k kernel_name -o kernel_name.mcu-rep ./program [程序参数]
```

多 GPU 示例：

```bash
musa52-run mcu --devices 0,2 -o multi_gpu.mcu-rep ./program
```

MCU 会多次重放应用以收集不同硬件计数器。被测程序应尽量满足：

- 每次重放执行相同的 kernel 序列；
- 输入和控制流具有确定性；
- 避免一次采集包含大量重复 kernel，否则采集时间和报告体积会明显增加；
- 优先使用 `-k` 缩小采集范围。

同一张 GPU 上采集 MCU 数据前，应停止 MTDCGM 性能字段 2002、2003、
2004、2005 的监控任务，采集完成后再恢复。

## 4. Moore Perf System

Moore Perf System 用于采集 CPU/GPU 时间线、MUSA API、kernel、显存操作和
系统运行时信息，适合定位调度、同步与 CPU-GPU 交互瓶颈。

检查安装：

```bash
msys --version
msys --help
```

采集 MUSA 时间线：

```bash
musa52-run msys profile \
  -t musa \
  -o report.msys-rep \
  ./program [程序参数]
```

限制采集时间：

```bash
musa52-run msys profile \
  -t musa \
  --duration 10 \
  -o report_10s.msys-rep \
  ./program [程序参数]
```

查看 kernel 汇总：

```bash
msys stats --report musa_gpu_kern_sum report.msys-rep
```

常用报告类型还包括：

```text
musa_api_sum
musa_api_trace
musa_gpu_kern_sum
musa_gpu_kern_gb_sum
musa_gpu_mem_size_sum
musa_gpu_mem_time_sum
musa_gpu_sum
musa_kern_exec_sum
musa_api_gpu_sum
```

查询当前版本支持的完整参数：

```bash
msys --help profile
msys --help stats
msys --help export
msys --help analyze
```

## 5. MUSA Compute Sanitizer

MUSA Compute Sanitizer 用于检测 GPU 越界访问、显存泄漏和 MUSA API 错误。
建议在性能分析前先完成正确性检查。

检查安装：

```bash
mt-compute-sanitizer --version
mt-compute-sanitizer --help
```

执行默认内存检查：

```bash
musa52-run mt-compute-sanitizer \
  --log-file program.san.log \
  ./program [程序参数]
```

检查显存泄漏：

```bash
musa52-run mt-compute-sanitizer \
  --leak-check=full \
  --log-file program-leak.san.log \
  ./program [程序参数]
```

限制详细错误打印数量：

```bash
musa52-run mt-compute-sanitizer \
  --print-limit 20 \
  --log-file program.san.log \
  ./program
```

检查结果：

```bash
rg 'Out of bounds|Leaked|Program hit musaError|ERROR SUMMARY' program.san.log
```

无错误时应看到：

```text
========= ERROR SUMMARY: 0 error
```

## 6. 推荐使用顺序

建议按以下顺序分析同一个程序：

1. 普通运行，确认输出和退出码正确；
2. 使用 MUSA Compute Sanitizer 检查内存和 API 错误；
3. 使用 Moore Perf System 查看时间线、API、同步和 kernel 分布；
4. 使用 Moore Perf Compute 针对热点 kernel 采集硬件计数器。

示例：

```bash
musa52-run ./program

musa52-run mt-compute-sanitizer \
  --log-file program.san.log \
  ./program

musa52-run msys profile \
  -t musa \
  -o program.msys-rep \
  ./program

musa52-run mcu \
  -k hot_kernel \
  -o hot_kernel.mcu-rep \
  ./program
```

## 7. GUI

Moore Perf Compute 和 Moore Perf System 的 GUI 命令分别为：

```bash
mcu-ui
msys-ui
```

当前运行环境是无桌面的 Docker/SSH 环境，因此主要使用 CLI。可以把
`.mcu-rep` 或 `.msys-rep` 报告复制到具有图形桌面的兼容环境中打开。

## 8. 安装与健康检查

检查 Debian 包状态：

```bash
dpkg-query -W -f='${Package}\t${Version}\t${Status}\n' \
  mcu msys mt-compute-sanitizer
dpkg --audit
```

检查 GPU：

```bash
mthreads-gmi
ls -l /dev/mtgpu*
```

检查三个工具和 SDK 5.2：

```bash
mcu --version
msys --version
mt-compute-sanitizer --version
musa52-run mcc --version
```

当前安装后的预期版本为：

```text
Moore Perf Compute Version: 1.3.0-85bf41325
Moore Perf System Version: 1.8.0-3260d06f3
mt-compute-sanitizer version: 1.1.1-2026.07.16-9c5dc5d
mcc version 5.2.0
```

## 9. 常见问题

### 找不到命令

先确认命令入口：

```bash
command -v mcu
command -v msys
command -v mt-compute-sanitizer
command -v musa52-run
```

当前预期路径分别为 `/usr/bin/mcu`、`/usr/bin/msys`、
`/usr/local/musa-4.3.5/bin/mt-compute-sanitizer` 和
`/usr/local/bin/musa52-run`。

### 程序找不到 `libmusart.so.5`

说明程序是用 SDK 5.2 编译的，但运行时没有启用对应库路径。使用：

```bash
musa52-run ./program
```

分析时同样把分析工具放在 `musa52-run` 后面：

```bash
musa52-run mcu ./program
musa52-run msys profile -t musa ./program
musa52-run mt-compute-sanitizer ./program
```

### MCU 采集时间很长

MCU 需要进行应用重放。使用 `-k` 只选择目标 kernel，并减少被测程序内重复
启动次数。性能分析期间的 kernel 时间包含工具采集开销，不能当作普通运行耗时。

### System 报告为空

确认程序确实执行了 MUSA API 或 GPU kernel，并显式启用 MUSA trace：

```bash
musa52-run msys profile -t musa -o report.msys-rep ./program
```

### 驱动版本警告

当前三套工具已经在本机 `3.3.5-server` 驱动上实测成功，但官方版本矩阵要求
5.2.0。若后续出现无法解释的采集错误、指标异常或升级工具版本，应优先在独立
维护窗口升级宿主机驱动，并在升级后重新完成本文第 8 节的健康检查。

## 10. 本机验证记录

2026-08-11 完成以下验证：

- Moore Perf Compute 能读取 S5000 硬件指标模块；
- Moore Perf System 成功采集 64 次 `pointer_chase` kernel，生成并解析
  `.msys-rep` 报告；
- MUSA Compute Sanitizer 完整运行同一 GPU kernel 程序，返回
  `ERROR SUMMARY: 0 error`；
- 安装后 `dpkg --audit` 无异常；
- `/usr/local/musa` 仍指向 4.3.5，原项目默认环境保持不变。
