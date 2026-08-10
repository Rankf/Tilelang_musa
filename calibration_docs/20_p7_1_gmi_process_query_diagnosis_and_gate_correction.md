# P7.1：`mthreads-gmi` 进程查询故障的定位与门禁修正

状态：已定位为 **S5000 当前驱动/管理工具组合不支持的进程监控能力**；原恢复门禁已修正，尚未据此启动任何 GPU kernel。
日期：2026-07-29（UTC+8）

## 结论

原 gate 把默认 `mthreads-gmi` 输出中的 `Failed to query running processes` 当作 runtime 未恢复的证据，设计不正确。只读复核显示：

- `mthreads-gmi -cf` 连续成功，报告 MTT S5000、80 GiB、P12、27 C、0% / 0 MiB；
- `mthreads-gmi -q --json -i 0` 成功，识别 GPU 0、PCI `00000000:03:00.0`、Driver `3.3.5-server`；
- `mthreads-gmi -pm` 明确返回 `the requested operation is not available on target device`；
- 工具帮助将 `-cf` 定义为“隐藏 MPC instance 和 process information”。

因此这是 process-monitor/query 能力不可用，而不是已经证明的 GPU runtime 故障。该结论也不反向证明此前 P7 的 `err 999` 已消失：两件事必须分开记录。

## 只读证据

诊断原始输出：`logs/p7_1_diagnosis/20260729_readonly/`。

| 检查 | 结果 | 可得出的结论 |
|---|---|---|
| concise / JSON 设备查询 | 成功 | 驱动管理路径可枚举设备并读取基本遥测 |
| PCIe | Gen5 x16，`mtgpu` 绑定 | PCIe/内核驱动绑定存在 |
| `/proc/*/fd` | 没有进程打开 `/dev/mtgpu.0` | 未见设备节点遗留持有者 |
| `/proc/*/maps` | 未见 `libmusa`、`libmusart` 或 `libmugraph` 映射 | 未见存活 MUSA runtime 进程 |
| mtlink error counters | replay/recovery 均为 0 | 未见该计数器上的链路恢复事件；FEC correction 为累计计数，未作健康/故障判定 |
| kernel log | 没有匹配行 | 不能据此证明系统日志正常，因为本次查询没有输出匹配记录 |

容器中没有 `systemctl`，故未对 systemd unit 状态作结论；`lspci` 有 libkmod-resource warning，但仍成功报告设备和 `Kernel driver in use: mtgpu`。

## 修正后的恢复门禁

`scripts/p7_1_recovery_gate.sh` 现在连续三次执行并验证：

1. `mthreads-gmi -cf`；
2. `mthreads-gmi -q --json -i 0` 的固定身份字段（GPU 数、型号、PCI Bus ID、显存）；
3. OS 进程快照没有残留 P2/P7 controller；
4. 没有进程打开 `/dev/mtgpu.0`，也没有存活进程映射 MUSA runtime 库。

它不再把不支持的 `mthreads-gmi` process monitor 当作失败条件。原 gate raw run 保留不改：它仍准确反映“旧规则被不支持能力阻止”。

该门禁只说明可以开始受控的 P7.1 runtime A/B 首个最小试验；它不是硬件认证，也不构成对 P7 间歇 `err 999` 的修复声明。若首个最小试验再次出现 runtime error，立即停止并保留日志，不能继续性能采样。
