# S5000 L1/Shared 多算子路径验证

## 结论

- ReLU、Add 的 TME 读取均为 0，输入经过普通 cached-load/L1 路径。
- 四个元素级用例的字节派生 L1 命中率均约为 0；MCU 的约 48–50% sector hit 不能作为时间复用命中率。
- 1M 预热工作集驻留 L2；16M ReLU/Add 的读取工作集超过 60 MiB L2，DDR 读取接近逻辑读取量。
- GEMM 4096³ 的 TMA 读取远大于普通 L1 下游读取，确认 TMA payload 基本旁路普通 L1。
- Shared Memory 是显式寻址存储，不输出命中率；模型输出搬运量、读写量和代价。

## 普通 cached-load 微基准校准

- 有效容量：32 KiB/MP，同一 MP 上的活跃 CTA 共享。
- Cache line：64 B 候选；sector：32 B 候选。
- MCU L1 Cache 物理峰值：7168 GB/s；当前验证有效吞吐约 5.2–5.6 TB/s。
- 冲突扫描从 13-way 开始出现额外流量，但 13–16-way 不单调，可能存在地址哈希；因此只记录条件冲突拐点，不宣称精确硬件关联度。

## 元素级算子：预测与 MCU

| 用例 | MCU sector hit | 字节派生 L1 hit | L1下游预测/实测 | APE | DDR读预测/实测 | DDR写预测/实测 | 写误差/逻辑写量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| relu_1m | 49.04% | 0.00% | 4194304/4198144 B | 0.091% | 10486/11264 B | 0/66464 B | 1.585% |
| relu_16m | 49.87% | 0.00% | 67108864/67112736 B | 0.006% | 67108864/67139200 B | 67108864/72036032 B | 7.342% |
| add_1m | 47.89% | 0.00% | 8388608/8402976 B | 0.171% | 20972/16256 B | 0/61184 B | 1.459% |
| add_16m | 47.98% | 0.00% | 134217728/134274208 B | 0.042% | 134217728/134259840 B | 67108864/67413728 B | 0.454% |

## 元素级算子延迟

| 用例 | MCU实测 | DeepStack原始 | 原始APE | 分层关键路径修正 | 修正APE |
|---|---:|---:|---:|---:|---:|
| relu_1m | 11.462 us | 3.570 us | 68.85% | 3.570 us | 68.85% |
| relu_16m | 115.304 us | 50.130 us | 56.52% | 95.760 us | 16.95% |
| add_1m | 14.985 us | 7.030 us | 53.09% | 7.030 us | 53.09% |
| add_16m | 148.274 us | 98.570 us | 33.52% | 143.640 us | 3.13% |

16M 流式用例接入分层 service cost 后明显改善；1M 短 kernel 仍由固定启动、prologue/tail-wave 等开销主导，不能仅靠内存带宽模型解决。

1M 用例预测 DDR 读为 0；MCU 的 11–16 KiB 为辅助/控制噪声，绝对量低于逻辑读取的 0.4%。
1M 用例的输出也在 kernel 采集窗口内驻留 L2，预测 DDR 写为 0；残余计数约占逻辑写量 1.5%。16M 用例 DDR 读误差低于 0.05%。当前最大偏差是 ReLU 16M 的 DDR 写约 7.3%，说明写回/事务粒度仍需继续校准，不能用 L1 读命中率修正。

## GEMM 4096³ TMA 路径

- MCU 普通 L1 下游读取：21120 B
- MCU TME/L2 读取计数：328223488 B
- 模型逻辑 TMA Tile 搬运请求：2147483648 B
- MCU DDR 读取：38452096 B

逻辑 Tile 请求量与 `l2pu__tme_read_l2_byte.avg` 的计数域不同，当前不能直接计算 APE。这里只用二者与普通 L1 的数量级差异验证路径分类。GEMM 的 L2/DDR 绝对流量模型仍需确认 TME counter 的聚合/归一化语义后再校准。

## 已接入的计算关系

```text
L1_eligible_read = ordinary_global_read
L1_bypass_read   = tma_global_to_shared
L2_read_input    = L1_bypass_read + L1_eligible_read × (1 - L1_hit)
DDR_read         = L2_read_input × (1 - L2_hit)
DDR_total        = DDR_read + writeback_reaching_DDR
Shared_hit_rate  = not_applicable
```

其中加号不可写成减号。冷启动与预热缓存通过 `--cache-condition cold|warm` 显式输入，不再暗中改变内存层级。
