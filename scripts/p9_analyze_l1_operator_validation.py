#!/usr/bin/env python3
"""Summarize P9 ordinary-L1 operator validation and model comparison."""

from __future__ import annotations

import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/profiling/l1_operator_validation_20260811"

METRICS = (
    "gpu__time_duration.max",
    "l1__load_requests.sum",
    "l1__store_requests.sum",
    "l1__downstream_read_bytes.sum",
    "l1__downstream_write_bytes.sum",
    "l1__sector_hit_rate.pct",
    "l1__throughput.avg.pct_of_peak_sustained_active",
    "l2pu__sector_hit_rate.pct",
    "l2pu__tme_read_l2_byte.avg",
    "ddr__read_bytes.sum",
    "ddr__write_bytes.sum",
)


def parse_log(path: pathlib.Path) -> tuple[dict[str, str], dict[str, float]]:
    metadata: dict[str, str] = {}
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("kernel="):
            metadata = dict(item.split("=", 1) for item in line.split(","))
        stripped = line.strip()
        for name in METRICS:
            if stripped.startswith(name + " "):
                metrics[name] = float(stripped.split()[-1])
    return metadata, metrics


def pct_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100 if measured else 0.0


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in ("relu_1m", "relu_16m", "add_1m", "add_16m"):
        metadata, metrics = parse_log(RESULT / f"{case}_mcu.log")
        missing = set(METRICS) - set(metrics)
        if missing:
            raise ValueError(f"{case}: missing MCU metrics {sorted(missing)}")
        prediction_report = json.loads(
            (RESULT / f"{case}_prediction.json").read_text(encoding="utf-8")
        )
        prediction = prediction_report["memory_hierarchy_prediction"]
        logical_read = float(metadata["ordinary_read_bytes"])
        logical_write = float(metadata["ordinary_write_bytes"])
        l1_downstream = metrics["l1__downstream_read_bytes.sum"]
        ddr_read = metrics["ddr__read_bytes.sum"]
        ddr_write = metrics["ddr__write_bytes.sum"]
        derived_l1_hit = max(0.0, 1.0 - l1_downstream / logical_read)
        derived_l2_hit = max(0.0, 1.0 - ddr_read / l1_downstream)
        predicted_l1_downstream = prediction["l1_cache"]["downstream_read_bytes"]
        predicted_ddr_read = prediction["ddr"]["read_bytes"]
        predicted_ddr_write = prediction["ddr"]["write_bytes"]
        measured_latency_us = metrics["gpu__time_duration.max"] / 1000.0
        base_latency_us = prediction_report["latency_us"]
        hierarchy_latency_us = prediction_report["latency_with_memory_hierarchy_us"]
        rows.append({
            "case": case,
            "op": metadata["op"],
            "elements": int(metadata["elements"]),
            "correctness": metadata["correctness"],
            "measured_latency_us": measured_latency_us,
            "deepstack_base_latency_us": base_latency_us,
            "hierarchy_latency_us": hierarchy_latency_us,
            "base_latency_ape_pct": pct_error(base_latency_us, measured_latency_us),
            "hierarchy_latency_ape_pct": pct_error(
                hierarchy_latency_us, measured_latency_us
            ),
            "logical_read_bytes": int(logical_read),
            "logical_write_bytes": int(logical_write),
            "mcu_l1_sector_hit_pct": metrics["l1__sector_hit_rate.pct"],
            "derived_l1_byte_hit_pct": derived_l1_hit * 100,
            "measured_l1_downstream_read_bytes": int(l1_downstream),
            "predicted_l1_downstream_read_bytes": int(predicted_l1_downstream),
            "l1_downstream_ape_pct": pct_error(predicted_l1_downstream, l1_downstream),
            "mcu_tme_read_bytes": int(metrics["l2pu__tme_read_l2_byte.avg"]),
            "derived_l2_byte_hit_pct": derived_l2_hit * 100,
            "measured_ddr_read_bytes": int(ddr_read),
            "predicted_ddr_read_bytes": int(predicted_ddr_read),
            "ddr_read_absolute_error_bytes": int(abs(predicted_ddr_read - ddr_read)),
            "measured_ddr_write_bytes": int(ddr_write),
            "predicted_ddr_write_bytes": int(predicted_ddr_write),
            "ddr_write_error_pct_of_logical": (
                abs(predicted_ddr_write - ddr_write) / logical_write * 100
            ),
            "cache_condition": prediction["cache_condition"],
        })
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with (RESULT / "operator_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT / "operator_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    gemm_log = ROOT / (
        "results/profiling/l1_characterization_20260811/"
        "gemm4096_tma_path_mcu.log"
    )
    _, gemm_metrics = parse_log(gemm_log)
    gemm_prediction = json.loads(
        (RESULT / "gemm_4096_prediction.json").read_text(encoding="utf-8")
    )["memory_hierarchy_prediction"]
    lines = [
        "# S5000 L1/Shared 多算子路径验证",
        "",
        "## 结论",
        "",
        "- ReLU、Add 的 TME 读取均为 0，输入经过普通 cached-load/L1 路径。",
        "- 四个元素级用例的字节派生 L1 命中率均约为 0；MCU 的约 48–50% sector hit 不能作为时间复用命中率。",
        "- 1M 预热工作集驻留 L2；16M ReLU/Add 的读取工作集超过 60 MiB L2，DDR 读取接近逻辑读取量。",
        "- GEMM 4096³ 的 TMA 读取远大于普通 L1 下游读取，确认 TMA payload 基本旁路普通 L1。",
        "- Shared Memory 是显式寻址存储，不输出命中率；模型输出搬运量、读写量和代价。",
        "",
        "## 普通 cached-load 微基准校准",
        "",
        "- 有效容量：32 KiB/MP，同一 MP 上的活跃 CTA 共享。",
        "- Cache line：64 B 候选；sector：32 B 候选。",
        "- MCU L1 Cache 物理峰值：7168 GB/s；当前验证有效吞吐约 5.2–5.6 TB/s。",
        "- 冲突扫描从 13-way 开始出现额外流量，但 13–16-way 不单调，可能存在地址哈希；因此只记录条件冲突拐点，不宣称精确硬件关联度。",
        "",
        "## 元素级算子：预测与 MCU",
        "",
        "| 用例 | MCU sector hit | 字节派生 L1 hit | L1下游预测/实测 | APE | DDR读预测/实测 | DDR写预测/实测 | 写误差/逻辑写量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['mcu_l1_sector_hit_pct']:.2f}% | "
            f"{row['derived_l1_byte_hit_pct']:.2f}% | "
            f"{row['predicted_l1_downstream_read_bytes']}/{row['measured_l1_downstream_read_bytes']} B | "
            f"{row['l1_downstream_ape_pct']:.3f}% | "
            f"{row['predicted_ddr_read_bytes']}/{row['measured_ddr_read_bytes']} B | "
            f"{row['predicted_ddr_write_bytes']}/{row['measured_ddr_write_bytes']} B | "
            f"{row['ddr_write_error_pct_of_logical']:.3f}% |"
        )
    lines.extend([
        "",
        "## 元素级算子延迟",
        "",
        "| 用例 | MCU实测 | DeepStack原始 | 原始APE | 分层关键路径修正 | 修正APE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['measured_latency_us']:.3f} us | "
            f"{row['deepstack_base_latency_us']:.3f} us | "
            f"{row['base_latency_ape_pct']:.2f}% | "
            f"{row['hierarchy_latency_us']:.3f} us | "
            f"{row['hierarchy_latency_ape_pct']:.2f}% |"
        )
    lines.extend([
        "",
        "16M 流式用例接入分层 service cost 后明显改善；1M 短 kernel 仍由固定启动、prologue/tail-wave 等开销主导，不能仅靠内存带宽模型解决。",
        "",
        "1M 用例预测 DDR 读为 0；MCU 的 11–16 KiB 为辅助/控制噪声，绝对量低于逻辑读取的 0.4%。",
        "1M 用例的输出也在 kernel 采集窗口内驻留 L2，预测 DDR 写为 0；残余计数约占逻辑写量 1.5%。16M 用例 DDR 读误差低于 0.05%。当前最大偏差是 ReLU 16M 的 DDR 写约 7.3%，说明写回/事务粒度仍需继续校准，不能用 L1 读命中率修正。",
        "",
        "## GEMM 4096³ TMA 路径",
        "",
        f"- MCU 普通 L1 下游读取：{gemm_metrics['l1__downstream_read_bytes.sum']:.0f} B",
        f"- MCU TME/L2 读取计数：{gemm_metrics['l2pu__tme_read_l2_byte.avg']:.0f} B",
        f"- 模型逻辑 TMA Tile 搬运请求：{gemm_prediction['l1_cache']['bypass_read_bytes']} B",
        f"- MCU DDR 读取：{gemm_metrics['ddr__read_bytes.sum']:.0f} B",
        "",
        "逻辑 Tile 请求量与 `l2pu__tme_read_l2_byte.avg` 的计数域不同，当前不能直接计算 APE。这里只用二者与普通 L1 的数量级差异验证路径分类。GEMM 的 L2/DDR 绝对流量模型仍需确认 TME counter 的聚合/归一化语义后再校准。",
        "",
        "## 已接入的计算关系",
        "",
        "```text",
        "L1_eligible_read = ordinary_global_read",
        "L1_bypass_read   = tma_global_to_shared",
        "L2_read_input    = L1_bypass_read + L1_eligible_read × (1 - L1_hit)",
        "DDR_read         = L2_read_input × (1 - L2_hit)",
        "DDR_total        = DDR_read + writeback_reaching_DDR",
        "Shared_hit_rate  = not_applicable",
        "```",
        "",
        "其中加号不可写成减号。冷启动与预热缓存通过 `--cache-condition cold|warm` 显式输入，不再暗中改变内存层级。",
    ])
    (RESULT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_report(rows)
    print(f"wrote {RESULT / 'operator_comparison.csv'}")
    print(f"wrote {RESULT / 'operator_comparison.json'}")
    print(f"wrote {RESULT / 'report.md'}")


if __name__ == "__main__":
    main()
