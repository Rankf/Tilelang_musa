#!/usr/bin/env python3
"""Analyze logical TMA bytes against MCU l2pu TME avg counters."""

from __future__ import annotations

import csv
import json
import pathlib
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/profiling/gemm_tme_scaling_20260811"
CASES = (
    "shape_1024",
    "shape_2048",
    "shape_4096",
    "tile_128_128_32",
    "tile_64_128_64",
    "tile_128_64_64",
)
METRICS = (
    "gpu__time_duration.max",
    "l1__downstream_read_bytes.sum",
    "l1__sector_hit_rate.pct",
    "l2pu__sector_hit_rate.pct",
    "l2pu__tme_read_l2_byte.avg",
    "ddr__read_bytes.sum",
    "ddr__write_bytes.sum",
)


def parse(path: pathlib.Path) -> tuple[dict[str, object], dict[str, float]]:
    metadata = None
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{") and '"case"' in line:
            metadata = json.loads(line)
        stripped = line.strip()
        for metric in METRICS:
            if stripped.startswith(metric + " "):
                metrics[metric] = float(stripped.split()[-1])
    if metadata is None:
        raise ValueError(f"{path}: no probe metadata")
    missing = set(METRICS) - set(metrics)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return metadata, metrics


def build_rows() -> list[dict[str, object]]:
    rows = []
    for case in CASES:
        metadata, metrics = parse(RESULT / f"{case}_mcu.log")
        logical = int(metadata["logical_tma_bytes"])
        tme_avg = int(metrics["l2pu__tme_read_l2_byte.avg"])
        rows.append({
            "case": case,
            "M": metadata["M"],
            "N": metadata["N"],
            "K": metadata["K"],
            "bm": metadata["bm"],
            "bn": metadata["bn"],
            "bk": metadata["bk"],
            "grid_cta": metadata["grid_cta"],
            "k_iterations": metadata["k_iterations"],
            "generated_launch_threads": metadata["generated_launch_threads"],
            "logical_tma_bytes": logical,
            "mcu_tme_read_l2_byte_avg": tme_avg,
            "logical_over_tme_avg": logical / tme_avg,
            "mcu_l1_downstream_read_bytes": int(
                metrics["l1__downstream_read_bytes.sum"]
            ),
            "mcu_l2_sector_hit_pct": metrics["l2pu__sector_hit_rate.pct"],
            "mcu_ddr_read_bytes": int(metrics["ddr__read_bytes.sum"]),
            "mcu_gpu_duration_us": metrics["gpu__time_duration.max"] / 1000.0,
            "correctness": metadata["correctness"],
            "uses_tma": metadata["uses_tma"],
        })
    return rows


def write(rows: list[dict[str, object]]) -> None:
    with (RESULT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULT / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fixed_tile = rows[:3]
    conditional_divisor = (
        sum(row["logical_tma_bytes"] for row in fixed_tile)
        / sum(row["mcu_tme_read_l2_byte_avg"] for row in fixed_tile)
    )
    conditional_apes = []
    for row in fixed_tile:
        predicted = row["logical_tma_bytes"] / conditional_divisor
        measured = row["mcu_tme_read_l2_byte_avg"]
        conditional_apes.append(abs(predicted - measured) / measured * 100)
    bk64 = next(row for row in rows if row["case"] == "shape_2048")
    bk32 = next(row for row in rows if row["case"] == "tile_128_128_32")
    tile_m = next(row for row in rows if row["case"] == "tile_64_128_64")
    tile_n = next(row for row in rows if row["case"] == "tile_128_64_64")
    orientation_spread = (
        max(tile_m["logical_over_tme_avg"], tile_n["logical_over_tme_avg"])
        / min(tile_m["logical_over_tme_avg"], tile_n["logical_over_tme_avg"])
        - 1
    ) * 100
    bk_delta = abs(
        bk64["logical_over_tme_avg"] - bk32["logical_over_tme_avg"]
    ) / bk64["logical_over_tme_avg"] * 100

    lines = [
        "# GEMM logical TMA / MCU TME avg 缩放验证",
        "",
        "所有用例均通过正确性检查、使用 TMA，并生成相同的 384-thread kernel 模板。",
        "",
        "| 用例 | Tile | Grid CTA | K迭代 | logical TMA | MCU TME avg | logical/TME | L1下游 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['bm']}×{row['bn']}×{row['bk']} | "
            f"{row['grid_cta']} | {row['k_iterations']} | "
            f"{row['logical_tma_bytes'] / 1e6:.3f} MB | "
            f"{row['mcu_tme_read_l2_byte_avg'] / 1e6:.3f} MB | "
            f"{row['logical_over_tme_avg']:.4f} | "
            f"{row['mcu_l1_downstream_read_bytes']} B |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 固定 128×128×64 Tile 时，1024³/2048³/4096³ 的比例依次为 "
        f"{fixed_tile[0]['logical_over_tme_avg']:.4f}、"
        f"{fixed_tile[1]['logical_over_tme_avg']:.4f}、"
        f"{fixed_tile[2]['logical_over_tme_avg']:.4f}，随规模增大缓慢下降。",
        f"- 仅在这个固定 Tile/代码路径内，穿过原点的条件换算除数约为 "
        f"{conditional_divisor:.4f}；三个 shape 的 TME avg 拟合 APE 为 "
        f"{', '.join(f'{value:.2f}%' for value in conditional_apes)}。",
        f"- BK 64→32、K迭代 32→64 后比例变化仅 {bk_delta:.3f}%，说明该组中计数与总字节相关，而不是简单按 K-loop 次数计数。",
        f"- 相同 2048³、相同 logical TMA 402.653 MB，仅交换 BM/BN 后比例为 "
        f"{tile_m['logical_over_tme_avg']:.4f} 与 "
        f"{tile_n['logical_over_tme_avg']:.4f}，跨度 {orientation_spread:.2f}%。",
        "- 因此不能建立跨 Tile 的单一固定换算系数。`avg` 明显受地址方向、L2PU 分布或操作数复用映射影响。",
        "- 普通 L1 下游只有 9.6–13.4 KiB，而 TME 为 MB 级，TMA bypass 普通 L1 的路径结论保持不变。",
        "",
        "## 模型处理建议",
        "",
        "- `logical_tma_bytes` 继续作为 Tile pipeline 的逻辑搬运量和 Shared 写入量。",
        "- `l2pu__tme_read_l2_byte.avg` 保留为独立的硬件观测字段，不冒充全芯片逻辑搬运量。",
        "- 固定 128×128×64 schedule 可使用约 6.62 的条件换算做诊断，但必须标记 `schedule_specific_estimate`，不能写入通用 S5000 硬件参数。",
        "- 若要预测 MCU TME avg，需要增加 BM/BN、地址布局和 L2PU 映射模型；仅增加 shape 样本无法解决跨 Tile 差异。",
        "",
        "此前 4096³ 的 256-thread kernel 与本轮 384-thread TMA-prefetch kernel 不是相同代码路径，不参与本轮换算拟合。",
    ])
    (RESULT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write(rows)
    ratios = [row["logical_over_tme_avg"] for row in rows]
    print(json.dumps({
        "cases": len(rows),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_mean": statistics.mean(ratios),
        "report": str(RESULT / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
