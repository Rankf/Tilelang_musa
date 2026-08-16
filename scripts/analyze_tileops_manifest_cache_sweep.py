#!/usr/bin/env python3
"""Analyze warm elementwise MCU cache sweeps anchored in TileOPs workloads."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/profiling/tileops_manifest_cache_sweep_20260814"
TILEOPS = ROOT / "TileOPs" / "src"
L2_CAPACITY_BYTES = 60 * 1024 * 1024
METRICS = (
    "gpu__time_duration.max",
    "l1__downstream_read_bytes.sum",
    "l1__downstream_write_bytes.sum",
    "l1__sector_hit_rate.pct",
    "l2pu__sector_hit_rate.pct",
    "ddr__read_bytes.sum",
    "ddr__write_bytes.sum",
)


def _manifest_anchors() -> dict[str, dict[str, object]]:
    sys.path.insert(0, str(TILEOPS))
    try:
        from tileops.manifest import load_workloads

        relu = load_workloads("ReluFwdOp")[0]
        add = load_workloads("AddFwdOp")[0]
    finally:
        sys.path.pop(0)
    return {
        "relu": {
            "op_name": "ReluFwdOp",
            "label": relu["label"],
            "shape": relu["input_shape"],
            "elements": math.prod(relu["input_shape"]),
            "manifest_dtypes": relu["dtypes"],
            "measurement_dtype": "float32",
            "provenance": "manifest_shape_dtype_normalized",
        },
        "add": {
            "op_name": "AddFwdOp",
            "label": add["label"],
            "shape": add["input_shape"],
            "elements": math.prod(add["input_shape"]),
            "manifest_dtypes": add["dtypes"],
            "measurement_dtype": "float32",
            "provenance": "manifest_exact_shape_and_dtype",
        },
    }


def _parse_summary_line(text: str, path: Path) -> dict[str, object]:
    summaries = []
    for line in text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            summaries.append(json.loads(line[start:]))
        except json.JSONDecodeError:
            continue
    if not summaries:
        raise ValueError(f"{path.name}: missing runtime JSON summary")
    if any(item != summaries[0] for item in summaries[1:]):
        raise ValueError(f"{path.name}: application replay summaries differ")
    return summaries[0]


def _parse_metrics(text: str, path: Path) -> dict[str, float]:
    metrics = {}
    for line in text.splitlines():
        stripped = line.strip()
        for name in METRICS:
            if stripped.startswith(name + " "):
                metrics[name] = float(stripped.split()[-1])
    required = {
        "gpu__time_duration.max",
        "l1__downstream_read_bytes.sum",
        "ddr__read_bytes.sum",
    }
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"{path.name}: missing MCU metrics {sorted(missing)}")
    return metrics


def _case_kind(filename: str, op: str, n: int, anchor_n: int) -> str:
    if n == anchor_n:
        return "tileops_manifest_anchor"
    if filename.endswith("_r2.console.txt"):
        return "capacity_control_repeat"
    return "capacity_control"


def build_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    anchors = _manifest_anchors()
    rows = []
    for path in sorted(RESULT.glob("*_warm*.console.txt")):
        text = path.read_text(encoding="utf-8")
        if "==ERROR==" in text or "No kernels were profiled" in text:
            raise ValueError(f"{path.name}: invalid MCU capture")
        runtime = _parse_summary_line(text, path)
        metrics = _parse_metrics(text, path)
        op = str(runtime["op"])
        n = int(runtime["shape"][0])
        logical_read = int(runtime["logical_read_bytes"])
        logical_write = int(runtime["logical_write_bytes"])
        l1_downstream = metrics["l1__downstream_read_bytes.sum"]
        ddr_read = metrics["ddr__read_bytes.sum"]
        raw_hit = (1.0 - ddr_read / l1_downstream) * 100.0
        current_model = 99.75 if logical_read <= L2_CAPACITY_BYTES else 0.0
        rows.append(
            {
                "case": path.name.removesuffix(".console.txt"),
                "case_kind": _case_kind(
                    path.name, op, n, int(anchors[op]["elements"])
                ),
                "op": op,
                "elements": n,
                "dtype": runtime["dtype"],
                "logical_read_bytes": logical_read,
                "logical_write_bytes": logical_write,
                "resident_working_set_bytes": logical_read + logical_write,
                "l1_downstream_read_bytes": int(l1_downstream),
                "ddr_read_bytes": int(ddr_read),
                "derived_l2_byte_hit_pct_raw": raw_hit,
                "derived_l2_byte_hit_pct": max(0.0, raw_hit),
                "current_tilesight_prediction_pct": current_model,
                "current_tilesight_absolute_error_points": abs(
                    current_model - max(0.0, raw_hit)
                ),
                "mcu_l2_sector_hit_pct": metrics.get(
                    "l2pu__sector_hit_rate.pct"
                ),
                "latency_us": metrics["gpu__time_duration.max"] / 1000.0,
                "correctness": runtime["correctness"],
                "kernel_source_sha256": runtime["kernel_source_sha256"],
                "cache_condition": runtime["cache_condition"],
                "mcu_launch_skip": runtime["mcu_expected_skip"],
                "mcu_launch_count": 1,
                "source_console": str(path),
                "source_report": str(
                    path.with_name(
                        path.name.removesuffix(".console.txt") + ".mcu-rep"
                    )
                ),
            }
        )
    return rows, anchors


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["op"]), int(row["elements"]))].append(row)
    aggregate = []
    for (op, elements), items in sorted(grouped.items()):
        hits = [float(item["derived_l2_byte_hit_pct"]) for item in items]
        first = items[0]
        aggregate.append(
            {
                "op": op,
                "elements": elements,
                "logical_read_mib": int(first["logical_read_bytes"]) / 2**20,
                "logical_write_mib": int(first["logical_write_bytes"]) / 2**20,
                "resident_working_set_mib": int(
                    first["resident_working_set_bytes"]
                )
                / 2**20,
                "run_count": len(items),
                "derived_l2_byte_hit_pct_mean": sum(hits) / len(hits),
                "derived_l2_byte_hit_pct_min": min(hits),
                "derived_l2_byte_hit_pct_max": max(hits),
                "current_tilesight_prediction_pct": first[
                    "current_tilesight_prediction_pct"
                ],
                "current_tilesight_absolute_error_points": abs(
                    float(first["current_tilesight_prediction_pct"])
                    - sum(hits) / len(hits)
                ),
                "case_kinds": sorted({str(item["case_kind"]) for item in items}),
            }
        )
    return aggregate


def write_outputs(
    rows: list[dict[str, object]],
    aggregate: list[dict[str, object]],
    anchors: dict[str, dict[str, object]],
) -> None:
    payload = {
        "schema_version": "s5000/tileops-manifest-cache-sweep-v1",
        "status": "valid",
        "scope": (
            "TileOPs manifest shape anchors executed by semantically equivalent "
            "S5000 TileLang-MUSA FP32 Add/ReLU kernels"
        ),
        "tileops_native_kernel_status": (
            "not_run_on_s5000; TileOPs elementwise kernels reject architecture 31"
        ),
        "cache_condition": "warm_after_5_skipped_launches",
        "l2_capacity_bytes": L2_CAPACITY_BYTES,
        "derived_hit_formula": (
            "max(0, 1 - ddr__read_bytes.sum / "
            "l1__downstream_read_bytes.sum)"
        ),
        "manifest_anchors": anchors,
        "runs": rows,
        "aggregate": aggregate,
        "conclusions": [
            "The current 99.75%/0% read-working-set step function is rejected.",
            "Warm hit rate declines before logical read bytes reach 60 MiB.",
            "Read plus dirty-output working set explains the transition better than read bytes alone.",
            "Unary and binary streams differ at similar resident byte counts, so one byte-only curve is insufficient.",
            "No replacement model is fitted from this sweep alone.",
        ],
    }
    (RESULT / "sweep_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TileOPs manifest锚点：S5000 elementwise warm L2容量扫描",
        "",
        "## 口径",
        "",
        "- manifest通过`tileops.manifest.load_workloads()`读取。",
        "- Add使用manifest的`2048×4096, float32`原始shape/dtype。",
        "- ReLU使用manifest的`2048×4096` shape；因manifest只声明FP16/BF16，而当前TileSight适配器只支持FP32，测量统一为FP32。",
        "- TileOPs原生elementwise Kernel只允许SM80/86/89/90，S5000架构31会被拒绝；本次使用语义等价的TileLang-MUSA Add/ReLU Kernel，未修改TileOPs白名单。",
        "- warm口径：连续发射6次，MCU `-s 5 -c 1`采第6次。",
        "- 字节命中率：`max(0, 1 - DDR read / L1 downstream read)`。",
        "",
        "## 聚合结果",
        "",
        "| 算子 | 元素 | 读MiB | 写MiB | 读+写MiB | 次数 | MCU字节命中率 | 当前模型 | 绝对误差点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        hit = float(item["derived_l2_byte_hit_pct_mean"])
        if int(item["run_count"]) > 1:
            hit_text = (
                f"{hit:.4f}% "
                f"[{float(item['derived_l2_byte_hit_pct_min']):.4f}, "
                f"{float(item['derived_l2_byte_hit_pct_max']):.4f}]"
            )
        else:
            hit_text = f"{hit:.4f}%"
        lines.append(
            f"| {item['op']} | {item['elements']} | "
            f"{float(item['logical_read_mib']):.1f} | "
            f"{float(item['logical_write_mib']):.1f} | "
            f"{float(item['resident_working_set_mib']):.1f} | "
            f"{item['run_count']} | {hit_text} | "
            f"{float(item['current_tilesight_prediction_pct']):.2f}% | "
            f"{float(item['current_tilesight_absolute_error_points']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 当前`read<=60 MiB → 99.75%`不成立。ReLU读取60 MiB但读+写120 MiB时仅0.0405%；Add读取60 MiB但读+写90 MiB时约4.0146%。",
            "2. 命中率在总驻留工作集接近60 MiB时已经连续下降，不是在读取工作集超过60 MiB后才突然变成0。",
            "3. 相同或相近总字节下，单输入ReLU与双输入Add仍有差异，后续模型至少需要输入流数量、交错顺序、输出write-allocate/dirty residency信息。",
            "4. 本轮足以否定当前二段常数，但不足以直接拟合稳定的新公式；还需要cold对照、反向二次扫描或只读对照来分离替换顺序与输出写分配。",
            "",
            "完整逐次数据见`sweep_summary.json`，原始证据为同目录的`.console.txt`和`.mcu-rep`。",
        ]
    )
    (RESULT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, anchors = build_rows()
    aggregate = aggregate_rows(rows)
    write_outputs(rows, aggregate, anchors)
    print(f"validated {len(rows)} MCU runs across {len(aggregate)} cases")
    print(RESULT / "sweep_summary.json")
    print(RESULT / "report.md")


if __name__ == "__main__":
    main()
