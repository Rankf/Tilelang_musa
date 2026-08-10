#!/usr/bin/env python3
"""Summarize staged P7 graph diagnostics; no performance parameter is emitted."""

import csv
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p7_large_gemm_graph"


def latest_complete():
    roots = []
    for root in (ROOT / "results/raw/p7_large_gemm_graph").glob("*"):
        config = root / "p7_large_gemm_graph_diagnostic.json"
        if not config.exists():
            continue
        cfg = json.loads(config.read_text())
        expected = [root / f"unroll_{unroll}" / f"trial_{trial}" / "result.json"
                    for unroll in cfg["graph_unrolls"] for trial in range(1, cfg["trials_per_unroll"] + 1)]
        if all(path.exists() for path in expected): roots.append(root)
    if not roots: raise RuntimeError("no complete P7 diagnostic run")
    return sorted(roots)[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = latest_complete()
    cfg = json.loads((root / "p7_large_gemm_graph_diagnostic.json").read_text())
    all_rows, output = [], []
    for unroll in cfg["graph_unrolls"]:
        rows = [json.loads((root / f"unroll_{unroll}" / f"trial_{trial}" / "result.json").read_text())
                for trial in range(1, cfg["trials_per_unroll"] + 1)]
        all_rows.extend(rows)
        passed = [row for row in rows if row["status"] == "pass"]
        failures = [row["failure_stage"] for row in rows if row["status"] != "pass"]
        output.append({"graph_unroll": unroll, "trials": len(rows), "passed": len(passed), "failed": len(failures),
                       "failure_stages": "|".join(failures) if failures else "",
                       "direct_event_us_median_successes": statistics.median([float(row["direct_event_us"]) for row in passed]) if passed else "",
                       "graph_replay_us_median_successes": statistics.median([float(row["graph_replay_us_median"]) for row in passed]) if passed else ""})
    with (OUT / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    report = {"run": str(root), "config": cfg, "summary": output, "raw_rows": all_rows,
              "parameter_status": "not_a_calibration_parameter; diagnostic only"}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(root), "summary": output}, indent=2))


if __name__ == "__main__": main()
