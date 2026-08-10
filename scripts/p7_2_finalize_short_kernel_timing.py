#!/usr/bin/env python3
"""Freeze P7.2 graph-free Event results without promoting rejected short cases."""
import argparse
import csv
import hashlib
import json
import pathlib
import statistics


def load_rows(root):
    rows = []
    for path in sorted(root.glob("**/result.json")):
        value = json.loads(path.read_text())
        if "case" in value:
            value["raw_result"] = str(path)
            rows.append(value)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--primary-root", type=pathlib.Path, required=True)
    ap.add_argument("--single-write-root", type=pathlib.Path, required=True)
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    timing = cfg["timing"]
    declared_case_ids = {case["id"] for case in cfg["cases"]}
    rows = load_rows(args.primary_root) + load_rows(args.single_write_root)
    groups = {}
    for row in rows:
        groups.setdefault(row["case"]["id"], []).append(row)
    result = []
    for case_id, trials in sorted(groups.items()):
        trials.sort(key=lambda x: x.get("trial", 0))
        measured = [x for x in trials if x.get("status") == "measured"]
        slopes = [x["fit"]["slope_us_per_launch"] for x in measured]
        required_trials = timing["trials_per_case"]
        span = (max(slopes) - min(slopes)) / statistics.mean(slopes) if slopes else None
        accepted = len(trials) == required_trials and len(measured) == required_trials and span is not None and span <= timing["max_trial_slope_span_fraction"]
        if case_id not in declared_case_ids:
            status, reason = "excluded_pre_revision_configuration_error", "raw trial belongs to a superseded invalid configuration; retained for audit but excluded from formal P7.2 case accounting"
        elif accepted:
            status, reason = "accepted_graph_free_event_slope", "all three trials passed individual fit gates and pre-registered cross-process slope span"
        elif len(trials) != required_trials:
            status, reason = "pending_short_kernel_method", "missing required trial count"
        elif len(measured) != required_trials:
            status, reason = "pending_short_kernel_method", "one or more trials failed compile/correctness/linear-fit gates"
        else:
            status, reason = "pending_short_kernel_method", "cross-process slope span exceeds pre-registered 5% threshold"
        exemplar = trials[0]
        result.append({
            "case_id": case_id, "op": exemplar["case"]["op"], "status": status, "reason": reason,
            "trials": len(trials), "measured_trials": len(measured), "slope_mean_us_per_launch": statistics.mean(slopes) if slopes else None,
            "slope_median_us_per_launch": statistics.median(slopes) if slopes else None,
            "slope_span_fraction": span,
            "intercept_mean_us": statistics.mean(x["fit"]["intercept_us"] for x in measured) if measured else None,
            "source_sha256s": sorted({x.get("source_sha256", "") for x in trials}),
            "trial_statuses": [x.get("status") for x in trials],
            "raw_results": [x["raw_result"] for x in trials]
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"phase": "P7.2", "protocol_version": cfg["protocol_version"], "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "primary_root": str(args.primary_root), "single_write_root": str(args.single_write_root), "rows": result}
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fields = ["case_id", "op", "status", "reason", "trials", "measured_trials", "slope_mean_us_per_launch", "slope_median_us_per_launch", "slope_span_fraction", "intercept_mean_us", "source_sha256s", "trial_statuses", "raw_results"]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in result:
            packed = {key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fields}
            writer.writerow(packed)
    print(json.dumps({"accepted": sum(x["status"].startswith("accepted") for x in result), "pending": sum(x["status"].startswith("pending") for x in result), "excluded": sum(x["status"].startswith("excluded") for x in result), "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__": main()
