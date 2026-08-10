#!/usr/bin/env python3
"""Aggregate P7.3 pilot without selecting only successful schedules."""
import argparse, csv, hashlib, json, pathlib, statistics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=pathlib.Path, required=True); ap.add_argument("--raw-root", type=pathlib.Path, required=True); ap.add_argument("--output-dir", type=pathlib.Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); required = cfg["timing"]["trials_per_schedule"]; maximum = cfg["timing"]["max_cross_trial_median_span_fraction"]
    groups = {}
    for path in sorted(args.raw_root.glob("**/result.json")):
        row = json.loads(path.read_text())
        if "case" not in row or "schedule" not in row: continue
        groups.setdefault((row["case"]["id"], row["schedule"]["id"]), []).append((path, row))
    output = []
    for (_, _), items in sorted(groups.items()):
        items.sort(key=lambda x: x[1].get("trial", 0)); rows = [x[1] for x in items]; measured = [x for x in rows if x.get("status") == "measured"]
        medians = [x["latency_us_median"] for x in measured]
        span = (max(medians) - min(medians)) / statistics.median(medians) if medians else None
        accepted = len(rows) == required and len(measured) == required and span is not None and span <= maximum
        if accepted: status, reason = "development_measured", "three correct/stable trials and pre-registered cross-trial median span passed"
        elif len(rows) != required: status, reason = "incomplete", "missing required trial count"
        elif len(measured) != required: status, reason = "not_development_measured", "one or more trials did not pass compile/correctness/within-trial stability"
        else: status, reason = "not_development_measured", "cross-trial median span exceeds pre-registered 5% threshold"
        ex = rows[0]
        output.append({"case_id": ex["case"]["id"], "shape_role": ex["case"]["shape_role"], "M": ex["case"]["M"], "N": ex["case"]["N"], "K": ex["case"]["K"], "schedule_id": ex["schedule"]["id"], "block_M": ex["schedule"]["block_M"], "block_N": ex["schedule"]["block_N"], "block_K": ex["schedule"]["block_K"], "num_warp": ex["schedule"]["num_warp"], "num_stages": ex["schedule"]["num_stages"], "status": status, "reason": reason, "trials": len(rows), "measured_trials": len(measured), "latency_us_median_of_trials": statistics.median(medians) if medians else None, "cross_trial_median_span_fraction": span, "effective_tops_median": statistics.median(x["effective_tops"] for x in measured) if measured else None, "grid_cta": ex.get("grid_cta"), "source_sha256s": sorted({x.get("source_sha256", "") for x in rows}), "source_features": ex.get("source_features"), "trial_statuses": [x.get("status") for x in rows], "raw_results": [str(path) for path, _ in items]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"phase": "P7.3", "protocol_version": cfg["protocol_version"], "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "raw_root": str(args.raw_root), "rows": output}
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    fields = list(output[0]) if output else ["case_id", "schedule_id", "status"]
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in output: writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    print(json.dumps({"development_measured": sum(x["status"] == "development_measured" for x in output), "not_measured": sum(x["status"] != "development_measured" for x in output), "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__": main()
