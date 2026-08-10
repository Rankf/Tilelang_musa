#!/usr/bin/env python3
"""Aggregate frozen-prediction P6-v3 measurements without changing P5 v2."""

import csv
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p6_v3_holdout"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def median(values): return statistics.median(values)


def latest_complete():
    choices = []
    for root in (ROOT / "results/raw/p6_v3_holdout").glob("*"):
        required = [root / "preflight.json"] + [root / category / f"process_{i}" / "summary.csv" for category in ("compute", "layout") for i in (1, 2, 3)]
        if all(path.exists() for path in required): choices.append(root)
    if not choices: raise RuntimeError("no complete P6-v3 run")
    return sorted(choices)[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = latest_complete()
    scope = json.loads((root / "p6_v3_holdout_scope.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    model_hash = digest(ROOT / scope["p5_model_path"])
    changed = model_hash != preflight["p5_model_sha256"]
    predicted = {x["case_id"]: x["p5_v2"]["prediction"] for x in preflight["records"]}
    grouped = {}
    for category, latency_key in (("compute", "latency_us_median"), ("layout", "warm_latency_us_median")):
        for i in (1, 2, 3):
            with (root / category / f"process_{i}" / "summary.csv").open() as file:
                for row in csv.DictReader(file): grouped.setdefault((category, row["case_id"]), []).append(row)
    output = []
    for (category, case_id), group in sorted(grouped.items()):
        if len(group) != 3: raise RuntimeError(f"missing process for {case_id}")
        key = "latency_us_median" if category == "compute" else "warm_latency_us_median"
        values = [float(row[key]) for row in group]
        actual = median(values)
        stable = all(row["status"] == "measured" and row["correctness"] == "pass" for row in group) and (max(values)-min(values))/actual <= .15
        pred = float(predicted[case_id]["predicted_latency_us"])
        output.append({"case_id": case_id, "category": category, "actual_latency_us_cross_run_median": actual,
                       "latency_us_run1": values[0], "latency_us_run2": values[1], "latency_us_run3": values[2],
                       "cross_run_relative_span": (max(values)-min(values))/actual, "measurement_eligible": stable,
                       "correctness_all_pass": all(row["correctness"] == "pass" for row in group),
                       "predicted_latency_us": pred, "signed_error_us": pred-actual,
                       "absolute_percentage_error": abs(pred/actual-1)*100,
                       "error_classification": "prediction_error" if stable else "measurement_gate_failed",
                       "source_sha256": group[0].get("source_sha256", "")})
    if changed: decision = "evidence_invalid_p5_v2_changed_after_preflight"
    elif not all(row["measurement_eligible"] for row in output): decision = "evidence_insufficient_measurement_gate_failed"
    else: decision = "review_required_numeric_holdout"
    report = {"run": str(root), "preflight": preflight, "p5_model_sha256_at_finalize": model_hash,
              "p5_model_changed_after_preflight": changed, "results": output, "decision": decision}
    with (OUT / "holdout_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    (OUT / "holdout_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(output), "decision": decision, "median_ape_percent": median([row["absolute_percentage_error"] for row in output])}, indent=2))


if __name__ == "__main__": main()
