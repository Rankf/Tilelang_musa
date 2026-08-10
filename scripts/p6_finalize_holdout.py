#!/usr/bin/env python3
"""Aggregate frozen P6 hold-out measurements without altering P5."""

import csv
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p6_holdout"
OUT.mkdir(parents=True, exist_ok=True)


def median(values): return statistics.median(values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_complete():
    candidates = []
    for root in (ROOT / "results/raw/p6_holdout").glob("*"):
        required = [root / "preflight.json"]
        for category in ("compute", "layout"):
            required.extend(root / category / f"process_{i}" / "summary.csv" for i in (1, 2, 3))
        if all(path.exists() for path in required): candidates.append(root)
    if not candidates: raise RuntimeError("no complete P6 run")
    return sorted(candidates)[-1]


def main():
    root = latest_complete()
    preflight = json.loads((root / "preflight.json").read_text())
    scope = json.loads((root / "p6_holdout_scope.json").read_text())
    p5_model_path = ROOT / scope["p5_model_path"]
    p5_model_sha256_at_finalize = sha256(p5_model_path)
    p5_model_changed_after_preflight = (
        p5_model_sha256_at_finalize != preflight["p5_model_sha256"]
    )
    p5 = {item["case_id"]: item["p5"] for item in preflight["records"]}
    tables = []
    for category in ("compute", "layout"):
        for index in (1, 2, 3):
            with (root / category / f"process_{index}" / "summary.csv").open() as file:
                tables.append((category, {row["case_id"]: row for row in csv.DictReader(file)}))
    cases = {}
    for category, table in tables:
        for case_id, row in table.items(): cases.setdefault((category, case_id), []).append(row)
    output = []
    for (category, case_id), group in sorted(cases.items()):
        if len(group) != 3: raise RuntimeError(f"missing P6 process for {case_id}")
        latency_key = "latency_us_median" if category == "compute" else "warm_latency_us_median"
        profiler_key = "profiler_self_device_us"
        values = [float(row[latency_key]) for row in group]
        statuses = [row["status"] for row in group]
        correctness = [row["correctness"] for row in group]
        stable = all(x == "measured" for x in statuses) and all(x == "pass" for x in correctness)
        cross_span = (max(values) - min(values)) / median(values)
        stable = stable and cross_span <= .15
        prediction = p5[case_id]
        if prediction["status"] == "unknown":
            predicted, signed, ape, classification = "unknown", "unknown", "unknown", "model_coverage_unknown"
        else:
            predicted = prediction["prediction"]["predicted_latency_us"]
            signed = float(predicted) - median(values)
            ape = abs(float(predicted) / median(values) - 1) * 100
            classification = "predicted"
        output.append({
            "case_id": case_id, "category": category, "run_statuses": "|".join(statuses),
            "correctness_all_pass": str(all(x == "pass" for x in correctness)).lower(),
            "latency_us_run1": values[0], "latency_us_run2": values[1], "latency_us_run3": values[2],
            "actual_latency_us_cross_run_median": median(values), "cross_run_relative_span": cross_span,
            "profiler_self_device_us_cross_run_median": median(float(row[profiler_key]) for row in group),
            "measurement_eligible": str(stable).lower(), "p5_prediction_status": prediction["status"],
            "predicted_latency_us": predicted, "signed_error_us": signed, "absolute_percentage_error": ape,
            "error_classification": classification, "source_sha256": group[0].get("source_sha256", ""),
        })
    if p5_model_changed_after_preflight:
        decision = "evidence_invalid_p5_changed_after_preflight"
    elif not all(row["measurement_eligible"] == "true" for row in output):
        decision = "evidence_insufficient_measurement_gate_failed"
    elif all(row["p5_prediction_status"] == "unknown" for row in output):
        decision = "evidence_insufficient_p5_has_no_holdout_coverage"
    else:
        decision = "review_required_mixed_prediction_coverage"
    with (OUT / "holdout_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    report = {
        "run": str(root), "preflight": preflight, "results": output,
        "decision": decision,
        "p5_model_sha256_at_finalize": p5_model_sha256_at_finalize,
        "p5_model_changed_after_preflight": p5_model_changed_after_preflight,
    }
    (OUT / "holdout_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(root), "decision": decision, "cases": len(output)}, indent=2))


if __name__ == "__main__": main()
