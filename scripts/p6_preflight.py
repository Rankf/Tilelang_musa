#!/usr/bin/env python3
"""Freeze P5's prediction state for P6 before any hold-out kernel is run."""

import argparse
import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_for(op):
    if op == "gemm": return "gemm"
    if op == "strided_copy": return "p3_global"
    return op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    model_path = ROOT / scope["p5_model_path"]
    observed_sha = sha256(model_path)
    if observed_sha != scope["p5_model_sha256"]:
        raise RuntimeError("P5 model hash differs from pre-registered P6 scope; regenerate scope before measurement")
    predictions_path = ROOT / "results/processed/p5_model/calibration_development_predictions_v1.csv"
    with predictions_path.open(newline="") as file:
        predictions = list(csv.DictReader(file))
    records = []
    for config_name in scope["holdout_configs"]:
        config_path = ROOT / config_name
        config = json.loads(config_path.read_text())
        for case in config["cases"]:
            condition = json.dumps(case, sort_keys=True, separators=(",", ":"))
            family = family_for(case["op"])
            matches = [row for row in predictions if row["family"] == family and row["condition_json"] == condition]
            if len(matches) == 1:
                outcome = {"status": "predicted_exact_condition", "prediction": matches[0]}
            elif len(matches) == 0:
                outcome = {"status": "unknown", "reason": "no exact P5 calibrated condition; cross-condition transfer is forbidden"}
            else:
                raise RuntimeError(f"ambiguous P5 condition for {case['id']}")
            records.append({"case_id": case["id"], "op": case["op"], "family": family,
                            "condition": case, "p5": outcome})
    output = {"scope_sha256": sha256(args.scope), "p5_model_sha256": observed_sha,
              "preflight_before_measurement": True, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
