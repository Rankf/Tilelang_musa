#!/usr/bin/env python3
"""Return an exact P5 condition prediction or an explicit unknown result."""

import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/processed/p5_model/calibration_development_predictions_v1.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, help="e.g. gemm, p3_global, vector_add")
    parser.add_argument("--condition-json", required=True, help="full condition JSON object")
    args = parser.parse_args()
    requested = json.dumps(json.loads(args.condition_json), sort_keys=True, separators=(",", ":"))
    with PREDICTIONS.open(newline="") as file:
        matches = [row for row in csv.DictReader(file)
                   if row["family"] == args.family and row["condition_json"] == requested]
    if len(matches) == 1:
        print(json.dumps({"status": "exact_condition_match", "prediction": matches[0]}, indent=2))
        return
    if len(matches) > 1:
        raise RuntimeError("condition is ambiguous; model registry needs a finer selector")
    print(json.dumps({
        "status": "unknown",
        "reason": "no exact calibrated condition; P5 forbids cross-condition rate transfer",
        "family": args.family,
        "condition_json": json.loads(args.condition_json),
    }, indent=2))


if __name__ == "__main__":
    main()
