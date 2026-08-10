#!/usr/bin/env python3
"""Validate P5 input hashes, provenance paths, and exact-condition equations."""

import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p5_model"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model = json.loads((OUT / "s5000_single_gpu_model_v1.json").read_text())
    for item in model["source_manifest"]:
        path = ROOT / item["path"]
        if not path.exists() or digest(path) != item["sha256"]:
            raise RuntimeError(f"source manifest mismatch: {item['path']}")
    with (OUT / "s5000_parameter_registry_v1.csv").open(newline="") as file:
        parameters = list(csv.DictReader(file))
    for parameter in parameters:
        for value in json.loads(parameter["raw_runs_json"]):
            path = pathlib.Path(value)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                raise RuntimeError(f"missing provenance path for {parameter['parameter_id']}: {value}")
    numeric_records = 0
    with (OUT / "calibration_development_predictions_v1.csv").open(newline="") as file:
        for record in csv.DictReader(file):
            if record["prediction_mode"] == "conditional_tail_interval":
                interval = json.loads(record["tail_wave_us"])
                assert interval["min"] <= float(record["actual_latency_us"]) <= interval["max"]
                continue
            condition = json.loads(record["condition_json"])
            rate = float(record["effective_rate"])
            if record["effective_rate_units"] == "TFLOP/s":
                expected = 2 * condition["M"] * condition["N"] * condition["K"] / rate / 1e6
            elif record["effective_rate_units"] == "Gitems/s":
                expected = condition.get("N", condition.get("M", 0) * condition.get("K", 0)) / rate / 1e3
            elif record["effective_rate_units"] == "GB/s_logical":
                expected = json.loads(record["resource_decomposition_json"])["logical_payload_bytes"] / rate / 1e3
            else:
                raise RuntimeError(f"unknown rate unit: {record['effective_rate_units']}")
            if abs(expected - float(record["predicted_latency_us"])) > 1e-8:
                raise RuntimeError(f"formula mismatch: {record['case_id']}")
            numeric_records += 1
    print(json.dumps({"status": "pass", "parameters": len(parameters), "numeric_predictions": numeric_records}, indent=2))


if __name__ == "__main__":
    main()
