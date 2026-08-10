#!/usr/bin/env python3
"""Audit P5 v2 provenance, isolation and formula-consistent predictions."""

import csv
import hashlib
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p5_1_transfer_model"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def predictor():
    path = ROOT / "scripts/p5_1_predict_transfer.py"
    spec = importlib.util.spec_from_file_location("p5_predict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    model = json.loads((OUT / "s5000_single_gpu_model_v2.json").read_text())
    for item in model["source_manifest"]:
        path = ROOT / item["path"]
        if "p6_holdout" in item["path"] or "p6_v2_holdout" in item["path"] or "p6_v3_holdout" in item["path"]:
            raise RuntimeError(f"hold-out path leaked into model source manifest: {item['path']}")
        if not path.exists() or digest(path) != item["sha256"]:
            raise RuntimeError(f"source manifest mismatch: {item['path']}")
    predict = predictor()
    checked = 0
    with (OUT / "development_predictions.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["prediction_mode"] != "development_fit_not_validation":
                continue
            result = predict.predict(model, row["family"], json.loads(row["condition_json"]))
            if result["status"] != "predicted":
                raise RuntimeError(f"development model no longer covers {row['case_id']}")
            observed = float(result["prediction"]["predicted_latency_us"])
            if abs(observed - float(row["predicted_latency_us"])) > 1e-8:
                raise RuntimeError(f"formula mismatch: {row['case_id']}")
            checked += 1
    scope = ROOT / "configs/p6_v3_holdout_scope.json"
    frozen = json.loads(scope.read_text())
    current = digest(ROOT / frozen["p5_model_path"])
    if current != frozen["p5_model_sha256"]:
        raise RuntimeError("P6-v3 scope model SHA mismatch")
    print(json.dumps({"status": "pass", "formula_checked_records": checked, "model_sha256": current}, indent=2))


if __name__ == "__main__": main()
