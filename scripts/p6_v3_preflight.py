#!/usr/bin/env python3
"""Freeze numerical P5 v2 predictions before any P6-v3 compilation."""

import argparse
import hashlib
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictor():
    path = ROOT / "scripts/p5_1_predict_transfer.py"
    spec = importlib.util.spec_from_file_location("p5_1_predict_transfer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family(op):
    return "p3_global" if op == "strided_copy" else op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    model_path = ROOT / scope["p5_model_path"]
    model_hash = digest(model_path)
    if model_hash != scope["p5_model_sha256"]:
        raise RuntimeError("frozen P5 v2 model hash does not match P6-v3 scope")
    predictor = load_predictor()
    model = json.loads(model_path.read_text())
    records = []
    for name in scope["holdout_configs"]:
        for case in json.loads((ROOT / name).read_text())["cases"]:
            outcome = predictor.predict(model, family(case["op"]), case)
            records.append({"case_id": case["id"], "family": family(case["op"]), "condition": case, "p5_v2": outcome})
    if any(record["p5_v2"]["status"] != "predicted" for record in records):
        raise RuntimeError("P6-v3 preflight requires a numerical prediction for every registered case")
    result = {"scope_sha256": digest(args.scope), "p5_model_sha256": model_hash, "preflight_before_measurement": True, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "pass", "predicted_cases": len(records), "p5_model_sha256": model_hash}, indent=2))


if __name__ == "__main__":
    main()
