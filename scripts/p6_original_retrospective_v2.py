#!/usr/bin/env python3
"""Read-only P5 v2 backtest on the already-observed original P6 set."""

import csv
import hashlib
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results/processed/p5_1_transfer_model/s5000_single_gpu_model_v2.json"
OLD = ROOT / "results/processed/p6_holdout/holdout_results.json"
OUT = ROOT / "results/processed/p6_original_retrospective_v2"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictor():
    spec = importlib.util.spec_from_file_location("p5_predict", ROOT / "scripts/p5_1_predict_transfer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads(MODEL_PATH.read_text())
    old = json.loads(OLD.read_text())
    conditions = {row["case_id"]: row["condition"] for row in old["preflight"]["records"]}
    predictor = load_predictor()
    output = []
    for row in old["results"]:
        case = conditions[row["case_id"]]
        family = "p3_global" if case["op"] == "strided_copy" else case["op"]
        prediction = predictor.predict(model, family, case)
        actual = float(row["actual_latency_us_cross_run_median"])
        if prediction["status"] == "predicted":
            value = float(prediction["prediction"]["predicted_latency_us"])
            signed, ape = value - actual, abs(value / actual - 1) * 100
        else:
            value = signed = ape = "unknown"
        output.append({"case_id": row["case_id"], "family": family, "actual_latency_us": actual,
                       "predicted_latency_us": value, "signed_error_us": signed, "absolute_percentage_error": ape,
                       "prediction_status": prediction["status"], "prediction_json": json.dumps(prediction, sort_keys=True)})
    with (OUT / "retrospective_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    report = {"kind": "retrospective only; original P6 values were observed before P5 v2 development and are not an independent acceptance test", "model_sha256": digest(MODEL_PATH), "source_original_p6": str(OLD), "results": output}
    (OUT / "retrospective_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(output), "all_predicted": all(row["prediction_status"] == "predicted" for row in output)}, indent=2))


if __name__ == "__main__": main()
