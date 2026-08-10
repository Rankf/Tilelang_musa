#!/usr/bin/env python3
"""Predict only inside the explicit P5 v2 transfer-model domains."""

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "results/processed/p5_1_transfer_model/s5000_single_gpu_model_v2.json"


def inside(value, bounds):
    return bounds["min"] <= value <= bounds["max"]


def linear(model, features):
    return sum(model["coefficients"][name] * features[name] for name in model["feature_names"])


def predict(model, family, c):
    models = model["models"]
    if family == "gemm":
        m = models["gemm"]; d = m["domains"]
        fixed = (c.get("op") == "gemm" and c.get("dtype") == d["dtype"] and c.get("accum_dtype") == d["accum_dtype"] and c.get("block_M") == d["block_M"] and c.get("block_N") == d["block_N"] and c.get("block_K") == d["block_K"] and c.get("num_warp") == d["num_warp"] and c.get("policy") == d["policy"] and c.get("num_stages") == d["num_stages"])
        if not fixed or any(c.get(key, -1) % d["divisibility"] for key in ("M", "N", "K")):
            return {"status": "unknown", "reason": "GEMM fixed schedule/dtype/divisibility outside v2 domain"}
        grid = (c["M"] // 64) * (c["N"] // 64); ops = 2 * c["M"] * c["N"] * c["K"] / 1e9
        f = {"bias": 1.0, "ops_giga": ops, "ops_giga_per_grid_cta": ops / grid, "grid_cta": grid}
        if not all(inside(f[name], d[name]) for name in ("ops_giga", "grid_cta", "ops_giga_per_grid_cta")):
            return {"status": "unknown", "reason": "GEMM derived feature outside v2 interpolation domain", "features": f}
        return {"status": "predicted", "prediction": {"predicted_latency_us": linear(m, f), "features": f, "formula": m["formula"]}}
    if family == "reduce_sum":
        m = models["reduce_sum"]; d = m["domains"]
        if not (c.get("op") == "reduce_sum" and c.get("dtype") == d["dtype"] and c.get("threads") == d["threads"]):
            return {"status": "unknown", "reason": "reduce fixed dtype/thread configuration outside v2 domain"}
        f = {"bias": 1.0, "input_million": c["M"] * c["K"] / 1e6, "cta_million": c["M"] / 4096}
        if not (inside(c["M"], d["M"]) and inside(c["K"], d["K"]) and inside(f["input_million"], d["input_million"]) and inside(f["cta_million"], d["cta_million"])):
            return {"status": "unknown", "reason": "reduce feature outside v2 interpolation domain", "features": f}
        return {"status": "predicted", "prediction": {"predicted_latency_us": linear(m, f), "features": f, "formula": m["formula"]}}
    if family == "fused_exp_add":
        m = models["fused_exp_add"]; d = m["domains"]
        if not (c.get("op") == "fused_exp_add" and c.get("dtype") == d["dtype"] and c.get("threads") == d["threads"] and c.get("vector") == d["vector"] and inside(c["N"], d["N"])):
            return {"status": "unknown", "reason": "fused exp+add feature outside v2 interpolation domain"}
        f = {"bias": 1.0, "N_million": c["N"] / 1e6}
        return {"status": "predicted", "prediction": {"predicted_latency_us": linear(m, f), "features": f, "formula": m["formula"]}}
    if family == "p3_global":
        if c.get("op") != "strided_copy":
            return {"status": "unknown", "reason": "p3_global v2 rule only supports strided_copy"}
        selector = {name: c.get(name) for name in ("N", "threads", "vector", "cache_condition")}
        for group in models["p3_global"]["groups"]:
            if group["selector"] != selector:
                continue
            points = group["points"]
            stride = c.get("stride")
            for left, right in zip(points, points[1:]):
                if left["stride"] <= stride <= right["stride"]:
                    weight = (math.log2(stride) - math.log2(left["stride"])) / (math.log2(right["stride"]) - math.log2(left["stride"]))
                    rate = left["rate_GBps"] + weight * (right["rate_GBps"] - left["rate_GBps"])
                    return {"status": "predicted", "prediction": {"predicted_latency_us": 8 * c["N"] / rate / 1e3, "effective_rate_GBps": rate, "bracket": [left["stride"], right["stride"]], "formula": models["p3_global"]["formula"]}}
            return {"status": "unknown", "reason": "stride is outside v2 interpolation brackets"}
        return {"status": "unknown", "reason": "no exact N/thread/vector/cache selector for stride interpolation"}
    return {"status": "unknown", "reason": f"unsupported family {family}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition-json", required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    output = predict(model, args.family, json.loads(args.condition_json))
    output.update({"model_version": model["model_version"], "model_path": str(args.model)})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
