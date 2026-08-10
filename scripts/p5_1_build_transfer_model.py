#!/usr/bin/env python3
"""Build a bounded P5 v2 transfer model without reading either P6 hold-out."""

import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p5_1_transfer_model"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def bool_value(value):
    return str(value).lower() == "true"


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def median(values):
    return statistics.median(values)


def latest_complete(root, categories):
    choices = []
    for run in root.glob("*"):
        required = [run / category / f"process_{i}" / "summary.csv"
                    for category in categories for i in (1, 2, 3)]
        if all(path.exists() for path in required):
            choices.append(run)
    if not choices:
        raise RuntimeError(f"no complete run under {root}")
    return sorted(choices)[-1]


def aggregate_run(root, category, config_name, latency_key):
    config = json.loads((root / config_name).read_text())
    by_id = {}
    source_paths = [root / config_name]
    for i in (1, 2, 3):
        path = root / category / f"process_{i}" / "summary.csv"
        source_paths.append(path)
        for row in rows(path):
            by_id.setdefault(row["case_id"], []).append(row)
    cases = {case["id"]: case for case in config["cases"]}
    output = []
    for case_id, case in sorted(cases.items()):
        group = by_id.get(case_id, [])
        if len(group) != 3:
            raise RuntimeError(f"{root}: {case_id} lacks three process records")
        values = [float(row[latency_key]) for row in group]
        statuses = [row["status"] for row in group]
        correctness = [row["correctness"] for row in group]
        span = (max(values) - min(values)) / median(values)
        eligible = all(value == "measured" for value in statuses) and all(value == "pass" for value in correctness) and span <= .15
        output.append({
            "case_id": case_id, "case": case, "category": category,
            "latency_us_cross_run_median": median(values), "cross_run_relative_span": span,
            "measurement_eligible": eligible, "run_statuses": "|".join(statuses),
            "correctness_all_pass": all(value == "pass" for value in correctness),
            "profiler_self_device_us_cross_run_median": median(float(row["profiler_self_device_us"]) for row in group),
            "source_sha256": sorted({row.get("source_sha256", "") for row in group}),
            "raw_run_paths": [str(root / category / f"process_{i}") for i in (1, 2, 3)],
        })
    return output, source_paths


def nonnegative_least_squares(feature_names, observations, required_positive):
    """Small deterministic NNLS by enumerating active feature subsets."""
    x = np.array([[obs["features"][name] for name in feature_names] for obs in observations], dtype=float)
    y = np.array([obs["actual_latency_us"] for obs in observations], dtype=float)
    best = None
    for mask in range(1, 1 << len(feature_names)):
        active = [i for i in range(len(feature_names)) if mask & (1 << i)]
        coefficient, _, _, _ = np.linalg.lstsq(x[:, active], y, rcond=None)
        if any(value < -1e-10 for value in coefficient):
            continue
        full = np.zeros(len(feature_names))
        full[active] = coefficient
        if any(full[feature_names.index(name)] <= 1e-12 for name in required_positive):
            continue
        predicted = x @ full
        sse = float(np.square(predicted - y).sum())
        candidate = (sse, full, predicted)
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
    if best is None:
        raise RuntimeError(f"no non-negative fit with required features {required_positive}")
    sse, coefficient, predicted = best
    return {
        "feature_names": feature_names,
        "coefficients": {name: float(coefficient[i]) for i, name in enumerate(feature_names)},
        "fit_sse": sse,
        "predictions": [float(value) for value in predicted],
    }


def range_of(observations, field):
    values = [obs["features"][field] for obs in observations]
    return {"min": min(values), "max": max(values)}


def add_linear_prediction(predictions, family, model, observations):
    coef = model["coefficients"]
    for obs, value in zip(observations, model["predictions"]):
        actual = obs["actual_latency_us"]
        predictions.append({
            "case_id": obs["case_id"], "family": family, "split": obs["split"],
            "prediction_mode": "development_fit_not_validation", "condition_json": compact(obs["case"]),
            "actual_latency_us": actual, "predicted_latency_us": value,
            "signed_error_us": value - actual,
            "absolute_percentage_error": abs(value / actual - 1) * 100,
            "features_json": compact(obs["features"]), "coefficients_json": compact(coef),
            "raw_runs_json": compact(obs["raw_runs"]),
        })


def p2_observations(p2):
    result = []
    for row in p2["cases"]:
        if not bool_value(row["p2_parameter_eligible"]):
            continue
        case = json.loads(row["case_config_json"])
        result.append({"case_id": row["case_id"], "case": case,
                       "actual_latency_us": float(row["graph_latency_us_cross_run_median"]),
                       "split": "P2_calibration", "raw_runs": p2["runs"]})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    development_run = latest_complete(ROOT / "results/raw/p5_1_transfer_development", ("compute", "layout"))
    refinement_run = latest_complete(ROOT / "results/raw/p5_1_transfer_refinement", ("compute", "layout"))
    dev_compute, dev_compute_paths = aggregate_run(development_run, "compute", "p5_1_transfer_development_compute.json", "latency_us_median")
    dev_layout, dev_layout_paths = aggregate_run(development_run, "layout", "p5_1_transfer_development_layout.json", "warm_latency_us_median")
    refine_compute, refine_compute_paths = aggregate_run(refinement_run, "compute", "p5_1_transfer_refinement_compute.json", "latency_us_median")
    refine_layout, refine_layout_paths = aggregate_run(refinement_run, "layout", "p5_1_transfer_refinement_layout.json", "warm_latency_us_median")
    ext_compute, ext_paths = [], []
    if not all(row["measurement_eligible"] for row in dev_compute + dev_layout + refine_compute + refine_layout):
        failures = [row["case_id"] for row in dev_compute + dev_layout + refine_compute + refine_layout if not row["measurement_eligible"]]
        raise RuntimeError(f"development measurement gate failed: {failures}")

    static_paths = [
        ROOT / "results/processed/p2_compute/aggregate.json",
        ROOT / "results/processed/p3_memory/aggregate_global_initial.csv",
        ROOT / "configs/p5_1_transfer_development_compute.json",
        ROOT / "configs/p5_1_transfer_development_layout.json",
        ROOT / "configs/p5_1_transfer_refinement_compute.json",
        ROOT / "configs/p5_1_transfer_refinement_layout.json",
        pathlib.Path(__file__),
    ]
    source_paths = static_paths + dev_compute_paths + dev_layout_paths + refine_compute_paths + refine_layout_paths + ext_paths
    if any("p6_holdout" in str(path) or "p6_v2_holdout" in str(path) for path in source_paths):
        raise RuntimeError("P6 path leaked into P5.1 input manifest")
    source_manifest = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in source_paths]

    p2 = json.loads((ROOT / "results/processed/p2_compute/aggregate.json").read_text())
    p2_obs = p2_observations(p2)
    aggregate_rows = []
    for row in dev_compute + dev_layout + refine_compute + refine_layout + ext_compute:
        aggregate_rows.append({
            "case_id": row["case_id"], "category": row["category"], "condition_json": compact(row["case"]),
            "latency_us_cross_run_median": row["latency_us_cross_run_median"],
            "cross_run_relative_span": row["cross_run_relative_span"],
            "measurement_eligible": row["measurement_eligible"], "run_statuses": row["run_statuses"],
            "correctness_all_pass": row["correctness_all_pass"],
            "profiler_self_device_us_cross_run_median": row["profiler_self_device_us_cross_run_median"],
            "source_sha256_json": compact(row["source_sha256"]), "raw_runs_json": compact(row["raw_run_paths"]),
        })

    predictions = []
    # FP16 GEMM: no source/resource identity claim; grid_cta is only the visible launch-grid feature.
    gemm = []
    for obs in p2_obs + [{"case_id": row["case_id"], "case": row["case"], "actual_latency_us": row["latency_us_cross_run_median"], "split": "P5_1_development", "raw_runs": row["raw_run_paths"]} for row in dev_compute + refine_compute + ext_compute]:
        c = obs["case"]
        if not (c.get("op") == "gemm" and c.get("dtype") == "float16" and c.get("accum_dtype") == "float32" and
                c.get("block_M") == c.get("block_N") == c.get("block_K") == 64 and c.get("num_warp") == 4 and
                c.get("policy") == "default" and c.get("num_stages") == 1 and
                all(c[key] % 64 == 0 for key in ("M", "N", "K"))):
            continue
        ops = 2 * c["M"] * c["N"] * c["K"] / 1e9
        grid = (c["M"] // 64) * (c["N"] // 64)
        obs["features"] = {"bias": 1.0, "ops_giga": ops, "ops_giga_per_grid_cta": ops / grid, "grid_cta": grid}
        gemm.append(obs)
    gemm_fit = nonnegative_least_squares(["bias", "ops_giga", "ops_giga_per_grid_cta"], gemm, ["ops_giga"])
    gemm_model = {**gemm_fit, "formula": "bias + ops_giga + ops_giga_per_grid_cta", "domains": {
        "dtype": "float16", "accum_dtype": "float32", "block_M": 64, "block_N": 64, "block_K": 64,
        "num_warp": 4, "policy": "default", "num_stages": 1, "divisibility": 64,
        "ops_giga": range_of(gemm, "ops_giga"), "grid_cta": range_of(gemm, "grid_cta"),
        "ops_giga_per_grid_cta": range_of(gemm, "ops_giga_per_grid_cta")}}
    add_linear_prediction(predictions, "gemm", gemm_fit, gemm)

    reduce = []
    for obs in p2_obs + [{"case_id": row["case_id"], "case": row["case"], "actual_latency_us": row["latency_us_cross_run_median"], "split": "P5_1_development", "raw_runs": row["raw_run_paths"]} for row in dev_compute + refine_compute + ext_compute]:
        c = obs["case"]
        if not (c.get("op") == "reduce_sum" and c.get("dtype") == "float32" and c.get("threads") == 128):
            continue
        obs["features"] = {"bias": 1.0, "input_million": c["M"] * c["K"] / 1e6, "cta_million": c["M"] / 4096}
        reduce.append(obs)
    reduce_fit = nonnegative_least_squares(["bias", "input_million", "cta_million"], reduce, ["input_million"])
    reduce_model = {**reduce_fit, "formula": "bias + input_million + cta_million", "domains": {
        "dtype": "float32", "threads": 128, "M": {"min": min(x["case"]["M"] for x in reduce), "max": max(x["case"]["M"] for x in reduce)},
        "K": {"min": min(x["case"]["K"] for x in reduce), "max": max(x["case"]["K"] for x in reduce)},
        "input_million": range_of(reduce, "input_million"), "cta_million": range_of(reduce, "cta_million")}}
    add_linear_prediction(predictions, "reduce_sum", reduce_fit, reduce)

    fused = []
    for row in dev_compute + refine_compute:
        c = row["case"]
        # The 4M point has a known short-kernel regime change.  P2 established
        # that short paths need a separate fixed-overhead study, so v2's steady
        # fused domain begins at 8M rather than fitting a false universal line.
        if c.get("op") != "fused_exp_add" or c["N"] < 8 * 1024 * 1024:
            continue
        fused.append({"case_id": row["case_id"], "case": c, "actual_latency_us": row["latency_us_cross_run_median"],
                      "split": "P5_1_development", "raw_runs": row["raw_run_paths"],
                      "features": {"bias": 1.0, "N_million": c["N"] / 1e6}})
    fused_fit = nonnegative_least_squares(["bias", "N_million"], fused, ["N_million"])
    fused_model = {**fused_fit, "formula": "bias + N_million", "domains": {
        "dtype": "float32", "threads": 256, "vector": 8, "N": {"min": min(x["case"]["N"] for x in fused), "max": max(x["case"]["N"] for x in fused)},
        "N_million": range_of(fused, "N_million")}}
    add_linear_prediction(predictions, "fused_exp_add", fused_fit, fused)

    # Stride rule: fixed log2(stride) interpolation of logical effective bandwidth only.
    stride = []
    p3_rows = rows(ROOT / "results/processed/p3_memory/aggregate_global_initial.csv")
    p3_run = json.loads((ROOT / "results/processed/p3_memory/aggregate_global_initial.json").read_text())["runs"]
    for row in p3_rows:
        c = json.loads(row["case_config_json"])
        if c.get("op") == "strided_copy" and bool_value(row["parameter_eligible"]):
            stride.append({"case_id": row["case_id"], "case": c, "rate_GBps": float(row["logical_effective_GBps_cross_run_median"]), "split": "P3_calibration", "raw_runs": p3_run})
    for row in dev_layout + refine_layout:
        c = row["case"]
        stride.append({"case_id": row["case_id"], "case": c, "rate_GBps": c and (8 * c["N"] / row["latency_us_cross_run_median"] / 1e3), "split": "P5_1_development", "raw_runs": row["raw_run_paths"]})
    groups = {}
    for item in stride:
        c = item["case"]
        key = compact({name: c[name] for name in ("N", "threads", "vector", "cache_condition")})
        groups.setdefault(key, []).append(item)
    stride_groups = []
    for key, points in sorted(groups.items()):
        points.sort(key=lambda item: item["case"]["stride"])
        if len({point["case"]["stride"] for point in points}) != len(points):
            raise RuntimeError(f"duplicate stride calibration point in {key}")
        stride_groups.append({"selector": json.loads(key), "points": [{"stride": point["case"]["stride"], "rate_GBps": point["rate_GBps"], "case_id": point["case_id"], "split": point["split"], "raw_runs": point["raw_runs"]} for point in points]})
        for index in range(1, len(points) - 1):
            left, point, right = points[index - 1], points[index], points[index + 1]
            weight = (math.log2(point["case"]["stride"]) - math.log2(left["case"]["stride"])) / (math.log2(right["case"]["stride"]) - math.log2(left["case"]["stride"]))
            predicted_rate = left["rate_GBps"] + weight * (right["rate_GBps"] - left["rate_GBps"])
            actual = 8 * point["case"]["N"] / point["rate_GBps"] / 1e3
            predicted = 8 * point["case"]["N"] / predicted_rate / 1e3
            predictions.append({"case_id": point["case_id"], "family": "p3_global", "split": point["split"], "prediction_mode": "development_leave_one_out_log2_stride_interpolation", "condition_json": compact(point["case"]), "actual_latency_us": actual, "predicted_latency_us": predicted, "signed_error_us": predicted - actual, "absolute_percentage_error": abs(predicted / actual - 1) * 100, "features_json": compact({"left_stride": left["case"]["stride"], "right_stride": right["case"]["stride"]}), "coefficients_json": "not_applicable", "raw_runs_json": compact(point["raw_runs"])})

    models = {"gemm": gemm_model, "reduce_sum": reduce_model, "fused_exp_add": fused_model,
              "p3_global": {"formula": "linear effective-bandwidth interpolation in log2(stride), then payload/rate", "groups": stride_groups}}
    numeric_apes = [float(row["absolute_percentage_error"]) for row in predictions]
    model = {"model_version": "s5000-single-gpu-v2-transfer", "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "scope": "bounded P5.1 development transfer model; P6/P6-v2 data excluded", "source_manifest": source_manifest,
             "development_runs": {"base": str(development_run), "refinement": str(refinement_run), "extension": "excluded: Graph-replay runtime failures"}, "models": models,
             "development_fit_summary": {"records": len(predictions), "median_ape_percent": statistics.median(numeric_apes), "maximum_ape_percent": max(numeric_apes), "is_holdout_validation": False},
             "known_unknowns": ["resource occupancy/wave mapping for unseen TileLang source", "cache-tier capacity", "extrapolation outside declared feature ranges", "other dtype/tile/pipeline/layout families"]}
    with (OUT / "development_aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate_rows[0])); writer.writeheader(); writer.writerows(aggregate_rows)
    (OUT / "development_aggregate.json").write_text(json.dumps({"base_run": str(development_run), "refinement_run": str(refinement_run), "extension_run": "excluded: Graph-replay runtime failures", "cases": aggregate_rows}, indent=2) + "\n")
    with (OUT / "development_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (OUT / "s5000_single_gpu_model_v2.json").write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps({"model": "s5000_single_gpu_model_v2.json", "development_records": len(predictions), "median_ape_percent": model["development_fit_summary"]["median_ape_percent"]}, indent=2))


if __name__ == "__main__":
    main()
