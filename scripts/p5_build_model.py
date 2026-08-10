#!/usr/bin/env python3
"""Build P5's evidence-only, condition-keyed S5000 single-GPU model.

This is deliberately a calibration/development registry, not a shape-transfer
model.  Any condition not found verbatim is reported as unknown for P6 rather
than borrowing a rate from a different cache, layout, or GEMM geometry.
"""

import csv
import datetime as dt
import hashlib
import json
import pathlib
import platform
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p5_model"
OUT.mkdir(parents=True, exist_ok=True)
SCOPE = ROOT / "configs/p5_scope.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def as_bool(value):
    return str(value).lower() == "true"


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def gemm_condition(case):
    # M/N/K are intentionally part of the key: P5 evidence proves that a
    # dtype/tile/stage-only key is not a safe rate-transfer rule.
    return {key: case[key] for key in ("dtype", "accum_dtype", "M", "N", "K", "block_M",
                                        "block_N", "block_K", "num_stages", "num_warp", "policy")}


def trace_runs_from_p3(aggregate_json):
    return json.loads(aggregate_json.read_text()).get("runs", [])


def p4_trace(parameter, manifests):
    evidence = parameter["evidence"]
    if "aggregate_native" in evidence:
        return [manifests["run"]["native_run"]]
    if "aggregate_tilelang.csv" in evidence:
        return [manifests["run"]["tilelang_run"]]
    if "aggregate_tilelang_pipeline" in evidence:
        return [manifests["pipeline"]["run"]]
    if "aggregate_register_pressure" in evidence:
        return [manifests["p4_1"]["register_pressure_run"]]
    if "aggregate_tilelang_wave_resources" in evidence:
        return [manifests["p4_1"]["tilelang_wave_resource_run"]]
    if "aggregate_tilelang_exact_wave" in evidence:
        return [manifests["p4_1"]["tilelang_exact_wave_run"]]
    if "aggregate_tilelang_compute_wave" in evidence:
        return [manifests["compute"]["run"], manifests["compute"]["resource_query_run"]]
    if "aggregate_tilelang_pipeline_resources" in evidence:
        return [manifests["p4_1"]["pipeline_resource_run"]]
    return ["unknown"]


def main():
    scope = json.loads(SCOPE.read_text())
    input_paths = [ROOT / path for path in scope["inputs"]]
    for path in input_paths:
        if not path.exists():
            raise RuntimeError(f"declared P5 input is missing: {path}")
    source_manifest = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in input_paths]
    source_manifest.extend([
        {"path": str(SCOPE.relative_to(ROOT)), "sha256": sha256(SCOPE), "role": "model_scope"},
        {"path": str(pathlib.Path(__file__).relative_to(ROOT)), "sha256": sha256(pathlib.Path(__file__)), "role": "builder"},
    ])

    p2 = json.loads((ROOT / "results/processed/p2_compute/aggregate.json").read_text())
    p2_manifest = json.loads((ROOT / "results/processed/p2_compute/formal_run_manifest.json").read_text())
    p3_params = json.loads((ROOT / "results/processed/p3_memory/s5000_memory_parameters_provisional.json").read_text())
    p4_params = json.loads((ROOT / "results/processed/p4_occupancy_wave/s5000_occupancy_wave_parameters_provisional.json").read_text())
    p4_pipeline = rows(ROOT / "results/processed/p4_occupancy_wave/aggregate_tilelang_pipeline.csv")
    p4_compute = json.loads((ROOT / "results/processed/p4_occupancy_wave/aggregate_tilelang_compute_wave.json").read_text())
    manifests = {
        "run": json.loads((ROOT / "results/processed/p4_occupancy_wave/run_manifest.json").read_text()),
        "pipeline": json.loads((ROOT / "results/processed/p4_occupancy_wave/aggregate_tilelang_pipeline.json").read_text()),
        "p4_1": json.loads((ROOT / "results/processed/p4_occupancy_wave/p4_1_manifest.json").read_text()),
        "compute": p4_compute,
    }

    p3_artifacts = [
        ("global", "aggregate_global_initial.csv", "aggregate_global_initial.json"),
        ("fixed_grid_cache", "aggregate_fixed_grid_cache.csv", "aggregate_fixed_grid_cache.json"),
        ("read", "aggregate_read_initial.csv", "aggregate_read_initial.json"),
        ("shared", "aggregate_shared_initial.csv", "aggregate_shared_initial.json"),
        ("pointer_chase", "aggregate_pointer_chase.csv", "aggregate_pointer_chase.json"),
    ]
    p3_tables = {}
    for family, csv_name, json_name in p3_artifacts:
        csv_path = ROOT / "results/processed/p3_memory" / csv_name
        json_path = ROOT / "results/processed/p3_memory" / json_name
        p3_tables[family] = {"csv": csv_name, "rows": rows(csv_path), "runs": trace_runs_from_p3(json_path)}

    registry = []
    def add_parameter(parameter_id, phase, family, value, units, status, applicability, evidence, raw_runs):
        registry.append({
            "parameter_id": parameter_id, "phase": phase, "family": family, "value": value,
            "units": units, "status": status, "applicability": applicability, "evidence": evidence,
            "raw_runs_json": compact(raw_runs),
        })

    # P3 summary parameters remain, but all P3 conditional curves are also
    # materialized below so a caller can select only an exact condition.
    for parameter in p3_params["parameters"]:
        evidence = parameter["evidence"]
        matching = [table["runs"] for table in p3_tables.values() if table["csv"] in evidence]
        trace = matching[0] if matching else ["logs/run/p3_musa_device_properties_stdout.log"]
        add_parameter(f"p3:summary:{parameter['parameter']}", "P3", "summary", parameter["value"],
                      parameter["units"], parameter["status"], parameter["applicability"], evidence,
                      trace)

    for family, table in p3_tables.items():
        for row in table["rows"]:
            if family == "pointer_chase":
                add_parameter(f"p3:pointer_chase:{row['workset_bytes']}", "P3", "pointer_chase_latency",
                              row["ns_per_dependent_load_cross_run_median"], "ns/dependent-load", "conditional",
                              compact({"workset_bytes": int(row["workset_bytes"])}),
                              f"{table['csv']}:workset={row['workset_bytes']}", table["runs"])
                continue
            case = json.loads(row["case_config_json"])
            if not as_bool(row["parameter_eligible"]):
                status = "not_parameter_eligible"
            else:
                status = "conditional_calibration_eligible"
            rate = row["logical_effective_GBps_cross_run_median"]
            if family == "shared" and row.get("incremental_shared_read_effective_GBps"):
                rate, units = row["incremental_shared_read_effective_GBps"], "GB/s_incremental_shared"
            else:
                units = "GB/s_logical"
            add_parameter(f"p3:{family}:{row['case_id']}", "P3", f"{family}_condition_rate", rate, units,
                          status, compact(case), f"{table['csv']}:{row['case_id']}; source_sha256={row.get('source_sha256', '')}",
                          table["runs"])

    p2_runs = p2["runs"]
    p2_gemm, p2_non_gemm = [], []
    for row in p2["cases"]:
        if not as_bool(row["p2_parameter_eligible"]):
            continue
        case = json.loads(row["case_config_json"])
        latency = float(row["graph_latency_us_cross_run_median"])
        if row["op"] == "gemm":
            rate = 2 * case["M"] * case["N"] * case["K"] / latency / 1e6
            item = {"row": row, "case": case, "latency": latency, "rate": rate}
            p2_gemm.append(item)
            add_parameter(f"p2:gemm:{row['case_id']}", "P2", "gemm_effective_rate", rate, "TFLOP/s",
                          "calibration_eligible", compact(gemm_condition(case)),
                          f"aggregate.json:{row['case_id']}; source_sha256={row['source_sha256']}", p2_runs)
        else:
            work_items = case.get("N", case.get("M", 0) * case.get("K", 0))
            # latency is microseconds: items/us == Gitems/s / 1000.
            rate = work_items / latency / 1e3
            item = {"row": row, "case": case, "latency": latency, "work_items": work_items, "rate": rate}
            p2_non_gemm.append(item)
            add_parameter(f"p2:{row['op']}:{row['case_id']}", "P2", f"{row['op']}_effective_rate", rate,
                          "Gitems/s", "calibration_eligible", compact(case),
                          f"aggregate.json:{row['case_id']}; source_sha256={row['source_sha256']}", p2_runs)

    for parameter in p4_params["parameters"]:
        add_parameter(f"p4:{parameter['parameter']}", "P4", "resource_or_wave", parameter["value"],
                      parameter["units"], parameter["status"], parameter["applicability"], parameter["evidence"],
                      p4_trace(parameter, manifests))

    pipeline_by_source = {row["source_sha256"]: row for row in p4_pipeline if as_bool(row["parameter_eligible"])}
    p4_by_name = {row["parameter"]: row for row in p4_params["parameters"]}
    predictions = []

    def add_prediction(**record):
        record.setdefault("prologue_us", "unknown")
        record.setdefault("tail_wave_us", "unknown")
        record.setdefault("epilogue_us", "unknown")
        record.setdefault("full_waves", "unknown")
        record.setdefault("tail_cta", "unknown")
        predictions.append(record)

    for item in p2_gemm:
        case, row, rate = item["case"], item["row"], item["rate"]
        predicted = 2 * case["M"] * case["N"] * case["K"] / rate / 1e6
        resource = {"resident_cta_per_mp": "unknown", "grid_cta": "unknown", "capacity_cta": "unknown",
                    "constraint_source": "unknown"}
        full_waves, tail_cta = "unknown", "unknown"
        pipeline = pipeline_by_source.get(row["source_sha256"])
        if pipeline:
            stage = int(pipeline["num_stages"])
            active = int(p4_by_name[f"tilelang_fp16_gemm_1024_stage{stage}_active_cta_per_mp"]["value"])
            grid = ((case["M"] + case["block_M"] - 1) // case["block_M"]) * ((case["N"] + case["block_N"] - 1) // case["block_N"])
            capacity = active * 60
            full_waves, tail_cta = grid // capacity, grid % capacity
            resource = {"resident_cta_per_mp": active, "grid_cta": grid, "capacity_cta": capacity,
                        "constraint_source": "P4 Driver API recompiled-MUBIN query; exact source SHA match"}
        add_prediction(case_id=row["case_id"], family="gemm", prediction_mode="exact_condition_lookup",
                       validation_eligible=False, actual_latency_us=item["latency"], predicted_latency_us=predicted,
                       signed_error_us=predicted-item["latency"], absolute_percentage_error=abs(predicted/item["latency"]-1)*100,
                       condition_json=compact(gemm_condition(case)), effective_rate=rate, effective_rate_units="TFLOP/s",
                       steady_effective_us=predicted, full_waves=full_waves, tail_cta=tail_cta,
                       tail_policy="unknown_for_gemm_family", resource_decomposition_json=compact(resource),
                       source_sha256=row["source_sha256"], raw_runs_json=compact(p2_runs))

    for item in p2_non_gemm:
        row, case, rate = item["row"], item["case"], item["rate"]
        predicted = item["work_items"] / rate / 1e3
        add_prediction(case_id=row["case_id"], family=row["op"], prediction_mode="exact_condition_lookup",
                       validation_eligible=False, actual_latency_us=item["latency"], predicted_latency_us=predicted,
                       signed_error_us=predicted-item["latency"], absolute_percentage_error=abs(predicted/item["latency"]-1)*100,
                       condition_json=compact(case), effective_rate=rate, effective_rate_units="Gitems/s",
                       steady_effective_us=predicted, tail_policy="unknown_for_non_gemm_family",
                       resource_decomposition_json=compact({"work_items": item["work_items"], "resource": "unknown"}),
                       source_sha256=row["source_sha256"], raw_runs_json=compact(p2_runs))

    # Exact P3 condition lookup: the complete config is the selector.  There
    # is no operation-only cache-rate transfer in this corrected model.
    for family, table in p3_tables.items():
        if family == "pointer_chase":
            continue
        for row in table["rows"]:
            if not as_bool(row["parameter_eligible"]):
                continue
            rate = float(row["logical_effective_GBps_cross_run_median"])
            if rate <= 0:
                continue
            actual = float(row["latency_us_cross_run_median"])
            predicted = float(row["logical_payload_bytes"]) / rate / 1e3
            case = json.loads(row["case_config_json"])
            add_prediction(case_id=row["case_id"], family=f"p3_{family}", prediction_mode="exact_condition_lookup",
                           validation_eligible=False, actual_latency_us=actual, predicted_latency_us=predicted,
                           signed_error_us=predicted-actual, absolute_percentage_error=abs(predicted/actual-1)*100,
                           condition_json=compact(case), effective_rate=rate, effective_rate_units="GB/s_logical",
                           steady_effective_us=predicted, tail_policy="unknown_or_not_parameterized",
                           resource_decomposition_json=compact({"logical_payload_bytes": int(row["logical_payload_bytes"]),
                                                                "cache_condition": case.get("cache_condition", "unknown"),
                                                                "layout": case.get("op"), "selector": "full_case_config"}),
                           source_sha256=row.get("source_sha256", ""), raw_runs_json=compact(table["runs"]))

    # This is the one measured family for which P4 provides a tail interval.
    for observation in p4_compute["tail_observations"]:
        add_prediction(case_id=f"p4_compute_wave_tail_{observation['boundary_cta']}", family="p4_compute_wave",
                       prediction_mode="conditional_tail_interval", validation_eligible=False,
                       actual_latency_us=observation["extra_us"], predicted_latency_us="interval",
                       signed_error_us="not_applicable", absolute_percentage_error="not_applicable",
                       condition_json=compact({"boundary_cta": observation["boundary_cta"], "threads": 256,
                                                "iterations": 2048, "active_cta_per_mp": 12}),
                       effective_rate="unknown", effective_rate_units="unknown", steady_effective_us="unknown",
                       tail_wave_us=compact({"min": p4_compute["conditional_tail_range_us"][0],
                                             "max": p4_compute["conditional_tail_range_us"][1],
                                             "median": p4_compute["conditional_tail_median_us"]}),
                       tail_policy="P4.2 exact kernel-family interval", full_waves="condition-dependent",
                       tail_cta=1, resource_decomposition_json=compact(p4_compute["resource_contract"]),
                       source_sha256="", raw_runs_json=compact([p4_compute["run"], p4_compute["resource_query_run"]]))

    numeric_errors = [float(row["absolute_percentage_error"]) for row in predictions
                      if isinstance(row["absolute_percentage_error"], (int, float))]
    model = {
        "model_version": scope["model_version"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder_python": platform.python_version(), "builder_platform": platform.platform(),
        "scope": "P5 calibration/development condition lookup; not a cross-condition predictor and not hold-out validated",
        "source_manifest": source_manifest,
        "equations": {
            "gemm": "effective_rate_us = 2*M*N*K / exact_condition_TFLOP_per_s / 1e6",
            "non_gemm": "effective_rate_us = work_items / exact_condition_Gitems_per_s / 1e3",
            "memory": "effective_rate_us = logical_payload_bytes / exact_condition_logical_GB_per_s / 1e3",
            "total": "prologue + steady + tail + epilogue; unavailable terms are unknown, never a fitted residual",
        },
        "tail_policy": {"compute_wave_only_range_us": p4_compute["conditional_tail_range_us"],
                          "compute_wave_only_median_us": p4_compute["conditional_tail_median_us"],
                          "copy_and_gemm": "unknown / no transfer"},
        "development_prediction_summary": {
            "exact_condition_records": len(predictions),
            "lookup_reconstruction_mape_percent": statistics.median(numeric_errors) if numeric_errors else "unknown",
            "lookup_reconstruction_is_validation": False,
            "cross_condition_rate_transfer_records": 0,
            "cross_condition_rate_transfer_status": "not implemented; requires P6 hold-out evidence",
        },
        "known_unknowns": ["physical cache tier selection", "scheduler-to-MP mapping", "universal tail-wave constant",
                           "GEMM prologue/epilogue split", "resource occupancy for unseen TileLang source"],
    }
    fieldnames = list(registry[0])
    with (OUT / "s5000_parameter_registry_v1.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(registry)
    (OUT / "s5000_parameter_registry_v1.json").write_text(json.dumps({"model_version": scope["model_version"], "source_manifest": source_manifest, "parameters": registry}, indent=2) + "\n")
    with (OUT / "calibration_development_predictions_v1.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    # Keep the former public filename, now with a formula-consistent exact condition lookup.
    with (OUT / "calibration_exact_lookup_predictions_v1.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (OUT / "s5000_single_gpu_model_v1.json").write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps(model["development_prediction_summary"], indent=2))


if __name__ == "__main__":
    main()
