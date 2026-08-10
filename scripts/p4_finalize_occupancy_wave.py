#!/usr/bin/env python3
"""Aggregate the P4 native-MUSA and TileLang-MUSA occupancy/wave evidence."""

import csv
import hashlib
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p4_occupancy_wave"
OUT.mkdir(parents=True, exist_ok=True)


def med(values):
    return statistics.median(values)


def span(values):
    return (max(values) - min(values)) / med(values) if med(values) else float("inf")


def newest_native_run():
    candidates = []
    for path in (ROOT / "results/raw/p4_occupancy_wave").glob("*"):
        if all((path / f"run_{i}.csv").exists() for i in (1, 2, 3)):
            candidates.append(path)
    if not candidates:
        raise RuntimeError("no complete three-process P4 native run")
    return sorted(candidates)[-1]


def newest_tilelang_run(config_sha):
    candidates = []
    for path in (ROOT / "results/raw/p4_tilelang_wave").glob("*"):
        processes = [path / f"process_{i}" for i in (1, 2, 3)]
        if not all((p / "summary.csv").exists() and (p / "metadata.json").exists() for p in processes):
            continue
        if all(json.loads((p / "metadata.json").read_text())["config_sha256"] == config_sha for p in processes):
            candidates.append(path)
    if not candidates:
        raise RuntimeError("no complete three-process P4 TileLang run with the current config")
    return sorted(candidates)[-1]


def native_aggregate(run_dir):
    tables = []
    for index in (1, 2, 3):
        with (run_dir / f"run_{index}.csv").open() as f:
            tables.append(list(csv.DictReader(f)))
    grouped = {}
    for table in tables:
        for row in table:
            if row["kind"] == "device_properties":
                key = ("device_properties", "device0", "0", "0")
            else:
                key = (row["kind"], row["kernel_id"], row["block_threads"], row["grid_blocks"])
            grouped.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise RuntimeError(f"native group not present in all processes: {key}")
        first = rows[0]
        record = {
            "kind": first["kind"], "kernel_id": first["kernel_id"],
            "requested_static_shared_bytes": first["requested_static_shared_bytes"],
            "block_threads": first["block_threads"], "grid_blocks": first["grid_blocks"],
            "iterations": first["iterations"],
            "api_active_blocks_per_mp": first["api_active_blocks_per_mp"],
            "func_num_regs": first["func_num_regs"],
            "func_static_shared_bytes": first["func_static_shared_bytes"],
            "func_max_threads_per_block": first["func_max_threads_per_block"],
            "correctness_all_pass": str(all(r["correctness"] in ("pass", "not_timed", "not_applicable") for r in rows)).lower(),
            "device_detail": first["error"],
        }
        if first["event_median_us"]:
            values = [float(r["event_median_us"]) for r in rows]
            rates = [float(r["thread_iterations_per_s"]) for r in rows]
            record.update({
                "event_latency_us_cross_run_median": med(values),
                "event_latency_us_cross_run_relative_span": span(values),
                "thread_iterations_per_s_cross_run_median": med(rates),
                "parameter_eligible": str(record["correctness_all_pass"] == "true" and span(values) <= .20).lower(),
            })
        else:
            record["parameter_eligible"] = "true"
        output.append(record)
    path = OUT / "aggregate_native.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in output for k in r})); writer.writeheader(); writer.writerows(output)
    return output


def tilelang_aggregate(run_dir, cfg):
    tables = []
    for index in (1, 2, 3):
        with (run_dir / f"process_{index}/summary.csv").open() as f:
            tables.append({row["case_id"]: row for row in csv.DictReader(f)})
    output = []
    for case in cfg["cases"]:
        # The former native-API reference positions (719/720/721 and
        # 1439/1440/1441) are not TileLang occupancy boundaries.  Keep their
        # raw runs for provenance, but exclude them from the current aggregate.
        if case["schedule_role"].startswith("native_api_reference"):
            continue
        rows = [table[case["id"]] for table in tables]
        values = [float(r["warm_latency_us_median"]) for r in rows]
        bandwidth = [float(r["logical_effective_GBps"] or 0) for r in rows]
        profiler = [float(r["profiler_self_device_us"]) for r in rows]
        eligible = all(r["status"] == "measured" and r["correctness"] == "pass" for r in rows) and span(values) <= .20
        output.append({
            "case_id": case["id"], "schedule_role": case["schedule_role"],
            "threads": case["threads"], "grid_blocks": case["grid_blocks"], "N": case["N"],
            "generated_launch_threads": rows[0]["generated_launch_threads"],
            "run_statuses": "|".join(r["status"] for r in rows),
            "correctness_all_pass": str(all(r["correctness"] == "pass" for r in rows)).lower(),
            "latency_us_run1": values[0], "latency_us_run2": values[1], "latency_us_run3": values[2],
            "latency_us_cross_run_median": med(values),
            "cross_run_relative_span": span(values),
            "profiler_self_device_us_cross_run_median": med(profiler),
            "logical_effective_GBps_cross_run_median": med(bandwidth),
            "source_sha256": rows[0]["source_sha256"],
            "parameter_eligible": str(eligible).lower(),
        })
    path = OUT / "aggregate_tilelang.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(output[0])); writer.writeheader(); writer.writerows(output)
    return output


def main():
    cfg_path = ROOT / "configs/p4_tilelang_wave.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    native_run = newest_native_run()
    tilelang_run = newest_tilelang_run(cfg_sha)
    native = native_aggregate(native_run)
    tilelang = tilelang_aggregate(tilelang_run, cfg)
    native_by = {(r["kind"], r["kernel_id"], r["block_threads"], r["grid_blocks"]): r for r in native}
    details = next(r["device_detail"] for r in native if r["kind"] == "device_properties")
    prop = {key: int(value) for key, value in re.findall(r"(shared_per_mp|regs_per_mp|max_blocks_per_mp|warp)=(\d+)", details)}
    prop["multiprocessors"] = 60
    s0_256 = native_by[("resource", "s0", "256", "0")]
    s32_256 = native_by[("resource", "s32k", "256", "0")]
    native_720 = native_by[("wave_sweep", "s0", "256", "720")]
    native_721 = native_by[("wave_sweep", "s0", "256", "721")]
    tail_pct = 100 * (float(native_721["event_latency_us_cross_run_median"]) / float(native_720["event_latency_us_cross_run_median"]) - 1)
    tl_by = {r["case_id"]: r for r in tilelang}
    parameters = [
        {"parameter": "multi_processor_count_driver_reported", "value": prop["multiprocessors"], "units": "MP", "status": "sdk_driver_reported", "applicability": "MUSA Runtime API native P4 probe", "evidence": "aggregate_native.csv:device_properties"},
        {"parameter": "shared_memory_per_mp_driver_reported", "value": prop["shared_per_mp"], "units": "bytes", "status": "sdk_driver_reported", "applicability": "MUSA Runtime API native P4 probe", "evidence": "aggregate_native.csv:device_properties"},
        {"parameter": "registers_per_mp_driver_reported", "value": prop["regs_per_mp"], "units": "32-bit registers", "status": "sdk_driver_reported", "applicability": "MUSA Runtime API native P4 probe", "evidence": "aggregate_native.csv:device_properties"},
        {"parameter": "max_blocks_per_mp_driver_reported", "value": prop["max_blocks_per_mp"], "units": "CTA", "status": "sdk_driver_reported", "applicability": "architectural device limit, not exact-kernel residency", "evidence": "aggregate_native.csv:device_properties"},
        {"parameter": "native_s0_256_active_cta_per_mp", "value": int(s0_256["api_active_blocks_per_mp"]), "units": "CTA", "status": "api_calculated_exact_kernel", "applicability": "P4 native s0 xorshift kernel, 256 threads, 12 B static shared, 4 regs/thread as reported", "evidence": "aggregate_native.csv:resource/s0/256"},
        {"parameter": "native_s32k_256_active_cta_per_mp", "value": int(s32_256["api_active_blocks_per_mp"]), "units": "CTA", "status": "api_calculated_exact_kernel", "applicability": "P4 native s32k xorshift kernel, 256 threads, 32780 B static shared, 4 regs/thread as reported", "evidence": "aggregate_native.csv:resource/s32k/256"},
        {"parameter": "native_s0_256_tail_720_to_721_percent", "value": tail_pct, "units": "percent", "status": "empirical_conditional", "applicability": "native s0 xorshift kernel, 256 threads, 512 iterations/thread; API capacity=60 MP*12 CTA", "evidence": "aggregate_native.csv:wave_sweep 720 and 721"},
        {"parameter": "tilelang_fixed_grid_copy_256_logical_GBps", "value": float(tl_by["tl_block_256"]["logical_effective_GBps_cross_run_median"]), "units": "GB/s", "status": "provisional_conditional", "applicability": "TileLang-MUSA FP32 fixed-grid copy, 256 threads, 2880 CTA, 94,371,840 elements, graph+profiler gated", "evidence": "aggregate_tilelang.csv:tl_block_256"},
        {"parameter": "tilelang_fixed_grid_copy_256_active_cta_per_mp", "value": 8, "units": "CTA", "status": "api_calculated_recompiled_mubin", "applicability": "TileLang-MUSA fixed-grid copy, 256 threads, 56 regs/thread, zero dynamic/static shared; queried MUBIN was recompiled from retained formal source/command, not the executed formal binary", "evidence": "aggregate_tilelang_wave_resources.csv:tl_block_256 and p4_1_manifest.json"},
        {"parameter": "tilelang_fixed_grid_copy_256_wave_capacity", "value": 480, "units": "CTA", "status": "derived_conditional", "applicability": "60 driver-reported MP * 8 API-calculated CTA/MP for the recompiled fixed-grid copy MUBIN", "evidence": "aggregate_tilelang_wave_resources.csv:tl_block_256"},
    ]
    pipeline_csv = OUT / "aggregate_tilelang_pipeline.csv"
    if pipeline_csv.exists():
        with pipeline_csv.open() as f:
            pipeline = {row["case_id"]: row for row in csv.DictReader(f)}
        stage2 = pipeline["pipeline_fp16_1024_stage2"]
        parameters.append({"parameter": "tilelang_fp16_gemm_1024_stage2_effective_TFLOPs", "value": float(stage2["effective_tops_cross_run_median"]), "units": "TFLOP/s", "status": "provisional_conditional", "applicability": "TileLang-MUSA FP16 GEMM 1024^3, BM=BN=BK=64, 4 warp, 2 pipeline stages, graph+profiler gated", "evidence": "aggregate_tilelang_pipeline.csv:pipeline_fp16_1024_stage2"})
    resource_csv = OUT / "aggregate_tilelang_pipeline_resources.csv"
    if resource_csv.exists():
        with resource_csv.open() as f:
            resources = list(csv.DictReader(f))
        for row in resources:
            stage = row["stage"]
            parameters.append({"parameter": f"tilelang_fp16_gemm_1024_stage{stage}_active_cta_per_mp", "value": int(row["active_cta_per_mp_256_threads"]), "units": "CTA", "status": "api_calculated_recompiled_mubin", "applicability": f"FP16 GEMM 1024^3 stage {stage}, 256 generated threads, 45 regs/thread, {row['dynamic_shared_bytes']} B TIR-declared dynamic shared; queried MUBIN recompiled from retained formal source/command", "evidence": f"aggregate_tilelang_pipeline_resources.csv:stage={stage}"})
    register_csv = OUT / "aggregate_register_pressure.csv"
    if register_csv.exists():
        with register_csv.open() as f:
            pressure = [r for r in csv.DictReader(f) if r["block_threads"] == "256"]
        for row in pressure:
            parameters.append({"parameter": f"native_register_pressure_live{row['live_values']}_256_active_cta_per_mp", "value": int(row["active_cta_per_mp"]), "units": "CTA", "status": "api_calculated_exact_kernel", "applicability": f"controlled native register-pressure kernel, {row['registers_per_thread']} runtime regs/thread, 256 threads, zero local/spill", "evidence": f"aggregate_register_pressure.csv:live_values={row['live_values']},block_threads=256"})
    exact_wave_csv = OUT / "aggregate_tilelang_exact_wave.csv"
    if exact_wave_csv.exists():
        with exact_wave_csv.open() as f:
            exact = {int(r["grid_blocks"]): r for r in csv.DictReader(f)}
        for boundary in (480, 960):
            delta = 100 * (float(exact[boundary + 1]["latency_us_cross_run_median"]) / float(exact[boundary]["latency_us_cross_run_median"]) - 1)
            parameters.append({"parameter": f"tilelang_fixed_grid_copy_tail_{boundary}_to_{boundary + 1}_percent", "value": delta, "units": "percent", "status": "empirical_inconclusive", "applicability": "fixed-grid copy exact-capacity scan; the 480 and 960 boundaries do not show a consistent positive tail step", "evidence": f"aggregate_tilelang_exact_wave.csv:grid={boundary},{boundary + 1}"})
    compute_wave_json = OUT / "aggregate_tilelang_compute_wave.json"
    if compute_wave_json.exists():
        compute_wave = json.loads(compute_wave_json.read_text())
        parameters.extend([
            {"parameter": "tilelang_compute_wave_tail_extra_us_median", "value": compute_wave["conditional_tail_median_us"], "units": "us", "status": "empirical_conditional_range", "applicability": "TileLang dependent FP32 recurrence, 2048 iterations/thread, 256 threads, 3 regs/thread, 12 CTA/MP; median across +1 CTA at 1x/2x/3x 720-CTA capacity", "evidence": "aggregate_tilelang_compute_wave.json:tail_observations"},
            {"parameter": "tilelang_compute_wave_tail_extra_us_min", "value": compute_wave["conditional_tail_range_us"][0], "units": "us", "status": "empirical_conditional_range", "applicability": "same P4.2 compute-wave kernel; lower observed boundary increment", "evidence": "aggregate_tilelang_compute_wave.json:conditional_tail_range_us"},
            {"parameter": "tilelang_compute_wave_tail_extra_us_max", "value": compute_wave["conditional_tail_range_us"][1], "units": "us", "status": "empirical_conditional_range", "applicability": "same P4.2 compute-wave kernel; upper observed boundary increment", "evidence": "aggregate_tilelang_compute_wave.json:conditional_tail_range_us"},
        ])
    for ext in ("csv", "json"):
        path = OUT / f"s5000_occupancy_wave_parameters_provisional.{ext}"
        if ext == "csv":
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(parameters[0])); writer.writeheader(); writer.writerows(parameters)
        else:
            path.write_text(json.dumps({"scope": "P4 conditional calibration, not a universal occupancy specification", "native_run": str(native_run), "tilelang_run": str(tilelang_run), "parameters": parameters}, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({"native_run": str(native_run), "tilelang_run": str(tilelang_run), "tilelang_config_sha256": cfg_sha}, indent=2) + "\n")
    print(json.dumps({"native_run": str(native_run), "tilelang_run": str(tilelang_run), "native_records": len(native), "tilelang_cases": len(tilelang), "native_tail_720_to_721_percent": tail_pct}, indent=2))


if __name__ == "__main__":
    main()
