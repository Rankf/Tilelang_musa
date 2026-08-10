#!/usr/bin/env python3
"""Aggregate the P4.1 exact TileLang resource, register, and wave evidence."""

import csv
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p4_occupancy_wave"
OUT.mkdir(parents=True, exist_ok=True)


def median(values):
    return statistics.median(values)


def relative_span(values):
    mid = median(values)
    return (max(values) - min(values)) / mid if mid else float("inf")


def newest_complete(base, required):
    candidates = [p for p in base.glob("*") if all((p / name).exists() for name in required)]
    if not candidates:
        raise RuntimeError(f"no complete run under {base}")
    return sorted(candidates)[-1]


def write_csv(name, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty aggregate {name}")
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_pipeline_resources():
    run = newest_complete(ROOT / "results/raw/p4_tilelang_resource", ["raw.csv"])
    with (run / "raw.csv").open() as f:
        source = list(csv.DictReader(f))
    rows = []
    for stage in (1, 2, 3, 4):
        group = [r for r in source if int(r["stage"]) == stage]
        if len(group) != 3 or any(r["status"] != "pass" for r in group):
            raise RuntimeError(f"pipeline stage {stage} lacks three passing resource queries")
        invariant = ["dynamic_shared_bytes", "max_threads_per_block", "static_shared_bytes",
                     "local_bytes_per_thread", "registers_per_thread", "active_cta_per_mp"]
        if any(len({r[key] for r in group}) != 1 for key in invariant):
            raise RuntimeError(f"pipeline stage {stage} resource queries disagree")
        first = group[0]
        rows.append({
            "stage": stage,
            "query_repetitions": 3,
            "dynamic_shared_bytes": int(first["dynamic_shared_bytes"]),
            "registers_per_thread": int(first["registers_per_thread"]),
            "static_shared_bytes": int(first["static_shared_bytes"]),
            "local_bytes_per_thread": int(first["local_bytes_per_thread"]),
            "max_threads_per_block": int(first["max_threads_per_block"]),
            "active_cta_per_mp_256_threads": int(first["active_cta_per_mp"]),
            "status": "pass_recompiled_mubin",
        })
    write_csv("aggregate_tilelang_pipeline_resources.csv", rows)
    return run, rows


def aggregate_register_pressure():
    run = newest_complete(ROOT / "results/raw/p4_register_pressure",
                          [f"run_{i}.csv" for i in (1, 2, 3)])
    tables = []
    for index in (1, 2, 3):
        with (run / f"run_{index}.csv").open() as f:
            tables.append(list(csv.DictReader(f)))
    rows = []
    keys = [(int(r["live_values"]), int(r["block_threads"])) for r in tables[0]]
    for live_values, threads in keys:
        group = [next(r for r in table if int(r["live_values"]) == live_values and
                      int(r["block_threads"]) == threads) for table in tables]
        invariant = ["func_num_regs", "func_local_bytes", "func_static_shared_bytes",
                     "active_cta_per_mp", "correctness"]
        if any(len({r[key] for r in group}) != 1 for key in invariant):
            raise RuntimeError(f"register-pressure repetitions disagree: {live_values}/{threads}")
        first = group[0]
        if first["correctness"] != "pass":
            raise RuntimeError(f"register-pressure correctness failed: {live_values}/{threads}")
        rows.append({
            "live_values": live_values,
            "block_threads": threads,
            "registers_per_thread": int(first["func_num_regs"]),
            "local_bytes_per_thread": int(first["func_local_bytes"]),
            "static_shared_bytes": int(first["func_static_shared_bytes"]),
            "active_cta_per_mp": int(first["active_cta_per_mp"]),
            "query_repetitions": 3,
            "correctness_all_pass": "true",
        })
    write_csv("aggregate_register_pressure.csv", rows)
    return run, rows


def aggregate_fixed_grid_resources():
    run = newest_complete(ROOT / "results/raw/p4_tilelang_wave_resource", ["raw.csv"])
    with (run / "raw.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["case_id"].startswith("tl_block_") or r["case_id"] == "tl_wave_1"]
    if not rows or any(r["status"] != "pass" for r in rows):
        raise RuntimeError("fixed-grid resource query contains failure")
    write_csv("aggregate_tilelang_wave_resources.csv", rows)
    return run, rows


def aggregate_exact_wave():
    cfg_path = ROOT / "configs/p4_tilelang_exact_wave.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    candidates = []
    for run in (ROOT / "results/raw/p4_tilelang_exact_wave").glob("*"):
        processes = [run / f"process_{i}" for i in (1, 2, 3)]
        if not all((p / "summary.csv").exists() and (p / "metadata.json").exists()
                   for p in processes):
            continue
        if all(json.loads((p / "metadata.json").read_text())["config_sha256"] == cfg_sha
               for p in processes):
            candidates.append(run)
    if not candidates:
        raise RuntimeError("no complete exact-wave run matching the current config")
    run = sorted(candidates)[-1]
    tables = []
    for index in (1, 2, 3):
        with (run / f"process_{index}/summary.csv").open() as f:
            tables.append({r["case_id"]: r for r in csv.DictReader(f)})
    rows = []
    for case in cfg["cases"]:
        group = [table[case["id"]] for table in tables]
        times = [float(r["warm_latency_us_median"]) for r in group]
        prof = [float(r["profiler_self_device_us"]) for r in group]
        eligible = (all(r["status"] == "measured" and r["correctness"] == "pass" for r in group)
                    and relative_span(times) <= 0.20)
        rows.append({
            "case_id": case["id"],
            "schedule_role": case["schedule_role"],
            "grid_blocks": case["grid_blocks"],
            "threads": case["threads"],
            "latency_us_run1": times[0],
            "latency_us_run2": times[1],
            "latency_us_run3": times[2],
            "latency_us_cross_run_median": median(times),
            "cross_run_relative_span": relative_span(times),
            "profiler_self_device_us_cross_run_median": median(prof),
            "correctness_all_pass": str(all(r["correctness"] == "pass" for r in group)).lower(),
            "parameter_eligible": str(eligible).lower(),
        })
    write_csv("aggregate_tilelang_exact_wave.csv", rows)
    by_grid = {int(r["grid_blocks"]): r for r in rows}
    deltas = {
        "tail_480_to_481_percent": 100 * (by_grid[481]["latency_us_cross_run_median"] /
                                            by_grid[480]["latency_us_cross_run_median"] - 1),
        "tail_960_to_961_percent": 100 * (by_grid[961]["latency_us_cross_run_median"] /
                                            by_grid[960]["latency_us_cross_run_median"] - 1),
    }
    return run, rows, deltas, cfg_sha


def main():
    pipeline_run, pipeline = aggregate_pipeline_resources()
    register_run, register = aggregate_register_pressure()
    wave_resource_run, wave_resource = aggregate_fixed_grid_resources()
    exact_wave_run, exact_wave, deltas, cfg_sha = aggregate_exact_wave()
    manifest = {
        "pipeline_resource_run": str(pipeline_run),
        "register_pressure_run": str(register_run),
        "tilelang_wave_resource_run": str(wave_resource_run),
        "tilelang_exact_wave_run": str(exact_wave_run),
        "tilelang_exact_wave_config_sha256": cfg_sha,
        "formal_mubin_retained": False,
        "resource_query_binary_provenance": "recompiled from retained formal .mu and exact mcc command",
        "exact_wave_tail_percent": deltas,
    }
    (OUT / "p4_1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "pipeline_resource_cases": len(pipeline),
        "register_pressure_cases": len(register),
        "fixed_grid_resource_cases": len(wave_resource),
        "exact_wave_cases": len(exact_wave),
        **deltas,
    }, indent=2))


if __name__ == "__main__":
    main()
