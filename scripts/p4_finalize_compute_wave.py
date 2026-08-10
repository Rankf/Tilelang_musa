#!/usr/bin/env python3
"""Aggregate P4.2 compute-dominant TileLang wave evidence."""

import csv
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/p4_tilelang_compute_wave.json"
OUT = ROOT / "results/processed/p4_occupancy_wave"


def median(values):
    return statistics.median(values)


def main():
    cfg = json.loads(CFG.read_text())
    sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    candidates = []
    for run in (ROOT / "results/raw/p4_tilelang_compute_wave").glob("*"):
        processes = sorted(run.glob("process_*"))
        matching = []
        for process in processes:
            summary = process / "summary.csv"
            metadata = process / "metadata.json"
            if summary.exists() and metadata.exists() and json.loads(metadata.read_text())["config_sha256"] == sha:
                matching.append(process)
        if len(matching) >= 3:
            candidates.append((run, matching))
    if not candidates:
        raise RuntimeError("no P4.2 run with at least three config-matching processes")
    run, processes = sorted(candidates, key=lambda item: item[0])[-1]
    accepted, rejected = [], []
    for process in processes:
        rows = list(csv.DictReader((process / "summary.csv").open()))
        failures = [r["case_id"] for r in rows
                    if r["status"] != "measured" or r["correctness"] != "pass"]
        if failures:
            rejected.append({"process": process.name, "failed_cases": failures})
        else:
            accepted.append((process, {r["case_id"]: r for r in rows}))
    if len(accepted) < 3:
        raise RuntimeError(f"fewer than three fully passing processes: {rejected}")
    selected = accepted[:3]
    rows = []
    for case in cfg["cases"]:
        group = [table[case["id"]] for _, table in selected]
        times = [float(r["warm_latency_us_median"]) for r in group]
        prof = [float(r["profiler_self_device_us"]) for r in group]
        rows.append({
            "case_id": case["id"], "schedule_role": case["schedule_role"],
            "grid_blocks": case["grid_blocks"], "threads": case["threads"],
            "iterations": case["iterations"],
            "latency_us_run1": times[0], "latency_us_run2": times[1],
            "latency_us_run3": times[2], "latency_us_cross_run_median": median(times),
            "cross_run_relative_span": (max(times) - min(times)) / median(times),
            "profiler_self_device_us_cross_run_median": median(prof),
            "source_sha256": group[0]["source_sha256"],
            "correctness_all_pass": "true", "parameter_eligible": "true",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "aggregate_tilelang_compute_wave.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_grid = {int(r["grid_blocks"]): r for r in rows}
    tail = []
    for boundary in (720, 1440, 2160):
        base = float(by_grid[boundary]["latency_us_cross_run_median"])
        after = float(by_grid[boundary + 1]["latency_us_cross_run_median"])
        tail.append({"boundary_cta": boundary, "extra_us": after - base,
                     "extra_percent": 100 * (after / base - 1)})
    extras_us = [r["extra_us"] for r in tail]
    extras_pct = [r["extra_percent"] for r in tail]
    resource_runs = [p for p in (ROOT / "results/raw/p4_tilelang_compute_wave_resource").glob("*")
                     if (p / "raw.csv").exists()]
    if not resource_runs:
        raise RuntimeError("missing P4.2 resource probe")
    resource_run = sorted(resource_runs)[-1]
    resource_rows = list(csv.DictReader((resource_run / "raw.csv").open()))
    fields = ("source_sha256", "dynamic_shared_bytes", "max_threads_per_block",
              "static_shared_bytes", "local_bytes_per_thread", "registers_per_thread",
              "active_cta_per_mp", "status")
    if len(resource_rows) != 3 or any(len({r[k] for r in resource_rows}) != 1 for k in fields):
        raise RuntimeError("P4.2 resource-query repetitions disagree")
    resource = resource_rows[0]
    if resource["status"] != "pass" or resource["source_sha256"] != rows[0]["source_sha256"]:
        raise RuntimeError("P4.2 resource source/status does not match timed kernel")
    result = {
        "run": str(run), "config_sha256": sha,
        "selected_processes": [p.name for p, _ in selected],
        "rejected_processes": rejected,
        "resource_query_run": str(resource_run),
        "resource_contract": {"threads": 256,
                              "registers_per_thread": int(resource["registers_per_thread"]),
                              "dynamic_shared_bytes": int(resource["dynamic_shared_bytes"]),
                              "static_shared_bytes": int(resource["static_shared_bytes"]),
                              "local_bytes_per_thread": int(resource["local_bytes_per_thread"]),
                              "active_cta_per_mp": int(resource["active_cta_per_mp"]),
                              "multiprocessors": 60, "capacity_cta": 720,
                              "mubin_provenance": "recompiled from retained formal source and command"},
        "tail_observations": tail,
        "conditional_tail_range_us": [min(extras_us), max(extras_us)],
        "conditional_tail_median_us": median(extras_us),
        "conditional_tail_range_percent": [min(extras_pct), max(extras_pct)],
        "conditional_tail_median_percent": median(extras_pct),
        "fit_status": "eligible_as_kernel-specific_range_not_universal_fixed_percent",
    }
    (OUT / "aggregate_tilelang_compute_wave.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
