#!/usr/bin/env python3
"""Aggregate the three independent P3 memory/cache measurement processes."""

import csv
import glob
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/p3_global_memory"
OUT = ROOT / "results/processed/p3_memory"
OUT.mkdir(parents=True, exist_ok=True)


def median(values):
    return statistics.median(values)


def select_runs(config_path, require_correct=True):
    cfg = json.loads((ROOT / config_path).read_text())
    expected = len(cfg["cases"])
    cfg_hash = __import__("hashlib").sha256((ROOT / config_path).read_bytes()).hexdigest()
    candidates = []
    for text in sorted(glob.glob(str(RAW / "*"))):
        path = pathlib.Path(text)
        metadata_path, summary_path = path / "metadata.json", path / "summary.csv"
        if not metadata_path.exists() or not summary_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        rows = list(csv.DictReader(summary_path.open()))
        if metadata.get("config_sha256") != cfg_hash or len(rows) != expected:
            continue
        if require_correct and not all(row.get("correctness") == "pass" for row in rows):
            continue
        candidates.append((path, rows))
    if len(candidates) < 3:
        raise RuntimeError(f"need three complete runs for {config_path}, found {len(candidates)}")
    return cfg, candidates[-3:]


def aggregate_config(name, config_path):
    cfg, runs = select_runs(config_path)
    tables = [{row["case_id"]: row for row in rows} for _, rows in runs]
    output = []
    for case in cfg["cases"]:
        cid = case["id"]
        rows = [table[cid] for table in tables]
        values = [float(row["warm_latency_us_median"]) for row in rows]
        prof = [float(row["profiler_self_device_us"]) for row in rows]
        cross_span = (max(values) - min(values)) / median(values)
        eligible = all(row["status"] == "measured" for row in rows) and cross_span <= .15
        cold_values = [float(row["cold_post_flush_latency_us_median"]) for row in rows if row["cold_post_flush_latency_us_median"]]
        record = {
            "case_id": cid, "op": case["op"], "case_config_json": json.dumps(case, sort_keys=True, separators=(",", ":")),
            "run_statuses": "|".join(row["status"] for row in rows),
            "correctness_all_pass": str(all(row["correctness"] == "pass" for row in rows)).lower(),
            "latency_us_run1": values[0], "latency_us_run2": values[1], "latency_us_run3": values[2],
            "latency_us_cross_run_median": median(values), "cross_run_relative_span": cross_span,
            "profiler_self_device_us_cross_run_median": median(prof),
            "logical_payload_bytes": rows[0]["logical_payload_bytes"],
            "logical_effective_GBps_cross_run_median": median(float(row["logical_effective_GBps"] or 0) for row in rows),
            "cold_post_flush_latency_us_cross_run_median": median(cold_values) if cold_values else "",
            "parameter_eligible": str(eligible).lower(),
            "source_sha256": rows[0].get("source_sha256", ""),
        }
        output.append(record)
    # Shared access is reported only after subtraction of the same-run r0 baseline.
    if name == "shared_initial":
        bases = [float(table["shared_base_r0"]["warm_latency_us_median"]) for table in tables]
        for record in output:
            if record["case_id"] == "shared_base_r0":
                continue
            bytes_ = float(record["logical_payload_bytes"])
            deltas = [float(tables[i][record["case_id"]]["warm_latency_us_median"]) - bases[i] for i in range(3)]
            record["incremental_latency_us_cross_run_median"] = median(deltas)
            record["incremental_shared_read_effective_GBps"] = bytes_ / median(deltas) / 1e3 if min(deltas) > 0 else ""
    fields = sorted({key for row in output for key in row})
    with (OUT / f"aggregate_{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    (OUT / f"aggregate_{name}.json").write_text(json.dumps({"config": config_path, "runs": [str(path) for path, _ in runs], "cases": output}, indent=2) + "\n")
    return output, [str(path) for path, _ in runs]


def aggregate_pointer_chase():
    runs = sorted(pathlib.Path(path) for path in glob.glob(str(ROOT / "results/raw/p3_pointer_chase/*")) if (pathlib.Path(path) / "raw_samples.csv").exists())[-3:]
    tables = []
    for run in runs:
        grouped = {}
        for row in csv.DictReader((run / "raw_samples.csv").open()):
            grouped.setdefault(row["workset_bytes"], []).append(float(row["ns_per_dependent_load"]))
        tables.append({key: median(value) for key, value in grouped.items()})
    rows = []
    for workset in tables[0]:
        values = [table[workset] for table in tables]
        rows.append({"workset_bytes": workset, "ns_per_dependent_load_run1": values[0], "ns_per_dependent_load_run2": values[1], "ns_per_dependent_load_run3": values[2], "ns_per_dependent_load_cross_run_median": median(values), "cross_run_relative_span": (max(values) - min(values)) / median(values)})
    with (OUT / "aggregate_pointer_chase.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "aggregate_pointer_chase.json").write_text(json.dumps({"runs": [str(path) for path in runs], "kernel": "benchmarks/p3_musa_pointer_chase.mu", "cases": rows}, indent=2) + "\n")
    return rows, [str(path) for path in runs]


def main():
    groups = {}
    groups["global_initial"] = aggregate_config("global_initial", "configs/p3_memory_global_initial.json")
    groups["fixed_grid_cache"] = aggregate_config("fixed_grid_cache", "configs/p3_memory_fixed_grid_cache.json")
    groups["shared_initial"] = aggregate_config("shared_initial", "configs/p3_shared_memory_initial.json")
    groups["read_initial"] = aggregate_config("read_initial", "configs/p3_memory_read_initial.json")
    groups["pointer_chase"] = aggregate_pointer_chase()
    global_rows = {row["case_id"]: row for row in groups["global_initial"][0]}
    read_rows = {row["case_id"]: row for row in groups["read_initial"][0]}
    shared_rows = {row["case_id"]: row for row in groups["shared_initial"][0]}
    parameters = [
        {"parameter": "global_copy_mixed_logical_GBps", "value": global_rows["copy_ws_512m"]["logical_effective_GBps_cross_run_median"], "units": "GB/s", "status": "provisional", "applicability": "FP32 TileLang vector copy, 512 MiB payload, 256 threads, preallocated, graph replay", "evidence": "aggregate_global_initial.csv:copy_ws_512m"},
        {"parameter": "global_fill_write_logical_GBps", "value": global_rows["fill_256m"]["logical_effective_GBps_cross_run_median"], "units": "GB/s", "status": "provisional", "applicability": "FP32 TileLang fill, 256 MiB output, 256 threads", "evidence": "aggregate_global_initial.csv:fill_256m"},
        {"parameter": "global_triad_mixed_logical_GBps", "value": global_rows["triad_256m"]["logical_effective_GBps_cross_run_median"], "units": "GB/s", "status": "provisional", "applicability": "FP32 TileLang A+B->C, 256 MiB per operand, 256 threads", "evidence": "aggregate_global_initial.csv:triad_256m"},
        {"parameter": "global_read_sum_logical_GBps", "value": read_rows["read_sum_512m"]["logical_effective_GBps_cross_run_median"], "units": "GB/s", "status": "provisional_lower_bound", "applicability": "FP32 TileLang read-dominated sum, 512 MiB input, 64 loads per output", "evidence": "aggregate_read_initial.csv:read_sum_512m"},
        {"parameter": "shared_repeated_read_effective_GBps", "value": shared_rows["shared_seq_r128"]["incremental_shared_read_effective_GBps"], "units": "GB/s", "status": "conditional_synthetic", "applicability": "FP32 16-element shared tile, 4096 CTA, 128 threads, 128 repeated reads; r0 baseline subtracted", "evidence": "aggregate_shared_initial.csv:shared_seq_r128"},
        {"parameter": "candidate_effective_cache_workset_boundary", "value": "unknown", "units": "bytes", "status": "unknown", "applicability": "fixed-grid 8-512 MiB copy plus 4 KiB-64 MiB native pointer chase", "evidence": "aggregate_fixed_grid_cache.csv and aggregate_pointer_chase.csv; no repeatable isolated cliff"},
    ]
    with (OUT / "s5000_memory_parameters_provisional.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(parameters[0])); writer.writeheader(); writer.writerows(parameters)
    (OUT / "s5000_memory_parameters_provisional.json").write_text(json.dumps({"scope": "P3 provisional, not a physical-memory specification", "parameters": parameters}, indent=2) + "\n")
    print(json.dumps({key: {"runs": value[1], "cases": len(value[0])} for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
