#!/usr/bin/env python3
"""Cross-check short TileLang GEMM timing with events, profiler and MUSAGraph."""

import argparse
import csv
import json
import pathlib
import statistics
import time
import traceback

import torch
import torch_musa  # noqa: F401
import tilelang

from p2_tilelang_fp16_gemm import PASS_CONFIGS, make_matmul, percentile


def stats(values):
    ordered = sorted(values)
    med = statistics.median(ordered)
    return {
        "count": len(values),
        "median_us": med,
        "p10_us": percentile(ordered, 0.1),
        "p90_us": percentile(ordered, 0.9),
        "max_us": max(values),
        "relative_span": (percentile(ordered, 0.9) - percentile(ordered, 0.1)) / med,
        "max_over_median": max(values) / med,
    }


def queued_events(fn, samples):
    starts = [torch.musa.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(samples)]
    host = []
    for start, end in zip(starts, ends):
        t0 = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        host.append((time.perf_counter_ns() - t0) / 1000.0)
    torch.musa.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)], host


def synchronized_events(fn, samples):
    values = []
    for _ in range(samples):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def graph_replay(fn, samples, unroll):
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        for _ in range(unroll):
            fn()
    torch.musa.synchronize()
    starts = [torch.musa.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.musa.synchronize()
    return [s.elapsed_time(e) * 1000.0 / unroll for s, e in zip(starts, ends)]


def profiler_average(fn, repeats):
    activities = [torch.profiler.ProfilerActivity.MUSA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(repeats):
            fn()
    torch.musa.synchronize()
    rows = []
    for event in prof.key_averages():
        device_us = float(event.self_device_time_total)
        if device_us > 0:
            rows.append({"key": event.key, "count": int(event.count), "self_device_us": device_us})
    kernel_rows = [row for row in rows if "KernelFill<" not in row["key"]]
    return rows, sum(row["self_device_us"] for row in kernel_rows) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {"id": "m256_default_s1", "M": 256, "N": 256, "K": 256, "block_M": 64, "block_N": 64, "block_K": 64, "num_warp": 4, "policy": "default", "num_stages": 1},
        {"id": "m512_default_s1", "M": 512, "N": 512, "K": 512, "block_M": 64, "block_N": 64, "block_K": 64, "num_warp": 4, "policy": "default", "num_stages": 1},
        {"id": "m512_b128_stage3", "M": 512, "N": 512, "K": 512, "block_M": 128, "block_N": 128, "block_K": 64, "num_warp": 4, "policy": "default", "num_stages": 3},
        {"id": "m1024_default_s1", "M": 1024, "N": 1024, "K": 1024, "block_M": 64, "block_N": 64, "block_K": 64, "num_warp": 4, "policy": "default", "num_stages": 1},
    ]
    summary = []
    raw = []
    profiler_rows = []
    torch.manual_seed(20260729)
    tilelang.disable_cache()
    for case in cases:
        row = {"case_id": case["id"], "status": "compile_failed", "error": ""}
        try:
            program = make_matmul(**{key: value for key, value in case.items() if key != "id"})
            kernel = tilelang.compile(program, out_idx=[], target="musa", pass_configs=PASS_CONFIGS)
            a = torch.randn((case["M"], case["K"]), dtype=torch.float16, device="musa")
            b = torch.randn((case["K"], case["N"]), dtype=torch.float16, device="musa")
            c = torch.empty((case["M"], case["N"]), dtype=torch.float16, device="musa")
            fn = lambda: kernel(a, b, c)
            fn()
            torch.musa.synchronize()
            torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
            for _ in range(200):
                fn()
            torch.musa.synchronize()
            queued, host = queued_events(fn, args.samples)
            sync = synchronized_events(fn, min(50, args.samples))
            methods = {"queued_single_call_event": queued, "synchronized_single_call_event": sync}
            graph_error = ""
            try:
                methods["musagraph_replay_unroll100"] = graph_replay(fn, 50, 100)
            except Exception:
                graph_error = traceback.format_exc(limit=3).replace("\n", " | ")
            p_rows, profiler_us = profiler_average(fn, 100)
            for p_row in p_rows:
                profiler_rows.append({"case_id": case["id"], **p_row})
            row = {"case_id": case["id"], "status": "measured", "error": "", "graph_error": graph_error,
                   "profiler_average_us": profiler_us, "host_submission": stats(host)}
            for method, values in methods.items():
                row[method] = stats(values)
                for index, value in enumerate(values):
                    raw.append({"case_id": case["id"], "method": method, "sample_index": index, "latency_us": value})
        except Exception:
            row["error"] = traceback.format_exc(limit=5).replace("\n", " | ")
        summary.append(row)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "raw_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "method", "sample_index", "latency_us"])
        writer.writeheader(); writer.writerows(raw)
    with (args.output_dir / "profiler_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "key", "count", "self_device_us"])
        writer.writeheader(); writer.writerows(profiler_rows)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
