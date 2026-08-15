#!/usr/bin/env python3
"""P3 TileLang-MUSA global-memory bandwidth and cache-working-set runner."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import time
import traceback
import uuid

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T

ROOT = pathlib.Path(__file__).resolve().parents[1]
PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True}


def percentile(values, q):
    ordered = sorted(values)
    x = (len(ordered) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def logical_bytes(case):
    # Payload accounting only; it is deliberately not claimed as DDR transaction traffic.
    if "N" in case:
        n = case["N"]
    elif case["op"] == "fixed_grid_compute":
        n = case["grid_blocks"] * case["threads"]
    else:
        n = case["grid_blocks"] * case["threads"] * case["cols"]
    op = case["op"]
    if op in ("copy", "strided_copy", "offset_copy", "fixed_grid_copy", "fixed_grid_compute"):
        return 8 * n
    if op == "fill":
        return 4 * n
    if op == "triad":
        return 12 * n
    if op == "shared_read":
        return 4 * case["grid_blocks"] * case["threads"] * case["cols"] * case["repeats"]
    if op == "read_sum":
        return 4 * n + 4 * (n // case["items_per_thread"])
    raise ValueError(f"unknown op {op}")


def make_program(case):
    op = case["op"]
    if op == "fixed_grid_compute":
        grid, threads, iterations = (case[k] for k in ("grid_blocks", "threads", "iterations"))
        n = grid * threads
        @T.prim_func
        def fixed_grid_compute(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(grid, threads=threads) as bx:
                acc = T.alloc_fragment((threads,), "float32")
                for i in T.Parallel(threads):
                    acc[i] = A[bx * threads + i]
                for k in T.Serial(iterations):
                    for i in T.Parallel(threads):
                        acc[i] = acc[i] * 1.0001 + 0.0001
                for i in T.Parallel(threads):
                    C[bx * threads + i] = acc[i]
        return fixed_grid_compute
    if op == "shared_read":
        grid, threads, cols, repeats, stride = (case[k] for k in ("grid_blocks", "threads", "cols", "repeats", "pattern_stride"))
        n = grid * threads * cols
        if stride == 1:
            @T.prim_func
            def shared_read(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
                with T.Kernel(grid, threads=threads) as bx:
                    tile = T.alloc_shared((threads * cols,), "float32")
                    acc = T.alloc_fragment((threads * cols,), "float32")
                    T.copy(A[bx * threads * cols], tile)
                    T.copy(tile, acc)
                    for r in T.Serial(repeats):
                        for i, j in T.Parallel(threads, cols):
                            p = i * cols + j
                            acc[p] = acc[p] + tile[p]
                    T.copy(acc, C[bx * threads * cols])
            return shared_read
        @T.prim_func
        def shared_read_risk(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(grid, threads=threads) as bx:
                tile = T.alloc_shared((threads * cols,), "float32")
                acc = T.alloc_fragment((threads * cols,), "float32")
                T.copy(A[bx * threads * cols], tile)
                T.copy(tile, acc)
                for r in T.Serial(repeats):
                    for i, j in T.Parallel(threads, cols):
                        p = i * cols + j
                        acc[p] = acc[p] + tile[i * cols + (j * stride) % cols]
                T.copy(acc, C[bx * threads * cols])
        return shared_read_risk
    if op == "read_sum":
        n, threads, items = case["N"], case["threads"], case["items_per_thread"]
        if n % (threads * items):
            raise ValueError("read_sum N must be divisible by threads*items_per_thread")
        output_n = n // items
        @T.prim_func
        def read_sum(A: T.Tensor((n,), "float32"), C: T.Tensor((output_n,), "float32")):
            with T.Kernel(T.ceildiv(n, threads * items), threads=threads) as bx:
                acc = T.alloc_fragment((threads,), "float32")
                T.clear(acc)
                for k in T.Serial(items):
                    for i in T.Parallel(threads):
                        acc[i] = acc[i] + A[(bx * threads + i) * items + k]
                for i in T.Parallel(threads):
                    C[bx * threads + i] = acc[i]
        return read_sum
    n, threads, vector = case["N"], case["threads"], case["vector"]
    if n % (threads * vector):
        raise ValueError("N must be divisible by threads*vector for this first controlled suite")

    @T.prim_func
    def copy(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            for i, j in T.Parallel(threads, vector):
                p = (bx * threads + i) * vector + j
                C[p] = A[p]

    @T.prim_func
    def fill(C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            for i, j in T.Parallel(threads, vector):
                p = (bx * threads + i) * vector + j
                C[p] = 1.25

    @T.prim_func
    def triad(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            for i, j in T.Parallel(threads, vector):
                p = (bx * threads + i) * vector + j
                C[p] = A[p] + B[p]

    if op == "fixed_grid_copy":
        grid = case["grid_blocks"]
        unit = grid * threads * vector
        if n % unit:
            raise ValueError("fixed-grid workset must be divisible by grid*threads*vector")
        iterations = n // unit
        @T.prim_func
        def fixed_grid_copy(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(grid, threads=threads) as bx:
                for k in T.Serial(iterations):
                    for i, j in T.Parallel(threads, vector):
                        p = ((k * grid + bx) * threads + i) * vector + j
                        C[p] = A[p]
        return fixed_grid_copy

    if op == "strided_copy":
        stride = case["stride"]
        @T.prim_func
        def strided(A: T.Tensor((n * stride,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
                for i, j in T.Parallel(threads, vector):
                    p = (bx * threads + i) * vector + j
                    C[p] = A[p * stride]
        return strided
    return {"copy": copy, "fill": fill, "triad": triad, "offset_copy": copy}[op]


def make_inputs(case):
    op = case["op"]
    if op == "fixed_grid_compute":
        n = case["grid_blocks"] * case["threads"]
        a = torch.ones(n, dtype=torch.float32, device="musa")
        c = torch.empty_like(a)
        # A scalar FP32 recurrence makes correctness independent of grid size
        # without adding a large CPU reference workload.
        expected_scalar = torch.tensor(1.0, dtype=torch.float32)
        for _ in range(case["iterations"]):
            expected_scalar = expected_scalar * 1.0001 + 0.0001
        expected = torch.full_like(a, float(expected_scalar))
        return [a, c], expected
    if op == "shared_read":
        n = case["grid_blocks"] * case["threads"] * case["cols"]
        a = torch.randn(n, dtype=torch.float32, device="musa")
        c = torch.empty_like(a)
        source = a.view(case["grid_blocks"], case["threads"], case["cols"])
        index = (torch.arange(case["cols"], device="musa") * case["pattern_stride"]) % case["cols"]
        expected = (source + case["repeats"] * source[:, :, index]).reshape(-1)
        return [a, c], expected
    if op == "read_sum":
        n = case["N"]
        a = torch.rand(n, dtype=torch.float32, device="musa")
        c = torch.empty(n // case["items_per_thread"], dtype=torch.float32, device="musa")
        return [a, c], a.reshape(-1, case["items_per_thread"]).sum(dim=1)
    n = case["N"]
    if op == "fill":
        c = torch.empty(n, dtype=torch.float32, device="musa")
        return [c], torch.full((n,), 1.25, dtype=torch.float32, device="musa")
    if op == "triad":
        a = torch.randn(n, dtype=torch.float32, device="musa")
        b = torch.randn(n, dtype=torch.float32, device="musa")
        c = torch.empty_like(a)
        return [a, b, c], a + b
    if op == "strided_copy":
        a = torch.randn(n * case["stride"], dtype=torch.float32, device="musa")
        c = torch.empty(n, dtype=torch.float32, device="musa")
        return [a, c], a[::case["stride"]]
    if op == "offset_copy":
        offset = case["offset_elements"]
        a_backing = torch.randn(n + offset, dtype=torch.float32, device="musa")
        c_backing = torch.empty(n + offset, dtype=torch.float32, device="musa")
        return [a_backing[offset:], c_backing[offset:]], a_backing[offset:]
    a = torch.randn(n, dtype=torch.float32, device="musa")
    c = torch.empty_like(a)
    return [a, c], a


def event_estimate_us(fn):
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 100.0


def graph_samples(fn, unroll, count):
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        for _ in range(unroll):
            fn()
    torch.musa.synchronize()
    starts = [torch.musa.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(count)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.musa.synchronize()
    return [start.elapsed_time(end) * 1000.0 / unroll for start, end in zip(starts, ends)]


def one_cold_graph_sample(fn, unroll, flush):
    flush.zero_()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        for _ in range(unroll):
            fn()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / unroll


def profiler_self_device_us(fn, repeats):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.MUSA]) as profiler:
        for _ in range(repeats):
            fn()
    torch.musa.synchronize()
    entries = []
    for event in profiler.key_averages():
        if event.self_device_time_total > 0:
            entries.append((event.key, int(event.count), float(event.self_device_time_total)))
    return sum(total for key, _, total in entries if "KernelFill<" not in key) / repeats, entries


def capture_gpu_state(path):
    result = subprocess.run(["mthreads-gmi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path.write_text(result.stdout, encoding="utf-8")


def stats(values):
    med = statistics.median(values)
    p10, p90 = percentile(values, .1), percentile(values, .9)
    return med, p10, p90, max(values), (p90 - p10) / med, max(values) / med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-dump-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.kernel_dump_dir.mkdir(parents=True, exist_ok=True)
    if config["phase"] not in ("P3", "P4", "P5.1-development", "P6", "P6-v2", "P6-v3") or config["benchmark"] != "tilelang_global_memory":
        raise ValueError("unexpected configuration")
    torch.manual_seed(20260729)
    tilelang.disable_cache()
    capture_gpu_state(args.output_dir / "gpu_before.txt")
    summary, raw, profiler_rows = [], [], []
    run_id = f"{config['phase'].lower()}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    for case in config["cases"]:
        row = {"run_id": run_id, "case_id": case["id"], "op": case["op"], "dtype": "float32", "case_config_json": json.dumps(case, sort_keys=True), "status": "compile_failed", "correctness": "not_run", "error": "", "logical_payload_bytes": logical_bytes(case), "timing_method": "MUSAGraph replay with profiler self-device cross-check", "allocation_mode": "preallocated"}
        try:
            kernel = tilelang.compile(make_program(case), out_idx=[], target="musa", pass_configs=PASS_CONFIGS)
            source = kernel.get_kernel_source()
            row["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            match = re.search(r"__launch_bounds__\((\d+)", source)
            row["generated_launch_threads"] = match.group(1) if match else "unknown"
            argsv, expected = make_inputs(case)
            fn = lambda: kernel(*argsv)
            fn(); torch.musa.synchronize()
            row["status"] = "incorrect"
            if case["op"] == "shared_read":
                row["correctness_tolerance"] = "rtol=1e-4,atol=1e-5 (FP32 repeated accumulation)"
                torch.testing.assert_close(argsv[-1], expected, rtol=1e-4, atol=1e-5)
            elif case["op"] == "fixed_grid_compute":
                row["correctness_tolerance"] = "rtol=5e-4,atol=5e-5 (FP32 dependent FMA recurrence)"
                torch.testing.assert_close(argsv[-1], expected, rtol=5e-4, atol=5e-5)
            else:
                row["correctness_tolerance"] = "rtol=1e-5,atol=1e-6"
                torch.testing.assert_close(argsv[-1], expected, rtol=1e-5, atol=1e-6)
            row["correctness"] = "pass"
            for _ in range(config["warmup"]):
                fn()
            torch.musa.synchronize()
            estimate = event_estimate_us(fn)
            unroll = max(1, min(config["max_graph_unroll"], round(config["target_graph_replay_us"] / estimate)))
            prof_repeats = max(1, min(config["max_profiler_repeats"], round(config["target_profiler_us"] / estimate)))
            warm_values = graph_samples(fn, unroll, config["graph_samples"])
            med, p10, p90, maximum, span, max_over = stats(warm_values)
            cold_values = []
            if case.get("cache_condition") == "warm_and_post_256m_flush":
                flush = torch.empty(config["cache_flush_bytes"] // 4, dtype=torch.float32, device="musa")
                cold_values = [one_cold_graph_sample(fn, unroll, flush) for _ in range(3)]
            profiler_us, entries = profiler_self_device_us(fn, prof_repeats)
            ratio_delta = abs(med / profiler_us - 1.0)
            stable = span <= config["stability"]["max_relative_span"] and max_over <= config["stability"]["max_over_median"] and ratio_delta <= config["stability"]["max_graph_profiler_ratio_delta"]
            row.update({"status": "measured" if stable else "runtime_failed", "estimate_direct_event_us": estimate, "graph_unroll": unroll, "profiler_repeats": prof_repeats, "warm_latency_us_median": med, "warm_latency_us_p10": p10, "warm_latency_us_p90": p90, "warm_max_latency_us": maximum, "warm_relative_span": span, "warm_max_over_median": max_over, "cold_post_flush_latency_us_median": statistics.median(cold_values) if cold_values else "", "cold_post_flush_samples": json.dumps(cold_values), "profiler_self_device_us": profiler_us, "graph_profiler_ratio_delta": ratio_delta, "logical_effective_GBps": logical_bytes(case) / med / 1e3 if stable else ""})
            if not stable:
                row["error"] = "stability or graph/profiler agreement gate failed"
            for i, value in enumerate(warm_values):
                raw.append({"run_id": run_id, "case_id": case["id"], "condition": "warm", "sample_index": i, "latency_us": value})
            for i, value in enumerate(cold_values):
                raw.append({"run_id": run_id, "case_id": case["id"], "condition": "post_256m_flush", "sample_index": i, "latency_us": value})
            for key, count, total in entries:
                profiler_rows.append({"case_id": case["id"], "key": key, "count": count, "self_device_us": total})
        except Exception:
            row["error"] = traceback.format_exc(limit=5).replace("\n", " | ")
        summary.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    capture_gpu_state(args.output_dir / "gpu_after.txt")
    fields = sorted({key for row in summary for key in row})
    with (args.output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "raw_samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "case_id", "condition", "sample_index", "latency_us"]); writer.writeheader(); writer.writerows(raw)
    with (args.output_dir / "profiler_events.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["case_id", "key", "count", "self_device_us"]); writer.writeheader(); writer.writerows(profiler_rows)
    metadata = {"run_id": run_id, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip(), "musa_home": os.environ.get("MUSA_HOME", "unknown"), "mcc_version": next(line for line in subprocess.check_output(["mcc", "--version"], text=True).splitlines() if line.startswith("mcc version")), "musa_capability": list(torch.musa.get_device_capability()), "kernel_dump_dir": str(args.kernel_dump_dir)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
