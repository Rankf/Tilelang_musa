#!/usr/bin/env python3
"""P7.2 graph-free Event slope protocol; deliberately excludes graph/profiler."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import traceback

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_program(case):
    op = case["op"]
    if op == "gemm":
        m, n, K, bm, bn, bk = (case[x] for x in ("M", "N", "K", "block_M", "block_N", "block_K"))
        dtype, accum = case["dtype"], case["accum_dtype"]
        policy = {"default": None, "m": GemmWarpPolicy.FullRow, "n": GemmWarpPolicy.FullCol,
                  "square": GemmWarpPolicy.Square}[case["policy"]]
        @T.prim_func
        def main(A: T.Tensor((m, K), dtype), B: T.Tensor((K, n), dtype), C: T.Tensor((m, n), dtype)):
            with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=case["num_warp"] * 32) as (bx, by):
                a = T.alloc_shared((bm, bk), dtype); b = T.alloc_shared((bk, bn), dtype)
                out = T.alloc_fragment((bm, bn), accum); T.clear(out)
                for k in T.Pipelined(T.ceildiv(K, bk), num_stages=case["num_stages"]):
                    T.copy(A[by * bm, k * bk], a); T.copy(B[k * bk, bx * bn], b)
                    if policy is None: T.gemm(a, b, out)
                    else: T.gemm(a, b, out, policy=policy)
                T.copy(out, C[by * bm, bx * bn])
        return main
    n, threads, vector, dtype = (case[x] for x in ("N", "threads", "vector", "dtype"))
    if n % (threads * vector):
        raise ValueError("N must be divisible by threads*vector")
    @T.prim_func
    def unary(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            for i, j in T.Parallel(threads, vector):
                p = (bx * threads + i) * vector + j
                if op == "single_write": C[p] = 1.0
                elif op == "exp": C[p] = T.exp(A[p])
                else: C[p] = T.rsqrt(A[p])
    @T.prim_func
    def binary(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            for i, j in T.Parallel(threads, vector):
                p = (bx * threads + i) * vector + j; C[p] = A[p] + B[p]
    return binary if op == "vector_add" else unary


def make_inputs(case):
    if case["op"] == "gemm":
        a = torch.randn((case["M"], case["K"]), device="musa", dtype=torch.float16)
        b = torch.randn((case["K"], case["N"]), device="musa", dtype=torch.float16)
        c = torch.empty((case["M"], case["N"]), device="musa", dtype=torch.float16)
        return [a, b, c], (a.float() @ b.float()).to(torch.float16)
    a = torch.rand(case["N"], device="musa", dtype=torch.float32) + 0.25
    c = torch.empty_like(a)
    if case["op"] == "single_write": return [a, c], torch.ones_like(a)
    if case["op"] == "exp": return [a, c], torch.exp(a)
    if case["op"] == "rsqrt": return [a, c], torch.rsqrt(a)
    b = torch.randn_like(a)
    return [a, b, c], a + b


def fit_line(points):
    xs, ys = zip(*points)
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / denom
    intercept = ybar - slope * xbar
    fitted = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, fitted))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return {"intercept_us": intercept, "slope_us_per_launch": slope,
            "r_squared": 1.0 if ss_tot == 0 and ss_res == 0 else 1.0 - ss_res / ss_tot,
            "mean_relative_residual": statistics.mean(abs(y - p) / max(abs(y), 1e-9) for y, p in zip(ys, fitted))}


def source_features(source):
    return {"tma": "MUtensorDescriptor" in source or "tl::tma_load<" in source,
            "descriptor_prefetch": "prefetch_tma_descriptor" in source,
            "async_barrier": "__musa_async_" in source,
            "launch_threads": int(re.search(r"__launch_bounds__\((\d+)", source).group(1)) if re.search(r"__launch_bounds__\((\d+)", source) else None}


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row, indent=2) + "\n"); print(json.dumps(row, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=pathlib.Path, required=True); ap.add_argument("--case", required=True); ap.add_argument("--trial", type=int, required=True); ap.add_argument("--output", type=pathlib.Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); case = next((x for x in cfg["cases"] if x["id"] == args.case), None)
    if case is None: raise ValueError("unknown case")
    timing = cfg["timing"]; row = {"phase": cfg["phase"], "protocol_version": cfg["protocol_version"], "case": case, "trial": args.trial, "status": "compile_failed", "compile": "not_run", "correctness": "not_run", "failure_stage": "compile", "error": "", "timing_method": "graph-free MUSA Event; synchronize each batch; no MUSAGraph/Profiler", "metadata": {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip()}}
    try:
        tilelang.disable_cache(); torch.manual_seed(20260729 + args.trial)
        kernel = tilelang.compile(make_program(case), out_idx=[], target="musa", pass_configs={tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True})
        source = kernel.get_kernel_source(); argv, expected = make_inputs(case); fn = lambda: kernel(*argv)
        row.update(compile="pass", source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_features=source_features(source))
        fn(); torch.musa.synchronize(); torch.testing.assert_close(argv[-1], expected, rtol=2e-2, atol=2e-2); row["correctness"] = "pass"
        for _ in range(timing["warmup"]): fn()
        torch.musa.synchronize(); all_samples, means = [], []
        for unroll in timing["batch_unrolls"]:
            values = []
            for _ in range(timing["samples_per_unroll"]):
                start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
                start.record()
                for _ in range(unroll): fn()
                end.record(); torch.musa.synchronize(); values.append(start.elapsed_time(end) * 1000.0)
            all_samples.append({"unroll": unroll, "total_us_samples": values, "total_us_median": statistics.median(values), "per_launch_us_median": statistics.median(values) / unroll})
            means.append((unroll, statistics.mean(values)))
        fit = fit_line([(x, y) for x, y in means if x in timing["fit_unrolls"]])
        accepted = fit["slope_us_per_launch"] > 0 and fit["intercept_us"] >= 0 and fit["r_squared"] >= timing["min_fit_r_squared"] and fit["mean_relative_residual"] <= timing["max_mean_relative_residual"]
        row.update(status="measured" if accepted else "measurement_rejected", failure_stage="" if accepted else "linear_fit_gate", samples=all_samples, fit=fit)
    except AssertionError:
        row.update(status="correctness_failed", failure_stage="correctness", error=traceback.format_exc(limit=6).replace("\n", " | "))
    except Exception:
        row.update(status="runtime_failed" if row["compile"] == "pass" else "compile_failed", error=traceback.format_exc(limit=6).replace("\n", " | "))
    try: row["metadata"]["musa_capability"] = list(torch.musa.get_device_capability())
    except Exception: row["metadata"]["musa_capability"] = "unavailable"
    write(args.output, row)


if __name__ == "__main__": main()
