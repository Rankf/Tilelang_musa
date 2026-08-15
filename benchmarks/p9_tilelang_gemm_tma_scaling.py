#!/usr/bin/env python3
"""Parameterized TileLang GEMM for MCU TME-counter scaling experiments."""

import argparse
import json
import math
import os
import re
import statistics

os.environ.setdefault("MUSA_HOME", "/usr/local/musa")

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T


def make_gemm(m, n, k, bm, bn, bk, nw, ns):
    @T.prim_func
    def main(
        a: T.Tensor((m, k), "float16"),
        b: T.Tensor((k, n), "float16"),
        c: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=nw * 32) as (bx, by):
            a_shared = T.alloc_shared((bm, bk), "float16")
            b_shared = T.alloc_shared((bk, bn), "float16")
            c_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=ns):
                T.copy(a[by * bm, ko * bk], a_shared)
                T.copy(b[ko * bk, bx * bn], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, c[by * bm, bx * bn])
    return main


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("M", "N", "K", "bm", "bn", "bk", "nw", "ns"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if min(args.M, args.N, args.K, args.bm, args.bn, args.bk,
           args.nw, args.ns, args.samples) <= 0:
        raise ValueError("all dimensions and counts must be positive")
    if args.M % args.bm or args.N % args.bn or args.K % args.bk:
        raise ValueError("this scaling probe requires exact tile divisibility")

    tilelang.disable_cache()
    kernel = tilelang.compile(
        make_gemm(args.M, args.N, args.K, args.bm, args.bn, args.bk,
                  args.nw, args.ns),
        out_idx=[],
        target="musa",
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
        },
    )
    source = kernel.get_kernel_source()
    symbol_match = re.search(r"__global__\s+void\s+(\w+)", source)
    kernel_symbol = symbol_match.group(1) if symbol_match else "unknown"
    uses_tma = "MUtensorDescriptor" in source or "tl::tma_load<" in source
    launch_match = re.search(r"__launch_bounds__\((\d+)", source)
    launch_threads = int(launch_match.group(1)) if launch_match else None
    if not uses_tma:
        raise RuntimeError("generated kernel does not contain the required TMA path")

    a = torch.ones((args.M, args.K), dtype=torch.float16, device="musa")
    b = torch.ones((args.K, args.N), dtype=torch.float16, device="musa")
    c = torch.empty((args.M, args.N), dtype=torch.float16, device="musa")

    kernel(a, b, c)
    torch.musa.synchronize()
    observed = c.flatten()[:16].float().cpu()
    expected = torch.full_like(observed, float(args.K))
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    samples_us = []
    for _ in range(args.samples):
        start = torch.musa.Event(enable_timing=True)
        stop = torch.musa.Event(enable_timing=True)
        start.record()
        kernel(a, b, c)
        stop.record()
        stop.synchronize()
        samples_us.append(start.elapsed_time(stop) * 1000.0)

    grid_m = math.ceil(args.M / args.bm)
    grid_n = math.ceil(args.N / args.bn)
    k_iterations = math.ceil(args.K / args.bk)
    logical_tma_bytes = (
        grid_m * grid_n * k_iterations
        * (args.bm * args.bk + args.bk * args.bn) * 2
    )
    print(json.dumps({
        "case": (
            f"gemm_{args.M}_{args.N}_{args.K}_"
            f"tile_{args.bm}_{args.bn}_{args.bk}"
        ),
        "M": args.M,
        "N": args.N,
        "K": args.K,
        "bm": args.bm,
        "bn": args.bn,
        "bk": args.bk,
        "nw_requested": args.nw,
        "ns": args.ns,
        "grid_m": grid_m,
        "grid_n": grid_n,
        "grid_cta": grid_m * grid_n,
        "k_iterations": k_iterations,
        "logical_tma_bytes": logical_tma_bytes,
        "kernel_symbol": kernel_symbol,
        "generated_launch_threads": launch_threads,
        "uses_tma": uses_tma,
        "latency_us_median": statistics.median(samples_us),
        "correctness": "pass",
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
