#!/usr/bin/env python3
"""Run the exact CompilerInfo Add/ReLU PrimFuncs for Event or MCU measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import torch
import torch_musa  # noqa: F401
import tilelang

from workloads.elementwise_tilelang import make_add_prim_func, make_relu_prim_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("add", "relu"))
    parser.add_argument(
        "--mode", choices=("profile", "profile-cold", "baseline"), required=True
    )
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    n = args.n
    threads = 256
    elements_per_thread = 8 if args.op == "add" else 4
    elements_per_block = threads * elements_per_thread
    if n <= 0 or n % elements_per_block:
        raise ValueError(
            f"n must be positive and divisible by {elements_per_block}"
        )
    factory = make_add_prim_func if args.op == "add" else make_relu_prim_func
    prim_func = factory(
        n=n,
        threads=threads,
        elements_per_thread=elements_per_thread,
    )
    kernel = tilelang.compile(prim_func, target={"kind": "musa", "arch": "mp_31"})
    source = kernel.get_kernel_source()

    left = torch.full((n,), -1.0 if args.op == "relu" else 1.0,
                      dtype=torch.float32, device="musa")
    output = torch.empty((n,), dtype=torch.float32, device="musa")
    right = (
        torch.full((n,), 2.0, dtype=torch.float32, device="musa")
        if args.op == "add" else None
    )
    torch.musa.synchronize()

    def launch() -> None:
        if args.op == "add":
            kernel(left, right, output)
        else:
            kernel(left, output)

    l2_eviction_bytes = None
    if args.mode == "profile-cold":
        # S5000 reports 60 MiB L2.  A 256 MiB read-only sweep after input
        # creation removes the tiny 64/128 KiB operand working set from L2.
        # Do not initialize this scratch allocation here: a streaming fill
        # leaves dirty cache lines whose delayed writeback can contaminate the
        # target kernel's DDR counters.  The arbitrary reduction value is not
        # consumed; only its memory reads and completion are required.
        l2_eviction_bytes = 256 * 1024 * 1024
        eviction = torch.empty(
            (l2_eviction_bytes // torch.float32.itemsize,),
            dtype=torch.float32,
            device="musa",
        )
        eviction_checksum = eviction.sum()
        torch.musa.synchronize()
        launches = 1
        launch()
        torch.musa.synchronize()
        samples_us: list[float] = []
    elif args.mode == "profile":
        launches = 6
        for _ in range(launches):
            launch()
        torch.musa.synchronize()
        samples_us: list[float] = []
    else:
        launches = 20
        for _ in range(launches):
            launch()
        torch.musa.synchronize()
        samples_us = []
        for _ in range(args.samples):
            start = torch.musa.Event(enable_timing=True)
            stop = torch.musa.Event(enable_timing=True)
            start.record()
            launch()
            stop.record()
            stop.synchronize()
            samples_us.append(start.elapsed_time(stop) * 1000.0)

    expected = torch.zeros_like(output) if args.op == "relu" else torch.full_like(output, 3.0)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    summary = {
        "schema_version": "s5000/elementwise-runtime-runner-v1",
        "op": args.op,
        "mode": args.mode,
        "correctness": "pass",
        "shape": [n],
        "dtype": "float32",
        "threads": threads,
        "elements_per_thread": elements_per_thread,
        "elements_per_block": threads * elements_per_thread,
        "grid_size": n // (threads * elements_per_thread),
        "logical_read_bytes": n * torch.float32.itemsize * (
            2 if args.op == "add" else 1
        ),
        "logical_write_bytes": n * torch.float32.itemsize,
        "target_launches": (
            launches if args.mode in {"profile", "profile-cold"} else args.samples
        ),
        "cache_condition": (
            "cold_explicit_l2_eviction"
            if args.mode == "profile-cold"
            else "warm_after_repeated_launches"
            if args.mode in {"profile", "baseline"}
            else None
        ),
        "l2_eviction_bytes": l2_eviction_bytes,
        "mcu_expected_skip": (
            5 if args.mode == "profile" else 0 if args.mode == "profile-cold" else None
        ),
        "kernel_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    if samples_us:
        summary["musa_event"] = {
            "samples": len(samples_us),
            "median_us": statistics.median(samples_us),
            "p10_us": percentile(samples_us, 0.10),
            "p90_us": percentile(samples_us, 0.90),
            "minimum_us": min(samples_us),
            "maximum_us": max(samples_us),
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
