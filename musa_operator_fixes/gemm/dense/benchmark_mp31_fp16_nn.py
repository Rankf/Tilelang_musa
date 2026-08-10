#!/usr/bin/env python3
"""Compare the repaired MP31 GEMM with torch_musa's native matmul."""

import argparse
import json
import statistics

import torch
import torch_musa  # noqa: F401

from tileops.ops import GemmOp

from musa_operator_fixes.gemm.dense import MusaMP31Fp16NNDenseGemmKernel


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def event_batch_us(fn, batch: int) -> float:
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(batch):
        output = fn()
    end.record()
    torch.musa.synchronize()
    assert output is not None
    return start.elapsed_time(end) * 1000.0 / batch


def summarize(values: list[float], m: int, n: int, k: int) -> dict:
    median_us = statistics.median(values)
    return {
        "median_us": median_us,
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
        "effective_tops": 2 * m * n * k / median_us / 1e6,
        "samples_us": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(20260729)
    m = n = k = 1024
    a = torch.randn((m, k), device="musa", dtype=torch.float16)
    b = torch.randn((k, n), device="musa", dtype=torch.float16)

    repaired_op = GemmOp(
        trans_a=False,
        trans_b=False,
        tune=False,
        kernel_map={"gemm_kernel": MusaMP31Fp16NNDenseGemmKernel},
    )
    repaired_fn = lambda: repaired_op(a, b)
    native_fn = lambda: torch.matmul(a, b)

    repaired = repaired_fn()
    native = native_fn()
    torch.musa.synchronize()
    torch.testing.assert_close(repaired, native, rtol=2e-2, atol=2e-2)

    for _ in range(args.warmup):
        repaired_fn()
        native_fn()
    torch.musa.synchronize()

    repaired_samples = []
    native_samples = []
    for _ in range(args.samples):
        repaired_samples.append(event_batch_us(repaired_fn, args.batch))
        native_samples.append(event_batch_us(native_fn, args.batch))

    repaired_summary = summarize(repaired_samples, m, n, k)
    native_summary = summarize(native_samples, m, n, k)
    result = {
        "status": "pass",
        "comparison_scope": "S5000 MUSA: repaired TileOPs kernel vs torch_musa torch.matmul",
        "shape": [m, n, k],
        "dtype": "float16",
        "layout": "NN",
        "warmup": args.warmup,
        "samples": args.samples,
        "batch": args.batch,
        "repaired": repaired_summary,
        "torch_musa_native": native_summary,
        "repaired_over_native_throughput_ratio": (
            native_summary["median_us"] / repaired_summary["median_us"]
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
