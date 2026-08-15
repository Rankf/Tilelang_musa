#!/usr/bin/env python3
"""P1 measurement-harness smoke benchmark for one TileLang-MUSA kernel.

This program deliberately reports timing quality, not a hardware calibration
parameter.  It must be run through scripts/p1_run_measurement_smoke.sh so the
environment snapshot and kernel-source dump are captured beside its CSV data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import uuid

import torch
import torch_musa  # noqa: F401 - registers torch.musa
import tilelang
import tilelang.language as T


ROOT = pathlib.Path(__file__).resolve().parents[1]
TILELANG_ROOT = ROOT / "tilelang-musa"


@tilelang.jit(target="musa", verbose=True)
def elementwise_add(A, B, num_per_thread=8, threads=256, dtype="float32"):
    n = T.const("N")
    A: T.Tensor[[n], dtype]
    B: T.Tensor[[n], dtype]
    C = T.empty((n,), dtype)
    with T.Kernel(T.ceildiv(n, threads * num_per_thread), threads=threads) as bx:
        for i, j in T.Parallel(threads, num_per_thread):
            offsets = (bx * threads + i) * num_per_thread
            C[offsets + j] = A[offsets + j] + B[offsets + j]
    return C


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(TILELANG_ROOT), *args], text=True).strip()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty sample set")
    index = (len(sorted_values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def capture_gpu_state(path: pathlib.Path) -> None:
    try:
        result = subprocess.run(["mthreads-gmi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        path.write_text(result.stdout, encoding="utf-8")
    except FileNotFoundError:
        path.write_text("mthreads-gmi unavailable\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--raw-csv", type=pathlib.Path, required=True)
    parser.add_argument("--summary-csv", type=pathlib.Path, required=True)
    parser.add_argument("--metadata-json", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-before", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-after", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-dump-dir", type=pathlib.Path, required=True)
    parser.add_argument("--runner-log", type=pathlib.Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["phase"] != "P1" or config["benchmark"] != "tilelang_elementwise_add":
        raise ValueError("unexpected P1 benchmark configuration")
    if config["dtype"] != "float32" or len(config["shape"]) != 1:
        raise ValueError("this smoke benchmark intentionally supports one float32 vector shape")

    for path in (args.raw_csv, args.summary_csv, args.metadata_json, args.gpu_before, args.gpu_after):
        path.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"p1-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    n = int(config["shape"][0])
    threads = int(config["threads"])
    num_per_thread = int(config["num_per_thread"])
    warmup = int(config["warmup"])
    sample_count = int(config["samples"])
    repeats = int(config["repeats_per_sample"])
    if n <= 0 or n % (threads * num_per_thread) != 0 or min(warmup, sample_count, repeats) <= 0:
        raise ValueError("shape must exactly cover the TileLang launch and timing counts must be positive")

    capture_gpu_state(args.gpu_before)
    tilelang.disable_cache()
    kernel = elementwise_add.compile(N=n, threads=threads, num_per_thread=num_per_thread, dtype=config["dtype"])
    source = kernel.get_kernel_source()
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    torch.manual_seed(20260728)
    a = torch.randn(n, dtype=torch.float32, device="musa")
    b = torch.randn(n, dtype=torch.float32, device="musa")
    expected = a + b
    actual = kernel(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    for _ in range(warmup):
        kernel(a, b)
    torch.musa.synchronize()

    samples_us: list[float] = []
    for _ in range(sample_count):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            kernel(a, b)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / repeats)

    capture_gpu_state(args.gpu_after)
    ordered = sorted(samples_us)
    median_us = statistics.median(ordered)
    p10_us = percentile(ordered, 0.10)
    p90_us = percentile(ordered, 0.90)
    relative_span = (p90_us - p10_us) / median_us
    stable = relative_span <= float(config["stability"]["max_relative_span"])
    status = "measured" if stable else "runtime_failed"
    patch_path = ROOT / "logs/env/tilelang_musa_local.patch"
    patch_hash = sha256(patch_path) if patch_path.exists() else "missing"
    command = " ".join(sys.argv)
    common = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "benchmark": config["benchmark"],
        "status": status,
        "dtype": config["dtype"],
        "shape": json.dumps(config["shape"]),
        "tile": "null",
        "threads": threads,
        "block": threads,
        "stages": "null",
        "smem_bytes": "unknown",
        "register_proxy": "unknown",
        "grid": n // (threads * num_per_thread),
        "warmup": warmup,
        "repeats": repeats,
        "musa_sdk": os.environ.get("MUSA_HOME", "unknown"),
        "mcc_version": subprocess.check_output(["mcc", "--version"], text=True).splitlines()[1],
        "tilelang_commit": git_value(["rev-parse", "HEAD"]),
        "tilelang_patch_sha256": patch_hash,
        "source_hash": source_hash,
        "harness_hash": sha256(pathlib.Path(__file__)),
        "config_hash": sha256(args.config),
        "kernel_dump_dir": str(args.kernel_dump_dir),
        "runner_log": str(args.runner_log),
        "command": command,
    }
    raw_fields = [*common.keys(), "sample_index", "latency_us"]
    with args.raw_csv.open("w", newline="", encoding="utf-8") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
        writer.writeheader()
        for index, latency_us in enumerate(samples_us):
            writer.writerow({**common, "sample_index": index, "latency_us": f"{latency_us:.6f}"})

    summary = {
        **common,
        "latency_us_median": f"{median_us:.6f}",
        "latency_us_p10": f"{p10_us:.6f}",
        "latency_us_p90": f"{p90_us:.6f}",
        "relative_span": f"{relative_span:.6f}",
        "stability_threshold": config["stability"]["max_relative_span"],
        "throughput": "not_applicable_p1",
        "correctness": "pass",
        "timing_scope": config["timing_scope"],
    }
    with args.summary_csv.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    metadata = {
        "run_id": run_id,
        "measurement_kind": "P1 stability smoke; excluded from P2-P5 parameter fitting",
        "config": config,
        "summary": summary,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_musa": getattr(torch_musa, "__version__", "unknown"),
        "musa_available": torch.musa.is_available(),
        "musa_capability": list(torch.musa.get_device_capability()),
        "kernel_source_hash": source_hash,
        "harness_sha256": sha256(pathlib.Path(__file__)),
        "config_sha256": sha256(args.config),
        "tilelang_patch_sha256": patch_hash,
        "artifacts": {
            "raw_csv": str(args.raw_csv),
            "summary_csv": str(args.summary_csv),
            "gpu_before": str(args.gpu_before),
            "gpu_after": str(args.gpu_after),
            "kernel_dump_dir": str(args.kernel_dump_dir),
            "runner_log": str(args.runner_log),
        },
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": status, "median_us": median_us, "relative_span": relative_span}, sort_keys=True))
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
