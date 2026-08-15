#!/usr/bin/env python3
"""Initial P2 FP16 GEMM vertical-slice runner for TileLang-MUSA/MP31."""

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
import sys
import time
import traceback
import uuid

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy


ROOT = pathlib.Path(__file__).resolve().parents[1]
TILELANG_ROOT = ROOT / "tilelang-musa"
PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True}


def make_matmul(M, N, K, block_M, block_N, block_K, num_warp, policy, num_stages):
    if policy == "default":
        warp_policy = None
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    elif policy == "square":
        warp_policy = GemmWarpPolicy.Square
    else:
        raise ValueError(f"unsupported GEMM policy: {policy}")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_warp * 32) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_K), "float16")
            b_shared = T.alloc_shared((block_K, block_N), "float16")
            c_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(c_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], a_shared)
                T.copy(B[k * block_K, bx * block_N], b_shared)
                if warp_policy is None:
                    T.gemm(a_shared, b_shared, c_local)
                else:
                    T.gemm(a_shared, b_shared, c_local, policy=warp_policy)
            T.copy(c_local, C[by * block_M, bx * block_N])

    return main


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(sorted_values: list[float], q: float) -> float:
    index = (len(sorted_values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(TILELANG_ROOT), *args], text=True).strip()


def capture_gpu_state(path: pathlib.Path) -> None:
    result = subprocess.run(["mthreads-gmi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path.write_text(result.stdout, encoding="utf-8")


def latest_dump(dump_dir: pathlib.Path, suffix: str) -> str:
    files = sorted(dump_dir.glob(f"*{suffix}"), key=lambda path: path.stat().st_mtime_ns)
    return str(files[-1]) if files else "missing"


def code_features(source: str) -> str:
    features = [
        "tl::gemm_ss",
        "tl::tma_load",
        "tl::prefetch_tma_descriptor",
        "__musa_async_wait",
        "__musa_async_arrive",
        "__musa_async_bar_record",
        "__shared__",
    ]
    return json.dumps([feature for feature in features if feature in source])


def generated_launch_threads(source: str) -> int | None:
    match = re.search(r"__launch_bounds__\((\d+)", source)
    return int(match.group(1)) if match else None


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
    if config["phase"] != "P2" or config["benchmark"] != "tilelang_fp16_gemm":
        raise ValueError("unexpected P2 GEMM configuration")
    for path in (args.raw_csv, args.summary_csv, args.metadata_json, args.gpu_before, args.gpu_after):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.kernel_dump_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"p2-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    config_hash = sha256(args.config)
    patch_path = ROOT / "logs/env/tilelang_musa_local.patch"
    patch_hash = sha256(patch_path) if patch_path.exists() else "missing"
    common = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "benchmark": config["benchmark"],
        "dtype": config["dtype"],
        "accum_dtype": config["accum_dtype"],
        "warmup": int(config["warmup"]),
        "repeats": int(config["repeats_per_sample"]),
        "musa_sdk": os.environ.get("MUSA_HOME", "unknown"),
        "mcc_version": next(line for line in subprocess.check_output(["mcc", "--version"], text=True).splitlines() if line.startswith("mcc version")),
        "tilelang_commit": git_value(["rev-parse", "HEAD"]),
        "tilelang_patch_sha256": patch_hash,
        "harness_hash": sha256(pathlib.Path(__file__)),
        "config_hash": config_hash,
        "kernel_dump_dir": str(args.kernel_dump_dir),
        "runner_log": str(args.runner_log),
        "timing_scope": config["timing_scope"],
    }
    summary_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    capture_gpu_state(args.gpu_before)
    tilelang.disable_cache()
    torch.manual_seed(20260729)

    for case in config["cases"]:
        record: dict[str, object] = {
            **common,
            "case_id": case["id"],
            "shape": json.dumps([case["M"], case["N"], case["K"]]),
            "tile": json.dumps([case["block_M"], case["block_N"], case["block_K"]]),
            "threads": case["num_warp"] * 32,
            "block": "unknown",
            "requested_threads": case["num_warp"] * 32,
            "generated_launch_threads": "unknown",
            "stages": case["num_stages"],
            "smem_bytes": "unknown",
            "register_proxy": "unknown",
            "grid": (case["N"] // case["block_N"]) * (case["M"] // case["block_M"]),
            "policy": case["policy"],
            "status": "compile_failed",
            "correctness": "not_run",
            "source_hash": "missing",
            "source_dump": "missing",
            "mcc_command": "missing",
            "code_features": "[]",
            "error": "",
            "latency_us_median": "",
            "latency_us_p10": "",
            "latency_us_p90": "",
            "relative_span": "",
            "max_latency_us": "",
            "max_over_median": "",
            "effective_tflops": "",
            "stability_gate": "not_run",
            "allocation_mode": "preallocated_output",
            "timing_method": "precreated_per_call_events_queued_single_sync",
        }
        try:
            program = make_matmul(
                M=case["M"], N=case["N"], K=case["K"],
                block_M=case["block_M"], block_N=case["block_N"], block_K=case["block_K"],
                num_warp=case["num_warp"], policy=case["policy"], num_stages=case["num_stages"],
            )
            kernel = tilelang.compile(
                program, out_idx=[], target="musa", verbose=True,
                pass_configs=PASS_CONFIGS,
            )
            source = kernel.get_kernel_source()
            record["source_hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            record["source_dump"] = latest_dump(args.kernel_dump_dir, ".mu")
            record["mcc_command"] = latest_dump(args.kernel_dump_dir, ".mu.cmd")
            record["code_features"] = code_features(source)
            launch_threads = generated_launch_threads(source)
            record["generated_launch_threads"] = launch_threads if launch_threads is not None else "unknown"
            record["block"] = launch_threads if launch_threads is not None else "unknown"
        except Exception:
            record["error"] = traceback.format_exc(limit=3).replace("\n", " | ")
            summary_rows.append(record)
            continue

        try:
            a = torch.randn((case["M"], case["K"]), dtype=torch.float16, device="musa")
            b = torch.randn((case["K"], case["N"]), dtype=torch.float16, device="musa")
            expected = a @ b
            actual = torch.empty((case["M"], case["N"]), dtype=torch.float16, device="musa")
            kernel(a, b, actual)
            torch.musa.synchronize()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            record["correctness"] = "pass"
        except Exception:
            record["status"] = "incorrect"
            record["correctness"] = "fail"
            record["error"] = traceback.format_exc(limit=3).replace("\n", " | ")
            summary_rows.append(record)
            continue

        try:
            for _ in range(int(config["warmup"])):
                kernel(a, b, actual)
            torch.musa.synchronize()
            sample_count = int(config["samples"])
            repeats = int(config["repeats_per_sample"])
            if repeats != 1:
                raise ValueError("precreated per-call event timing requires repeats_per_sample=1")
            starts = [torch.musa.Event(enable_timing=True) for _ in range(sample_count)]
            ends = [torch.musa.Event(enable_timing=True) for _ in range(sample_count)]
            submission_samples_us: list[float] = []
            torch.musa.synchronize()
            for start, end in zip(starts, ends):
                host_start = time.perf_counter_ns()
                start.record()
                kernel(a, b, actual)
                end.record()
                host_end = time.perf_counter_ns()
                submission_samples_us.append((host_end - host_start) / 1000.0)
            torch.musa.synchronize()
            samples_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
            ordered = sorted(samples_us)
            median_us = statistics.median(ordered)
            p10_us = percentile(ordered, 0.10)
            p90_us = percentile(ordered, 0.90)
            relative_span = (p90_us - p10_us) / median_us
            max_latency_us = max(samples_us)
            max_over_median = max_latency_us / median_us
            stable = (
                relative_span <= float(config["stability"]["max_relative_span"])
                and max_over_median <= float(config["stability"]["max_over_median"])
            )
            record.update({
                "status": "measured" if stable else "runtime_failed",
                "latency_us_median": f"{median_us:.6f}",
                "latency_us_p10": f"{p10_us:.6f}",
                "latency_us_p90": f"{p90_us:.6f}",
                "relative_span": f"{relative_span:.6f}",
                "max_latency_us": f"{max_latency_us:.6f}",
                "max_over_median": f"{max_over_median:.6f}",
                "effective_tflops": f"{(2.0 * case['M'] * case['N'] * case['K']) / (median_us * 1.0e6):.6f}" if stable else "",
                "stability_gate": "pass" if stable else "fail",
            })
            if not stable:
                record["error"] = (
                    f"stability gate failed: relative_span={relative_span:.6f} "
                    f"(threshold={float(config['stability']['max_relative_span']):.6f}), "
                    f"max_over_median={max_over_median:.6f} "
                    f"(threshold={float(config['stability']['max_over_median']):.6f})"
                )
            for sample_index, (latency_us, submission_us) in enumerate(zip(samples_us, submission_samples_us)):
                raw_rows.append({
                    **record,
                    "sample_index": sample_index,
                    "latency_us": f"{latency_us:.6f}",
                    "host_submission_us": f"{submission_us:.6f}",
                })
        except Exception:
            record["status"] = "runtime_failed"
            record["error"] = traceback.format_exc(limit=3).replace("\n", " | ")
        summary_rows.append(record)

    capture_gpu_state(args.gpu_after)
    summary_fields = list(summary_rows[0].keys())
    with args.summary_csv.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.raw_csv.open("w", newline="", encoding="utf-8") as raw_file:
        writer = csv.DictWriter(
            raw_file,
            fieldnames=[*summary_fields, "sample_index", "latency_us", "host_submission_us"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    metadata = {
        "run_id": run_id,
        "measurement_kind": "P2 initial FP16 GEMM vertical slice",
        "config": config,
        "tilelang_commit": common["tilelang_commit"],
        "tilelang_patch_sha256": patch_hash,
        "harness_sha256": common["harness_hash"],
        "musa_capability": list(torch.musa.get_device_capability()),
        "artifacts": {
            "raw_csv": str(args.raw_csv), "summary_csv": str(args.summary_csv),
            "gpu_before": str(args.gpu_before), "gpu_after": str(args.gpu_after),
            "kernel_dump_dir": str(args.kernel_dump_dir), "runner_log": str(args.runner_log),
        },
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "statuses": {row["case_id"]: row["status"] for row in summary_rows}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
