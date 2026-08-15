#!/usr/bin/env python3
"""Single-process staged MUSA Graph diagnosis for the known large GEMM case."""

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import traceback

import torch
import torch_musa  # noqa: F401
import tilelang

from p2_tilelang_compute_suite import PASS_CONFIGS, inputs, make_gemm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def event_latency_us(fn, repeats):
    start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def error(args, stage, row):
    row.update(status="runtime_failed", failure_stage=stage,
               error=traceback.format_exc(limit=6).replace("\n", " | "))
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--unroll", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    case = config["case"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260729 + args.trial)
    tilelang.disable_cache()
    row = {"case_id": case["id"], "graph_unroll": args.unroll, "trial": args.trial,
           "status": "compile_failed", "correctness": "not_run", "failure_stage": "compile", "error": ""}
    try:
        kernel = tilelang.compile(make_gemm(case), out_idx=[], target="musa", pass_configs=PASS_CONFIGS)
        source = kernel.get_kernel_source()
        row["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        row["compile"] = "pass"
    except Exception:
        return error(args, "compile", row)
    try:
        argv, expected = inputs(case)
        fn = lambda: kernel(*argv)
        fn(); torch.musa.synchronize()
        torch.testing.assert_close(argv[-1], expected, rtol=2e-2, atol=2e-2)
        row["correctness"] = "pass"
    except Exception:
        return error(args, "correctness_or_first_launch", row)
    try:
        for _ in range(config["direct_warmup"]):
            fn()
        torch.musa.synchronize()
        row["direct_event_us"] = event_latency_us(fn, config["direct_event_repeats"])
        row["direct_event"] = "pass"
    except Exception:
        return error(args, "direct_event", row)
    try:
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            for _ in range(args.unroll):
                fn()
        row["graph_capture"] = "pass"
    except Exception:
        return error(args, "graph_capture", row)
    try:
        torch.musa.synchronize()
        row["graph_capture_synchronize"] = "pass"
    except Exception:
        return error(args, "graph_capture_synchronize", row)
    try:
        samples = [event_latency_us(graph.replay, 1) / args.unroll for _ in range(config["graph_replay_samples"])]
        row.update(status="pass", failure_stage="", graph_replay="pass", graph_replay_samples_us=samples,
                   graph_replay_us_median=statistics.median(samples), graph_replay_us_min=min(samples), graph_replay_us_max=max(samples))
    except Exception:
        return error(args, "graph_replay_or_synchronize", row)
    row["metadata"] = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                       "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip(),
                       "mcc_version": next(line for line in subprocess.check_output(["mcc", "--version"], text=True).splitlines() if line.startswith("mcc version")),
                       "musa_capability": list(torch.musa.get_device_capability())}
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__": main()
