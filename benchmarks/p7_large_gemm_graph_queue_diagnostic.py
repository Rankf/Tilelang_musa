#!/usr/bin/env python3
"""P2-style queued graph replay diagnostic in one fresh process."""

import argparse
import hashlib
import json
import pathlib
import statistics
import traceback

import torch
import torch_musa  # noqa: F401
import tilelang

from p2_tilelang_compute_suite import PASS_CONFIGS, inputs, make_gemm


def fail(args, row, stage):
    row.update(status="runtime_failed", failure_stage=stage, error=traceback.format_exc(limit=6).replace("\n", " | "))
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--queue-depth", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text()); case = cfg["case"]
    torch.manual_seed(20260729 + args.trial); tilelang.disable_cache()
    row = {"case_id": case["id"], "graph_unroll": cfg["graph_unroll"], "queue_depth": args.queue_depth, "trial": args.trial,
           "status": "compile_failed", "correctness": "not_run", "failure_stage": "compile", "error": ""}
    try:
        kernel = tilelang.compile(make_gemm(case), out_idx=[], target="musa", pass_configs=PASS_CONFIGS)
        row.update(compile="pass", source_sha256=hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest())
    except Exception: return fail(args, row, "compile")
    try:
        argv, expected = inputs(case); fn = lambda: kernel(*argv)
        fn(); torch.musa.synchronize(); torch.testing.assert_close(argv[-1], expected, rtol=2e-2, atol=2e-2)
        row["correctness"] = "pass"
        for _ in range(cfg["direct_warmup"]): fn()
        torch.musa.synchronize()
    except Exception: return fail(args, row, "correctness_or_direct_warmup")
    try:
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            for _ in range(cfg["graph_unroll"]): fn()
        torch.musa.synchronize(); row["graph_capture"] = "pass"
    except Exception: return fail(args, row, "graph_capture_or_synchronize")
    try:
        starts = [torch.musa.Event(enable_timing=True) for _ in range(args.queue_depth)]
        ends = [torch.musa.Event(enable_timing=True) for _ in range(args.queue_depth)]
        for start, end in zip(starts, ends):
            start.record(); graph.replay(); end.record()
        row["queued_submission"] = "pass"
    except Exception: return fail(args, row, "queued_submission")
    try:
        torch.musa.synchronize()
        samples = [start.elapsed_time(end) * 1000.0 / cfg["graph_unroll"] for start, end in zip(starts, ends)]
        row.update(status="pass", failure_stage="", queued_synchronize="pass", queued_replay_samples_us=samples,
                   queued_replay_us_median=statistics.median(samples), queued_replay_us_max=max(samples))
        args.output.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    except Exception: return fail(args, row, "queued_synchronize_or_event_elapsed")


if __name__ == "__main__": main()
