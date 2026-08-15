#!/usr/bin/env python3
"""P7.3 FP16 GEMM pilot: graph-free direct batch timing and source provenance."""
import argparse, hashlib, json, pathlib, re, statistics, subprocess, traceback
import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_gemm(case, schedule):
    m, n, K = case["M"], case["N"], case["K"]
    bm, bn, bk = schedule["block_M"], schedule["block_N"], schedule["block_K"]
    policy = {"default": None, "m": GemmWarpPolicy.FullRow, "n": GemmWarpPolicy.FullCol, "square": GemmWarpPolicy.Square}[schedule["policy"]]
    @T.prim_func
    def main(A: T.Tensor((m, K), "float16"), B: T.Tensor((K, n), "float16"), C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=schedule["num_warp"] * 32) as (bx, by):
            a = T.alloc_shared((bm, bk), "float16"); b = T.alloc_shared((bk, bn), "float16")
            out = T.alloc_fragment((bm, bn), "float32"); T.clear(out)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=schedule["num_stages"]):
                T.copy(A[by * bm, k * bk], a); T.copy(B[k * bk, bx * bn], b)
                if policy is None: T.gemm(a, b, out)
                else: T.gemm(a, b, out, policy=policy)
            T.copy(out, C[by * bm, bx * bn])
    return main


def features(source):
    return {"tma": "MUtensorDescriptor" in source or "tl::tma_load<" in source, "descriptor_prefetch": "prefetch_tma_descriptor" in source, "async_barrier": "__musa_async_" in source, "warp_specialized": "__musa_async_wait" in source and "__musa_async_arrive" in source, "launch_threads": int(re.search(r"__launch_bounds__\((\d+)", source).group(1)) if re.search(r"__launch_bounds__\((\d+)", source) else None}


def percentile(values, q):
    values = sorted(values); p = (len(values) - 1) * q; lo = int(p); hi = min(lo + 1, len(values) - 1); return values[lo] + (values[hi] - values[lo]) * (p - lo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=pathlib.Path, required=True); ap.add_argument("--case", required=True); ap.add_argument("--schedule", required=True); ap.add_argument("--trial", type=int, required=True); ap.add_argument("--output", type=pathlib.Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); case = next(x for x in cfg["cases"] if x["id"] == args.case); schedule = next(x for x in cfg["schedules"] if x["id"] == args.schedule); tm = cfg["timing"]
    row = {"phase": cfg["phase"], "protocol_version": cfg["protocol_version"], "data_role": cfg["data_role"], "case": case, "schedule": schedule, "trial": args.trial, "status": "compile_failed", "compile": "not_run", "correctness": "not_run", "failure_stage": "compile", "error": "", "timing_method": "graph-free MUSA Event batch-32; synchronize each batch; no MUSAGraph/Profiler", "metadata": {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip()}}
    try:
        tilelang.disable_cache(); torch.manual_seed(20260729 + args.trial); kernel = tilelang.compile(make_gemm(case, schedule), out_idx=[], target="musa", pass_configs={tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True})
        source = kernel.get_kernel_source(); a = torch.randn((case["M"], case["K"]), device="musa", dtype=torch.float16); b = torch.randn((case["K"], case["N"]), device="musa", dtype=torch.float16); c = torch.empty((case["M"], case["N"]), device="musa", dtype=torch.float16); expected = (a.float() @ b.float()).to(torch.float16); fn = lambda: kernel(a, b, c)
        row.update(compile="pass", source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_features=features(source), grid_cta=((case["M"] + schedule["block_M"] - 1) // schedule["block_M"]) * ((case["N"] + schedule["block_N"] - 1) // schedule["block_N"]))
        fn(); torch.musa.synchronize(); torch.testing.assert_close(c, expected, rtol=2e-2, atol=2e-2); row["correctness"] = "pass"
        for _ in range(tm["warmup"]): fn()
        torch.musa.synchronize(); values = []
        for _ in range(tm["samples_per_trial"]):
            start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True); start.record()
            for _ in range(tm["batch_unroll"]): fn()
            end.record(); torch.musa.synchronize(); values.append(start.elapsed_time(end) * 1000.0 / tm["batch_unroll"])
        median = statistics.median(values); p10, p90 = percentile(values, .1), percentile(values, .9); span = (p90 - p10) / median
        row.update(status="measured" if span <= tm["max_trial_p90_p10_fraction"] else "measurement_rejected", failure_stage="" if span <= tm["max_trial_p90_p10_fraction"] else "within_trial_stability_gate", samples_us=values, latency_us_median=median, latency_us_p10=p10, latency_us_p90=p90, relative_p90_p10_span=span, effective_tops=(2 * case["M"] * case["N"] * case["K"]) / (median * 1e6))
    except AssertionError: row.update(status="correctness_failed", failure_stage="correctness", error=traceback.format_exc(limit=6).replace("\n", " | "))
    except Exception: row.update(status="runtime_failed" if row["compile"] == "pass" else "compile_failed", error=traceback.format_exc(limit=6).replace("\n", " | "))
    try: row["metadata"]["musa_capability"] = list(torch.musa.get_device_capability())
    except Exception: row["metadata"]["musa_capability"] = "unavailable"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(row, indent=2) + "\n"); print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__": main()
