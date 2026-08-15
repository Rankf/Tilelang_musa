#!/usr/bin/env python3
"""P7.3.1: graph-free slope timing with retained TileLang-MUSA MUBIN."""
import argparse, hashlib, json, pathlib, statistics, subprocess, traceback
import torch
import torch_musa  # noqa: F401
import tilelang
from p7_3_gemm_development_pilot import make_gemm, features

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks" / "p7_3_gemm_development_pilot.py"


def fit_line(points):
    xs, ys = zip(*points); xb, yb = statistics.mean(xs), statistics.mean(ys); denom = sum((x - xb) ** 2 for x in xs)
    slope = sum((x - xb) * (y - yb) for x, y in points) / denom; intercept = yb - slope * xb
    pred = [intercept + slope * x for x in xs]; residual = sum((y - p) ** 2 for y, p in zip(ys, pred)); total = sum((y - yb) ** 2 for y in ys)
    return {"slope_us_per_launch": slope, "intercept_us": intercept, "r_squared": 1.0 if total == residual == 0 else 1.0 - residual / total, "mean_relative_residual": statistics.mean(abs(y - p) / max(abs(y), 1e-9) for y, p in zip(ys, pred))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=pathlib.Path, required=True); ap.add_argument("--case", required=True); ap.add_argument("--schedule", required=True); ap.add_argument("--trial", type=int, required=True); ap.add_argument("--artifact-dir", type=pathlib.Path, required=True); ap.add_argument("--output", type=pathlib.Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); case = next(x for x in cfg["cases"] if x["id"] == args.case); schedule = next(x for x in cfg["schedules"] if x["id"] == args.schedule); tm = cfg["timing"]; args.artifact_dir.mkdir(parents=True, exist_ok=True)
    row = {"phase": cfg["phase"], "protocol_version": cfg["protocol_version"], "data_role": cfg["data_role"], "case": case, "schedule": schedule, "trial": args.trial, "status": "compile_failed", "compile": "not_run", "correctness": "not_run", "failure_stage": "compile", "error": "", "timing_method": "graph-free MUSA Event linear slope; synchronize each batch; no MUSAGraph/Profiler", "metadata": {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "base_gemm_builder_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(), "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip()}}
    try:
        tilelang.disable_cache(); torch.manual_seed(20260729 + args.trial)
        passes = {tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True, "musa.kernels_output_dir": str(args.artifact_dir)}
        kernel = tilelang.compile(make_gemm(case, schedule), out_idx=[], target="musa", pass_configs=passes)
        source = kernel.get_kernel_source(); mubins = sorted(args.artifact_dir.glob("*.mubin"))
        if len(mubins) != 1: raise RuntimeError(f"expected exactly one retained MUBIN, found {len(mubins)}")
        mubin = mubins[0]; a = torch.randn((case["M"], case["K"]), device="musa", dtype=torch.float16); b = torch.randn((case["K"], case["N"]), device="musa", dtype=torch.float16); c = torch.empty((case["M"], case["N"]), device="musa", dtype=torch.float16); expected = (a.float() @ b.float()).to(torch.float16); fn = lambda: kernel(a, b, c)
        row.update(compile="pass", source_sha256=hashlib.sha256(source.encode()).hexdigest(), mubin_path=str(mubin), mubin_sha256=hashlib.sha256(mubin.read_bytes()).hexdigest(), source_features=features(source), grid_cta=((case["M"] + schedule["block_M"] - 1) // schedule["block_M"]) * ((case["N"] + schedule["block_N"] - 1) // schedule["block_N"]))
        fn(); torch.musa.synchronize(); torch.testing.assert_close(c, expected, rtol=2e-2, atol=2e-2); row["correctness"] = "pass"
        for _ in range(tm["warmup"]): fn()
        torch.musa.synchronize(); samples = []; means = []
        for unroll in tm["batch_unrolls"]:
            values = []
            for _ in range(tm["samples_per_unroll"]):
                start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True); start.record()
                for _ in range(unroll): fn()
                end.record(); torch.musa.synchronize(); values.append(start.elapsed_time(end) * 1000.0)
            samples.append({"unroll": unroll, "total_us_samples": values, "total_us_median": statistics.median(values)}); means.append((unroll, statistics.mean(values)))
        fit = fit_line([(x, y) for x, y in means if x in tm["fit_unrolls"]]); ok = fit["slope_us_per_launch"] > 0 and fit["intercept_us"] >= 0 and fit["r_squared"] >= tm["min_fit_r_squared"] and fit["mean_relative_residual"] <= tm["max_mean_relative_residual"]
        row.update(status="measured" if ok else "measurement_rejected", failure_stage="" if ok else "linear_fit_gate", samples=samples, fit=fit)
    except AssertionError: row.update(status="correctness_failed", failure_stage="correctness", error=traceback.format_exc(limit=6).replace("\n", " | "))
    except Exception: row.update(status="runtime_failed" if row["compile"] == "pass" else "compile_failed", error=traceback.format_exc(limit=6).replace("\n", " | "))
    try: row["metadata"]["musa_capability"] = list(torch.musa.get_device_capability())
    except Exception: row["metadata"]["musa_capability"] = "unavailable"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(row, indent=2) + "\n"); print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__": main()
