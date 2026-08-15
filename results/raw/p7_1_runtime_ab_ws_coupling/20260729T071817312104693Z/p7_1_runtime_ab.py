#!/usr/bin/env python3
"""P7.1 graph-free runtime A/B for the known large FP16 TileLang-MUSA GEMM.

This harness is versioned independently of P2.  It never runs MUSAGraph or
torch Profiler: its purpose is to isolate compile/correctness/repeated-launch
runtime behavior.  The runner executes each variant in a fresh process.
"""

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import traceback

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_gemm(case, *, disable_tma):
    """Create the same mathematical GEMM, optionally disabling TMA per copy."""
    m, n, K, bm, bn, bk = (case[key] for key in ("M", "N", "K", "block_M", "block_N", "block_K"))
    dtype, accum = case["dtype"], case["accum_dtype"]
    policy = {"default": None, "m": GemmWarpPolicy.FullRow, "n": GemmWarpPolicy.FullCol,
              "square": GemmWarpPolicy.Square}[case["policy"]]

    @T.prim_func
    def main(A: T.Tensor((m, K), dtype), B: T.Tensor((K, n), dtype), C: T.Tensor((m, n), accum if dtype == "int8" else dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=case["num_warp"] * 32) as (bx, by):
            a = T.alloc_shared((bm, bk), dtype)
            b = T.alloc_shared((bk, bn), dtype)
            out = T.alloc_fragment((bm, bn), accum)
            T.clear(out)
            # Keep this induction-variable spelling aligned with the historical
            # P7 kernel so baseline source identity is testable byte-for-byte.
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=case["num_stages"]):
                T.copy(A[by * bm, k * bk], a, disable_tma=disable_tma)
                T.copy(B[k * bk, bx * bn], b, disable_tma=disable_tma)
                if policy is None:
                    T.gemm(a, b, out)
                else:
                    T.gemm(a, b, out, policy=policy)
            T.copy(out, C[by * bm, bx * bn], disable_tma=disable_tma)
    return main


def make_inputs(case):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "int8": torch.int8}[case["dtype"]]
    if dtype == torch.int8:
        a = torch.randint(-4, 5, (case["M"], case["K"]), device="musa", dtype=dtype)
        b = torch.randint(-4, 5, (case["K"], case["N"]), device="musa", dtype=dtype)
        out = torch.empty((case["M"], case["N"]), device="musa", dtype=torch.int32)
        expected = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)).to("musa")
    else:
        a = torch.randn((case["M"], case["K"]), device="musa", dtype=dtype)
        b = torch.randn((case["K"], case["N"]), device="musa", dtype=dtype)
        out = torch.empty((case["M"], case["N"]), device="musa", dtype=dtype)
        expected = (a.float() @ b.float()).to(dtype)
    return [a, b, out], expected


def source_features(source):
    return {
        "tma": "MUtensorDescriptor" in source or "tl::tma_load<" in source,
        "descriptor_prefetch": "prefetch_tma_descriptor" in source,
        "warp_specialized": bool(re.search(r"threadIdx\.x\)\)\) \{|threadIdx\.x\)\) \{", source))
                            and "__musa_async_wait" in source and "__musa_async_arrive" in source,
        "async_barrier": "__musa_async_" in source,
        "launch_threads": int(re.search(r"__launch_bounds__\((\d+)", source).group(1))
                          if re.search(r"__launch_bounds__\((\d+)", source) else None,
    }


def contract_errors(features, contract):
    return {key: {"expected": expected, "observed": features.get(key)}
            for key, expected in contract.items()
            if expected != "any" and features.get(key) != expected}


def pass_configs(variant):
    return {
        tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: variant["enable_musa_tma_prefetch"],
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: variant["disable_warp_specialized"],
    }


def write_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def static_metadata(config_path):
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "tilelang_commit": subprocess.check_output(["git", "-C", str(ROOT / "tilelang-musa"), "rev-parse", "HEAD"], text=True).strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    variant = next((item for item in cfg["variants"] if item["id"] == args.variant), None)
    if variant is None:
        raise ValueError(f"unknown variant: {args.variant}")
    case = cfg["case"]
    row = {"phase": cfg["phase"], "case_id": case["id"], "variant": variant["id"], "trial": args.trial,
           "compile": "not_run", "correctness": "not_run", "status": "compile_failed", "failure_stage": "compile",
           "error": "", "timing_method": "graph-free MUSA Event batch; synchronize each batch; no graph/profiler",
           "metadata": static_metadata(args.config)}
    completed = []
    try:
        tilelang.disable_cache()
        torch.manual_seed(20260729 + args.trial)
        kernel = tilelang.compile(make_gemm(case, disable_tma=variant["disable_tma"]), out_idx=[], target="musa",
                                  pass_configs=pass_configs(variant))
        source = kernel.get_kernel_source()
        features = source_features(source)
        row.update(compile="pass", source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_features=features,
                   pass_configs={str(key.value if hasattr(key, "value") else key): value for key, value in pass_configs(variant).items()})
        mismatch = contract_errors(features, variant["source_contract"])
        if mismatch:
            row.update(status="source_contract_failed", failure_stage="generated_source_contract", source_contract_errors=mismatch)
            return write_result(args.output, row)
        argv, expected = make_inputs(case)
        fn = lambda: kernel(*argv)
        fn()
        torch.musa.synchronize()
        torch.testing.assert_close(argv[-1], expected, rtol=2e-2, atol=2e-2)
        row["correctness"] = "pass"
        row["failure_stage"] = "warmup"
        for _ in range(cfg["warmup"]):
            fn()
        torch.musa.synchronize()
        for unroll in cfg["batch_unrolls"]:
            row["failure_stage"] = f"direct_batch_unroll_{unroll}"
            values = []
            for _ in range(cfg["samples_per_unroll"]):
                start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
                start.record()
                for _ in range(unroll):
                    fn()
                end.record()
                torch.musa.synchronize()
                values.append(start.elapsed_time(end) * 1000.0 / unroll)
            completed.append({"batch_unroll": unroll, "samples_us": values,
                              "median_us": sorted(values)[len(values) // 2], "samples_completed": len(values)})
        row.update(status="pass", failure_stage="", completed_unrolls=completed)
    except AssertionError:
        row.update(status="correctness_failed", failure_stage="correctness_assertion", completed_unrolls=completed,
                   error=traceback.format_exc(limit=8).replace("\n", " | "))
    except Exception:
        row.update(status="runtime_failed" if row["compile"] == "pass" else "compile_failed",
                   completed_unrolls=completed,
                   error=traceback.format_exc(limit=8).replace("\n", " | "))
    try:
        row["metadata"]["musa_capability"] = list(torch.musa.get_device_capability())
    except Exception:
        row["metadata"]["musa_capability"] = "unavailable"
    write_result(args.output, row)


if __name__ == "__main__":
    main()
