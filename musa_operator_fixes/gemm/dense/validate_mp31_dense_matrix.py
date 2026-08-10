#!/usr/bin/env python3
"""Correctness matrix for repaired MP31 dense GEMM layouts and workloads."""

import argparse
import json

import torch
import torch_musa  # noqa: F401

from tileops.ops import GemmOp

from musa_operator_fixes.gemm.dense import MusaMP31DenseGemmKernel


CASES = {
    "fp16_nn_square": (1024, 1024, 1024, torch.float16, False, False),
    "bf16_nn_square": (1024, 1024, 1024, torch.bfloat16, False, False),
    "fp16_nt_small": (128, 256, 512, torch.float16, False, True),
    "fp16_tn_small": (128, 256, 512, torch.float16, True, False),
    "fp16_tt_small": (128, 256, 512, torch.float16, True, True),
    "fp16_all_tail_nn": (65, 130, 129, torch.float16, False, False),
    "bf16_manifest_decode_down_nt": (128, 7168, 2048, torch.bfloat16, False, True),
    "bf16_manifest_decode_gate_up_nt": (128, 2112, 7168, torch.bfloat16, False, True),
    "bf16_manifest_prefill_gate_up_nt": (4096, 2112, 7168, torch.bfloat16, False, True),
    "bf16_manifest_prefill_down_nt": (4096, 7168, 2048, torch.bfloat16, False, True),
    "fp16_manifest_prefill_attn_nt": (4096, 4096, 7168, torch.float16, False, True),
    "bf16_manifest_prefill_attn_nt": (4096, 4096, 7168, torch.bfloat16, False, True),
    "bf16_manifest_k_dominant_nt": (4096, 7168, 16384, torch.bfloat16, False, True),
    "bf16_manifest_wide_n_nt": (4096, 24576, 1536, torch.bfloat16, False, True),
    "bf16_manifest_mid_m16_nt": (16, 4096, 7168, torch.bfloat16, False, True),
    "bf16_manifest_mid_m32_nt": (32, 4096, 7168, torch.bfloat16, False, True),
    "bf16_manifest_mid_m64_nt": (64, 7168, 2048, torch.bfloat16, False, True),
    "bf16_manifest_mid_m96_nt": (96, 2112, 7168, torch.bfloat16, False, True),
}


def run_case(case_id: str) -> dict:
    m, n, k, dtype, trans_a, trans_b = CASES[case_id]
    torch.manual_seed(20260729)
    a = torch.randn((k, m) if trans_a else (m, k), device="musa", dtype=dtype)
    b = torch.randn((n, k) if trans_b else (k, n), device="musa", dtype=dtype)
    expected_a = a.T if trans_a else a
    expected_b = b.T if trans_b else b
    expected = (expected_a.float() @ expected_b.float()).to(dtype)
    op = GemmOp(
        trans_a=trans_a,
        trans_b=trans_b,
        kernel_map={"gemm_kernel": MusaMP31DenseGemmKernel},
    )
    actual = op(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    return {
        "status": "pass",
        "case": case_id,
        "shape": [m, n, k],
        "dtype": str(dtype).replace("torch.", ""),
        "layout": f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    print(json.dumps(run_case(args.case), sort_keys=True))


if __name__ == "__main__":
    main()
