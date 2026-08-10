#!/usr/bin/env python3
"""Compile and run one correctness-only TileOPs dense GEMM repair smoke."""

import torch
import torch_musa  # noqa: F401

from tileops.ops import GemmOp

from musa_operator_fixes.gemm.dense import MusaMP31Fp16NNDenseGemmKernel


def main() -> None:
    torch.manual_seed(20260729)
    m = n = k = 1024
    a = torch.randn((m, k), device="musa", dtype=torch.float16)
    b = torch.randn((k, n), device="musa", dtype=torch.float16)
    expected = (a.float() @ b.float()).to(torch.float16)

    op = GemmOp(
        trans_a=False,
        trans_b=False,
        tune=False,
        kernel_map={"gemm_kernel": MusaMP31Fp16NNDenseGemmKernel},
    )
    actual = op(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(
        {
            "status": "pass",
            "shape": [m, n, k],
            "dtype": "float16",
            "layout": "NN",
            "kernel": type(op.kernel).__name__,
        }
    )


if __name__ == "__main__":
    main()
