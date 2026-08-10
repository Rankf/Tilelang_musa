"""Dense GEMM kernel for the MUSA MP31 target.

This is deliberately a target-specific replacement for TileOPs' Hopper-only
``GemmKernel``.  It uses synchronous ``T.gemm`` so TileLang-MUSA selects the
MP31 GEMM implementation instead of forcing Hopper ``T.wgmma_gemm``.
"""

import functools
from typing import Callable, Optional

import torch
import tilelang
import tilelang.language as T

from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _mp31_dense_gemm_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
) -> Callable:
    if min(m, n, k) <= 0:
        raise ValueError(f"GEMM dimensions must be positive, got {(m, n, k)}")
    @tilelang.jit(out_idx=[2], compile_flags=["-O3"])
    def _kernel(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_warps: int = 4,
        num_stages: int = 1,
    ) -> Callable:
        if (block_m, block_n, block_k, num_warps, num_stages) != (64, 64, 64, 4, 1):
            raise ValueError(
                "The initial repaired kernel only supports the audited "
                "64x64x64, 4-warp, stage-1 MP31 schedule"
            )

        @T.prim_func
        def main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),
            c: T.Tensor((m, n), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=num_warps * 32,
            ) as (bx, by):
                a_shared = T.alloc_shared(
                    (block_k, block_m) if trans_a else (block_m, block_k), dtype
                )
                b_shared = T.alloc_shared(
                    (block_n, block_k) if trans_b else (block_k, block_n), dtype
                )
                c_local = T.alloc_fragment((block_m, block_n), "float32")

                T.clear(c_local)
                for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    if trans_a:
                        T.copy(a[ko * block_k, by * block_m], a_shared)
                    else:
                        T.copy(a[by * block_m, ko * block_k], a_shared)
                    if trans_b:
                        T.copy(b[bx * block_n, ko * block_k], b_shared)
                    else:
                        T.copy(b[ko * block_k, bx * block_n], b_shared)
                    T.gemm(
                        a_shared,
                        b_shared,
                        c_local,
                        transpose_A=trans_a,
                        transpose_B=trans_b,
                    )
                T.copy(c_local, c[by * block_m, bx * block_n])

        return main

    return _kernel


class MusaMP31DenseGemmKernel(Kernel):
    """TileOPs-compatible FP16/BF16 dense GEMM for MP31.

    The fixed schedule is intentionally reused across layouts while TileLang
    handles partial M/N/K tiles through its copy and GEMM lowering.  This is a
    portability repair, not a schedule search implementation.
    """

    supported_archs: list[int] = [31]

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "MP31 dense GEMM supports torch.float16 or torch.bfloat16, "
                f"got {dtype}"
            )
        if tune:
            raise ValueError("Autotuning is not enabled for the initial repaired kernel")

        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.kernel = _mp31_dense_gemm_kernel(
            m,
            n,
            k,
            self.dtype_str,
            trans_a,
            trans_b,
        )
        self.init_config(config=config, tune=False)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_warps": 4,
            "num_stages": 1,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.device.type != "musa" or b.device.type != "musa":
            raise ValueError(f"Expected MUSA inputs, got {a.device} and {b.device}")
        compiled = self.kernel(**self.config)
        return compiled(a, b)


# Retain the original import name for callers that only need FP16 NN.  New
# callers should use MusaMP31DenseGemmKernel.
MusaMP31Fp16NNDenseGemmKernel = MusaMP31DenseGemmKernel
