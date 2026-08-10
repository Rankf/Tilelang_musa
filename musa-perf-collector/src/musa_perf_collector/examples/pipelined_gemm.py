"""A small TileLang factory that builds IR but never allocates or runs MUSA tensors."""

from __future__ import annotations

from typing import Any


def build_prim_func(config: dict[str, Any]):
    import tilelang
    import tilelang.language as T

    m = int(config.get("M", 128))
    n = int(config.get("N", 128))
    k = int(config.get("K", 128))
    block_m = int(config.get("block_M", 64))
    block_n = int(config.get("block_N", 64))
    block_k = int(config.get("block_K", 32))
    stages = int(config.get("num_stages", 2))
    dtype = str(config.get("dtype", "float16"))

    @tilelang.jit(target="musa")
    def kernel(A, B):
        A: T.Tensor[[m, k], dtype]
        B: T.Tensor[[k, n], dtype]
        C = T.empty((m, n), dtype)
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), dtype)
            b_shared = T.alloc_shared((block_k, block_n), dtype)
            c_local = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(c_local)
            for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=stages):
                T.copy(A[by * block_m, kk * block_k], a_shared)
                T.copy(B[kk * block_k, bx * block_n], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, C[by * block_m, bx * block_n])
        return C

    return kernel.get_tir()
