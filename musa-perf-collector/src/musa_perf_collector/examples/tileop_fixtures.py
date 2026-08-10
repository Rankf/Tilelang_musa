"""Metadata-only TileLang factories for non-GEMM TileOp coverage tests."""

from __future__ import annotations


def build_cumsum_prim_func(config=None):
    """Build a shared-memory cumsum IR; callers must not lower/launch it here."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target="musa")
    def kernel(A):
        A: T.Tensor[[128], "float32"]
        B = T.empty((128,), "float32")
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            T.copy(A, shared)
            T.cumsum(shared, shared, dim=0)
            T.copy(shared, B)
        return B

    return kernel.get_tir()


def build_atomic_prim_func(config=None):
    """Build a tile-region atomic add IR for adapter verification only."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target="musa")
    def kernel(A):
        A: T.Tensor[[128], "float32"]
        B = T.empty((128,), "float32")
        with T.Kernel(1, threads=128):
            T.atomic_add(B, A)
        return B

    return kernel.get_tir()


def build_reduce_prim_func(config=None):
    """Build a shared-to-fragment reduction IR for structural adapter tests."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target="musa")
    def kernel(A):
        A: T.Tensor[[128], "float32"]
        B = T.empty((1,), "float32")
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            result = T.alloc_fragment((1,), "float32")
            T.copy(A, shared)
            T.reduce_sum(shared, result, dim=0)
            T.copy(result, B)
        return B

    return kernel.get_tir()


def build_gemm_reduce_prim_func(config=None):
    """Build the runtime GEMM+row-reduce TileOp sequence without execution."""
    import tilelang
    import tilelang.language as T

    config = config or {}
    m, n, k = (int(config.get(name, default)) for name, default in (("M", 128), ("N", 128), ("K", 64)))
    threads = int(config.get("threads", 256))
    reduction = str(config.get("reduction", "sum"))
    if reduction not in {"sum", "max"}:
        raise ValueError("reduction must be sum or max")

    @tilelang.jit(target="musa")
    def kernel(A, B):
        A: T.Tensor[[m, k], T.float16]
        B: T.Tensor[[n, k], T.float16]
        out = T.empty((m,), "float32")
        with T.Kernel(1, threads=threads) as _:
            a_shared = T.alloc_shared((m, k), "float16")
            b_shared = T.alloc_shared((n, k), "float16")
            acc = T.alloc_fragment((m, n), "float32")
            out_local = T.alloc_fragment((m,), "float32")
            T.clear(acc)
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(a_shared, b_shared, acc, transpose_B=True)
            if reduction == "sum":
                T.reduce_sum(acc, out_local, dim=1, clear=True)
            else:
                T.reduce_max(acc, out_local, dim=1, clear=True)
            T.copy(out_local, out)
        return out

    return kernel.get_tir()
