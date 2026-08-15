"""Strict FP32 element-wise PrimFuncs for compiler-info integration tests."""

import tilelang.language as T


def make_add_prim_func(n=16384, threads=256, elements_per_thread=8):
    """Return C[i] = A[i] + B[i] with exact one-dimensional coverage."""
    if n <= 0 or threads <= 0 or elements_per_thread <= 0:
        raise ValueError("shape and launch parameters must be positive")
    elements_per_block = threads * elements_per_thread
    if n % elements_per_block:
        raise ValueError("n must be divisible by threads * elements_per_thread")

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(n // elements_per_block, threads=threads) as bx:
            for tx, local in T.Parallel(threads, elements_per_thread):
                index = (bx * threads + tx) * elements_per_thread + local
                C[index] = A[index] + B[index]

    return main


def make_relu_prim_func(n=16384, threads=256, elements_per_thread=4):
    """Return B[i] = max(A[i], 0) with exact one-dimensional coverage."""
    if n <= 0 or threads <= 0 or elements_per_thread <= 0:
        raise ValueError("shape and launch parameters must be positive")
    elements_per_block = threads * elements_per_thread
    if n % elements_per_block:
        raise ValueError("n must be divisible by threads * elements_per_thread")

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(n // elements_per_block, threads=threads) as bx:
            for tx, local in T.Parallel(threads, elements_per_thread):
                index = (bx * threads + tx) * elements_per_thread + local
                B[index] = T.max(A[index], 0.0)

    return main
