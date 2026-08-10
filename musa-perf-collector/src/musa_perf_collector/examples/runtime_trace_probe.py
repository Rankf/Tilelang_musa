"""Minimal MUSA `clock64` trace probe used to validate the deep-collection ABI."""

from __future__ import annotations


def build_kernel():
    import tilelang
    import tilelang.language as T

    tilelang.disable_cache()

    @tilelang.jit(target="musa")
    def trace_probe(A):
        A: T.Tensor[[16], T.float32]
        trace = T.empty((2,), "int64")
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                trace[0] = T.call_extern("int64", "clock64")
            A[tx % 16] = A[tx % 16] + T.float32(1)
            if tx == 0:
                trace[1] = T.call_extern("int64", "clock64")
        return trace

    return trace_probe
