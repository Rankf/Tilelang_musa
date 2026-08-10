"""Standalone TileLang application used to validate online JIT collection.

This intentionally does not use RuntimeWorkload, a collector factory, or a
hand-written instrumented kernel.  The application owns inputs and correctness;
the online hook only observes/augments the normal ``kernel.compile()`` call.
"""

from __future__ import annotations

import json

from musa_perf_collector.online import auto_install


def make_kernel(*, m: int = 128, n: int = 128, k: int = 64, threads: int = 256):
    """Build the real online GEMM+reduce kernel without allocating tensors."""
    import tilelang
    import tilelang.language as T

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
            T.reduce_sum(acc, out_local, dim=1, clear=True)
            T.copy(out_local, out)
        return out

    return kernel


def build_prim_func(config: dict[str, object]):
    """Offline factory for the exact PrimFunc used by :func:`main`."""
    return make_kernel(
        m=int(config.get("M", 128)),
        n=int(config.get("N", 128)),
        k=int(config.get("K", 64)),
        threads=int(config.get("threads", 256)),
    ).get_tir()


def main() -> None:
    import torch

    m, n, k, threads = 128, 128, 64, 256
    kernel = make_kernel(m=m, n=n, k=k, threads=threads)

    torch.manual_seed(20260729)
    torch.musa.manual_seed_all(20260729)
    a = torch.randn((m, k), device="musa", dtype=torch.float16)
    b = torch.randn((n, k), device="musa", dtype=torch.float16)
    compiled = kernel.compile()
    outputs = [compiled(a, b) for _ in range(3)]
    torch.musa.synchronize()
    reference = torch.matmul(a.float(), b.float().transpose(0, 1)).sum(dim=1)
    for output in outputs:
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    session = auto_install.current_session()
    if session is not None:
        session.report_correctness(status="pass", source="application_assert_close")
        session.flush()
        print(json.dumps({
            "correctness": "application_reported_pass",
            "online_session": str(session.directory),
            "kernel_mode": compiled.online_instrumentation["mode"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
