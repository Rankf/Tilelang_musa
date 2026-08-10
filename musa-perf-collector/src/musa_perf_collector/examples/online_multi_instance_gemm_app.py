"""Real MUSA application for one-to-many compiler provenance validation.

The scalar index copy before the pipelined loop feeds both the producer-side
TMA coordinate and the consumer-side output coordinate.  Warp specialization
therefore has to materialize one branch-private lowered instance in each
branch while retaining the source TileOp provenance ordinal.
"""

from __future__ import annotations

import json

from musa_perf_collector.online import auto_install


def make_kernel():
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target="musa")
    def kernel(A, B, offset):
        A: T.Tensor[[128, 128], T.float16]
        B: T.Tensor[[128, 128], T.float16]
        offset: T.Tensor[[1], T.int32]
        C = T.empty((128, 128), T.float16)
        with T.Kernel(2, 2, threads=256) as (bx, by):
            a_shared = T.alloc_shared((64, 32), T.float16)
            b_shared = T.alloc_shared((32, 64), T.float16)
            acc = T.alloc_fragment((64, 64), T.float32)
            offset_local = T.alloc_local((1,), T.int32)
            T.copy(offset, offset_local)
            T.clear(acc)
            for kk in T.Pipelined(4, num_stages=2):
                T.copy(A[(by + offset_local[0]) * 64, kk * 32], a_shared)
                T.copy(B[kk * 32, bx * 64], b_shared)
                T.gemm(a_shared, b_shared, acc)
            T.copy(acc, C[(by + offset_local[0]) * 64, bx * 64])
        return C

    return kernel


def build_prim_func(_config: dict[str, object]):
    """Offline factory for the exact fixed-shape PrimFunc used by :func:`main`.

    The offline CLI uniformly supplies a workload configuration.  This
    controlled example has no configurable shape parameters, but accepting
    the argument keeps its compiler-exact lineage artifact reproducible
    through that CLI.
    """
    return make_kernel().get_tir()


def main() -> None:
    import torch

    torch.manual_seed(20260730)
    torch.musa.manual_seed_all(20260730)
    a = torch.randn((128, 128), device="musa", dtype=torch.float16)
    b = torch.randn((128, 128), device="musa", dtype=torch.float16)
    offset = torch.zeros((1,), device="musa", dtype=torch.int32)
    compiled = make_kernel().compile()
    outputs = [compiled(a, b, offset) for _ in range(3)]
    torch.musa.synchronize()
    reference = torch.matmul(a.float(), b.float())
    for output in outputs:
        torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)
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
