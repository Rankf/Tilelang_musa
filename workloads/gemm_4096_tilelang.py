"""4096^3 FP16 GEMM used by the 5.1 metadata-to-model example."""

import tilelang.language as T


def make_prim_func():
    M = N = K = 4096
    BM, BN, BK = 128, 128, 64
    warps, stages = 4, 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM),
                      threads=warps * 32) as (bx, by):
            a_shared = T.alloc_shared((BM, BK), "float16")
            b_shared = T.alloc_shared((BK, BN), "float16")
            c_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(c_local)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM, k * BK], a_shared)
                T.copy(B[k * BK, bx * BN], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, C[by * BM, bx * BN])

    return main
