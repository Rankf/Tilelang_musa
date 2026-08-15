"""MCU runner for one cold or sixth warm S5000 GEMM launch."""

import argparse
import json
import os

os.environ.setdefault("MUSA_HOME", "/usr/local/musa")

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T


M = N = K = 4096
BM = BN = 128
BK = 64
NW = 4
NS = 2


@T.prim_func
def gemm_4096(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=NW * 32) as (
        bx,
        by,
    ):
        a_shared = T.alloc_shared((BM, BK), "float16")
        b_shared = T.alloc_shared((BK, BN), "float16")
        c_local = T.alloc_fragment((BM, BN), "float32")
        T.clear(c_local)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=NS):
            T.copy(A[by * BM, k * BK], a_shared)
            T.copy(B[k * BK, bx * BN], b_shared)
            T.gemm(a_shared, b_shared, c_local)
        T.copy(c_local, C[by * BM, bx * BN])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = tilelang.compile(gemm_4096)
    a = torch.randn(M, K, dtype=torch.float16, device="musa")
    b = torch.randn(K, N, dtype=torch.float16, device="musa")
    c = torch.empty(M, N, dtype=torch.float16, device="musa")
    torch.musa.synchronize()

    if args.mode == "cold":
        # S5000 L2 is 60 MiB.  Copy 128 MiB of host-backed data to the device,
        # then read the full buffer so the target starts behind a >2x-L2 sweep.
        host_evict = torch.ones(64 * 1024 * 1024, dtype=torch.float16)
        device_evict = host_evict.to("musa")
        torch.musa.synchronize()
        eviction_checksum = device_evict.sum()
        torch.musa.synchronize()
        del host_evict
        launches = 1
    else:
        eviction_checksum = None
        launches = 6

    print(json.dumps({
        "mode": args.mode,
        "target_launches": launches,
        "mcu_expected_skip": 0 if args.mode == "cold" else 5,
        "l2_capacity_bytes": 60 * 1024 * 1024,
        "cold_eviction_read_bytes": (
            64 * 1024 * 1024 * 2 if args.mode == "cold" else 0
        ),
    }))

    for _ in range(launches):
        kernel(a, b, c)
    torch.musa.synchronize()

    # Keep the eviction allocation and reduction live through the target.
    if eviction_checksum is not None:
        _ = eviction_checksum


if __name__ == "__main__":
    main()
