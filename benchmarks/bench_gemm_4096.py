import torch, time, os
os.environ['MUSA_HOME'] = '/usr/local/musa'
import torch_musa
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 64
NW, NS = 4, 2

@T.prim_func
def gemm_4096(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=NW * 32) as (bx, by):
        a_shared = T.alloc_shared((BM, BK), "float16")
        b_shared = T.alloc_shared((BK, BN), "float16")
        c_local = T.alloc_fragment((BM, BN), "float32")
        T.clear(c_local)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=NS):
            T.copy(A[by * BM, k * BK], a_shared)
            T.copy(B[k * BK, bx * BN], b_shared)
            T.gemm(a_shared, b_shared, c_local)
        T.copy(c_local, C[by * BM, bx * BN])

print("Compiling kernel...")
kernel = tilelang.compile(gemm_4096)
print("Kernel compiled")

A = torch.randn(M, K, dtype=torch.float16, device='musa')
B = torch.randn(K, N, dtype=torch.float16, device='musa')
C = torch.empty(M, N, dtype=torch.float16, device='musa')

# 预热
for _ in range(5):
    kernel(A, B, C)

# 计时（10次取平均）
torch.musa.synchronize()
start = time.time()
for _ in range(10):
    kernel(A, B, C)
torch.musa.synchronize()
end = time.time()
avg_us = (end - start) / 10 * 1e6
tflops = 2 * M * N * K / avg_us / 1e6
print(f"\n实测延迟: {avg_us:.1f} us")
print(f"实测吞吐: {tflops:.1f} TFLOP/s")
print(f"\n=== DeepStack 预测 ===")
print(f"预测延迟: 967.3 us")
print(f"预测吞吐: 142085.0 GB/s")