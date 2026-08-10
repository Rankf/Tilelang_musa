#!/usr/bin/env python3
"""S5000 算子性能分析工具

用法:
  python3 analyze_op.py gemm --M 1024 --N 1024 --K 1024 --arch S5000 -o report.json
  python3 analyze_op.py relu --N 16777216 --arch S5000 -o report.json
"""
import os, sys, json, argparse, importlib

os.environ.setdefault("MUSA_HOME", "/usr/local/musa")
BASE = "/root/s5000-calibration"
sys.path = [BASE, BASE+"/TileOPs", BASE+"/DeepStack-ae-source/src/tilesight",
            BASE+"/DeepStack-ae/src"] + sys.path

# ── S5000 架构（使用校准后的 ConfigurableStackedGpu 配置）──
from deepstack.mosaic.arch import make_s5000_config, make_custom_stacked_gpu

def analyze_gemm(M, N, K, bm, bn, bk, nw, ns, arch):
    from tilesight.fused_op_pipeline_wave.matmul_pipeline_wave import calculate_matmul_pipeline_wave
    mem_levels = {"in1": [1,1,1,2], "in2": [0,1,1,2], "out1": [0,0,1,4]}
    r = calculate_matmul_pipeline_wave((M,N,K), (bm,bn,bk), (bm//nw,bn//nw,bk), ns, arch, mem_levels, mma_type="wmma")
    return {
        "op": "gemm", "shape": [M,N,K], "tile": [bm,bn,bk], "warp": nw, "stage": ns,
        "latency_us": round(r.total_latency*1e6, 2),
        "bandwidth_gbps": round(2*M*N*K/r.total_latency/1e9, 1),
        "ddr_util_pct": round(r.ddr_util*100, 1),
        "l2_util_pct": round(r.l2_util*100, 1),
        "smem_util_pct": round(r.smem_util*100, 1),
        "compute_util_pct": round(r.compute_util*100, 1),
        "l2_hit_rate_pct": round(r.l2_hit_rate*100, 1),
        "tiles_per_sm": r.tiles_per_sm,
        "waves": round(r.waves, 1),
        "pipeline": {
            "prologue_us": round(r.pipeline_detail.prologue_time*1e6, 2) if r.pipeline_detail else 0,
            "steady_per_iter_us": round(r.pipeline_detail.steady_time_per_iter*1e6, 2) if r.pipeline_detail else 0,
            "epilogue_us": round(r.pipeline_detail.epilogue_time*1e6, 2) if r.pipeline_detail else 0,
        }
    }

def analyze_relu(N, threads, arch):
    from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import calculate_N_0_elementwise_pipeline_wave
    mem_levels = {"in1": [1,0,0,4], "out1": [0,0,1,4]}
    r = calculate_N_0_elementwise_pipeline_wave((N,), (threads*4,), (threads,), arch, mem_levels)
    return {
        "op": "relu", "N": N, "threads": threads,
        "latency_us": round(r.total_latency*1e6, 2),
        "bandwidth_gbps": round(N*4*2/r.total_latency/1e9, 1),
        "ddr_util_pct": round(r.ddr_util*100, 1),
        "l2_util_pct": round(r.l2_util*100, 1),
        "smem_util_pct": round(r.smem_util*100, 1),
        "compute_util_pct": round(r.compute_util*100, 1),
        "l2_hit_rate_pct": round(r.l2_hit_rate*100, 1),
        "tiles_per_sm": r.tiles_per_sm,
        "waves": round(r.waves, 1),
    }

def analyze_add(N, threads, arch):
    from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import calculate_N_N_elementwise_pipeline_wave
    mem_levels = {"in1": [1,0,0,4], "in2": [1,0,0,4], "out1": [0,0,1,4]}
    r = calculate_N_N_elementwise_pipeline_wave((N,), (threads*4,), (threads,), arch, mem_levels)
    return {
        "op": "add", "N": N, "threads": threads,
        "latency_us": round(r.total_latency*1e6, 2),
        "bandwidth_gbps": round(N*4*3/r.total_latency/1e9, 1),  # 2读+1写
        "ddr_util_pct": round(r.ddr_util*100, 1),
        "l2_util_pct": round(r.l2_util*100, 1),
        "smem_util_pct": round(r.smem_util*100, 1),
        "compute_util_pct": round(r.compute_util*100, 1),
        "l2_hit_rate_pct": round(r.l2_hit_rate*100, 1),
        "tiles_per_sm": r.tiles_per_sm,
        "waves": round(r.waves, 1),
    }

def main():
    parser = argparse.ArgumentParser(description="S5000 算子性能分析工具")
    parser.add_argument("op", choices=["gemm", "relu", "add"], help="算子类型")
    parser.add_argument("-o", "--output", default="", help="输出文件 (默认打印到终端)")
    parser.add_argument("--arch", default="S5000", help="架构 (默认 S5000)")

    # GEMM 参数
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--bm", type=int, default=128, help="tile M")
    parser.add_argument("--bn", type=int, default=128, help="tile N")
    parser.add_argument("--bk", type=int, default=64, help="tile K")
    parser.add_argument("--nw", type=int, default=4, help="warp 数")
    parser.add_argument("--ns", type=int, default=2, help="pipeline stage")
    # Elementwise 参数
    parser.add_argument("--threads", type=int, default=256, help="线程数")

    args = parser.parse_args()
    arch = make_custom_stacked_gpu(make_s5000_config())

    if args.op == "gemm":
        report = analyze_gemm(args.M, args.N, args.K, args.bm, args.bn, args.bk, args.nw, args.ns, arch)
    elif args.op == "relu":
        report = analyze_relu(args.N, args.threads, arch)
    elif args.op == "add":
        report = analyze_add(args.N, args.threads, arch)

    # 添加元信息
    report["arch"] = args.arch
    report["core_freq_ghz"] = arch.core_freq / 1e9
    report["sm_count"] = arch.sm_count

    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"报告已保存到: {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()