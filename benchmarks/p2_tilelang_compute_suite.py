#!/usr/bin/env python3
"""P2 TileLang compute suite with graph-replay timing and profiler audit."""

import argparse, csv, hashlib, json, pathlib, re, statistics, subprocess, traceback
import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]
PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True}


def percentile(v, q):
    v = sorted(v); x = (len(v)-1)*q; lo = int(x); hi = min(lo+1, len(v)-1)
    return v[lo] + (v[hi]-v[lo])*(x-lo)


def make_gemm(c):
    M,N,K,bm,bn,bk = (c[k] for k in ("M","N","K","block_M","block_N","block_K"))
    dtype, accum = c["dtype"], c["accum_dtype"]
    policy = {"default":None,"m":GemmWarpPolicy.FullRow,"n":GemmWarpPolicy.FullCol,"square":GemmWarpPolicy.Square}[c["policy"]]
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype), B:T.Tensor((K,N),dtype), C:T.Tensor((M,N),accum if dtype=="int8" else dtype)):
        with T.Kernel(T.ceildiv(N,bn),T.ceildiv(M,bm),threads=c["num_warp"]*32) as (bx,by):
            a=T.alloc_shared((bm,bk),dtype); b=T.alloc_shared((bk,bn),dtype); out=T.alloc_fragment((bm,bn),accum)
            T.clear(out)
            for k in T.Pipelined(T.ceildiv(K,bk),num_stages=c["num_stages"]):
                T.copy(A[by*bm,k*bk],a); T.copy(B[k*bk,bx*bn],b)
                if policy is None: T.gemm(a,b,out)
                else: T.gemm(a,b,out,policy=policy)
            T.copy(out,C[by*bm,bx*bn])
    return main


def make_vector(c):
    N,threads,vec,dtype,op = c["N"],c["threads"],c["vector"],c["dtype"],c["op"]
    @T.prim_func
    def binary(A:T.Tensor((N,),dtype), B:T.Tensor((N,),dtype), C:T.Tensor((N,),dtype)):
        with T.Kernel(T.ceildiv(N,threads*vec),threads=threads) as bx:
            for i,j in T.Parallel(threads,vec):
                p=(bx*threads+i)*vec+j; C[p]=A[p]+B[p]
    @T.prim_func
    def unary(A:T.Tensor((N,),dtype), C:T.Tensor((N,),dtype)):
        with T.Kernel(T.ceildiv(N,threads*vec),threads=threads) as bx:
            for i,j in T.Parallel(threads,vec):
                p=(bx*threads+i)*vec+j
                if op=="exp": C[p]=T.exp(A[p])
                else: C[p]=T.rsqrt(A[p])
    @T.prim_func
    def fused_exp_add(A:T.Tensor((N,),dtype), B:T.Tensor((N,),dtype), C:T.Tensor((N,),dtype)):
        with T.Kernel(T.ceildiv(N,threads*vec),threads=threads) as bx:
            for i,j in T.Parallel(threads,vec):
                p=(bx*threads+i)*vec+j; C[p]=T.exp(A[p])+B[p]
    if op=="vector_add": return binary
    if op=="fused_exp_add": return fused_exp_add
    return unary


def make_reduce(c):
    M,K,dtype,threads=c["M"],c["K"],c["dtype"],c["threads"]
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype), C:T.Tensor((M,),dtype)):
        with T.Kernel(M,threads=threads) as bx:
            x=T.alloc_fragment((1,K),dtype); y=T.alloc_fragment((1,),dtype)
            T.copy(A[bx,0],x); T.reduce_sum(x,y,dim=1,clear=True); C[bx]=y[0]
    return main


def inputs(c):
    op=c["op"]
    if op=="gemm":
        td={"float16":torch.float16,"bfloat16":torch.bfloat16,"int8":torch.int8}[c["dtype"]]
        if td==torch.int8:
            a=torch.randint(-4,5,(c["M"],c["K"]),device="musa",dtype=td); b=torch.randint(-4,5,(c["K"],c["N"]),device="musa",dtype=td); out=torch.empty((c["M"],c["N"]),device="musa",dtype=torch.int32)
            expected=(a.cpu().to(torch.int32)@b.cpu().to(torch.int32)).to("musa")
        else:
            a=torch.randn((c["M"],c["K"]),device="musa",dtype=td); b=torch.randn((c["K"],c["N"]),device="musa",dtype=td); out=torch.empty((c["M"],c["N"]),device="musa",dtype=td)
            expected=(a.float()@b.float()).to(td)
        return [a,b,out],expected
    if op=="vector_add":
        a=torch.randn(c["N"],device="musa"); b=torch.randn(c["N"],device="musa"); out=torch.empty_like(a); return [a,b,out],a+b
    if op=="fused_exp_add":
        a=torch.rand(c["N"],device="musa")+.25; b=torch.randn(c["N"],device="musa"); out=torch.empty_like(a); return [a,b,out],torch.exp(a)+b
    if op in ("exp","rsqrt"):
        a=torch.rand(c["N"],device="musa")+.25; out=torch.empty_like(a); return [a,out],torch.exp(a) if op=="exp" else torch.rsqrt(a)
    a=torch.randn((c["M"],c["K"]),device="musa"); out=torch.empty(c["M"],device="musa"); return [a,out],a.sum(dim=1)


def graph_samples(fn, unroll, count):
    graph=torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        for _ in range(unroll): fn()
    torch.musa.synchronize(); starts=[torch.musa.Event(enable_timing=True) for _ in range(count)]; ends=[torch.musa.Event(enable_timing=True) for _ in range(count)]
    for s,e in zip(starts,ends): s.record(); graph.replay(); e.record()
    torch.musa.synchronize(); return [s.elapsed_time(e)*1000/unroll for s,e in zip(starts,ends)]


def profiler_us(fn,repeats):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.MUSA]) as p:
        for _ in range(repeats): fn()
    torch.musa.synchronize(); rows=[]
    for e in p.key_averages():
        if e.self_device_time_total>0: rows.append((e.key,e.count,float(e.self_device_time_total)))
    return sum(x[2] for x in rows if "KernelFill<" not in x[0])/repeats,rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=pathlib.Path,required=True); ap.add_argument("--output-dir",type=pathlib.Path,required=True); args=ap.parse_args()
    cfg=json.loads(args.config.read_text()); args.output_dir.mkdir(parents=True,exist_ok=True); tilelang.disable_cache(); torch.manual_seed(20260729)
    summary=[]; raw=[]; prows=[]
    for c in cfg["cases"]:
        row={"case_id":c["id"],"op":c["op"],"dtype":c["dtype"],"status":"compile_failed","correctness":"not_run","error":""}
        compiled = False
        try:
            program=make_gemm(c) if c["op"]=="gemm" else make_reduce(c) if c["op"]=="reduce_sum" else make_vector(c)
            kernel=tilelang.compile(program,out_idx=[],target="musa",pass_configs=PASS_CONFIGS,compile_flags=cfg.get("compile_flags")); compiled = True; source=kernel.get_kernel_source(); argv,expected=inputs(c); fn=lambda:kernel(*argv)
            device_funcs=list(kernel.adapter.device_mod.functions.values())
            if len(device_funcs)==1:
                attrs=device_funcs[0].attrs
                row["dynamic_shared_bytes"]=int(attrs.get("dyn_shared_memory_buf",0))
                row["kernel_global_symbol"]=str(attrs.get("global_symbol","unknown"))
            fn(); torch.musa.synchronize()
            if c["dtype"]=="int8": torch.testing.assert_close(argv[-1],expected,rtol=0,atol=0)
            else: torch.testing.assert_close(argv[-1],expected,rtol=2e-2,atol=2e-2)
            row["correctness"]="pass"
            for _ in range(cfg["warmup"]): fn()
            torch.musa.synchronize(); vals=graph_samples(fn,cfg["graph_unroll"],cfg["graph_samples"]); pus,events=profiler_us(fn,cfg["profiler_repeats"])
            med=statistics.median(vals); p10=percentile(vals,.1); p90=percentile(vals,.9); mx=max(vals); ratio=abs(med/pus-1)
            stable=(p90-p10)/med<=cfg["stability"]["max_relative_span"] and mx/med<=cfg["stability"]["max_over_median"] and ratio<=cfg["stability"]["max_graph_profiler_ratio_delta"]
            row.update(status="measured" if stable else "runtime_failed",latency_us_median=med,latency_us_p10=p10,latency_us_p90=p90,max_latency_us=mx,relative_span=(p90-p10)/med,max_over_median=mx/med,profiler_self_device_us=pus,graph_profiler_ratio_delta=ratio,source_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_launch_threads=(re.search(r"__launch_bounds__\((\d+)",source).group(1) if re.search(r"__launch_bounds__\((\d+)",source) else "unknown"),timing_method="MUSAGraph replay, 100 captured kernels per replay; profiler self-device cross-check",allocation_mode="preallocated_output")
            if c["op"]=="gemm" and stable: row["effective_tops"]=(2*c["M"]*c["N"]*c["K"])/(med*1e6)
            elif c["op"]!="gemm" and stable: row["elements_per_second"]=((c.get("N") or c["M"]*c["K"])/(med*1e-6))
            for i,v in enumerate(vals): raw.append({"case_id":c["id"],"sample_index":i,"latency_us":v})
            for key,count,total in events: prows.append({"case_id":c["id"],"key":key,"count":count,"self_device_us":total})
            if not stable: row["error"]="stability or graph/profiler agreement gate failed"
        except Exception:
            if compiled:
                row["status"] = "runtime_failed"
            row["error"]=traceback.format_exc(limit=5).replace("\n"," | ")
        summary.append(row); print(json.dumps(row),flush=True)
    fields=sorted({k for r in summary for k in r});
    with (args.output_dir/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    with (args.output_dir/"raw_samples.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["case_id","sample_index","latency_us"]); w.writeheader(); w.writerows(raw)
    with (args.output_dir/"profiler_events.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["case_id","key","count","self_device_us"]); w.writeheader(); w.writerows(prows)
    (args.output_dir/"metadata.json").write_text(json.dumps({"config":cfg,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"harness_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"tilelang_commit":subprocess.check_output(["git","-C",str(ROOT/"tilelang-musa"),"rev-parse","HEAD"],text=True).strip(),"tilelang_local_patch_sha256":hashlib.sha256((ROOT/"logs/env/tilelang_musa_local.patch").read_bytes()).hexdigest(),"musa_home":__import__("os").environ.get("MUSA_HOME","unknown"),"mcc_version":next(x for x in subprocess.check_output(["mcc","--version"],text=True).splitlines() if x.startswith("mcc version")),"musa_capability":torch.musa.get_device_capability()},indent=2)+"\n")

if __name__=="__main__": main()
