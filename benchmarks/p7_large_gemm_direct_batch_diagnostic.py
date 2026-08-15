#!/usr/bin/env python3
"""Graph-free device-event batch timing for P7 recovery diagnostics."""
import argparse, hashlib, json, pathlib, statistics, subprocess, traceback
import torch
import torch_musa  # noqa: F401
import tilelang

from p2_tilelang_compute_suite import PASS_CONFIGS, inputs, make_gemm, percentile, profiler_us

ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=pathlib.Path,required=True)
    ap.add_argument("--trial",type=int,required=True)
    ap.add_argument("--batch-unroll",type=int)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args(); cfg=json.loads(args.config.read_text()); c=cfg["case"]
    batch_unroll=args.batch_unroll if args.batch_unroll is not None else cfg["batch_unroll"]
    row={"case_id":c["id"],"trial":args.trial,"compile":"not_run","correctness":"not_run","status":"runtime_failed","failure_stage":"","error":"","timing_method":"graph-free MUSA event batch"}
    vals=[]
    try:
        tilelang.disable_cache(); torch.manual_seed(20260729)
        kernel=tilelang.compile(make_gemm(c),out_idx=[],target="musa",pass_configs=PASS_CONFIGS)
        row["compile"]="pass"; source=kernel.get_kernel_source(); argv,expected=inputs(c); fn=lambda:kernel(*argv)
        functions=list(kernel.adapter.device_mod.functions.values())
        if len(functions)==1:
            attrs=functions[0].attrs; row["dynamic_shared_bytes"]=int(attrs.get("dyn_shared_memory_buf",0)); row["kernel_global_symbol"]=str(attrs.get("global_symbol","unknown"))
        fn(); torch.musa.synchronize(); torch.testing.assert_close(argv[-1],expected,rtol=2e-2,atol=2e-2); row["correctness"]="pass"
        for _ in range(cfg["warmup"]): fn()
        torch.musa.synchronize()
        for _ in range(cfg["samples"]):
            start=torch.musa.Event(enable_timing=True); end=torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(batch_unroll): fn()
            end.record(); torch.musa.synchronize(); vals.append(start.elapsed_time(end)*1000/batch_unroll)
        profiler,events=profiler_us(fn,cfg["profiler_repeats"])
        med=statistics.median(vals); p10=percentile(vals,.1); p90=percentile(vals,.9); mx=max(vals); ratio=abs(med/profiler-1)
        stable=(p90-p10)/med<=cfg["stability"]["max_relative_span"] and mx/med<=cfg["stability"]["max_over_median"] and ratio<=cfg["stability"]["max_batch_profiler_ratio_delta"]
        row.update(status="pass" if stable else "stability_failed",failure_stage="" if stable else "stability_or_profiler_gate",batch_unroll=batch_unroll,samples=vals,latency_us_median=med,latency_us_p10=p10,latency_us_p90=p90,max_latency_us=mx,relative_span=(p90-p10)/med,max_over_median=mx/med,profiler_self_device_us=profiler,batch_profiler_ratio_delta=ratio,effective_tops=(2*c["M"]*c["N"]*c["K"])/(med*1e6),source_sha256=hashlib.sha256(source.encode()).hexdigest(),profiler_events=events)
    except Exception:
        row["batch_unroll"]=batch_unroll; row["samples_completed"]=len(vals)
        row["failure_stage"]="compile" if row["compile"]!="pass" else "correctness" if row["correctness"]!="pass" else "direct_batch_timing_or_profiler"
        row["error"]=traceback.format_exc(limit=6).replace("\n"," | ")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(row,indent=2)+"\n"); print(json.dumps(row),flush=True)

if __name__=="__main__": main()
