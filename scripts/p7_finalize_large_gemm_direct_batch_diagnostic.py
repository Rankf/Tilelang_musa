#!/usr/bin/env python3
import argparse, csv, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-root",type=pathlib.Path,default=ROOT/"results/raw/p7_large_gemm_direct_batch")
    ap.add_argument("--processed-root",type=pathlib.Path,default=ROOT/"results/processed/p7_large_gemm_direct_batch")
    args=ap.parse_args(); raw_root=args.raw_root.resolve(); out=args.processed_root.resolve(); candidates=[]
    for run in raw_root.glob("*"):
        cfg=run/"p7_large_gemm_direct_batch_diagnostic.json"
        if cfg.exists():
            data=json.loads(cfg.read_text()); paths=[run/f"trial_{i}"/"result.json" for i in range(1,data["trials"]+1)]
            if all(p.exists() for p in paths): candidates.append((run,data,paths))
    if not candidates: raise RuntimeError("no complete direct-batch P7 run")
    run,cfg,paths=sorted(candidates)[-1]; rows=[json.loads(p.read_text()) for p in paths]; passed=[r for r in rows if r["status"]=="pass"]
    result={"run":str(run),"config":cfg,"trials":len(rows),"passed":len(passed),"failed":len(rows)-len(passed),"median_latency_us_of_passes":statistics.median(r["latency_us_median"] for r in passed) if passed else None,"parameter_status":"diagnostic only; no P5/P6 model input","rows":rows}
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="") as f:
        fields=["trial","status","compile","correctness","failure_stage","latency_us_median","latency_us_p10","latency_us_p90","max_latency_us","relative_span","max_over_median","profiler_self_device_us","batch_profiler_ratio_delta","effective_tops","source_sha256","error"]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
