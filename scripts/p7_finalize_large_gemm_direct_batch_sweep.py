#!/usr/bin/env python3
import csv, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/"results/processed/p7_large_gemm_direct_batch_sweep"
def main():
    choices=[]
    for run in (ROOT/"results/raw/p7_large_gemm_direct_batch_sweep").glob("*"):
        cfg_path=run/"p7_large_gemm_direct_batch_sweep.json"
        if not cfg_path.exists(): continue
        cfg=json.loads(cfg_path.read_text()); expected=[run/f"unroll_{u}"/f"trial_{t}"/"result.json" for u in cfg["batch_unrolls"] for t in range(1,cfg["trials_per_unroll"]+1)]
        if all(p.exists() for p in expected): choices.append((run,cfg))
    if not choices: raise RuntimeError("no complete direct-batch sweep")
    run,cfg=sorted(choices)[-1]; summary=[]
    for u in cfg["batch_unrolls"]:
        rows=[json.loads((run/f"unroll_{u}"/f"trial_{t}"/"result.json").read_text()) for t in range(1,cfg["trials_per_unroll"]+1)]; good=[r for r in rows if r["status"]=="pass"]
        summary.append({"batch_unroll":u,"trials":len(rows),"passed":len(good),"failed":len(rows)-len(good),"failure_stages":"|".join(r["failure_stage"] for r in rows if r["status"]!="pass"),"samples_completed_on_failures":"|".join(str(r.get("samples_completed","")) for r in rows if r["status"]!="pass"),"latency_us_median_successes":statistics.median(r["latency_us_median"] for r in good) if good else ""})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    result={"run":str(run),"config":cfg,"summary":summary,"parameter_status":"diagnostic only; no P5/P6 model input"}; (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
