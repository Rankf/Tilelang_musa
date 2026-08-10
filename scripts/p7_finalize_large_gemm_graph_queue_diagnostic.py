#!/usr/bin/env python3
import csv, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed/p7_large_gemm_graph_queue"

def main():
    choices=[]
    for root in (ROOT / "results/raw/p7_large_gemm_graph_queue").glob("*"):
        cfg_path=root / "p7_large_gemm_graph_queue_diagnostic.json"
        if not cfg_path.exists(): continue
        cfg=json.loads(cfg_path.read_text())
        expected=[root/f"depth_{d}"/f"trial_{t}"/"result.json" for d in cfg["queue_depths"] for t in range(1,cfg["trials_per_depth"]+1)]
        if all(x.exists() for x in expected): choices.append((root,cfg))
    if not choices: raise RuntimeError("no complete queued-replay P7 run")
    root,cfg=sorted(choices)[-1]; OUT.mkdir(parents=True,exist_ok=True); summary=[]; raw=[]
    for d in cfg["queue_depths"]:
        rows=[json.loads((root/f"depth_{d}"/f"trial_{t}"/"result.json").read_text()) for t in range(1,cfg["trials_per_depth"]+1)]; raw += rows
        good=[x for x in rows if x["status"]=="pass"]; bad=[x["failure_stage"] for x in rows if x["status"]!="pass"]
        summary.append({"queue_depth":d,"trials":len(rows),"passed":len(good),"failed":len(bad),"failure_stages":"|".join(bad),"queued_replay_us_median_successes":statistics.median([x["queued_replay_us_median"] for x in good]) if good else ""})
    with (OUT/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (OUT/"summary.json").write_text(json.dumps({"run":str(root),"config":cfg,"summary":summary,"raw_rows":raw,"parameter_status":"diagnostic only"},indent=2)+"\n")
    print(json.dumps({"run":str(root),"summary":summary},indent=2))

if __name__=="__main__": main()
