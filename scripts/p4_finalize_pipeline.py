#!/usr/bin/env python3
import csv, hashlib, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/p4_tilelang_pipeline.json"
OUT = ROOT / "results/processed/p4_occupancy_wave"

def main():
    cfg = json.loads(CFG.read_text())
    sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    candidates = []
    for root in (ROOT / "results/raw/p4_tilelang_pipeline").glob("*"):
        ps = [root / f"process_{i}" for i in (1, 2, 3)]
        if all((p / "summary.csv").exists() and (p / "metadata.json").exists() for p in ps) and all(json.loads((p / "metadata.json").read_text())["config_sha256"] == sha for p in ps):
            candidates.append(root)
    if not candidates: raise RuntimeError("need a complete three-process pipeline run")
    root = sorted(candidates)[-1]
    tables = []
    for i in (1, 2, 3):
        with (root / f"process_{i}/summary.csv").open() as f: tables.append({r["case_id"]: r for r in csv.DictReader(f)})
    rows = []
    for case in cfg["cases"]:
        source = [t[case["id"]] for t in tables]
        times = [float(r["latency_us_median"]) for r in source]
        tops = [float(r["effective_tops"] or 0) for r in source]
        mid = statistics.median(times); cross = (max(times)-min(times))/mid
        dynamic_shared = {r["dynamic_shared_bytes"] for r in source}
        symbols = {r["kernel_global_symbol"] for r in source}
        launches = {r["generated_launch_threads"] for r in source}
        if len(dynamic_shared) != 1 or len(symbols) != 1 or len(launches) != 1:
            raise RuntimeError(f"lowered device attributes disagree for {case['id']}")
        rows.append({"case_id":case["id"],"num_stages":case["num_stages"],"threads":int(source[0]["generated_launch_threads"]),"run_statuses":"|".join(r["status"] for r in source),"correctness_all_pass":str(all(r["correctness"]=="pass" for r in source)).lower(),"latency_us_run1":times[0],"latency_us_run2":times[1],"latency_us_run3":times[2],"latency_us_cross_run_median":mid,"cross_run_relative_span":cross,"effective_tops_cross_run_median":statistics.median(tops),"generated_launch_threads":source[0]["generated_launch_threads"],"dynamic_shared_bytes":int(source[0]["dynamic_shared_bytes"]),"kernel_global_symbol":source[0]["kernel_global_symbol"],"source_sha256":source[0]["source_sha256"],"parameter_eligible":str(all(r["status"]=="measured" and r["correctness"]=="pass" for r in source) and cross <= .20).lower()})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "aggregate_tilelang_pipeline.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "aggregate_tilelang_pipeline.json").write_text(json.dumps({"run":str(root),"config_sha256":sha,"cases":rows},indent=2)+"\n")
    print(json.dumps({"run":str(root),"cases":rows},indent=2))
if __name__ == "__main__": main()
