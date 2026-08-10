#!/usr/bin/env python3
"""Aggregate the three independent P2 compute-suite processes."""
import csv, datetime, glob, hashlib, json, os, pathlib, statistics, subprocess

root=pathlib.Path(__file__).resolve().parents[1]
runs=[pathlib.Path(x) for x in sorted(glob.glob(str(root/"results/raw/p2_compute/*")))[-3:]]
tables=[]
for run in runs:
    with (run/"summary.csv").open() as f: tables.append({r["case_id"]:r for r in csv.DictReader(f)})
config=json.loads((root/"configs/p2_compute_suite.json").read_text())
case_config={c["id"]:c for c in config["cases"]}
rows=[]
for cid in tables[0]:
    rs=[t[cid] for t in tables]; medians=[float(r["latency_us_median"]) for r in rs]
    cross=(max(medians)-min(medians))/statistics.median(medians)
    all_correct=all(r["correctness"]=="pass" for r in rs); all_measured=all(r["status"]=="measured" for r in rs)
    eligible=all_correct and all_measured and cross<=.15
    rows.append({"case_id":cid,"op":rs[0]["op"],"dtype":rs[0]["dtype"],"case_config_json":json.dumps(case_config[cid],sort_keys=True,separators=(",",":")),"run_statuses":"|".join(r["status"] for r in rs),"correctness_all_pass":str(all_correct).lower(),"graph_latency_us_run1":medians[0],"graph_latency_us_run2":medians[1],"graph_latency_us_run3":medians[2],"graph_latency_us_cross_run_median":statistics.median(medians),"cross_run_relative_span":cross,"profiler_self_device_us_cross_run_median":statistics.median(float(r["profiler_self_device_us"]) for r in rs),"p2_parameter_eligible":str(eligible).lower(),"disposition":"eligible" if eligible else "measured_but_pending_short_kernel_method" if all_correct else "incorrect","source_sha256":rs[0]["source_sha256"]})
out=root/"results/processed/p2_compute"; out.mkdir(parents=True,exist_ok=True)
with (out/"aggregate.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(out/"aggregate.json").write_text(json.dumps({"runs":[str(x) for x in runs],"policy":{"all_three_correct":True,"all_three_status_measured":True,"cross_run_relative_span_max":.15},"cases":rows},indent=2)+"\n")
(out/"formal_run_manifest.json").write_text(json.dumps({"generated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"measurement_dates_utc":[x.name for x in runs],"reproduce_command":"scripts/p2_run_compute_suite.sh","config":"configs/p2_compute_suite.json","config_sha256":hashlib.sha256((root/"configs/p2_compute_suite.json").read_bytes()).hexdigest(),"harness":"benchmarks/p2_tilelang_compute_suite.py","measured_run_harness_sha256":[json.loads((x/"metadata.json").read_text())["harness_sha256"] for x in runs],"current_reproduction_harness_sha256":hashlib.sha256((root/"benchmarks/p2_tilelang_compute_suite.py").read_bytes()).hexdigest(),"post_run_harness_change":"metadata-only: future runs additionally record mcc, MUSA_HOME and local patch hashes; kernel generation and timing logic unchanged","tilelang_commit":subprocess.check_output(["git","-C",str(root/"tilelang-musa"),"rev-parse","HEAD"],text=True).strip(),"tilelang_local_patch_sha256":hashlib.sha256((root/"logs/env/tilelang_musa_local.patch").read_bytes()).hexdigest(),"musa_home":os.environ.get("MUSA_HOME","/usr/local/musa-4.3.5"),"mcc_version":next(x for x in subprocess.check_output(["mcc","--version"],text=True).splitlines() if x.startswith("mcc version")),"device":"MTT S5000","musa_capability":[3,1],"timing_scope":"preallocated buffers; MUSAGraph 100-call capture/replay plus MUSA profiler self-device cross-check"},indent=2)+"\n")
print(json.dumps({"runs":[str(x) for x in runs],"eligible":sum(r["p2_parameter_eligible"]=="true" for r in rows),"pending":sum(r["p2_parameter_eligible"]!="true" for r in rows)}))
