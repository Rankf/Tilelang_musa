#!/usr/bin/env python3
"""Summarize only a complete P7 exact-P2 single-case reproduction run."""
import argparse, csv, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]

def read_row(path):
    with path.open(newline="") as f:
        return next(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=pathlib.Path, default=ROOT / "results/raw/p7_p2_exact_single_gemm")
    ap.add_argument("--processed-root", type=pathlib.Path, default=ROOT / "results/processed/p7_p2_exact_single_gemm")
    args=ap.parse_args()
    raw_root=args.raw_root.resolve(); out=args.processed_root.resolve()
    candidates=[]
    for run in raw_root.glob("*"):
        config=run / "p7_p2_exact_single_gemm.json"
        paths=[run/f"process_{i}"/"summary.csv" for i in range(1,4)]
        if config.exists() and all(p.exists() for p in paths):
            candidates.append((run, config, paths))
    if not candidates:
        raise RuntimeError("no complete three-process exact-P2 P7 run")
    run, config_path, paths=sorted(candidates)[-1]
    rows=[]
    for index, path in enumerate(paths, 1):
        row=read_row(path)
        row["process_index"]=index
        row["process_exit_code"]=(path.parent / "process_exit_code.txt").read_text().strip() if (path.parent / "process_exit_code.txt").exists() else "unknown"
        rows.append(row)
    measured=[r for r in rows if r["status"] == "measured" and r.get("correctness") == "pass"]
    status={
        "run":str(run), "config":json.loads(config_path.read_text()), "trials":len(rows),
        "compile_passes":sum(r["status"] != "compile_failed" for r in rows),
        "correctness_passes":sum(r.get("correctness") == "pass" for r in rows),
        "measured_passes":len(measured),
        "runtime_or_gate_failures":sum(r["status"] == "runtime_failed" for r in rows),
        "median_latency_us_of_measured":statistics.median(float(r["latency_us_median"]) for r in measured) if measured else None,
        "parameter_status":"diagnostic only; no P5/P6 model input",
        "rows":rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    fields=sorted({key for row in rows for key in row})
    with (out / "summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
