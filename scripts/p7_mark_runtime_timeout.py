#!/usr/bin/env python3
"""Write a raw-data marker when a P7 diagnostic child times out before output."""
import argparse, json, pathlib

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=pathlib.Path,required=True)
    ap.add_argument("--case-id",required=True)
    ap.add_argument("--trial",type=int,required=True)
    ap.add_argument("--context",required=True)
    ap.add_argument("--timeout-seconds",type=int,required=True)
    args=ap.parse_args()
    payload={"case_id":args.case_id,"trial":args.trial,"status":"timeout_terminated","correctness":"not_run","failure_stage":"process_timeout","error":"P7 runner timed out a single diagnostic child before it wrote result.json.","timeout_context":args.context,"timeout_seconds":args.timeout_seconds}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,sort_keys=True))
if __name__=="__main__": main()
