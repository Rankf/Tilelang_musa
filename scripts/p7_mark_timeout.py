#!/usr/bin/env python3
"""Write a factual timeout marker for a terminated P7 child process."""

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--unroll", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--rss-kib", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    payload = {"case_id": "p7_fp16_gemm_1536x1024x1536", "graph_unroll": args.unroll, "trial": args.trial,
               "status": "timeout_terminated", "correctness": "not_run", "failure_stage": "pre_compile_or_runtime_initialization_timeout",
               "error": "P7 controller terminated only this child after it exceeded peer startup duration without TileLang compile output.",
               "timeout_evidence": {"pid": args.pid, "elapsed_seconds_before_termination": args.elapsed_seconds,
                                    "rss_kib_before_termination": args.rss_kib, "threads_before_termination": args.threads}}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__": main()
