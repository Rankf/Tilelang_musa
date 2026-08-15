#!/usr/bin/env python3
"""Aggregate TileSight kernel JSON reports into one offline session report."""

import argparse
import json
from pathlib import Path

from analyze_op import build_session_report, render_session_summary


def main():
    parser = argparse.ArgumentParser(description="TileSight多Kernel离线Session报告")
    parser.add_argument("inputs", nargs="+", help="analyze_op.py生成的Kernel JSON")
    parser.add_argument("--session-id", default="offline_session")
    parser.add_argument("--report-type", choices=["json", "summary"],
                        default="summary")
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8"))
               for path in args.inputs]
    session = build_session_report(reports, session_id=args.session_id)
    output = (json.dumps(session, indent=2, ensure_ascii=False)
              if args.report_type == "json"
              else render_session_summary(session).rstrip("\n"))
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"Session报告已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
