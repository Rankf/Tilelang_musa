"""CLI for CPU-only offline analysis and bounded MUSA runtime collection."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from .offline import analyze
from .runtime import collect as runtime_collect
from .collectors.runtime.capability import probe


def _factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("factory must use package.module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"factory {spec} is not callable")
    return function


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def _offline_analyze(args: argparse.Namespace) -> int:
    config = _config(args.config)
    prim_func = _factory(args.factory)(config)
    output = analyze(
        prim_func,
        config=config,
        output_root=args.output,
        arch=args.arch,
        pass_config=config.get("pass_config"),
        tilelang_root=args.tilelang_root,
    )
    print(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    top = parser.add_subparsers(dest="mode", required=True)
    offline = top.add_parser("offline", help="compiler-only analysis; never launches a MUSA kernel")
    offline_sub = offline.add_subparsers(dest="offline_command", required=True)
    analyze_parser = offline_sub.add_parser("analyze", help="analyze a PrimFunc factory")
    analyze_parser.add_argument("--factory", required=True)
    analyze_parser.add_argument("--config", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--arch", default="mp_31")
    analyze_parser.add_argument("--tilelang-root", type=Path, default=Path("/root/s5000-calibration/tilelang-musa"))
    analyze_parser.set_defaults(handler=_offline_analyze)

    runtime = top.add_parser("runtime", help="bounded MUSA runtime measurement")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    probe_parser = runtime_sub.add_parser("probe", help="inspect Event/profiler/MUPTI/telemetry availability")
    probe_parser.add_argument("--device", type=int, default=0)
    probe_parser.set_defaults(handler=lambda args: (print(json.dumps(probe(args.device), indent=2, sort_keys=True, default=str)) or 0))
    collect_parser = runtime_sub.add_parser("collect", help="run a reviewed workload adapter under a bounded mode")
    collect_parser.add_argument("--factory", required=True, help="package.module:build_workload")
    collect_parser.add_argument("--config", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.set_defaults(handler=lambda args: (print(runtime_collect(args.factory, config=_config(args.config), output_root=args.output)) or 0))
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
