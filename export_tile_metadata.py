#!/usr/bin/env python3
"""Export TileSight item-5.1 metadata from a TileLang PrimFunc factory."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MUSA_HOME", "/usr/local/musa")
sys.path[:0] = [
    str(BASE),
    str(BASE / "TileOPs"),
    str(BASE / "tilelang-musa"),
    str(BASE / "DeepStack-ae-source" / "src" / "tilesight"),
    str(BASE / "DeepStack-ae" / "src"),
]

from deepstack.mosaic.arch import make_custom_stacked_gpu, make_s5000_config
from tilesight.tir_interface import analyze_prim_func


def _load_factory(spec):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--factory must use package.module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"factory {spec!r} is not callable")
    return factory


def main():
    parser = argparse.ArgumentParser(
        description="从TileLang PrimFunc导出tilesight/tile-metadata-v1"
    )
    parser.add_argument(
        "--factory", required=True,
        help="返回PrimFunc/IRModule的Python factory，格式package.module:function",
    )
    parser.add_argument("-o", "--output", default="",
                        help="输出JSON；省略则打印到终端")
    args = parser.parse_args()

    prim_func = _load_factory(args.factory)()
    arch = make_custom_stacked_gpu(make_s5000_config())
    metadata = analyze_prim_func(prim_func, arch)
    output = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print(f"5.1 Tile metadata已保存到: {path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
