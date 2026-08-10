"""Human-readable report with explicit static-analysis limitations."""

from __future__ import annotations

import html
from typing import Any


def render_markdown(manifest: dict[str, Any], metadata: dict[str, Any], domains: dict[str, Any], dag: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    kernel = metadata["kernel"]
    per_block = domains["per_block"]
    per_kernel = domains["per_kernel_launch"]
    lines = [
        f"# Offline TileLang-MUSA analysis — {kernel['name']}",
        "",
        "## Scope",
        "",
        "This report was generated without MUSA kernel execution, Event timing, profiler counters, or an S5000 architecture profile.",
        "",
        "## Compiler provenance",
        "",
        f"- Target: `{manifest['target']}`",
        f"- TileLang revision: `{manifest.get('tilelang_revision') or 'unknown'}`",
        f"- Device execution: `{manifest['device_execution']}`",
        f"- Captured passes: {len(manifest['pass_sequence'])}",
        "",
        "## Static demand by execution domain",
        "",
        f"- Per block FLOPs: `{per_block['flops']}`",
        f"- Per block logical bytes: `{per_block['logical_total_bytes']}`",
        f"- Per block AI (all logical): `{per_block['arithmetic_intensity_all_logical']}` FLOPs/byte",
        f"- Per kernel-launch FLOPs: `{per_kernel['flops']}`",
        f"- Per kernel-launch logical bytes: `{per_kernel['logical_total_bytes']}`",
        f"- Per kernel-launch AI (global logical): `{per_kernel['arithmetic_intensity_global_logical']}` FLOPs/byte",
        "",
        "## Structure",
        "",
        f"- Buffers: {len(metadata['buffers'])}",
        f"- TileOp actions: {len(metadata['tile_ops'])}",
        f"- TileOp opaque records: {metadata['tileop_coverage']['opaque_records']}",
        f"- Pipeline loops: {len(metadata['pipeline']['pipeline_loops'])}",
        f"- Data-dependence edges: {len(dag['edges'])}",
        f"- Offline-static status: `{diagnostics['status']}`",
        "",
        "## Important limitations",
        "",
        "- Logical bytes are compiler-derived accesses, not measured MUSA bus traffic.",
        "- Capacity traffic uses a bounded static schedule and is not an S5000 cache-hit measurement.",
        "- The upper-bound formula contains uncalibrated throughput and bandwidth variables, so no absolute latency is reported.",
    ]
    return "\n".join(lines) + "\n"


def render_html(markdown: str) -> str:
    """Emit a dependency-free review copy without pretending to be rich HTML."""
    return "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>Offline TileLang-MUSA analysis</title></head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n"
