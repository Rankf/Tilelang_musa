"""Runtime report renderer that keeps evidence layers separate."""

from __future__ import annotations

import html
from typing import Any


def render_markdown(manifest: dict[str, Any], correctness: dict[str, Any], baseline: dict[str, Any],
                    diagnostics: dict[str, Any], overhead: dict[str, Any],
                    offline_runtime_join: dict[str, Any] | None = None) -> str:
    lines = [
        f"# Runtime TileLang-MUSA measurement — {manifest['workload']}", "",
        "## Validity", "",
        f"- Status: `{manifest['status']}`",
        f"- Correctness: `{correctness['status']}`",
        f"- Requested mode: `{manifest['requested_mode']}`",
        f"- Actual mode: `{manifest['actual_mode']}`",
        f"- Baseline kernel hash: `{manifest['baseline_kernel_hash']}`",
        f"- Instrumented: `{manifest['instrumented']}`", "",
        "## Uninstrumented Event baseline", "",
        f"- Samples: {baseline.get('sample_count', 0)}",
        f"- Median per-launch latency: `{baseline.get('median_per_launch_ms')}` ms",
        f"- Timing kind: `{baseline.get('timing_kind')}`", "",
        "## Collection overhead", "",
        f"- Status: `{overhead.get('status')}`",
        f"- Relative overhead: `{overhead.get('relative_overhead')}`", "",
        "## Offline TileOp association", "",
        f"- Status: `{(offline_runtime_join or {}).get('status', 'not_applicable')}`",
        f"- Mode: `{(offline_runtime_join or {}).get('association_mode')}`",
        f"- Selected source op IDs: `{(offline_runtime_join or {}).get('selected_source_op_ids', [])}`", "",
        "## Important interpretation rules", "",
        "- Baseline Event data is not replaced by profiler or instrumented measurements.",
        "- Telemetry is a measurement-window observation; it is not per-tile power.",
        "- Instrumented trace cycles describe selected program boundaries, not exclusive hardware time.", "",
        "## Diagnostics", "",
    ]
    lines.extend(f"- `{item['code']}`: {item.get('message', '')}" for item in diagnostics.get("diagnostics", []))
    return "\n".join(lines) + "\n"


def render_html(markdown: str) -> str:
    return "<!doctype html><html><head><meta charset=\"utf-8\"><title>Runtime TileLang-MUSA measurement</title></head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n"
