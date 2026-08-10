"""S5000 integration coverage for the non-GEMM ``scan`` trace adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.runtime import collect


@unittest.skipUnless(__import__("torch").musa.is_available(), "requires a MUSA device")
class RuntimeCumsumTest(unittest.TestCase):
    def test_deep_cumsum_has_exact_scan_lineage(self):
        config = {
            "mode": "deep", "device": 0, "input_seed": 1, "size": 128, "threads": 128,
            "baseline": {"warmup_launches": 1, "repeat_launches": 4, "event_batch_size": 2, "cache_policy": "preserve"},
            "telemetry": {"enabled": False}, "standard": {"max_profiled_launches": 1},
            "deep": {"launch_indices": [0], "block_coordinates": [[0, 0, 0]], "tileop_families": ["scan"],
                     "max_tileops_per_block": 1, "max_iterations_per_tileop": 1, "max_trace_records": 128,
                     "max_trace_bytes": 8192, "readback_policy": "async_after_window"},
            "budget": {"max_total_duration_s": 30, "max_profiled_launches": 1, "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 1, "max_iterations_per_tileop": 1,
                       "max_trace_records": 128, "max_trace_bytes": 8192, "max_counter_passes": 1,
                       "max_dropped_records": 0, "on_budget_exceeded": "degrade"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = collect("musa_perf_collector.examples.runtime_cumsum:build_workload", config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrument = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["workload"], "cumsum_fp32")
            self.assertEqual([item["record_kind"] for item in trace], ["cumsum_begin_issue_boundary", "cumsum_end_issue_boundary"])
            self.assertEqual({item["canonical_family"] for item in instrument["lineage"]}, {"scan"})
            self.assertEqual({item["lineage"] for item in instrument["lineage"]}, {"compiler_exact_lineage"})
            self.assertEqual({item["source_ordinal"] for item in instrument["lineage"]}, {1})
            self.assertEqual(instrument["trace_validation"]["status"], "pass")
            self.assertEqual(instrument["trace_validation"]["lineage_mode"], "compiler_exact")
