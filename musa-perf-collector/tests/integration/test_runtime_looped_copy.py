"""S5000 regression coverage for compiler-exact source-loop iteration trace."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.examples.runtime_looped_copy import build_workload
from musa_perf_collector.runtime import collect


@unittest.skipUnless(__import__("torch").musa.is_available(), "requires a MUSA device")
class RuntimeLoopedCopyTest(unittest.TestCase):
    factory = "musa_perf_collector.examples.runtime_looped_copy:build_workload"

    def _config(self) -> dict:
        return {
            "mode": "deep", "device": 0, "input_seed": 1, "size": 128, "threads": 128, "iterations": 4,
            "baseline": {"warmup_launches": 1, "repeat_launches": 4, "event_batch_size": 2, "cache_policy": "preserve"},
            "telemetry": {"enabled": False}, "standard": {"max_profiled_launches": 1},
            "deep": {"trace_source_ordinals": [0], "trace_loop_ordinal": 0, "trace_iteration": 2,
                     "launch_indices": [0], "block_coordinates": [[0, 0, 0]], "tileop_families": ["copy"],
                     "max_tileops_per_block": 1, "max_iterations_per_tileop": 1,
                     "max_trace_records": 2, "max_trace_bytes": 16, "readback_policy": "async_after_window"},
            "budget": {"max_total_duration_s": 30, "max_profiled_launches": 1, "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 2, "max_trace_bytes": 16,
                       "max_counter_passes": 1, "max_dropped_records": 0, "on_budget_exceeded": "degrade"},
        }

    def test_selected_source_iteration_is_correct_and_auditable(self):
        config = self._config()
        case = build_workload(config).prepare(config, variant="instrumented")
        self.assertEqual(case.kernel_source.count("clock64"), 2)
        self.assertIn("(__1 == 2)", case.kernel_source)
        with tempfile.TemporaryDirectory() as temporary:
            output = collect(self.factory, config=config, output_root=Path(temporary))
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            self.assertEqual([(item["source_loop_ordinal"], item["relative_iteration"]) for item in trace],
                             [(0, 2), (0, 2)])
            self.assertEqual(instrumentation["trace_validation"]["iteration_selector"],
                             {"source_loop_ordinal": 0, "relative_iteration": 2})
            self.assertTrue((output / "baseline" / "kernel.mu").is_file())
            self.assertTrue((output / "deep" / "instrumented_kernel.mu").is_file())
