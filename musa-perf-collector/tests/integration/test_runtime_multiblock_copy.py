"""S5000 regression coverage for bounded fixed-slot multi-block tracing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.examples.runtime_multiblock_copy import build_workload
from musa_perf_collector.runtime import collect


@unittest.skipUnless(__import__("torch").musa.is_available(), "requires a MUSA device")
class RuntimeMultiBlockCopyTest(unittest.TestCase):
    factory = "musa_perf_collector.examples.runtime_multiblock_copy:build_workload"

    def _config(self) -> dict:
        return {
            "mode": "deep", "device": 0, "input_seed": 1, "size": 512, "block_size": 128,
            "blocks": 4, "threads": 128,
            "baseline": {"warmup_launches": 1, "repeat_launches": 4, "event_batch_size": 2, "cache_policy": "preserve"},
            "telemetry": {"enabled": False}, "standard": {"max_profiled_launches": 1},
            "deep": {"trace_source_ordinals": [0], "launch_indices": [0],
                     "block_coordinates": [[0, 0, 0], [3, 0, 0]], "tileop_families": ["copy"],
                     "max_tileops_per_block": 1, "max_iterations_per_tileop": 1,
                     "max_trace_records": 4, "max_trace_bytes": 32, "readback_policy": "async_after_window"},
            "budget": {"max_total_duration_s": 30, "max_profiled_launches": 1, "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 2, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 4, "max_trace_bytes": 32,
                       "max_counter_passes": 1, "max_dropped_records": 0, "on_budget_exceeded": "degrade"},
        }

    def test_selected_blocks_have_isolated_slots_and_exact_output(self):
        config = self._config()
        config["execution_stream"] = "default"
        case = build_workload(config).prepare(config, variant="instrumented")
        self.assertEqual(case.kernel_source.count("clock64"), 4)
        self.assertIn("Trace[0]", case.kernel_source)
        self.assertIn("Trace[2]", case.kernel_source)
        with tempfile.TemporaryDirectory() as temporary:
            output = collect(self.factory, config=config, output_root=Path(temporary))
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            overhead = json.loads((output / "overhead.json").read_text())
            stream_sync = json.loads((output / "stream_sync.json").read_text())
            self.assertEqual([(item["trace_slot"], item["block"]) for item in trace], [
                (0, [0, 0, 0]), (1, [0, 0, 0]), (2, [3, 0, 0]), (3, [3, 0, 0]),
            ])
            self.assertEqual(instrumentation["trace_metadata"]["selected_blocks"], [[0, 0, 0], [3, 0, 0]])
            self.assertFalse(instrumentation["trace_validation"]["trace_clock_monotonic"])
            self.assertTrue(instrumentation["trace_validation"]["trace_pair_monotonic"])
            self.assertEqual(instrumentation["trace_validation"]["selected_source_ordinals"], [0])
            self.assertEqual(instrumentation["instrumented_launch_usage"]["used"], 1)
            self.assertEqual(instrumentation["instrumented_launch_usage"]["budget"], 1)
            self.assertEqual(overhead["status"], "unavailable")
            self.assertEqual(stream_sync["status"], "pass")
            self.assertEqual(stream_sync["observed_baseline_stream_ids"], [0])

    def test_periodic_launch_and_exact_block_selectors_control_execution(self):
        config = self._config()
        config["execution_stream"] = "custom"
        config["deep"].pop("launch_indices")
        config["deep"].pop("block_coordinates")
        config["deep"].update({
            "launch_candidate_count": 4,
            "launch_selector": {"every": 2, "start": 1},
            "block_selector": {"indices": [1, 2]},
            "max_trace_records": 8,
            "max_trace_bytes": 64,
        })
        config["budget"].update({"max_instrumented_launches": 2, "max_trace_records": 8,
                                 "max_trace_bytes": 64})
        with tempfile.TemporaryDirectory() as temporary:
            output = collect(self.factory, config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            selection = json.loads((output / "deep" / "selection.json").read_text())
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            stream_sync = json.loads((output / "stream_sync.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(selection["launch"]["selected"], [1, 3])
            self.assertEqual(selection["launch"]["selector_kind"], "periodic")
            self.assertEqual(selection["block"]["selected_coordinates"], [[1, 0, 0], [2, 0, 0]])
            self.assertEqual({item["launch_index"] for item in trace}, {1, 3})
            self.assertEqual({tuple(item["block"]) for item in trace}, {(1, 0, 0), (2, 0, 0)})
            self.assertEqual(len(trace), 8)
            self.assertEqual(instrumentation["instrumented_launch_usage"]["used"], 2)
            self.assertEqual([item["launch_index"] for item in instrumentation["readback"]["runs"]], [1, 3])
            custom_stream_id = stream_sync["expected_baseline_stream_id"]
            self.assertNotEqual(custom_stream_id, 0)
            self.assertEqual(stream_sync["observed_baseline_stream_ids"], [custom_stream_id])
            self.assertEqual({item["producer_stream_id"] for item in stream_sync["async_readbacks"]},
                             {custom_stream_id})
            self.assertNotIn(custom_stream_id,
                             {item["collector_stream_id"] for item in stream_sync["async_readbacks"]})
