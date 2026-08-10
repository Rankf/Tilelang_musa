"""Device integration coverage for reviewed runtime collection paths."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.runtime import collect


@unittest.skipUnless(__import__("torch").musa.is_available(), "requires a MUSA device")
class RuntimeGemmReduceTest(unittest.TestCase):
    factory = "musa_perf_collector.examples.runtime_gemm_reduce:build_workload"

    def _config(self, mode: str) -> dict:
        return {
            "mode": mode, "device": 0, "sampling_seed": 1, "input_seed": 1,
            "M": 128, "N": 128, "K": 64, "threads": 256,
            "baseline": {"warmup_launches": 1, "repeat_launches": 4, "event_batch_size": 2, "cache_policy": "preserve"},
            "telemetry": {"enabled": False}, "standard": {"max_profiled_launches": 1},
            "deep": {"launch_indices": [0], "block_coordinates": [[0, 0, 0]], "tileop_families": ["gemm_dense", "reduce"],
                     "max_tileops_per_block": 3, "max_iterations_per_tileop": 1, "max_trace_records": 128, "max_trace_bytes": 8192},
            "budget": {"max_total_duration_s": 30, "max_profiled_launches": 1, "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 3, "max_iterations_per_tileop": 1,
                       "max_trace_records": 128, "max_trace_bytes": 8192, "max_counter_passes": 1,
                       "max_dropped_records": 0, "on_budget_exceeded": "degrade"},
        }

    def test_light_keeps_correctness_and_baseline_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = collect(self.factory, config=self._config("light"), output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            correctness = json.loads((output / "correctness.json").read_text())
            self.assertEqual(manifest["actual_mode"], "light")
            self.assertFalse(manifest["instrumented"])
            self.assertEqual(correctness["status"], "pass")
            self.assertTrue((output / "baseline" / "event_samples.jsonl").is_file())

    def test_deep_emits_bounded_trace_and_separate_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config("deep")
            # One correctness/trace launch plus two two-launch ABBA B groups.
            config["budget"]["max_instrumented_launches"] = 5
            output = collect(self.factory, config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            overhead = json.loads((output / "overhead.json").read_text())
            self.assertEqual(manifest["actual_mode"], "deep")
            self.assertTrue(manifest["instrumented"])
            self.assertNotEqual(manifest["baseline_kernel_hash"], manifest["instrumented_kernel_hash"])
            self.assertEqual(len(trace), 2)
            self.assertTrue(all(item["block"] == [0, 0, 0] for item in trace))
            self.assertTrue(overhead["instrumented"])
            self.assertLessEqual(overhead["instrumented_launch_usage"]["used"],
                                 overhead["instrumented_launch_usage"]["budget"])
            self.assertEqual(overhead["instrumented_launch_usage"]["used"], 5)
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            self.assertEqual(instrumentation["trace_metadata"]["timestamps_per_sample"], 2)
            self.assertEqual(instrumentation["readback"]["actual_policy"], "async_after_window")
            self.assertTrue(instrumentation["readback"]["async_probe"]["data_verified"])
            transitions = [item["to"] for item in instrumentation["readback"]["slot"]["transitions"]]
            self.assertEqual(transitions, ["DEVICE_WRITING", "DEVICE_READY", "D2H_COPYING", "HOST_READY", "FREE"])
            self.assertEqual(instrumentation["readback"]["details"]["generation_id"],
                             instrumentation["readback"]["slot"]["generation_id"])
            self.assertEqual(instrumentation["trace_validation"]["status"], "pass")
            self.assertEqual(instrumentation["trace_validation"]["lineage_mode"], "compiler_exact")
            self.assertEqual({item["lineage"] for item in instrumentation["lineage"]}, {"compiler_exact_lineage"})
            self.assertEqual({item["source_ordinal"] for item in instrumentation["lineage"]}, {4})

    def test_deep_reduce_max_preserves_family_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config("deep")
            config["reduction"] = "max"
            output = collect(self.factory, config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["workload"], "gemm_reduce_max_fp16")
            self.assertEqual(trace[-1]["record_kind"], "reduce_max_end_issue_boundary")
            self.assertEqual(instrumentation["lineage"][-1]["semantic_op"], "reduce_max")
            self.assertEqual(instrumentation["trace_validation"]["lineage_mode"], "compiler_exact")

    def test_deep_multi_tileop_trace_uses_four_ordered_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config("deep")
            config["deep"]["trace_source_ordinals"] = [3, 4]
            output = collect(self.factory, config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            trace = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            instrumentation = json.loads((output / "deep" / "instrumentation_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(trace), 4)
            self.assertEqual([item["source_ordinal"] for item in trace], [3, 3, 4, 4])
            self.assertEqual([item["boundary"] for item in trace], ["begin", "end", "begin", "end"])
            self.assertEqual(instrumentation["trace_validation"]["selected_source_ordinals"], [3, 4])
            self.assertEqual(instrumentation["readback"]["details"]["copy_bytes"], 32)

    def test_offline_op_ids_join_to_runtime_slots_and_tile_metadata(self):
        artifact = Path("/root/musa-perf-collector/results/offline/analysis_0a0d26cf0bd4ffde")
        if not artifact.is_dir():
            self.skipTest("reviewed offline exact-lineage artifact is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config("deep")
            config["deep"].pop("trace_source_ordinals", None)
            config["deep"].update({
                "lineage_artifact": str(artifact),
                "trace_op_ids": ["op_ba6a0cf9b64ded14", "op_712a09f7641bd83c"],
            })
            output = collect(self.factory, config=config, output_root=Path(temporary))
            association = json.loads((output / "deep" / "offline_runtime_join.json").read_text())
            tileops = [json.loads(line) for line in
                       (output / "deep" / "tileop_context.jsonl").read_text().splitlines()]
            enriched = [json.loads(line) for line in
                        (output / "deep" / "trace_enriched.jsonl").read_text().splitlines()]
            raw = [json.loads(line) for line in (output / "deep" / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(association["status"], "complete")
            self.assertEqual(association["tileop_count"], 2)
            self.assertEqual([item["raw_op_name"] for item in tileops], ["tl.tileop.gemm", "tl.tileop.reduce"])
            self.assertEqual(tileops[0]["operation_details"]["gemm"]["m"], 128)
            self.assertEqual(tileops[1]["operation_details"]["reduction"]["kind"], '"sum"')
            self.assertEqual([item["tileop"]["source_op_id"] for item in enriched],
                             ["op_ba6a0cf9b64ded14"] * 2 + ["op_712a09f7641bd83c"] * 2)
            self.assertEqual([{key: value for key, value in item.items() if key != "tileop"} for item in enriched], raw)

    def test_unavailable_profiler_requests_are_explicitly_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config("standard")
            config["standard"].update({"metrics": ["sm_active"], "pc_sampling": {"enabled": True}})
            output = collect(self.factory, config=config, output_root=Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            diagnostics = json.loads((output / "diagnostics.json").read_text())["diagnostics"]
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual({item["code"] for item in diagnostics}, {"MUPTI_COUNTER_ADAPTER_UNAVAILABLE", "PC_SAMPLING_UNAVAILABLE"})
