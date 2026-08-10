"""Integration test that lowers only; it never allocates or launches MUSA work."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.examples.pipelined_gemm import build_prim_func
from musa_perf_collector.offline import analyze


class OfflinePipelinedGemmTest(unittest.TestCase):
    def test_cpu_only_artifact_bundle(self):
        config = {
            "M": 128, "N": 128, "K": 128,
            "block_M": 64, "block_N": 64, "block_K": 32,
            "num_stages": 2, "dtype": "float16",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = analyze(
                build_prim_func(config), config=config, output_root=Path(temporary),
                arch="mp_31", tilelang_root=Path("/root/s5000-calibration/tilelang-musa"),
            )
            manifest = json.loads((output / "manifest.json").read_text())
            per_block = json.loads((output / "metrics" / "workload_per_block.json").read_text())
            per_kernel = json.loads((output / "metrics" / "workload_per_kernel.json").read_text())
            coverage = json.loads((output / "metadata" / "tileop_coverage.json").read_text())
            diagnostics = json.loads((output / "diagnostics.json").read_text())
            lineage = json.loads((output / "metadata" / "lineage.json").read_text())
            tile_op_records = [json.loads(line) for line in (output / "metadata" / "tile_ops.jsonl").read_text().splitlines()]
            reuse = json.loads((output / "metrics" / "reuse_distance.json").read_text())
            dag = json.loads((output / "metadata" / "dag.json").read_text())
            tile_ops = (output / "metadata" / "tile_ops.jsonl").read_text().splitlines()
            self.assertFalse(manifest["device_execution"])
            self.assertFalse(manifest["device_compile"])
            self.assertIn("S00_INPUT_TILE_IR", manifest["snapshot_stages"])
            self.assertIn("S60_FINAL_DEVICE_IR", manifest["snapshot_stages"])
            self.assertEqual(per_block["flops"], 2 * 64 * 64 * 32 * 4)
            self.assertEqual(per_kernel["flops"], 2 * 64 * 64 * 32 * 4 * 4)
            self.assertEqual(per_block["logical_total_bytes"], 270336)
            self.assertEqual(per_block["scope_transfer_bytes"]["global->shared.dyn"], 32768)
            self.assertEqual(per_block["arithmetic_intensity_global_logical"], 25.6)
            self.assertEqual(coverage["observed_action_calls"], 5)
            self.assertEqual(coverage["ignored_constructor_calls"], 10)
            self.assertEqual(coverage["unexported_action_calls"], 0)
            self.assertEqual(diagnostics["status"], "complete")
            self.assertEqual(lineage["status"], "compiler_exact_lineage")
            self.assertTrue(lineage["exact_runtime_instrumentation_allowed"])
            self.assertTrue(all(item["eligible_for_exact_runtime_instrumentation"] for item in lineage["mappings"]))
            self.assertEqual([record["execution"]["calls_per_block_value"] for record in tile_op_records], [1, 4, 4, 4, 1])
            self.assertEqual([record["execution"]["calls_per_kernel_value"] for record in tile_op_records], [4, 16, 16, 16, 4])
            self.assertEqual(reuse["status"], "complete_bounded_static_schedule")
            self.assertEqual(reuse["expanded_event_count_per_block"], 14)
            self.assertEqual(reuse["capacity_sweep"][0]["potential_global_read_traffic_bytes"], 32768)
            self.assertEqual(len(tile_ops), 5)
            self.assertTrue(any(edge["buffer"] == "a_shared" for edge in dag["edges"]))
            self.assertTrue(any(edge["buffer"] == "c_local" for edge in dag["edges"]))
            self.assertTrue((output / "generated" / "kernel.mu").is_file())
            self.assertTrue((output / "report" / "report.html").is_file())
