"""Compiler-exact one-to-many provenance coverage for the MUSA pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.examples.online_multi_instance_gemm_app import build_prim_func
from musa_perf_collector.offline import analyze


class OfflineMultiInstanceGemmTest(unittest.TestCase):
    def test_source_index_copy_lowers_to_two_stable_device_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = analyze(
                build_prim_func(),
                config={"workload": "online_multi_instance_gemm"},
                output_root=Path(temporary),
                arch="mp_31",
                tilelang_root=Path("/root/s5000-calibration/tilelang-musa"),
            )
            lineage = json.loads((output / "metadata" / "lineage.json").read_text())
            mappings = lineage["mappings"]
            duplicated = [item for item in mappings if item["lowered_instance_count"] > 1]
            self.assertEqual(lineage["status"], "compiler_exact_lineage")
            self.assertEqual(len(duplicated), 1)
            self.assertEqual(duplicated[0]["source_kind"], "copy")
            self.assertEqual(duplicated[0]["source_ordinal"], 0)
            self.assertEqual(duplicated[0]["lowered_instance_count"], 2)
            self.assertEqual(duplicated[0]["final_device_instance_count"], 2)
            self.assertEqual(duplicated[0]["source_op_id"], "op_bfdf2ff536ef6911")
