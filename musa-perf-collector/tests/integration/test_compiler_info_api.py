"""Compiler-information API tests; no MUSA binary compile or launch occurs."""

from __future__ import annotations

from pathlib import Path
import unittest

from musa_perf_collector.adapters.tilelang import (
    TileLangCompilerInfoCollector,
    collect_compiler_information,
)
from musa_perf_collector.examples.pipelined_gemm import build_prim_func


class CompilerInfoApiTest(unittest.TestCase):
    def test_public_collector_factory_uses_tilelang_bundled_tvm(self):
        collector = TileLangCompilerInfoCollector.create()
        self.assertIsNotNone(collector.instrument)

    def test_collects_structured_multistage_compiler_facts(self):
        config = {
            "M": 128, "N": 128, "K": 128,
            "block_M": 64, "block_N": 64, "block_K": 32,
            "num_stages": 2, "dtype": "float16",
        }
        information = collect_compiler_information(
            build_prim_func(config),
            arch="mp_31",
            tilelang_root=Path("/root/s5000-calibration/tilelang-musa"),
        )

        self.assertEqual(
            information["schema_version"],
            "musa-perf-collector/tilelang-compiler-info-v1",
        )
        self.assertFalse(information["identity"]["device_compile"])
        self.assertTrue(information["coverage"]["complete_required_stage_capture"])
        self.assertTrue(information["coverage"]["source_tileop_lossless_export"])
        self.assertTrue(information["coverage"]["compiler_exact_lineage"])

        pipeline_calls = information["pipeline_planning"]["kernels"][0]["annotated_calls"]
        self.assertTrue(any(call["source_ordinal"] == 1 for call in pipeline_calls))
        self.assertTrue(any("is_tma_copy" in call["annotations"] for call in pipeline_calls))

        layout_blocks = information["layout_inference"]["kernels"][0]["blocks"]
        self.assertTrue(any("layout_map" in block["annotations"] for block in layout_blocks))

        lowered_regions = information["tileop_lowering"]["kernels"][0]["provenance_regions"]
        self.assertEqual({region["source_ordinal"] for region in lowered_regions}, {0, 1, 2, 3, 4})

        launch = information["final_implementation"]["launch"][0]
        self.assertEqual(launch["grid_size"], 4)
        self.assertEqual(
            information["source"]["kernel"]["launch_context"]["thread_count"],
            128,
        )
        # The MUSA lowering expands this kernel from the source request to the
        # actual launch configuration.  Both values must remain observable.
        self.assertEqual(launch["threads_per_block"], 256)
        self.assertEqual(launch["warps_per_block"], 8)
        self.assertGreater(launch["dynamic_shared_memory_bytes"], 0)

        final_ops = information["final_implementation"]["lowered_operations"]["kernels"][0]
        self.assertTrue(final_ops["synchronization_calls"])


if __name__ == "__main__":
    unittest.main()
