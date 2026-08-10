"""Structural compiler anchors never masquerade as exact provenance."""

from __future__ import annotations

import unittest

from musa_perf_collector.collectors.compiler.lineage import structural_candidates


class LineageTest(unittest.TestCase):
    def test_unique_cumsum_anchor_is_candidate_not_exact(self):
        result = structural_candidates([{"op_id": "op_1", "kind": "cumsum"}], {"S40_TILE_OP_LOWERED": 'T.call_extern("tl::CumSum1D<128>::run")'})
        mapping = result["mappings"][0]
        self.assertEqual(mapping["confidence"], "high")
        self.assertFalse(mapping["eligible_for_exact_runtime_instrumentation"])

    def test_repeated_gemm_anchor_is_ambiguous(self):
        result = structural_candidates([{"op_id": "op_1", "kind": "gemm"}, {"op_id": "op_2", "kind": "gemm"}], {"S40_TILE_OP_LOWERED": "T.tl_gemm(\nT.tl_gemm("})
        self.assertTrue(all(item["confidence"] == "ambiguous" for item in result["mappings"]))

    def test_complete_compiler_ordinals_are_exact(self):
        tile_ops = [
            {"op_id": "op_copy", "kind": "copy", "ordinal": 0},
            {"op_id": "op_scan", "kind": "cumsum", "ordinal": 1},
        ]
        snapshots = {
            stage: ('T.attr(0, "tl.provenance_ordinal", 0)\n'
                    'T.attr(0, "tl.provenance_ordinal", 1)')
            for stage in ("S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S60_FINAL_DEVICE_IR")
        }
        result = structural_candidates(tile_ops, snapshots)
        self.assertEqual(result["status"], "compiler_exact_lineage")
        self.assertTrue(result["exact_runtime_instrumentation_allowed"])
        self.assertTrue(all(item["eligible_for_exact_runtime_instrumentation"] for item in result["mappings"]))

    def test_exact_lineage_exports_source_loop_context(self):
        tile_ops = [{
            "op_id": "op_copy", "kind": "copy", "ordinal": 0,
            "loop_context": [{"loop_id": "lexical_loop/2", "min": "0", "extent": "4",
                              "extent_value": 4, "kind": "serial", "thread_binding": None}],
        }]
        snapshots = {
            stage: 'T.attr(0, "tl.provenance_ordinal", 0)'
            for stage in ("S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S60_FINAL_DEVICE_IR")
        }
        mapping = structural_candidates(tile_ops, snapshots)["mappings"][0]
        self.assertEqual(mapping["source_loop_context"], [{
            "lexical_loop_ordinal": 2, "min": "0", "extent": "4", "extent_value": 4,
            "kind": "serial", "thread_binding": None,
        }])

    def test_partial_compiler_ordinals_are_rejected(self):
        tile_ops = [
            {"op_id": "op_copy", "kind": "copy", "ordinal": 0},
            {"op_id": "op_scan", "kind": "cumsum", "ordinal": 1},
        ]
        snapshots = {
            "S30_LAYOUT_INFERRED": 'T.attr(0, "tl.provenance_ordinal", 0)',
            "S40_TILE_OP_LOWERED": 'T.attr(0, "tl.provenance_ordinal", 0)',
            "S60_FINAL_DEVICE_IR": 'T.attr(0, "tl.provenance_ordinal", 0)',
        }
        result = structural_candidates(tile_ops, snapshots)
        self.assertEqual(result["status"], "compiler_provenance_incomplete")
        self.assertFalse(result["exact_runtime_instrumentation_allowed"])
