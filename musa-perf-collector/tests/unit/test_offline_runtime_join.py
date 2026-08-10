"""Exact offline TileOp to runtime slot association contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from musa_perf_collector.collectors.runtime.offline_join import associate


class OfflineRuntimeJoinTest(unittest.TestCase):
    def _artifact(self, root: Path) -> Path:
        (root / "metadata").mkdir()
        (root / "metrics").mkdir()
        (root / "manifest.json").write_text(json.dumps({
            "analysis_status": "complete", "device_execution": False, "analysis_id": "analysis_1",
            "kernel_id": "kernel_1", "tilelang_revision": "rev1", "input_ir_sha256": "hash1",
        }))
        (root / "metadata" / "lineage.json").write_text(json.dumps({
            "status": "compiler_exact_lineage", "exact_runtime_instrumentation_allowed": True,
            "mappings": [{"source_op_id": "gemm", "source_ordinal": 3, "source_kind": "gemm",
                          "status": "compiler_exact_lineage",
                          "eligible_for_exact_runtime_instrumentation": True}],
        }))
        (root / "metadata" / "tile_ops.jsonl").write_text(json.dumps({
            "op_id": "gemm", "ordinal": 3, "kind": "gemm", "canonical_family": "gemm_dense",
            "raw_op_name": "tl.tileop.gemm", "raw_ir": "T.gemm(A, B, C)", "statement_path": "tileop/3",
            "support_level": "structured", "gemm": {"m": 64, "n": 64, "k": 32, "flops": 262144},
            "reads": [{"buffer": "A", "shape": ["64", "32"]}],
            "writes": [{"buffer": "C", "shape": ["64", "64"]}],
            "loop_context": [], "condition_context": [], "execution": {"calls_per_kernel_value": 1},
        }) + "\n")
        (root / "metrics" / "workload_per_invocation.jsonl").write_text(json.dumps({
            "op_id": "gemm", "flops": 262144, "logical_total_bytes": 24576,
        }) + "\n")
        (root / "metadata" / "dag.json").write_text(json.dumps({"edges": []}))
        (root / "metadata" / "pipeline.json").write_text(json.dumps({"pipeline_loops": []}))
        return root

    def _case(self, artifact: Path, *, ordinal: int = 3):
        lineage_path = artifact / "metadata" / "lineage.json"
        return SimpleNamespace(trace_metadata={"selection_evidence": {
            "mode": "offline_compiler_exact_lineage", "lineage_artifact": str(lineage_path),
            "offline_tilelang_revision": "rev1", "offline_input_ir_sha256": "hash1",
            "selected": [{"source_op_id": "gemm", "source_ordinal": ordinal, "source_kind": "gemm"}],
        }}, lineage=[
            {"trace_slot": 0, "boundary": "begin", "block": [0, 0, 0], "source_op_id": "gemm",
             "source_ordinal": ordinal, "canonical_family": "gemm_dense"},
            {"trace_slot": 1, "boundary": "end", "block": [0, 0, 0], "source_op_id": "gemm",
             "source_ordinal": ordinal, "canonical_family": "gemm_dense"},
        ])

    @staticmethod
    def _trace():
        return [
            {"trace_slot": 0, "boundary": "begin", "block": [0, 0, 0], "clock_cycles": 10},
            {"trace_slot": 1, "boundary": "end", "block": [0, 0, 0], "clock_cycles": 20},
        ]

    def test_exact_join_exports_tile_ir_shape_metrics_and_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            result = associate(self._case(artifact), self._trace(), {
                "lineage_artifact": str(artifact), "trace_op_ids": ["gemm"],
            })
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["tileop_count"], 1)
            self.assertEqual(result["trace_record_count"], 2)
            tileop = result["tileops"][0]
            self.assertEqual(tileop["raw_ir"], "T.gemm(A, B, C)")
            self.assertEqual(tileop["reads"][0]["shape"], ["64", "32"])
            self.assertEqual(tileop["metrics_per_invocation"]["flops"], 262144)
            self.assertEqual(result["trace_records"][1]["tileop"]["source_op_id"], "gemm")

    def test_ordinal_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "ordinal conflicts"):
                associate(self._case(artifact, ordinal=4), self._trace(), {
                    "lineage_artifact": str(artifact), "trace_op_ids": ["gemm"],
                })

    def test_selection_artifact_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            case = self._case(artifact)
            case.trace_metadata["selection_evidence"]["lineage_artifact"] = "/tmp/different/metadata/lineage.json"
            with self.assertRaisesRegex(RuntimeError, "different lineage artifact"):
                associate(case, self._trace(), {
                    "lineage_artifact": str(artifact), "trace_op_ids": ["gemm"],
                })

    def test_missing_artifact_is_explicit_not_requested(self):
        result = associate(SimpleNamespace(trace_metadata={}, lineage=[]), [], {})
        self.assertEqual(result["status"], "not_requested")
