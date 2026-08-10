"""Offline exact lineage is required for automatic runtime target selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from musa_perf_collector.collectors.runtime.lineage_selection import resolve


class LineageSelectionTest(unittest.TestCase):
    def _artifact(self, root: Path) -> Path:
        path = root / "metadata"
        path.mkdir()
        (path / "lineage.json").write_text(json.dumps({
            "status": "compiler_exact_lineage", "exact_runtime_instrumentation_allowed": True,
            "mappings": [
                {"source_op_id": "gemm", "source_kind": "gemm", "source_ordinal": 3,
                 "eligible_for_exact_runtime_instrumentation": True,
                 "source_loop_context": [{"lexical_loop_ordinal": 0, "kind": "serial", "thread_binding": None,
                                          "extent_value": 4}]},
                {"source_op_id": "reduce", "source_kind": "reduce_sum", "source_ordinal": 4,
                 "eligible_for_exact_runtime_instrumentation": True},
            ],
        }))
        (root / "manifest.json").write_text(json.dumps({"input_config": {"M": 128, "N": 128}}))
        return root

    def test_resolves_explicit_op_ids_in_source_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            ordinals, evidence = resolve({"lineage_artifact": str(artifact), "trace_op_ids": ["reduce", "gemm"]})
            self.assertEqual(ordinals, [3, 4])
            self.assertEqual([item["source_op_id"] for item in evidence["selected"]], ["gemm", "reduce"])

    def test_artifact_requires_explicit_op_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "trace_op_ids"):
                resolve({"lineage_artifact": str(artifact)})

    def test_rejects_mismatched_workload_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                resolve({"lineage_artifact": str(artifact), "trace_op_ids": ["gemm"]},
                        workload_config={"M": 64, "N": 128})

    def test_exact_iteration_selector_requires_enclosing_source_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            ordinals, evidence = resolve({
                "lineage_artifact": str(artifact), "trace_op_ids": ["gemm"],
                "trace_loop_ordinal": 0, "trace_iteration": 2,
            })
            self.assertEqual(ordinals, [3])
            self.assertEqual(evidence["iteration_selector"],
                             {"source_loop_ordinal": 0, "relative_iteration": 2})

    def test_iteration_selector_rejects_unrelated_source_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "not enclosed"):
                resolve({"lineage_artifact": str(artifact), "trace_op_ids": ["reduce"],
                         "trace_loop_ordinal": 0, "trace_iteration": 0})
