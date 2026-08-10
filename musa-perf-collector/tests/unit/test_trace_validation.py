"""Deep trace contract checks must reject incomplete or misleading evidence."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from musa_perf_collector.collectors.runtime.trace_validation import validate


def _case(*, lineage: str = "manual_exact_workload_contract"):
    return SimpleNamespace(instrumented=True, trace_metadata={"timestamps_per_sample": 2}, lineage=[
        {"trace_slot": 0, "boundary": "begin", "canonical_family": "gemm_dense", "lineage": lineage},
        {"trace_slot": 1, "boundary": "end", "canonical_family": "reduce", "lineage": lineage},
    ])


def _compiler_multi_case():
    lineage = []
    for slot, ordinal in enumerate((3, 3, 4, 4)):
        lineage.append({"trace_slot": slot, "boundary": "begin" if slot % 2 == 0 else "end",
                        "canonical_family": "gemm_dense" if ordinal == 3 else "reduce",
                        "lineage": "compiler_exact_lineage", "source_ordinal": ordinal,
                        "compiler_provenance_annotation": "tl.provenance_ordinal"})
    return SimpleNamespace(instrumented=True, trace_metadata={"timestamps_per_sample": 4}, lineage=lineage)


class TraceValidationTest(unittest.TestCase):
    trace = [{"trace_slot": 0, "boundary": "begin", "record_kind": "gemm_issue_boundary", "clock_cycles": 3},
             {"trace_slot": 1, "boundary": "end", "record_kind": "reduce_sum_issue_boundary", "clock_cycles": 4}]

    def test_exact_two_point_trace_passes(self):
        self.assertEqual(validate(_case(), self.trace, {"tileop_families": ["gemm_dense", "reduce"]})["status"], "pass")

    def test_heuristic_lineage_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exact lineage"):
            validate(_case(lineage="heuristic_mapping"), self.trace, {})

    def test_nonmonotonic_clock_is_rejected(self):
        trace = [dict(self.trace[0]), dict(self.trace[1])]
        trace[0]["clock_cycles"], trace[1]["clock_cycles"] = 4, 3
        with self.assertRaisesRegex(RuntimeError, "not monotonic"):
            validate(_case(), trace, {})

    def test_compiler_lineage_requires_provenance_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "provenance ordinal"):
            validate(_case(lineage="compiler_exact_lineage"), self.trace, {})

    def test_compiler_multi_tileop_pairs_pass(self):
        trace = [
            {"trace_slot": 0, "boundary": "begin", "source_ordinal": 3, "record_kind": "gemm_begin_issue_boundary", "clock_cycles": 1},
            {"trace_slot": 1, "boundary": "end", "source_ordinal": 3, "record_kind": "gemm_end_issue_boundary", "clock_cycles": 2},
            {"trace_slot": 2, "boundary": "begin", "source_ordinal": 4, "record_kind": "reduce_begin_issue_boundary", "clock_cycles": 3},
            {"trace_slot": 3, "boundary": "end", "source_ordinal": 4, "record_kind": "reduce_end_issue_boundary", "clock_cycles": 4},
        ]
        result = validate(_compiler_multi_case(), trace, {"tileop_families": ["gemm_dense", "reduce"]})
        self.assertEqual(result["selected_source_ordinals"], [3, 4])

    def test_iteration_trace_requires_matching_record_context(self):
        case = _case()
        case.trace_metadata["iteration_selector"] = {"source_loop_ordinal": 0, "relative_iteration": 2}
        trace = [
            {"record_kind": "gemm_issue_boundary", "clock_cycles": 3,
             "trace_slot": 0, "boundary": "begin", "source_loop_ordinal": 0, "relative_iteration": 2},
            {"record_kind": "reduce_sum_issue_boundary", "clock_cycles": 4,
             "trace_slot": 1, "boundary": "end", "source_loop_ordinal": 0, "relative_iteration": 1},
        ]
        with self.assertRaisesRegex(RuntimeError, "selected exact source-loop iteration"):
            validate(case, trace, {"tileop_families": ["gemm_dense", "reduce"]})

    def test_multiblock_trace_checks_pairs_without_global_clock_order(self):
        lineage = []
        for slot, block in enumerate(([0, 0, 0], [0, 0, 0], [3, 0, 0], [3, 0, 0])):
            lineage.append({"trace_slot": slot, "boundary": "begin" if slot % 2 == 0 else "end",
                            "block": block, "canonical_family": "copy",
                            "lineage": "compiler_exact_lineage", "source_ordinal": 0,
                            "compiler_provenance_annotation": "tl.provenance_ordinal"})
        case = SimpleNamespace(instrumented=True, trace_metadata={
            "timestamps_per_sample": 4, "selected_blocks": [[0, 0, 0], [3, 0, 0]],
        }, lineage=lineage)
        trace = [
            {"trace_slot": 0, "boundary": "begin", "source_ordinal": 0, "block": [0, 0, 0], "clock_cycles": 100, "record_kind": "copy_begin_issue_boundary"},
            {"trace_slot": 1, "boundary": "end", "source_ordinal": 0, "block": [0, 0, 0], "clock_cycles": 120, "record_kind": "copy_end_issue_boundary"},
            {"trace_slot": 2, "boundary": "begin", "source_ordinal": 0, "block": [3, 0, 0], "clock_cycles": 90, "record_kind": "copy_begin_issue_boundary"},
            {"trace_slot": 3, "boundary": "end", "source_ordinal": 0, "block": [3, 0, 0], "clock_cycles": 110, "record_kind": "copy_end_issue_boundary"},
        ]
        result = validate(case, trace, {"tileop_families": ["copy"]})
        self.assertFalse(result["trace_clock_monotonic"])
        self.assertTrue(result["trace_pair_monotonic"])
        self.assertEqual(result["selected_source_ordinals"], [0])

    def test_swapped_boundary_is_rejected(self):
        trace = [dict(self.trace[0]), dict(self.trace[1])]
        trace[0]["boundary"] = "end"
        with self.assertRaisesRegex(RuntimeError, "must be the begin boundary"):
            validate(_case(), trace, {})

    def test_record_ordinal_must_match_lineage(self):
        case = _compiler_multi_case()
        trace = [
            {"trace_slot": slot, "boundary": "begin" if slot % 2 == 0 else "end",
             "source_ordinal": ordinal, "record_kind": "copy_issue_boundary", "clock_cycles": slot + 1}
            for slot, ordinal in enumerate((3, 3, 99, 4))
        ]
        with self.assertRaisesRegex(RuntimeError, "source ordinal"):
            validate(case, trace, {})
