from __future__ import annotations

import unittest

from musa_perf_collector.collectors.runtime.sampling import (BudgetExceeded, CollectionBudget, apply_budget,
                                                             choose_indices, linear_block_coordinates,
                                                             identity_selection_evidence, selection_evidence,
                                                             stable_selected)
from musa_perf_collector.runtime import _bound_deep_config, _resolve_deep_scope


class RuntimeSamplingTest(unittest.TestCase):
    def test_probability_selection_is_stable(self):
        first = [stable_selected(str(value), seed=7, probability=0.4) for value in range(20)]
        second = [stable_selected(str(value), seed=7, probability=0.4) for value in range(20)]
        self.assertEqual(first, second)

    def test_selector_precedence_is_exact_then_periodic_then_probability(self):
        selector = {"indices": [2, 4], "every": 2, "probability": 1.0}
        self.assertEqual(choose_indices(range(6), selector, seed=7), [2, 4])

    def test_probability_evidence_is_replayable_and_budgeted(self):
        selected, evidence, diagnostic = selection_evidence(
            range(20), {"probability": 1.0}, seed=19, dimension="launch",
            budget_limit=3, policy="degrade")
        self.assertEqual(selected, [0, 1, 2])
        self.assertEqual(evidence["algorithm"], "sha256_threshold_v1")
        self.assertEqual(evidence["selected_before_budget"], list(range(20)))
        self.assertTrue(evidence["budget_truncated"])
        self.assertTrue(evidence["candidates_hash"].startswith("sha256:"))
        self.assertEqual(diagnostic["label"], "launch_selection")

    def test_exact_selection_rejects_duplicate_and_out_of_range_indices(self):
        for indices in ([1, 1], [0, 4]):
            with self.assertRaises(ValueError):
                selection_evidence(range(4), {"indices": indices}, seed=0, dimension="block",
                                   budget_limit=4, policy="fail")

    def test_static_grid_linearization_is_x_fastest(self):
        self.assertEqual(linear_block_coordinates([2, 2]), [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ])

    def test_named_scope_probability_is_stable(self):
        first, evidence = identity_selection_evidence(
            ["kernel_a", "kernel_b"], {"probability": 0.5}, seed=29, dimension="kernel")
        second, _ = identity_selection_evidence(
            ["kernel_a", "kernel_b"], {"probability": 0.5}, seed=29, dimension="kernel")
        self.assertEqual(first, second)
        self.assertEqual(evidence["algorithm"], "sha256_threshold_v1")

    def test_case_kernel_stream_scope_can_exclude_only_deep(self):
        class Case:
            name = "case_a"
            kernel_id = "kernel_a"
        resolved = {"sampling_seed": 7, "deep": {"case_selector": {"values": ["case_b"]}}}
        self.assertFalse(_resolve_deep_scope(resolved, Case(), 123))
        evidence = resolved["deep"]["selection_evidence"]
        self.assertEqual(evidence["case"]["selected"], [])
        self.assertEqual(evidence["kernel"]["selected"], ["kernel_a"])
        self.assertEqual(evidence["stream"]["selected"], ["123"])

    def test_every_selector_and_budget_degrade(self):
        self.assertEqual(choose_indices(range(8), {"every": 3, "start": 1}, seed=0), [1, 4, 7])
        selected, diagnostic = apply_budget([1, 2, 3], 2, label="blocks", policy="degrade")
        self.assertEqual(selected, [1, 2])
        self.assertEqual(diagnostic["code"], "BUDGET_EXCEEDED")

    def test_budget_fail_is_not_silent(self):
        with self.assertRaises(BudgetExceeded):
            apply_budget([1, 2], 1, label="blocks", policy="fail")

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            CollectionBudget.from_config({"max_trace_records": -1})

    def test_deep_selection_is_reduced_before_compilation(self):
        resolved = {
            "deep": {"launch_indices": [0, 1], "block_coordinates": [[0, 0, 0], [1, 0, 0]],
                     "tileop_families": ["gemm_dense", "reduce", "copy"], "max_trace_records": 999},
            "budget": {"on_budget_exceeded": "degrade", "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 2,
                       "max_iterations_per_tileop": 1, "max_trace_records": 4, "max_trace_bytes": 64},
        }
        diagnostics = []
        _bound_deep_config(resolved, diagnostics)
        self.assertEqual(resolved["deep"]["launch_indices"], [0])
        self.assertEqual(resolved["deep"]["block_coordinates"], [[0, 0, 0]])
        self.assertEqual(resolved["deep"]["tileop_families"], ["gemm_dense", "reduce"])
        self.assertEqual(resolved["deep"]["max_trace_records"], 4)
        self.assertEqual(len(diagnostics), 4)

    def test_invalid_or_duplicate_block_coordinates_are_rejected(self):
        base = {
            "deep": {"launch_indices": [0], "tileop_families": ["copy"]},
            "budget": {"on_budget_exceeded": "degrade", "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 2, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 4, "max_trace_bytes": 64},
        }
        for coordinates in ([[0, 0]], [[0, 0, 0], [0, 0, 0]], [[-1, 0, 0]]):
            resolved = {**base, "deep": {**base["deep"], "block_coordinates": coordinates}}
            with self.assertRaisesRegex(RuntimeError, "block coordinate"):
                _bound_deep_config(resolved, [])

    def test_block_coordinate_must_be_inside_runtime_grid(self):
        resolved = {
            "deep": {"launch_indices": [0], "block_coordinates": [[4, 0, 0]], "tileop_families": ["copy"]},
            "budget": {"on_budget_exceeded": "degrade", "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 2, "max_trace_bytes": 16},
        }
        with self.assertRaisesRegex(RuntimeError, "outside runtime grid"):
            _bound_deep_config(resolved, [], launch={"grid": [4, 1, 1]})

    def test_dynamic_runtime_grid_is_rejected_for_deep_blocks(self):
        resolved = {
            "deep": {"launch_indices": [0], "block_coordinates": [[0, 0, 0]], "tileop_families": ["copy"]},
            "budget": {"on_budget_exceeded": "degrade", "max_instrumented_launches": 1,
                       "max_sampled_blocks_per_launch": 1, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 2, "max_trace_bytes": 16},
        }
        with self.assertRaisesRegex(RuntimeError, "integer runtime grid"):
            _bound_deep_config(resolved, [], launch={"grid": ["n", 1, 1]})

    def test_periodic_launch_and_block_selectors_drive_resolved_config(self):
        resolved = {
            "sampling_seed": 23,
            "deep": {"launch_candidate_count": 6, "launch_selector": {"every": 2, "start": 1},
                     "block_selector": {"indices": [0, 3]}, "tileop_families": ["copy"]},
            "budget": {"on_budget_exceeded": "degrade", "max_instrumented_launches": 3,
                       "max_sampled_blocks_per_launch": 2, "max_tileops_per_block": 1,
                       "max_iterations_per_tileop": 1, "max_trace_records": 8, "max_trace_bytes": 64},
        }
        diagnostics = []
        _bound_deep_config(resolved, diagnostics, launch={"grid": [4, 1, 1]})
        self.assertEqual(resolved["deep"]["launch_indices"], [1, 3, 5])
        self.assertEqual(resolved["deep"]["block_coordinates"], [[0, 0, 0], [3, 0, 0]])
        self.assertEqual(resolved["deep"]["selection_evidence"]["launch"]["selector_kind"], "periodic")
        self.assertEqual(resolved["deep"]["selection_evidence"]["block"]["selected_coordinates"],
                         [[0, 0, 0], [3, 0, 0]])
        self.assertEqual(diagnostics, [])
