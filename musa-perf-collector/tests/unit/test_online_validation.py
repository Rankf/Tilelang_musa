from types import SimpleNamespace
import unittest

from musa_perf_collector.online.validation import audit_session


class OnlineSessionAuditTest(unittest.TestCase):
    def setUp(self):
        self.budget = SimpleNamespace(
            max_instrumented_launches=1,
            max_trace_records=2,
            max_trace_bytes=16,
        )
        self.launches = [{
            "kernel_id": "kernel_a", "launch_index": 0, "status": "instrumented",
            "schema": "online-launch-observation-v1",
            "pid": 10, "device_id": 0, "stream_id": 7,
            "selection_reason": "selected"}]
        self.trace = [
            {"kernel_id": "kernel_a", "launch_index": 0, "slot": 0,
             "generation": 1, "op_id": "op_a", "ordinal": 3,
             "source_kind": "gemm", "boundary": "begin", "block": [0, 0, 0],
             "lowered_instance_index": None, "lowering_semantics": "single",
             "clock_cycles": 100, "valid": True, "interval_cycles": None,
             "interval_semantics": None, "kernel_event_ms": 0.1,
             "schema": "online-tile-trace-v1",
             "evidence_level": "compiler_exact_instrumented_issue_boundary",
             "plan_sha256": "plan_a", "readback_policy": "native_deferred_at_flush",
             "process_id": 10, "device_id": 0, "stream_id": 7,
             "instrumented_variant_launch_index": 0,
             "instrumented_variant_cold_start": True,
             "tileop": {"op_id": "op_a"}},
            {"kernel_id": "kernel_a", "launch_index": 0, "slot": 1,
             "generation": 1, "op_id": "op_a", "ordinal": 3,
             "source_kind": "gemm", "boundary": "end", "block": [0, 0, 0],
             "lowered_instance_index": None, "lowering_semantics": "single",
             "clock_cycles": 112, "valid": True, "interval_cycles": 12,
             "interval_semantics": "lane0_issue_boundary_envelope",
             "kernel_event_ms": 0.1, "schema": "online-tile-trace-v1",
             "evidence_level": "compiler_exact_instrumented_issue_boundary",
             "plan_sha256": "plan_a", "readback_policy": "native_deferred_at_flush",
             "process_id": 10, "device_id": 0, "stream_id": 7,
             "instrumented_variant_launch_index": 0,
             "instrumented_variant_cold_start": True,
             "tileop": {"op_id": "op_a"}},
        ]
        self.layout = [{
            key: record[key] for key in (
                "slot", "op_id", "ordinal", "source_kind", "boundary",
                "block", "lowered_instance_index", "lowering_semantics")
        } for record in self.trace]

    def audit(self, **updates):
        values = {
            "launches": self.launches,
            "trace_records": self.trace,
            "enriched_records": self.trace,
            "expected_trace_extents": {"kernel_a": 2},
            "expected_trace_layouts": {"kernel_a": self.layout},
            "expected_plan_hashes": {"kernel_a": "plan_a"},
            "expected_readback_policies": {
                "kernel_a": "native_deferred_at_flush"},
            "total_launch_count": 1,
            "fast_path_launch_count": 0,
            "fast_path_skip_reasons": {},
            "instrumented_launch_count": 1,
            "instrumented_dispatch_count": 1,
            "reserved_instrumented_launches": 1,
            "reserved_trace_records": 2,
            "reserved_trace_bytes": 16,
            "budget": self.budget,
        }
        values.update(updates)
        return audit_session(**values)

    def test_consistent_session_passes(self):
        result = self.audit()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["violation_count"], 0)

    def test_launch_error_and_missing_slots_fail(self):
        result = self.audit(
            launches=[{"kernel_id": "kernel_a", "launch_index": 0,
                       "status": "instrumented_error"}],
            trace_records=[], enriched_records=[], instrumented_launch_count=0)
        self.assertEqual(result["status"], "fail")
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("LAUNCH_ERROR_STATUS", codes)

    def test_trace_from_baseline_and_budget_overrun_fail(self):
        result = self.audit(
            launches=[{"kernel_id": "kernel_a", "launch_index": 0, "status": "baseline"}],
            instrumented_launch_count=0,
            reserved_trace_records=3,
        )
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("TRACE_WITHOUT_INSTRUMENTED_LAUNCH", codes)
        self.assertIn("TRACE_RECORD_BUDGET_EXCEEDED", codes)

    def test_invalid_slot_cannot_publish_interval(self):
        trace = [dict(item) for item in self.trace]
        trace[1].update(valid=False, interval_cycles=1)
        result = self.audit(trace_records=trace, enriched_records=trace)
        self.assertIn(
            "INVALID_SLOT_HAS_INTERVAL",
            {item["code"] for item in result["violations"]})

    def test_concurrent_completion_order_does_not_create_false_gap(self):
        launches = [
            {"kernel_id": "kernel_a", "launch_index": 1, "status": "baseline",
             "schema": "online-launch-observation-v1",
             "pid": 10, "device_id": 0, "stream_id": 7},
            self.launches[0],
        ]
        result = self.audit(launches=launches, total_launch_count=2)
        self.assertNotIn(
            "INVALID_OR_DUPLICATE_LAUNCH_INDEX",
            {item["code"] for item in result["violations"]})

    def test_exact_layout_and_enrichment_mismatches_fail(self):
        trace = [dict(item) for item in self.trace]
        trace[1]["op_id"] = "wrong_op"
        enriched = [dict(item) for item in self.trace]
        enriched[0]["tileop"] = {"op_id": "wrong_op"}
        result = self.audit(trace_records=trace, enriched_records=enriched)
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("TRACE_LAYOUT_FIELD_MISMATCH", codes)
        self.assertIn("ENRICHED_TRACE_FIELD_MISMATCH", codes)
        self.assertIn("ENRICHED_TILEOP_ID_MISMATCH", codes)

    def test_total_launch_accounting_gap_fails(self):
        result = self.audit(total_launch_count=2)
        self.assertIn(
            "TOTAL_LAUNCH_ACCOUNTING_MISMATCH",
            {item["code"] for item in result["violations"]})

    def test_generation_reuse_across_launches_fails(self):
        launches = self.launches + [{
            "kernel_id": "kernel_a", "launch_index": 1, "status": "instrumented",
            "schema": "online-launch-observation-v1",
            "pid": 10, "device_id": 0, "stream_id": 7}]
        second = [{
            **item, "launch_index": 1,
            "instrumented_variant_launch_index": 1,
            "instrumented_variant_cold_start": False,
        } for item in self.trace]
        result = self.audit(
            launches=launches, trace_records=self.trace + second,
            enriched_records=self.trace + second, total_launch_count=2,
            instrumented_launch_count=2, instrumented_dispatch_count=2,
            reserved_instrumented_launches=2, reserved_trace_records=4,
            reserved_trace_bytes=32,
            budget=SimpleNamespace(
                max_instrumented_launches=2,
                max_trace_records=4, max_trace_bytes=32))
        self.assertIn(
            "TRACE_GENERATION_REUSED",
            {item["code"] for item in result["violations"]})

    def test_plan_launch_event_interval_and_fast_reason_corruption_fail(self):
        trace = [dict(item) for item in self.trace]
        trace[0].update(plan_sha256="wrong", process_id=99, kernel_event_ms=0.2)
        trace[1]["interval_cycles"] = 11
        result = self.audit(
            trace_records=trace, enriched_records=self.trace,
            total_launch_count=2, fast_path_launch_count=1,
            fast_path_skip_reasons={"launch_not_selected": 0})
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("TRACE_PLAN_HASH_MISMATCH", codes)
        self.assertIn("TRACE_LAUNCH_IDENTITY_MISMATCH", codes)
        self.assertIn("TRACE_LAUNCH_EVENT_TIME_MISMATCH", codes)
        self.assertIn("TRACE_INTERVAL_MISMATCH", codes)
        self.assertIn("FAST_PATH_REASON_ACCOUNTING_MISMATCH", codes)

    def test_reordered_layout_cannot_bypass_interval_audit(self):
        trace = [dict(item) for item in self.trace]
        trace[1]["interval_cycles"] = 11
        result = self.audit(
            trace_records=trace,
            enriched_records=trace,
            expected_trace_layouts={"kernel_a": list(reversed(self.layout))},
        )
        self.assertIn(
            "TRACE_INTERVAL_MISMATCH",
            {item["code"] for item in result["violations"]},
        )

    def test_pair_validity_mismatch_fails(self):
        trace = [dict(item) for item in self.trace]
        trace[0]["valid"] = False
        result = self.audit(trace_records=trace, enriched_records=trace)
        self.assertIn(
            "TRACE_PAIR_VALIDITY_MISMATCH",
            {item["code"] for item in result["violations"]},
        )

    def test_lowered_instance_layout_requires_zero_based_contiguous_indices(self):
        layout = [dict(item) for item in self.layout]
        for item in layout:
            item["lowering_semantics"] = "lowered_instance"
            item["lowered_instance_index"] = 1
        result = self.audit(expected_trace_layouts={"kernel_a": layout})
        self.assertIn(
            "NONCONTIGUOUS_EXPECTED_LOWERED_INSTANCES",
            {item["code"] for item in result["violations"]},
        )

    def test_each_expected_instance_requires_one_begin_and_one_end(self):
        layout = [dict(item) for item in self.layout]
        layout[1]["boundary"] = "begin"
        result = self.audit(expected_trace_layouts={"kernel_a": layout})
        self.assertIn(
            "EXPECTED_TRACE_PAIR_LAYOUT_MISMATCH",
            {item["code"] for item in result["violations"]},
        )

    def test_malformed_layout_is_a_violation_not_an_audit_exception(self):
        layout = [dict(item) for item in self.layout]
        layout[0].update(
            block=None, op_id=[], ordinal=True, lowering_semantics={})
        layout[1].update(boundary=[], lowered_instance_index={})
        result = self.audit(expected_trace_layouts={"kernel_a": layout})
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("MALFORMED_EXPECTED_TRACE_BLOCK", codes)
        self.assertIn("INVALID_EXPECTED_TRACE_BOUNDARY", codes)
        self.assertIn("INVALID_EXPECTED_TRACE_OP_ID", codes)
        self.assertIn("INVALID_EXPECTED_TRACE_ORDINAL", codes)
        self.assertIn("INVALID_EXPECTED_LOWERING_SEMANTICS", codes)

    def test_boolean_and_string_identity_fields_fail_strict_schema(self):
        trace = [dict(item) for item in self.trace]
        trace[0]["generation"] = True
        trace[1]["generation"] = "1"
        launches = [dict(item) for item in self.launches]
        launches[0]["launch_index"] = True
        result = self.audit(
            launches=launches, trace_records=trace, enriched_records=trace)
        codes = {item["code"] for item in result["violations"]}
        self.assertIn("MALFORMED_LAUNCH_INDEX", codes)
        self.assertIn("MALFORMED_TRACE_GENERATION", codes)
