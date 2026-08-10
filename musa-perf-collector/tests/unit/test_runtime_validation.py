from __future__ import annotations

import unittest

from musa_perf_collector.core.runtime_validation import validate_session


class RuntimeValidationTest(unittest.TestCase):
    def _bundle(self):
        manifest = {
            "schema_version": "musa-perf-collector/runtime-measurement-v1", "status": "complete",
            "requested_mode": "deep", "actual_mode": "deep", "instrumented": True,
            "baseline_kernel_hash": "baseline", "instrumented_kernel_hash": "instrumented",
        }
        deep = {
            "status": "complete", "instrumented_correctness": {"status": "pass"},
            "trace_validation": {"status": "pass"},
            "selection": {name: {} for name in ("case", "kernel", "stream", "launch", "block")},
            "trace": [{"launch_index": 1}, {"launch_index": 1}],
            "readback": {"runs": [{"launch_index": 1}]},
            "instrumented_launch_usage": {"used": 1, "budget": 1},
        }
        resolved = {"deep": {"launch_indices": [1], "max_trace_records": 2}}
        return manifest, deep, resolved

    def test_complete_deep_bundle_passes(self):
        manifest, deep, resolved = self._bundle()
        result = validate_session(manifest, {"status": "pass"}, {"instrumented": False}, [], deep, resolved)
        self.assertEqual(result["status"], "pass")

    def test_trace_launch_or_budget_conflict_fails(self):
        manifest, deep, resolved = self._bundle()
        deep["trace"].append({"launch_index": 2})
        result = validate_session(manifest, {"status": "pass"}, {"instrumented": False}, [], deep, resolved)
        self.assertEqual(result["status"], "fail")
        self.assertEqual({item["code"] for item in result["violations"]},
                         {"TRACE_LAUNCH_SET_MISMATCH", "TRACE_RECORD_BUDGET_EXCEEDED"})

    def test_complete_cannot_hide_diagnostics_or_hash_alias(self):
        manifest, deep, resolved = self._bundle()
        manifest["instrumented_kernel_hash"] = "baseline"
        result = validate_session(manifest, {"status": "pass"}, {"instrumented": False},
                                  [{"code": "DEGRADED"}], deep, resolved)
        self.assertEqual(result["status"], "fail")
        self.assertEqual({item["code"] for item in result["violations"]},
                         {"COMPLETE_HAS_DIAGNOSTICS", "KERNEL_HASH_NOT_ISOLATED"})

    def test_failed_correctness_requires_invalid_status(self):
        manifest, _, resolved = self._bundle()
        manifest.update({"requested_mode": "light", "actual_mode": "light", "instrumented": False})
        result = validate_session(manifest, {"status": "fail"}, {"instrumented": False}, [], None, resolved)
        self.assertEqual(result["status"], "fail")
        self.assertIn("CORRECTNESS_NOT_INVALID", {item["code"] for item in result["violations"]})

    def test_required_stability_must_pass(self):
        manifest, deep, resolved = self._bundle()
        resolved["stability"] = {"required": True}
        result = validate_session(manifest, {"status": "pass"}, {"instrumented": False}, [], deep, resolved,
                                  stability={"status": "insufficient"})
        self.assertIn("STABILITY_GATE_INVALID", {item["code"] for item in result["violations"]})

    def test_stream_sync_failure_invalidates_bundle(self):
        manifest, deep, resolved = self._bundle()
        result = validate_session(manifest, {"status": "pass"}, {"instrumented": False}, [], deep, resolved,
                                  stream_sync={"status": "fail"})
        self.assertIn("STREAM_SYNC_GATE_INVALID", {item["code"] for item in result["violations"]})


if __name__ == "__main__":
    unittest.main()
