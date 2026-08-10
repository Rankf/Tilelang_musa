from __future__ import annotations

import unittest

from musa_perf_collector.collectors.runtime.stream_sync import audit


class RuntimeStreamSyncTest(unittest.TestCase):
    def test_baseline_and_async_readback_evidence_pass(self):
        samples = [{"sample_index": 0, "stream_id": 7, "completion_event_synchronized": True,
                    "synchronization_scope": "batch_end_event"}]
        deep = {"readback": {"actual_policy": "async_after_window", "runs": [{"launch_index": 2, "details": {
            "producer_stream_id": 7, "collector_stream_id": 9, "producer_event_recorded": True,
            "collector_waited_for_producer": True, "copy_done_event_recorded": True,
        }}]}}
        self.assertEqual(audit(samples, 7, None, deep)["status"], "pass")

    def test_stream_mismatch_and_missing_wait_fail(self):
        samples = [{"sample_index": 0, "stream_id": 8, "completion_event_synchronized": False}]
        deep = {"readback": {"actual_policy": "async_after_window", "runs": [{"launch_index": 2, "details": {
            "producer_stream_id": 7, "collector_stream_id": 7, "producer_event_recorded": True,
            "collector_waited_for_producer": False, "copy_done_event_recorded": True,
        }}]}}
        result = audit(samples, 7, None, deep)
        self.assertEqual(result["status"], "fail")
        self.assertEqual({item["code"] for item in result["violations"]},
                         {"BASELINE_STREAM_MISMATCH", "BASELINE_EVENT_COMPLETION_MISSING",
                          "ASYNC_READBACK_STREAM_SYNC_INVALID"})


if __name__ == "__main__":
    unittest.main()
