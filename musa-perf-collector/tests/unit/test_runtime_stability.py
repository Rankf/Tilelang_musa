from __future__ import annotations

import json
import unittest

from musa_perf_collector.collectors.runtime.stability import assess


def snapshot(temp: float, freq: float, power: float, timestamp: float) -> dict:
    return {"returncode": 0, "host_monotonic_s": timestamp, "raw_json": json.dumps({"GPU": [{
        "Temperature": {"GPU Current Temp": f"{temp}C"}, "Clocks": {"Graphics": f"{freq}MHz"},
        "Power Readings": {"Power Draw ": f"{power}W"},
    }]})}


class RuntimeStabilityTest(unittest.TestCase):
    def test_stable_window_uses_snapshots_active_and_timing(self):
        active = {"samples": [{"host_monotonic_s": 2.0, "temperature_c": 41.0,
                               "frequency_mhz": 1750.0, "power_w": 410.0}]}
        timing = [{"per_launch_ms": value} for value in (1.0, 1.01, 1.02, 1.01)]
        result = assess(snapshot(40, 1750, 400, 1), active, snapshot(42, 1750, 405, 3), timing,
                        {"required": True})
        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["thermal"]["point_count"], 3)
        self.assertEqual(result["semantics"], "session_window_only_not_per_tileop")

    def test_temperature_frequency_and_timing_drift_are_explicit(self):
        active = {"samples": [{"host_monotonic_s": 2.0, "temperature_c": 60.0,
                               "frequency_mhz": 1500.0, "power_w": 500.0}]}
        timing = [{"per_launch_ms": value} for value in (1.0, 1.0, 1.4, 1.4)]
        result = assess(snapshot(40, 1750, 400, 1), active, snapshot(70, 1200, 450, 3), timing,
                        {"max_temperature_delta_c": 5, "max_frequency_delta_pct": 5,
                         "max_timing_drift_pct": 10})
        self.assertEqual(result["status"], "unstable")
        self.assertEqual({item["code"] for item in result["reasons"]},
                         {"TEMPERATURE_DRIFT_EXCEEDED", "FREQUENCY_DRIFT_EXCEEDED", "TIMING_DRIFT_EXCEEDED"})

    def test_missing_evidence_is_insufficient_not_stable(self):
        result = assess({}, {"samples": []}, {}, [], {"required": True})
        self.assertEqual(result["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
