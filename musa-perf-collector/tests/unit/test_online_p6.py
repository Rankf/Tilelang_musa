"""P6 online hook/config/session contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from musa_perf_collector.online.config import load_online_config
from musa_perf_collector.online.dual_variant import OnlineDualVariantKernel
from musa_perf_collector.online.session import OnlineObservedKernel, OnlineSession
from musa_perf_collector.online.plan import build_plan

try:
    from musa_perf_collector.online._native_dispatch import NativePeriodicDispatcher
except ImportError:
    NativePeriodicDispatcher = None


class _Baseline:
    artifact = "baseline-artifact"
    prim_func = "baseline-primfunc"

    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return value + 1

    def get_kernel_source(self):
        return "baseline-source"


def _write_config(root: Path, **updates) -> Path:
    value = {
        "schema": "musa-perf-collector/online-config-v1",
        "enabled": True,
        "compile_policy": "observe_only",
        "correctness_policy": "application_owned",
        "telemetry": {"enabled": False},
        "output": str(root / "results"),
    }
    value.update(updates)
    path = root / "online.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@unittest.skipIf(NativePeriodicDispatcher is None, "native dispatcher is not built")
class NativePeriodicDispatcherTest(unittest.TestCase):
    def test_every_start_kwargs_and_attribute_forwarding(self):
        baseline_calls = []
        selected_indices = []

        class Baseline:
            artifact = "native-baseline-artifact"

            def __call__(self, value, *, scale=1):
                baseline_calls.append(value)
                return value * scale

        def selected(index, value, *, scale=1):
            selected_indices.append(index)
            return value * scale, None

        dispatcher = NativePeriodicDispatcher(
            Baseline(), selected, Baseline(), 4, 1, os.getpid(), None,
            {"dispatch_path": "native_c_vectorcall"})
        self.assertEqual(
            [dispatcher(value, scale=2) for value in range(10)],
            [value * 2 for value in range(10)])
        self.assertEqual(selected_indices, [1, 5, 9])
        self.assertEqual(baseline_calls, [0, 2, 3, 4, 6, 7, 8])
        self.assertEqual(dispatcher.artifact, "native-baseline-artifact")
        self.assertEqual(dispatcher.online_launch_summary(), {
            "total_launch_count": 10,
            "fast_path_launch_count": 7,
            "fast_path_skip_reasons": {"launch_not_selected": 7},
        })

    def test_terminal_reason_stops_future_selected_callbacks(self):
        selected_indices = []

        def selected(index, value):
            selected_indices.append(index)
            return value, "instrumented_launch_budget_exhausted"

        dispatcher = NativePeriodicDispatcher(
            lambda value: value, selected, lambda value: value,
            2, 0, os.getpid(), None, {})
        self.assertEqual([dispatcher(value) for value in range(5)], list(range(5)))
        self.assertEqual(selected_indices, [0])
        self.assertEqual(dispatcher.online_launch_summary(), {
            "total_launch_count": 5,
            "fast_path_launch_count": 4,
            "fast_path_skip_reasons": {
                "instrumented_launch_budget_exhausted": 2,
                "launch_not_selected": 2,
            },
        })

    def test_fast_path_error_is_recorded_without_replacing_exception(self):
        errors = []

        def baseline(_value):
            raise LookupError("original failure")

        dispatcher = NativePeriodicDispatcher(
            baseline, lambda index, value: (value, None), baseline, 64, 1,
            os.getpid(), lambda *items: errors.append(items), {})
        with self.assertRaisesRegex(LookupError, "original failure"):
            dispatcher(3)
        self.assertEqual(errors, [(0, "launch_not_selected", "LookupError")])

    def test_process_mismatch_uses_fork_safe_callable(self):
        calls = []
        dispatcher = NativePeriodicDispatcher(
            lambda value: calls.append("baseline") or value,
            lambda index, value: (calls.append("selected") or value, None),
            lambda value: calls.append("forked") or value,
            64, 0, os.getpid() + 1, None, {})
        self.assertEqual(dispatcher(7), 7)
        self.assertEqual(calls, ["forked"])


def _write_kernel_metadata(artifact: Path, *, grid=(4, 1, 1), threads=(128, 1, 1)) -> None:
    (artifact / "metadata" / "kernel.json").write_text(json.dumps({
        "launch_context": {
            "block_axes": {f"blockIdx.{axis}": value for axis, value in zip("xyz", grid)},
            "thread_axes": {f"threadIdx.{axis}": value for axis, value in zip("xyz", threads)},
            "block_count": grid[0] * grid[1] * grid[2],
            "thread_count": threads[0] * threads[1] * threads[2],
        }
    }), encoding="utf-8")


def _write_persisted_plan(session: OnlineSession, kernel_id: str, plan: SimpleNamespace) -> None:
    path = session.directory / "kernels" / kernel_id / "instrumentation_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "online-instrumentation-plan-v1",
        "trace_extent": plan.trace_extent,
        "slot_layout": list(plan.slot_layout),
        "plan_sha256": plan.plan_sha256,
    }), encoding="utf-8")


class OnlineConfigTest(unittest.TestCase):
    def test_config_is_validated_and_content_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = load_online_config(_write_config(root))
            second = load_online_config(_write_config(root))
            self.assertEqual(first.config_sha256, second.config_sha256)
            self.assertTrue(first.config_sha256.startswith("sha256:"))
            self.assertEqual(first.output, (root / "results").resolve())
            self.assertEqual(first.launch_selector, {"every": 64, "start": 0})
            self.assertEqual(first.event_timing_every, 1)

    def test_periodic_interval_is_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_online_config(_write_config(
                root, launch_selector={"every": 32, "start": 7}))
            self.assertEqual(config.launch_selector, {"every": 32, "start": 7})

    def test_event_timing_sampling_is_configurable_and_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_online_config(_write_config(root, event_timing_every=8))
            self.assertEqual(config.event_timing_every, 8)
            with self.assertRaisesRegex(ValueError, "event_timing_every"):
                load_online_config(_write_config(root, event_timing_every=0))

    def test_unknown_keys_and_invalid_probability_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unknown online config"):
                load_online_config(_write_config(root, typo=True))
            with self.assertRaisesRegex(ValueError, "probability"):
                load_online_config(_write_config(root, launch_selector={"probability": 1.5}))

    def test_dual_variant_requires_exact_artifact_and_op_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "lineage_artifact"):
                load_online_config(_write_config(root, compile_policy="eager_dual_variant"))
            artifact = root / "lineage.json"
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace_op_ids"):
                load_online_config(_write_config(
                    root, compile_policy="eager_dual_variant", lineage_artifact=str(artifact)))

    def test_telemetry_configuration_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_online_config(_write_config(
                root, telemetry={"enabled": True, "device": 1}))
            self.assertEqual(config.telemetry, {"enabled": True, "device": 1})
            with self.assertRaisesRegex(ValueError, "telemetry"):
                load_online_config(_write_config(root, telemetry={"interval_ms": 1}))

    def test_unsupported_selectors_and_evidence_claims_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "kernel_selector"):
                load_online_config(_write_config(root, kernel_selector={"indices": [0]}))
            with self.assertRaisesRegex(ValueError, "unknown launch_selector"):
                load_online_config(_write_config(
                    root, launch_selector={"probability": 0.5, "typo": True}))
            with self.assertRaisesRegex(ValueError, "block_selector"):
                load_online_config(_write_config(root, block_selector={"probability": 0.5}))
            with self.assertRaisesRegex(ValueError, "correctness_policy"):
                load_online_config(_write_config(root, correctness_policy="application_reported"))

    def test_zero_pool_capacity_and_non_integer_record_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "max_pending_readbacks"):
                load_online_config(_write_config(
                    root, budget={"max_pending_readbacks": 0}))
            with self.assertRaisesRegex(ValueError, "max_trace_records"):
                load_online_config(_write_config(
                    root, budget={"max_trace_records": 2.5}))
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_online_config(_write_config(
                    root, budget={"max_total_duration_s": float("nan")}))
            with self.assertRaisesRegex(ValueError, "must not exceed 4096"):
                load_online_config(_write_config(
                    root, budget={"max_pending_readbacks": 4097}))


class OnlineObserveSessionTest(unittest.TestCase):
    def test_flush_after_close_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(Path(directory))))
            session.close()
            with mock.patch.object(session, "_flush_locked") as flush_locked:
                session.flush()
            flush_locked.assert_not_called()

    def test_observe_wrapper_preserves_result_and_baseline_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_online_config(_write_config(Path(directory)))
            session = OnlineSession(config)
            request = SimpleNamespace(
                baseline_cache_identity="sha256:" + "1" * 64,
                prim_func_sha256="2" * 64,
                prim_func=SimpleNamespace(attrs={"global_symbol": "kernel"}),
                normalized_out_idx=(2,), original_out_idx=(-1,), target="musa",
                target_host=None, execution_backend="tvm_ffi", compile_origin="user_baseline",
            )
            # Mock tilelang.jit.hooks to avoid broken torch import in test env
            class _MockCompileAugmentation:
                def __init__(self, **kwargs):
                    self._wrap_kernel = kwargs.get("wrap_kernel")
                def wrap_kernel(self, baseline, internal_compile):
                    return self._wrap_kernel(baseline, internal_compile)
            mock_tilelang = __import__("unittest").mock.MagicMock()
            mock_tilelang.jit.hooks.CompileAugmentation = _MockCompileAugmentation
            with __import__("unittest").mock.patch.dict(sys.modules, {
                "tilelang": mock_tilelang,
                "tilelang.jit": mock_tilelang.jit,
                "tilelang.jit.hooks": mock_tilelang.jit.hooks,
            }):
                augmentation = session(request)
            baseline = _Baseline()
            wrapped = augmentation.wrap_kernel(baseline, None)
            self.assertIsInstance(wrapped, OnlineObservedKernel)
            self.assertEqual(wrapped(4), 5)
            self.assertEqual(baseline.calls, 1)
            self.assertEqual(wrapped.artifact, "baseline-artifact")
            self.assertEqual(wrapped.get_kernel_source(), "baseline-source")
            session.close()

            compiles = (session.directory / "compile_observations.jsonl").read_text().splitlines()
            launches = (session.directory / "launch_observations.jsonl").read_text().splitlines()
            self.assertEqual(len(compiles), 1)
            self.assertEqual(len(launches), 1)
            self.assertEqual(json.loads(launches[0])["status"], "observed")
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["launch_observation_count"], 1)
            self.assertFalse(validation["automatic_instrumentation"])

    def test_application_exception_is_recorded_and_re_raised(self):
        class Failing:
            def __call__(self):
                raise LookupError("application failure")

        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(Path(directory))))
            wrapped = OnlineObservedKernel(Failing(), session, "kernel_error")
            with self.assertRaisesRegex(LookupError, "application failure"):
                wrapped()
            session.flush()
            record = json.loads((session.directory / "launch_observations.jsonl").read_text().strip())
            self.assertEqual(record["status"], "application_error")
            self.assertEqual(record["error_type"], "LookupError")
            session.close()


class _PoolStub:
    readback_policy = "sync_at_flush"
    device = 0

    def __init__(self, completions=None, *, pending_count=0):
        self.completions = list(completions or [])
        self.poll_count = 0
        self.pending_count = pending_count

    def poll(self, *, wait=False):
        self.poll_count += 1
        values, self.completions = self.completions, []
        return values

    def try_acquire(self, _identity):
        raise AssertionError("instrumented pool must not be acquired")


class OnlineDualVariantReliabilityTest(unittest.TestCase):
    def test_terminal_budget_state_stays_on_direct_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(
                root, launch_candidate_count=0,
                launch_selector={"every": 2, "start": 0})))
            pool = _PoolStub()
            baseline = _Baseline()
            wrapper = OnlineDualVariantKernel(
                baseline, object(), session, "kernel_terminal",
                SimpleNamespace(plan_sha256="plan", trace_extent=2), pool)
            self.assertEqual([wrapper(4) for _ in range(5)], [5] * 5)
            self.assertEqual(baseline.calls, 5)
            self.assertEqual(pool.poll_count, 0)
            session.close()
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["total_launch_count"], 5)
            self.assertEqual(validation["fast_path_launch_count"], 5)
            self.assertEqual(validation["fast_path_skip_reasons"], {
                "launch_not_selected": 2,
                "sampling_window_closed": 3,
            })

    def test_default_every_64_selector_hits_zero_and_sixty_four(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(root)))
            decisions = [
                session.prepare_fast_launch("kernel", trace_extent=2)
                for _ in range(65)
            ]
            self.assertTrue(decisions[0][1])
            self.assertTrue(decisions[64][1])
            self.assertFalse(any(item[1] for item in decisions[1:64]))
            session.close()
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["total_launch_count"], 65)
            self.assertEqual(validation["fast_path_launch_count"], 63)
            self.assertEqual(validation["fast_path_skip_reasons"], {"launch_not_selected": 63})

    def test_unselected_periodic_launch_skips_metadata_pool_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(
                root, launch_selector={"every": 64, "start": 1})))
            pool = _PoolStub()
            wrapper = OnlineDualVariantKernel(
                _Baseline(), object(), session, "kernel_fast",
                SimpleNamespace(plan_sha256="plan", trace_extent=2), pool)
            with mock.patch.object(session, "stream_metadata") as metadata:
                self.assertEqual(wrapper(4), 5)
            metadata.assert_not_called()
            self.assertEqual(pool.poll_count, 0)
            session.close()
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["total_launch_count"], 1)
            self.assertEqual(validation["fast_path_launch_count"], 1)
            self.assertEqual(validation["launch_observation_count"], 0)

    def test_flush_timeout_is_bounded_and_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(root)))
            plan = SimpleNamespace(
                trace_extent=2,
                lowering_semantics="single",
                plan_sha256="plan_stuck",
                slot_layout=tuple({
                    "slot": slot,
                    "op_id": "op_stuck",
                    "ordinal": 0,
                    "source_kind": "copy",
                    "boundary": boundary,
                    "block": [0, 0, 0],
                    "lowered_instance_index": None,
                    "lowering_semantics": "single",
                } for slot, boundary in enumerate(("begin", "end"))),
            )
            _write_persisted_plan(session, "kernel_stuck", plan)
            session._trace_pools["kernel_stuck"] = (
                plan, _PoolStub(pending_count=1))
            with mock.patch(
                "musa_perf_collector.online.session._TRACE_DRAIN_TIMEOUT_S", 0.0):
                session.flush()
            diagnostics = [
                json.loads(line)
                for line in (session.directory / "diagnostics.jsonl").read_text().splitlines()
                if line
            ]
            self.assertEqual(diagnostics[-1]["code"], "ONLINE_TRACE_DRAIN_TIMEOUT")
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["status"], "partial")
            session._trace_pools.clear()
            session.close()

    def test_native_drain_failure_is_landed_as_invalid_evidence(self):
        class FailingPool:
            readback_policy = "native_deferred_at_flush"

            def poll(self, *, wait=False):
                raise RuntimeError("injected native drain failure")

        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(
                _write_config(Path(directory))))
            plan = SimpleNamespace(
                trace_extent=2, lowering_semantics="single", plan_sha256="plan_drain",
                slot_layout=())
            _write_persisted_plan(session, "kernel_drain_error", plan)
            session._trace_pools["kernel_drain_error"] = (plan, FailingPool())
            session.flush()
            diagnostics = [
                json.loads(line)
                for line in (session.directory / "diagnostics.jsonl").read_text().splitlines()
                if line
            ]
            self.assertEqual(diagnostics[-1]["code"], "ONLINE_TRACE_DRAIN_FAILED")
            validation = json.loads(
                (session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["status"], "invalid")
            session._trace_pools.clear()
            session.close()

    def test_persisted_evidence_read_failure_invalidates_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(Path(directory))))
            with mock.patch(
                "musa_perf_collector.online.session._read_jsonl_objects",
                side_effect=ValueError("injected persisted evidence failure"),
            ):
                session.flush()
            audit = json.loads((session.directory / "session_audit.json").read_text())
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(audit["status"], "fail")
            self.assertEqual(audit["violations"][0]["code"], "PERSISTED_EVIDENCE_READ_FAILED")
            self.assertEqual(validation["status"], "invalid")
            session.close()

    def test_deferred_pool_exhaustion_becomes_terminal_partial_state(self):
        class ExhaustedPool:
            readback_policy = "native_deferred_at_flush"
            device = 0

            def __init__(self):
                self.acquire_count = 0

            def poll(self, *, wait=False):
                return []

            def try_acquire(self, _identity):
                self.acquire_count += 1
                return None

        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(
                Path(directory), launch_selector={"every": 1, "start": 0})))
            pool = ExhaustedPool()
            wrapper = OnlineDualVariantKernel(
                _Baseline(), object(), session, "kernel_exhausted",
                SimpleNamespace(plan_sha256="plan", trace_extent=2), pool)
            self.assertEqual([wrapper(1) for _ in range(5)], [2] * 5)
            self.assertEqual(pool.acquire_count, 1)
            session.close()
            validation = json.loads(
                (session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["status"], "partial")
            self.assertEqual(validation["total_launch_count"], 5)
            self.assertEqual(validation["fast_path_launch_count"], 4)

    def test_abort_failure_does_not_mask_instrumented_error(self):
        class AbortFailPool:
            readback_policy = "sync_at_flush"
            device = 0

            def poll(self, *, wait=False):
                return []

            def try_acquire(self, _identity):
                return SimpleNamespace(device_tensor="trace")

            def update_identity(self, _lease, _identity):
                return None

            def record_start(self, _lease):
                return None

            def abort(self, _lease):
                raise RuntimeError("injected abort failure")

        def failing_instrumented(*_args, **_kwargs):
            raise LookupError("original instrumented failure")

        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(
                Path(directory), launch_selector={"probability": 1.0})))
            wrapper = OnlineDualVariantKernel(
                _Baseline(), failing_instrumented, session, "kernel_abort",
                SimpleNamespace(plan_sha256="plan", trace_extent=2),
                AbortFailPool())
            with self.assertRaisesRegex(LookupError, "original instrumented failure"):
                wrapper(1)
            self.assertEqual(
                session._diagnostics[-1]["code"], "ONLINE_TRACE_ABORT_FAILED")
            self.assertEqual(
                session._launch_records[-1]["error_type"], "LookupError")
            session.close()

    def test_selected_device_mismatch_never_acquires_trace_tensor(self):
        class DevicePool:
            readback_policy = "native_deferred_at_flush"
            device = 0

            def poll(self, *, wait=False):
                return []

            def try_acquire(self, _identity):
                raise AssertionError("device-mismatched trace pool was acquired")

        with tempfile.TemporaryDirectory() as directory:
            session = OnlineSession(load_online_config(_write_config(
                Path(directory), launch_selector={"every": 1, "start": 0})))
            wrapper = OnlineDualVariantKernel(
                _Baseline(), object(), session, "kernel_device",
                SimpleNamespace(plan_sha256="plan", trace_extent=2), DevicePool())
            with mock.patch.object(session, "stream_metadata", return_value=(1, 99)):
                self.assertEqual(wrapper(1), 2)
            self.assertEqual(
                session._diagnostics[-1]["code"], "ONLINE_TRACE_DEVICE_MISMATCH")
            self.assertEqual(
                session._launch_records[-1]["selection_reason"],
                "trace_device_mismatch")
            session.close()

    def test_trace_record_budget_is_aggregate_across_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(
                root,
                launch_selector={"probability": 1.0},
                budget={
                    "max_instrumented_launches": 3,
                    "max_trace_records": 4,
                    "max_trace_bytes": 32,
                    "max_pending_readbacks": 1,
                    "max_total_duration_s": 30,
                })))
            self.assertTrue(session.select_launch(
                "kernel", 0, 0, trace_extent=2)[0])
            self.assertTrue(session.select_launch(
                "kernel", 1, 0, trace_extent=2)[0])
            selected, reason, _ = session.select_launch(
                "kernel", 2, 0, trace_extent=2)
            self.assertFalse(selected)
            self.assertEqual(reason, "trace_record_budget_exhausted")
            session.close()

    def test_baseline_error_is_recorded_on_unselected_launch(self):
        class Failing:
            def __call__(self):
                raise LookupError("baseline failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(
                root, launch_candidate_count=0)))
            wrapper = OnlineDualVariantKernel(
                Failing(), object(), session, "kernel_error",
                SimpleNamespace(plan_sha256="plan", trace_extent=2), _PoolStub())
            with self.assertRaisesRegex(LookupError, "baseline failure"):
                wrapper()
            session.flush()
            record = json.loads((session.directory / "launch_observations.jsonl").read_text())
            self.assertEqual(record["status"], "application_error")
            self.assertEqual(record["error_type"], "LookupError")
            self.assertEqual(record["selection_reason"], "sampling_window_closed")
            validation = json.loads((session.directory / "session_validation.json").read_text())
            self.assertEqual(validation["status"], "invalid")
            self.assertEqual(validation["independent_audit"], "fail")
            session.close()

    def test_wrapper_never_uses_trace_pool_outside_creation_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(root)))
            pool = _PoolStub()
            wrapper = OnlineDualVariantKernel(
                _Baseline(), object(), session, "kernel_fork",
                SimpleNamespace(plan_sha256="plan"), pool)
            wrapper._owner_pid = -1
            self.assertEqual(wrapper(4), 5)
            self.assertEqual(pool.poll_count, 0)
            session.flush()
            record = json.loads((session.directory / "launch_observations.jsonl").read_text())
            self.assertEqual(record["status"], "baseline")
            self.assertEqual(
                record["selection_reason"], "forked_child_instrumentation_disabled")
            session.close()

    def test_partial_trace_preserves_valid_pairs_and_never_derives_sentinel_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = OnlineSession(load_online_config(_write_config(root)))
            sentinel = -(1 << 62)
            layout = []
            for slot, (op_id, boundary) in enumerate((
                ("op_a", "begin"), ("op_a", "end"),
                ("op_b", "begin"), ("op_b", "end"),
            )):
                layout.append({
                    "slot": slot, "op_id": op_id, "ordinal": slot // 2,
                    "source_kind": "copy", "lowered_instance_index": None,
                    "lowering_semantics": "single", "block": [0, 0, 0],
                    "boundary": boundary,
                })
            plan = SimpleNamespace(trace_extent=4, slot_layout=tuple(layout))
            pool = _PoolStub([{
                "values": [100, 125, sentinel, sentinel], "sentinel": sentinel,
                "generation": 1, "readback_policy": "async_after_window",
                "kernel_event_ms": 0.1,
                "launch_identity": {"kernel_id": "kernel_partial", "launch_index": 0},
            }])
            session.poll_trace_pool("kernel_partial", plan, pool)
            self.assertEqual([item["valid"] for item in session._trace_records], [True, True, False, False])
            self.assertEqual(session._trace_records[1]["interval_cycles"], 25)
            self.assertIsNone(session._trace_records[3]["interval_cycles"])
            self.assertIsNone(session._trace_records[3]["interval_semantics"])
            self.assertEqual(session._diagnostics[-1]["code"], "ONLINE_TRACE_PARTIAL")
            session.close()


class OnlinePlanTest(unittest.TestCase):
    def test_exact_plan_freezes_op_block_and_boundary_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            (artifact / "metadata").mkdir(parents=True)
            (artifact / "manifest.json").write_text(json.dumps({
                "input_ir_sha256": "a" * 64,
                "tilelang_revision": "revision-a",
            }), encoding="utf-8")
            _write_kernel_metadata(artifact)
            (artifact / "metadata" / "lineage.json").write_text(json.dumps({
                "status": "compiler_exact_lineage",
                "exact_runtime_instrumentation_allowed": True,
                "mappings": [
                    {"source_op_id": "op_a", "source_ordinal": 3, "source_kind": "gemm",
                     "eligible_for_exact_runtime_instrumentation": True,
                     "lowered_instance_count": 1, "final_device_instance_count": 1},
                    {"source_op_id": "op_b", "source_ordinal": 4, "source_kind": "reduce",
                     "eligible_for_exact_runtime_instrumentation": True,
                     "lowered_instance_count": 1, "final_device_instance_count": 1},
                ],
            }), encoding="utf-8")
            config = load_online_config(_write_config(
                root, compile_policy="eager_dual_variant", lineage_artifact=str(artifact),
                trace_op_ids=["op_b", "op_a"], block_selector={"indices": [0, 2]},
                budget={"max_trace_records": 8, "max_trace_bytes": 64}))
            request = SimpleNamespace(prim_func_sha256="a" * 64, normalized_out_idx=(2,))
            with mock.patch("musa_perf_collector.online.plan._tilelang_revision", return_value="revision-a"):
                plan = build_plan(config, request, target="musa -arch=mp_31", execution_backend="tvm_ffi")
            self.assertEqual(plan.source_ordinals, (3, 4))
            self.assertEqual(plan.source_op_ids, ("op_a", "op_b"))
            self.assertEqual(plan.trace_extent, 8)
            self.assertEqual([item["slot"] for item in plan.slot_layout], list(range(8)))
            self.assertEqual(plan.slot_layout[4]["op_id"], "op_b")
            self.assertEqual(plan.slot_layout[6]["block"], [2, 0, 0])

    def test_plan_rejects_ir_or_one_to_many_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            (artifact / "metadata").mkdir(parents=True)
            (artifact / "manifest.json").write_text(json.dumps({
                "input_ir_sha256": "a" * 64, "tilelang_revision": "revision-a"}), encoding="utf-8")
            _write_kernel_metadata(artifact)
            lineage_path = artifact / "metadata" / "lineage.json"
            lineage_path.write_text(json.dumps({
                "status": "compiler_exact_lineage", "exact_runtime_instrumentation_allowed": True,
                "mappings": [{"source_op_id": "op_a", "source_ordinal": 1, "source_kind": "copy",
                              "eligible_for_exact_runtime_instrumentation": True,
                              "lowered_instance_count": 2, "final_device_instance_count": 2}],
            }), encoding="utf-8")
            config = load_online_config(_write_config(
                root, compile_policy="eager_dual_variant", lineage_artifact=str(artifact),
                trace_op_ids=["op_a"]))
            with mock.patch("musa_perf_collector.online.plan._tilelang_revision", return_value="revision-a"):
                with self.assertRaisesRegex(RuntimeError, "IR hash"):
                    build_plan(config, SimpleNamespace(
                        prim_func_sha256="b" * 64, normalized_out_idx=(1,)),
                        target="musa", execution_backend="tvm_ffi")
                with self.assertRaisesRegex(RuntimeError, "one-to-one"):
                    build_plan(config, SimpleNamespace(
                        prim_func_sha256="a" * 64, normalized_out_idx=(1,)),
                        target="musa", execution_backend="tvm_ffi")

    def test_explicit_multi_instance_semantics_expand_slots_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            (artifact / "metadata").mkdir(parents=True)
            (artifact / "manifest.json").write_text(json.dumps({
                "input_ir_sha256": "a" * 64, "tilelang_revision": "revision-a"}), encoding="utf-8")
            _write_kernel_metadata(artifact)
            (artifact / "metadata" / "lineage.json").write_text(json.dumps({
                "status": "compiler_exact_lineage", "exact_runtime_instrumentation_allowed": True,
                "mappings": [{"source_op_id": "op_a", "source_ordinal": 7, "source_kind": "copy",
                              "eligible_for_exact_runtime_instrumentation": True,
                              "lowered_instance_count": 2, "final_device_instance_count": 2}],
            }), encoding="utf-8")
            request = SimpleNamespace(prim_func_sha256="a" * 64, normalized_out_idx=(1,))
            with mock.patch("musa_perf_collector.online.plan._tilelang_revision", return_value="revision-a"):
                envelope = build_plan(load_online_config(_write_config(
                    root, compile_policy="eager_dual_variant", lineage_artifact=str(artifact),
                    trace_op_ids=["op_a"], lowering_semantics="source_envelope")), request,
                    target="musa", execution_backend="tvm_ffi")
                instances = build_plan(load_online_config(_write_config(
                    root, compile_policy="eager_dual_variant", lineage_artifact=str(artifact),
                    trace_op_ids=["op_a"], lowering_semantics="lowered_instance")), request,
                    target="musa", execution_backend="tvm_ffi")
            self.assertEqual(envelope.trace_extent, 2)
            self.assertTrue(all(item["lowered_instance_index"] is None for item in envelope.slot_layout))
            self.assertEqual(instances.trace_extent, 4)
            self.assertEqual(
                [item["lowered_instance_index"] for item in instances.slot_layout], [0, 0, 1, 1])
