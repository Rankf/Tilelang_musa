"""Coverage and completeness contracts independent of device execution."""

from __future__ import annotations

import unittest

from musa_perf_collector.adapters.tilelang.tileop_discovery import classify_call
from musa_perf_collector.collectors.compiler.tileop_registry import FAMILY_BY_NAME, build_tileop_record
from musa_perf_collector.collectors.compiler.execution_counts import launch_context
from musa_perf_collector.core.validation import validate_offline_static


class _FakeOp:
    def __init__(self, name: str, builder: bool = True):
        self.name = name
        self._builder = builder

    def get_attr(self, name: str):
        return object() if name == "TLOpBuilder" and self._builder else None


class _FakeCall:
    def __init__(self, name: str, builder: bool = True, args=None):
        self.op = _FakeOp(name, builder)
        self.args = args or []
        self.annotations = {}

    def __str__(self):
        return self.op.name


class _FakeBuffer:
    def __init__(self, name: str, dtype: str = "float16", scope: str = "shared"):
        self.name, self.dtype, self._scope = name, dtype, scope

    def scope(self):
        return self._scope


class _FakeLoad:
    def __init__(self, buffer: _FakeBuffer):
        self.buffer, self.indices = buffer, [0, 0]


def _region(name: str, shape=(16, 16), dtype="float16", scope="shared"):
    return _FakeCall("tl.tileop.region", args=[_FakeLoad(_FakeBuffer(name, dtype, scope)), 1, *shape])


class TileOpContractTest(unittest.TestCase):
    def test_current_action_catalog_has_a_family(self):
        expected = {
            "copy", "async_copy", "tma_copy", "transpose", "im2col", "c2d_im2col",
            "gemm", "gemm_py", "gemm_sp", "wgmma_gemm", "tcgen05_gemm",
            "wgmma_gemm_sp", "tcgen05_gemm_sp", "fill", "reduce", "cumsum",
            "finalize_reducer", "atomicadd", "atomicmax", "atomicmin",
        }
        self.assertEqual(set(FAMILY_BY_NAME), expected)

    def test_future_unregistered_tileop_is_preserved_as_opaque(self):
        call = _FakeCall("tl.tileop.future_op", builder=False)
        self.assertEqual(classify_call(call), "unregistered_action")
        record = build_tileop_record(call, {
            "ordinal": 0, "statement_path": "tileop/0", "loop_context": [],
            "condition_context": [], "classification": "unregistered_action",
        })
        self.assertEqual(record["support_level"], "opaque")
        self.assertEqual(record["reads"], "unknown")
        self.assertIn("UNREGISTERED_TILEOP_CALL", record["diagnostic_codes"])

    def test_opaque_action_prevents_complete_without_dropping_coverage(self):
        opaque = {
            "op_id": "future", "raw_op_name": "tl.tileop.future_op", "support_level": "opaque",
            "canonical_family": "unknown", "reads": "unknown", "writes": "unknown",
            "execution": {"calls_per_block_expr": "1"}, "diagnostic_codes": [],
        }
        diagnostics = validate_offline_static(
            snapshots={stage: "ir" for stage in (
                "S00_INPUT_TILE_IR", "S10_PIPELINE_PLANNED", "S20_PIPELINE_INJECTED",
                "S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S50_HOST_DEVICE_SPLIT", "S60_FINAL_DEVICE_IR",
            )},
            kernel_source="kernel", coverage={"observed_action_calls": 1, "exported_action_records": 1, "unexported_action_calls": 0}, tile_ops=[opaque],
            dag={"nodes": [{"op_id": "future"}], "edges": [], "unknown_effect_barriers": ["future"]},
            domains={"per_block": {"execution_domain": "per_block", "symbolic_multiplicity_ops": [], "unknown_access_effect_ops": ["future"]}, "per_kernel_launch": {"execution_domain": "per_kernel_launch", "symbolic_multiplicity_ops": [], "unknown_access_effect_ops": ["future"]}},
        )
        self.assertEqual(diagnostics["status"], "partial")
        self.assertTrue(diagnostics["complete_gate"]["tileop_lossless_export"])

    def test_cycle_or_schema_error_prevents_complete(self):
        record = {
            "op_id": "one", "raw_op_name": "tl.tileop.copy", "canonical_family": "copy",
            "support_level": "structured", "reads": [], "writes": [],
            "execution": {"calls_per_block_expr": "1"}, "diagnostic_codes": [],
        }
        diagnostics = validate_offline_static(
            snapshots={stage: "ir" for stage in (
                "S00_INPUT_TILE_IR", "S10_PIPELINE_PLANNED", "S20_PIPELINE_INJECTED",
                "S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S50_HOST_DEVICE_SPLIT", "S60_FINAL_DEVICE_IR",
            )},
            kernel_source="kernel", coverage={"observed_action_calls": 1, "exported_action_records": 1, "unexported_action_calls": 0}, tile_ops=[record],
            dag={"nodes": [{"op_id": "one"}], "edges": [{"from": "one", "to": "one"}], "unknown_effect_barriers": []},
            domains={"per_block": {"execution_domain": "per_block", "symbolic_multiplicity_ops": [], "unknown_access_effect_ops": []}, "per_kernel_launch": {"execution_domain": "per_kernel_launch", "symbolic_multiplicity_ops": [], "unknown_access_effect_ops": []}},
        )
        self.assertEqual(diagnostics["status"], "failed")
        self.assertIn("DAG_CYCLE_IN_SINGLE_ITERATION", [item["code"] for item in diagnostics["diagnostics"]])

    def test_every_registered_family_has_a_structural_adapter_contract(self):
        a, b, c, e = _region("A"), _region("B"), _region("C", dtype="float32", scope="local.fragment"), _region("E")
        calls = {
            **{name: _FakeCall(f"tl.tileop.{name}", args=[a, b]) for name in ("copy", "async_copy", "tma_copy", "transpose", "im2col", "c2d_im2col")},
            **{name: _FakeCall(f"tl.tileop.{name}", args=[a, b, c, False, False, 16, 16, 16]) for name in ("gemm", "gemm_py", "wgmma_gemm", "tcgen05_gemm")},
            **{name: _FakeCall(f"tl.tileop.{name}", args=[a, e, b, c, False, False, False, 16, 16, 32]) for name in ("gemm_sp", "wgmma_gemm_sp", "tcgen05_gemm_sp")},
            "fill": _FakeCall("tl.tileop.fill", args=[c, 0]),
            "reduce": _FakeCall("tl.tileop.reduce", args=[a, c, "sum", 1, True]),
            "cumsum": _FakeCall("tl.tileop.cumsum", args=[a, c, 1, False]),
            "finalize_reducer": _FakeCall("tl.tileop.finalize_reducer", args=[c]),
            **{name: _FakeCall(f"tl.tileop.{name}", args=[a, c]) for name in ("atomicadd", "atomicmax", "atomicmin")},
        }
        self.assertEqual(set(calls), set(FAMILY_BY_NAME))
        for ordinal, (name, call) in enumerate(calls.items()):
            with self.subTest(tileop=name):
                record = build_tileop_record(call, {
                    "ordinal": ordinal, "statement_path": f"tileop/{ordinal}", "loop_context": [],
                    "condition_context": [], "classification": "action",
                })
                self.assertIn(record["support_level"], {"structured", "structural"})
                self.assertIsInstance(record["reads"], list)
                self.assertIsInstance(record["writes"], list)

    def test_symbolic_launch_extent_is_preserved(self):
        class PrimFunc:
            def script(self):
                return 'bx = T.launch_thread("blockIdx.x", blocks)\ntx = T.launch_thread("threadIdx.x", 128)'
        launch = launch_context(PrimFunc())
        self.assertEqual(launch["block_count"], "blocks")
        self.assertEqual(launch["thread_count"], 128)
