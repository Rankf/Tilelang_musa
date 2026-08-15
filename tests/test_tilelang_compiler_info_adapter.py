"""Formal TileLang compiler-info to TileSight adapter tests."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from deepstack.mosaic.arch import make_custom_stacked_gpu, make_s5000_config

from tilelang_compiler_info_adapter import (
    CompilerInfoAdapterError,
    ModelConsistencyError,
    build_tilesight_result,
    invoke_tilesight_model,
    prepare_tilesight_request,
    run_tilesight_request,
)


def make_compiler_info():
    """Return the smallest exact-lineage FP16 GEMM compiler artifact."""
    operations = [
        {
            "op_id": "copy_a",
            "canonical_family": "copy",
            "reads": [{"buffer": "A", "scope": "global"}],
            "writes": [{"buffer": "a_shared", "scope": "shared.dyn"}],
            "loop_ids": ["loop/0"],
        },
        {
            "op_id": "copy_b",
            "canonical_family": "copy",
            "reads": [{"buffer": "B", "scope": "global"}],
            "writes": [{"buffer": "b_shared", "scope": "shared.dyn"}],
            "loop_ids": ["loop/0"],
        },
        {
            "op_id": "gemm",
            "canonical_family": "gemm",
            "reads": [
                {"buffer": "a_shared", "scope": "shared.dyn"},
                {"buffer": "b_shared", "scope": "shared.dyn"},
                {"buffer": "c_local", "scope": "local.fragment"},
            ],
            "writes": [{"buffer": "c_local", "scope": "local.fragment"}],
            "loop_ids": ["loop/0"],
            "gemm": {"m": 128, "n": 128, "k": 64},
        },
        {
            "op_id": "copy_c",
            "canonical_family": "copy",
            "reads": [{"buffer": "c_local", "scope": "local.fragment"}],
            "writes": [{"buffer": "C", "scope": "global"}],
            "loop_ids": [],
        },
    ]
    return {
        "schema_version": "tilelang/compiler-info-v1",
        "compilation_id": "compilation_test",
        "identity": {"target": "musa -arch=mp_31"},
        "lineage": {"status": "compiler_exact_lineage"},
        "source": {
            "kernels": [
                {
                    "kernel_id": "kernel_test",
                    "buffers": [
                        {"name": "A", "shape": [4096, 4096], "dtype": "float16"},
                        {"name": "B", "shape": [4096, 4096], "dtype": "float16"},
                        {"name": "C", "shape": [4096, 4096], "dtype": "float16"},
                        {"name": "c_local", "shape": [128, 128], "dtype": "float32"},
                    ],
                    "tile_ops": operations,
                    "dependencies": {
                        "edges": [
                            {"from": "copy_a", "to": "gemm"},
                            {"from": "copy_b", "to": "gemm"},
                            {"from": "gemm", "to": "copy_c"},
                        ]
                    },
                    "pipeline_declarations": [
                        {
                            "pipeline_id": "pipeline_source",
                            "loop_id": "loop/0",
                            "loop_ordinal": 0,
                            "num_stages": 2,
                        }
                    ],
                    "static_workload": {
                        "per_kernel_launch": {
                            "flops": 2 * 4096 * 4096 * 4096,
                        }
                    },
                }
            ],
        },
        "pipeline_planning": {
            "kernels": [
                {
                    "pipelines": [
                        {
                            "pipeline_id": "pipeline_planned",
                            "source_loop_ordinal": 0,
                        }
                    ]
                }
            ]
        },
        "tileop_lowering": {
            "kernels": [
                {
                    "all_call_kinds": {"tl.tl_gemm": 1},
                }
            ]
        },
        "final_implementation": {
            "kernels": [
                {
                    "launch": {
                        "grid_size": 1024,
                        "threads_per_block": 256,
                        "warp_size": 32,
                        "warps_per_block": 8,
                        "dynamic_shared_memory_bytes": 65536,
                    }
                }
            ]
        },
    }


def make_elementwise_compiler_info(kind="add", n=16384):
    """Return a minimal exact-lineage FP32 Add or ReLU compiler artifact."""
    input_count = 2 if kind == "add" else 1
    expression_kind = "add" if kind == "add" else "max_zero"
    elements_per_thread = 8 if kind == "add" else 4
    elements_per_block = 256 * elements_per_thread
    names = ["A", "B"] if kind == "add" else ["A"]
    output = "C" if kind == "add" else "B"
    buffers = [
        {
            "name": name,
            "shape": [n],
            "dtype": "float32",
            "scope": "global",
        }
        for name in sorted(set([*names, output]))
    ]
    operation = {
        "op_id": f"elementwise_{kind}",
        "source_ordinal": 0,
        "expression": {
            "kind": expression_kind,
            "input_count": input_count,
            "operations_per_element": 1,
        },
        "reads": [
            {"buffer": name, "scope": "global", "dtype": "float32"} for name in names
        ],
        "writes": [{"buffer": output, "scope": "global", "dtype": "float32"}],
        "shape": [n],
        "dtype": "float32",
        "condition_context": [],
        "elements_per_block": elements_per_block,
        "elements_per_thread": elements_per_thread,
        "access_pattern": "same_index_contiguous",
    }
    return {
        "schema_version": "tilelang/compiler-info-v1",
        "compilation_id": f"compilation_{kind}",
        "identity": {"target": "musa -arch=mp_31"},
        "lineage": {"status": "unavailable_or_incomplete"},
        "elementwise_lineage": {
            "status": "compiler_exact_lineage",
            "mappings": [
                {
                    "source_op_id": operation["op_id"],
                    "source_ordinal": 0,
                    "lowered_instance_count": 1,
                }
            ],
        },
        "source": {
            "kernels": [
                {
                    "kernel_id": f"kernel_{kind}",
                    "buffers": buffers,
                    "tile_ops": [],
                    "elementwise_ops": [operation],
                }
            ]
        },
        "final_implementation": {
            "kernels": [
                {
                    "launch": {
                        "grid_size": n // elements_per_block,
                        "threads_per_block": 256,
                        "warp_size": 32,
                        "warps_per_block": 8,
                    }
                }
            ]
        },
    }


class CompilerInfoAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = make_custom_stacked_gpu(make_s5000_config())

    def test_request_preserves_evidence_classes_and_model_semantics(self):
        request = prepare_tilesight_request(make_compiler_info())
        self.assertEqual(request["schema_version"], "tilesight/model-request-v1")
        self.assertEqual(request["model"]["arguments"]["wp_shape"], [32, 64, 64])
        self.assertEqual(
            request["model"]["arguments"]["mem_levels"]["out1"],
            [1, 0, 1, 2],
        )
        self.assertIn("wp_shape", request["evidence"]["model_assumption"])
        self.assertEqual(request["model_execution"]["numpy_random_seed"], 20260814)
        self.assertIn(
            "dynamic_shared_memory_bytes",
            request["evidence"]["compiler_observed"],
        )
        self.assertEqual(request["compiler_facts"]["accumulator_dtype"], "float32")
        self.assertIn(
            "mixed_precision_exact_register_footprint", request["unavailable"]
        )
        self.assertIn("runtime_measurements", request["unavailable"])
        json.dumps(request, sort_keys=True)

    def test_native_model_is_non_mutating_and_matches_compiler_facts(self):
        request = prepare_tilesight_request(make_compiler_info())
        before = deepcopy(request)
        np.random.seed(314159)
        random_state_before = np.random.get_state()
        native = invoke_tilesight_model(request, self.arch)
        random_state_after = np.random.get_state()
        self.assertEqual(random_state_before[0], random_state_after[0])
        np.testing.assert_array_equal(random_state_before[1], random_state_after[1])
        self.assertEqual(random_state_before[2:], random_state_after[2:])
        result = build_tilesight_result(request, native, self.arch)
        repeated = run_tilesight_request(request, self.arch)
        self.assertEqual(request, before)
        self.assertEqual(result["prediction"], repeated["prediction"])
        self.assertEqual(result["schema_version"], "tilesight/model-result-v1")
        self.assertEqual(result["compiler_validation"]["status"], "all_consistent")
        self.assertEqual(result["prediction"]["modeled_smem_footprint_bytes"], 65536)
        json.dumps(result, sort_keys=True)

    def test_unknown_or_inconsistent_inputs_fail_closed(self):
        incomplete = make_compiler_info()
        incomplete["lineage"]["status"] = "partial"
        with self.assertRaisesRegex(
            CompilerInfoAdapterError, "incomplete_compiler_lineage"
        ):
            prepare_tilesight_request(incomplete)

        wrong_dtype = make_compiler_info()
        wrong_dtype["source"]["kernels"][0]["buffers"][2]["dtype"] = "float32"
        with self.assertRaisesRegex(CompilerInfoAdapterError, "unsupported_dtype"):
            prepare_tilesight_request(wrong_dtype)

        wrong_accumulator = make_compiler_info()
        wrong_accumulator["source"]["kernels"][0]["buffers"][3]["dtype"] = "float16"
        with self.assertRaisesRegex(
            CompilerInfoAdapterError, "unsupported_accumulator_dtype"
        ):
            prepare_tilesight_request(wrong_accumulator)

        wrong_target = make_compiler_info()
        wrong_target["identity"]["target"] = "cuda -arch=sm_90"
        with self.assertRaisesRegex(CompilerInfoAdapterError, "unsupported_target"):
            prepare_tilesight_request(wrong_target)

        request = prepare_tilesight_request(make_compiler_info())
        native = invoke_tilesight_model(request, self.arch)
        request["compiler_facts"]["dynamic_shared_memory_bytes"] = 32768
        with self.assertRaises(ModelConsistencyError) as caught:
            build_tilesight_result(request, native, self.arch)
        self.assertEqual(
            caught.exception.diagnostics[0]["field"],
            "dynamic_shared_memory_bytes",
        )

    def test_add_and_relu_dispatch_to_native_elementwise_models(self):
        cases = [
            ("add", "elementwise_n_n_pipeline_wave", 2048, 8),
            ("relu", "elementwise_n_0_pipeline_wave", 1024, 4),
        ]
        for kind, model_name, elements_per_block, elements_per_thread in cases:
            request = prepare_tilesight_request(
                make_elementwise_compiler_info(kind), cache_condition="cold"
            )
            self.assertEqual(request["model"]["name"], model_name)
            self.assertEqual(request["model"]["arguments"]["op_shape"], [16384])
            self.assertEqual(
                request["model"]["arguments"]["tb_shape"], [elements_per_block]
            )
            self.assertEqual(request["model"]["arguments"]["dim_threads"], [256])
            self.assertEqual(
                request["model"]["arguments"]["mem_levels"]["out1"],
                [1, 0, 1, 4],
            )
            self.assertEqual(
                request["compiler_facts"]["elements_per_thread"],
                elements_per_thread,
            )
            native = invoke_tilesight_model(request, self.arch)
            result = build_tilesight_result(request, native, self.arch)
            self.assertEqual(result["compiler_validation"]["status"], "all_consistent")
            self.assertEqual(result["model_scenario"]["cache_condition"], "cold")
            self.assertEqual(result["prediction"]["l2_hit_rate_pct"], 0.0)
            cache = result["prediction"]["cache_detail"]
            self.assertEqual(cache["kernel_scope_ddr_write_bytes"], 0)
            self.assertEqual(cache["deferred_writeback_bytes"], 16384 * 4)
            json.dumps(result, sort_keys=True)

    def test_elementwise_cache_scenarios_are_native_model_predictions(self):
        for kind in ("add", "relu"):
            warm_request = prepare_tilesight_request(
                make_elementwise_compiler_info(kind), cache_condition="warm"
            )
            self.assertNotIn("runtime_measurements", warm_request)
            warm = run_tilesight_request(warm_request, self.arch)
            self.assertEqual(warm["prediction"]["l2_hit_rate_pct"], 99.75)
            self.assertEqual(
                warm["prediction"]["cache_detail"]["l2_prediction_method"],
                "calibrated_warm_working_set_fits_l2",
            )

            large_request = prepare_tilesight_request(
                make_elementwise_compiler_info(kind, n=1 << 24),
                cache_condition="warm",
            )
            large = run_tilesight_request(large_request, self.arch)
            self.assertEqual(large["prediction"]["l2_hit_rate_pct"], 0.0)
            large_cache = large["prediction"]["cache_detail"]
            self.assertEqual(
                large_cache["l2_prediction_method"],
                "warm_working_set_exceeds_l2_capacity",
            )
            self.assertGreater(large_cache["kernel_scope_ddr_write_bytes"], 0)
            self.assertEqual(large_cache["deferred_writeback_bytes"], 0)

    def test_native_elementwise_default_remains_explicit_cold(self):
        from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import (
            calculate_N_0_elementwise_pipeline_wave,
            calculate_N_N_elementwise_pipeline_wave,
        )

        functions = {
            "add": calculate_N_N_elementwise_pipeline_wave,
            "relu": calculate_N_0_elementwise_pipeline_wave,
        }
        for kind, function in functions.items():
            request = prepare_tilesight_request(
                make_elementwise_compiler_info(kind), cache_condition="cold"
            )
            arguments = deepcopy(request["model"]["arguments"])
            legacy_style = function(arch=self.arch, **arguments)
            explicit = function(
                arch=self.arch, cache_condition="cold", **arguments
            )
            self.assertEqual(legacy_style.l2_hit_rate, explicit.l2_hit_rate)
            self.assertEqual(legacy_style.total_latency, explicit.total_latency)
            self.assertEqual(
                legacy_style.cache_detail["cache_condition"], "cold"
            )

    def test_elementwise_adapter_fails_closed_and_does_not_precompute_predictions(self):
        with self.assertRaisesRegex(
            CompilerInfoAdapterError, "missing_cache_condition"
        ):
            prepare_tilesight_request(make_elementwise_compiler_info("add"))
        with self.assertRaisesRegex(
            CompilerInfoAdapterError, "unsupported_cache_condition"
        ):
            prepare_tilesight_request(
                make_elementwise_compiler_info("add"), cache_condition="unknown"
            )
        broadcast = make_elementwise_compiler_info("add")
        broadcast["source"]["kernels"][0]["elementwise_ops"][0][
            "access_pattern"
        ] = "unsupported"
        with self.assertRaisesRegex(
            CompilerInfoAdapterError, "unsupported_access_pattern"
        ):
            prepare_tilesight_request(broadcast, cache_condition="cold")

        request = prepare_tilesight_request(
            make_elementwise_compiler_info("add"), cache_condition="cold"
        )

        def all_keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from all_keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from all_keys(item)

        forbidden = {
            "latency_us",
            "ddr_util_pct",
            "l2_util_pct",
            "smem_util_pct",
            "compute_util_pct",
            "l2_hit_rate_pct",
            "tiles_per_sm",
            "waves",
            "physical_traffic_bytes",
            "occupancy",
        }
        self.assertTrue(forbidden.isdisjoint(set(all_keys(request))))

    def test_analyze_op_cli_uses_formal_request_and_keeps_old_path_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiler-info.json"
            path.write_text(json.dumps(make_compiler_info()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "analyze_op.py", "--compiler-info", str(path)],
                check=True,
                text=True,
                capture_output=True,
            )
        report = json.loads(completed.stdout)
        self.assertEqual(report["input_source"], "compiler_info")
        self.assertNotIn("metadata_index", report)
        self.assertEqual(
            report["model_request"]["schema_version"],
            "tilesight/model-request-v1",
        )
        self.assertEqual(
            report["model_result"]["compiler_validation"]["status"],
            "all_consistent",
        )

    def test_analyze_op_cli_dispatches_real_elementwise_tile_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "add-compiler-info.json"
            path.write_text(
                json.dumps(make_elementwise_compiler_info("add")),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "analyze_op.py",
                    "--compiler-info",
                    str(path),
                    "--cache-condition",
                    "cold",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        report = json.loads(completed.stdout)
        self.assertEqual(report["op"], "add")
        self.assertEqual(report["input_source"], "compiler_info")
        self.assertEqual(report["elements_per_block"], 2048)
        self.assertEqual(report["elements_per_thread"], 8)
        self.assertEqual(
            report["model_request"]["model"]["name"],
            "elementwise_n_n_pipeline_wave",
        )
        self.assertEqual(
            report["model_result"]["compiler_validation"]["status"],
            "all_consistent",
        )

    def test_real_tilelang_4096_compiler_artifact_when_available(self):
        try:
            from tilelang.tools.compiler_info import collect_compiler_information
            from workloads.gemm_4096_tilelang import make_prim_func
        except (ImportError, OSError) as error:
            self.skipTest(f"TileLang-MUSA compiler environment unavailable: {error}")
        information = collect_compiler_information(
            make_prim_func(), target={"kind": "musa", "arch": "mp_31"}
        )
        request = prepare_tilesight_request(information)
        result = run_tilesight_request(request, self.arch)
        self.assertEqual(request["model"]["arguments"]["op_shape"], [4096] * 3)
        self.assertEqual(result["prediction"]["modeled_smem_footprint_bytes"], 65536)

    def test_real_tilelang_add_relu_compiler_artifacts_when_available(self):
        try:
            from analyze_op import analyze_from_compiler_info
            from tilelang.tools.compiler_info import collect_compiler_information
            from workloads.elementwise_tilelang import (
                make_add_prim_func,
                make_relu_prim_func,
            )
        except (ImportError, OSError) as error:
            self.skipTest(f"TileLang-MUSA compiler environment unavailable: {error}")
        for factory, expected_model in [
            (make_add_prim_func, "elementwise_n_n_pipeline_wave"),
            (make_relu_prim_func, "elementwise_n_0_pipeline_wave"),
        ]:
            information = collect_compiler_information(
                factory(), target={"kind": "musa", "arch": "mp_31"}
            )
            request = prepare_tilesight_request(information, cache_condition="cold")
            result = run_tilesight_request(request, self.arch)
            self.assertEqual(request["model"]["name"], expected_model)
            self.assertEqual(result["compiler_validation"]["status"], "all_consistent")
            report = analyze_from_compiler_info(
                information, arch=self.arch, cache_condition="cold"
            )
            self.assertEqual(report["input_source"], "compiler_info")
            self.assertEqual(
                report["elements_per_block"],
                request["model"]["arguments"]["tb_shape"][0],
            )
            self.assertEqual(
                report["model_result"]["compiler_validation"]["status"],
                "all_consistent",
            )


if __name__ == "__main__":
    unittest.main()
