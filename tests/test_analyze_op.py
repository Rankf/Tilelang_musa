import unittest
import numpy as np
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from analyze_op import (
    _derive_warp_shape,
    analyze_add,
    analyze_gemm_from_metadata,
    analyze_relu,
    build_hierarchical_report,
    build_session_report,
    derive_gemm_model_input,
    render_report_summary,
    render_session_summary,
)
from deepstack.mosaic.arch import make_s5000_config, make_custom_stacked_gpu
from tilesight.util.L2_hit_rate_flow_sim_reuse_distance import (
    L2_hit_rate_flow_sim_reuse_distance,
)


class AnalyzeOpTest(unittest.TestCase):
    metadata = "docs/tilesight_5_1_gemm_4096_output.json"

    @classmethod
    def setUpClass(cls):
        cls.arch = make_custom_stacked_gpu(make_s5000_config())

    def test_warp_count_is_not_squared(self):
        self.assertEqual(_derive_warp_shape(128, 128, 64, 4), (64, 64, 64))

    def test_gemm_exposes_5_2_cost_and_critical_path(self):
        report = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        model = report["cost_model"]

        self.assertEqual(model["status"], "available")
        self.assertEqual(model["value_kind"], "model_estimate")
        self.assertGreater(report["pipeline"]["memory_per_iter_us"], 0)
        self.assertGreater(report["pipeline"]["compute_per_iter_us"], 0)
        self.assertIn(model["critical_path"]["bottleneck"],
                      {"compute", "ddr", "l2", "shared_memory"})

        for level in ("ddr", "l2", "shared_memory"):
            cost = model["memory_level_cost"][level]
            self.assertGreater(cost["traffic_bytes"], 0)
            self.assertGreater(cost["peak_roofline_cost_us"], 0)
            self.assertGreater(cost["calibrated_service_cost_us"], 0)

    def test_metadata_api_does_not_require_cli_or_explicit_arch(self):
        report = analyze_gemm_from_metadata(self.metadata)
        self.assertEqual(report["op"], "gemm")
        self.assertEqual(
            report["memory_path_kind"],
            "s5000_fp16_global_shared_from_tile_metadata",
        )
        self.assertEqual(report["memory_levels"]["out1"], [1, 0, 1, 2])
        self.assertEqual(report["cost_model"]["status"], "available")

    def test_gemm_tma_bypasses_ordinary_l1(self):
        report = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        hierarchy = report["memory_hierarchy_prediction"]
        self.assertEqual(hierarchy["l1_cache"]["eligible_read_bytes"], 0)
        self.assertGreater(hierarchy["l1_cache"]["bypass_read_bytes"], 0)
        self.assertEqual(
            hierarchy["l1_cache"]["predicted_hit_rate_pct"],
            "not_applicable",
        )
        self.assertEqual(
            hierarchy["shared_memory"]["hit_rate"], "not_applicable"
        )

    def test_gemm_exposes_existing_per_operand_cache_data(self):
        report = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        detail = report["tile_cache_predictions"]
        self.assertEqual(detail["status"], "available")
        self.assertEqual(detail["calculation_change"],
                         "none; formats existing cache-model accumulators")
        operands = {item["operand"]: item for item in detail["operands"]}
        self.assertEqual(set(operands), {"A", "B"})
        self.assertEqual(operands["A"]["tile_request_bytes_per_k_iter"], 16384)
        self.assertEqual(operands["B"]["tile_request_bytes_per_k_iter"], 16384)
        self.assertEqual(operands["A"]["ordinary_l1"]["status"], "bypass")

        aggregate = detail["aggregate"]
        self.assertEqual(
            sum(item["l2"]["request_bytes"] for item in operands.values()),
            aggregate["l2_request_bytes"],
        )
        self.assertEqual(
            sum(item["ddr"]["predicted_read_bytes"]
                for item in operands.values()),
            aggregate["predicted_ddr_read_bytes"],
        )
        self.assertAlmostEqual(
            aggregate["weighted_operand_l2_hit_rate_pct"],
            report["l2_hit_rate_pct"], places=1,
        )

    def test_l2_detail_mode_preserves_legacy_aggregate_calculation(self):
        args = (
            1024, 1536, 2048, 128, 128, 64,
            self.arch.l2_capacity, self.arch.sm_count,
            {"in1": [1, 1, 1, 2], "in2": [1, 1, 1, 2],
             "out1": [1, 0, 1, 2]},
            1,
        )
        np.random.seed(20260812)
        legacy = L2_hit_rate_flow_sim_reuse_distance(*args)
        np.random.seed(20260812)
        detail = L2_hit_rate_flow_sim_reuse_distance(
            *args, return_detail=True
        )
        self.assertEqual(legacy, detail["l2_hit_rate"])

    def test_elementwise_ordinary_load_and_explicit_cache_condition(self):
        cold_relu = analyze_relu(16384, 256, self.arch, cache_condition="cold")
        relu = analyze_relu(1 << 20, 256, self.arch, cache_condition="warm")
        add = analyze_add(1 << 24, 256, self.arch, cache_condition="warm")
        cold_h = cold_relu["memory_hierarchy_prediction"]
        relu_h = relu["memory_hierarchy_prediction"]
        add_h = add["memory_hierarchy_prediction"]

        self.assertEqual(cold_h["l2"]["predicted_hit_rate_pct"], 0.0)
        self.assertEqual(cold_h["ddr"]["logical_output_write_bytes"], 65536)
        self.assertEqual(cold_h["ddr"]["kernel_scope_write_bytes"], 0)
        self.assertEqual(cold_h["ddr"]["deferred_writeback_bytes"], 65536)
        self.assertEqual(relu_h["l1_cache"]["bypass_read_bytes"], 0)
        self.assertEqual(relu_h["l1_cache"]["predicted_hit_rate_pct"], 0.0)
        self.assertEqual(relu_h["l2"]["predicted_hit_rate_pct"], 99.75)
        self.assertEqual(
            relu_h["l2"]["calibrated_prediction_range_pct"], [99.0, 100.0]
        )
        self.assertGreater(relu_h["ddr"]["read_bytes"], 0)
        self.assertEqual(relu_h["ddr"]["write_bytes"], 0)
        self.assertEqual(relu_h["ddr"]["deferred_writeback_bytes"], 1 << 22)
        self.assertEqual(relu["l2_hit_rate_pct"], 99.75)
        self.assertGreater(relu["l2_util_pct"], 0)
        self.assertEqual(add_h["l2"]["predicted_hit_rate_pct"], 0.0)
        self.assertGreater(add_h["ddr"]["read_bytes"], 0)
        self.assertEqual(add_h["ddr"]["write_bytes"], 1 << 26)
        self.assertEqual(add_h["ddr"]["deferred_writeback_bytes"], 0)
        self.assertEqual(add_h["critical_memory_path"]["bottleneck"], "ddr")
        self.assertEqual(
            add["latency_with_memory_hierarchy_us"], add["latency_us"]
        )

    def test_5_7_transfer_anomaly_output_is_model_only_and_structured(self):
        report = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        analysis = report["transfer_anomaly_analysis"]
        self.assertEqual(analysis["status"], "available")
        self.assertEqual(
            analysis["analysis_scope"], "tile_pipeline_and_hardware_model"
        )
        self.assertNotIn("measurement_comparison", analysis)
        rule_ids = {item["rule_id"] for item in analysis["findings"]}
        self.assertIn("cross_level_read_conservation", rule_ids)
        self.assertIn("transfer_service_pressure", rule_ids)
        self.assertIn("logical_transfer_amplification", rule_ids)
        self.assertNotIn("tme_counter_comparability", rule_ids)

    def test_5_5_hierarchical_report_uses_existing_model_fields(self):
        report = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        report.update({"arch": "S5000", "sm_count": self.arch.sm_count})
        view = build_hierarchical_report(report)

        self.assertEqual(view["schema"], "tilesight/hierarchical-report-v1")
        self.assertEqual(view["calculation_change"],
                         "none; report-layer transformation only")
        self.assertEqual(view["report_scope"], "kernel")
        self.assertEqual(
            {item["resource"] for item in
             view["hotspots"]["resource_by_utilization"]},
            {"compute", "shared_memory", "l2", "global_memory_ddr"},
        )
        self.assertEqual(
            {item["operand"] for item in
             view["hotspots"]["tile_transfer_by_ddr_bytes"]},
            {"A", "B"},
        )
        self.assertEqual(
            [item["event"] for item in view["timeline"]["events"]],
            ["pipeline_prologue", "pipeline_steady", "pipeline_epilogue"],
        )
        self.assertEqual(view["scale_units"]["kernel"]["grid_tile_count"],
                         32 * 32)
        self.assertIn("不是运行期实测trace",
                      render_report_summary(report, view))

    def test_cli_report_type_selection(self):
        base = [
            sys.executable, "analyze_op.py",
            "--tile-metadata", self.metadata,
        ]
        hierarchical = subprocess.run(
            base + ["--report-type", "hierarchical"],
            check=True, text=True, capture_output=True,
        )
        self.assertEqual(
            json.loads(hierarchical.stdout)["schema"],
            "tilesight/hierarchical-report-v1",
        )
        summary = subprocess.run(
            base + ["--report-type", "summary"],
            check=True, text=True, capture_output=True,
        )
        self.assertIn("资源热点", summary.stdout)
        self.assertIn("Pipeline模型时间线", summary.stdout)

    def test_5_5_session_report_ranks_existing_kernel_reports(self):
        small = analyze_gemm_from_metadata(self.metadata, arch=self.arch)
        large = deepcopy(small)
        small["latency_with_memory_hierarchy_us"] /= 2
        for report in (small, large):
            report.update({"arch": "S5000", "sm_count": self.arch.sm_count})
        session = build_session_report([small, large], "test_session")
        self.assertEqual(session["schema"], "tilesight/session-report-v1")
        self.assertEqual(session["summary"]["kernel_count"], 2)
        self.assertEqual(session["kernel_hotspots_by_latency"][0]["shape"],
                         [4096, 4096, 4096])
        self.assertEqual(len(session["timeline"]), 2)
        self.assertIn("不是运行期实测Session trace",
                      render_session_summary(session))

    def test_5_1_metadata_drives_gemm_model_and_backfills_ids(self):
        metadata = self.metadata
        derived = derive_gemm_model_input(metadata)
        self.assertEqual(derived["model_input"], {
            "M": 4096, "N": 4096, "K": 4096,
            "bm": 128, "bn": 128, "bk": 64,
            "nw": 4, "ns": 2,
        })
        report = analyze_gemm_from_metadata(metadata, arch=self.arch)
        index = report["metadata_index"]
        self.assertEqual(report["input_source"], "tile_metadata")
        self.assertEqual(report["kernel_id"], "kernel_a0316b506d1f69cc")
        self.assertEqual(report["pipeline_id"],
                         "pipeline_55cb7cd195173372")
        self.assertEqual(
            report["cost_model"]["critical_path"]["tile_ops"][0]["op_id"],
            index["operation_ids"]["gemm"],
        )
        operands = {
            item["operand"]: item for item in
            report["tile_cache_predictions"]["operands"]
        }
        self.assertEqual(operands["A"]["op_id"],
                         index["operation_ids"]["copy_a"])
        self.assertEqual(operands["B"]["op_id"],
                         index["operation_ids"]["copy_b"])

    def test_cli_accepts_metadata_without_manual_model_parameters(self):
        completed = subprocess.run([
            sys.executable, "analyze_op.py",
            "--tile-metadata", "docs/tilesight_5_1_gemm_4096_output.json",
            "--report-type", "hierarchical",
        ], check=True, text=True, capture_output=True)
        view = json.loads(completed.stdout)
        self.assertEqual(view["kernel_summary"]["shape"], [4096, 4096, 4096])
        self.assertEqual(
            view["kernel_summary"]["critical_path"]["tile_ops"][0]["op_id"],
            "op_1543d5dc0c6f297c",
        )
        summary = subprocess.run([
            sys.executable, "analyze_op.py",
            "--tile-metadata", "docs/tilesight_5_1_gemm_4096_output.json",
            "--report-type", "summary",
        ], check=True, text=True, capture_output=True)
        self.assertIn("op_1543d5dc0c6f297c", summary.stdout)
        self.assertIn("op_a4e41ee4df136690", summary.stdout)

    def test_cli_rejects_removed_manual_gemm_interface(self):
        completed = subprocess.run([
            sys.executable, "analyze_op.py", "gemm",
            "--M", "1024", "--N", "1024", "--K", "1024",
        ], text=True, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid choice", completed.stderr)

    def test_metadata_export_cli_reproduces_4096_gemm_contract(self):
        exporter_python = ".venv-tilelang-musa/bin/python"
        if not Path(exporter_python).exists():
            self.skipTest("TileLang-MUSA virtual environment is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.json"
            subprocess.run([
                exporter_python, "export_tile_metadata.py",
                "--factory", "workloads.gemm_4096_tilelang:make_prim_func",
                "-o", str(output),
            ], check=True, text=True, capture_output=True)
            metadata = json.loads(output.read_text())
            derived = derive_gemm_model_input(metadata)
            self.assertEqual(derived["model_input"]["M"], 4096)
            self.assertEqual(derived["model_input"]["bm"], 128)
            self.assertEqual(derived["model_input"]["nw"], 4)
            self.assertEqual(derived["model_input"]["ns"], 2)


if __name__ == "__main__":
    unittest.main()
