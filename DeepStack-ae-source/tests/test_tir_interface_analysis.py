"""Tests for TileSight's self-contained TIR analysis entry points."""

import pytest


tilelang = pytest.importorskip("tilelang")
import tilelang.language as T

from tilesight.tir_interface import (
    analyze_prim_func,
    analyze_tir_after_vectorize,
    estimate_flops_from_mod,
)


class TestArch:
    core = "test-gpu"
    fp16_tensor_flops = 100e12
    fp32_cuda_core_flops = 20e12
    compute_max_util = 0.8
    ddr_bandwidth = 1e12
    ddr_max_util = 0.8
    l2_bandwidth = 4e12
    l2_max_util = 0.5
    smem_bandwidth = 8e12
    l1_max_util = 0.9


def make_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((256, 256), "float16"),
        B: T.Tensor((256, 256), "float16"),
        C: T.Tensor((256, 256), "float16"),
    ):
        with T.Kernel(4, 4, threads=128) as (bx, by):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.clear(c_local)
            for k in T.Pipelined(8, num_stages=2):
                T.copy(A[by * 64, k * 32], a_shared)
                T.copy(B[k * 32, bx * 64], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, C[by * 64, bx * 64])

    return main


def make_lowered_elementwise():
    @T.prim_func
    def add(
        A: T.Tensor((256,), "float32"),
        B: T.Tensor((256,), "float32"),
        C: T.Tensor((256,), "float32"),
    ):
        for i in T.serial(256):
            C[i] = A[i] + B[i]

    return add


def test_pre_lowering_gemm_extracts_metadata_pipeline_and_dag():
    function = make_gemm()
    first = analyze_prim_func(function, TestArch())
    second = analyze_prim_func(function, TestArch())
    kernel = first["kernels"][0]

    assert first["schema_version"] == "tilesight/tile-metadata-v1"
    assert kernel["kernel_id"] == second["kernels"][0]["kernel_id"]
    assert [op["op_id"] for op in kernel["operations"]] == [
        op["op_id"] for op in second["kernels"][0]["operations"]
    ]
    assert kernel["launch"]["grid_size"] == 16
    assert kernel["launch"]["threads_per_block"] == 128
    assert [op["category"] for op in kernel["operations"]] == [
        "fill", "copy", "copy", "gemm", "copy"
    ]
    assert kernel["pipeline"]["status"] == "explicit"
    assert kernel["pipeline"]["pipelines"][0]["stage_count"] == 2
    assert len(kernel["pipeline"]["pipelines"][0]["operation_ids"]) == 3
    assert estimate_flops_from_mod(function) == 2 * 256**3
    assert {edge["kind"] for edge in kernel["dependencies"]["edges"]} >= {"RAW", "WAW"}
    gemm = next(op for op in kernel["operations"] if op["category"] == "gemm")
    assert gemm["hardware_levels"] == {
        "read": ["local", "shared"], "write": ["local"]
    }
    assert len(gemm["dependency_ids"]) == 3
    assert "workload" not in kernel
    assert "prediction" not in kernel


def test_after_vectorize_falls_back_without_fabricating_tileops():
    report = analyze_tir_after_vectorize(make_lowered_elementwise(), TestArch())
    kernel = report["kernels"][0]

    assert report["analysis_phase"] == "after_vectorize"
    assert kernel["coverage"]["tileop_semantics_available"] is False
    assert kernel["coverage"]["lossless_action_claim"] is False
    assert kernel["pipeline"]["status"] == "unavailable"
    assert any(
        item["code"] == "HIGH_LEVEL_TILEOP_SEMANTICS_UNAVAILABLE"
        for item in kernel["diagnostics"]
    )
    assert any(op["category"] == "elementwise" for op in kernel["operations"])
    assert "workload" not in kernel
    assert "prediction" not in kernel


def test_rejects_invalid_inputs_and_missing_architecture():
    with pytest.raises(TypeError):
        analyze_prim_func(object(), TestArch())
    with pytest.raises(TypeError):
        analyze_prim_func(make_gemm(), None)
