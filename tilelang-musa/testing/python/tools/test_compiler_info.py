"""Tests for TileLang's compiler-only information collector."""

import inspect
import json

import tilelang
import tilelang.language as T

from tilelang.tools.compiler_info import (
    SCHEMA_VERSION,
    CompilerInfoCollector,
    collect_compiler_information,
)


@T.prim_func
def pipelined_gemm(
    A: T.Tensor((128, 64), "float16"),
    B: T.Tensor((64, 128), "float16"),
    C: T.Tensor((128, 128), "float16"),
):
    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((128, 32), "float16")
        B_shared = T.alloc_shared((32, 128), "float16")
        C_local = T.alloc_fragment((128, 128), "float32")
        T.clear(C_local)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(A[0, k * 32], A_shared)
            T.copy(B[k * 32, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C)


@T.prim_func
def strided_add(
    A: T.Tensor((32768,), "float32"),
    B: T.Tensor((32768,), "float32"),
    C: T.Tensor((16384,), "float32"),
):
    with T.Kernel(8, threads=256) as bx:
        for tx, local in T.Parallel(256, 8):
            output_index = (bx * 256 + tx) * 8 + local
            input_index = output_index * 2
            C[output_index] = A[input_index] + B[input_index]


def test_collector_is_a_pass_instrument_and_does_not_change_compile_api():
    collector = CompilerInfoCollector()
    assert isinstance(collector, tilelang.tvm.instrument.PassInstrument)
    assert collector.pass_sequence == []
    parameters = inspect.signature(tilelang.compile).parameters
    assert "instruments" in parameters
    assert "compiler_info" not in parameters


def test_offline_collection_is_json_serializable_and_captures_pipeline():
    information = collect_compiler_information(
        pipelined_gemm,
        target={"kind": "musa", "arch": "mp_31"},
    )
    assert information["schema_version"] == SCHEMA_VERSION
    assert information["collection_scope"] == (
        "compiler_only_no_kernel_launch_no_profiler_no_performance_model"
    )
    assert information["coverage"]["complete_required_stage_capture"]
    assert information["source"]["kernels"][0]["coverage"]["tile_op_count"] >= 5
    declarations = information["source"]["kernels"][0]["pipeline_declarations"]
    assert len(declarations) == 1
    assert declarations[0]["pipeline_id"].startswith("pipeline_")
    assert declarations[0]["num_stages"] == 2
    planning = information["pipeline_planning"]
    assert planning is not None
    pipelines = planning["kernels"][0]["pipelines"]
    assert len(pipelines) >= 1
    assert all(pipeline["statement_alignment_valid"] for pipeline in pipelines)
    assert all(
        pipeline["implementation_style"] == "mbarrier_or_tma_call_annotations"
        for pipeline in pipelines
    )
    assert any(
        statement["pipeline_calls"]
        for pipeline in pipelines
        for statement in pipeline["statements"]
    )
    pipeline_call_kinds = {
        call_kind
        for pipeline in pipelines
        for statement in pipeline["statements"]
        for call_kind in statement["call_kinds"]
    }
    assert "tl.tileop.region" not in pipeline_call_kinds
    assert any("barrier" in call_kind for call_kind in pipeline_call_kinds)
    synchronization_call_kinds = {
        call["call_kind"]
        for pipeline in pipelines
        for statement in pipeline["statements"]
        for call in statement["pipeline_calls"]
        if call["role"] == "synchronization"
    }
    assert "tirx.ptx_arrive_barrier" in synchronization_call_kinds
    layout_blocks = information["layout_inference"]["kernels"][0]["blocks"]
    assert len(layout_blocks) == 1
    assert set(layout_blocks[0]["block_names"]) == {"root", "tilelang_root"}
    assert layout_blocks[0]["deduplicated_equivalent_block_count"] == 2
    assert information["final_implementation"]["kernels"][0]["launch"][
        "threads_per_block"
    ]
    assert information["unavailable"]["register_count"].startswith(
        "requires structured MCC"
    )
    json.dumps(information, sort_keys=True)


def test_existing_lower_artifact_can_finalize_without_device_compile():
    collector = CompilerInfoCollector()
    target = tilelang.tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target, tilelang.tvm.transform.PassContext(
        opt_level=3, instruments=[collector]
    ):
        artifact = tilelang.lower(
            pipelined_gemm,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    information = collector.finalize(
        source_prim_func=pipelined_gemm,
        compiled_artifact=artifact,
        target=target,
    )
    assert information["identity"]["kernel_source_sha256"]
    assert information["final_implementation"]["kernel_source_character_count"] > 0


def test_add_and_relu_have_independent_exact_elementwise_lineage():
    from workloads.elementwise_tilelang import make_add_prim_func, make_relu_prim_func

    cases = [
        (make_add_prim_func(), "add", 2, 2048, 8),
        (make_relu_prim_func(), "max_zero", 1, 1024, 4),
    ]
    for prim_func, kind, input_count, elements_per_block, elements_per_thread in cases:
        information = collect_compiler_information(
            prim_func, target={"kind": "musa", "arch": "mp_31"}
        )
        kernel = information["source"]["kernels"][0]
        assert kernel["coverage"]["tile_op_count"] == 0
        assert kernel["coverage"]["elementwise_op_count"] == 1
        operation = kernel["elementwise_ops"][0]
        assert operation["expression"]["kind"] == kind
        assert operation["expression"]["input_count"] == input_count
        assert operation["expression"]["operations_per_element"] == 1
        assert operation["access_pattern"] == "same_index_contiguous"
        assert operation["shape"] == ["16384"]
        assert operation["dtype"] == "float32"
        assert operation["elements_per_block"] == elements_per_block
        assert operation["elements_per_thread"] == elements_per_thread
        assert len(operation["writes"]) == 1
        assert information["lineage"]["status"] == "unavailable_or_incomplete"
        assert information["elementwise_lineage"]["status"] == "compiler_exact_lineage"
        assert information["coverage"]["compiler_exact_elementwise_lineage"]
        assert information["coverage"]["source_tile_op_count"] == 0
        assert information["coverage"]["source_elementwise_op_count"] == 1


def test_strided_elementwise_access_is_not_claimed_contiguous():
    information = collect_compiler_information(
        strided_add, target={"kind": "musa", "arch": "mp_31"}
    )
    operation = information["source"]["kernels"][0]["elementwise_ops"][0]
    assert operation["expression"]["kind"] == "add"
    assert not operation["expression"]["same_index"]
    assert operation["access_pattern"] == "unsupported"
    assert information["elementwise_lineage"]["status"] == "compiler_exact_lineage"
