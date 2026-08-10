"""ABI tests for automatic provenance Trace attachment."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm


@T.prim_func
def copy_primfunc(
    A: T.Tensor((16,), "float32"),
    B: T.Tensor((16,), "float32"),
):
    for i in T.serial(16):
        B[i] = A[i]


@T.prim_func
def duplicated_lowered_tileop(A_handle: T.handle, Trace_handle: T.handle):
    A = T.match_buffer(A_handle, (32,), "float32")
    Trace = T.match_buffer(Trace_handle, (4,), "int64")
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    with T.attr(0, "tl.provenance_ordinal", 7):
        A[tx] = A[tx] + T.float32(1)
    with T.attr(0, "tl.provenance_ordinal", 7):
        A[tx] = A[tx] + T.float32(2)


@T.prim_func
def duplicated_warp_specialized_tileop(A_handle: T.handle, Trace_handle: T.handle):
    A = T.match_buffer(A_handle, (32,), "float32")
    Trace = T.match_buffer(Trace_handle, (4,), "int64")
    tx = T.launch_thread("threadIdx.x", 32)
    with T.attr([16, 16], "kWarpSpecializationScope", 0):
        if 16 <= tx:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(1)
        else:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(2)


@T.prim_func
def malformed_warp_specialized_tileop(A_handle: T.handle, Trace_handle: T.handle):
    A = T.match_buffer(A_handle, (32,), "float32")
    Trace = T.match_buffer(Trace_handle, (4,), "int64")
    tx = T.launch_thread("threadIdx.x", 32)
    with T.attr([16, 16], "kWarpSpecializationScope", 0):
        if tx < 16:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(1)
        else:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(2)


@T.prim_func
def malformed_warp_partition_extent(A_handle: T.handle, Trace_handle: T.handle):
    A = T.match_buffer(A_handle, (32,), "float32")
    Trace = T.match_buffer(Trace_handle, (4,), "int64")
    tx = T.launch_thread("threadIdx.x", 32)
    with T.attr([8, 16], "kWarpSpecializationScope", 0):
        if 16 <= tx:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(1)
        else:
            with T.attr(0, "tl.provenance_ordinal", 7):
                A[tx] = A[tx] + T.float32(2)


def test_trace_buffer_is_appended_without_mutating_baseline():
    original = copy_primfunc.script(show_meta=True)
    module = tvm.IRModule({"copy_primfunc": copy_primfunc})
    transformed = tilelang.transform.AttachProvenanceTraceBuffer("Trace", 4)(module)["copy_primfunc"]
    assert len(copy_primfunc.params) == 2
    assert len(transformed.params) == 3
    assert copy_primfunc.script(show_meta=True) == original
    trace = transformed.buffer_map[transformed.params[-1]]
    assert str(trace.dtype) == "int64"
    assert [int(value) for value in trace.shape] == [4]
    assert int(transformed.attrs["tl.provenance_trace_abi_version"]) == 1
    assert int(transformed.attrs["tl.provenance_trace_extent"]) == 4
    assert int(transformed.attrs["tl.provenance_trace_parameter_index"]) == 2


def test_trace_buffer_rejects_name_collision_and_invalid_extent():
    module = tvm.IRModule({"copy_primfunc": copy_primfunc})
    with pytest.raises(Exception, match="name already exists"):
        tilelang.transform.AttachProvenanceTraceBuffer("A", 2)(module)
    with pytest.raises(ValueError, match="positive"):
        tilelang.transform.AttachProvenanceTraceBuffer("Trace", 0)


def test_multi_instance_trace_semantics_are_explicit_and_bounded():
    module = tvm.IRModule({"duplicated_lowered_tileop": duplicated_lowered_tileop})
    with pytest.raises(Exception, match="exactly one"):
        tilelang.transform.ProvenanceTraceInjection(
            "Trace", 7, trace_semantics="single")(module)

    envelope = tilelang.transform.ProvenanceTraceInjection(
        "Trace", 7, trace_semantics="source_envelope")(module)
    envelope_script = envelope.script()
    assert envelope_script.count('T.call_extern("int64", "clock64")') == 2

    instances = tilelang.transform.ProvenanceTraceInjection(
        "Trace", 7, trace_semantics="lowered_instance")(module)
    instance_script = instances.script()
    assert instance_script.count('T.call_extern("int64", "clock64")') == 4
    for slot in range(4):
        assert f"Trace[{slot}]" in instance_script


def test_multi_instance_trace_uses_each_warp_partition_leader():
    module = tvm.IRModule({
        "duplicated_warp_specialized_tileop": duplicated_warp_specialized_tileop,
    })
    instances = tilelang.transform.ProvenanceTraceInjection(
        "Trace", 7, trace_semantics="lowered_instance")(module)
    script = instances.script()
    assert script.count('T.call_extern("int64", "clock64")') == 4
    scope = script.index('T.attr([16, 16], "kWarpSpecializationScope", 0)')
    producer = script.index("if ", scope)
    consumer = script.index("else:", producer)
    assert script.index("if v == 16:", producer, consumer) >= 0
    assert script.index("if v == 0:", consumer) >= 0


def test_malformed_warp_partition_condition_fails_closed():
    module = tvm.IRModule({
        "malformed_warp_specialized_tileop": malformed_warp_specialized_tileop,
    })
    with pytest.raises(Exception, match="threadIdx.x >= consumer extent"):
        tilelang.transform.ProvenanceTraceInjection(
            "Trace", 7, trace_semantics="lowered_instance")(module)


def test_malformed_warp_partition_extent_fails_closed():
    module = tvm.IRModule({
        "malformed_warp_partition_extent": malformed_warp_partition_extent,
    })
    with pytest.raises(Exception, match="partition extents must sum"):
        tilelang.transform.ProvenanceTraceInjection(
            "Trace", 7, trace_semantics="lowered_instance")(module)
