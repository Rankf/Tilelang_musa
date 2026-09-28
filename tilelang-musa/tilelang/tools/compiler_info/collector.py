"""Structured compiler facts for TileLang programs.

Collection uses TVM's existing :class:`PassInstrument` callbacks and the
existing :class:`CompiledArtifact` fields.  Values in ``compiler_observed``
sections come directly from TileLang IR or compiler artifacts.  Workload and
dependency summaries are explicitly labelled ``semantic_derived``.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from tilelang import tvm


SCHEMA_VERSION = "tilelang/compiler-info-v1"
COMPILER_OBSERVED = "compiler_observed"
SEMANTIC_DERIVED = "semantic_derived"
UNAVAILABLE = "unavailable"

_STAGE_RULES = (
    ("S10_PIPELINE_PLANNED", "PipelinePlanning"),
    ("S20_PIPELINE_INJECTED", "InjectSoftwarePipeline"),
    ("S30_LAYOUT_INFERRED", "LayoutInference"),
    ("S40_TILE_OP_LOWERED", "LowerTileOp"),
    ("S45_VECTORIZED", "VectorizeLoop"),
    ("S50_STORAGE_REWRITTEN", "StorageRewrite"),
)
_REQUIRED_STAGES = {
    "S10_PIPELINE_PLANNED",
    "S20_PIPELINE_INJECTED",
    "S30_LAYOUT_INFERRED",
    "S40_TILE_OP_LOWERED",
    "S50_STORAGE_REWRITTEN",
    "S60_FINAL_DEVICE_IR",
}
_LAYOUT_KEYS = {
    "layout_map",
    "k_major_map",
    "sqmma_map",
    "sqmma_inst_split_map",
    "parallel_loop_layout",
    "parallel_loop_predicate",
}
_PIPELINE_ARRAY_KEYS = {
    "software_pipeline_stage": "stage",
    "software_pipeline_order": "order",
    "software_pipeline_tma_copies": "tma_copy",
    "software_pipeline_async_producers": "async_producer",
    "software_pipeline_async_producer_groups": "async_producer_group",
}
_PIPELINE_CALL_KEYS = {
    "barrier",
    "emit_arrive",
    "is_tma_copy",
    "tl.pipeline_mbar_phase_expr",
}
_SYNC_TOKENS = (
    "barrier",
    "wait",
    "fence",
    "storage_sync",
    "commit_group",
    "arrive",
)
_ACTION_FAMILIES = {
    "copy": "copy",
    "async_copy": "copy",
    "tma_copy": "copy",
    "gemm": "gemm",
    "gemm_py": "gemm",
    "gemm_sp": "gemm_sparse",
    "reduce": "reduce",
    "cumsum": "scan",
    "atomicadd": "atomic",
    "atomicmax": "atomic",
    "atomicmin": "atomic",
    "fill": "fill",
    "clear": "fill",
    "im2col": "transform_copy",
    "transpose": "transform_copy",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}_{_sha256(encoded)[:16]}"


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    for candidate in (getattr(value, "value", None), value):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass
    return None


def _name(value: Any) -> str:
    for attribute in ("name", "name_hint"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return str(candidate)
    return str(value)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    parsed = _integer(value)
    if parsed is not None:
        return parsed
    if hasattr(value, "items"):
        try:
            return {_name(key): _json_value(item) for key, item in value.items()}
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)) or type(value).__name__ in {
        "Array",
        "ShapeTuple",
    }:
        try:
            return [_json_value(item) for item in value]
        except TypeError:
            pass
    return str(value)


def _attrs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        return {str(key): _json_value(item) for key, item in value.items()}
    except AttributeError:
        return {}


def _module_script(value: Any) -> str:
    try:
        return value.script(show_meta=True)
    except TypeError:
        return value.script()


def _post_order(root: Any) -> list[Any]:
    nodes: list[Any] = []
    tvm.tirx.stmt_functor.post_order_visit(root, nodes.append)
    return nodes


def _prim_funcs(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, tvm.tirx.PrimFunc):
        symbol = value.attrs.get("global_symbol", "main") if value.attrs else "main"
        return [(str(symbol), value)]
    functions = getattr(value, "functions", None)
    if functions is None:
        return []
    result = []
    for global_var, function in functions.items():
        if isinstance(function, tvm.tirx.PrimFunc):
            result.append((_name(global_var), function))
    return result


def _op_name(call: Any) -> str:
    op = getattr(call, "op", None)
    return str(getattr(op, "name", op))


def _call_annotations(call: Any) -> dict[str, Any]:
    return _attrs(getattr(call, "annotations", None))


def _span(value: Any) -> dict[str, Any] | None:
    span = getattr(value, "span", None)
    if span is None:
        return None
    source_name = getattr(span, "source_name", None)
    return {
        "source_name": _name(source_name) if source_name is not None else None,
        "line": _integer(getattr(span, "line", None)),
        "end_line": _integer(getattr(span, "end_line", None)),
        "column": _integer(getattr(span, "column", None)),
        "end_column": _integer(getattr(span, "end_column", None)),
    }


def _dtype_bytes(dtype: Any) -> int | None:
    try:
        data_type = tvm.DataType(str(dtype))
        bits = int(data_type.bits) * int(data_type.lanes)
        return (bits + 7) // 8
    except (AttributeError, TypeError, ValueError):
        return None


def _static_product(values: list[Any]) -> int | None:
    product = 1
    for value in values:
        parsed = _integer(value)
        if parsed is None:
            return None
        product *= parsed
    return product


def _buffer_record(buffer: Any, origin: str) -> dict[str, Any]:
    shape = [str(value) for value in buffer.shape]
    elements = _static_product(list(buffer.shape))
    width = _dtype_bytes(buffer.dtype)
    return {
        "buffer_id": _stable_id(
            "buffer",
            str(buffer.name),
            origin,
            shape,
            str(buffer.dtype),
            str(buffer.scope()),
        ),
        "name": str(buffer.name),
        "origin": origin,
        "shape": shape,
        "dtype": str(buffer.dtype),
        "scope": str(buffer.scope()),
        "elements": elements,
        "allocation_bytes": elements * width
        if elements is not None and width is not None
        else None,
        "data_alignment": _integer(getattr(buffer, "data_alignment", None)),
        "offset_factor": _integer(getattr(buffer, "offset_factor", None)),
        "evidence_level": COMPILER_OBSERVED,
    }


def _region(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, tvm.tirx.Call) or not value.args:
        return None
    load = value.args[0]
    buffer = getattr(load, "buffer", None)
    if buffer is None:
        return None
    shape_args = list(value.args[2:])
    elements = _static_product(shape_args)
    width = _dtype_bytes(buffer.dtype)
    return {
        "buffer": str(buffer.name),
        "scope": str(buffer.scope()),
        "indices": [str(item) for item in getattr(load, "indices", [])],
        "shape": [str(item) for item in shape_args],
        "elements": elements,
        "logical_bytes": elements * width
        if elements is not None and width is not None
        else None,
        "region_op": _op_name(value),
        "evidence_level": COMPILER_OBSERVED,
    }


def _action_family(op_name: str) -> str:
    short = op_name.rsplit(".", 1)[-1]
    return _ACTION_FAMILIES.get(short, "unknown")


def _action_accesses(
    call: Any, family: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regions = [_region(argument) for argument in call.args]
    if family in {"copy", "transform_copy"}:
        read_indices, write_indices = (0,), (1,)
    elif family in {"gemm", "gemm_sparse"}:
        read_indices, write_indices = (0, 1, 2), (2,)
    elif family in {"reduce", "scan"}:
        read_indices, write_indices = (0,), (1,)
    elif family == "atomic":
        read_indices, write_indices = (0, 1), (1,)
    elif family == "fill":
        read_indices, write_indices = (), (0,)
    else:
        read_indices, write_indices = tuple(range(len(regions))), ()
    reads = [
        regions[index]
        for index in read_indices
        if index < len(regions) and regions[index] is not None
    ]
    writes = [
        regions[index]
        for index in write_indices
        if index < len(regions) and regions[index] is not None
    ]
    return reads, writes


def _gemm_information(call: Any, family: str) -> dict[str, Any] | None:
    if family not in {"gemm", "gemm_sparse"}:
        return None
    offsets = (5, 6, 7) if family == "gemm" else (7, 8, 9)
    values = [call.args[index] if index < len(call.args) else None for index in offsets]
    dimensions = [_integer(value) for value in values]
    flops = None
    if all(value is not None for value in dimensions):
        m, n, k = dimensions
        flops = 2 * m * n * k
    return {
        "m": dimensions[0] if dimensions[0] is not None else str(values[0]),
        "n": dimensions[1] if dimensions[1] is not None else str(values[1]),
        "k": dimensions[2] if dimensions[2] is not None else str(values[2]),
        "flops_per_invocation": flops,
        "sparse_dense_equivalent": family == "gemm_sparse",
        "evidence_level": SEMANTIC_DERIVED,
    }


def _buffer_access(buffer: Any, indices: Any) -> dict[str, Any]:
    """Return one observed scalar buffer access without predicting traffic."""
    return {
        "buffer": str(buffer.name),
        "scope": str(buffer.scope()),
        "indices": [str(index) for index in indices],
        "dtype": str(buffer.dtype),
        "evidence_level": COMPILER_OBSERVED,
    }


def _is_zero_immediate(value: Any) -> bool:
    return type(value).__name__ in {"IntImm", "FloatImm"} and _json_value(value) == 0


def _same_indices(left: Any, right: Any) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(tvm.ir.structural_equal(a, b) for a, b in zip(left, right))
    except AttributeError:
        return all(str(a) == str(b) for a, b in zip(left, right))


def _is_contiguous_1d_index(
    index: Any,
    function: Any,
    loop_vars: list[Any],
    loop_records: list[dict[str, Any]],
    elements_per_block: Any,
) -> bool:
    """Prove the exact row-major block/loop flattening used by version one."""
    if not isinstance(elements_per_block, int) or elements_per_block <= 0:
        return False
    serial_loops = [
        (loop_var, record)
        for loop_var, record in zip(loop_vars, loop_records)
        if record.get("thread_binding") is None
    ]
    extents = [record.get("extent_value") for _, record in serial_loops]
    if not extents or any(
        not isinstance(extent, int) or extent <= 0 for extent in extents
    ):
        return False
    block_axes = []
    for node in _post_order(function.body):
        if type(node).__name__ == "For":
            binding = getattr(node, "thread_binding", None)
            if binding is not None:
                axis = getattr(binding, "thread_tag", None)
                axis = getattr(axis, "value", axis)
                extent = _integer(node.extent)
                if (
                    axis is not None
                    and str(axis).startswith("blockIdx")
                    and extent != 1
                ):
                    block_axes.append(node.loop_var)
            continue
        if type(node).__name__ != "AttrStmt" or str(node.attr_key) != "thread_extent":
            continue
        axis = getattr(getattr(node, "node", None), "thread_tag", None)
        axis = getattr(axis, "value", axis)
        extent = _integer(node.value)
        if axis is not None and str(axis).startswith("blockIdx") and extent != 1:
            variable = getattr(getattr(node, "node", None), "var", None)
            if variable is not None:
                block_axes.append(variable)
    if len(block_axes) != 1:
        return False
    variables = [block_axes[0], *(loop_var for loop_var, _ in serial_loops)]
    try:
        coefficients = list(tvm.arith.detect_linear_equation(index, variables))
        analyzer = tvm.arith.Analyzer()
        parsed = [_integer(analyzer.simplify(value)) for value in coefficients]
    except (AttributeError, TypeError, ValueError):
        return False
    expected_loop_coefficients = []
    stride = 1
    for extent in reversed(extents):
        expected_loop_coefficients.append(stride)
        stride *= extent
    expected_loop_coefficients.reverse()
    return parsed == [elements_per_block, *expected_loop_coefficients, 0]


def _elementwise_expression(value: Any, store_indices: Any) -> dict[str, Any]:
    """Classify exact unfused elementwise forms supported downstream."""
    def collect_loads(expr: Any) -> list[Any]:
        found: list[Any] = []
        try:
            tvm.tirx.stmt_functor.post_order_visit(
                expr,
                lambda node: found.append(node)
                if isinstance(node, tvm.tirx.BufferLoad) else None,
            )
        except (AttributeError, TypeError):
            found = [
                item for item in (getattr(expr, "a", None), getattr(expr, "b", None))
                if isinstance(item, tvm.tirx.BufferLoad)
            ]
        return found

    root_kind = type(value).__name__
    operands = [getattr(value, "a", None), getattr(value, "b", None)]
    loads = collect_loads(value)
    kind = "unsupported"
    if root_kind == "Add" and len(loads) == 2:
        kind = "add"
    elif (
        root_kind == "Max"
        and len(loads) == 1
        and any(_is_zero_immediate(item) for item in operands)
    ):
        kind = "max_zero"
    else:
        node = value
        while type(node).__name__ == "Cast":
            node = getattr(node, "value", None)
        if isinstance(node, tvm.tirx.Call) and len(loads) == 1:
            name = _op_name(node).lower()
            for candidate in ("exp", "log", "sqrt"):
                if name == candidate or name.endswith("." + candidate):
                    kind = candidate
                    break
    same_index = bool(loads) and all(
        _same_indices(load.indices, store_indices) for load in loads
    )
    return {
        "kind": kind,
        "root_node_type": root_kind,
        "input_count": len(loads),
        "operations_per_element": 1 if kind != "unsupported" else None,
        "same_index": same_index,
        "raw_expression": str(value),
        "evidence": {
            "root_node_type": COMPILER_OBSERVED,
            "raw_expression": COMPILER_OBSERVED,
            "kind": SEMANTIC_DERIVED,
            "input_count": SEMANTIC_DERIVED,
            "operations_per_element": SEMANTIC_DERIVED,
            "same_index": SEMANTIC_DERIVED,
        },
    }


def _multiply(values: list[int | str]) -> int | str:
    numeric = 1
    symbolic = []
    for value in values:
        if isinstance(value, int):
            numeric *= value
        else:
            symbolic.append(str(value))
    if not symbolic:
        return numeric
    terms = ([str(numeric)] if numeric != 1 else []) + symbolic
    return " * ".join(terms) if terms else 1


def _launch_information(function: Any) -> dict[str, Any]:
    axes: dict[str, int | str] = {}
    for node in _post_order(function.body):
        if type(node).__name__ == "For":
            binding = getattr(node, "thread_binding", None)
            if binding is not None:
                axis = getattr(binding, "thread_tag", None)
                axis = getattr(axis, "value", axis)
                if axis is not None:
                    extent = _integer(node.extent)
                    axes[str(axis)] = (
                        extent if extent is not None else str(node.extent)
                    )
            continue
        if type(node).__name__ != "AttrStmt" or str(node.attr_key) != "thread_extent":
            continue
        axis = getattr(getattr(node, "node", None), "thread_tag", None)
        axis = getattr(axis, "value", axis)
        if axis is None:
            continue
        extent = _integer(node.value)
        axes[str(axis)] = extent if extent is not None else str(node.value)
    attrs = _attrs(function.attrs)
    attr_extents = attrs.get("thread_extent")
    if isinstance(attr_extents, dict):
        for axis, extent in attr_extents.items():
            axes.setdefault(str(axis), extent)
    block_axes = {
        key: value for key, value in axes.items() if key.startswith("blockIdx")
    }
    thread_axes = {
        key: value for key, value in axes.items() if key.startswith("threadIdx")
    }
    grid_size = _multiply(list(block_axes.values()))
    threads = _multiply(list(thread_axes.values()))
    target = function.attrs.get("target") if function.attrs else None
    target_attrs = getattr(target, "attrs", None)
    warp_size = (
        _integer(target_attrs.get("thread_warp_size"))
        if target_attrs is not None
        else None
    )
    warps = (
        threads // warp_size
        if isinstance(threads, int)
        and isinstance(warp_size, int)
        and warp_size > 0
        and threads % warp_size == 0
        else None
    )
    return {
        "block_axes": block_axes,
        "thread_axes": thread_axes,
        "grid_size": grid_size,
        "threads_per_block": threads,
        "warp_size": warp_size,
        "warps_per_block": warps,
        "kernel_launch_params": attrs.get("tirx.kernel_launch_params"),
        "dynamic_shared_memory_bytes": attrs.get("dyn_shared_memory_buf"),
        "cluster_dims": attrs.get("cluster_dims"),
        "tma_descriptor_args": attrs.get("tma_descriptor_args"),
        "target": attrs.get("target"),
        "evidence_level": COMPILER_OBSERVED,
    }


def _source_kernel(name: str, function: Any) -> dict[str, Any]:
    buffers: dict[str, dict[str, Any]] = {
        str(buffer.name): _buffer_record(buffer, "parameter")
        for buffer in function.buffer_map.values()
    }
    loops: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    elementwise_actions: list[dict[str, Any]] = []

    @tvm.tirx.functor.visitor
    class SourceVisitor(tvm.tirx.PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.loop_stack: list[dict[str, Any]] = []
            self.loop_vars: list[Any] = []
            self.let_bindings: dict[Any, Any] = {}
            self.conditions: list[str] = []
            self.loop_ordinal = 0
            self.action_ordinal = 0
            self.elementwise_ordinal = 0

        def visit_for_(self, op):
            record = {
                "loop_id": f"loop/{self.loop_ordinal}",
                "ordinal": self.loop_ordinal,
                "var": str(op.loop_var),
                "min": str(op.min),
                "extent": str(op.extent),
                "extent_value": _integer(op.extent),
                "kind": str(op.kind),
                "thread_binding": (
                    str(op.thread_binding)
                    if op.thread_binding is not None
                    else None
                ),
                "annotations": _attrs(op.annotations),
                "span": _span(op),
                "evidence_level": COMPILER_OBSERVED,
            }
            self.loop_ordinal += 1
            loops.append(record)
            self.loop_stack.append(record)
            self.loop_vars.append(op.loop_var)
            try:
                super().visit_for_(op)
            finally:
                self.loop_vars.pop()
                self.loop_stack.pop()

        def visit_let_stmt_(self, op):
            self.visit_expr(op.value)
            marker = object()
            previous = self.let_bindings.get(op.var, marker)
            self.let_bindings[op.var] = tvm.tirx.stmt_functor.substitute(
                op.value, self.let_bindings
            )
            try:
                self.visit_stmt(op.body)
            finally:
                if previous is marker:
                    del self.let_bindings[op.var]
                else:
                    self.let_bindings[op.var] = previous

        def visit_bind_(self, op):
            self.visit_expr(op.value)
            self.let_bindings[op.var] = tvm.tirx.stmt_functor.substitute(
                op.value, self.let_bindings
            )

        def visit_seq_stmt_(self, op):
            previous = dict(self.let_bindings)
            try:
                for statement in op.seq:
                    self.visit_stmt(statement)
            finally:
                self.let_bindings = previous

        def visit_if_then_else_(self, op):
            self.visit_expr(op.condition)
            condition = str(op.condition)
            self.conditions.append(condition)
            self.visit_stmt(op.then_case)
            self.conditions.pop()
            if op.else_case is not None:
                self.conditions.append(f"not ({condition})")
                self.visit_stmt(op.else_case)
                self.conditions.pop()

        def visit_evaluate_(self, op):
            call = op.value
            if isinstance(call, tvm.tirx.Call):
                op_name = _op_name(call)
                if op_name.startswith("tl."):
                    family = _action_family(op_name)
                    reads, writes = _action_accesses(call, family)
                    annotations = _call_annotations(call)
                    source_ordinal = _integer(annotations.get("tl_provenance_ordinal"))
                    if source_ordinal is None:
                        source_ordinal = self.action_ordinal
                    serial_factors: list[int | str] = []
                    for loop in self.loop_stack:
                        if loop["thread_binding"]:
                            continue
                        serial_factors.append(
                            loop["extent_value"]
                            if loop["extent_value"] is not None
                            else loop["extent"]
                        )
                    calls_per_block = _multiply(serial_factors)
                    action_id = _stable_id(
                        "tileop", name, source_ordinal, op_name, str(call)
                    )
                    record = {
                        "op_id": action_id,
                        "source_ordinal": source_ordinal,
                        "op_name": op_name,
                        "kind": op_name.rsplit(".", 1)[-1],
                        "canonical_family": family,
                        "reads": reads,
                        "writes": writes,
                        "raw_args": [str(value) for value in call.args],
                        "annotations": annotations,
                        "loop_ids": [loop["loop_id"] for loop in self.loop_stack],
                        "condition_context": list(self.conditions),
                        "calls_per_block": calls_per_block,
                        "gemm": _gemm_information(call, family),
                        "span": _span(call) or _span(op),
                        "evidence_level": COMPILER_OBSERVED,
                    }
                    actions.append(record)
                    self.action_ordinal += 1
            super().visit_evaluate_(op)

        def visit_buffer_store_(self, op):
            if not str(op.buffer.scope()).startswith("global"):
                super().visit_buffer_store_(op)
                return
            value = tvm.tirx.stmt_functor.substitute(op.value, self.let_bindings)
            indices = [
                tvm.tirx.stmt_functor.substitute(index, self.let_bindings)
                for index in op.indices
            ]
            expression = _elementwise_expression(value, indices)
            if expression["kind"] != "unsupported":
                loads = []
                tvm.tirx.stmt_functor.post_order_visit(
                    value,
                    lambda node: loads.append(node)
                    if isinstance(node, tvm.tirx.BufferLoad) else None,
                )
                serial_factors: list[int | str] = []
                for loop in self.loop_stack:
                    if loop["thread_binding"]:
                        continue
                    serial_factors.append(
                        loop["extent_value"]
                        if loop["extent_value"] is not None
                        else loop["extent"]
                    )
                elements_per_block = _multiply(serial_factors)
                threads = _launch_information(function)["threads_per_block"]
                elements_per_thread = (
                    elements_per_block // threads
                    if isinstance(elements_per_block, int)
                    and isinstance(threads, int)
                    and threads > 0
                    and elements_per_block % threads == 0
                    else None
                )
                contiguous = (
                    len(indices) == 1
                    and expression["same_index"]
                    and _is_contiguous_1d_index(
                        indices[0],
                        function,
                        self.loop_vars,
                        self.loop_stack,
                        elements_per_block,
                    )
                )
                source_ordinal = self.elementwise_ordinal
                elementwise_actions.append(
                    {
                        "op_id": _stable_id(
                            "elementwise",
                            name,
                            source_ordinal,
                            expression["kind"],
                            str(op),
                        ),
                        "source_ordinal": source_ordinal,
                        "expression": expression,
                        "reads": [
                            _buffer_access(load.buffer, load.indices) for load in loads
                        ],
                        "writes": [_buffer_access(op.buffer, indices)],
                        "shape": [str(value) for value in op.buffer.shape],
                        "dtype": str(op.buffer.dtype),
                        "loop_ids": [loop["loop_id"] for loop in self.loop_stack],
                        "condition_context": list(self.conditions),
                        "elements_per_block": elements_per_block,
                        "elements_per_thread": elements_per_thread,
                        "access_pattern": (
                            "same_index_contiguous" if contiguous else "unsupported"
                        ),
                        "span": _span(op),
                        "evidence": {
                            "reads": COMPILER_OBSERVED,
                            "writes": COMPILER_OBSERVED,
                            "shape": COMPILER_OBSERVED,
                            "dtype": COMPILER_OBSERVED,
                            "loop_ids": COMPILER_OBSERVED,
                            "elements_per_block": SEMANTIC_DERIVED,
                            "elements_per_thread": SEMANTIC_DERIVED,
                            "access_pattern": SEMANTIC_DERIVED,
                        },
                    }
                )
                self.elementwise_ordinal += 1
            super().visit_buffer_store_(op)

        def visit_block_(self, op):
            for buffer in getattr(op, "alloc_buffers", []):
                buffers.setdefault(
                    str(buffer.name), _buffer_record(buffer, "allocation")
                )
            super().visit_block_(op)

        def visit_s_block_(self, op):
            for buffer in getattr(op, "alloc_buffers", []):
                buffers.setdefault(
                    str(buffer.name), _buffer_record(buffer, "allocation")
                )
            super().visit_s_block_(op)

    SourceVisitor().visit_stmt(function.body)
    # Some TVM variants do not dispatch a Python block visitor.  The fallback
    # only adds buffers and cannot duplicate records because the name is keyed.
    for node in _post_order(function.body):
        for buffer in getattr(node, "alloc_buffers", []):
            buffers.setdefault(str(buffer.name), _buffer_record(buffer, "allocation"))

    launch = _launch_information(function)
    grid_size = launch["grid_size"]
    for action in actions:
        count = action["calls_per_block"]
        action["calls_per_kernel"] = (
            count * grid_size
            if isinstance(count, int) and isinstance(grid_size, int)
            else f"({count}) * ({grid_size})"
        )

    writers: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for action in actions:
        for region in action["reads"]:
            producer = writers.get(region["buffer"])
            if producer:
                edges.append(
                    {
                        "edge_id": _stable_id(
                            "edge", producer, action["op_id"], region["buffer"]
                        ),
                        "from": producer,
                        "to": action["op_id"],
                        "buffer": region["buffer"],
                        "kind": "data",
                        "evidence_level": SEMANTIC_DERIVED,
                    }
                )
        for region in action["writes"]:
            writers[region["buffer"]] = action["op_id"]

    per_operation = []
    kernel_totals = {"flops": 0, "logical_read_bytes": 0, "logical_write_bytes": 0}
    unknown_totals = []
    for action in actions:
        read_bytes = sum(
            item["logical_bytes"]
            for item in action["reads"]
            if isinstance(item["logical_bytes"], int)
        )
        write_bytes = sum(
            item["logical_bytes"]
            for item in action["writes"]
            if isinstance(item["logical_bytes"], int)
        )
        flops = (action.get("gemm") or {}).get("flops_per_invocation", 0) or 0
        count = action["calls_per_kernel"]
        row = {
            "op_id": action["op_id"],
            "flops_per_invocation": flops,
            "logical_read_bytes_per_invocation": read_bytes,
            "logical_write_bytes_per_invocation": write_bytes,
            "calls_per_block": action["calls_per_block"],
            "calls_per_kernel": count,
            "evidence_level": SEMANTIC_DERIVED,
        }
        per_operation.append(row)
        if isinstance(count, int):
            kernel_totals["flops"] += flops * count
            kernel_totals["logical_read_bytes"] += read_bytes * count
            kernel_totals["logical_write_bytes"] += write_bytes * count
        else:
            unknown_totals.append(action["op_id"])
    kernel_totals["logical_total_bytes"] = (
        kernel_totals["logical_read_bytes"] + kernel_totals["logical_write_bytes"]
    )

    script = _module_script(function)
    kernel_id = _stable_id("kernel", name, _sha256(script))
    pipeline_declarations = []
    for loop in loops:
        annotations = loop["annotations"]
        if not any(
            key in annotations
            for key in (
                "num_stages",
                "tl_pipelined_num_stages",
                "software_pipeline_stage",
            )
        ):
            continue
        pipeline_declarations.append(
            {
                "pipeline_id": _stable_id(
                    "pipeline",
                    name,
                    loop["ordinal"],
                    loop["var"],
                    loop["extent"],
                ),
                "loop_id": loop["loop_id"],
                "loop_ordinal": loop["ordinal"],
                "num_stages": annotations.get(
                    "num_stages", annotations.get("tl_pipelined_num_stages")
                ),
                "annotations": annotations,
                "evidence_level": COMPILER_OBSERVED,
            }
        )
    return {
        "kernel_id": kernel_id,
        "name": str(function.attrs.get("global_symbol", name))
        if function.attrs
        else name,
        "attrs": _attrs(function.attrs),
        "launch": launch,
        "buffers": list(buffers.values()),
        "loops": loops,
        "pipeline_declarations": pipeline_declarations,
        "tile_ops": actions,
        "elementwise_ops": elementwise_actions,
        "dependencies": {
            "nodes": [
                {"op_id": item["op_id"], "family": item["canonical_family"]}
                for item in actions
            ],
            "edges": edges,
            "scope": "single source Tile IR snapshot; conservative last-writer buffer matching",
            "evidence_level": SEMANTIC_DERIVED,
        },
        "static_workload": {
            "per_operation": per_operation,
            "per_kernel_launch": kernel_totals,
            "unknown_or_symbolic_operation_ids": unknown_totals,
            "value_kind": "logical_static_workload_not_measured_bus_traffic",
            "evidence_level": SEMANTIC_DERIVED,
        },
        "coverage": {
            "tile_op_count": len(actions),
            "known_family_count": sum(
                item["canonical_family"] != "unknown" for item in actions
            ),
            "unknown_family_count": sum(
                item["canonical_family"] == "unknown" for item in actions
            ),
            "elementwise_op_count": len(elementwise_actions),
            "supported_elementwise_op_count": sum(
                item["expression"]["kind"] in {"add", "max_zero", "exp", "log", "sqrt"}
                and item["access_pattern"] == "same_index_contiguous"
                for item in elementwise_actions
            ),
        },
        "evidence_level": COMPILER_OBSERVED,
    }


def _source_information(value: Any) -> dict[str, Any]:
    return {
        "snapshot": "S00_INPUT_TILE_IR",
        "kernels": [
            _source_kernel(name, function) for name, function in _prim_funcs(value)
        ],
        "evidence_level": COMPILER_OBSERVED,
    }


def _statement_sequence(body: Any) -> list[Any]:
    sequence = getattr(body, "seq", None)
    return list(sequence) if sequence is not None else [body]


def _statement_calls(
    statement: Any,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    calls = []
    ordinals = []
    pipeline_calls = []
    for node in _post_order(statement):
        if not isinstance(node, tvm.tirx.Call):
            continue
        name = _op_name(node)
        annotations = _call_annotations(node)
        is_region_helper = name in {"tl.region", "tl.tileop.region"}
        is_synchronization = _is_sync_call(name)
        if (name.startswith("tl.") and not is_region_helper) or is_synchronization:
            calls.append(name)
        selected = {
            key: value
            for key, value in annotations.items()
            if key in _PIPELINE_CALL_KEYS
        }
        if selected or is_synchronization:
            pipeline_calls.append(
                {
                    "call_kind": name,
                    "annotations": selected,
                    "role": "synchronization"
                    if is_synchronization
                    else "pipeline_annotated",
                    "evidence_level": COMPILER_OBSERVED,
                }
            )
        ordinal = _integer(annotations.get("tl_provenance_ordinal"))
        if ordinal is not None:
            ordinals.append(ordinal)
    return sorted(set(calls)), sorted(set(ordinals)), pipeline_calls


def _pipeline_information(value: Any, snapshot: str) -> dict[str, Any]:
    kernels = []
    for name, function in _prim_funcs(value):
        pipelines = []
        pipeline_ordinal = 0
        for node in _post_order(function.body):
            if type(node).__name__ != "For":
                continue
            annotations = _attrs(node.annotations)
            statements = _statement_sequence(node.body)
            statement_metadata = [
                _statement_calls(statement) for statement in statements
            ]
            has_pipeline_calls = any(item[2] for item in statement_metadata)
            has_pipeline_loop_annotations = any(
                key in annotations
                for key in (
                    "num_stages",
                    "tl_pipelined_num_stages",
                    "software_pipeline_stage",
                )
            )
            if not has_pipeline_loop_annotations and not has_pipeline_calls:
                continue
            arrays = {
                output_name: annotations.get(annotation_name)
                for annotation_name, output_name in _PIPELINE_ARRAY_KEYS.items()
            }
            records = []
            for index, statement in enumerate(statements):
                calls, ordinals, pipeline_calls = statement_metadata[index]
                record = {
                    "statement_index": index,
                    "call_kinds": calls,
                    "source_ordinals": ordinals,
                    "pipeline_calls": pipeline_calls,
                    "statement_sha256": _sha256(str(statement)),
                    "evidence_level": COMPILER_OBSERVED,
                }
                for output_name, array in arrays.items():
                    record[output_name] = (
                        array[index]
                        if isinstance(array, list) and index < len(array)
                        else None
                    )
                records.append(record)
            declared_lengths = {
                key: len(value_)
                for key, value_ in arrays.items()
                if isinstance(value_, list)
            }
            pipelines.append(
                {
                    "pipeline_id": _stable_id(
                        "pipeline",
                        name,
                        pipeline_ordinal,
                        str(node.loop_var),
                        str(node.extent),
                    ),
                    "loop_var": str(node.loop_var),
                    "loop_min": str(node.min),
                    "loop_extent": str(node.extent),
                    "loop_extent_value": _integer(node.extent),
                    "source_loop_ordinal": _integer(
                        annotations.get("tl_provenance_loop_ordinal")
                    ),
                    "num_stages": annotations.get(
                        "tl_pipelined_num_stages", annotations.get("num_stages")
                    ),
                    "implementation_style": (
                        "software_pipeline_arrays"
                        if "software_pipeline_stage" in annotations
                        else "mbarrier_or_tma_call_annotations"
                        if has_pipeline_calls
                        else "source_pipeline_declaration"
                    ),
                    "async_stages": annotations.get("software_pipeline_async_stages"),
                    "statements": records,
                    "annotation_lengths": declared_lengths,
                    "statement_alignment_valid": all(
                        length == len(statements)
                        for length in declared_lengths.values()
                    ),
                    "annotations": annotations,
                    "span": _span(node),
                    "evidence_level": COMPILER_OBSERVED,
                }
            )
            pipeline_ordinal += 1
        kernels.append({"name": name, "pipelines": pipelines})
    return {
        "snapshot": snapshot,
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _layout_information(value: Any) -> dict[str, Any]:
    kernels = []
    for name, function in _prim_funcs(value):
        blocks = []
        block_index_by_annotations: dict[str, int] = {}
        loops = []
        for node in _post_order(function.body):
            annotations = _attrs(getattr(node, "annotations", None))
            selected = {
                key: item for key, item in annotations.items() if key in _LAYOUT_KEYS
            }
            if not selected:
                continue
            if type(node).__name__ in {"Block", "SBlock"}:
                block_name = str(getattr(node, "name_hint", ""))
                annotation_key = json.dumps(
                    selected, sort_keys=True, separators=(",", ":")
                )
                existing_index = block_index_by_annotations.get(annotation_key)
                if existing_index is None:
                    block_index_by_annotations[annotation_key] = len(blocks)
                    blocks.append(
                        {
                            "name": block_name,
                            "block_names": [block_name],
                            "annotations": selected,
                            "deduplicated_equivalent_block_count": 1,
                            "evidence_level": COMPILER_OBSERVED,
                        }
                    )
                else:
                    record = blocks[existing_index]
                    if block_name not in record["block_names"]:
                        record["block_names"].append(block_name)
                    record["deduplicated_equivalent_block_count"] += 1
            elif type(node).__name__ == "For":
                loops.append(
                    {
                        "var": str(node.loop_var),
                        "extent": str(node.extent),
                        "annotations": selected,
                        "evidence_level": COMPILER_OBSERVED,
                    }
                )
        kernels.append({"name": name, "blocks": blocks, "loops": loops})
    return {
        "snapshot": "S30_LAYOUT_INFERRED",
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _is_sync_call(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SYNC_TOKENS)


def _implementation_information(value: Any, snapshot: str) -> dict[str, Any]:
    kernels = []
    for name, function in _prim_funcs(value):
        all_calls = [
            _op_name(node)
            for node in _post_order(function.body)
            if isinstance(node, tvm.tirx.Call)
        ]
        regions = []
        elementwise_regions = []
        for node in _post_order(function.body):
            if type(node).__name__ != "AttrStmt":
                continue
            attr_key = str(node.attr_key)
            if attr_key not in {
                "tl.provenance_ordinal",
                "tl.elementwise_provenance_ordinal",
            }:
                continue
            calls = [
                _op_name(child)
                for child in _post_order(node.body)
                if isinstance(child, tvm.tirx.Call)
            ]
            record = {
                "source_ordinal": _integer(node.value),
                "call_kinds": dict(sorted(Counter(calls).items())),
                "synchronization_calls": dict(
                    sorted(
                        Counter(call for call in calls if _is_sync_call(call)).items()
                    )
                ),
                "body_sha256": _sha256(str(node.body)),
                "evidence_level": COMPILER_OBSERVED,
            }
            if attr_key == "tl.provenance_ordinal":
                regions.append(record)
            else:
                stores = [
                    child
                    for child in _post_order(node.body)
                    if isinstance(child, tvm.tirx.BufferStore)
                ]
                record["store_count"] = len(stores)
                record["store_buffers"] = sorted(
                    Counter(str(store.buffer.name) for store in stores).items()
                )
                elementwise_regions.append(record)
        kernels.append(
            {
                "name": name,
                "provenance_regions": regions,
                "elementwise_provenance_regions": elementwise_regions,
                "all_call_kinds": dict(sorted(Counter(all_calls).items())),
                "synchronization_calls": dict(
                    sorted(
                        Counter(
                            call for call in all_calls if _is_sync_call(call)
                        ).items()
                    )
                ),
                "function_attrs": _attrs(function.attrs),
            }
        )
    return {
        "snapshot": snapshot,
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _final_information(
    value: Any, params: list[Any], kernel_source: str
) -> dict[str, Any]:
    kernels = []
    for fallback_name, function in _prim_funcs(value):
        kernels.append(
            {
                "name": str(function.attrs.get("global_symbol", fallback_name))
                if function.attrs
                else fallback_name,
                "launch": _launch_information(function),
                "attrs": _attrs(function.attrs),
                "evidence_level": COMPILER_OBSERVED,
            }
        )
    parameter_records = [
        {
            "index": index,
            "dtype": str(getattr(param, "dtype", "unknown")),
            "shape": [str(value_) for value_ in getattr(param, "shape", [])],
            "is_scalar": len(getattr(param, "shape", [])) == 0,
            "evidence_level": COMPILER_OBSERVED,
        }
        for index, param in enumerate(params or [])
    ]
    return {
        "snapshot": "S60_FINAL_DEVICE_IR",
        "kernels": kernels,
        "parameters": parameter_records,
        "lowered_operations": _implementation_information(value, "S60_FINAL_DEVICE_IR"),
        "kernel_source_sha256": _sha256(kernel_source),
        "kernel_source_character_count": len(kernel_source),
        "evidence_level": COMPILER_OBSERVED,
    }


def _lineage(
    source: dict[str, Any], final_implementation: dict[str, Any]
) -> dict[str, Any]:
    source_ops = [op for kernel in source["kernels"] for op in kernel["tile_ops"]]
    regions = [
        region
        for kernel in final_implementation["lowered_operations"]["kernels"]
        for region in kernel["provenance_regions"]
    ]
    by_ordinal: dict[int, list[dict[str, Any]]] = {}
    for region in regions:
        ordinal = region.get("source_ordinal")
        if isinstance(ordinal, int):
            by_ordinal.setdefault(ordinal, []).append(region)
    mappings = []
    for op in source_ops:
        matched = by_ordinal.get(op["source_ordinal"], [])
        mappings.append(
            {
                "source_op_id": op["op_id"],
                "source_ordinal": op["source_ordinal"],
                "source_kind": op["kind"],
                "lowered_instance_count": len(matched),
                "lowered_regions": matched,
                "evidence_level": COMPILER_OBSERVED if matched else UNAVAILABLE,
            }
        )
    exact = bool(source_ops) and all(
        item["lowered_instance_count"] > 0 for item in mappings
    )
    return {
        "status": "compiler_exact_lineage" if exact else "unavailable_or_incomplete",
        "mappings": mappings,
        "limitations": []
        if exact
        else [
            "exact lineage requires compiler provenance annotations to survive final lowering"
        ],
        "evidence_level": COMPILER_OBSERVED if exact else UNAVAILABLE,
    }


def _elementwise_lineage(
    source: dict[str, Any], final_implementation: dict[str, Any]
) -> dict[str, Any]:
    mappings = []
    source_ops = []
    final_kernels = final_implementation["lowered_operations"]["kernels"]
    for kernel_index, source_kernel in enumerate(source["kernels"]):
        kernel_ops = source_kernel["elementwise_ops"]
        source_ops.extend(kernel_ops)
        regions = (
            final_kernels[kernel_index]["elementwise_provenance_regions"]
            if kernel_index < len(final_kernels)
            else []
        )
        by_ordinal: dict[int, list[dict[str, Any]]] = {}
        for region in regions:
            ordinal = region.get("source_ordinal")
            if isinstance(ordinal, int):
                by_ordinal.setdefault(ordinal, []).append(region)
        for op in kernel_ops:
            matched = by_ordinal.get(op["source_ordinal"], [])
            mappings.append(
                {
                    "source_kernel_id": source_kernel["kernel_id"],
                    "source_op_id": op["op_id"],
                    "source_ordinal": op["source_ordinal"],
                    "source_kind": op["expression"]["kind"],
                    "lowered_instance_count": len(matched),
                    "lowered_regions": matched,
                    "evidence_level": COMPILER_OBSERVED if matched else UNAVAILABLE,
                }
            )
    exact = bool(source_ops) and all(
        item["lowered_instance_count"] > 0 for item in mappings
    )
    return {
        "status": "compiler_exact_lineage" if exact else "unavailable_or_incomplete",
        "mappings": mappings,
        "limitations": []
        if exact
        else [
            "exact elementwise lineage requires provenance attributes to survive final lowering"
        ],
        "evidence_level": COMPILER_OBSERVED if exact else UNAVAILABLE,
    }


@tvm.instrument.pass_instrument
class CompilerInfoCollector:
    """Capture compiler stages without changing the modules being compiled."""

    def __init__(self, *, retain_ir_modules: bool = True):
        self.retain_ir_modules = bool(retain_ir_modules)
        self.stage_modules: dict[str, Any] = {}
        self.stage_scripts: dict[str, str] = {}
        self.pass_sequence: list[str] = []
        self.stage_occurrences: Counter[str] = Counter()

    def run_after_pass(self, mod, pass_info):
        pass_name = str(pass_info.name)
        self.pass_sequence.append(pass_name)
        for stage, suffix in _STAGE_RULES:
            if pass_name.endswith(suffix):
                script = _module_script(mod)
                self.stage_scripts[stage] = script
                self.stage_occurrences[stage] += 1
                if self.retain_ir_modules:
                    self.stage_modules[stage] = mod

    def get_stage_module(self, stage: str) -> Any:
        """Return one retained IRModule, failing clearly when it was not kept."""
        if not self.retain_ir_modules:
            raise RuntimeError("collector was created with retain_ir_modules=False")
        if stage not in self.stage_modules:
            raise KeyError(f"compiler stage was not captured: {stage}")
        return self.stage_modules[stage]

    def finalize(
        self,
        *,
        source_prim_func: Any,
        compiled_artifact: Any,
        target: Any = None,
        target_host: Any = None,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | tuple[str, ...] | None = None,
        include_ir: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-serializable compiler-information artifact."""
        source_script = _module_script(source_prim_func)
        final_script = _module_script(compiled_artifact.device_mod)
        kernel_source = str(compiled_artifact.kernel_source)
        scripts = dict(self.stage_scripts)
        scripts["S00_INPUT_TILE_IR"] = source_script
        scripts["S60_FINAL_DEVICE_IR"] = final_script

        modules = dict(self.stage_modules)
        modules["S60_FINAL_DEVICE_IR"] = compiled_artifact.device_mod
        source = _source_information(source_prim_func)
        final_implementation = _final_information(
            compiled_artifact.device_mod,
            list(compiled_artifact.params or []),
            kernel_source,
        )
        missing = sorted(_REQUIRED_STAGES - set(scripts))
        stage_manifest = {
            stage: {
                "tir_sha256": _sha256(script),
                "character_count": len(script),
                "capture_occurrences": self.stage_occurrences.get(stage, 1),
                "evidence_level": COMPILER_OBSERVED,
            }
            for stage, script in sorted(scripts.items())
        }
        identity_payload = {
            "source_ir_sha256": _sha256(source_script),
            "final_device_ir_sha256": _sha256(final_script),
            "kernel_source_sha256": _sha256(kernel_source),
            "target": str(target) if target is not None else None,
            "target_host": str(target_host) if target_host is not None else None,
            "pass_configs": _json_value(pass_configs or {}),
            "compile_flags": list(compile_flags or []),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "compilation_id": _stable_id("compilation", identity_payload),
            "collection_scope": "compiler_only_no_kernel_launch_no_profiler_no_performance_model",
            "identity": {**identity_payload, "evidence_level": COMPILER_OBSERVED},
            "stage_manifest": stage_manifest,
            "pass_sequence": list(self.pass_sequence),
            "source": source,
            "pipeline_planning": (
                _pipeline_information(
                    modules["S10_PIPELINE_PLANNED"], "S10_PIPELINE_PLANNED"
                )
                if "S10_PIPELINE_PLANNED" in modules
                else None
            ),
            "pipeline_implementation": (
                _pipeline_information(
                    modules["S20_PIPELINE_INJECTED"], "S20_PIPELINE_INJECTED"
                )
                if "S20_PIPELINE_INJECTED" in modules
                else None
            ),
            "layout_inference": (
                _layout_information(modules["S30_LAYOUT_INFERRED"])
                if "S30_LAYOUT_INFERRED" in modules
                else None
            ),
            "tileop_lowering": (
                _implementation_information(
                    modules["S40_TILE_OP_LOWERED"], "S40_TILE_OP_LOWERED"
                )
                if "S40_TILE_OP_LOWERED" in modules
                else None
            ),
            "final_implementation": final_implementation,
            "lineage": _lineage(source, final_implementation),
            "elementwise_lineage": _elementwise_lineage(source, final_implementation),
            "coverage": {
                "required_stages": sorted(_REQUIRED_STAGES),
                "captured_stages": sorted(scripts),
                "missing_required_stages": missing,
                "complete_required_stage_capture": not missing,
                "source_tile_op_count": sum(
                    len(kernel["tile_ops"]) for kernel in source["kernels"]
                ),
                "source_elementwise_op_count": sum(
                    len(kernel["elementwise_ops"]) for kernel in source["kernels"]
                ),
                "compiler_exact_lineage": False,
                "compiler_exact_elementwise_lineage": False,
            },
            "unavailable": {
                "register_count": "requires structured MCC/backend compiler resource output",
                "spill_bytes": "requires structured MCC/backend compiler resource output",
                "measured_cache_hit_rate": "runtime profiler data, not a compiler fact",
                "measured_pipeline_stage_time": "runtime trace data, not a compiler fact",
                "measured_scale_unit_load": "runtime profiler/trace data, not a compiler fact",
                "evidence_level": UNAVAILABLE,
            },
        }
        result["coverage"]["compiler_exact_lineage"] = (
            result["lineage"]["status"] == "compiler_exact_lineage"
        )
        result["coverage"]["compiler_exact_elementwise_lineage"] = (
            result["elementwise_lineage"]["status"] == "compiler_exact_lineage"
        )
        if include_ir:
            result["ir_snapshots"] = scripts
        # Fail at the producer boundary if a future compiler object silently
        # escapes the conversion helpers.
        json.dumps(result, sort_keys=True)
        return result


def collect_compiler_information(
    prim_func: Any,
    *,
    target: Any = "auto",
    target_host: Any = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    include_ir: bool = False,
) -> dict[str, Any]:
    """Lower without device compilation or execution and collect compiler facts."""
    import tilelang
    from tilelang.backend.target import determine_target

    resolved_target = determine_target(target, return_object=True)
    collector = CompilerInfoCollector()
    configs = dict(pass_configs or {})
    if compile_flags:
        from tilelang.transform import PassConfigKey

        configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = list(compile_flags)
    with resolved_target, tvm.transform.PassContext(
        opt_level=3, config=configs, instruments=[collector]
    ):
        artifact = tilelang.lower(
            prim_func,
            target=resolved_target,
            target_host=target_host,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return collector.finalize(
        source_prim_func=prim_func,
        compiled_artifact=artifact,
        target=resolved_target,
        target_host=target_host,
        pass_configs=configs,
        compile_flags=compile_flags,
        include_ir=include_ir,
    )
