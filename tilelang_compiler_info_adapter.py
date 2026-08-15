"""Strict TileLang compiler-info adapter for TileSight's native models.

This module is deliberately owned by the performance-tool side.  It consumes
``tilelang/compiler-info-v1`` artifacts, maps proven FP16 GEMM and strict FP32
Add/ReLU facts to TileSight's existing scalar/tuple APIs, and keeps compiler
facts, adapter rules, model assumptions, and predictions separate.  It does
not import TileLang, run a kernel, or read profiler data.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any


COMPILER_INFO_SCHEMA = "tilelang/compiler-info-v1"
MODEL_REQUEST_SCHEMA = "tilesight/model-request-v1"
MODEL_RESULT_SCHEMA = "tilesight/model-result-v1"
S5000_FP16_GEMM_MEMORY_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [1, 1, 1, 2],
    # The generated MUSA kernel writes the FP32 fragment directly to an FP16
    # global output.  It does not allocate an output tile in shared memory.
    "out1": [1, 0, 1, 2],
}
S5000_FP32_RELU_MEMORY_LEVELS = {
    "in1": [1, 0, 0, 4],
    # Compiler lineage proves that the standalone kernel writes global output;
    # [0, 0, 1, 4] would describe a register-only fused intermediate instead.
    "out1": [1, 0, 1, 4],
}
S5000_FP32_ADD_MEMORY_LEVELS = {
    "in1": [1, 0, 0, 4],
    "in2": [1, 0, 0, 4],
    "out1": [1, 0, 1, 4],
}
_TILESIGHT_RNG_LOCK = threading.Lock()


class CompilerInfoAdapterError(ValueError):
    """A compiler artifact cannot be mapped without guessing."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class ModelConsistencyError(ValueError):
    """TileSight-derived structure disagrees with final compiler facts."""

    def __init__(self, diagnostics: list[dict[str, Any]]):
        super().__init__("compiler/model consistency validation failed")
        self.diagnostics = diagnostics


def _error(code: str, message: str) -> None:
    raise CompilerInfoAdapterError(code, message)


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _load_artifact(value: dict[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        artifact = value
        source = "python_object"
    else:
        path = Path(value).expanduser().resolve()
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            _error("invalid_compiler_info_file", f"cannot read {path}: {error}")
        source = str(path)
    if artifact.get("schema_version") != COMPILER_INFO_SCHEMA:
        _error(
            "unsupported_schema",
            f"expected {COMPILER_INFO_SCHEMA!r}, got {artifact.get('schema_version')!r}",
        )
    return artifact, source


def _require_s5000_compiler_target(artifact: dict[str, Any]) -> Any:
    """Reject artifacts compiled for a target other than MUSA mp_31."""
    target = artifact.get("identity", {}).get("target")
    parsed = target
    if isinstance(target, str):
        try:
            parsed = json.loads(target)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        kind = parsed.get("kind")
        architecture = parsed.get("arch")
        matches = kind == "musa" and architecture == "mp_31"
    else:
        matches = (
            isinstance(target, str)
            and "musa" in target.lower()
            and "mp_31" in target.lower()
        )
    if not matches:
        _error(
            "unsupported_target",
            f"formal S5000 modeling requires MUSA mp_31, got {target!r}",
        )
    return target


def _one(items: list[Any], code: str, description: str) -> Any:
    if len(items) != 1:
        _error(code, f"expected exactly one {description}, found {len(items)}")
    return items[0]


def _positive_int(value: Any, code: str, description: str) -> int:
    if isinstance(value, bool):
        _error(code, f"{description} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _error(code, f"{description} must be a positive integer, got {value!r}")
    if parsed <= 0 or str(value).strip() != str(parsed):
        _error(code, f"{description} must be a static positive integer, got {value!r}")
    return parsed


def _static_shape(record: dict[str, Any], description: str) -> list[int]:
    shape = record.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        _error(
            "unsupported_dynamic_shape", f"{description} requires a static 2-D shape"
        )
    return [
        _positive_int(value, "unsupported_dynamic_shape", f"{description} shape")
        for value in shape
    ]


def _static_vector_shape(record: dict[str, Any], description: str) -> list[int]:
    shape = record.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        _error(
            "unsupported_dynamic_shape", f"{description} requires a static 1-D shape"
        )
    return [
        _positive_int(value, "unsupported_dynamic_shape", f"{description} shape")
        for value in shape
    ]


def _is_scope(record: dict[str, Any], prefix: str) -> bool:
    return str(record.get("scope", "")).startswith(prefix)


def _select_source_kernel(
    artifact: dict[str, Any], kernel_id: str | None
) -> tuple[dict[str, Any], int]:
    kernels = artifact.get("source", {}).get("kernels")
    if not isinstance(kernels, list) or not kernels:
        _error("missing_source_kernel", "compiler artifact contains no source kernels")
    if kernel_id is None:
        kernel = _one(kernels, "ambiguous_kernel", "source kernel")
    else:
        kernel = _one(
            [item for item in kernels if item.get("kernel_id") == kernel_id],
            "unknown_kernel_id",
            f"source kernel with kernel_id={kernel_id!r}",
        )
    return kernel, kernels.index(kernel)


def derive_warp_shape(
    bm: int, bn: int, bk: int, warps_per_block: int
) -> tuple[int, int, int]:
    """Apply TileSight's named near-square factorization assumption."""
    if warps_per_block <= 0:
        _error("invalid_warp_count", "warps_per_block must be positive")
    candidates = []
    for split_n in range(1, warps_per_block + 1):
        if warps_per_block % split_n:
            continue
        split_m = warps_per_block // split_n
        if bm % split_m == 0 and bn % split_n == 0:
            wm, wn = bm // split_m, bn // split_n
            candidates.append((abs(math.log(wm / wn)), -split_m, wm, wn))
    if not candidates:
        _error(
            "warp_tile_unavailable",
            f"cannot factor {warps_per_block} warps over block tile ({bm}, {bn})",
        )
    _, _, wm, wn = min(candidates)
    return wm, wn, bk


def _final_kernel(artifact: dict[str, Any], source_index: int) -> dict[str, Any]:
    kernels = artifact.get("final_implementation", {}).get("kernels")
    if not isinstance(kernels, list) or not kernels:
        _error(
            "missing_final_kernel", "compiler artifact contains no final device kernel"
        )
    if len(kernels) == 1:
        return kernels[0]
    if source_index >= len(kernels):
        _error(
            "ambiguous_final_kernel", "source and final kernel lists cannot be aligned"
        )
    return kernels[source_index]


def _operation_ids(operations: list[dict[str, Any]]) -> None:
    ids = [item.get("op_id") for item in operations]
    if None in ids or len(set(ids)) != len(ids):
        _error(
            "invalid_operation_ids",
            "source TileOps require unique non-null op_id values",
        )


def _has_dependency(kernel: dict[str, Any], producer: str, consumer: str) -> bool:
    edges = kernel.get("dependencies", {}).get("edges", [])
    return any(
        item.get("from") == producer and item.get("to") == consumer for item in edges
    )


def _pipeline_for_operation(
    kernel: dict[str, Any], operation: dict[str, Any]
) -> dict[str, Any]:
    loop_ids = set(operation.get("loop_ids", []))
    candidates = [
        item
        for item in kernel.get("pipeline_declarations", [])
        if item.get("loop_id") in loop_ids
    ]
    pipeline = _one(candidates, "ambiguous_pipeline", "source pipeline containing GEMM")
    if not pipeline.get("pipeline_id"):
        _error(
            "missing_pipeline_id",
            "source pipeline requires compiler-generated pipeline_id",
        )
    return pipeline


def _planned_pipeline_ids(
    artifact: dict[str, Any], source_index: int, loop_ordinal: int
) -> list[str]:
    kernels = artifact.get("pipeline_planning", {}).get("kernels", [])
    if source_index >= len(kernels):
        return []
    return [
        item["pipeline_id"]
        for item in kernels[source_index].get("pipelines", [])
        if item.get("source_loop_ordinal") == loop_ordinal and item.get("pipeline_id")
    ]


def _require_musa_gemm_lowering(artifact: dict[str, Any], source_index: int) -> str:
    kernels = artifact.get("tileop_lowering", {}).get("kernels", [])
    calls = (
        kernels[source_index].get("all_call_kinds", {})
        if source_index < len(kernels)
        else {}
    )
    if "tl.tl_gemm" not in calls:
        _error(
            "unsupported_mma_mapping",
            "MUSA FP16 GEMM mapping requires an observed tl.tl_gemm lowered call",
        )
    return "wmma"


def _prepare_elementwise_request(
    artifact: dict[str, Any],
    source: str,
    compiler_target: Any,
    kernel: dict[str, Any],
    source_index: int,
    cache_condition: str | None,
) -> dict[str, Any]:
    """Map one strict FP32 Add/ReLU compiler fact set to a native model call."""
    if (
        artifact.get("elementwise_lineage", {}).get("status")
        != "compiler_exact_lineage"
    ):
        _error(
            "incomplete_elementwise_lineage",
            "formal elementwise requests require exact source-to-lowered store lineage",
        )
    if kernel.get("tile_ops"):
        _error(
            "unsupported_fusion", "elementwise formal modeling does not accept TileOps"
        )
    operations = kernel.get("elementwise_ops")
    if not isinstance(operations, list):
        _error(
            "missing_elementwise_ops", "source kernel elementwise_ops must be a list"
        )
    _operation_ids(operations)
    operation = _one(
        operations, "ambiguous_elementwise", "source elementwise operation"
    )
    mappings = artifact.get("elementwise_lineage", {}).get("mappings", [])
    mapping = _one(
        [
            item
            for item in mappings
            if item.get("source_op_id") == operation.get("op_id")
        ],
        "inconsistent_elementwise_lineage",
        "lineage mapping for source elementwise operation",
    )
    if (
        not isinstance(mapping.get("lowered_instance_count"), int)
        or mapping["lowered_instance_count"] <= 0
    ):
        _error(
            "inconsistent_elementwise_lineage",
            "elementwise store has no lowered region",
        )

    expression = operation.get("expression") or {}
    kind = expression.get("kind")
    expected_inputs = {"max_zero": 1, "add": 2}
    if kind not in expected_inputs:
        _error(
            "unsupported_elementwise_expression",
            f"unsupported expression kind {kind!r}",
        )
    if expression.get("input_count") != expected_inputs[kind]:
        _error(
            "unsupported_elementwise_expression",
            "expression input count is inconsistent",
        )
    if expression.get("operations_per_element") != 1:
        _error(
            "unsupported_fusion",
            "first elementwise version requires exactly one operation",
        )
    if operation.get("access_pattern") != "same_index_contiguous":
        _error(
            "unsupported_access_pattern",
            "first elementwise version requires same-index contiguous accesses",
        )
    if operation.get("condition_context"):
        _error(
            "unsupported_predication",
            "first elementwise version does not accept predicates",
        )

    reads = operation.get("reads")
    writes = operation.get("writes")
    if not isinstance(reads, list) or len(reads) != expected_inputs[kind]:
        _error(
            "unsupported_memory_path", "elementwise input access count is inconsistent"
        )
    if not isinstance(writes, list) or len(writes) != 1:
        _error(
            "unsupported_memory_path", "elementwise requires exactly one output access"
        )
    if not all(_is_scope(item, "global") for item in [*reads, *writes]):
        _error(
            "unsupported_memory_path",
            "first elementwise version requires global buffers",
        )
    if any(item.get("dtype") != "float32" for item in [*reads, *writes]):
        _error("unsupported_dtype", "current elementwise adapter supports FP32 only")

    operation_shape = _static_vector_shape(operation, "elementwise operation")
    buffers = {item.get("name"): item for item in kernel.get("buffers", [])}
    for access in [*reads, *writes]:
        buffer = buffers.get(access.get("buffer"))
        if buffer is None:
            _error(
                "unknown_buffer",
                f"operation references unknown buffer {access.get('buffer')!r}",
            )
        if buffer.get("dtype") != "float32" or not _is_scope(buffer, "global"):
            _error("unsupported_memory_path", "referenced buffers must be global FP32")
        if (
            _static_vector_shape(buffer, f"buffer {buffer.get('name')}")
            != operation_shape
        ):
            _error(
                "inconsistent_elementwise_shapes",
                "all elementwise buffers must have one shape",
            )
    n = operation_shape[0]

    elements_per_block = _positive_int(
        operation.get("elements_per_block"),
        "invalid_elementwise_tile",
        "elements_per_block",
    )
    elements_per_thread = _positive_int(
        operation.get("elements_per_thread"),
        "invalid_elementwise_tile",
        "elements_per_thread",
    )
    if n % elements_per_block:
        _error("unsupported_tail_tile", "N must be divisible by elements_per_block")
    final = _final_kernel(artifact, source_index)
    launch = final.get("launch", {})
    grid_size = _positive_int(
        launch.get("grid_size"), "invalid_launch", "final grid_size"
    )
    threads = _positive_int(
        launch.get("threads_per_block"), "invalid_launch", "final threads_per_block"
    )
    warp_size = _positive_int(
        launch.get("warp_size"), "invalid_launch", "target warp_size"
    )
    if threads % warp_size:
        _error(
            "invalid_launch", "threads_per_block is not divisible by target warp_size"
        )
    warps = threads // warp_size
    if launch.get("warps_per_block") != warps:
        _error(
            "invalid_launch", "final warps_per_block disagrees with threads / warp_size"
        )
    if elements_per_block != threads * elements_per_thread:
        _error(
            "compiler_model_tile_mismatch",
            "elements_per_block disagrees with threads * elements_per_thread",
        )
    if grid_size != n // elements_per_block:
        _error(
            "compiler_model_grid_mismatch",
            f"compiler grid {grid_size} != shape/tile grid {n // elements_per_block}",
        )
    if cache_condition is None:
        _error(
            "missing_cache_condition",
            "elementwise modeling requires an explicit cache condition",
        )
    if cache_condition not in {"cold", "warm"}:
        _error(
            "unsupported_cache_condition",
            "elementwise cache_condition must be either 'cold' or 'warm'",
        )

    model_name = {
        "max_zero": "elementwise_n_0_pipeline_wave",
        "add": "elementwise_n_n_pipeline_wave",
    }[kind]
    mem_levels = (
        S5000_FP32_RELU_MEMORY_LEVELS
        if kind == "max_zero"
        else S5000_FP32_ADD_MEMORY_LEVELS
    )
    arguments = {
        "op_shape": [n],
        "tb_shape": [elements_per_block],
        "dim_threads": [threads],
        "mem_levels": deepcopy(mem_levels),
        "num_ops": 1,
    }
    identity = {
        "compilation_id": artifact.get("compilation_id"),
        "kernel_id": kernel["kernel_id"],
        "pipeline_id": None,
        "planned_pipeline_ids": [],
        "operation_ids": {"elementwise": operation["op_id"]},
    }
    compiler_facts = {
        "grid_size": grid_size,
        "threads_per_block": threads,
        "warp_size": warp_size,
        "warps_per_block": warps,
        "elements_per_block": elements_per_block,
        "elements_per_thread": elements_per_thread,
        "dtype": "float32",
        "target": compiler_target,
        "lineage_status": artifact["elementwise_lineage"]["status"],
    }
    request = {
        "schema_version": MODEL_REQUEST_SCHEMA,
        "request_id": None,
        "input_source": source,
        "identity": identity,
        "model": {"name": model_name, "arguments": arguments},
        "model_scenario": {"cache_condition": cache_condition},
        "model_execution": {"randomness": "none"},
        "compiler_facts": compiler_facts,
        "evidence": {
            "compiler_observed": [
                "op_shape",
                "dim_threads",
                "grid_size",
                "threads_per_block",
                "warps_per_block",
                "dtype",
                "operation_id",
            ],
            "semantic_derived": [
                "expression_kind",
                "tb_shape",
                "elements_per_thread",
                "num_ops",
                "access_pattern",
            ],
            "adapter_derived": ["operand_roles", "mem_levels", "model_dispatch"],
            "model_assumption": ["cache_condition"],
            "model_predicted": [],
        },
        "assumptions": [
            {
                "field": "cache_condition",
                "rule": "tilesight_elementwise_cache_condition_v1",
                "reason": (
                    "caller-selected initial L2 state; TileSight predicts cache "
                    "residency and traffic without runtime counter input"
                ),
            }
        ],
        "unavailable": [
            "compiler_observed_register_count",
            "compiler_observed_spill_bytes",
            "runtime_measurements",
        ],
        "diagnostics": [],
    }
    request["request_id"] = _stable_id(
        "model_request",
        {
            "identity": identity,
            "model": request["model"],
            "model_scenario": request["model_scenario"],
        },
    )
    json.dumps(request, sort_keys=True)
    return request


def prepare_tilesight_request(
    compiler_info: dict[str, Any] | str | Path,
    *,
    kernel_id: str | None = None,
    cache_condition: str | None = None,
) -> dict[str, Any]:
    """Build one strict GEMM/Add/ReLU request from compiler-only information."""
    artifact, source = _load_artifact(compiler_info)
    compiler_target = _require_s5000_compiler_target(artifact)
    kernel, source_index = _select_source_kernel(artifact, kernel_id)
    if kernel.get("elementwise_ops"):
        return _prepare_elementwise_request(
            artifact,
            source,
            compiler_target,
            kernel,
            source_index,
            cache_condition,
        )
    if artifact.get("lineage", {}).get("status") != "compiler_exact_lineage":
        _error(
            "incomplete_compiler_lineage",
            "formal model requests require exact source-to-lowered TileOp lineage",
        )
    operations = kernel.get("tile_ops", [])
    if not isinstance(operations, list):
        _error("missing_tile_ops", "source kernel tile_ops must be a list")
    _operation_ids(operations)
    gemm = _one(
        [item for item in operations if item.get("canonical_family") == "gemm"],
        "ambiguous_gemm",
        "GEMM TileOp",
    )
    gemm_detail = gemm.get("gemm") or {}
    bm = _positive_int(gemm_detail.get("m"), "invalid_gemm_tile", "GEMM tile M")
    bn = _positive_int(gemm_detail.get("n"), "invalid_gemm_tile", "GEMM tile N")
    bk = _positive_int(gemm_detail.get("k"), "invalid_gemm_tile", "GEMM tile K")

    shared_inputs = [
        item for item in gemm.get("reads", []) if _is_scope(item, "shared")
    ]
    if len(shared_inputs) != 2:
        _error(
            "unsupported_memory_path", "GEMM requires exactly two shared-memory inputs"
        )

    def producer_for(shared_buffer: str) -> dict[str, Any]:
        return _one(
            [
                item
                for item in operations
                if item.get("canonical_family") == "copy"
                and any(
                    write.get("buffer") == shared_buffer and _is_scope(write, "shared")
                    for write in item.get("writes", [])
                )
                and any(_is_scope(read, "global") for read in item.get("reads", []))
                and _has_dependency(kernel, item["op_id"], gemm["op_id"])
            ],
            "unsupported_memory_path",
            f"global-to-shared producer for {shared_buffer!r}",
        )

    copy_a = producer_for(shared_inputs[0]["buffer"])
    copy_b = producer_for(shared_inputs[1]["buffer"])
    if copy_a["op_id"] == copy_b["op_id"]:
        _error(
            "unsupported_memory_path", "GEMM inputs resolve to the same copy operation"
        )
    read_a = _one(
        [item for item in copy_a.get("reads", []) if _is_scope(item, "global")],
        "unsupported_memory_path",
        "global A read",
    )
    read_b = _one(
        [item for item in copy_b.get("reads", []) if _is_scope(item, "global")],
        "unsupported_memory_path",
        "global B read",
    )
    output_copy = _one(
        [
            item
            for item in operations
            if item.get("canonical_family") == "copy"
            and any(_is_scope(read, "local") for read in item.get("reads", []))
            and any(_is_scope(write, "global") for write in item.get("writes", []))
            and _has_dependency(kernel, gemm["op_id"], item["op_id"])
        ],
        "unsupported_memory_path",
        "local-to-global GEMM output copy",
    )
    write_c = _one(
        [item for item in output_copy.get("writes", []) if _is_scope(item, "global")],
        "unsupported_memory_path",
        "global C write",
    )
    accumulator_access = _one(
        [item for item in gemm.get("writes", []) if _is_scope(item, "local")],
        "unsupported_memory_path",
        "local GEMM accumulator",
    )

    buffers = {item.get("name"): item for item in kernel.get("buffers", [])}
    try:
        buffer_a = buffers[read_a["buffer"]]
        buffer_b = buffers[read_b["buffer"]]
        buffer_c = buffers[write_c["buffer"]]
        accumulator_buffer = buffers[accumulator_access["buffer"]]
    except KeyError as error:
        _error(
            "unknown_buffer", f"operation references unknown buffer {error.args[0]!r}"
        )
    if tuple(item.get("dtype") for item in (buffer_a, buffer_b, buffer_c)) != (
        "float16",
        "float16",
        "float16",
    ):
        _error("unsupported_dtype", "current formal adapter supports FP16 A/B/C only")
    if accumulator_buffer.get("dtype") != "float32":
        _error(
            "unsupported_accumulator_dtype",
            "current formal adapter supports an FP32 GEMM accumulator only",
        )
    shape_a = _static_shape(buffer_a, "A")
    shape_b = _static_shape(buffer_b, "B")
    shape_c = _static_shape(buffer_c, "C")
    m, k = shape_a
    k_b, n = shape_b
    if k != k_b or shape_c != [m, n]:
        _error(
            "inconsistent_gemm_shapes",
            f"inconsistent GEMM shapes A={shape_a}, B={shape_b}, C={shape_c}",
        )

    pipeline = _pipeline_for_operation(kernel, gemm)
    stage_num = _positive_int(
        pipeline.get("num_stages"), "invalid_pipeline_stage", "pipeline num_stages"
    )
    final = _final_kernel(artifact, source_index)
    launch = final.get("launch", {})
    grid_size = _positive_int(
        launch.get("grid_size"), "invalid_launch", "final grid_size"
    )
    threads = _positive_int(
        launch.get("threads_per_block"), "invalid_launch", "final threads_per_block"
    )
    warp_size = _positive_int(
        launch.get("warp_size"), "invalid_launch", "target warp_size"
    )
    if threads % warp_size:
        _error(
            "invalid_launch", "threads_per_block is not divisible by target warp_size"
        )
    warps = threads // warp_size
    observed_warps = launch.get("warps_per_block")
    if observed_warps != warps:
        _error(
            "invalid_launch", "final warps_per_block disagrees with threads / warp_size"
        )
    warp_shape = derive_warp_shape(bm, bn, bk, warps)
    expected_grid = math.ceil(m / bm) * math.ceil(n / bn)
    if grid_size != expected_grid:
        _error(
            "compiler_model_grid_mismatch",
            f"compiler grid {grid_size} != shape/tile grid {expected_grid}",
        )
    compiler_flops = (
        kernel.get("static_workload", {}).get("per_kernel_launch", {}).get("flops")
    )
    expected_flops = 2 * m * n * k
    if compiler_flops != expected_flops:
        _error(
            "compiler_model_flops_mismatch",
            f"compiler FLOPs {compiler_flops!r} != GEMM FLOPs {expected_flops}",
        )
    dynamic_shared = _positive_int(
        launch.get("dynamic_shared_memory_bytes"),
        "invalid_resource_fact",
        "final dynamic shared memory bytes",
    )
    mma_type = _require_musa_gemm_lowering(artifact, source_index)
    arguments = {
        "op_shape": [m, n, k],
        "tb_shape": [bm, bn, bk],
        "wp_shape": list(warp_shape),
        "stage_num": stage_num,
        "mem_levels": deepcopy(S5000_FP16_GEMM_MEMORY_LEVELS),
        "row_panel": 1,
        "batch": 1,
        "mma_type": mma_type,
        "column_panel": None,
        "raster_axis": "legacy",
    }
    identity = {
        "compilation_id": artifact.get("compilation_id"),
        "kernel_id": kernel["kernel_id"],
        "pipeline_id": pipeline["pipeline_id"],
        "planned_pipeline_ids": _planned_pipeline_ids(
            artifact, source_index, pipeline["loop_ordinal"]
        ),
        "operation_ids": {
            "copy_a": copy_a["op_id"],
            "copy_b": copy_b["op_id"],
            "gemm": gemm["op_id"],
            "copy_c": output_copy["op_id"],
        },
    }
    compiler_facts = {
        "grid_size": grid_size,
        "threads_per_block": threads,
        "warp_size": warp_size,
        "warps_per_block": warps,
        "dynamic_shared_memory_bytes": dynamic_shared,
        "accumulator_buffer": accumulator_access["buffer"],
        "accumulator_dtype": accumulator_buffer["dtype"],
        "flops": compiler_flops,
        "target": compiler_target,
        "lineage_status": artifact["lineage"]["status"],
    }
    request = {
        "schema_version": MODEL_REQUEST_SCHEMA,
        "request_id": None,
        "input_source": source,
        "identity": identity,
        "model": {"name": "matmul_pipeline_wave", "arguments": arguments},
        "model_execution": {
            "numpy_random_seed": 20260814,
            "rng_scope": "adapter_lock_save_seed_restore",
        },
        "compiler_facts": compiler_facts,
        "evidence": {
            "compiler_observed": [
                "op_shape",
                "tb_shape",
                "stage_num",
                "grid_size",
                "threads_per_block",
                "warps_per_block",
                "dynamic_shared_memory_bytes",
                "accumulator_dtype",
                "operation_and_pipeline_ids",
            ],
            "semantic_derived": ["compiler_flops"],
            "adapter_derived": ["operand_roles", "mem_levels", "mma_type"],
            "model_assumption": [
                "wp_shape",
                "row_panel",
                "raster_axis",
                "mem_levels.out1.element_bytes",
                "cache_random_seed",
            ],
            "model_predicted": [],
        },
        "assumptions": [
            {
                "field": "wp_shape",
                "rule": "tilesight_near_square_warp_factorization_v1",
                "reason": "compiler layout does not expose a stable warp-tile tuple",
            },
            {
                "field": "row_panel/raster_axis",
                "rule": "tilesight_legacy_traversal_v1",
                "reason": "compiler artifact has no normalized raster schedule",
            },
            {
                "field": "mem_levels.out1.element_bytes",
                "rule": "tilesight_single_output_width_uses_external_fp16_v1",
                "reason": (
                    "the native model has one output byte width while the compiler "
                    "observes an FP32 accumulator and FP16 global output; FP16 is "
                    "used for output traffic and modeled register footprint is not "
                    "treated as a compiler fact"
                ),
            },
            {
                "field": "cache_random_seed",
                "rule": "fixed_numpy_seed_20260814_v1",
                "reason": (
                    "TileSight's reuse-distance traversal samples permutations; "
                    "the adapter saves, seeds, and restores NumPy RNG state so "
                    "identical requests produce reproducible predictions"
                ),
            },
        ],
        "unavailable": [
            "compiler_observed_register_count",
            "compiler_observed_spill_bytes",
            "mixed_precision_exact_register_footprint",
            "runtime_measurements",
        ],
        "diagnostics": [],
    }
    request["request_id"] = _stable_id(
        "model_request",
        {
            "identity": identity,
            "model": request["model"],
            "model_execution": request["model_execution"],
        },
    )
    json.dumps(request, sort_keys=True)
    return request


def invoke_tilesight_model(request: dict[str, Any], arch: Any) -> Any:
    """Call TileSight using only validated compiler facts and model scenario."""
    if request.get("schema_version") != MODEL_REQUEST_SCHEMA:
        _error("unsupported_model_request", "invalid model request schema")
    model_name = request.get("model", {}).get("name")
    supported_models = {
        "matmul_pipeline_wave",
        "elementwise_n_0_pipeline_wave",
        "elementwise_n_n_pipeline_wave",
    }
    if model_name not in supported_models:
        _error("unsupported_model", f"unsupported TileSight model {model_name!r}")
    if getattr(arch, "core", None) != "S5000":
        _error(
            "unsupported_architecture",
            f"this request profile requires S5000, got {getattr(arch, 'core', None)!r}",
        )
    arguments = deepcopy(request["model"]["arguments"])
    if model_name != "matmul_pipeline_wave":
        from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import (
            calculate_N_0_elementwise_pipeline_wave,
            calculate_N_N_elementwise_pipeline_wave,
        )

        function = (
            calculate_N_0_elementwise_pipeline_wave
            if model_name == "elementwise_n_0_pipeline_wave"
            else calculate_N_N_elementwise_pipeline_wave
        )
        return function(
            arch=arch,
            cache_condition=request["model_scenario"]["cache_condition"],
            **arguments,
        )

    from tilesight.fused_op_pipeline_wave.matmul_pipeline_wave import (
        calculate_matmul_pipeline_wave,
    )
    import numpy as np

    execution = request.get("model_execution", {})
    seed = execution.get("numpy_random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _error("invalid_model_execution", "numpy_random_seed must be non-negative")
    # TileSight's legacy cache traversal uses NumPy's process-global RNG.  The
    # lock prevents concurrent adapter calls from interleaving while restoring
    # the caller's exact RNG state after the native API returns.
    with _TILESIGHT_RNG_LOCK:
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            return calculate_matmul_pipeline_wave(arch=arch, **arguments)
        finally:
            np.random.set_state(random_state)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _build_gemm_result(
    request: dict[str, Any], model_result: Any, arch: Any
) -> dict[str, Any]:
    """Wrap one native result and enforce compiler/model consistency."""
    arguments = request["model"]["arguments"]
    compiler = request["compiler_facts"]
    m, n, k = arguments["op_shape"]
    bm, bn, _ = arguments["tb_shape"]
    wm, wn, _ = arguments["wp_shape"]
    modeled = {
        "grid_size": math.ceil(m / bm) * math.ceil(n / bn),
        "flops": 2 * m * n * k,
        "warps_per_block": (bm // wm) * (bn // wn),
        "dynamic_shared_memory_bytes": model_result.smem_footprint,
    }
    validation = {}
    diagnostics = []
    for field, model_value in modeled.items():
        compiler_value = compiler[field]
        matches = math.isclose(
            float(compiler_value), float(model_value), rel_tol=0.0, abs_tol=1e-6
        )
        validation[field] = {
            "compiler_value": compiler_value,
            "model_derived_value": _json_value(model_value),
            "matches": matches,
        }
        if not matches:
            diagnostics.append(
                {
                    "severity": "error",
                    "kind": "compiler_model_inconsistency",
                    "field": field,
                    "compiler_value": compiler_value,
                    "model_derived_value": _json_value(model_value),
                }
            )
    if diagnostics:
        raise ModelConsistencyError(diagnostics)
    detail = model_result.pipeline_detail
    prediction = {
        "value_kind": "model_predicted",
        "latency_us": model_result.total_latency * 1e6,
        "per_tile_latency_us": model_result.per_tile_latency * 1e6,
        "compute_util_pct": model_result.compute_util * 100,
        "ddr_util_pct": model_result.ddr_util * 100,
        "l2_util_pct": model_result.l2_util * 100,
        "smem_util_pct": model_result.smem_util * 100,
        "l2_hit_rate_pct": model_result.l2_hit_rate * 100,
        "tiles_per_sm": model_result.tiles_per_sm,
        "waves": model_result.waves,
        "modeled_smem_footprint_bytes": model_result.smem_footprint,
        "modeled_register_footprint": model_result.reg_footprint,
        "modeled_register_footprint_metadata": {
            "value_kind": "model_predicted",
            "compiler_validated": False,
            "unit": "TileSight internal 4-byte register units",
            "limitation": "mixed FP32 accumulator / FP16 output width is not exact",
        },
        "pipeline": {
            "prologue_us": detail.prologue_time * 1e6 if detail else None,
            "steady_per_iteration_us": detail.steady_time_per_iter * 1e6
            if detail
            else None,
            "epilogue_us": detail.epilogue_time * 1e6 if detail else None,
            "memory_per_iteration_us": detail.mem_time_per_iter * 1e6
            if detail
            else None,
            "compute_per_iteration_us": detail.compute_time_per_iter * 1e6
            if detail
            else None,
        },
        "cache_detail": _json_value(getattr(model_result, "cache_detail", None)),
    }
    result = {
        "schema_version": MODEL_RESULT_SCHEMA,
        "request_id": request["request_id"],
        "identity": deepcopy(request["identity"]),
        "architecture": str(getattr(arch, "core", type(arch).__name__)),
        "prediction": _json_value(prediction),
        "compiler_validation": {
            "status": "all_consistent",
            "checks": validation,
        },
        "evidence": {
            **deepcopy(request["evidence"]),
            "model_predicted": list(prediction),
        },
        "assumptions": deepcopy(request["assumptions"]),
        "unavailable": deepcopy(request["unavailable"]),
        "diagnostics": deepcopy(request["diagnostics"]),
    }
    json.dumps(result, sort_keys=True)
    return result


def _build_elementwise_result(
    request: dict[str, Any], model_result: Any, arch: Any
) -> dict[str, Any]:
    arguments = request["model"]["arguments"]
    compiler = request["compiler_facts"]
    n = arguments["op_shape"][0]
    elements_per_block = arguments["tb_shape"][0]
    threads = arguments["dim_threads"][0]
    modeled = {
        "grid_size": math.ceil(n / elements_per_block),
        "threads_per_block": threads,
        "warps_per_block": max(threads // compiler["warp_size"], 1),
        "elements_per_block": elements_per_block,
    }
    validation = {}
    diagnostics = []
    for field, model_value in modeled.items():
        compiler_value = compiler[field]
        matches = compiler_value == model_value
        validation[field] = {
            "compiler_value": compiler_value,
            "model_derived_value": _json_value(model_value),
            "matches": matches,
        }
        if not matches:
            diagnostics.append(
                {
                    "severity": "error",
                    "kind": "compiler_model_inconsistency",
                    "field": field,
                    "compiler_value": compiler_value,
                    "model_derived_value": _json_value(model_value),
                }
            )
    if diagnostics:
        raise ModelConsistencyError(diagnostics)
    detail = model_result.pipeline_detail
    prediction = {
        "value_kind": "model_predicted",
        "latency_us": model_result.total_latency * 1e6,
        "per_tile_latency_us": model_result.per_tile_latency * 1e6,
        "compute_util_pct": model_result.compute_util * 100,
        "ddr_util_pct": model_result.ddr_util * 100,
        "l2_util_pct": model_result.l2_util * 100,
        "smem_util_pct": model_result.smem_util * 100,
        "l2_hit_rate_pct": model_result.l2_hit_rate * 100,
        "tiles_per_sm": model_result.tiles_per_sm,
        "waves": model_result.waves,
        "modeled_smem_footprint_bytes": model_result.smem_footprint,
        "modeled_register_footprint": model_result.reg_footprint,
        "pipeline": {
            "prologue_us": detail.prologue_time * 1e6 if detail else None,
            "steady_per_iteration_us": detail.steady_time_per_iter * 1e6
            if detail
            else None,
            "epilogue_us": detail.epilogue_time * 1e6 if detail else None,
            "memory_per_iteration_us": detail.mem_time_per_iter * 1e6
            if detail
            else None,
            "compute_per_iteration_us": detail.compute_time_per_iter * 1e6
            if detail
            else None,
        },
        "cache_detail": _json_value(getattr(model_result, "cache_detail", None)),
    }
    result = {
        "schema_version": MODEL_RESULT_SCHEMA,
        "request_id": request["request_id"],
        "identity": deepcopy(request["identity"]),
        "architecture": str(getattr(arch, "core", type(arch).__name__)),
        "model_scenario": deepcopy(request["model_scenario"]),
        "prediction": _json_value(prediction),
        "compiler_validation": {
            "status": "all_consistent",
            "checks": validation,
        },
        "evidence": {
            **deepcopy(request["evidence"]),
            "model_predicted": list(prediction),
        },
        "assumptions": deepcopy(request["assumptions"]),
        "unavailable": deepcopy(request["unavailable"]),
        "diagnostics": deepcopy(request["diagnostics"]),
    }
    json.dumps(result, sort_keys=True)
    return result


def build_tilesight_result(
    request: dict[str, Any], model_result: Any, arch: Any
) -> dict[str, Any]:
    """Wrap one native model result without changing TileSight calculations."""
    if request.get("schema_version") != MODEL_REQUEST_SCHEMA:
        _error("unsupported_model_request", "invalid model request schema")
    model_name = request.get("model", {}).get("name")
    if model_name == "matmul_pipeline_wave":
        return _build_gemm_result(request, model_result, arch)
    if model_name in {"elementwise_n_0_pipeline_wave", "elementwise_n_n_pipeline_wave"}:
        return _build_elementwise_result(request, model_result, arch)
    _error("unsupported_model", f"unsupported TileSight model {model_name!r}")


def run_tilesight_request(request: dict[str, Any], arch: Any) -> dict[str, Any]:
    """Run and wrap one validated request; no kernel or profiler is invoked."""
    return build_tilesight_result(request, invoke_tilesight_model(request, arch), arch)


def analyze_compiler_info(
    compiler_info: dict[str, Any] | str | Path,
    arch: Any,
    *,
    kernel_id: str | None = None,
    cache_condition: str | None = None,
) -> dict[str, Any]:
    """Convenience entry combining strict preparation and native modeling."""
    return run_tilesight_request(
        prepare_tilesight_request(
            compiler_info,
            kernel_id=kernel_id,
            cache_condition=cache_condition,
        ),
        arch,
    )


__all__ = [
    "COMPILER_INFO_SCHEMA",
    "MODEL_REQUEST_SCHEMA",
    "MODEL_RESULT_SCHEMA",
    "S5000_FP16_GEMM_MEMORY_LEVELS",
    "S5000_FP32_ADD_MEMORY_LEVELS",
    "S5000_FP32_RELU_MEMORY_LEVELS",
    "CompilerInfoAdapterError",
    "ModelConsistencyError",
    "analyze_compiler_info",
    "build_tilesight_result",
    "derive_warp_shape",
    "invoke_tilesight_model",
    "prepare_tilesight_request",
    "run_tilesight_request",
]
