"""Structured TileLang-MUSA compiler information collected via public APIs.

This module deliberately stops at compiler facts.  It does not import
TileSight, invoke a performance model, read profiler counters, or launch a
device kernel.  Collection uses only TileLang's existing ``instruments`` /
``PassInstrument`` hook and the public ``CompiledArtifact`` fields.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from ...collectors.compiler.lineage import structural_candidates
from ...collectors.compiler.metadata import extract_metadata
from ...core.evidence import COMPILER_OBSERVED
from ...core.ids import sha256_text, stable_id
from .pass_capture import make_snapshot_instrument


SCHEMA_VERSION = "musa-perf-collector/tilelang-compiler-info-v1"
_LAYOUT_KEYS = {
    "layout_map",
    "k_major_map",
    "sqmma_map",
    "sqmma_inst_split_map",
    "parallel_loop_layout",
    "parallel_loop_predicate",
}
_SYNC_TOKENS = ("barrier", "wait", "fence", "storage_sync", "commit_group")


def _integer(value: Any) -> int | None:
    try:
        return int(value.value)
    except (AttributeError, TypeError, ValueError):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None


def _object_name(value: Any) -> str:
    for attribute in ("name", "name_hint"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return str(candidate)
    return str(value)


def _json_value(value: Any) -> Any:
    """Convert compiler objects to deterministic, non-lossy-enough JSON facts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    parsed = _integer(value)
    if parsed is not None:
        return parsed
    if hasattr(value, "items"):
        try:
            return {
                _object_name(key): _json_value(item)
                for key, item in value.items()
            }
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)) or type(value).__name__ in {"Array", "ShapeTuple"}:
        try:
            return [_json_value(item) for item in value]
        except TypeError:
            pass
    return str(value)


def _annotations(node: Any) -> dict[str, Any]:
    raw = getattr(node, "annotations", None)
    if raw is None:
        return {}
    try:
        return {str(key): _json_value(value) for key, value in raw.items()}
    except AttributeError:
        return {}


def _op_name(call: Any) -> str:
    op = getattr(call, "op", None)
    name = getattr(op, "name", None)
    return str(name) if name is not None else str(op)


def _post_order(root: Any) -> list[Any]:
    import tvm.tirx as tirx

    nodes: list[Any] = []
    tirx.stmt_functor.post_order_visit(root, nodes.append)
    return nodes


def _prim_funcs(value: Any) -> list[tuple[str, Any]]:
    import tvm.tirx as tirx

    if isinstance(value, tirx.PrimFunc):
        symbol = value.attrs.get("global_symbol", "main") if value.attrs else "main"
        return [(str(symbol), value)]
    functions = getattr(value, "functions", None)
    if functions is None:
        return []
    result: list[tuple[str, Any]] = []
    for global_var, function in functions.items():
        if isinstance(function, tirx.PrimFunc):
            result.append((str(getattr(global_var, "name_hint", global_var)), function))
    return result


def _product(values: dict[str, int | str]) -> int | str:
    numeric = 1
    symbolic: list[str] = []
    for value in values.values():
        if isinstance(value, int):
            numeric *= value
        else:
            symbolic.append(str(value))
    if not symbolic:
        return numeric
    terms = ([str(numeric)] if numeric != 1 else []) + symbolic
    return " * ".join(terms) if terms else "1"


def _target_warp_size(attrs: Any) -> int | None:
    if not attrs:
        return None
    target = attrs.get("target")
    if target is None:
        return None
    target_attrs = getattr(target, "attrs", None)
    if target_attrs is None:
        return None
    try:
        return _integer(target_attrs.get("thread_warp_size"))
    except AttributeError:
        return None


def _launch_information(value: Any) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    for fallback_name, function in _prim_funcs(value):
        axes: dict[str, int | str] = {}
        for node in _post_order(function.body):
            if type(node).__name__ != "AttrStmt" or str(node.attr_key) != "thread_extent":
                continue
            axis = getattr(getattr(node, "node", None), "thread_tag", None)
            axis = getattr(axis, "value", axis)
            if axis is None:
                continue
            extent = _integer(node.value)
            axes[str(axis)] = extent if extent is not None else str(node.value)
        grid_axes = {key: value for key, value in axes.items() if key.startswith("blockIdx")}
        thread_axes = {key: value for key, value in axes.items() if key.startswith("threadIdx")}
        attrs = {
            str(key): _json_value(value)
            for key, value in (function.attrs.items() if function.attrs else [])
        }
        warp_size = _target_warp_size(function.attrs)
        thread_count = _product(thread_axes)
        warps = (
            thread_count // warp_size
            if isinstance(thread_count, int) and isinstance(warp_size, int)
            and warp_size > 0 and thread_count % warp_size == 0
            else None
        )
        kernels.append({
            "name": str(function.attrs.get("global_symbol", fallback_name)) if function.attrs else fallback_name,
            "grid_axes": grid_axes,
            "thread_axes": thread_axes,
            "grid_size": _product(grid_axes),
            "threads_per_block": thread_count,
            "warp_size": warp_size,
            "warps_per_block": warps,
            "dynamic_shared_memory_bytes": _integer(function.attrs.get("dyn_shared_memory_buf")) if function.attrs else None,
            "kernel_launch_params": _json_value(function.attrs.get("tirx.kernel_launch_params")) if function.attrs else None,
            "attrs": attrs,
            "source_snapshot": "S60_FINAL_DEVICE_IR",
            "evidence_level": COMPILER_OBSERVED,
        })
    return kernels


def _pipeline_information(value: Any) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    for name, function in _prim_funcs(value):
        calls: list[dict[str, Any]] = []
        loops: list[dict[str, Any]] = []
        for node in _post_order(function.body):
            kind = type(node).__name__
            if kind == "Call":
                annotations = _annotations(node)
                ordinal = _integer(annotations.get("tl_provenance_ordinal"))
                if ordinal is None and not any(
                    key in annotations for key in ("is_tma_copy", "emit_arrive", "tl.pipeline_mbar_phase_expr")
                ):
                    continue
                calls.append({
                    "op_name": _op_name(node),
                    "source_ordinal": ordinal,
                    "annotations": annotations,
                    "raw_ir": str(node),
                    "evidence_level": COMPILER_OBSERVED,
                })
            elif kind == "For":
                annotations = _annotations(node)
                if annotations:
                    loops.append({
                        "var": str(node.loop_var),
                        "min": str(node.min),
                        "extent": str(node.extent),
                        "extent_value": _integer(node.extent),
                        "kind": str(node.kind),
                        "thread_binding": str(node.thread_binding) if node.thread_binding else None,
                        "annotations": annotations,
                        "evidence_level": COMPILER_OBSERVED,
                    })
        kernels.append({"name": name, "annotated_calls": calls, "annotated_loops": loops})
    return {
        "source_snapshot": "S10_PIPELINE_PLANNED",
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _layout_information(value: Any) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    for name, function in _prim_funcs(value):
        blocks: list[dict[str, Any]] = []
        loops: list[dict[str, Any]] = []
        for node in _post_order(function.body):
            annotations = _annotations(node)
            selected = {key: item for key, item in annotations.items() if key in _LAYOUT_KEYS}
            if not selected:
                continue
            if type(node).__name__ == "SBlock":
                blocks.append({
                    "block_name": str(getattr(node, "name_hint", "")),
                    "annotations": selected,
                    "evidence_level": COMPILER_OBSERVED,
                })
            elif type(node).__name__ == "For":
                loops.append({
                    "var": str(node.loop_var),
                    "extent": str(node.extent),
                    "annotations": selected,
                    "evidence_level": COMPILER_OBSERVED,
                })
        kernels.append({"name": name, "blocks": blocks, "loops": loops})
    return {
        "source_snapshot": "S30_LAYOUT_INFERRED",
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _is_sync_call(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SYNC_TOKENS)


def _provenance_regions(value: Any, source_snapshot: str) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    for name, function in _prim_funcs(value):
        regions: list[dict[str, Any]] = []
        all_calls = [_op_name(node) for node in _post_order(function.body) if type(node).__name__ == "Call"]
        for node in _post_order(function.body):
            if type(node).__name__ != "AttrStmt" or str(node.attr_key) != "tl.provenance_ordinal":
                continue
            ordinal = _integer(node.value)
            calls = [_op_name(child) for child in _post_order(node.body) if type(child).__name__ == "Call"]
            regions.append({
                "source_ordinal": ordinal,
                "call_count": len(calls),
                "call_kinds": dict(sorted(Counter(calls).items())),
                "synchronization_calls": dict(sorted(Counter(item for item in calls if _is_sync_call(item)).items())),
                "body_sha256": sha256_text(str(node.body)),
                "evidence_level": COMPILER_OBSERVED,
            })
        kernels.append({
            "name": name,
            "provenance_regions": regions,
            "all_call_kinds": dict(sorted(Counter(all_calls).items())),
            "synchronization_calls": dict(sorted(Counter(item for item in all_calls if _is_sync_call(item)).items())),
        })
    return {
        "source_snapshot": source_snapshot,
        "kernels": kernels,
        "evidence_level": COMPILER_OBSERVED,
    }


def _git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


@dataclass
class TileLangCompilerInfoCollector:
    """Collect compiler stages through TileLang's existing instrument API."""

    tvm: Any

    def __post_init__(self) -> None:
        self.instrument = make_snapshot_instrument(self.tvm)

    @classmethod
    def create(cls) -> "TileLangCompilerInfoCollector":
        """Create a collector after loading TileLang's bundled TVM lazily."""
        from tilelang import tvm

        return cls(tvm)

    def finalize(
        self,
        *,
        source_prim_func: Any,
        compiled_artifact: Any,
        target: Any,
        pass_config: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        tilelang_version: str | None = None,
        tilelang_root: Path | None = None,
        device_compile: bool | None = None,
    ) -> dict[str, Any]:
        """Return JSON-serializable compiler facts after compilation/lowering."""
        source_metadata = extract_metadata(source_prim_func)
        snapshots = {"S00_INPUT_TILE_IR": source_prim_func.script(), **self.instrument.snapshots}
        modules = dict(self.instrument.stage_modules)
        final_device_ir = compiled_artifact.device_mod.script()
        snapshots["S60_FINAL_DEVICE_IR"] = final_device_ir
        modules["S60_FINAL_DEVICE_IR"] = compiled_artifact.device_mod
        kernel_source = str(compiled_artifact.kernel_source)
        stage_manifest = {
            stage: {
                "tir_sha256": sha256_text(text),
                "character_count": len(text),
                "captured": True,
            }
            for stage, text in sorted(snapshots.items())
        }
        required = {
            "S00_INPUT_TILE_IR", "S10_PIPELINE_PLANNED", "S20_PIPELINE_INJECTED",
            "S30_LAYOUT_INFERRED", "S40_TILE_OP_LOWERED", "S60_FINAL_DEVICE_IR",
        }
        missing = sorted(required - set(snapshots))
        compilation_id = stable_id("compile", [
            sha256_text(snapshots["S00_INPUT_TILE_IR"]), str(target), pass_config or {},
            compile_flags or [], sha256_text(kernel_source),
        ])
        lineage = structural_candidates(source_metadata["tile_ops"], snapshots)
        return {
            "schema_version": SCHEMA_VERSION,
            "compilation_id": compilation_id,
            "collection_scope": "compiler_only_no_device_execution_or_performance_model",
            "identity": {
                "source_ir_sha256": sha256_text(snapshots["S00_INPUT_TILE_IR"]),
                "final_device_ir_sha256": sha256_text(final_device_ir),
                "kernel_source_sha256": sha256_text(kernel_source),
                "target": str(target),
                "tilelang_version": tilelang_version,
                "tilelang_revision": _git_revision(tilelang_root),
                "pass_config": pass_config or {},
                "compile_flags": compile_flags or [],
                "device_compile": device_compile,
                "evidence_level": COMPILER_OBSERVED,
            },
            "stage_manifest": stage_manifest,
            "pass_sequence": list(self.instrument.pass_sequence),
            "source": source_metadata,
            "pipeline_planning": _pipeline_information(modules["S10_PIPELINE_PLANNED"]) if "S10_PIPELINE_PLANNED" in modules else None,
            "layout_inference": _layout_information(modules["S30_LAYOUT_INFERRED"]) if "S30_LAYOUT_INFERRED" in modules else None,
            "tileop_lowering": _provenance_regions(modules["S40_TILE_OP_LOWERED"], "S40_TILE_OP_LOWERED") if "S40_TILE_OP_LOWERED" in modules else None,
            "final_implementation": {
                "launch": _launch_information(compiled_artifact.device_mod),
                "lowered_operations": _provenance_regions(compiled_artifact.device_mod, "S60_FINAL_DEVICE_IR"),
                "kernel_source_sha256": sha256_text(kernel_source),
                "kernel_source_character_count": len(kernel_source),
                "evidence_level": COMPILER_OBSERVED,
            },
            "lineage": lineage,
            "coverage": {
                "required_stages": sorted(required),
                "captured_stages": sorted(snapshots),
                "missing_required_stages": missing,
                "complete_required_stage_capture": not missing,
                "source_tileop_lossless_export": source_metadata["tileop_coverage"]["lossless_export"],
                "compiler_exact_lineage": lineage.get("status") == "compiler_exact_lineage",
            },
        }


def collect_compiler_information(
    prim_func: Any,
    *,
    arch: str = "mp_31",
    pass_config: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
    tilelang_root: Path | None = None,
) -> dict[str, Any]:
    """Lower without device compile/launch and return structured compiler facts."""
    import tilelang
    from tilelang import tvm

    target = tvm.target.Target({"kind": "musa", "arch": arch})
    collector = TileLangCompilerInfoCollector(tvm)
    configs = dict(pass_config or {})
    if compile_flags:
        from tilelang.transform import PassConfigKey

        configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = list(compile_flags)
    with target, tvm.transform.PassContext(
        opt_level=3, config=configs, instruments=[collector.instrument]
    ):
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return collector.finalize(
        source_prim_func=prim_func,
        compiled_artifact=artifact,
        target=target,
        pass_config=pass_config,
        compile_flags=compile_flags,
        tilelang_version=getattr(tilelang, "__version__", "unknown"),
        tilelang_root=tilelang_root,
        device_compile=False,
    )


__all__ = [
    "SCHEMA_VERSION",
    "TileLangCompilerInfoCollector",
    "collect_compiler_information",
]
