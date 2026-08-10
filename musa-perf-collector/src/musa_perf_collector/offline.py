"""Offline analysis orchestration and artifact-bundle export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.tilelang.offline_session import lower_offline
from .collectors.compiler.analysis import build_dag, reuse_distance, upper_bound_formula, workload_demand, workload_domains
from .collectors.compiler.metadata import extract_metadata
from .collectors.compiler.lineage import structural_candidates
from .core.ids import sha256_text, stable_id
from .core.io import write_json, write_jsonl, write_text
from .core.validation import validate_offline_static
from .exporters.report import render_markdown


DEFAULT_CAPACITIES = [32 * 1024, 128 * 1024, 512 * 1024, 2 * 1024 * 1024]


def analyze(
    prim_func: Any,
    *,
    config: dict[str, Any],
    output_root: Path,
    arch: str = "mp_31",
    pass_config: dict[str, Any] | None = None,
    tilelang_root: Path | None = None,
) -> Path:
    metadata = extract_metadata(prim_func)
    artifact = lower_offline(
        prim_func,
        arch=arch,
        pass_config=pass_config,
        tilelang_root=tilelang_root,
    )
    # Deliberately deterministic: the same input, configuration, compiler
    # revision, and generated source resolve to the same artifact directory.
    analysis_id = stable_id("analysis", [metadata["kernel"]["name"], config, artifact.manifest["kernel_source_sha256"]])
    output_dir = output_root / analysis_id
    domains = workload_domains(metadata["tile_ops"])
    demand = workload_demand(metadata["tile_ops"])
    dag = build_dag(metadata["tile_ops"])
    reuse = reuse_distance(metadata["tile_ops"], config.get("capacity_sweep_bytes", DEFAULT_CAPACITIES))
    diagnostics = validate_offline_static(
        snapshots=artifact.snapshots,
        kernel_source=artifact.kernel_source,
        coverage=metadata["tileop_coverage"],
        tile_ops=metadata["tile_ops"],
        dag=dag,
        domains=domains,
    )
    manifest = {
        "schema_version": "musa-perf-collector/offline-static-v1",
        "analysis_id": analysis_id,
        "kernel_id": stable_id("kernel", [metadata["kernel"]["name"], prim_func.script()]),
        "input_config": config,
        "input_ir_sha256": sha256_text(prim_func.script()),
        "pass_sequence": artifact.pass_sequence,
        "snapshot_stages": sorted(artifact.snapshots),
        "analysis_status": diagnostics["status"],
        **artifact.manifest,
    }
    for stage, text in artifact.snapshots.items():
        write_text(output_dir / "ir" / f"{stage}.tir", text)
    write_text(output_dir / "generated" / "kernel.mu", artifact.kernel_source)
    write_text(output_dir / "ir" / "host.tir", artifact.host_ir)
    write_text(output_dir / "ir" / "device.tir", artifact.device_ir)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "metadata" / "kernel.json", metadata["kernel"])
    write_jsonl(output_dir / "metadata" / "buffers.jsonl", metadata["buffers"])
    write_jsonl(output_dir / "metadata" / "tile_ops.jsonl", metadata["tile_ops"])
    write_json(output_dir / "metadata" / "tileop_coverage.json", metadata["tileop_coverage"])
    write_json(output_dir / "metadata" / "execution_tree.json", [
        {"op_id": op["op_id"], "statement_path": op["statement_path"], "loop_context": op["loop_context"], "condition_context": op["condition_context"], "execution": op["execution"]}
        for op in metadata["tile_ops"]
    ])
    write_jsonl(output_dir / "metadata" / "diagnostics.jsonl", diagnostics["diagnostics"])
    write_json(output_dir / "metadata" / "pipeline.json", metadata["pipeline"])
    write_json(output_dir / "metadata" / "lineage.json", structural_candidates(metadata["tile_ops"], artifact.snapshots))
    write_json(output_dir / "metadata" / "loops.json", metadata["loops"])
    write_json(output_dir / "metadata" / "dag.json", dag)
    write_json(output_dir / "metrics" / "workload_demand.json", demand)
    write_jsonl(output_dir / "metrics" / "workload_per_invocation.jsonl", domains["per_invocation"])
    write_json(output_dir / "metrics" / "workload_per_block.json", domains["per_block"])
    write_json(output_dir / "metrics" / "workload_per_kernel.json", domains["per_kernel_launch"])
    write_json(output_dir / "metrics" / "reuse_distance.json", reuse)
    write_json(output_dir / "metrics" / "upper_bound_formula.json", upper_bound_formula(domains["per_kernel_launch"]))
    write_json(output_dir / "generated" / "compile_config.json", artifact.manifest)
    markdown = render_markdown(manifest, metadata, domains, dag, diagnostics)
    write_text(output_dir / "report" / "report.md", markdown)
    from .exporters.report import render_html
    write_text(output_dir / "report" / "report.html", render_html(markdown))
    return output_dir
