"""CPU-only TileLang lowering session and compiler-artifact capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from ...core.evidence import COMPILER_OBSERVED
from ...core.ids import sha256_text
from .pass_capture import make_snapshot_instrument


@dataclass
class OfflineArtifact:
    snapshots: dict[str, str]
    pass_sequence: list[str]
    kernel_source: str
    host_ir: str
    device_ir: str
    manifest: dict[str, Any]


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


def lower_offline(
    prim_func: Any,
    *,
    arch: str,
    pass_config: dict[str, Any] | None,
    tilelang_root: Path | None,
) -> OfflineArtifact:
    """Lower to MUSA C without compile, allocation, launch, or device execution."""
    import tilelang
    from tilelang import tvm

    target = tvm.target.Target({"kind": "musa", "arch": arch})
    instrument = make_snapshot_instrument(tvm)
    input_ir = prim_func.script()
    with target, tvm.transform.PassContext(
        opt_level=3, config=pass_config or {}, instruments=[instrument]
    ):
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    snapshots = {"S00_INPUT_TILE_IR": input_ir, **instrument.snapshots}
    host_ir = artifact.host_mod.script()
    device_ir = artifact.device_mod.script()
    # The lowering pipeline continues after LowerDeviceKernelLaunch.  The
    # artifact's device module is therefore the only defensible final snapshot.
    snapshots["S60_FINAL_DEVICE_IR"] = device_ir
    manifest = {
        "target": str(target),
        "arch": arch,
        "pass_config": pass_config or {},
        "tilelang_version": getattr(tilelang, "__version__", "unknown"),
        "tilelang_root": str(tilelang_root) if tilelang_root else None,
        "tilelang_revision": _git_revision(tilelang_root),
        "device_execution": False,
        "device_compile": False,
        "evidence_level": COMPILER_OBSERVED,
        "kernel_source_sha256": sha256_text(artifact.kernel_source),
        "host_ir_sha256": sha256_text(host_ir),
        "device_ir_sha256": sha256_text(device_ir),
    }
    return OfflineArtifact(
        snapshots=snapshots,
        pass_sequence=instrument.pass_sequence,
        kernel_source=artifact.kernel_source,
        host_ir=host_ir,
        device_ir=device_ir,
        manifest=manifest,
    )
