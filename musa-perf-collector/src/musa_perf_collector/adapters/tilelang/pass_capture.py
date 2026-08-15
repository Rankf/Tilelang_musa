"""PassInstrument based IR capture without changes to TileLang-MUSA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGE_MATCHERS = {
    "S10_PIPELINE_PLANNED": "PipelinePlanning",
    "S20_PIPELINE_INJECTED": "InjectSoftwarePipeline",
    "S30_LAYOUT_INFERRED": "LayoutInference",
    "S40_TILE_OP_LOWERED": "LowerTileOp",
    "S50_HOST_DEVICE_SPLIT": "SplitHostDevice",
    "S60_FINAL_DEVICE_IR": "LowerDeviceKernelLaunch",
}


def make_snapshot_instrument(tvm: Any) -> Any:
    """Create a registered TVM pass instrument lazily after TileLang imports."""

    @tvm.instrument.pass_instrument
    class SnapshotInstrument:
        def __init__(self):
            self.snapshots: dict[str, str] = {}
            # Keep the immutable IRModule snapshots in memory as well as their
            # text form.  Project-side collectors can then extract structured
            # compiler facts without parsing pretty-printed TIR.  This uses
            # TVM's existing PassInstrument API and does not require a
            # TileLang-MUSA extension.
            self.stage_modules: dict[str, Any] = {}
            self.pass_sequence: list[str] = []

        def run_after_pass(self, mod, pass_info):
            name = str(pass_info.name)
            self.pass_sequence.append(name)
            for stage, matcher in STAGE_MATCHERS.items():
                if matcher in name:
                    self.snapshots[stage] = mod.script()
                    self.stage_modules[stage] = mod

    return SnapshotInstrument()
