"""Compiler provenance helpers shared by backend pass pipelines."""

from __future__ import annotations

import tilelang
from tvm import IRModule


def annotate_source_tileops(mod: IRModule) -> IRModule:
    """Attach stable source TileOp ordinals before pipeline expansion."""
    return tilelang.transform.ProvenanceAnnotation()(mod)


def inject_requested_trace(mod: IRModule) -> IRModule:
    """Inject an explicitly requested, bounded provenance trace after lowering."""
    trace_cfg = tilelang.transform.get_pass_context().config
    if not trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_ENABLE, False):
        return mod

    buffer_name = trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_BUFFER)
    ordinals = trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_ORDINALS)
    if ordinals is None:
        ordinal = trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_ORDINAL)
        ordinals = [] if ordinal is None else [ordinal]
    ordinals = [int(ordinal) for ordinal in ordinals]
    if not ordinals or len(set(ordinals)) != len(ordinals) or buffer_name is None:
        raise ValueError("provenance trace requires ordinal and trace buffer configuration")

    trace_loop_ordinal = int(
        trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_LOOP_ORDINAL, -1)
    )
    trace_iteration = int(
        trace_cfg.get(tilelang.PassConfigKey.TL_PROVENANCE_TRACE_ITERATION, -1)
    )
    if (trace_loop_ordinal >= 0) != (trace_iteration >= 0):
        raise ValueError(
            "provenance iteration trace requires both source loop ordinal "
            "and non-negative iteration"
        )

    trace_blocks = [
        int(value)
        for value in trace_cfg.get(
            tilelang.PassConfigKey.TL_PROVENANCE_TRACE_BLOCKS, [0, 0, 0]
        )
    ]
    if not trace_blocks or len(trace_blocks) % 3:
        raise ValueError(
            "provenance trace blocks must be flattened non-empty x/y/z triples"
        )
    if any(value < 0 for value in trace_blocks):
        raise ValueError("provenance trace block coordinates must be non-negative")
    coordinates = [
        tuple(trace_blocks[index : index + 3])
        for index in range(0, len(trace_blocks), 3)
    ]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("provenance trace block coordinates must be unique")

    trace_semantics = str(
        trace_cfg.get(
            tilelang.PassConfigKey.TL_PROVENANCE_TRACE_SEMANTICS, "single"
        )
    )
    if trace_semantics not in {"single", "source_envelope", "lowered_instance"}:
        raise ValueError(
            "provenance trace semantics must be single, source_envelope, "
            "or lowered_instance"
        )

    instance_counts = [
        int(value)
        for value in trace_cfg.get(
            tilelang.PassConfigKey.TL_PROVENANCE_TRACE_INSTANCE_COUNTS,
            [1] * len(ordinals),
        )
    ]
    if len(instance_counts) != len(ordinals) or any(
        value <= 0 for value in instance_counts
    ):
        raise ValueError(
            "provenance trace instance counts must align with selected ordinals"
        )

    trace_slot_offset = 0
    trace_block_count = len(coordinates)
    for trace_index, ordinal in enumerate(ordinals):
        mod = tilelang.transform.ProvenanceTraceInjection(
            str(buffer_name),
            ordinal,
            trace_slot_offset,
            trace_loop_ordinal,
            trace_iteration,
            trace_blocks,
            trace_semantics,
        )(mod)
        instance_factor = (
            instance_counts[trace_index]
            if trace_semantics == "lowered_instance"
            else 1
        )
        trace_slot_offset += instance_factor * trace_block_count * 2
    return mod
