"""Fail-closed validation shared by every reviewed deep-trace adapter."""

from __future__ import annotations

from typing import Any


def validate(case: Any, trace: list[dict[str, Any]], deep_config: dict[str, Any]) -> dict[str, Any]:
    """Validate bounded trace completeness before it can become ``complete``.

    This validates the adapter contract rather than attempting unsafe heuristic
    lineage recovery from generated MUSA C.  Future automatic instrumentation
    must provide the same evidence fields.
    """
    if not case.instrumented:
        raise RuntimeError("deep trace case is not marked instrumented")
    metadata = case.trace_metadata or {}
    expected = int(metadata.get("timestamps_per_sample", 0))
    if expected < 2 or expected % 2 != 0 or len(trace) != expected:
        raise RuntimeError(
            f"trace record count {len(trace)} does not match the paired-boundary trace contract ({expected})"
        )
    lineage = case.lineage or []
    slots = [item.get("trace_slot") for item in lineage]
    if sorted(slots) != list(range(expected)):
        raise RuntimeError("trace lineage slots must exactly cover the trace records")
    if bool(deep_config.get("require_exact_lineage", True)):
        non_exact = [item for item in lineage if item.get("lineage") not in {"manual_exact_workload_contract", "compiler_exact_lineage"}]
        if non_exact:
            raise RuntimeError("deep trace requires exact lineage; heuristic or missing lineage was supplied")
    ordered_lineage = sorted(lineage, key=lambda item: int(item.get("trace_slot", -1)))
    trace_by_slot = {int(item.get("trace_slot", -1)): item for item in trace}
    if set(trace_by_slot) != set(range(expected)):
        raise RuntimeError("trace records must cover every fixed trace slot exactly once")
    for pair_start in range(0, expected, 2):
        for offset, expected_boundary in enumerate(("begin", "end")):
            slot = pair_start + offset
            lineage_item = ordered_lineage[slot]
            trace_item = trace_by_slot[slot]
            if lineage_item.get("boundary") != expected_boundary:
                raise RuntimeError(f"trace lineage slot {slot} must be the {expected_boundary} boundary")
            if trace_item.get("boundary") != expected_boundary:
                raise RuntimeError(f"trace record slot {slot} must be the {expected_boundary} boundary")
            if trace_item.get("source_ordinal") is not None and lineage_item.get("source_ordinal") is not None:
                if int(trace_item["source_ordinal"]) != int(lineage_item["source_ordinal"]):
                    raise RuntimeError("trace record source ordinal does not match fixed-slot lineage")
            lineage_block = lineage_item.get("block")
            trace_block = trace_item.get("block")
            if lineage_block is not None or trace_block is not None:
                if tuple(lineage_block or ()) != tuple(trace_block or ()):
                    raise RuntimeError("trace record block does not match fixed-slot lineage")
    compiler_lineage = [item for item in ordered_lineage if item.get("lineage") == "compiler_exact_lineage"]
    if compiler_lineage:
        ordinal_sequence = [item.get("source_ordinal") for item in compiler_lineage]
        if not all(isinstance(item, int) and item >= 0 for item in ordinal_sequence):
            raise RuntimeError("compiler exact trace requires non-negative source provenance ordinals")
        ordinal_pairs = [ordinal_sequence[index:index + 2] for index in range(0, expected, 2)]
        if any(len(pair) != 2 or pair[0] != pair[1] for pair in ordinal_pairs):
            raise RuntimeError("each compiler exact TileOp must own exactly one begin/end trace pair")
        selected_ordinals = [pair[0] for pair in ordinal_pairs]
        selected_blocks = metadata.get("selected_blocks")
        if selected_blocks is None:
            if selected_ordinals != sorted(set(selected_ordinals)):
                raise RuntimeError("compiler exact TileOp ordinals must be unique and in source order")
        else:
            normalized_blocks = [tuple(int(value) for value in block) for block in selected_blocks]
            if not normalized_blocks or len(set(normalized_blocks)) != len(normalized_blocks):
                raise RuntimeError("multi-block trace metadata must contain unique selected block coordinates")
            if expected % (2 * len(normalized_blocks)):
                raise RuntimeError("multi-block trace record count is incompatible with selected blocks")
            lineage_by_slot = {int(item.get("trace_slot", -1)): item for item in ordered_lineage}
            observed_blocks: dict[tuple[int, int, int], list[int]] = {block: [] for block in normalized_blocks}
            for pair_start in range(0, expected, 2):
                pair_lineage = [lineage_by_slot[pair_start], lineage_by_slot[pair_start + 1]]
                pair_block = tuple(int(value) for value in pair_lineage[0].get("block", []))
                if len(pair_block) != 3 or pair_block not in observed_blocks:
                    raise RuntimeError("multi-block trace lineage has an unselected or invalid block coordinate")
                if tuple(int(value) for value in pair_lineage[1].get("block", [])) != pair_block:
                    raise RuntimeError("one TileOp boundary pair must belong to one block")
                pair_trace = [trace_by_slot[pair_start], trace_by_slot[pair_start + 1]]
                if any(tuple(int(value) for value in item.get("block", [])) != pair_block for item in pair_trace):
                    raise RuntimeError("multi-block trace record block does not match fixed-slot lineage")
                if int(pair_trace[1]["clock_cycles"]) < int(pair_trace[0]["clock_cycles"]):
                    raise RuntimeError("trace end clock precedes begin clock within one block")
                observed_blocks[pair_block].append(int(pair_lineage[0]["source_ordinal"]))
            expected_tileops_per_block = expected // (2 * len(normalized_blocks))
            if any(len(ordinals) != expected_tileops_per_block for ordinals in observed_blocks.values()):
                raise RuntimeError("one or more selected blocks did not produce all fixed trace pairs")
            if any(ordinals != sorted(set(ordinals)) for ordinals in observed_blocks.values()):
                raise RuntimeError("each selected block must preserve unique source TileOp order")
            selected_ordinals = list(dict.fromkeys(selected_ordinals))
        if any(item.get("compiler_provenance_annotation") != "tl.provenance_ordinal" for item in compiler_lineage):
            raise RuntimeError("compiler exact trace is missing the compiler provenance annotation evidence")
    else:
        selected_ordinals = []
    selected_families = set(deep_config.get("tileop_families", []))
    if selected_families and not {item.get("canonical_family") for item in lineage}.issubset(selected_families):
        raise RuntimeError("trace lineage contains a TileOp family outside the selected families")
    if metadata.get("selected_blocks") is None:
        clocks = [int(trace_by_slot[slot]["clock_cycles"]) for slot in range(expected)]
        if any(current < previous for previous, current in zip(clocks, clocks[1:])):
            raise RuntimeError("trace clock cycles are not monotonic")
    if any(not str(item.get("record_kind", "")).endswith("issue_boundary") for item in trace):
        raise RuntimeError("trace records must use boundary semantics, not exclusive hardware-time claims")
    iteration_selector = metadata.get("iteration_selector")
    if iteration_selector is not None:
        expected_loop = int(iteration_selector["source_loop_ordinal"])
        expected_iteration = int(iteration_selector["relative_iteration"])
        if any(item.get("source_loop_ordinal") != expected_loop or
               item.get("relative_iteration") != expected_iteration for item in trace):
            raise RuntimeError("trace records do not carry the selected exact source-loop iteration")
    return {"status": "pass", "timestamps_per_sample": expected, "lineage_mode": "compiler_exact" if compiler_lineage else "exact",
            "trace_clock_monotonic": metadata.get("selected_blocks") is None,
            "trace_pair_monotonic": True, "selected_families": sorted(selected_families),
            "selected_source_ordinals": selected_ordinals,
            "iteration_selector": iteration_selector}
