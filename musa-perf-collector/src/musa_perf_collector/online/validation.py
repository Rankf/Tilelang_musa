"""Independent structural audit for one online collection session."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


def audit_session(
    *,
    launches: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    enriched_records: list[dict[str, Any]],
    expected_trace_extents: dict[str, int],
    expected_trace_layouts: dict[str, list[dict[str, Any]] | tuple[dict[str, Any], ...]],
    expected_plan_hashes: dict[str, str | None],
    expected_readback_policies: dict[str, str],
    total_launch_count: int,
    fast_path_launch_count: int,
    fast_path_skip_reasons: dict[str, int],
    instrumented_launch_count: int,
    instrumented_dispatch_count: int,
    reserved_instrumented_launches: int,
    reserved_trace_records: int,
    reserved_trace_bytes: int,
    budget: Any,
) -> dict[str, Any]:
    """Cross-check counters, launch identities, slots, and hard budgets.

    This audit consumes only already-recorded evidence.  It does not trust the
    session's proposed status and never repairs or fabricates missing records.
    """
    violations: list[dict[str, Any]] = []

    def fail(code: str, **details: Any) -> None:
        violations.append({"code": code, **details})

    def nonnegative_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    def layout_block(
        layout: dict[str, Any], *, kernel_id: str, slot: int,
    ) -> tuple[Any, ...] | None:
        block = layout.get("block")
        if (
            not isinstance(block, (list, tuple))
            or len(block) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in block)
        ):
            fail(
                "MALFORMED_EXPECTED_TRACE_BLOCK",
                kernel_id=kernel_id, slot=slot, block=block)
            return None
        return tuple(block)

    def layout_pair_key(layout: dict[str, Any]) -> tuple[Any, ...]:
        raw_op_id = layout.get("op_id")
        op_id = raw_op_id if isinstance(raw_op_id, str) and raw_op_id else None
        ordinal = nonnegative_int(layout.get("ordinal"))
        block = layout.get("block")
        normalized_block = (
            tuple(block) if isinstance(block, (list, tuple)) and len(block) == 3
            and all(nonnegative_int(value) is not None for value in block)
            else None)
        raw_semantics = layout.get("lowering_semantics")
        semantics = raw_semantics if isinstance(raw_semantics, str) and raw_semantics in {
            "single", "source_envelope", "lowered_instance"} else None
        instance = nonnegative_int(layout.get("lowered_instance_index"))
        return op_id, ordinal, normalized_block, instance, semantics

    layout_by_kernel: dict[str, dict[int, dict[str, Any]]] = {}
    for kernel_id, layouts in expected_trace_layouts.items():
        indexed: dict[int, dict[str, Any]] = {}
        boundaries_by_pair: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        instances_by_source: dict[tuple[Any, ...], set[int]] = defaultdict(set)
        if not isinstance(layouts, (list, tuple)):
            fail("MALFORMED_EXPECTED_TRACE_LAYOUTS", kernel_id=kernel_id)
            layout_by_kernel[str(kernel_id)] = indexed
            continue
        for layout in layouts:
            if not isinstance(layout, dict):
                fail("MALFORMED_EXPECTED_TRACE_LAYOUT", kernel_id=kernel_id)
                continue
            slot = nonnegative_int(layout.get("slot"))
            if slot is None:
                fail("MALFORMED_EXPECTED_TRACE_LAYOUT", kernel_id=kernel_id)
                continue
            if slot in indexed:
                fail(
                    "DUPLICATE_EXPECTED_TRACE_SLOT", kernel_id=kernel_id,
                    slot=slot)
            indexed[slot] = layout
            raw_semantics = layout.get("lowering_semantics")
            semantics = raw_semantics if isinstance(raw_semantics, str) and raw_semantics in {
                "single", "source_envelope", "lowered_instance"} else None
            raw_boundary = layout.get("boundary")
            boundary = raw_boundary if isinstance(raw_boundary, str) and raw_boundary in {
                "begin", "end"} else None
            block = layout_block(layout, kernel_id=str(kernel_id), slot=slot)
            raw_op_id = layout.get("op_id")
            op_id = raw_op_id if isinstance(raw_op_id, str) and raw_op_id else None
            if op_id is None:
                fail(
                    "INVALID_EXPECTED_TRACE_OP_ID",
                    kernel_id=kernel_id, slot=slot, op_id=raw_op_id)
            raw_ordinal = layout.get("ordinal")
            ordinal = nonnegative_int(raw_ordinal)
            if ordinal is None:
                fail(
                    "INVALID_EXPECTED_TRACE_ORDINAL",
                    kernel_id=kernel_id, slot=slot, ordinal=raw_ordinal)
            if (
                not isinstance(layout.get("source_kind"), str)
                or not layout["source_kind"]
            ):
                fail(
                    "INVALID_EXPECTED_TRACE_SOURCE_KIND",
                    kernel_id=kernel_id, slot=slot,
                    source_kind=layout.get("source_kind"))
            if boundary is None:
                fail(
                    "INVALID_EXPECTED_TRACE_BOUNDARY",
                    kernel_id=kernel_id, slot=slot, boundary=raw_boundary)
            if semantics is None:
                fail(
                    "INVALID_EXPECTED_LOWERING_SEMANTICS",
                    kernel_id=kernel_id, slot=slot, semantics=raw_semantics)
            raw_instance = layout.get("lowered_instance_index")
            instance = nonnegative_int(raw_instance)
            if semantics == "lowered_instance":
                if instance is None:
                    fail(
                        "INVALID_EXPECTED_LOWERED_INSTANCE_INDEX",
                        kernel_id=kernel_id, slot=slot, instance=raw_instance)
                elif op_id is not None and ordinal is not None and block is not None:
                    instances_by_source[(
                        op_id, ordinal,
                        block, semantics,
                    )].add(instance)
            elif raw_instance is not None:
                fail(
                    "UNEXPECTED_LOWERED_INSTANCE_INDEX", kernel_id=kernel_id,
                    slot=slot, instance=raw_instance,
                    semantics=semantics)
            pair_key = (
                op_id, ordinal,
                block, instance, semantics,
            )
            boundaries_by_pair[pair_key].append(boundary)
        for pair_key, boundaries in boundaries_by_pair.items():
            if len(boundaries) != 2 or set(boundaries) != {"begin", "end"}:
                fail(
                    "EXPECTED_TRACE_PAIR_LAYOUT_MISMATCH",
                    kernel_id=kernel_id, pair=list(map(str, pair_key)),
                    boundaries=boundaries)
        for source_key, instances in instances_by_source.items():
            expected = set(range(len(instances)))
            if instances != expected:
                fail(
                    "NONCONTIGUOUS_EXPECTED_LOWERED_INSTANCES",
                    kernel_id=kernel_id, source=list(map(str, source_key)),
                    instances=sorted(instances), expected=sorted(expected))
        expected_slots = set(range(len(indexed)))
        if set(indexed) != expected_slots:
            fail(
                "NONCONTIGUOUS_EXPECTED_TRACE_SLOTS",
                kernel_id=kernel_id, slots=sorted(indexed),
                expected=sorted(expected_slots))
        layout_by_kernel[str(kernel_id)] = indexed

    for kernel_id, indexed in layout_by_kernel.items():
        extent = expected_trace_extents.get(kernel_id)
        if (
            isinstance(extent, bool) or not isinstance(extent, int)
            or extent < 0 or extent != len(indexed)
        ):
            fail(
                "EXPECTED_TRACE_EXTENT_LAYOUT_MISMATCH",
                kernel_id=kernel_id, extent=extent, layout_count=len(indexed))

    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    launch_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in launches:
        if not isinstance(record, dict):
            fail("MALFORMED_LAUNCH_RECORD")
            continue
        by_kernel[str(record.get("kernel_id"))].append(record)
        if record.get("schema") != "online-launch-observation-v1":
            fail(
                "LAUNCH_SCHEMA_MISMATCH", kernel_id=record.get("kernel_id"),
                launch_index=record.get("launch_index"))
        if record.get("status") not in {
            "observed", "baseline", "instrumented", "application_error",
            "instrumented_error",
        }:
            fail(
                "UNKNOWN_LAUNCH_STATUS", kernel_id=record.get("kernel_id"),
                launch_index=record.get("launch_index"),
                status=record.get("status"))
        if record.get("status") in {"application_error", "instrumented_error"}:
            fail(
                "LAUNCH_ERROR_STATUS", kernel_id=record.get("kernel_id"),
                launch_index=record.get("launch_index"), status=record.get("status"))
        kernel_id = record.get("kernel_id")
        launch_index = nonnegative_int(record.get("launch_index"))
        if not isinstance(kernel_id, str) or not kernel_id or launch_index is None:
            fail("MALFORMED_LAUNCH_INDEX", kernel_id=kernel_id)
            continue
        identity = (kernel_id, launch_index)
        launch_by_identity[identity] = record
    for kernel_id, records in by_kernel.items():
        try:
            indexes = [nonnegative_int(record.get("launch_index")) for record in records]
        except AttributeError:
            fail("MALFORMED_LAUNCH_INDEX", kernel_id=kernel_id)
            continue
        if any(index is None for index in indexes) or len(set(indexes)) != len(indexes):
            fail("INVALID_OR_DUPLICATE_LAUNCH_INDEX", kernel_id=kernel_id, indexes=indexes)

    if not isinstance(fast_path_skip_reasons, dict):
        fail("MALFORMED_FAST_PATH_REASONS")
        fast_path_skip_reasons = {}
    if nonnegative_int(total_launch_count) is None:
        fail("MALFORMED_TOTAL_LAUNCH_COUNT", value=total_launch_count)
        total_launch_count = 0
    if nonnegative_int(fast_path_launch_count) is None:
        fail("MALFORMED_FAST_PATH_LAUNCH_COUNT", value=fast_path_launch_count)
        fast_path_launch_count = 0
    if total_launch_count < len(launches):
        fail(
            "TOTAL_LAUNCH_COUNT_UNDERFLOW",
            total=total_launch_count, recorded=len(launches))
    if fast_path_launch_count > total_launch_count:
        fail(
            "FAST_PATH_LAUNCH_COUNT_OVERFLOW",
            total=total_launch_count, fast_path=fast_path_launch_count)
    if fast_path_launch_count + len(launches) != total_launch_count:
        fail(
            "TOTAL_LAUNCH_ACCOUNTING_MISMATCH", total=total_launch_count,
            fast_path=fast_path_launch_count, recorded=len(launches))
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fast_path_skip_reasons.values())
        or sum(fast_path_skip_reasons.values()) != fast_path_launch_count
    ):
        fail(
            "FAST_PATH_REASON_ACCOUNTING_MISMATCH",
            fast_path=fast_path_launch_count,
            reason_total=sum(
                value for value in fast_path_skip_reasons.values()
                if isinstance(value, int) and not isinstance(value, bool)))

    instrumented: set[tuple[str, int]] = set()
    for record in launches:
        if not isinstance(record, dict):
            continue
        if record.get("status") != "instrumented":
            continue
        kernel_id = record.get("kernel_id")
        launch_index = nonnegative_int(record.get("launch_index"))
        if not isinstance(kernel_id, str) or not kernel_id or launch_index is None:
            fail("MALFORMED_INSTRUMENTED_LAUNCH_IDENTITY")
            continue
        identity = (kernel_id, launch_index)
        instrumented.add(identity)
    if instrumented_launch_count != len(instrumented):
        fail(
            "INSTRUMENTED_LAUNCH_COUNT_MISMATCH",
            recorded=len(instrumented), counter=instrumented_launch_count)
    attempted = sum(
        isinstance(record, dict)
        and record.get("status") in {"instrumented", "instrumented_error"}
        for record in launches)
    if instrumented_dispatch_count != attempted:
        fail(
            "INSTRUMENTED_DISPATCH_COUNT_MISMATCH",
            recorded=attempted, counter=instrumented_dispatch_count)
    if reserved_instrumented_launches < instrumented_launch_count:
        fail(
            "INSTRUMENTED_RESERVATION_UNDERFLOW",
            reserved=reserved_instrumented_launches, actual=instrumented_launch_count)

    grouped_trace: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    generation_owners: dict[tuple[str, int], tuple[str, int]] = {}
    for record in trace_records:
        if not isinstance(record, dict):
            fail("MALFORMED_TRACE_RECORD")
            continue
        kernel_id = record.get("kernel_id")
        launch_index = nonnegative_int(record.get("launch_index"))
        if not isinstance(kernel_id, str) or not kernel_id or launch_index is None:
            fail("MALFORMED_TRACE_LAUNCH_IDENTITY", slot=record.get("slot"))
            continue
        identity = (kernel_id, launch_index)
        grouped_trace[identity].append(record)
        if identity not in instrumented:
            fail(
                "TRACE_WITHOUT_INSTRUMENTED_LAUNCH",
                kernel_id=identity[0], launch_index=identity[1])
        boundary = record.get("boundary")
        interval = record.get("interval_cycles")
        if boundary == "begin" and interval is not None:
            fail("BEGIN_HAS_INTERVAL", kernel_id=identity[0], slot=record.get("slot"))
        if not record.get("valid") and interval is not None:
            fail("INVALID_SLOT_HAS_INTERVAL", kernel_id=identity[0], slot=record.get("slot"))
        if record.get("schema") != "online-tile-trace-v1":
            fail(
                "TRACE_SCHEMA_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        if record.get("evidence_level") != "compiler_exact_instrumented_issue_boundary":
            fail(
                "TRACE_EVIDENCE_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        if record.get("plan_sha256") != expected_plan_hashes.get(identity[0]):
            fail(
                "TRACE_PLAN_HASH_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        if record.get("readback_policy") != expected_readback_policies.get(identity[0]):
            fail(
                "TRACE_READBACK_POLICY_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        launch = launch_by_identity.get(identity)
        if launch is not None:
            for trace_field, launch_field in (
                ("process_id", "pid"), ("device_id", "device_id"),
                ("stream_id", "stream_id"),
            ):
                if record.get(trace_field) != launch.get(launch_field):
                    fail(
                        "TRACE_LAUNCH_IDENTITY_MISMATCH", kernel_id=identity[0],
                        launch_index=identity[1], slot=record.get("slot"),
                        field=trace_field, trace=record.get(trace_field),
                        launch=launch.get(launch_field))
        clock = record.get("clock_cycles")
        if record.get("valid") and (
            isinstance(clock, bool) or not isinstance(clock, int)
        ):
            fail(
                "VALID_TRACE_CLOCK_MALFORMED", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        event_ms = record.get("kernel_event_ms")
        if event_ms is not None and (
            isinstance(event_ms, bool) or not isinstance(event_ms, (int, float))
            or not math.isfinite(float(event_ms)) or event_ms < 0
        ):
            fail(
                "TRACE_EVENT_TIME_MALFORMED", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"),
                value=event_ms)
        generation = nonnegative_int(record.get("generation"))
        if generation is None or generation == 0:
            fail(
                "MALFORMED_TRACE_GENERATION", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        else:
            generation_key = (identity[0], generation)
            owner = generation_owners.setdefault(generation_key, identity)
            if owner != identity:
                fail(
                    "TRACE_GENERATION_REUSED", kernel_id=identity[0],
                    generation=generation, owners=[owner[1], identity[1]])

        slot = nonnegative_int(record.get("slot"))
        if slot is None or slot not in layout_by_kernel.get(identity[0], {}):
            fail(
                "MISSING_EXPECTED_TRACE_LAYOUT", kernel_id=identity[0],
                launch_index=identity[1], slot=record.get("slot"))
        else:
            layout = layout_by_kernel[identity[0]][slot]
            for field in (
                "slot", "op_id", "ordinal", "source_kind", "boundary",
                "block", "lowered_instance_index", "lowering_semantics",
            ):
                if record.get(field) != layout.get(field):
                    fail(
                        "TRACE_LAYOUT_FIELD_MISMATCH", kernel_id=identity[0],
                        launch_index=identity[1], slot=slot, field=field,
                        expected=layout.get(field), actual=record.get(field))

    for identity in instrumented:
        extent = expected_trace_extents.get(identity[0])
        if extent is None:
            fail("MISSING_EXPECTED_TRACE_EXTENT", kernel_id=identity[0])
            continue
        records = grouped_trace.get(identity, [])
        try:
            slots = [nonnegative_int(record.get("slot")) for record in records]
        except AttributeError:
            fail(
                "MALFORMED_TRACE_SLOT", kernel_id=identity[0],
                launch_index=identity[1])
            continue
        if any(slot is None for slot in slots):
            fail("MALFORMED_TRACE_SLOT", kernel_id=identity[0], launch_index=identity[1])
            continue
        slots.sort()
        if slots != list(range(extent)):
            fail(
                "TRACE_SLOT_SET_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], expected_extent=extent, slots=slots)
            continue
        generations = {nonnegative_int(record.get("generation")) for record in records}
        if len(generations) != 1:
            fail(
                "TRACE_LAUNCH_GENERATION_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1], generations=sorted(map(str, generations)))
        event_values = {
            float(value) if (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and value >= 0
            ) else None
            for value in (record.get("kernel_event_ms") for record in records)
        }
        if len(event_values) != 1:
            fail(
                "TRACE_LAUNCH_EVENT_TIME_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1])
        variant_indexes = {
            nonnegative_int(record.get("instrumented_variant_launch_index"))
            for record in records}
        if len(variant_indexes) != 1:
            fail(
                "TRACE_VARIANT_INDEX_MISMATCH", kernel_id=identity[0],
                launch_index=identity[1])
        by_slot = {nonnegative_int(record.get("slot")): record for record in records}
        layouts = layout_by_kernel.get(identity[0], {})
        # Build the complete begin map before checking ends.  The compiler
        # currently emits layouts in slot order, but an independent audit must
        # not silently skip an interval merely because the serialized plan was
        # reordered.
        begins: dict[tuple[Any, ...], dict[str, Any]] = {}
        for slot, layout in layouts.items():
            pair_key = layout_pair_key(layout)
            if layout.get("boundary") == "begin":
                begins[pair_key] = by_slot.get(slot, {})
        for slot, layout in layouts.items():
            pair_key = layout_pair_key(layout)
            if layout.get("boundary") != "end":
                continue
            begin = begins.get(pair_key)
            end = by_slot.get(slot)
            if not begin or not end:
                continue
            if bool(begin.get("valid")) != bool(end.get("valid")):
                fail(
                    "TRACE_PAIR_VALIDITY_MISMATCH", kernel_id=identity[0],
                    launch_index=identity[1], slot=slot,
                    begin_valid=begin.get("valid"), end_valid=end.get("valid"))
            if begin.get("valid") and end.get("valid"):
                begin_clock = begin.get("clock_cycles")
                end_clock = end.get("clock_cycles")
                if (
                    isinstance(begin_clock, int) and not isinstance(begin_clock, bool)
                    and isinstance(end_clock, int) and not isinstance(end_clock, bool)
                ):
                    expected_interval = (
                        (end_clock & ((1 << 64) - 1))
                        - (begin_clock & ((1 << 64) - 1))) & ((1 << 64) - 1)
                    actual_interval = end.get("interval_cycles")
                    if (
                        isinstance(actual_interval, bool)
                        or not isinstance(actual_interval, int)
                        or actual_interval != expected_interval
                    ):
                        fail(
                            "TRACE_INTERVAL_MISMATCH", kernel_id=identity[0],
                            launch_index=identity[1], slot=slot,
                            expected=expected_interval,
                            actual=end.get("interval_cycles"))
                    if end.get("interval_semantics") != "lane0_issue_boundary_envelope":
                        fail(
                            "TRACE_INTERVAL_SEMANTICS_MISMATCH",
                            kernel_id=identity[0], launch_index=identity[1],
                            slot=slot)
            elif end.get("interval_cycles") is not None:
                fail(
                    "PARTIAL_PAIR_HAS_INTERVAL", kernel_id=identity[0],
                    launch_index=identity[1], slot=slot)

    variant_ordinals: list[int] = []
    for records in grouped_trace.values():
        if not records:
            continue
        value = records[0].get("instrumented_variant_launch_index")
        if isinstance(value, int) and not isinstance(value, bool):
            variant_ordinals.append(value)
            expected_cold = value == 0
            if any(record.get("instrumented_variant_cold_start") != expected_cold
                   for record in records):
                fail("TRACE_COLD_START_MISMATCH", variant_index=value)
    if sorted(variant_ordinals) != list(range(len(instrumented))):
        fail(
            "TRACE_VARIANT_INDEX_SEQUENCE_MISMATCH",
            indexes=sorted(variant_ordinals), expected_count=len(instrumented))

    if len(enriched_records) != len(trace_records):
        fail(
            "TRACE_ENRICHMENT_COUNT_MISMATCH",
            raw=len(trace_records), enriched=len(enriched_records))
    raw_by_slot: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in trace_records:
        if not isinstance(record, dict):
            continue
        kernel_id = record.get("kernel_id")
        launch_index = nonnegative_int(record.get("launch_index"))
        slot = nonnegative_int(record.get("slot"))
        if not isinstance(kernel_id, str) or not kernel_id or launch_index is None or slot is None:
            continue
        key = (kernel_id, launch_index, slot)
        raw_by_slot[key] = record
    enriched_keys: set[tuple[str, int, int]] = set()
    for record in enriched_records:
        if not isinstance(record, dict):
            fail("MALFORMED_ENRICHED_TRACE_IDENTITY")
            continue
        kernel_id = record.get("kernel_id")
        launch_index = nonnegative_int(record.get("launch_index"))
        slot = nonnegative_int(record.get("slot"))
        if not isinstance(kernel_id, str) or not kernel_id or launch_index is None or slot is None:
            fail("MALFORMED_ENRICHED_TRACE_IDENTITY")
            continue
        key = (kernel_id, launch_index, slot)
        if key in enriched_keys:
            fail(
                "DUPLICATE_ENRICHED_TRACE_SLOT", kernel_id=key[0],
                launch_index=key[1], slot=key[2])
        enriched_keys.add(key)
        raw = raw_by_slot.get(key)
        if raw is None:
            fail(
                "ENRICHED_TRACE_WITHOUT_RAW", kernel_id=key[0],
                launch_index=key[1], slot=key[2])
            continue
        for field in (
            "generation", "op_id", "ordinal", "source_kind", "boundary",
            "block", "clock_cycles", "valid", "interval_cycles",
            "interval_semantics", "lowered_instance_index", "lowering_semantics",
            "plan_sha256", "process_id", "device_id", "stream_id",
            "kernel_event_ms", "readback_policy", "evidence_level",
            "instrumented_variant_launch_index", "instrumented_variant_cold_start",
        ):
            if record.get(field) != raw.get(field):
                fail(
                    "ENRICHED_TRACE_FIELD_MISMATCH", kernel_id=key[0],
                    launch_index=key[1], slot=key[2], field=field)
        tileop = record.get("tileop")
        if not isinstance(tileop, dict) or tileop.get("op_id") != raw.get("op_id"):
            fail(
                "ENRICHED_TILEOP_ID_MISMATCH", kernel_id=key[0],
                launch_index=key[1], slot=key[2],
                trace_op_id=raw.get("op_id"),
                tileop_op_id=tileop.get("op_id") if isinstance(tileop, dict) else None)
    if len(trace_records) > reserved_trace_records:
        fail(
            "TRACE_RECORD_RESERVATION_UNDERFLOW",
            reserved=reserved_trace_records, actual=len(trace_records))
    if len(trace_records) * 8 > reserved_trace_bytes:
        fail(
            "TRACE_BYTE_RESERVATION_UNDERFLOW",
            reserved=reserved_trace_bytes, actual=len(trace_records) * 8)
    if reserved_instrumented_launches > budget.max_instrumented_launches:
        fail("INSTRUMENTED_LAUNCH_BUDGET_EXCEEDED")
    if reserved_trace_records > budget.max_trace_records:
        fail("TRACE_RECORD_BUDGET_EXCEEDED")
    if reserved_trace_bytes > budget.max_trace_bytes:
        fail("TRACE_BYTE_BUDGET_EXCEEDED")

    return {
        "schema": "online-session-audit-v1",
        "audit_revision": "P6.9.4",
        "checks": [
            "launch_accounting", "fast_reason_accounting",
            "instrumented_dispatch_accounting", "trace_slot_layout",
            "plan_hash", "launch_trace_identity", "generation_ownership",
            "trace_pair_layout", "lowered_instance_layout",
            "trace_pair_validity", "clock_interval_derivation",
            "event_time_consistency",
            "raw_enriched_identity", "tileop_identity", "strict_schema_types",
            "budget_bounds",
        ],
        "status": "pass" if not violations else "fail",
        "violation_count": len(violations),
        "violations": violations,
        "observed": {
            "launches": len(launches),
            "total_launches": total_launch_count,
            "fast_path_launches": fast_path_launch_count,
            "instrumented_launches": len(instrumented),
            "instrumented_dispatches": attempted,
            "trace_records": len(trace_records),
            "enriched_trace_records": len(enriched_records),
        },
        "reserved": {
            "instrumented_launches": reserved_instrumented_launches,
            "trace_records": reserved_trace_records,
            "trace_bytes": reserved_trace_bytes,
        },
    }
