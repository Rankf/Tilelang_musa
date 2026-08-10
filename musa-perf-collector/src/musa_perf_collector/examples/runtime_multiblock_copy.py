"""Reviewed four-block MUSA workload for bounded fixed-slot Tile tracing."""

from __future__ import annotations

from typing import Any

from ..adapters.tilelang.runtime_workload import PreparedRuntimeCase, RuntimeWorkload
from ..collectors.runtime.lineage_selection import resolve as resolve_trace_lineage
from ..core.ids import stable_id


class MultiBlockCopyWorkload(RuntimeWorkload):
    """A copy TileOp executed once by each of four blocks.

    Selected static-grid blocks own disjoint boundary slot pairs; no atomic
    append or cross-block timestamp ordering is used.
    """

    _selected_blocks = [[0, 0, 0], [3, 0, 0]]

    def prepare(self, config: dict[str, Any], *, variant: str = "baseline") -> PreparedRuntimeCase:
        import tilelang
        import tilelang.language as T
        import torch

        block_size = int(config.get("block_size", 128))
        blocks = int(config.get("blocks", 4))
        size = int(config.get("size", block_size * blocks))
        threads = int(config.get("threads", block_size))
        seed = int(config.get("input_seed", 20260730))
        if blocks != 4 or block_size != 128 or size != 512 or threads != 128:
            raise ValueError("reviewed multi-block copy prototype requires blocks=4, block_size=size/4=threads=128")
        tilelang.disable_cache()
        execution_stream = self.resolve_execution_stream(config)
        deep = config.get("deep", {})
        trace_ordinals, selection_evidence = resolve_trace_lineage(deep, workload_config=config)
        selected_blocks = [list(map(int, block)) for block in deep.get("block_coordinates", self._selected_blocks)]
        trace_slots = len(trace_ordinals) * len(selected_blocks) * 2

        if variant == "instrumented":
            if list(deep.get("tileop_families", ["copy"])) != ["copy"]:
                raise RuntimeError("multi-block copy trace requires exactly tileop_families=[copy]")
            if trace_ordinals != [0]:
                raise RuntimeError("multi-block copy only permits exact source TileOp ordinal 0")
            if not selected_blocks or any(block[1:] != [0, 0] or not 0 <= block[0] < blocks
                                          for block in selected_blocks):
                raise RuntimeError("multi-block copy selected block is outside the static 4x1x1 grid")
            if int(deep.get("max_tileops_per_block", 1)) < 1:
                raise RuntimeError("copy trace requires max_tileops_per_block >= 1")
            if int(deep.get("max_trace_records", 128)) < trace_slots or int(deep.get("max_trace_bytes", 8192)) < trace_slots * 8:
                raise RuntimeError("trace budget cannot hold every selected block boundary pair")

        if variant == "instrumented":
            @tilelang.jit(target="musa", pass_configs={
                "tl.provenance_trace_enable": True,
                "tl.provenance_trace_ordinal": 0,
                "tl.provenance_trace_buffer": "Trace",
                "tl.provenance_trace_blocks": [value for block in selected_blocks for value in block],
            })
            def kernel(A, Trace):
                A: T.Tensor[[512], T.float32]
                Trace: T.Tensor[[trace_slots], T.int64]
                out = T.empty((512,), "float32")
                with T.Kernel(4, threads=128) as bx:
                    # Keep the dynamic trace extent in the eager builder's
                    # closure; Simplify removes this compile-time no-op.
                    T.evaluate(trace_slots)
                    shared = T.alloc_shared((128,), "float32")
                    T.copy(A[bx * 128], shared)
                    T.copy(shared, out[bx * 128])
                return out
        else:
            @tilelang.jit(target="musa")
            def kernel(A):
                A: T.Tensor[[512], T.float32]
                out = T.empty((512,), "float32")
                with T.Kernel(4, threads=128) as bx:
                    shared = T.alloc_shared((128,), "float32")
                    T.copy(A[bx * 128], shared)
                    T.copy(shared, out[bx * 128])
                return out

        torch.manual_seed(seed)
        torch.musa.manual_seed_all(seed)
        a = torch.randn((size,), device="musa", dtype=torch.float32)
        execution_stream.wait_stream(torch.musa.current_stream())
        compiled = kernel.compile()
        trace_device = torch.empty((trace_slots,), device="musa", dtype=torch.int64) if variant == "instrumented" else None
        trace_host = torch.empty((trace_slots,), dtype=torch.int64).pin_memory() if variant == "instrumented" else None
        collector_stream = torch.musa.Stream() if variant == "instrumented" else None
        trace_ready = torch.musa.Event() if variant == "instrumented" else None
        copy_done = torch.musa.Event() if variant == "instrumented" else None
        execution_stream.wait_stream(torch.musa.current_stream())
        trace_generation = 0
        readback_generation = 0

        def run():
            nonlocal trace_generation
            with torch.musa.stream(execution_stream):
                value = compiled(a, trace_device) if variant == "instrumented" else compiled(a)
            if variant == "instrumented":
                trace_generation += 1
            return value

        def compare(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

        source = compiled.get_kernel_source()
        lineage = None
        trace_reader = None
        enqueue_trace_readback = None
        finish_trace_readback = None
        if variant == "instrumented":
            source_op_id = stable_id("op", ["multiblock_copy", "source_copy"])
            lineage = [
                {"trace_slot": block_index * 2 + boundary_index, "semantic_op_id": source_op_id,
                 "source_op_id": source_op_id, "semantic_op": "copy", "canonical_family": "copy",
                 "boundary": boundary, "block": block, "lineage": "compiler_exact_lineage",
                 "source_ordinal": 0, "compiler_provenance_annotation": "tl.provenance_ordinal"}
                for block_index, block in enumerate(selected_blocks)
                for boundary_index, boundary in enumerate(("begin", "end"))
            ]

            def records(values):
                return [
                    {"record_kind": f"copy_{boundary}_issue_boundary", "block": block,
                     "clock_cycles": values[block_index * 2 + boundary_index], "instrumented": True,
                     "source_ordinal": 0, "trace_slot": block_index * 2 + boundary_index,
                     "boundary": boundary}
                    for block_index, block in enumerate(selected_blocks)
                    for boundary_index, boundary in enumerate(("begin", "end"))
                ]

            def trace_reader():
                if trace_generation == 0:
                    raise RuntimeError("instrumented trace has not been produced")
                return records(trace_device.cpu().tolist())

            def enqueue_trace_readback():
                nonlocal readback_generation
                if trace_generation == 0:
                    raise RuntimeError("instrumented trace has not been produced")
                producer_stream_id = int(execution_stream.stream_id)
                trace_ready.record(execution_stream)
                with torch.musa.stream(collector_stream):
                    collector_stream.wait_event(trace_ready)
                    trace_host.copy_(trace_device, non_blocking=True)
                    copy_done.record(collector_stream)
                readback_generation = trace_generation
                return {"generation_id": readback_generation, "copy_done_immediately": bool(copy_done.query()),
                        "host_is_pinned_musa": bool(trace_host.is_pinned()), "copy_bytes": trace_slots * 8,
                        "producer_stream_id": producer_stream_id,
                        "collector_stream_id": int(collector_stream.stream_id),
                        "producer_event_recorded": True, "collector_waited_for_producer": True,
                        "copy_done_event_recorded": True}

            def finish_trace_readback():
                if readback_generation == 0:
                    raise RuntimeError("asynchronous trace readback has not been enqueued")
                copy_done.synchronize()
                return records(trace_host.tolist())

        return PreparedRuntimeCase(
            name="multiblock_copy_fp32", kernel_id=stable_id("kernel", ["multiblock_copy", variant, size, threads]),
            kernel_source=source, run=run, reference=lambda: a, compare=compare,
            launch={"grid": [blocks, 1, 1], "block": [threads, 1, 1], "shared_memory_bytes": None,
                    "stream": str(config.get("execution_stream", "current")),
                    "stream_id": int(execution_stream.stream_id)},
            instrumented=variant == "instrumented", lineage=lineage, tile_trace=trace_reader, run_timed=run,
            trace_metadata={"variant": "compiler_exact_provenance_multiblock_fixed_slots", "timestamps_per_sample": trace_slots,
                            "selected_blocks": selected_blocks, "trace_layout": "op_major_block_major_paired",
                            "cross_block_clock_order": "not_compared", "readback_policy": "capability_gated_async_after_window",
                            "device_trace_bytes": trace_slots * 8, "trace_storage": "caller_preallocated_input",
                            "clock_semantics": "issue_boundary"} if variant == "instrumented" else None,
            enqueue_trace_readback=enqueue_trace_readback, finish_trace_readback=finish_trace_readback,
            current_trace_generation=(lambda: trace_generation) if variant == "instrumented" else None,
            execution_stream=execution_stream,
        )


def build_workload(config: dict[str, Any]) -> RuntimeWorkload:
    return MultiBlockCopyWorkload()
