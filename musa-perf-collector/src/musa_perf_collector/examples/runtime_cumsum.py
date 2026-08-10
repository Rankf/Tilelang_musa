"""Reviewed MUSA runtime workload for the ``scan`` TileOp family."""

from __future__ import annotations

from typing import Any

from ..adapters.tilelang.runtime_workload import PreparedRuntimeCase, RuntimeWorkload
from ..core.ids import stable_id


class CumsumWorkload(RuntimeWorkload):
    """One-block scan adapter; explicit bounds keep deep instrumentation safe."""

    def prepare(self, config: dict[str, Any], *, variant: str = "baseline") -> PreparedRuntimeCase:
        import tilelang
        import tilelang.language as T
        import torch

        size = int(config.get("size", 128))
        threads = int(config.get("threads", 128))
        seed = int(config.get("input_seed", 20260729))
        if size != threads:
            raise ValueError("runtime cumsum prototype requires size == threads")
        tilelang.disable_cache()
        execution_stream = self.resolve_execution_stream(config)
        deep = config.get("deep", {})
        if variant == "instrumented":
            if list(deep.get("tileop_families", ["scan"])) != ["scan"]:
                raise RuntimeError("cumsum trace requires exactly tileop_families=[scan]")
            if int(deep.get("max_tileops_per_block", 1)) < 1:
                raise RuntimeError("scan trace requires max_tileops_per_block >= 1")
            if int(deep.get("max_trace_records", 128)) < 2 or int(deep.get("max_trace_bytes", 8192)) < 16:
                raise RuntimeError("trace budget cannot hold two int64 timestamps")
            if deep.get("block_coordinates", [[0, 0, 0]]) != [[0, 0, 0]]:
                raise RuntimeError("this single-block prototype only permits block_coordinates=[[0,0,0]]")

        if variant == "instrumented":
            @tilelang.jit(target="musa", pass_configs={
                "tl.provenance_trace_enable": True,
                "tl.provenance_trace_ordinal": 1,
                "tl.provenance_trace_buffer": "Trace",
            })
            def kernel(A, Trace):
                A: T.Tensor[[size], T.float32]
                Trace: T.Tensor[[2], T.int64]
                out = T.empty((size,), "float32")
                with T.Kernel(1, threads=threads) as _:
                    shared = T.alloc_shared((size,), "float32")
                    T.copy(A, shared)
                    T.cumsum(shared, shared, dim=0)
                    T.copy(shared, out)
                return out
        else:
            @tilelang.jit(target="musa")
            def kernel(A):
                A: T.Tensor[[size], T.float32]
                out = T.empty((size,), "float32")
                with T.Kernel(1, threads=threads) as _:
                    shared = T.alloc_shared((size,), "float32")
                    T.copy(A, shared)
                    T.cumsum(shared, shared, dim=0)
                    T.copy(shared, out)
                return out

        torch.manual_seed(seed)
        torch.musa.manual_seed_all(seed)
        a = torch.randn((size,), device="musa", dtype=torch.float32)
        execution_stream.wait_stream(torch.musa.current_stream())
        compiled = kernel.compile()
        trace_device = torch.empty((2,), device="musa", dtype=torch.int64) if variant == "instrumented" else None
        trace_host = torch.empty((2,), dtype=torch.int64).pin_memory() if variant == "instrumented" else None
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

        def reference():
            return torch.cumsum(a, dim=0)

        def compare(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

        source = compiled.get_kernel_source()
        lineage = None
        trace_reader = None
        enqueue_trace_readback = None
        finish_trace_readback = None
        if variant == "instrumented":
            lineage = [{"trace_slot": 0, "boundary": "begin", "block": [0, 0, 0],
                        "semantic_op_id": stable_id("op", ["cumsum", "cumsum_begin"]),
                        "semantic_op": "cumsum", "canonical_family": "scan", "lineage": "compiler_exact_lineage",
                        "source_ordinal": 1, "compiler_provenance_annotation": "tl.provenance_ordinal"},
                       {"trace_slot": 1, "boundary": "end", "block": [0, 0, 0],
                        "semantic_op_id": stable_id("op", ["cumsum", "cumsum_end"]),
                        "semantic_op": "cumsum", "canonical_family": "scan", "lineage": "compiler_exact_lineage",
                        "source_ordinal": 1, "compiler_provenance_annotation": "tl.provenance_ordinal"}]

            def records(values):
                return [{"trace_slot": 0, "boundary": "begin", "source_ordinal": 1,
                         "record_kind": "cumsum_begin_issue_boundary", "block": [0, 0, 0],
                         "clock_cycles": values[0], "instrumented": True},
                        {"trace_slot": 1, "boundary": "end", "source_ordinal": 1,
                         "record_kind": "cumsum_end_issue_boundary", "block": [0, 0, 0],
                         "clock_cycles": values[1], "instrumented": True}]

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
                        "host_is_pinned_musa": bool(trace_host.is_pinned()), "copy_bytes": 16,
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
            name="cumsum_fp32", kernel_id=stable_id("kernel", ["cumsum", variant, size, threads]), kernel_source=source,
            run=run, reference=reference, compare=compare,
            launch={"grid": [1, 1, 1], "block": [threads, 1, 1], "shared_memory_bytes": None,
                    "stream": str(config.get("execution_stream", "current")),
                    "stream_id": int(execution_stream.stream_id)},
            instrumented=variant == "instrumented", lineage=lineage, tile_trace=trace_reader, run_timed=run,
            trace_metadata={"variant": "compiler_exact_provenance_two_point", "timestamps_per_sample": 2,
                            "readback_policy": "capability_gated_async_after_window", "device_trace_bytes": 16,
                            "trace_storage": "caller_preallocated_input", "clock_semantics": "issue_boundary"} if variant == "instrumented" else None,
            enqueue_trace_readback=enqueue_trace_readback, finish_trace_readback=finish_trace_readback,
            current_trace_generation=(lambda: trace_generation) if variant == "instrumented" else None,
            execution_stream=execution_stream,
        )


def build_workload(config: dict[str, Any]) -> RuntimeWorkload:
    return CumsumWorkload()
