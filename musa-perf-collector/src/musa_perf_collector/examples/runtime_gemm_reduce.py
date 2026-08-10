"""Reviewed MUSA runtime workload: FP16 GEMM followed by a row reduction."""

from __future__ import annotations

from typing import Any

from ..adapters.tilelang.runtime_workload import PreparedRuntimeCase, RuntimeWorkload
from ..collectors.runtime.lineage_selection import resolve as resolve_trace_lineage
from ..core.ids import stable_id


class GemmReduceSumWorkload(RuntimeWorkload):
    def prepare(self, config: dict[str, Any], *, variant: str = "baseline") -> PreparedRuntimeCase:
        import tilelang
        import tilelang.language as T
        import torch

        m, n, k = (int(config.get(name, default)) for name, default in (("M", 128), ("N", 128), ("K", 64)))
        threads = int(config.get("threads", 256))
        seed = int(config.get("input_seed", 20260729))
        reduction = str(config.get("reduction", "sum"))
        if reduction not in {"sum", "max"}:
            raise ValueError("reduction must be one of: sum, max")
        tilelang.disable_cache()
        execution_stream = self.resolve_execution_stream(config)

        deep = config.get("deep", {})
        trace_ordinals, selection_evidence = resolve_trace_lineage(deep, workload_config=config)
        trace_slots = len(trace_ordinals) * 2
        if variant == "instrumented":
            selected_families = list(deep.get("tileop_families", ["gemm_dense", "reduce"]))
            if selected_families != ["gemm_dense", "reduce"]:
                raise RuntimeError("this trace prototype requires the exact selected families [gemm_dense, reduce]")
            if int(deep.get("max_tileops_per_block", 3)) < len(selected_families):
                raise RuntimeError("selected TileOps exceed max_tileops_per_block")
            if not trace_ordinals or trace_ordinals != sorted(set(trace_ordinals)):
                raise RuntimeError("trace_source_ordinals must be unique and in source order")
            if any(item not in {3, 4} for item in trace_ordinals):
                raise RuntimeError("this reviewed workload only permits exact TileOp ordinals 3 (gemm) and 4 (reduce)")
            if len(trace_ordinals) > int(deep.get("max_tileops_per_block", 3)):
                raise RuntimeError("selected exact TileOps exceed max_tileops_per_block")
            if int(deep.get("max_trace_records", 128)) < trace_slots or int(deep.get("max_trace_bytes", 8192)) < trace_slots * 8:
                raise RuntimeError("trace budget cannot hold selected TileOp boundary timestamps")
            coordinates = deep.get("block_coordinates", [[0, 0, 0]])
            if coordinates != [[0, 0, 0]]:
                raise RuntimeError("this single-block prototype only permits block_coordinates=[[0,0,0]]")

        if variant == "instrumented":
            @tilelang.jit(target="musa", pass_configs={
                "tl.provenance_trace_enable": True,
                "tl.provenance_trace_ordinals": trace_ordinals,
                "tl.provenance_trace_buffer": "Trace",
            })
            def kernel(A, B, Trace):
                A: T.Tensor[[m, k], T.float16]
                B: T.Tensor[[n, k], T.float16]
                Trace: T.Tensor[[trace_slots], T.int64]
                out = T.empty((m,), "float32")
                with T.Kernel(1, threads=threads) as _:
                    # Keep the dynamic trace extent in the eager builder's
                    # closure; this compile-time no-op is removed by Simplify.
                    T.evaluate(trace_slots)
                    a_shared = T.alloc_shared((m, k), "float16")
                    b_shared = T.alloc_shared((n, k), "float16")
                    acc = T.alloc_fragment((m, n), "float32")
                    out_local = T.alloc_fragment((m,), "float32")
                    T.clear(acc)
                    T.copy(A, a_shared)
                    T.copy(B, b_shared)
                    T.gemm(a_shared, b_shared, acc, transpose_B=True)
                    if reduction == "sum":
                        T.reduce_sum(acc, out_local, dim=1, clear=True)
                    else:
                        T.reduce_max(acc, out_local, dim=1, clear=True)
                    T.copy(out_local, out)
                return out
        else:
            @tilelang.jit(target="musa")
            def kernel(A, B):
                A: T.Tensor[[m, k], T.float16]
                B: T.Tensor[[n, k], T.float16]
                out = T.empty((m,), "float32")
                with T.Kernel(1, threads=threads) as _:
                    a_shared = T.alloc_shared((m, k), "float16")
                    b_shared = T.alloc_shared((n, k), "float16")
                    acc = T.alloc_fragment((m, n), "float32")
                    out_local = T.alloc_fragment((m,), "float32")
                    T.clear(acc)
                    T.copy(A, a_shared)
                    T.copy(B, b_shared)
                    T.gemm(a_shared, b_shared, acc, transpose_B=True)
                    if reduction == "sum":
                        T.reduce_sum(acc, out_local, dim=1, clear=True)
                    else:
                        T.reduce_max(acc, out_local, dim=1, clear=True)
                    T.copy(out_local, out)
                return out

        torch.manual_seed(seed)
        torch.musa.manual_seed_all(seed)
        a = torch.randn((m, k), device="musa", dtype=torch.float16)
        b = torch.randn((n, k), device="musa", dtype=torch.float16)
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
                value = compiled(a, b, trace_device) if variant == "instrumented" else compiled(a, b)
            if variant == "instrumented":
                trace_generation += 1
            return value

        def reference():
            matrix = torch.matmul(a.float(), b.float().transpose(0, 1))
            return matrix.sum(dim=1) if reduction == "sum" else matrix.amax(dim=1)

        def compare(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        source = compiled.get_kernel_source()
        lineage = None
        trace_reader = None
        enqueue_trace_readback = None
        finish_trace_readback = None
        if variant == "instrumented":
            source_op_ids = {
                int(item["source_ordinal"]): str(item["source_op_id"])
                for item in (selection_evidence or {}).get("selected", [])
            }
            descriptors = []
            for ordinal in trace_ordinals:
                semantic_op = "gemm" if ordinal == 3 else f"reduce_{reduction}"
                family = "gemm_dense" if ordinal == 3 else "reduce"
                for boundary in ("begin", "end"):
                    descriptors.append({"ordinal": ordinal, "semantic_op": semantic_op,
                                        "canonical_family": family, "boundary": boundary})
            lineage = [
                {"trace_slot": slot,
                 "semantic_op_id": source_op_ids.get(item["ordinal"], stable_id("op", [f"gemm_reduce_{reduction}", item["semantic_op"]])),
                 "source_op_id": source_op_ids.get(item["ordinal"]),
                 "semantic_op": item["semantic_op"], "canonical_family": item["canonical_family"],
                 "block": [0, 0, 0],
                 "boundary": item["boundary"], "lineage": "compiler_exact_lineage",
                 "source_ordinal": item["ordinal"], "compiler_provenance_annotation": "tl.provenance_ordinal"}
                for slot, item in enumerate(descriptors)
            ]

            def records(values):
                return [
                    {"record_kind": f'{item["semantic_op"]}_{item["boundary"]}_issue_boundary',
                     "block": [0, 0, 0], "clock_cycles": values[slot], "instrumented": True,
                     "source_ordinal": item["ordinal"], "trace_slot": slot, "boundary": item["boundary"]}
                    for slot, item in enumerate(descriptors)
                ]

            def trace_reader():
                if trace_generation == 0:
                    raise RuntimeError("instrumented trace has not been produced")
                # This intentional synchronization happens only after the
                # timed window.  `run_timed` never calls it.
                values = trace_device.cpu().tolist()
                return records(values)

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
                values = trace_host.tolist()
                return records(values)

        return PreparedRuntimeCase(
            name=f"gemm_reduce_{reduction}_fp16",
            kernel_id=stable_id("kernel", [f"gemm_reduce_{reduction}", variant, m, n, k, threads]),
            kernel_source=source,
            run=run,
            reference=reference,
            compare=compare,
            launch={"grid": [1, 1, 1], "block": [threads, 1, 1], "shared_memory_bytes": None,
                    "stream": str(config.get("execution_stream", "current")),
                    "stream_id": int(execution_stream.stream_id)},
            instrumented=variant == "instrumented",
            lineage=lineage,
            tile_trace=trace_reader,
            run_timed=run,
            trace_metadata={"variant": "compiler_exact_provenance_paired_boundaries", "timestamps_per_sample": trace_slots,
                            "selected_source_ordinals": trace_ordinals,
                            "selection_evidence": selection_evidence,
                            "readback_policy": "capability_gated_async_after_window", "device_trace_bytes": trace_slots * 8,
                            "trace_storage": "caller_preallocated_input", "clock_semantics": "issue_boundary"} if variant == "instrumented" else None,
            enqueue_trace_readback=enqueue_trace_readback,
            finish_trace_readback=finish_trace_readback,
            current_trace_generation=(lambda: trace_generation) if variant == "instrumented" else None,
            execution_stream=execution_stream,
        )


def build_workload(config: dict[str, Any]) -> RuntimeWorkload:
    return GemmReduceSumWorkload()
