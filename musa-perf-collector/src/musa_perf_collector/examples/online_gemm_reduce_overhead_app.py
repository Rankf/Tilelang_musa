"""S5000 ABBA validation for the native online periodic dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

from musa_perf_collector.online import auto_install


def _batch_event_ms(torch, run, launches: int) -> float:
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        run()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def main() -> None:
    import tilelang
    import tilelang.language as T
    import torch

    m, n, k, threads = 128, 128, 64, 256

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
            T.reduce_sum(acc, out_local, dim=1, clear=True)
            T.copy(out_local, out)
        return out

    torch.manual_seed(20260730)
    torch.musa.manual_seed_all(20260730)
    a = torch.randn((m, k), device="musa", dtype=torch.float16)
    b = torch.randn((n, k), device="musa", dtype=torch.float16)
    online = kernel.compile()
    metadata = online.online_instrumentation
    if metadata.get("dispatch_path") != "native_c_vectorcall":
        raise RuntimeError(f"native dispatcher unavailable: {metadata.get('dispatch_path')}")
    baseline = metadata["baseline_kernel"]

    # Three application-owned correctness launches preserve the same 4099
    # online-launch population as the reviewed Python-dispatch experiment.
    outputs = [online(a, b) for _ in range(3)]
    torch.musa.synchronize()
    reference = torch.matmul(a.float(), b.float().transpose(0, 1)).sum(dim=1)
    for output in outputs:
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    launches = 512
    direct_samples = []
    online_samples = []
    direct_run = lambda: baseline(a, b)
    online_run = lambda: online(a, b)
    for _ in range(4):
        direct_samples.append(_batch_event_ms(torch, direct_run, launches))
        online_samples.append(_batch_event_ms(torch, online_run, launches))
        online_samples.append(_batch_event_ms(torch, online_run, launches))
        direct_samples.append(_batch_event_ms(torch, direct_run, launches))

    direct_median = statistics.median(direct_samples)
    online_median = statistics.median(online_samples)
    absolute_ms = online_median - direct_median
    relative = absolute_ms / direct_median if direct_median else None
    session = auto_install.current_session()
    if session is None:
        raise RuntimeError("online session is unavailable")
    session.report_correctness(status="pass", source="application_assert_close")
    session.flush()
    result = {
        "schema": "online-native-dispatch-overhead-v1",
        "measurement": "MUSA Event ABBA batch average",
        "abba_rounds": 4,
        "launches_per_batch": launches,
        "direct_samples_ms": direct_samples,
        "online_samples_ms": online_samples,
        "direct_median_ms": direct_median,
        "online_median_ms": online_median,
        "absolute_overhead_ms": absolute_ms,
        "relative_overhead": relative,
        "dispatch_path": metadata["dispatch_path"],
        "session": str(session.directory),
    }
    output = Path(session.directory) / "native_dispatch_overhead.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
