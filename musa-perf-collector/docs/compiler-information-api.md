# TileLang-MUSA compiler information API

This API collects compiler facts through TileLang-MUSA's existing public
`instruments`/`PassInstrument` interface. It does not modify TileLang-MUSA,
invoke TileSight, run a performance model, compile a device binary, or launch a
MUSA kernel when the offline convenience function is used.

## Offline collection

```python
from pathlib import Path

from musa_perf_collector.adapters.tilelang import collect_compiler_information

information = collect_compiler_information(
    prim_func,
    arch="mp_31",
    tilelang_root=Path("/root/s5000-calibration/tilelang-musa"),
)
```

The returned value is JSON-serializable and uses schema
`musa-perf-collector/tilelang-compiler-info-v1`.

## Collection during a normal TileLang compile

```python
import tilelang

from musa_perf_collector.adapters.tilelang import TileLangCompilerInfoCollector

collector = TileLangCompilerInfoCollector.create()
kernel = tilelang.compile(
    prim_func,
    target="musa",
    instruments=[collector.instrument],
)
information = collector.finalize(
    source_prim_func=prim_func,
    compiled_artifact=kernel.artifact,
    target=kernel.target,
    pass_config=kernel.pass_configs,
    compile_flags=kernel.compile_flags,
    tilelang_version=getattr(tilelang, "__version__", "unknown"),
    device_compile=True,
)
```

## Collected facts

- input Tile IR, buffers, loops, TileOps, access regions, and source launch;
- pipeline-planning and software-pipeline IR stages;
- inferred layouts, K-major and matrix-layout annotations;
- source TileOp provenance through TileOp lowering and final device IR;
- TMA/copy annotations, barriers, waits, fences, and lowered call classes;
- final kernel symbol, grid, block, warp, dynamic Shared Memory, target attrs,
  and launch parameters;
- compiler/pass configuration, source/final/source-code hashes, stage coverage,
  and TileLang revision when supplied.

Register count and spill usage are not claimed because they require structured
MCC/backend compiler resource output rather than ordinary TileLang TIR.
