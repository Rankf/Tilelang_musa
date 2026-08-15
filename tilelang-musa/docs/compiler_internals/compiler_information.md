# TileLang compiler information collection

`tilelang.tools.compiler_info` collects compiler facts through TileLang's
existing `instruments` argument and `CompiledArtifact`. It does not run a
kernel, read profiler counters, invoke a performance model, or modify the
`compile`, `jit`, or `lower` interfaces.

## Offline collection

```python
from tilelang.tools.compiler_info import collect_compiler_information

information = collect_compiler_information(
    prim_func,
    target={"kind": "musa", "arch": "mp_31"},
)
```

The returned dictionary is JSON-serializable and uses schema
`tilelang/compiler-info-v1`.

## Collection during normal compilation

```python
import tilelang
from tilelang.tools.compiler_info import CompilerInfoCollector

collector = CompilerInfoCollector()
kernel = tilelang.compile(
    prim_func,
    target={"kind": "musa", "arch": "mp_31"},
    instruments=[collector],
)
information = collector.finalize(
    source_prim_func=prim_func,
    compiled_artifact=kernel.artifact,
    target=kernel.target,
    pass_configs=kernel.pass_configs,
    compile_flags=kernel.compile_flags,
)
```

## Information and evidence boundary

The artifact contains source TileOps and direct element-wise stores, buffers,
loops, access regions, conservative dependencies and logical workload,
pipeline planning and injected pipeline information, inferred layouts, lowered
implementation calls, final launch information, compiler lineage where
available, pass coverage, and compiler/source hashes.

Direct Add and `max(input, 0)` stores use a provenance domain independent from
TileOps: `source.kernels[].elementwise_ops` and `elementwise_lineage`. This
keeps TileOp ordinals and existing GEMM lineage stable. The collector records
the observed expression, buffer accesses, shape, dtype, and loop identity. It
derives expression kind, operation count, elements per block/thread, and access
pattern. `same_index_contiguous` is emitted only after the bound index is
substituted and its one-dimensional affine block/loop coefficients prove exact
contiguous coverage; otherwise the access pattern is `unsupported`.

Pipeline information covers both forms emitted by TileLang: conventional
`software_pipeline_*` arrays and MUSA's warp-specialized mbarrier/TMA call
annotations. The source pipeline declaration retains the requested
`num_stages`; planned producer and consumer loops retain their source-loop
ordinal and the exact call annotations observed after `PipelinePlanning`.
Region-construction helpers are excluded from pipeline call kinds, while both
TileLang and TIRX synchronization calls are retained. Equivalent layout
annotations on nested blocks are represented once, with every original block
name preserved in `block_names`.

Logical bytes and dependency edges are static semantic derivations, not
measured bus traffic. Register count and spills require structured MCC/backend
compiler output. Cache hit rate, pipeline-stage duration, and scale-unit load
require runtime profiler or trace data and are explicitly reported as
unavailable rather than inferred as compiler facts.
