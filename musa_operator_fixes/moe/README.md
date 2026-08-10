# MoE family

`routing/` contains top-k and dispatch logic.
`permute_unpermute/` contains token layout transforms.
`experts/` contains non-GEMM expert integration.

Grouped matrix bodies belong in `../gemm/moe_grouped/`.
