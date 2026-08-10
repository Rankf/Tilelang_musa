# Attention family

`gqa/` contains GQA/MHA attention integration outside the reusable GEMM bodies.
`mla_nsa/` contains MLA, NSA, and related specialized attention paths.

Reusable QK/PV matrix kernels belong in `../gemm/attention_qk_pv/`.
