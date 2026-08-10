# DeepStack S5000 patches

This directory records the small changes made to the upstream DeepStack/
TileSight pipeline model without vendoring another complete source checkout.

- `matmul_pipeline_wave_s5000.patch` adds modeled L2 and shared-memory
  utilization to `matmul_pipeline_wave.py`.
- `elementwise_pipeline_wave.py` currently has no delta from the checked
  upstream version, so there is no elementwise patch to record.

Apply the patch from the root of a compatible DeepStack checkout with:

```bash
git apply /path/to/matmul_pipeline_wave_s5000.patch
```
