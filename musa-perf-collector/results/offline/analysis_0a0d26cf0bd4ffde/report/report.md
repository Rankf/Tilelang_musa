# Offline TileLang-MUSA analysis — kernel

## Scope

This report was generated without MUSA kernel execution, Event timing, profiler counters, or an S5000 architecture profile.

## Compiler provenance

- Target: `{"kind":"musa","tag":"","keys":["musa","gpu"],"max_num_threads":1024,"thread_warp_size":32,"arch":"mp_31"}`
- TileLang revision: `4d666968a005846314e9a84d8ff121c5dbffa9ee`
- Device execution: `False`
- Captured passes: 81

## Static demand by execution domain

- Per block FLOPs: `2097152`
- Per block logical bytes: `361984`
- Per block AI (all logical): `5.793493635077794` FLOPs/byte
- Per kernel-launch FLOPs: `2097152`
- Per kernel-launch logical bytes: `361984`
- Per kernel-launch AI (global logical): `63.01538461538462` FLOPs/byte

## Structure

- Buffers: 7
- TileOp actions: 6
- TileOp opaque records: 0
- Pipeline loops: 0
- Data-dependence edges: 5
- Offline-static status: `complete`

## Important limitations

- Logical bytes are compiler-derived accesses, not measured MUSA bus traffic.
- Capacity traffic uses a bounded static schedule and is not an S5000 cache-hit measurement.
- The upper-bound formula contains uncalibrated throughput and bandwidth variables, so no absolute latency is reported.
