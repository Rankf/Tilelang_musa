#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_env=/root/miniconda3/envs/tilefoundry_lq
tilelang_root=/root/iie/ruankefan/tilefoundry_musa/third_party/tilelang_musa
driver52_lib="$python_env/musa-driver-5.2/usr/lib/x86_64-linux-gnu"

# Keep the installed kernel driver unchanged.  These paths select only the
# SDK 5.2 user-space/runtime stack required by torch_musa and TileLang.
export LD_LIBRARY_PATH="$python_env/lib:$driver52_lib:/usr/mpi/gcc/openmpi-4.1.7a1/lib:$tilelang_root/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root:$tilelang_root${PYTHONPATH:+:$PYTHONPATH}"

# This host has GCC 12 executables without the matching libstdc++ headers.
# Point mcc 5.2 at the complete GCC 11 headers without changing operator code.
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

exec /usr/local/bin/musa52-run "$python_env/bin/python" \
  "$repo_root/benchmarks/compiler_info_elementwise_runtime.py" "$@"
