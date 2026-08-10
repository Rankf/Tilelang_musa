#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_large_gemm_graph_queue/${STAMP}"
CONFIG="${ROOT_DIR}/configs/p7_large_gemm_graph_queue_diagnostic.json"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"; cp "${CONFIG}" "${RUN_ROOT}/"; mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true
mapfile -t DEPTHS < <("${ROOT_DIR}/.venv-tilelang-musa/bin/python" -c "import json; print(*json.load(open('${CONFIG}'))['queue_depths'], sep='\\n')")
TRIALS="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
trial_count="$(${TRIALS} -c "import json; print(json.load(open('${CONFIG}'))['trials_per_depth'])")"
for depth in "${DEPTHS[@]}"; do
  for trial in $(seq 1 "${trial_count}"); do
    out="${RUN_ROOT}/depth_${depth}/trial_${trial}"; dump="${ROOT_DIR}/logs/build/p7_large_gemm_graph_queue/${STAMP}/depth_${depth}/trial_${trial}"
    mkdir -p "${out}" "${dump}"; export TILELANG_DUMP_KERNEL_SOURCE="${dump}"
    "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p7_large_gemm_graph_queue_diagnostic.py" --config "${CONFIG}" --queue-depth "${depth}" --trial "${trial}" --output "${out}/result.json" 2>&1 | tee "${out}/run.log"
  done
done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true; printf '%s\n' "${RUN_ROOT}"
