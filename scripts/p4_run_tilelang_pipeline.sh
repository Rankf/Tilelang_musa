#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/p4_tilelang_pipeline.json"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p4_tilelang_pipeline/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"; cp "${CONFIG}" "${RUN_ROOT}/config.json"
for i in 1 2 3; do
  OUT="${RUN_ROOT}/process_${i}"; DUMP="${ROOT_DIR}/logs/build/p4_tilelang_pipeline/${STAMP}/process_${i}"
  mkdir -p "${OUT}" "${DUMP}"; export TILELANG_DUMP_KERNEL_SOURCE="${DUMP}"
  mthreads-gmi > "${OUT}/gpu_before.txt" 2>&1
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" --config "${CONFIG}" --output-dir "${OUT}" 2>&1 | tee "${OUT}/run.log"
  mthreads-gmi > "${OUT}/gpu_after.txt" 2>&1
done
printf '%s\n' "${RUN_ROOT}"
