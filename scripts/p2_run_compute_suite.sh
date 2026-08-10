#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUT="${ROOT_DIR}/results/raw/p2_compute/${STAMP}"
DUMP="${ROOT_DIR}/logs/build/p2_compute/${STAMP}"
mkdir -p "${OUT}" "${DUMP}"
export TILELANG_DUMP_KERNEL_SOURCE="${DUMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mthreads-gmi > "${OUT}/gpu_before.txt"
"${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" --config "${ROOT_DIR}/configs/p2_compute_suite.json" --output-dir "${OUT}" 2>&1 | tee "${OUT}/run.log"
mthreads-gmi > "${OUT}/gpu_after.txt"
echo "${OUT}"
