#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/p3_memory_global_initial.json}"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUT_DIR="${ROOT_DIR}/results/raw/p3_global_memory/${STAMP}"
DUMP_DIR="${ROOT_DIR}/logs/build/p3_global_memory/${STAMP}"
mkdir -p "${OUT_DIR}" "${DUMP_DIR}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TILELANG_DUMP_KERNEL_SOURCE="${DUMP_DIR}"
test -f "${CONFIG_PATH}"
"${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p3_tilelang_global_memory.py" --config "${CONFIG_PATH}" --output-dir "${OUT_DIR}" --kernel-dump-dir "${DUMP_DIR}" 2>&1 | tee "${OUT_DIR}/run.log"
printf '%s\n' "${OUT_DIR}"
