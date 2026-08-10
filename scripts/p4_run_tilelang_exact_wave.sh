#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/p4_tilelang_exact_wave.json"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p4_tilelang_exact_wave/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"; cp "${CONFIG}" "${RUN_ROOT}/config.json"
for i in 1 2 3; do
  OUT="${RUN_ROOT}/process_${i}"; DUMP="${ROOT_DIR}/logs/build/p4_tilelang_exact_wave/${STAMP}/process_${i}"
  mkdir -p "${OUT}" "${DUMP}"; export TILELANG_DUMP_KERNEL_SOURCE="${DUMP}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p3_tilelang_global_memory.py" --config "${CONFIG}" --output-dir "${OUT}" --kernel-dump-dir "${DUMP}" 2>&1 | tee "${OUT}/run.log"
done
printf '%s\n' "${RUN_ROOT}"
