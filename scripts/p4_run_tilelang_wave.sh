#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/p4_tilelang_wave.json"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p4_tilelang_wave/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"
cp "${CONFIG_PATH}" "${RUN_ROOT}/config.json"
for process_index in 1 2 3; do
  OUT_DIR="${RUN_ROOT}/process_${process_index}"
  DUMP_DIR="${ROOT_DIR}/logs/build/p4_tilelang_wave/${STAMP}/process_${process_index}"
  mkdir -p "${OUT_DIR}" "${DUMP_DIR}"
  export TILELANG_DUMP_KERNEL_SOURCE="${DUMP_DIR}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p3_tilelang_global_memory.py" --config "${CONFIG_PATH}" --output-dir "${OUT_DIR}" --kernel-dump-dir "${DUMP_DIR}" 2>&1 | tee "${OUT_DIR}/run.log"
done
printf '%s\n' "${RUN_ROOT}"
