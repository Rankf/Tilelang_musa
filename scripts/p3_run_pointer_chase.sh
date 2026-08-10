#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUT_DIR="${ROOT_DIR}/results/raw/p3_pointer_chase/${STAMP}"
BIN_DIR="${ROOT_DIR}/results/derived/p3_pointer_chase"
BIN="${BIN_DIR}/p3_pointer_chase_${STAMP}"
mkdir -p "${OUT_DIR}" "${BIN_DIR}" "${ROOT_DIR}/logs/build" 
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mthreads-gmi > "${OUT_DIR}/gpu_before.txt"
printf '%q ' mcc -O2 "${ROOT_DIR}/benchmarks/p3_musa_pointer_chase.mu" "-L${MUSA_HOME}/lib" -lmusart -o "${BIN}" > "${OUT_DIR}/mcc_command.txt"
printf '\n' >> "${OUT_DIR}/mcc_command.txt"
mcc -O2 "${ROOT_DIR}/benchmarks/p3_musa_pointer_chase.mu" "-L${MUSA_HOME}/lib" -lmusart -o "${BIN}" > "${OUT_DIR}/mcc.log" 2>&1
sha256sum "${ROOT_DIR}/benchmarks/p3_musa_pointer_chase.mu" "${BIN}" > "${OUT_DIR}/sha256.txt"
"${BIN}" > "${OUT_DIR}/raw_samples.csv" 2> "${OUT_DIR}/stderr.log"
mthreads-gmi > "${OUT_DIR}/gpu_after.txt"
printf '%s\n' "${OUT_DIR}"
