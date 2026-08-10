#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/logs/build"
RUN_DIR="${ROOT_DIR}/logs/run"
OUT_BIN="${ROOT_DIR}/results/derived/p0_musa_smoke"
MUSA_LIB_DIR="/usr/local/musa-4.3.5/lib"
mkdir -p "${BUILD_DIR}" "${RUN_DIR}" "$(dirname "${OUT_BIN}")"

printf '%q ' mcc -O2 "${ROOT_DIR}/musa_smoke.mu" "-L${MUSA_LIB_DIR}" -lmusart -o "${OUT_BIN}" > "${BUILD_DIR}/p0_musa_smoke_command.txt"
echo >> "${BUILD_DIR}/p0_musa_smoke_command.txt"

mthreads-gmi > "${RUN_DIR}/p0_musa_smoke_gpu_before.txt" 2>&1
mcc -O2 "${ROOT_DIR}/musa_smoke.mu" "-L${MUSA_LIB_DIR}" -lmusart -o "${OUT_BIN}" > "${BUILD_DIR}/p0_musa_smoke_mcc.log" 2>&1
sha256sum "${ROOT_DIR}/musa_smoke.mu" "${OUT_BIN}" > "${BUILD_DIR}/p0_musa_smoke_sha256.txt"
"${OUT_BIN}" > "${RUN_DIR}/p0_musa_smoke_stdout.log" 2> "${RUN_DIR}/p0_musa_smoke_stderr.log"
mthreads-gmi > "${RUN_DIR}/p0_musa_smoke_gpu_after.txt" 2>&1
