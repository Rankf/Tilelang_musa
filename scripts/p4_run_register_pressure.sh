#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="${ROOT_DIR}/results/raw/p4_register_pressure/${STAMP}"
BUILD_DIR="${ROOT_DIR}/logs/build/p4_register_pressure/${STAMP}"
BIN="${ROOT_DIR}/results/derived/p4_musa_register_pressure"
mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "$(dirname "${BIN}")"
mcc -O2 --cuda-gpu-arch=mp_31 -resource-usage "${ROOT_DIR}/benchmarks/p4_musa_register_pressure.mu" \
  -L/usr/local/musa-4.3.5/lib -lmusart -o "${BIN}" > "${BUILD_DIR}/mcc_resource_usage.log" 2>&1
sha256sum "${ROOT_DIR}/benchmarks/p4_musa_register_pressure.mu" "${BIN}" > "${BUILD_DIR}/sha256.txt"
for i in 1 2 3; do "${BIN}" "${RUN_DIR}/run_${i}.csv" > "${RUN_DIR}/stdout_${i}.log" 2> "${RUN_DIR}/stderr_${i}.log"; done
printf '%s\n' "${RUN_DIR}"
