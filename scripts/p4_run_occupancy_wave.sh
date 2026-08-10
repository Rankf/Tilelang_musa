#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p4_occupancy_wave/${STAMP}"
BIN="${ROOT_DIR}/results/derived/p4_musa_occupancy_wave"
mkdir -p "${RUN_ROOT}" "${ROOT_DIR}/logs/build" "${ROOT_DIR}/results/derived"

printf '%q ' mcc -O2 --cuda-gpu-arch=mp_31 "${ROOT_DIR}/benchmarks/p4_musa_occupancy_wave.mu" -L/usr/local/musa-4.3.5/lib -lmusart -o "${BIN}" > "${ROOT_DIR}/logs/build/p4_musa_occupancy_wave_mcc_command.txt"
echo >> "${ROOT_DIR}/logs/build/p4_musa_occupancy_wave_mcc_command.txt"
mcc -O2 --cuda-gpu-arch=mp_31 "${ROOT_DIR}/benchmarks/p4_musa_occupancy_wave.mu" -L/usr/local/musa-4.3.5/lib -lmusart -o "${BIN}" > "${ROOT_DIR}/logs/build/p4_musa_occupancy_wave_mcc.log" 2>&1
sha256sum "${ROOT_DIR}/benchmarks/p4_musa_occupancy_wave.mu" "${BIN}" > "${ROOT_DIR}/logs/build/p4_musa_occupancy_wave_sha256.txt"
cp "${ROOT_DIR}/configs/p4_occupancy_wave.json" "${RUN_ROOT}/config.json"

for process_index in 1 2 3; do
  mthreads-gmi > "${RUN_ROOT}/gpu_before_${process_index}.txt" 2>&1
  "${BIN}" "${RUN_ROOT}/run_${process_index}.csv" > "${RUN_ROOT}/stdout_${process_index}.log" 2> "${RUN_ROOT}/stderr_${process_index}.log"
  mthreads-gmi > "${RUN_ROOT}/gpu_after_${process_index}.txt" 2>&1
done
printf '%s\n' "${RUN_ROOT}"
