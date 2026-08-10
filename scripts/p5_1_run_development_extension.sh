#!/usr/bin/env bash
# Additional development coverage approved before P5 v2 fitting.  It uses no
# P6/P6-v2 data and does not alter either hold-out configuration.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p5_1_transfer_development_extension/${STAMP}"
CONFIG="${ROOT_DIR}/configs/p5_1_transfer_development_extension_compute.json"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}"
cp "${CONFIG}" "${RUN_ROOT}/"
mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true
for process_index in 1 2 3; do
  out="${RUN_ROOT}/compute/process_${process_index}"
  dump="${ROOT_DIR}/logs/build/p5_1_transfer_development_extension/${STAMP}/compute/process_${process_index}"
  mkdir -p "${out}" "${dump}"
  export TILELANG_DUMP_KERNEL_SOURCE="${dump}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" \
    --config "${CONFIG}" --output-dir "${out}" 2>&1 | tee "${out}/run.log"
done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true
printf '%s\n' "${RUN_ROOT}"
