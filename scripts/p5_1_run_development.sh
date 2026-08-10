#!/usr/bin/env bash
# Run the pre-registered P5.1 development set only.  This script never reads
# or runs either P6 hold-out directory.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p5_1_transfer_development/${STAMP}"
COMPUTE_CONFIG="${ROOT_DIR}/configs/p5_1_transfer_development_compute.json"
LAYOUT_CONFIG="${ROOT_DIR}/configs/p5_1_transfer_development_layout.json"

export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}"
cp "${COMPUTE_CONFIG}" "${LAYOUT_CONFIG}" "${RUN_ROOT}/"
mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true

for process_index in 1 2 3; do
  compute_out="${RUN_ROOT}/compute/process_${process_index}"
  compute_dump="${ROOT_DIR}/logs/build/p5_1_transfer_development/${STAMP}/compute/process_${process_index}"
  mkdir -p "${compute_out}" "${compute_dump}"
  export TILELANG_DUMP_KERNEL_SOURCE="${compute_dump}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" \
    --config "${COMPUTE_CONFIG}" --output-dir "${compute_out}" 2>&1 | tee "${compute_out}/run.log"

  layout_out="${RUN_ROOT}/layout/process_${process_index}"
  layout_dump="${ROOT_DIR}/logs/build/p5_1_transfer_development/${STAMP}/layout/process_${process_index}"
  mkdir -p "${layout_out}" "${layout_dump}"
  export TILELANG_DUMP_KERNEL_SOURCE="${layout_dump}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p3_tilelang_global_memory.py" \
    --config "${LAYOUT_CONFIG}" --output-dir "${layout_out}" --kernel-dump-dir "${layout_dump}" \
    2>&1 | tee "${layout_out}/run.log"
done

mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true
printf '%s\n' "${RUN_ROOT}"
