#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p6_holdout/${STAMP}"
SCOPE="${ROOT_DIR}/configs/p6_holdout_scope.json"
COMPUTE_CONFIG="${ROOT_DIR}/configs/p6_compute_holdout.json"
LAYOUT_CONFIG="${ROOT_DIR}/configs/p6_layout_holdout.json"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"
cp "${SCOPE}" "${COMPUTE_CONFIG}" "${LAYOUT_CONFIG}" "${RUN_ROOT}/"
"${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/scripts/p6_preflight.py" \
  --scope "${SCOPE}" --output "${RUN_ROOT}/preflight.json" | tee "${RUN_ROOT}/preflight.log"
mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true
for process_index in 1 2 3; do
  COMPUTE_OUT="${RUN_ROOT}/compute/process_${process_index}"
  COMPUTE_DUMP="${ROOT_DIR}/logs/build/p6_holdout/${STAMP}/compute/process_${process_index}"
  mkdir -p "${COMPUTE_OUT}" "${COMPUTE_DUMP}"
  export TILELANG_DUMP_KERNEL_SOURCE="${COMPUTE_DUMP}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" \
    --config "${COMPUTE_CONFIG}" --output-dir "${COMPUTE_OUT}" 2>&1 | tee "${COMPUTE_OUT}/run.log"
  LAYOUT_OUT="${RUN_ROOT}/layout/process_${process_index}"
  LAYOUT_DUMP="${ROOT_DIR}/logs/build/p6_holdout/${STAMP}/layout/process_${process_index}"
  mkdir -p "${LAYOUT_OUT}" "${LAYOUT_DUMP}"
  export TILELANG_DUMP_KERNEL_SOURCE="${LAYOUT_DUMP}"
  "${ROOT_DIR}/.venv-tilelang-musa/bin/python" "${ROOT_DIR}/benchmarks/p3_tilelang_global_memory.py" \
    --config "${LAYOUT_CONFIG}" --output-dir "${LAYOUT_OUT}" --kernel-dump-dir "${LAYOUT_DUMP}" \
    2>&1 | tee "${LAYOUT_OUT}/run.log"
done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true
printf '%s\n' "${RUN_ROOT}"
