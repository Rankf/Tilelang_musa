#!/usr/bin/env bash
# Isolates the original P2 compute harness to one large GEMM per fresh process.
# A per-process failure is recorded but must not suppress later independent trials.
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_p2_exact_single_gemm/${STAMP}"
CONFIG="${ROOT_DIR}/configs/p7_p2_exact_single_gemm.json"
PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}"
cp "${CONFIG}" "${RUN_ROOT}/"
mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true
for process_index in 1 2 3; do
  out="${RUN_ROOT}/process_${process_index}"
  dump="${ROOT_DIR}/logs/build/p7_p2_exact_single_gemm/${STAMP}/process_${process_index}"
  mkdir -p "${out}" "${dump}"
  export TILELANG_DUMP_KERNEL_SOURCE="${dump}"
  "${PYTHON}" "${ROOT_DIR}/benchmarks/p2_tilelang_compute_suite.py" \
    --config "${CONFIG}" --output-dir "${out}" 2>&1 | tee "${out}/run.log"
  status=${PIPESTATUS[0]}
  printf '%s\n' "${status}" > "${out}/process_exit_code.txt"
done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true
printf '%s\n' "${RUN_ROOT}"
