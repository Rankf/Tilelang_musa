#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TILELANG_ROOT="${ROOT_DIR}/tilelang-musa"
TILELANG_PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
EXPECTED_COMMIT="4d666968a005846314e9a84d8ff121c5dbffa9ee"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="${ROOT_DIR}/results/raw/p1/${RUN_STAMP}"
LOG_FILE="${ROOT_DIR}/logs/run/p1_measurement_smoke_${RUN_STAMP}.log"
DUMP_DIR="${ROOT_DIR}/logs/build/p1_tilelang_kernel/${RUN_STAMP}"

test -x "${TILELANG_PYTHON}"
test "$(git -C "${TILELANG_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
mkdir -p "${RUN_DIR}" "${DUMP_DIR}" "$(dirname "${LOG_FILE}")"

export MUSA_HOME="/usr/local/musa-4.3.5"
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TILELANG_DUMP_KERNEL_SOURCE="${DUMP_DIR}"

"${ROOT_DIR}/scripts/p0_capture_environment.sh"
"${TILELANG_PYTHON}" "${ROOT_DIR}/benchmarks/p1_tilelang_elementwise_add.py" \
  --config "${ROOT_DIR}/configs/p1_elementwise_add_smoke.json" \
  --raw-csv "${RUN_DIR}/raw_samples.csv" \
  --summary-csv "${RUN_DIR}/summary.csv" \
  --metadata-json "${RUN_DIR}/metadata.json" \
  --gpu-before "${RUN_DIR}/gpu_before.txt" \
  --gpu-after "${RUN_DIR}/gpu_after.txt" \
  --kernel-dump-dir "${DUMP_DIR}" \
  --runner-log "${LOG_FILE}" \
  2>&1 | tee "${LOG_FILE}"
