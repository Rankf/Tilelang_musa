#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/p7_1_runtime_ab.json"
PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_1_runtime_ab/${STAMP}"
BUILD_ROOT="${ROOT_DIR}/logs/build/p7_1_runtime_ab/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}" "${BUILD_ROOT}"
cp "${CONFIG}" "${RUN_ROOT}/"
cp "${ROOT_DIR}/benchmarks/p7_1_runtime_ab.py" "${RUN_ROOT}/"
cp "${BASH_SOURCE[0]}" "${RUN_ROOT}/p7_1_run_runtime_ab.sh"
GATE_RUN="$(bash "${ROOT_DIR}/scripts/p7_1_recovery_gate.sh")"
GATE_STATUS=$?
printf '%s\n' "${GATE_RUN}" > "${RUN_ROOT}/recovery_gate_run.txt"
if [[ "${GATE_STATUS}" -ne 0 ]]; then
  printf '{"phase":"P7.1","status":"blocked","reason":"formal recovery gate failed; no GPU kernel was started"}\n' > "${RUN_ROOT}/controller_status.json"
  printf '%s\n' "${RUN_ROOT}"
  exit 2
fi
mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" 2>&1

mapfile -t VARIANTS < <("${PYTHON}" -c "import json; print(*(x['id'] for x in json.load(open('${CONFIG}'))['variants']), sep='\\n')")
TRIALS="$("${PYTHON}" -c "import json; print(json.load(open('${CONFIG}'))['trials_per_variant'])")"
for variant in "${VARIANTS[@]}"; do
  for trial in $(seq 1 "${TRIALS}"); do
    OUT_DIR="${RUN_ROOT}/${variant}/trial_${trial}"
    DUMP_DIR="${BUILD_ROOT}/${variant}/trial_${trial}"
    mkdir -p "${OUT_DIR}" "${DUMP_DIR}"
    export TILELANG_DUMP_KERNEL_SOURCE="${DUMP_DIR}"
    timeout --signal=TERM --kill-after=10s 180s "${PYTHON}" "${ROOT_DIR}/benchmarks/p7_1_runtime_ab.py" \
      --config "${CONFIG}" --variant "${variant}" --trial "${trial}" --output "${OUT_DIR}/result.json" 2>&1 | tee "${OUT_DIR}/run.log"
    command_status=${PIPESTATUS[0]}
    if [[ ! -f "${OUT_DIR}/result.json" ]]; then
      printf '{"phase":"P7.1","status":"runtime_failed","failure_stage":"controller_timeout_or_abnormal_exit","error":"runner exit code %s; no result.json"}\n' "${command_status}" > "${OUT_DIR}/result.json"
    fi
  done
done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" 2>&1
printf '%s\n' "${RUN_ROOT}"
