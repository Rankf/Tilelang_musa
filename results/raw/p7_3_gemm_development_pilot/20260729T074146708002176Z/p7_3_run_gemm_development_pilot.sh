#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/p7_3_gemm_development_pilot_v1.json"
PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_3_gemm_development_pilot/${STAMP}"
BUILD_ROOT="${ROOT_DIR}/logs/build/p7_3_gemm_development_pilot/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}" "${BUILD_ROOT}"
cp "${CONFIG}" "${RUN_ROOT}/"; cp "${ROOT_DIR}/benchmarks/p7_3_gemm_development_pilot.py" "${RUN_ROOT}/"; cp "${BASH_SOURCE[0]}" "${RUN_ROOT}/"
GATE_RUN="$(bash "${ROOT_DIR}/scripts/p7_1_recovery_gate.sh")"; GATE_STATUS=$?
printf '%s\n' "${GATE_RUN}" > "${RUN_ROOT}/recovery_gate_run.txt"
if [[ "${GATE_STATUS}" -ne 0 ]]; then printf '{"phase":"P7.3","status":"blocked","reason":"P7.1 recovery gate failed; no kernel started"}\n' > "${RUN_ROOT}/controller_status.json"; printf '%s\n' "${RUN_ROOT}"; exit 2; fi
mthreads-gmi -cf > "${RUN_ROOT}/gpu_before.txt" 2>&1
mapfile -t CASES < <("${PYTHON}" -c "import json; print(*(x['id'] for x in json.load(open('${CONFIG}'))['cases']), sep='\\n')")
mapfile -t SCHEDULES < <("${PYTHON}" -c "import json; print(*(x['id'] for x in json.load(open('${CONFIG}'))['schedules']), sep='\\n')")
TRIALS="$("${PYTHON}" -c "import json; print(json.load(open('${CONFIG}'))['timing']['trials_per_schedule'])")"
for case_id in "${CASES[@]}"; do for schedule_id in "${SCHEDULES[@]}"; do for trial in $(seq 1 "${TRIALS}"); do
  OUT="${RUN_ROOT}/${case_id}/${schedule_id}/trial_${trial}"; DUMP="${BUILD_ROOT}/${case_id}/${schedule_id}/trial_${trial}"; mkdir -p "${OUT}" "${DUMP}"; export TILELANG_DUMP_KERNEL_SOURCE="${DUMP}"
  timeout --signal=TERM --kill-after=10s 300s "${PYTHON}" "${ROOT_DIR}/benchmarks/p7_3_gemm_development_pilot.py" --config "${CONFIG}" --case "${case_id}" --schedule "${schedule_id}" --trial "${trial}" --output "${OUT}/result.json" 2>&1 | tee "${OUT}/run.log"
  rc=${PIPESTATUS[0]}; if [[ ! -f "${OUT}/result.json" ]]; then printf '{"phase":"P7.3","status":"runtime_failed","failure_stage":"controller_timeout_or_abnormal_exit","error":"runner exit code %s; no result.json"}\n' "${rc}" > "${OUT}/result.json"; fi
done; done; done
mthreads-gmi -cf > "${RUN_ROOT}/gpu_after.txt" 2>&1; printf '%s\n' "${RUN_ROOT}"
