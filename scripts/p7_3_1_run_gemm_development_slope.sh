#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/p7_3_1_gemm_development_slope_v1.json"
PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_3_1_gemm_development_slope/${STAMP}"
BUILD_ROOT="${ROOT_DIR}/logs/build/p7_3_1_gemm_development_slope/${STAMP}"
export MUSA_HOME=/usr/local/musa-4.3.5
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}" "${BUILD_ROOT}"
cp "${CONFIG}" "${RUN_ROOT}/"; cp "${ROOT_DIR}/benchmarks/p7_3_1_gemm_development_slope.py" "${ROOT_DIR}/benchmarks/p7_3_gemm_development_pilot.py" "${BASH_SOURCE[0]}" "${RUN_ROOT}/"
env | sort > "${RUN_ROOT}/environment.txt"; mcc --version > "${RUN_ROOT}/mcc_version.txt" 2>&1; "${PYTHON}" --version > "${RUN_ROOT}/python_version.txt" 2>&1
GATE_RUN="$(bash "${ROOT_DIR}/scripts/p7_1_recovery_gate.sh")"; GATE_STATUS=$?
printf '%s\n' "${GATE_RUN}" > "${RUN_ROOT}/recovery_gate_run.txt"
if [[ "${GATE_STATUS}" -ne 0 ]]; then printf '{"phase":"P7.3.1","status":"blocked","reason":"recovery gate failed; no kernel started"}\n' > "${RUN_ROOT}/controller_status.json"; printf '%s\n' "${RUN_ROOT}"; exit 2; fi
mthreads-gmi -q --json -i 0 > "${RUN_ROOT}/gpu_before.json" 2>&1
mapfile -t CASES < <("${PYTHON}" -c "import json; print(*(x['id'] for x in json.load(open('${CONFIG}'))['cases']), sep='\\n')")
mapfile -t SCHEDULES < <("${PYTHON}" -c "import json; print(*(x['id'] for x in json.load(open('${CONFIG}'))['schedules']), sep='\\n')")
TRIALS="$("${PYTHON}" -c "import json; print(json.load(open('${CONFIG}'))['timing']['trials_per_schedule'])")"
for case_id in "${CASES[@]}"; do for schedule_id in "${SCHEDULES[@]}"; do for trial in $(seq 1 "${TRIALS}"); do
  OUT="${RUN_ROOT}/${case_id}/${schedule_id}/trial_${trial}"; DUMP="${BUILD_ROOT}/${case_id}/${schedule_id}/trial_${trial}"; ART="${OUT}/mubin"; mkdir -p "${OUT}" "${DUMP}" "${ART}"; export TILELANG_DUMP_KERNEL_SOURCE="${DUMP}"
  timeout --signal=TERM --kill-after=10s 360s "${PYTHON}" "${ROOT_DIR}/benchmarks/p7_3_1_gemm_development_slope.py" --config "${CONFIG}" --case "${case_id}" --schedule "${schedule_id}" --trial "${trial}" --artifact-dir "${ART}" --output "${OUT}/result.json" 2>&1 | tee "${OUT}/run.log"
  rc=${PIPESTATUS[0]}; if [[ ! -f "${OUT}/result.json" ]]; then printf '{"phase":"P7.3.1","status":"runtime_failed","failure_stage":"controller_timeout_or_abnormal_exit","error":"runner exit code %s; no result.json"}\n' "${rc}" > "${OUT}/result.json"; fi
done; done; done
mthreads-gmi -q --json -i 0 > "${RUN_ROOT}/gpu_after.json" 2>&1; printf '%s\n' "${RUN_ROOT}"
