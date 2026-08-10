#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_large_gemm_direct_batch_sweep/${STAMP}"; CONFIG="${ROOT_DIR}/configs/p7_large_gemm_direct_batch_sweep.json"; PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
export MUSA_HOME=/usr/local/musa-4.3.5; export PATH="${MUSA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_ROOT}"; cp "${CONFIG}" "${RUN_ROOT}/"; mthreads-gmi > "${RUN_ROOT}/gpu_before.txt" || true
mapfile -t UNROLLS < <("${PYTHON}" -c "import json; print(*json.load(open('${CONFIG}'))['batch_unrolls'],sep='\\n')")
TRIALS="$("${PYTHON}" -c "import json; print(json.load(open('${CONFIG}'))['trials_per_unroll'])")"
for unroll in "${UNROLLS[@]}"; do for trial in $(seq 1 "${TRIALS}"); do
  out="${RUN_ROOT}/unroll_${unroll}/trial_${trial}"; dump="${ROOT_DIR}/logs/build/p7_large_gemm_direct_batch_sweep/${STAMP}/unroll_${unroll}/trial_${trial}"
  mkdir -p "${out}" "${dump}"; export TILELANG_DUMP_KERNEL_SOURCE="${dump}"
  timeout --signal=TERM --kill-after=10s 120s "${PYTHON}" "${ROOT_DIR}/benchmarks/p7_large_gemm_direct_batch_diagnostic.py" --config "${CONFIG}" --batch-unroll "${unroll}" --trial "${trial}" --output "${out}/result.json" 2>&1 | tee "${out}/run.log"
  status=${PIPESTATUS[0]}; if [[ ! -f "${out}/result.json" ]]; then "${PYTHON}" "${ROOT_DIR}/scripts/p7_mark_runtime_timeout.py" --output "${out}/result.json" --case-id p7_fp16_gemm_1536x1024x1536 --trial "${trial}" --context "direct_batch_sweep_unroll_${unroll}" --timeout-seconds 120; fi
done; done
mthreads-gmi > "${RUN_ROOT}/gpu_after.txt" || true; printf '%s\n' "${RUN_ROOT}"
