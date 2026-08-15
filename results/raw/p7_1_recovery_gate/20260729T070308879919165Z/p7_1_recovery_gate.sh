#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_1_recovery_gate/${STAMP}"
mkdir -p "${RUN_ROOT}"
cp "${BASH_SOURCE[0]}" "${RUN_ROOT}/p7_1_recovery_gate.sh"

gate_ok=1
for index in 1 2 3; do
  mthreads-gmi > "${RUN_ROOT}/mthreads_gmi_${index}.txt" 2>&1
  if rg -q "Failed to query running processes" "${RUN_ROOT}/mthreads_gmi_${index}.txt"; then
    gate_ok=0
  fi
done
ps -eo pid,ppid,stat,etime,cmd > "${RUN_ROOT}/process_snapshot.txt"
if rg -q '[[:space:]](p2_|p7_)[^[:space:]]*' "${RUN_ROOT}/process_snapshot.txt"; then
  gate_ok=0
fi

if [[ "${gate_ok}" -eq 1 ]]; then
  printf '{"phase":"P7.1","status":"pass","reason":"three mthreads-gmi process-table queries succeeded; no residual P2/P7 process matched"}\n' > "${RUN_ROOT}/gate_status.json"
  printf '%s\n' "${RUN_ROOT}"
  exit 0
fi
printf '{"phase":"P7.1","status":"blocked","reason":"recovery gate failed; do not launch GPU kernel; inspect mthreads-gmi snapshots and process snapshot"}\n' > "${RUN_ROOT}/gate_status.json"
printf '%s\n' "${RUN_ROOT}"
exit 2
