#!/usr/bin/env bash
set -u -o pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_ROOT="${ROOT_DIR}/results/raw/p7_1_recovery_gate/${STAMP}"
mkdir -p "${RUN_ROOT}"
cp "${BASH_SOURCE[0]}" "${RUN_ROOT}/p7_1_recovery_gate.sh"

gate_ok=1
for index in 1 2 3; do
  # Process-monitoring is unavailable on this physical S5000/driver combination.
  # Use only documented device-level queries here; process cleanliness is checked
  # below through /proc, rather than treating an unsupported gmi capability as a
  # runtime-health failure.
  mthreads-gmi -cf > "${RUN_ROOT}/mthreads_gmi_concise_${index}.txt" 2>&1
  mthreads-gmi -q --json -i 0 > "${RUN_ROOT}/mthreads_gmi_json_${index}.json" 2>&1
  if ! python3 - "${RUN_ROOT}/mthreads_gmi_json_${index}.json" <<'PY'
import json
import sys

try:
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    gpu = report["GPU"][0]
    ok = (
        report["Attached GPUs"] == "1"
        and gpu["Index"] == "0"
        and gpu["Product Name"] == "MTT S5000"
        and gpu["PCI"]["Bus ID"] == "00000000:03:00.0"
        and gpu["FB Memory Usage"]["Total"] == "81920MiB"
        and gpu["FB Memory Usage"]["Used"] == "0MiB"
        and gpu["Utilization"]["Gpu"] == "0%"
    )
except (OSError, ValueError, KeyError, IndexError, TypeError):
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    gate_ok=0
  fi
done
ps -eo pid,ppid,stat,etime,cmd > "${RUN_ROOT}/process_snapshot.txt"
if rg -q '[[:space:]](p2_|p7_)[^[:space:]]*' "${RUN_ROOT}/process_snapshot.txt"; then
  gate_ok=0
fi

# A direct OS-level check catches processes that actually retain the GPU device
# or MUSA runtime, independently of mthreads-gmi's unsupported process monitor.
for fd in /proc/[0-9]*/fd/*; do
  if [[ "$(readlink "${fd}" 2>/dev/null || true)" == "/dev/mtgpu.0" ]]; then
    gate_ok=0
    break
  fi
done
for maps in /proc/[0-9]*/maps; do
  if rg -q 'libmusa|libmusart|libmugraph' "${maps}" 2>/dev/null; then
    gate_ok=0
    break
  fi
done

if [[ "${gate_ok}" -eq 1 ]]; then
  printf '{"phase":"P7.1","status":"pass","reason":"three device-level mthreads-gmi JSON queries validated; no residual P2/P7 process, open mtgpu device, or mapped MUSA runtime found"}\n' > "${RUN_ROOT}/gate_status.json"
  printf '%s\n' "${RUN_ROOT}"
  exit 0
fi
printf '{"phase":"P7.1","status":"blocked","reason":"device-level recovery gate failed or a residual GPU/MUSA process was found; do not launch GPU kernel; inspect snapshots"}\n' > "${RUN_ROOT}/gate_status.json"
printf '%s\n' "${RUN_ROOT}"
exit 2
