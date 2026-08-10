#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_STAMP="20260729T030002304541927Z"
SOURCE_DIR="${ROOT_DIR}/logs/build/p4_tilelang_pipeline/${PIPELINE_STAMP}/process_1"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="${ROOT_DIR}/results/raw/p4_tilelang_resource/${STAMP}"
BUILD_DIR="${ROOT_DIR}/logs/build/p4_tilelang_resource/${STAMP}"
BIN="${ROOT_DIR}/results/derived/p4_musa_mubin_resource_probe"
mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "$(dirname "${BIN}")"

g++ -O2 -std=c++17 -I/usr/local/musa-4.3.5/include \
  "${ROOT_DIR}/benchmarks/p4_musa_mubin_resource_probe.cpp" \
  -L/usr/lib/x86_64-linux-gnu -lmusa -o "${BIN}" \
  > "${BUILD_DIR}/probe_build.log" 2>&1
sha256sum "${ROOT_DIR}/benchmarks/p4_musa_mubin_resource_probe.cpp" "${BIN}" \
  > "${BUILD_DIR}/probe_sha256.txt"

printf 'process_index,stage,source_mubin,dynamic_shared_bytes,max_threads_per_block,static_shared_bytes,local_bytes_per_thread,registers_per_thread,active_cta_per_mp,status,error\n' > "${RUN_DIR}/raw.csv"
for process_index in 1 2 3; do
  for stage in 1 2 3 4; do
    index=$((stage - 1))
    source="${SOURCE_DIR}/tilelang_musa_kernel_000${index}.mu"
    original_mubin="${SOURCE_DIR}/tilelang_musa_kernel_000${index}.mubin"
    dynamic_bytes=$((stage * 16384))
    resource_log="${BUILD_DIR}/stage_${stage}_resource_usage.log"
    resource_mubin="${BUILD_DIR}/stage_${stage}_resource_usage.mubin"
    if [[ "${process_index}" == 1 ]]; then
      /usr/local/musa-4.3.5/bin/mcc --cuda-device-only --cuda-gpu-arch=mp_31 -std=c++17 \
        -Wno-error=address-of-temporary -I"${ROOT_DIR}/tilelang-musa/src" \
        -I"${ROOT_DIR}/tilelang-musa/3rdparty/mutlass/include" -D__MUSA_ARCH_LIST__=310 \
        -mllvm -mtgpu-alloc-shared-memory-from-zero=1 -resource-usage \
        -o "${resource_mubin}" "${source}" > "${resource_log}" 2>&1
      sha256sum "${source}" "${resource_mubin}" > "${BUILD_DIR}/stage_${stage}_sha256.txt"
      if [[ -f "${original_mubin}" ]]; then
        sha256sum "${original_mubin}" >> "${BUILD_DIR}/stage_${stage}_sha256.txt"
      else
        printf 'formal_dump_mubin=not_retained; queried_mubin=recompiled_from_formal_source_and_command\n' >> "${BUILD_DIR}/stage_${stage}_sha256.txt"
      fi
    fi
    output="$(${BIN} "${resource_mubin}" main_kernel 256 "${dynamic_bytes}" 2>&1)" || {
      escaped="${output//$'\n'/ | }"
      printf '%s,%s,%s,%s,,,,,,failed,%q\n' "${process_index}" "${stage}" "${resource_mubin}" "${dynamic_bytes}" "${escaped}" >> "${RUN_DIR}/raw.csv"
      continue
    }
    value() { sed -n "s/^$1=//p" <<<"${output}"; }
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,pass,\n' \
      "${process_index}" "${stage}" "${resource_mubin}" "${dynamic_bytes}" \
      "$(value max_threads_per_block)" "$(value static_shared_bytes)" \
      "$(value local_bytes_per_thread)" "$(value registers_per_thread)" \
      "$(value active_cta_per_mp)" >> "${RUN_DIR}/raw.csv"
  done
done
cp "${ROOT_DIR}/configs/p4_tilelang_pipeline.json" "${RUN_DIR}/pipeline_config.json"
printf '%s\n' "${RUN_DIR}"
