#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_STAMP="20260729T033936508084786Z"
SOURCE="${ROOT_DIR}/logs/build/p4_tilelang_compute_wave/${FORMAL_STAMP}/process_1/tilelang_musa_kernel_0000.mu"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="${ROOT_DIR}/results/raw/p4_tilelang_compute_wave_resource/${STAMP}"
BUILD_DIR="${ROOT_DIR}/logs/build/p4_tilelang_compute_wave_resource/${STAMP}"
MUBIN="${BUILD_DIR}/fixed_grid_compute_resource_usage.mubin"
PROBE="${ROOT_DIR}/results/derived/p4_musa_mubin_resource_probe"
mkdir -p "${RUN_DIR}" "${BUILD_DIR}"
/usr/local/musa-4.3.5/bin/mcc --cuda-device-only --cuda-gpu-arch=mp_31 -std=c++17 \
  -Wno-error=address-of-temporary -I"${ROOT_DIR}/tilelang-musa/src" \
  -I"${ROOT_DIR}/tilelang-musa/3rdparty/mutlass/include" -D__MUSA_ARCH_LIST__=310 \
  -mllvm -mtgpu-alloc-shared-memory-from-zero=1 -resource-usage \
  -o "${MUBIN}" "${SOURCE}" > "${BUILD_DIR}/mcc_resource_usage.log" 2>&1
sha256sum "${SOURCE}" "${MUBIN}" > "${BUILD_DIR}/sha256.txt"
printf 'query_index,source_sha256,dynamic_shared_bytes,max_threads_per_block,static_shared_bytes,local_bytes_per_thread,registers_per_thread,active_cta_per_mp,status\n' > "${RUN_DIR}/raw.csv"
source_sha="$(sha256sum "${SOURCE}" | awk '{print $1}')"
for query_index in 1 2 3; do
  output="$("${PROBE}" "${MUBIN}" fixed_grid_compute_kernel 256 0)"
  value() { sed -n "s/^$1=//p" <<<"${output}"; }
  printf '%s,%s,0,%s,%s,%s,%s,%s,pass\n' \
    "${query_index}" "${source_sha}" "$(value max_threads_per_block)" \
    "$(value static_shared_bytes)" "$(value local_bytes_per_thread)" \
    "$(value registers_per_thread)" "$(value active_cta_per_mp)" >> "${RUN_DIR}/raw.csv"
done
printf '%s\n' "${RUN_DIR}"
