#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_STAMP="20260729T025312594585659Z"
SOURCE_DIR="${ROOT_DIR}/logs/build/p4_tilelang_wave/${FORMAL_STAMP}/process_1"
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="${ROOT_DIR}/results/raw/p4_tilelang_wave_resource/${STAMP}"
BUILD_DIR="${ROOT_DIR}/logs/build/p4_tilelang_wave_resource/${STAMP}"
PROBE="${ROOT_DIR}/results/derived/p4_musa_mubin_resource_probe"
mkdir -p "${RUN_DIR}" "${BUILD_DIR}"
case_ids=(tl_block_64 tl_block_128 tl_block_256 tl_block_384 tl_block_512 tl_wave_1 tl_wave_719 tl_wave_720 tl_wave_721 tl_wave_1439 tl_wave_1440 tl_wave_1441)
threads=(64 128 256 384 512 256 256 256 256 256 256 256)
printf 'case_id,block_threads,dynamic_shared_bytes,max_threads_per_block,static_shared_bytes,local_bytes_per_thread,registers_per_thread,active_cta_per_mp,status\n' > "${RUN_DIR}/raw.csv"
for index in $(seq 0 11); do
  source=$(printf '%s/tilelang_musa_kernel_%04d.mu' "${SOURCE_DIR}" "${index}")
  mubin=$(printf '%s/case_%02d.mubin' "${BUILD_DIR}" "${index}")
  log=$(printf '%s/case_%02d_resource_usage.log' "${BUILD_DIR}" "${index}")
  /usr/local/musa-4.3.5/bin/mcc --cuda-device-only --cuda-gpu-arch=mp_31 -std=c++17 \
    -Wno-error=address-of-temporary -I"${ROOT_DIR}/tilelang-musa/src" \
    -I"${ROOT_DIR}/tilelang-musa/3rdparty/mutlass/include" -D__MUSA_ARCH_LIST__=310 \
    -mllvm -mtgpu-alloc-shared-memory-from-zero=1 -resource-usage -o "${mubin}" "${source}" > "${log}" 2>&1
  output="$(${PROBE} "${mubin}" fixed_grid_copy_kernel "${threads[$index]}" 0)"
  value() { sed -n "s/^$1=//p" <<<"${output}"; }
  printf '%s,%s,0,%s,%s,%s,%s,%s,pass\n' "${case_ids[$index]}" "${threads[$index]}" \
    "$(value max_threads_per_block)" "$(value static_shared_bytes)" "$(value local_bytes_per_thread)" \
    "$(value registers_per_thread)" "$(value active_cta_per_mp)" >> "${RUN_DIR}/raw.csv"
done
printf '%s\n' "${RUN_DIR}"
