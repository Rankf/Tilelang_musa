#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TILELANG_ROOT="${ROOT_DIR}/tilelang-musa"
TILELANG_PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
DUMP_DIR="${ROOT_DIR}/logs/build/p0_tilelang_kernel"
RUN_LOG="${ROOT_DIR}/logs/run/p0_tilelang_add_global_repro.log"
EXPECTED_COMMIT="4d666968a005846314e9a84d8ff121c5dbffa9ee"

test -x "${TILELANG_PYTHON}"
test -d "${TILELANG_ROOT}/.git"
test "$(git -C "${TILELANG_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
mkdir -p "${DUMP_DIR}" "$(dirname "${RUN_LOG}")"

export MUSA_HOME="/usr/local/musa-4.3.5"
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TILELANG_DUMP_KERNEL_SOURCE="${DUMP_DIR}"

cd "${TILELANG_ROOT}"
"${TILELANG_PYTHON}" testing/musa/common/test_add_global.py 2>&1 | tee "${RUN_LOG}"
