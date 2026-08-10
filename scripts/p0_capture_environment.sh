#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/logs/env"
TILESIGHT_ROOT="${ROOT_DIR}/DeepStack-ae-source/src/tilesight"
TILELANG_ROOT="${ROOT_DIR}/tilelang-musa"
TILELANG_PYTHON="${ROOT_DIR}/.venv-tilelang-musa/bin/python"
mkdir -p "${OUT_DIR}"

mthreads-gmi > "${OUT_DIR}/mthreads-gmi_before.txt" 2>&1
mcc --version > "${OUT_DIR}/mcc_version.txt" 2>&1
python3 --version > "${OUT_DIR}/python_version.txt" 2>&1
env | rg '^(MUSA|PATH|PYTHONPATH|LD_LIBRARY_PATH|TILELANG)' > "${OUT_DIR}/musa_environment.txt" || true
python3 -c 'import torch; print("torch=" + torch.__version__); print("torch_path=" + torch.__file__); import torch_musa; print("torch_musa=" + getattr(torch_musa, "__version__", "unknown")); print("torch_musa_path=" + torch_musa.__file__)' > "${OUT_DIR}/python_musa_packages.txt" 2>&1

if [ -x "${TILELANG_PYTHON}" ] && "${TILELANG_PYTHON}" -c 'import tilelang' > "${OUT_DIR}/tilelang_import.txt" 2>&1; then
  "${TILELANG_PYTHON}" -c 'import tilelang, torch, torch_musa; print("tilelang_path=" + tilelang.__file__); print("tilelang_version=" + getattr(tilelang, "__version__", "unknown")); print("torch=" + torch.__version__); print("torch_musa=" + getattr(torch_musa, "__version__", "unknown")); print("musa_available=" + str(torch.musa.is_available())); print("musa_capability=" + str(torch.musa.get_device_capability()))' >> "${OUT_DIR}/tilelang_import.txt" 2>&1
  printf 'available\n' > "${OUT_DIR}/tilelang_status.txt"
else
  printf 'unavailable\n' > "${OUT_DIR}/tilelang_status.txt"
fi

find /root -maxdepth 4 -type d -iname '*tilelang*' -print > "${OUT_DIR}/tilelang_worktrees.txt" 2>&1
if [ -d "${TILELANG_ROOT}/.git" ]; then
  git -C "${TILELANG_ROOT}" rev-parse HEAD > "${OUT_DIR}/tilelang_musa_commit.txt"
  git -C "${TILELANG_ROOT}" status --short > "${OUT_DIR}/tilelang_musa_status.txt"
  git -C "${TILELANG_ROOT}" submodule status --recursive > "${OUT_DIR}/tilelang_musa_submodules.txt"
  git -C "${TILELANG_ROOT}" diff --binary > "${OUT_DIR}/tilelang_musa_local.patch"
  sha256sum "${OUT_DIR}/tilelang_musa_local.patch" > "${OUT_DIR}/tilelang_musa_local_patch_sha256.txt"
fi
sha256sum "${ROOT_DIR}/musa_smoke.mu" "${ROOT_DIR}/musa_smoke" > "${OUT_DIR}/musa_smoke_sha256.txt"

if [ -f "${TILESIGHT_ROOT}/README.md" ]; then
  git -C "${ROOT_DIR}/DeepStack-ae-source" rev-parse HEAD > "${OUT_DIR}/deepstack_ae_commit.txt" 2>&1
  sha256sum "${TILESIGHT_ROOT}/README.md" > "${OUT_DIR}/tilesight_readme_sha256.txt"
  find "${TILESIGHT_ROOT}/tilesight" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "${OUT_DIR}/tilesight_source_areas.txt"
  env PYTHONPATH="${TILESIGHT_ROOT}" python3 -c 'import tilesight; print("tilesight_path=" + tilesight.__file__)' > "${OUT_DIR}/tilesight_source_import.txt" 2>&1 || true
else
  printf 'unavailable\n' > "${OUT_DIR}/deepstack_tilesight_source_status.txt"
fi

mthreads-gmi > "${OUT_DIR}/mthreads-gmi_after.txt" 2>&1
