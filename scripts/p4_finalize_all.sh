#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python3 scripts/p4_finalize_extensions.py
python3 scripts/p4_finalize_pipeline.py
python3 scripts/p4_finalize_compute_wave.py
python3 scripts/p4_finalize_occupancy_wave.py
