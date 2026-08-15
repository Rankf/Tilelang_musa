#!/bin/sh
set -eu

export MUSA_HOME=/usr/local/musa-4.3.5
export LD_LIBRARY_PATH=/root/s5000-calibration/tilelang-musa/build/lib:/usr/local/musa-4.3.5/lib:/usr/local/musa-4.3.5/lib64:/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
export PYTHONPATH=/root/s5000-calibration:/root/s5000-calibration/tilelang-musa

exec /root/s5000-calibration/.venv-tilelang-musa/bin/python \
  /root/s5000-calibration/benchmarks/compiler_info_elementwise_runtime.py "$@"
