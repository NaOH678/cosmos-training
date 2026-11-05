#!/usr/bin/env bash
set -euo pipefail
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk
export LD_LIBRARY_PATH=''
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=third_party/fast_foundation_runtime:third_party/foundation_runtime
export CPATH=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/third_party/fast_foundation_sysroot/usr/include/python3.10:/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/third_party/fast_foundation_sysroot/usr/include
export TORCHINDUCTOR_CACHE_DIR=/tmp/fast_foundation_inductor
export TRITON_CACHE_DIR=/tmp/fast_foundation_triton
exec /data/shichaojian/Track4World_portable/Track4World/.venv/bin/python -u tools/run_fast_foundation_v4.py "$@"
