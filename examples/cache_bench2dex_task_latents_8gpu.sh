#!/usr/bin/env bash
# Foreground, resumable task21-style RGB/VAE preparation for four Bench2Dex tasks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  80) SHORT=task80_desk ;;
  07) SHORT=task07_citrus ;;
  06) SHORT=task06_fruit ;;
  27) SHORT=task27_ball ;;
  *) echo "Usage: bash $0 {80|07|06|27}" >&2; exit 2 ;;
esac
TASK="$1"
shift
cd "$ROOT"
export LD_LIBRARY_PATH='' PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
OUT="/data/shichaojian/raw_data/bench2dex/$SHORT/cosmos-cache"
mkdir -p "$ROOT/pointflow_outputs/bench2dex_latent_jobs" "$OUT/logs"
# All four commands use the same GPUs: never overlap them in separate terminals.
exec 9>"$ROOT/pointflow_outputs/bench2dex_latent_jobs/eight_gpu.lock"
flock -n 9 || { echo "Another Bench2Dex 8-GPU cache job is running; run tasks sequentially." >&2; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8, "Need eight visible CUDA GPUs"'
LOG="$OUT/logs/convert_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
"$PYTHON" -m tools.cache_bench2dex_multitask_latents \
  --task "$TASK" --output "$OUT" --devices 0,1,2,3,4,5,6,7 \
  --workers 8 --seed 42 --wait-downloads "$@" 2>&1 | tee "$LOG"
