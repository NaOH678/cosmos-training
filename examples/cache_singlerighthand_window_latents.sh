#!/usr/bin/env bash
# Real-data adapter: existing RGB cache -> independent training-window latents.
# Usage: bash <this-script> sandwich|dropper|dropper_mix [Python tool arguments...]
# Override VIDEO_CACHE_ROOT, ALLOWLIST, OUTPUT_ROOT or WAN_VAE_PATH as needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${1:-sandwich}"
case "$DATA" in
  sandwich)
    CACHE=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
    LIST="$ROOT/examples/pointflow_sandwich924_all_101_episodes.txt" ;;
  dropper)
    CACHE=/data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache
    LIST="$ROOT/examples/pointflow_dropper_all_101_episodes.txt" ;;
  dropper_mix)
    CACHE=/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache
    LIST="$ROOT/examples/pointflow_dropper_dagger_mix_71_episodes.txt" ;;
  -h|--help)
    echo "Usage: bash $0 {sandwich|dropper|dropper_mix} [tool arguments...]"
    echo "Env: VIDEO_CACHE_ROOT ALLOWLIST OUTPUT_ROOT WAN_VAE_PATH PYTHON_BIN"
    echo "     WORKERS=8 DEVICES=0,1,2,3,4,5,6,7 BATCH_SIZE=1 OVERWRITE=0"
    exit 0 ;;
  *) echo "Unknown dataset: $DATA (choose sandwich, dropper or dropper_mix)" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
cd "$ROOT"
export LD_LIBRARY_PATH='' PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
: "${PYTHON_BIN:=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python}"
: "${VIDEO_CACHE_ROOT:=$CACHE}"
: "${ALLOWLIST:=$LIST}"
: "${WAN_VAE_PATH:=/data/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=$VIDEO_CACHE_ROOT/vae_window_latents}"
: "${WORKERS:=8}"
: "${DEVICES:=0,1,2,3,4,5,6,7}"
: "${BATCH_SIZE:=1}"
: "${OVERWRITE:=0}"

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$ALLOWLIST" ]] || { echo "missing allowlist: $ALLOWLIST" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "missing VAE: $WAN_VAE_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.writer.lock"
flock -n 9 || { echo "Another cache writer is using $OUTPUT_ROOT" >&2; exit 1; }
LOG="$OUTPUT_ROOT/logs/convert_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
extra=()
[[ "$OVERWRITE" == 1 ]] && extra+=(--overwrite)
"$PYTHON_BIN" -m tools.cache_window_vae_latents \
  --cache-root "$VIDEO_CACHE_ROOT" --vae-path "$WAN_VAE_PATH" \
  --episode-allowlist "$ALLOWLIST" --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" --devices "$DEVICES" --batch-size "$BATCH_SIZE" \
  "${extra[@]}" "$@" 2>&1 | tee "$LOG"
