#!/usr/bin/env bash
# Match outputs/wam-task27-align5k-01; original launcher owns all data/numerical settings.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export COSMOS_TRAINING_BACKEND=litegen
export LITEGEN_DEVICE="${LITEGEN_DEVICE:-cuda}"
SMOKE=0
for arg in "$@"; do
    if [[ "$arg" == -- ]]; then break; fi
    if [[ "$arg" == --smoke ]]; then SMOKE=1; fi
done
if [[ "$SMOKE" == 1 ]]; then
    export MAX_ITER="${MAX_ITER:-2}" SAVE_ITER="${SAVE_ITER:-2}"
else
    export MAX_ITER="${MAX_ITER:-5000}" SAVE_ITER="${SAVE_ITER:-5000}"
fi
# Fail on an accidental restart: comparison must begin at the pretrained DCP.
# Set ALLOW_RESUME=1 explicitly for interruption recovery, with the same output directory.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/../outputs/wam-task27-litegen-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "${ALLOW_RESUME:-0}" != 1 && -d "$OUTPUT_ROOT" ]] &&
    [[ -n "$(find "$OUTPUT_ROOT" -name latest_checkpoint.txt -print -quit)" ]]; then
    printf 'Output contains a resumable checkpoint: %s (use a fresh OUTPUT_ROOT or ALLOW_RESUME=1)\n' "$OUTPUT_ROOT" >&2
    exit 1
fi
exec bash "$REPO_ROOT/mrlv/launch_wam_task.sh" "$@"
