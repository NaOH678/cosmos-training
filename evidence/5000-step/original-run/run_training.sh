#!/usr/bin/env bash
set -Eeuo pipefail
RUN_DIR="/mnt/lvminrui/workspace/mrlv/wam-910b-8x-5000-20261105T131003Z"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
date -u +%FT%TZ > "$RUN_DIR/started_at.txt"
finish() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
    date -u +%FT%TZ > "$RUN_DIR/finished_at.txt"
    if (( code != 0 )); then printf 'failed\n' > "$RUN_DIR/status.txt"; fi
    exit "$code"
}
trap finish EXIT
printf 'preflight\n' > "$RUN_DIR/status.txt"
source "$RUN_DIR/training.env"
[[ ! -e "$OUTPUT_ROOT" ]] || { printf 'Fresh training requires an unused output: %s\n' "$OUTPUT_ROOT" >&2; exit 1; }
bash run.sh --check-inputs > "$RUN_DIR/input-check.log" 2>&1
cp outputs/input_check.json "$RUN_DIR/input_check.json"
"$TRAIN_PYTHON" -m mrlv.npu.preflight --expected-devices 8 --report "$RUN_DIR/preflight.json" > "$RUN_DIR/preflight.log" 2>&1
printf 'config_check\n' > "$RUN_DIR/status.txt"
OUTPUT_ROOT="$RUN_DIR/config-preview" bash cosmos-training/mrlv/launch_wam_litegen.sh --dry-run > "$RUN_DIR/config-preview.log" 2>&1
"$TRAIN_PYTHON" "$RUN_DIR/verify_config.py" "$RUN_DIR/config-preview" "$RUN_DIR/config-check.json"
printf 'training\n' > "$RUN_DIR/status.txt"
bash cosmos-training/mrlv/launch_wam_litegen.sh
printf 'comparing\n' > "$RUN_DIR/status.txt"
"$TRAIN_PYTHON" "$RUN_DIR/verify_config.py" "$OUTPUT_ROOT" "$RUN_DIR/final-config-check.json"
"$TRAIN_PYTHON" cosmos-training/mrlv/compare_wam_cross_device.py \
    --baseline reference/wam-task27-align5k-01 --candidate "$OUTPUT_ROOT" \
    --report "$RUN_DIR/alignment.json"
printf 'passed\n' > "$RUN_DIR/status.txt"
