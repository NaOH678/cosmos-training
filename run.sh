#!/usr/bin/env bash
# One task command, independent of checkout path and platform vendor.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -r "$BUNDLE/config/cluster.env" ]]; then source "$BUNDLE/config/cluster.env"; fi
export COSMOS_NPU_ENV="${COSMOS_NPU_ENV:-$BUNDLE/.venv-npu}"
export PYTHONNOUSERSITE=1
[[ "$COSMOS_NPU_ENV" = /* ]] || export COSMOS_NPU_ENV="$BUNDLE/$COSMOS_NPU_ENV"
export OSS_ROOT="${OSS_ROOT:-/oss}"
export EDGE_DROID_MODEL_PATH="${EDGE_DROID_MODEL_PATH:-$OSS_ROOT/models/cosmos3-edge-droid}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$OSS_ROOT/models/cosmos3-edge-droid-dcp}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
export SIM_POINTFK_BUNDLE="${SIM_POINTFK_BUNDLE:-$OSS_ROOT/datasets/sim_task27_ball_wam}"
PYTHON="${NPU_BOOTSTRAP_PYTHON:-python3.12}"
case "${1:-}" in
    --help|-h)
        printf 'Usage: bash run.sh [--verify|--check-inputs|--bootstrap-only|--checks-only|--report-root NEW_DIR]\nDefault: install locked environment, check inputs, then run NPU acceptance and 8-rank WAM two-step smoke.\n'
        exit 0 ;;
    --verify)
        exec "$PYTHON" "$BUNDLE/scripts/verify_bundle.py" ;;
    --check-inputs)
        shift
        exec "$PYTHON" "$BUNDLE/scripts/check_inputs.py" "$@" ;;
    --bootstrap-only)
        exec bash "$BUNDLE/scripts/bootstrap.sh" ;;
esac
bash "$BUNDLE/scripts/bootstrap.sh"
# checks-only covers kernels/tiny FSDP and does not require the 4B assets.
CHECKS_ONLY=0
for argument in "$@"; do
    if [[ "$argument" == --checks-only ]]; then CHECKS_ONLY=1; fi
done
if [[ "$CHECKS_ONLY" != 1 ]]; then
    "$COSMOS_NPU_ENV/bin/python" "$BUNDLE/scripts/check_inputs.py"
fi
exec bash "$BUNDLE/cosmos-training/mrlv/npu/submit_job.sh" "$@"

