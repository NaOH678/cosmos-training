#!/usr/bin/env bash
# Platform executes once per node; use a prepared environment without installing.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE"
if [[ -r config/cluster.env ]]; then source config/cluster.env; fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:?Missing platform node count}}}"
export NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:?Missing platform node rank}}}"
: "${MASTER_ADDR:?Missing MASTER_ADDR}"
: "${MASTER_PORT:?Missing MASTER_PORT}"
[[ "${OUTPUT_ROOT:-}" = /* ]] || { printf 'Set OUTPUT_ROOT to a new absolute shared directory.\n' >&2; exit 1; }
printf 'Host=%s nodes=%s cards/node=%s node_rank=%s master=%s:%s\n' \
    "$(hostname)" "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
source scripts/activate.sh
REPORT_ROOT="$OUTPUT_ROOT/setup/$(hostname)"
mkdir -p "$REPORT_ROOT"
# CANN adds profiler distributions; check the venv without changing training paths.
env -u PYTHONPATH "$TRAIN_PYTHON" scripts/check_environment.py --require-ready \
    --lock environment/requirements.lock --report "$REPORT_ROOT/installed.json" \
    2>&1 | tee "$REPORT_ROOT/environment.log"
env -u PYTHONPATH "$TRAIN_PYTHON" -m pip check 2>&1 | tee "$REPORT_ROOT/pip-check.log"

DRY_RUN=0
for argument in "$@"; do
    [[ "$argument" != -- ]] || break
    if [[ "$argument" == --dry-run ]]; then DRY_RUN=1; fi
done
if [[ "$DRY_RUN" == 0 ]]; then
    "$TRAIN_PYTHON" -m migration.npu.preflight --expected-devices "$NPROC_PER_NODE" \
        --report "$REPORT_ROOT/preflight.json" 2>&1 | tee "$REPORT_ROOT/preflight.log"
fi
exec bash train.sh "$@"
