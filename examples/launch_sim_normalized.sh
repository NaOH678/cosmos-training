#!/usr/bin/env bash
# Usage: OUTPUT_ROOT=... bash examples/launch_sim_normalized.sh 21_v3|27_ball|<task> wam|nonlocal|local|pointfk
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?select a task, e.g. 21_v3 or 27_ball}"
VARIANT="${2:?select wam, nonlocal or local}"
[[ "$TASK" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid task: $TASK" >&2; exit 2; }
case "$TASK" in 21) TASK=21_v3 ;; 27) TASK=27_ball ;; esac
[[ "$VARIANT" != pointfk ]] || VARIANT=local
case "$VARIANT" in wam|nonlocal|local) ;; *) echo "Unknown variant: $VARIANT" >&2; exit 2;; esac
export SIM_POINTFK_VARIANT="$VARIANT"
export SIM_POINTFK_BUNDLE="${SIM_POINTFK_BUNDLE:-/data/shichaojian/sim_data/bench2dex_task${TASK}_training}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the run directory}"
export OUTPUT_ROOT
JOB_NAME="action_policy_sim_${TASK}_${VARIANT}"
export POINTFLOW_SONATA_STAGE=2
export POINTFLOW_FREEZE_SONATA=false
export REAL_ACTION_VARIANT='' REAL_ACTION_STATS_PATH='' REAL_ACTION_STATS_SHA256=''
export POINTFLOW_DISPLACEMENT_FRAME_SCALES='' POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=''
export LOGURU_COLORIZE="${LOGURU_COLORIZE:-true}" PYTHONUNBUFFERED=1
export LOG_FILENAME="launcher_$(date -u +%Y%m%dT%H%M%S)_$$.log"
export EXTRA_TAIL_OVERRIDES="job.name=$JOB_NAME trainer.max_iter=20000 trainer.run_validation=true trainer.validation_iter=500 checkpoint.save_iter=500 ${EXTRA_TAIL_OVERRIDES:-}"
exec bash "$ROOT/examples/launch_sim_pointfk_v2.sh"
