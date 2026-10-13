#!/usr/bin/env bash
# V3 persistent tracks, 94/6 split, baseline action statistics; established v2 model recipe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SIM_POINTFK_BUNDLE=/data/shichaojian/sim_data/bench2dex_task21_v3_training
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shichaojian/runs/sim_v3_94train_6val_handguided1024_stage2_8gpu_20k}"
export POINTFLOW_SONATA_STAGE=2
export POINTFLOW_FREEZE_SONATA=false
export REAL_ACTION_VARIANT='' REAL_ACTION_STATS_PATH='' REAL_ACTION_STATS_SHA256=''
export POINTFLOW_DISPLACEMENT_FRAME_SCALES='' POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=''
export LOGURU_COLORIZE=true PYTHONUNBUFFERED=1
export LOG_FILENAME="launcher_$(date -u +%Y%m%dT%H%M%S)_$$.log"
export EXTRA_TAIL_OVERRIDES="job.name=action_policy_sim_pointfk_v3 trainer.max_iter=20000 trainer.run_validation=true trainer.validation_iter=500 checkpoint.save_iter=500 ${EXTRA_TAIL_OVERRIDES:-}"
exec bash "$ROOT/examples/launch_sim_pointfk_v2.sh"
