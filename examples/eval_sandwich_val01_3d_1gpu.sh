#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export POINTFLOW_3D_SOURCE_JOB=/data/shichaojian/runs/pointflow_fk_localrope_qkvo_sac_16gpu_20261002/cosmos3_action/action_sft/action_policy_fk_point_singlerighthand_edge
export POINTFLOW_3D_CASE=val_01
export BASE_CHECKPOINT_PATH="$POINTFLOW_3D_SOURCE_JOB/checkpoints/iter_000010000"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shichaojian/sandwich_e6b_step10000_val01_3d_20261006}"
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Use a fresh OUTPUT_ROOT: $OUTPUT_ROOT already exists" >&2
    exit 1
fi
[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
export NNODES=1 NPROC_PER_NODE=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29631}"
export POINTFLOW_FK_LOCAL_ROPE=true FK_PROJECT_ANCHORS=true POINTFLOW_FK_UNIFIED_EVAL=true
export POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128
export POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_GEOMETRY_MOTION_FUSION=false
export POINTFLOW_DISPLACEMENT_SCALE=0.0528 FK_DISPLACEMENT_SCALE=0.083745
export EXTRA_TAIL_OVERRIDES="checkpoint.load_path=$BASE_CHECKPOINT_PATH checkpoint.load_training_state=false checkpoint.strict_resume=true trainer.run_validation=true trainer.run_validation_on_start=true trainer.max_iter=1 trainer.max_val_iter=1 model.config.compile.enabled=false dataloader_train.max_samples_per_batch=1 dataloader_val.max_samples_per_batch=1 trainer.callbacks.pointflow_eval._target_=cosmos_framework.callbacks.pointflow_fk_single_case_3d.SingleCase3DEval"
exec bash "$ROOT/examples/launch_pointflow_fk_sandwich101.sh"
