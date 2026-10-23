#!/usr/bin/env bash
# Usage: bash examples/launch_real_normalized.sh sandwich|dropper_mix wam|nonlocal|local
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${1:?select sandwich or dropper_mix}"
VARIANT="${2:?select wam, nonlocal or local}"
export REAL_ACTION_VARIANT="$VARIANT"
case "$VARIANT" in wam|nonlocal|local) ;; *) echo "Unknown variant: $VARIANT" >&2; exit 2;; esac
export PATH="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin:$PATH"
export LD_LIBRARY_PATH='' PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export LOGURU_COLORIZE="${LOGURU_COLORIZE:-true}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export COSMOS_CPU_AFFINITY_MODE=partition COSMOS_FLASH2_VARLEN=1 I4_ATTN_BACKENDS=flash2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export WAN_VAE_PATH="$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth"
case "$DATA" in
  sandwich)
    export SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/singlerighthand_sandwich_100
    export SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
    export POINTFLOW_EPISODE_ALLOWLIST="$ROOT/examples/pointflow_sandwich924_all_101_episodes.txt"
    export POINTFLOW_MANIFEST="$ROOT/pointflow_outputs/sandwich_924_20260928/manifest.json"
    export FK_ANNOTATION_ROOT=/data/shichaojian/raw_data/sandwich_fk21
    export FK_CAMERA_PROFILE=legacy
    export POINTFLOW_DISPLACEMENT_SCALE=0.0528 FK_DISPLACEMENT_SCALE=0.083745
    STATS="$ROOT/pointflow_outputs/real_action_normalization_20261006/sandwich101_val5pct/action_stats.json"
    VAL_EPISODES=5 ;;
  dropper_mix)
    export SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/dropper_dagger_mix
    export SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache
    export POINTFLOW_EPISODE_ALLOWLIST="$ROOT/examples/pointflow_dropper_dagger_mix_71_episodes.txt"
    export POINTFLOW_MANIFEST="$ROOT/pointflow_outputs/dagger_mix_71_20261001/manifest.json"
    export FK_ANNOTATION_ROOT=/data/shichaojian/raw_data/dropper_dagger_mix_fk21
    export FK_CAMERA_PROFILE=dagger
    # New camera mount: full 71-episode rescan still rounds to FK scale 0.043357.
    export POINTFLOW_DISPLACEMENT_SCALE=0.037493 FK_DISPLACEMENT_SCALE=0.043357
    STATS="$ROOT/pointflow_outputs/real_action_normalization_20261006/dropper_mix71_val5pct/action_stats.json"
    VAL_EPISODES=4 ;;
  *) echo "Unknown dataset: $DATA" >&2; exit 2 ;;
esac
export SINGLERIGHTHAND_USE_VIDEO_CACHE=true
export SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT="$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents"
export POINTFLOW_WINDOW_CACHE_ROOT="$SINGLERIGHTHAND_CACHE_ROOT/pointflow_windows"
export POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128
export POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_GEOMETRY_MOTION_FUSION=false
export POINTFLOW_SELECT_TOP_N=500 POINTFLOW_MIN_VOXEL_MEMBERS=3 POINTFLOW_SELECT_MIN_VALID_STEPS=16
export POINTFLOW_SELECT_REGION_QUOTAS=2:0.40,3:0.45,4:0.15 POINTFLOW_SELECT_PHANTOM_GUARD=true
export POINTFLOW_FREEZE_SONATA=true POINTFLOW_REFERENCE_ATTENTION=false
export POINTFLOW_DISPLACEMENT_FRAME_SCALES='' POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=''
export POINTFLOW_SONATA_CHECKPOINT=/data/shichaojian/checkpoints/ptv3/sonata_small.pth
export FK_ENCODER_CHECKPOINT=1 FK_PROJECT_ANCHORS=true POINTFLOW_FK_UNIFIED_EVAL=true POINTFLOW_EVAL_JOINT=true
export POINTFLOW_FK_LOCAL_ROPE=false POINTFLOW_FK_ATTN_IMPL=partition
[[ "$VARIANT" != local ]] || export POINTFLOW_FK_LOCAL_ROPE=true
if [[ "$VARIANT" == wam ]]; then
    export POINTFLOW_SONATA_CHECKPOINT='' POINTFLOW_MANIFEST='' FK_ENCODER_CHECKPOINT='' FK_ANNOTATION_ROOT=''
    export FK_PROJECT_ANCHORS=false POINTFLOW_FK_UNIFIED_EVAL=false POINTFLOW_WINDOW_CACHE_ROOT=''
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-1}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shichaojian/runs/real_normalized/${DATA}_${VARIANT}_20k_20261006}"
export REAL_ACTION_STATS_PATH="$STATS"
export REAL_ACTION_STATS_SHA256="$(sha256sum "$STATS" | cut -d' ' -f1)"
TOML_FILE=examples/toml/sft_config/action_policy_fk_point_singlerighthand_edge.toml
DATASET_PATH="$SINGLERIGHTHAND_RAW_ROOT"
TAIL_OVERRIDES=(
  "job.name=action_policy_real_${DATA}_${VARIANT}"
  trainer.max_iter=20000 'scheduler.cycle_lengths=[20000]' trainer.seed=42 trainer.grad_accum_iter=1
  trainer.run_validation=true trainer.run_validation_on_start=true trainer.validation_iter=500 checkpoint.save_iter=500
  dataloader_train.max_samples_per_batch=16 dataloader_train.dataloader.batch_size=16
  dataloader_train.dataloader.in_order=true
  dataloader_train.dataloader.datasets.singlerighthand.dataset.split_val_ratio=0.05
  dataloader_val.dataloader.datasets.singlerighthand.dataset.split_val_ratio=0.05
  "model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"
  "model.config.parallelism.data_parallel_replicate_degree=$NNODES"
  model.config.compile.enabled=true model.config.compile.compiled_region=language model.config.compile.compile_dynamic=true
  model.config.activation_checkpointing.mode=selective
  'model.config.activation_checkpointing.save_ops_regex=[_flash_attn.*forward]'
  'model.config.activation_checkpointing.save_mm_shapes=[[2048,2048],[2048,1024]]'
  model.config.rectified_flow_training_config.independent_pointflow_schedule=false
  model.config.rectified_flow_training_config.independent_fk_schedule=false
  "model.config.rectified_flow_training_config.pointflow_displacement_scale=$POINTFLOW_DISPLACEMENT_SCALE"
)
if [[ "$VARIANT" != wam ]]; then
    TAIL_OVERRIDES+=("trainer.callbacks.pointflow_eval.val_episodes=$VAL_EPISODES" trainer.callbacks.pointflow_eval.val_windows=1)
fi
# Extra Hydra overrides appended last (later value wins), e.g. EXTRA_TAIL_OVERRIDES='trainer.max_iter=40000 scheduler.cycle_lengths=[40000]'
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then read -r -a _extra <<< "$EXTRA_TAIL_OVERRIDES"; TAIL_OVERRIDES+=("${_extra[@]}"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    export IMAGINAIRE_OUTPUT_ROOT="$ROOT/pointflow_outputs/real_action_normalization_20261006/resolved_configs"
    python -m cosmos_framework.scripts.train --dryrun --sft-toml="$TOML_FILE" -- "${TAIL_OVERRIDES[@]}"
    exit
fi
RESULT="$(python tools/prepare_normalized_run.py --kind real --stats "$STATS" --output-root "$OUTPUT_ROOT" --fk-camera-profile "$FK_CAMERA_PROFILE" --variant "$VARIANT")"
export REAL_ACTION_STATS_PATH="$(head -n1 <<< "$RESULT")"
export REAL_ACTION_STATS_SHA256="$(tail -n1 <<< "$RESULT")"
# Shared launcher resolves platform node topology and tees per-node stdout/stderr.
LOG_FILENAME="launcher_${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${HOSTNAME:-node}}}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
source "$ROOT/examples/_sft_launcher_common.sh"
