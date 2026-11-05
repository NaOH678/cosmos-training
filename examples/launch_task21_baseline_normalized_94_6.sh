#!/usr/bin/env bash
# Task21 replay baseline: 47 trajectory groups train / 3 groups held out.
set -euo pipefail
export PATH="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin:$PATH"
export LD_LIBRARY_PATH=''
export PYTHONUNBUFFERED=1
# tee makes stdout a pipe; explicitly retain Loguru's terminal colors.
export LOGURU_COLORIZE="${LOGURU_COLORIZE:-true}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export COSMOS_FLASH2_VARLEN=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export COSMOS_CPU_AFFINITY_MODE="${COSMOS_CPU_AFFINITY_MODE:-partition}"
export BENCH2DEX_CACHE_ROOT=/data/shichaojian/raw_data/bench2dex/task21/cosmos-cache
export BENCH2DEX_LATENT_ROOT="$BENCH2DEX_CACHE_ROOT/vae_window_latents"
export BENCH2DEX_ACTION_STATS_PATH=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/pointflow_outputs/task21_baseline_split94_6_20261005/action_stats.json
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export WAN_VAE_PATH="$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shichaojian/runs/bench2dex/task21_normalized_94train_6val_20k_20261005}"
mkdir -p "$OUTPUT_ROOT/logs"
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex
bash examples/launch_sft_action_policy_bench2dex_normalized.sh \
  dataloader_train.dataloader.datasets.bench2dex.dataset.split=train \
  dataloader_train.dataloader.datasets.bench2dex.dataset.split_seed=42 \
  dataloader_train.dataloader.datasets.bench2dex.dataset.split_val_ratio=0.06 "$@" \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/launcher.log"
