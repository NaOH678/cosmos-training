#!/usr/bin/env bash
# 在任务平台申请单节点 GPU 后执行一次；默认 8 卡基线，--smoke 可用 2 卡。
# 默认训练 Cosmos3-Edge-DROID 4B / task27 / WAM，运行 2000 步。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=baseline
DRY_RUN=0
EXTRA_OVERRIDES=()

usage() {
    cat <<'HELP'
Usage: bash mrlv/launch_wam_task.sh [--smoke] [--dry-run] [-- HYDRA_OVERRIDE ...]

Default: activate Conda cosmos, then train 4B Edge-DROID WAM on task27.
         1 node / 8 GPUs, batch 16/GPU, 2000 steps, 40000-step LR schedule.
--smoke: 2 steps, batch 1/GPU; disable compile and validation, save at step 2.
         Set SENSECORE_ACCELERATE_DEVICE_COUNT=2 for a two-GPU smoke test.
--dry-run: activate the environment and resolve the config without GPU workers.
Checkpoint policy: after a successful save, retain only the newest complete checkpoint.

Defaults below can be overridden with environment variables:
  LITEGEN_DEVICE (cuda|npu), TRAIN_PYTHON (skip Conda activation), LITEGEN_ROOT,
  CONDA_INIT, CONDA_ENV, OSS_ROOT, OUTPUT_ROOT, WAM_TMPDIR (default /tmp),
  WAM_COMPILER_CACHE_ROOT (default: a fresh directory under WAM_TMPDIR),
  EDGE_DROID_MODEL_PATH, BASE_CHECKPOINT_PATH, WAN_VAE_PATH, SIM_POINTFK_BUNDLE,
  MAX_ITER, BATCH_SIZE, GRAD_ACCUM_ITER, CYCLE_LENGTH, COMPILE, RUN_VALIDATION, SAVE_ITER,
  VALIDATION_ITER, LOGGING_ITER, TRAIN_WORKERS, TRAIN_SEED, JOB_NAME.
  BASELINE_MONITOR (1), TENSORBOARD_DIR ($OUTPUT_ROOT/tensorboard),
  BENCH_WARMUP_STEPS (50; skipped when summarizing steady-state performance).

Platform topology: SENSECORE_ACCELERATE_DEVICE_COUNT=8,
  SENSECORE_PYTORCH_NNODES=1, SENSECORE_PYTORCH_NODE_RANK=0.
MASTER_ADDR and MASTER_PORT are preserved when provided by the platform.
Mount /mnt/lvminrui and /oss in the task container. Run this script with bash.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (( $# )); do
    case "$1" in
        --smoke) MODE=smoke; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_OVERRIDES=("$@"); break ;;
        *) die "Unknown argument: $1 (use --help)" ;;
    esac
done

# 1. Select an interpreter and preserve the vendor runtime environment.
DEVICE_KIND="${LITEGEN_DEVICE:-${COSMOS_DEVICE:-cuda}}"
[[ "$DEVICE_KIND" == cuda || "$DEVICE_KIND" == npu ]] || die "Unsupported training device: $DEVICE_KIND"
export COSMOS_DEVICE="$DEVICE_KIND" LITEGEN_DEVICE="$DEVICE_KIND"
if [[ -z "${TRAIN_PYTHON:-}" ]]; then
    CONDA_INIT="${CONDA_INIT:-/mnt/lvminrui/conda_init.sh}"
    CONDA_ENV="${CONDA_ENV:-cosmos}"
    [[ -r "$CONDA_INIT" ]] || die "Conda init is not readable: $CONDA_INIT"
    set +u
    source "$CONDA_INIT"
    conda activate "$CONDA_ENV"
    set -u
    TRAIN_PYTHON="$(command -v python)"
fi
[[ -x "$TRAIN_PYTHON" ]] || die "TRAIN_PYTHON is not executable: $TRAIN_PYTHON"
cd "$REPO_ROOT"
# The original CUDA image needs this workaround; NPU must retain CANN libraries.
if [[ "$DEVICE_KIND" == cuda ]]; then export LD_LIBRARY_PATH=''; fi
export PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1
LITEGEN_ROOT="${LITEGEN_ROOT:-$REPO_ROOT/../LiteGen.dev}"
if [[ -r "$LITEGEN_ROOT/litegen/__init__.py" ]]; then export PYTHONPATH="$REPO_ROOT:$LITEGEN_ROOT"; fi
if [[ "${COSMOS_TRAINING_BACKEND:-native}" == litegen ]]; then
    "$TRAIN_PYTHON" - <<'PY_CHECK'
import inspect
from litegen import LiteGen
if "materialize_meta" not in inspect.signature(LiteGen.prepare).parameters:
    raise SystemExit("LiteGen needs the Cosmos materialize_meta extension from this workspace.")
PY_CHECK
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 多进程 IPC 临时文件放在容器本地目录，避免共享卷退出清理时 EBUSY。
export TMPDIR="${WAM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# 2. 设置 torchrun 需要的变量。保留平台注入的值；未注入时使用单机 8 卡默认值。
export SENSECORE_ACCELERATE_DEVICE_COUNT="${SENSECORE_ACCELERATE_DEVICE_COUNT:-${NPROC_PER_NODE:-8}}"
export SENSECORE_PYTORCH_NNODES="${SENSECORE_PYTORCH_NNODES:-${NNODES:-1}}"
export SENSECORE_PYTORCH_NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29644}"
[[ "$SENSECORE_ACCELERATE_DEVICE_COUNT" =~ ^[1-9][0-9]*$ ]] || die "Accelerator count must be a positive integer"
[[ "$SENSECORE_PYTORCH_NNODES" == 1 && "$SENSECORE_PYTORCH_NODE_RANK" == 0 ]] ||
    die "This script requires 1 node / node rank 0; received $SENSECORE_PYTORCH_NNODES nodes / rank $SENSECORE_PYTORCH_NODE_RANK"
[[ "$MODE" == smoke || "$SENSECORE_ACCELERATE_DEVICE_COUNT" == 8 ]] ||
    die "The baseline requires 8 accelerators; use --smoke to check training on fewer accelerators"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( MASTER_PORT <= 65535 )) || die "Invalid MASTER_PORT: $MASTER_PORT"
export NPROC_PER_NODE="$SENSECORE_ACCELERATE_DEVICE_COUNT"
export NNODES="$SENSECORE_PYTORCH_NNODES" NODE_RANK="$SENSECORE_PYTORCH_NODE_RANK"

# 3. 设置 4B HF 资源、DCP 初始权重、VAE 和 task27 数据的位置。
OSS_ROOT="${OSS_ROOT:-/oss}"
export EDGE_DROID_MODEL_PATH="${EDGE_DROID_MODEL_PATH:-$OSS_ROOT/shichaojian/models/cosmos3-edge-droid}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$OSS_ROOT/shichaojian/models/cosmos3-edge-droid-dcp}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
export SIM_POINTFK_BUNDLE="${SIM_POINTFK_BUNDLE:-$OSS_ROOT/shichaojian/wam_datasets_20261029/sim_task27_ball_wam}"
SIM_CACHE="$SIM_POINTFK_BUNDLE/datasets/cosmos-cache"
export SIM_POINTFK_SELECTION_ROOT="$SIM_CACHE/pointflow_fk_handguided1024"
export SIM_ACTION_STATS_PATH="$SIM_POINTFK_BUNDLE/normalization_train_only.json"
TOML_FILE="$REPO_ROOT/examples/toml/sft_config/action_policy_sim_pointfk_edge.toml"

# 4. 原生 WAM：联合训练视频和动作，关闭 PointFlow/FK 分支。
export SIM_POINTFK_VARIANT=wam
export POINTFLOW_SONATA_CHECKPOINT='' FK_ENCODER_CHECKPOINT='' FK_SKIP_RESUME_KEYS=''
export POINTFLOW_FK_LOCAL_ROPE=false POINTFLOW_FK_UNIFIED_EVAL=false POINTFLOW_EVAL_JOINT=false
export COSMOS_GPU_VIDEO_AUGMENTATION=false FK_ROLLOUT=false FK_SAMPLER_PROBE=false
export REAL_ACTION_VARIANT='' REAL_ACTION_STATS_PATH='' REAL_ACTION_STATS_SHA256=''
export POINTFLOW_DISPLACEMENT_FRAME_SCALES='' POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=''
export WANDB_MODE=offline
if [[ "$DEVICE_KIND" == cuda ]]; then
    export COSMOS_FLASH2_VARLEN=1 I4_ATTN_BACKENDS=flash2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    export COSMOS_FLASH2_VARLEN=0 I4_ATTN_BACKENDS=npu
fi
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0
export COSMOS_CPU_AFFINITY_MODE=partition

# 5. 训练参数和输出目录。显式指定相同 OUTPUT_ROOT 时会自动续训。
if [[ "$MODE" == smoke ]]; then
    MAX_ITER="${MAX_ITER:-2}"; BATCH_SIZE="${BATCH_SIZE:-1}"
    COMPILE="${COMPILE:-false}"; RUN_VALIDATION="${RUN_VALIDATION:-false}"
    SAVE_ITER="${SAVE_ITER:-2}"; LOGGING_ITER="${LOGGING_ITER:-1}"
else
    MAX_ITER="${MAX_ITER:-2000}"; BATCH_SIZE="${BATCH_SIZE:-16}"
    DEFAULT_COMPILE=false
    if [[ "$DEVICE_KIND" == cuda ]]; then DEFAULT_COMPILE=true; fi
    COMPILE="${COMPILE:-$DEFAULT_COMPILE}"; RUN_VALIDATION="${RUN_VALIDATION:-true}"
    SAVE_ITER="${SAVE_ITER:-500}"; LOGGING_ITER="${LOGGING_ITER:-10}"
fi
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"
CYCLE_LENGTH="${CYCLE_LENGTH:-40000}"
VALIDATION_ITER="${VALIDATION_ITER:-2000}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
JOB_NAME="${JOB_NAME:-sim_task27_wam_task_$MODE}"
for setting in MAX_ITER BATCH_SIZE CYCLE_LENGTH SAVE_ITER LOGGING_ITER VALIDATION_ITER TRAIN_WORKERS GRAD_ACCUM_ITER; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || die "$setting must be a positive integer"
done
[[ "$TRAIN_SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die "TRAIN_SEED must be a non-negative integer"
[[ "$COMPILE" == true || "$COMPILE" == false ]] || die "COMPILE must be true or false"
[[ "$RUN_VALIDATION" == true || "$RUN_VALIDATION" == false ]] || die "RUN_VALIDATION must be true or false"
[[ "$JOB_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "JOB_NAME contains unsupported characters"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
# 覆盖 Conda 激活脚本的共享盘缓存默认值，避免并发缓存文件读取失败。
WAM_COMPILER_CACHE_ROOT="${WAM_COMPILER_CACHE_ROOT:-$TMPDIR/cosmos-wam-compile-$RUN_STAMP}"
export TRITON_CACHE_DIR="$WAM_COMPILER_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$WAM_COMPILER_CACHE_ROOT/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/train/${JOB_NAME}_$RUN_STAMP}"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
BASELINE_MONITOR="${BASELINE_MONITOR:-1}"
BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS:-50}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_ROOT/tensorboard}"
[[ "$BASELINE_MONITOR" == 0 || "$BASELINE_MONITOR" == 1 ]] || die "BASELINE_MONITOR must be 0 or 1"
[[ "$BENCH_WARMUP_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] || die "BENCH_WARMUP_STEPS must be a non-negative integer"
MONITOR_OVERRIDES=()
if [[ "$BASELINE_MONITOR" == 1 ]]; then
    "$TRAIN_PYTHON" - <<'PY'
from importlib.util import find_spec
if find_spec("tensorboard") is None:
    raise SystemExit("TensorBoard is missing in the active environment; install tensorboard or set BASELINE_MONITOR=0.")
PY
    MONITOR_OVERRIDES=(
        "+trainer.callbacks.baseline_monitor._target_=mrlv.training_monitor.TrainingMonitor"
        "+trainer.callbacks.baseline_monitor.log_dir=$TENSORBOARD_DIR"
        "+trainer.callbacks.baseline_monitor.warmup_steps=$BENCH_WARMUP_STEPS"
    )
fi

# 6. 检查输入，并读取共享清单中的元数据；WAM 不向模型提供几何标签。
for input in "$TOML_FILE" "$EDGE_DROID_MODEL_PATH/config.json" \
    "$EDGE_DROID_MODEL_PATH/tokenizer_config.json" "$EDGE_DROID_MODEL_PATH/tokenizer.json" \
    "$BASE_CHECKPOINT_PATH/model/.metadata" "$WAN_VAE_PATH" "$SIM_ACTION_STATS_PATH" \
    "$SIM_POINTFK_SELECTION_ROOT/manifest.json" "$SIM_CACHE/window_index.json" \
    "$SIM_CACHE/vae_window_latents/window_manifest.json"; do
    [[ -r "$input" ]] || die "Required input is not readable: $input"
done
[[ -d "$SIM_CACHE/video_frames" ]] || die "Video frame cache is missing: $SIM_CACHE/video_frames"
SELECTION_INFO="$("$TRAIN_PYTHON" - "$SIM_POINTFK_SELECTION_ROOT/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
if manifest["schema"] != "sim_pointfk_handguided1024_v1":
    raise SystemExit("Unexpected selection manifest schema")
print(manifest["train_only_scales"]["pointflow"])
print(manifest["train_only_scales"]["fk"])
print(manifest.get("frame", "camera"))
PY
)"
mapfile -t SELECTION_FIELDS <<< "$SELECTION_INFO"
export POINTFLOW_DISPLACEMENT_SCALE="${SELECTION_FIELDS[0]}"
export FK_DISPLACEMENT_SCALE="${SELECTION_FIELDS[1]}" POINTFK_GEOMETRY_FRAME="${SELECTION_FIELDS[2]}"
STATS_DIGEST="$(sha256sum "$SIM_ACTION_STATS_PATH")"
export SIM_ACTION_STATS_SHA256="${STATS_DIGEST%% *}"

if [[ "$DRY_RUN" == 0 ]]; then
    "$TRAIN_PYTHON" - <<'PY'
import os
import torch
from litegen.device import get_backend, device_type
expected = int(os.environ["NPROC_PER_NODE"])
backend = get_backend(device_type())
reason = backend.unavailable_reason()
if reason is not None:
    raise SystemExit(reason)
count = backend.device_count()
if count != expected:
    raise SystemExit(f"Expected {expected} visible {backend.device_type} devices; found {count}.")
print(f"PyTorch {torch.__version__}; device {backend.device_type}; visible devices: {count}")
PY
    STATS_RESULT="$("$TRAIN_PYTHON" tools/prepare_normalized_run.py --kind sim --stats "$SIM_ACTION_STATS_PATH" --output-root "$OUTPUT_ROOT")"
    mapfile -t STATS_FIELDS <<< "$STATS_RESULT"
    export SIM_ACTION_STATS_PATH="${STATS_FIELDS[0]}" SIM_ACTION_STATS_SHA256="${STATS_FIELDS[1]}"
fi
mkdir -p "$OUTPUT_ROOT/logs"

TAIL_OVERRIDES=(
    "job.name=$JOB_NAME" "job.wandb_mode=offline"
    "trainer.max_iter=$MAX_ITER" "scheduler.cycle_lengths=[$CYCLE_LENGTH]"
    "trainer.seed=$TRAIN_SEED" "trainer.grad_accum_iter=$GRAD_ACCUM_ITER" "trainer.logging_iter=$LOGGING_ITER"
    "trainer.run_validation=$RUN_VALIDATION" "trainer.run_validation_on_start=$RUN_VALIDATION"
    "trainer.validation_iter=$VALIDATION_ITER" "trainer.max_val_iter=1"
    "checkpoint.save_iter=$SAVE_ITER"
    "dataloader_train.max_samples_per_batch=$BATCH_SIZE"
    "dataloader_train.dataloader.batch_size=$BATCH_SIZE"
    "dataloader_train.dataloader.num_workers=$TRAIN_WORKERS"
    "dataloader_train.dataloader.in_order=true"
    "model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"
    "model.config.parallelism.data_parallel_replicate_degree=$NNODES"
    "model.config.parallelism.context_parallel_shard_degree=1"
    "model.config.compile.enabled=$COMPILE"
    "model.config.rectified_flow_training_config.independent_action_schedule=false"
    "model.config.rectified_flow_training_config.independent_pointflow_schedule=false"
    "model.config.rectified_flow_training_config.independent_fk_schedule=false"
    "+trainer.callbacks.latest_checkpoint._target_=mrlv.checkpoint_retention.KeepLatestCheckpoint"
    "${MONITOR_OVERRIDES[@]}"
    "${EXTRA_OVERRIDES[@]}"
)
printf 'Conda=%s Python=%s\n' "${CONDA_DEFAULT_ENV:-external}" "$TRAIN_PYTHON"
printf 'Temporary directory=%s\n' "$TMPDIR"
printf 'Triton cache=%s\nInductor cache=%s\n' "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
printf 'Baseline monitor=%s TensorBoard=%s performance_warmup_steps=%s\n' "$BASELINE_MONITOR" "$TENSORBOARD_DIR" "$BENCH_WARMUP_STEPS"
printf 'Accelerators/node=%s nodes=%s node_rank=%s master=%s:%s\n' \
    "$SENSECORE_ACCELERATE_DEVICE_COUNT" "$SENSECORE_PYTORCH_NNODES" "$SENSECORE_PYTORCH_NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT"
printf 'Model=%s\nCheckpoint=%s\nData=%s\nOutput=%s\n' \
    "$EDGE_DROID_MODEL_PATH" "$BASE_CHECKPOINT_PATH" "$SIM_POINTFK_BUNDLE" "$OUTPUT_ROOT"
printf 'Mode=%s steps=%s batch/rank=%s global_batch=%s compile=%s dry_run=%s\n' \
    "$MODE" "$MAX_ITER" "$BATCH_SIZE" "$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM_ITER))" "$COMPILE" "$DRY_RUN"

# 7. 只解析配置时不运行 torchrun、不申请 GPU worker。
if [[ "$DRY_RUN" == 1 ]]; then
    exec "$TRAIN_PYTHON" -m cosmos_framework.scripts.train --dryrun --sft-toml="$TOML_FILE" -- "${TAIL_OVERRIDES[@]}"
fi

# 8. 在已经分配 GPU 的任务容器中执行真正的训练命令。
"$TRAIN_PYTHON" -m torch.distributed.run \
    --nproc_per_node="$SENSECORE_ACCELERATE_DEVICE_COUNT" \
    --nnodes="$SENSECORE_PYTORCH_NNODES" \
    --node_rank="$SENSECORE_PYTORCH_NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/logs/launcher_$RUN_STAMP.log"
