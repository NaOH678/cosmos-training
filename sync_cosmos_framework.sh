#!/usr/bin/env bash
# Sync CODE from the live Cosmos Framework working copy into the WorldAct cosmos branch.
#
#   原仓库 (跑训练/推理, 带数据/venv/outputs)  ──code only──►  WorldAct cosmos branch root
#   /mnt/hwfile/shichaojian/cosmos-framework                  /mnt/petrelfs/shichaojian/WorldAct-cosmos
#
# 工作流: 在原仓库改代码/测试 → 跑这个脚本 → 在 WorldAct-cosmos 里 git add/commit。
# 单向镜像 (--delete): 原仓库删了的文件这里也删; 但被 exclude 的东西
# (数据/权重/.venv/outputs/.git/软链) 受保护, 不会被 --delete 动到。
#
# 用法:
#   bash sync_cosmos_framework.sh            # 真同步
#   bash sync_cosmos_framework.sh --dry-run  # 先看会改什么
set -euo pipefail

SRC=/mnt/hwfile/shichaojian/cosmos-framework
DEST="${WORLDACT_COSMOS_WORKTREE:-/mnt/petrelfs/shichaojian/WorldAct-cosmos}"
DRY="${1:-}"

if [ -n "$DRY" ] && [ "$DRY" != "--dry-run" ]; then
  echo "usage: bash sync_cosmos_framework.sh [--dry-run]" >&2
  exit 2
fi

if [ ! -d "$DEST" ]; then
  cat >&2 <<EOF
error: missing WorldAct cosmos checkout: $DEST

Create it first:
  cd /mnt/petrelfs/shichaojian
  git clone /mnt/petrelfs/shichaojian/WorldAct "$DEST"
  cd "$DEST"
  git remote set-url origin https://github.com/liuhangxu-robin/WorldAct.git
  git checkout -b cosmos origin/main

If the cosmos branch already exists:
  git fetch origin cosmos
  git checkout -b cosmos origin/cosmos
EOF
  exit 1
fi

BRANCH="$(cd "$DEST" && git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [ "$BRANCH" != "cosmos" ]; then
  echo "error: destination must be a WorldAct checkout on branch 'cosmos'; got '$BRANCH' at $DEST" >&2
  exit 1
fi

STALE_LFS_GLOB="$DEST/inputs/prompt_upsampler/image_inputs/*.jpg"
if compgen -G "$STALE_LFS_GLOB" >/dev/null; then
  if [ "$DRY" = "--dry-run" ]; then
    for path in $STALE_LFS_GLOB; do
      echo "[sync] would remove stale LFS example image: ${path#$DEST/}"
    done
  else
    for path in $STALE_LFS_GLOB; do
      echo "[sync] remove stale LFS example image: ${path#$DEST/}"
      rm -f "$path"
    done
  fi
fi

EXC=(
  --exclude='.git' --exclude='.git/'
  --exclude='.venv/' --exclude='.venv-cosmos/' --exclude='venv/' --exclude='env/' --exclude='ENV/'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo' --exclude='*.so'
  --exclude='.cache/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.mypy_cache/'
  --exclude='.pyre/' --exclude='.pytype/' --exclude='.hypothesis/' --exclude='.nox/' --exclude='.tox/'
  --exclude='.tmp_*' --exclude='*.swp' --exclude='.DS_Store'
  --exclude='assets/' --exclude='credentials/' --exclude='datasets/' --exclude='outputs/' --exclude='output/'
  --exclude='tmp/' --exclude='checkpoints/' --exclude='training_output/' --exclude='cosmos_training_meta/'
  --exclude='examples/data/' --exclude='examples/checkpoints/'
  --exclude='examples/**/data/' --exclude='examples/**/checkpoints/'
  --exclude='examples/**/datasets/' --exclude='examples/**/output/' --exclude='examples/**/outputs/'
  --exclude='inputs/prompt_upsampler/image_inputs/*.jpg'
  --exclude='runs/' --exclude='wandb/' --exclude='logs/' --exclude='ray_results/'
  --exclude='build/' --exclude='dist/' --exclude='downloads/' --exclude='*.egg-info/' --exclude='.eggs/'
  --exclude='htmlcov/' --exclude='cover/' --exclude='docs/_build/' --exclude='site/'
  --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin' --exclude='*.ckpt'
  --exclude='*.nemo' --exclude='*.onnx' --exclude='*.engine'
  --exclude='*.npy' --exclude='*.npz' --exclude='*.arrow' --exclude='*.parquet' --exclude='*.lmdb'
  --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.tgz' --exclude='*.zip' --exclude='*.sif'
  --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' --exclude='*.mkv' --exclude='*.webm'
  --exclude='*.log' --exclude='*.out' --exclude='*.err'
)

echo "[sync] SRC=$SRC"
echo "[sync] DEST=$DEST"
echo "[sync] DEST_BRANCH=$BRANCH"
[ "$DRY" = "--dry-run" ] && echo "[sync] DRY-RUN (no changes)"
rsync -a --delete ${DRY:+--dry-run} -v "${EXC[@]}" "$SRC/" "$DEST/" | tail -80
echo "[sync] done. Next: cd $DEST && git add . && git commit && git push -u origin cosmos"
