# 同步到 WorldAct monorepo

本仓库(Cosmos Framework + 本地场景适配)有两份:

| 角色 | 路径 | 用途 |
| --- | --- | --- |
| **原仓库(source of truth)** | `/mnt/hwfile/shichaojian/cosmos-framework` | 在这里改代码、跑训练/推理/测试。可以是官方 clone, 不需要能 push。 |
| **WorldAct cosmos 分支副本(发布快照)** | `/mnt/petrelfs/shichaojian/WorldAct-cosmos` | 单独 checkout/clone WorldAct 的 `cosmos` 分支, 根目录直接放 Cosmos 代码。 |

WorldAct 是独立 git 仓库, 远程按分支存放不同项目代码: `lingbot-va` 分支放 LingBot-VA, `cosmos` 分支放 Cosmos。Cosmos 同步时刻意不带原仓库 `.git/`, 由 WorldAct 的 `cosmos` 分支统一 `git add/commit/push`。

## 纪律: 只在原仓库改代码

所有代码改动都在原仓库(`/mnt/hwfile/...`)做并测试, WorldAct 这边只负责提交发布快照。

> 注意: 不要在 WorldAct 的 `cosmos` 分支里直接改代码。同步是 `rsync --delete` 单向镜像, 会用原仓库覆盖 WorldAct `cosmos` 分支里的同路径本地改动。

## 同步流程

```bash
# 1) 在原仓库改代码并测试
cd /mnt/hwfile/shichaojian/cosmos-framework
# uv run ruff check .
# uv run pytest ...

# 2) 准备 WorldAct 的 cosmos 分支 checkout
cd /mnt/petrelfs/shichaojian
git clone /mnt/petrelfs/shichaojian/WorldAct WorldAct-cosmos
cd WorldAct-cosmos
git remote set-url origin https://github.com/liuhangxu-robin/WorldAct.git
git checkout -b cosmos origin/main
# 如果远程 cosmos 分支已经存在, 用:
# git checkout -b cosmos origin/cosmos

# 3) 从原仓库单向镜像到 WorldAct-cosmos 根目录, 先 dry-run 看范围
cd /mnt/hwfile/shichaojian/cosmos-framework
bash sync_cosmos_framework.sh --dry-run
bash sync_cosmos_framework.sh

# 4) 在 cosmos 分支提交
cd /mnt/petrelfs/shichaojian/WorldAct-cosmos
git add .
git commit -m "update cosmos-framework"
git push -u origin cosmos
```

原仓库不需要 `git add`: 同步脚本按文件系统镜像, 未提交但未被排除的文件也会同步。提交只发生在 WorldAct 仓库。

## 同步脚本做了什么

`sync_cosmos_framework.sh`:

- 使用 `rsync -a --delete` 把原仓库代码单向镜像到 WorldAct 的 `cosmos` 分支 checkout 根目录。
- 同步前检查目标 checkout 当前分支必须是 `cosmos`, 防止误刷 `lingbot-va` 分支。
- 排除 `.git`, `.git/`, `.venv/`, `.venv-cosmos/`, cache, 日志, 临时 `.tmp_*` 文件。
- 排除大体积或本地生成目录: `assets/`, `credentials/`, `datasets/`, `outputs/`, `tmp/`, `training_output/`, `cosmos_training_meta/`, `examples/data/`, `examples/checkpoints/`, `examples/**/output/`, `examples/**/datasets/`。
- 排除当前无法拉取的 LFS 示例图片: `inputs/prompt_upsampler/image_inputs/*.jpg`; 若目标 checkout 已有旧同步残留, 脚本会在真实同步时删除这些 stale pointer 文件。
- 排除常见权重/数据/视频归档文件: `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.ckpt`, `*.npy`, `*.npz`, `*.parquet`, `*.mp4`, `*.tar`, `*.zip`, `*.sif` 等。

`--delete` 只删除会被同步管理的代码文件; 被 exclude 的数据、权重、虚拟环境和日志受保护, 不会被同步脚本删除。

## 要在 WorldAct 里跑代码?

WorldAct `cosmos` 分支副本默认不带数据、权重、虚拟环境和输出目录。若确实需要在 WorldAct checkout 里运行, 用软链指回原仓库或共享存储:

```bash
cd /mnt/petrelfs/shichaojian/WorldAct-cosmos
ln -s /mnt/hwfile/shichaojian/cosmos-framework/.venv-cosmos .venv-cosmos
ln -s /mnt/hwfile/shichaojian/cosmos-framework/outputs outputs
# checkpoints / datasets 同理按需软链
```

软链目标应保持在同步脚本 exclude 列表里, 避免被提交或镜像删除。

## 本文档自身

`WORLDACT_SYNC.md` 和 `sync_cosmos_framework.sh` 存在原仓库根目录, 会随同步流入 WorldAct 的 `cosmos` 分支。要改它们时, 去原仓库改, 再重新同步。
