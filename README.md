<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
    <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> |
    🤗 <a href="https://huggingface.co/collections/nvidia/cosmos3">Cosmos 3 </a>
</p>

<p align="center">
    Part of the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> project family — the training and serving framework repository.
</p>

# Cosmos-Framework

**Cosmos-Framework** is an end-to-end framework for training and serving world models, including the **Cosmos3** model family. Everything lives in a single top-level [`cosmos_framework/`](./cosmos_framework) Python package:

- **Training** — distributed FSDP / TP / CP / PP trainer, native DCP checkpoints with HuggingFace `safetensors` import/export, JSONL / WebDataset / LeRobot dataset adapters. Entry point: `cosmos_framework.scripts.train`. See [`docs/training.md`](./docs/training.md).
- **Inference** — Diffusers / Transformers / vLLM backends with offline batch generation and online serving (Ray + Gradio). Entry point: `cosmos_framework.scripts.inference`. Ecosystem-facing shim libraries (lightweight standalone wrappers for downstream projects) live under [`packages/`](./packages).

## WorldAct 训练主线（本分支）

基于 NVIDIA cosmos-framework，加入单右手真机（sandwich / dropper）与 Bench2Dex 仿真的 WAM 后训练，以及可选的 PointFlow/FK 几何分支。

原生 WAM 训练入口（第二个参数选 `wam`；`nonlocal` / `local` 是 PointFlow/FK 变体，默认不需要）：

| 用途 | 入口 |
|---|---|
| 真机，action 归一化 | `examples/launch_real_normalized.sh sandwich\|dropper_mix wam` |
| 仿真，action 归一化 | `OUTPUT_ROOT=... examples/launch_sim_normalized.sh <task> wam` |
| 实验配置 | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py`、`action_policy_sim_pointfk_edge.py` |
| 归一化统计 | `tools/compute_real_action_stats.py` → `action_stats.json`（q01/q99，sha256 写入训练配置），实现见 `cosmos_framework/data/generator/action/real_normalization.py` |
| 视频帧 / VAE latent 缓存 | `docs/pointflow_data_pipeline.md`，`examples/cache_singlerighthand_window_latents.sh`（按训练窗口编码，与在线路径逐位一致） |
| 数据集 | `cosmos_framework/data/generator/action/datasets/singlerighthand_raw_dataset.py`、`sim_pointfk_dataset.py` |

- 主干网络、loss、采样器与官方一致；PointFlow/FK 相关开关（`sync_plain_param_grads`、`joint_attn_mask`、FK 一致性 loss 等）默认关闭，关闭时计算路径与原版相同。
- 启动脚本里的数据、模型和 venv 路径是训练机上的绝对路径，迁移时需要改。
- 推理与部署见 `docs/inference.md`、`docs/cosmos_protocol_v2_server.md`。

### 原生 WAM 训练上手（给新硬件适配的同事）

本节说明我们实际怎么提交原生 WAM 训练，以及一条训练从启动到写出 checkpoint 经过了哪些代码。适配新显卡前，建议先把下面的数据加载和训练逻辑看一遍。目前只需要适配 `wam` 变体，`nonlocal` / `local`（PointFlow/FK 分支）不在范围内。

#### 1. 一条命令启动

我们在 A800 集群上每个任务占 1 节点 8 卡，任务命令栏只写一行。

仿真（Bench2Dex，例如 task27）：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/<run_name> \
EXTRA_TAIL_OVERRIDES='job.name=sim_task27_wam trainer.max_iter=40000 scheduler.cycle_lengths=[40000] trainer.validation_iter=2000' \
bash examples/launch_sim_normalized.sh 27_ball wam
```

真机（单右手 sandwich / dropper_mix）：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/real_normalized/<run_name> \
bash examples/launch_real_normalized.sh sandwich wam
```

- 第一个参数选数据：仿真可以是 `21_v3`、`27_ball`、`06_fruit`、`07_citrus`，对应 `/data/shichaojian/sim_data/bench2dex_task<TASK>_training`；真机可以是 `sandwich` 或 `dropper_mix`。第二个参数固定写 `wam`。
- `EXTRA_TAIL_OVERRIDES` 是追加在最后的 Hydra 覆盖项，后写的生效。
  - 我们的仿真实验统一用 40k 的学习率周期，训到 30k 手动停止，在 20k / 25k / 30k 测评。
  - `trainer.max_iter`、batch、保存间隔等都可以在这里改。
- 同一个 `OUTPUT_ROOT` 重新提交时，会从 `checkpoints/latest_checkpoint.txt` 自动续训。`tools/prepare_normalized_run.py` 会检查归一化统计和变体有没有变；变了就报错，要求换新的 `OUTPUT_ROOT`。
- 多机训练只需要把 `NNODES` / `NODE_RANK` / `MASTER_ADDR` 交给 `_sft_launcher_common.sh`，见 `docs/pointflow_multinode_training.md`。

#### 2. 启动链路

```
launch_sim_normalized.sh <task> wam        # 选数据 bundle，设 SIM_POINTFK_VARIANT=wam
 └─ launch_sim_pointfk_v2.sh               # 模型/VAE/venv 路径，注意力与编译开关，归一化统计
     └─ tools/prepare_normalized_run.py    # 把 action 统计固定到 OUTPUT_ROOT，并做续训检查
     └─ _sft_launcher_common.sh            # 组装 torchrun 参数（单机/多机），日志写到 OUTPUT_ROOT/logs
         └─ torchrun -m cosmos_framework.scripts.train --sft-toml=<toml> -- <Hydra 覆盖项>
             ├─ examples/toml/sft_config/action_policy_sim_pointfk_edge.toml           # lr、batch、调度、保存
             └─ configs/base/experiment/action/posttrain_config/action_policy_sim_pointfk_edge.py
                  # 继承 action_policy_singlerighthand_edge.py（模型、loss、回调），只换数据集
```

真机这条链路更短：`launch_real_normalized.sh` 直接调用 `prepare_normalized_run.py` 和 `_sft_launcher_common.sh`。用的 TOML 是 `action_policy_fk_point_singlerighthand_edge.toml`，实验配置是 `action_policy_singlerighthand_edge.py`。

启动脚本里写死了三类 A800 上的绝对路径，换机器时要改：

- venv：`/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv`，Python 3.13，PyTorch 2.10.0+cu128。
- 模型：`/data/shichaojian/models/cosmos3-edge-droid`（HF）及其 DCP 版本 `cosmos3-edge-droid-dcp`；VAE 是 `vae/Wan2.2_VAE.pth`。
- 数据：`/data/shichaojian/sim_data/...`、`/data/shichaojian/raw_data/...`、`/data/shichaojian/datasets/...`。

#### 3. 数据加载逻辑

**仿真**：`cosmos_framework/data/generator/action/datasets/sim_pointfk_dataset.py`。数据目录规范见 `docs/sim_pointfk_data_format.md`，生成流程见 `docs/SIM_DATA_PIPELINE.md`。

- 一个任务一个 bundle：`bench2dex_task<TASK>_training/`。
  - `normalization_train_only.json`：动作统计，只用训练集计算。
  - `datasets/cosmos-cache/video_frames/<episode>.npy`：三视角拼好的 RGB，`uint8 [T,3,720,640]`，以 memmap 方式读取。
  - `datasets/cosmos-cache/vae_window_latents/`：每个训练窗口预先编码好的 VAE latent，形状 `[48,9,40,40]`，以 uint16 存储的 bfloat16。
  - `datasets/cosmos-cache/window_index.json`：所有有效窗口的索引。
  - `pointflow_fk_handguided1024/manifest.json`：窗口清单和训练/验证划分，94/6，按轨迹组划分。WAM 也从这里读清单和划分，只是不读 PointFlow/FK 内容（`include_geometry=False`）。
- 读取时会核对三份清单的 SHA256 是否一致：选点清单、latent 清单、窗口索引。不一致直接报错，避免视频、动作和窗口错位。
- 每个样本是一个 33 帧窗口，20 Hz，跨度 1.6 s。
  - `vae_latent_cache`：训练时直接用缓存的 latent，不在线跑 VAE 编码；训练集也只读第 0 帧 RGB（`cached_video_only`）。
  - `action`：`[33, D]`。第 0 行是当前关节状态，后 32 行是绝对关节目标。按 q01/q99 仿射归一化：`offset=(q01+q99)/2`，`scale=max((q99-q01)/2, 0.05)`，不截断。最后补零到 64 维，补的维度不参与 loss。D 随机器人不同：Wuji 手 52 维，06 为 36 维，07 为 48 维。
  - `ai_caption`：任务描述文本，训练时有 10% 概率丢弃，用于 CFG。
  - 其余固定字段：`domain_id=28`，`conditioning_fps=20`，`viewpoint=concat_view`。
- dataloader 每卡 4 个 worker，按 episode 成块打乱，`in_order=True`。

**真机**：`singlerighthand_raw_dataset.py`（入口在 `action_sft_dataset.py`），合同见 `docs/action_policy_singlerighthand_edge_posttrain.md`。

- 原始数据：30 fps 的 `head.mp4`、`right_wrist.mp4` 和 LMDB。训练按 15 fps 取 33 帧观测、32 步动作，窗口跨度 2.13 s，起点每次滑动 1 个源帧。
- 动作是 27 维：右手末端 xyz、四元数 xyzw、20 个手指关节（弧度）。第 0 行是当前状态。
- 训练读两层缓存：`video_frames`（解码、拼图、缩放后的帧）和 `vae_window_latents`（每个窗口的 latent）。缓存生成方法见 `docs/pointflow_data_pipeline.md` 和 `examples/cache_singlerighthand_window_latents.sh`。验证集按 episode 留 5%。

#### 4. 训练逻辑（WAM）

原理详见 `docs/cosmos_wam_mode.md`。

- 视频 latent 和动作 token 拼在同一条序列上，同一个噪声水平 σ，一次前向同时去噪。
  - 条件部分不加噪、不进 loss：视频 latent 0（当前帧）和动作 token 0（当前状态）。
  - 预测部分：视频 latent 1–8 和动作 token 1–32。
- loss = 10 × 视频 MSE + 动作 MSE（`loss_scale=10.0`），用的是 rectified flow。代码里 `x_0` 是噪声，`x_1` 是干净数据，不要看反。
- 初始化：从 Cosmos3-Edge-DROID 的 DCP checkpoint 加载（`checkpoint.load_path=$BASE_CHECKPOINT_PATH`）。
- 优化：lr 2e-5，warmup 50 步后在 `cycle_lengths` 内衰减到 0。bf16，FSDP 分片数等于每节点卡数（`data_parallel_shard_degree=$NPROC_PER_NODE`）。
- batch：每卡 16，8 卡全局 128，不做梯度累积。
- 每 500 步存一次 checkpoint，到 `OUTPUT_ROOT/cosmos3_action/action_sft/<job.name>/checkpoints/iter_XXXXXXXXX/`。每个约 30 GB，不会自动删除旧的，磁盘要预留够。
- 验证：每 `validation_iter` 步在验证集上算一次。

**A800 参考值**（`sim_task06_wam_8gpu_40k_20261012`，8×A800 80GB）：

| 指标 | 数值 |
|---|---|
| 速度 | 约 3.55 s/步 |
| loss（每 1000 步的均值）：0–1k | 2.50 |
| 1k–2k | 1.06 |
| 5k–6k | 0.78 |
| 10k–11k | 0.67 |
| 20k–21k | 0.55 |
| 29k–30k | 0.49 |

新硬件上前几千步的 loss 应该和这条曲线基本重合，可以作为数值正确性的判断依据。

#### 5. 新显卡上要看的开关

这些都和硬件有关，适配时逐项确认：

| 开关 | 位置 | 作用 / 适配要点 |
|---|---|---|
| `I4_ATTN_BACKENDS=flash2`、`COSMOS_FLASH2_VARLEN=1` | 启动脚本 | 注意力后端。目前强制用 FlashAttention-2 的 varlen 接口；新卡不支持时，要换成可用的后端（选择逻辑在 `cosmos_framework/model/attention/`） |
| `model.config.compile.enabled=true`、`compile_dynamic=true` | TOML / 覆盖项 | torch.compile（Inductor/Triton）。`TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` 是为了绕过 A800 上反向时 Triton 共享内存溢出。新卡出现编译问题时，可以先关掉 compile 跑通 |
| `activation_checkpointing.mode=selective`、`save_ops_regex=[_flash_attn.*forward]`、`save_mm_shapes` | TOML | 选择性重计算，按 flash-attn 算子名和矩阵乘形状决定保存哪些激活。换了注意力后端，正则就匹配不到了，要一起改 |
| `dataloader_train.max_samples_per_batch` / `batch_size` | TOML / 覆盖项 | 每卡 batch 16 是按 80 GB 显存定的。显存小时调小，同时记下全局 batch 变化 |
| `NPROC_PER_NODE`、`data_parallel_shard_degree` | 环境变量 / 覆盖项 | 必须满足 `shard × replicate × CP = 总卡数` |
| `precision=bfloat16` | TOML | 需要 bf16 支持 |
| `trainer.callbacks.compile_tokenizer.enabled=false` | TOML | A800 环境没有 nvcc，所以关闭 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`COSMOS_CPU_AFFINITY_MODE=partition` | 启动脚本 | 显存分配器和 CPU 绑核，按新机器的拓扑调整 |

建议的适配顺序：

1. `DRY_RUN=1 bash examples/launch_sim_normalized.sh 27_ball wam`：只解析配置，不占 GPU。
2. 两步冒烟测试：
   ```bash
   EXTRA_TAIL_OVERRIDES='trainer.max_iter=2 checkpoint.save_iter=2' \
   OUTPUT_ROOT=<tmp> bash examples/launch_sim_normalized.sh 27_ball wam
   ```
3. 训 1–2k 步，对照上面的 loss 和速度参考值。
4. 再跑完整训练。

#### 6. 参考文档（建议阅读顺序）

| 文档 | 内容 |
|---|---|
| [`docs/cosmos_wam_mode.md`](./docs/cosmos_wam_mode.md) | WAM 原理：时间网格、条件/生成位置、加噪公式、loss，以及为什么要编码全部 33 帧 |
| [`docs/sim_pointfk_data_format.md`](./docs/sim_pointfk_data_format.md) | 仿真数据目录规范：训练前数据和训练缓存（latent、窗口索引、划分、归一化）的格式，以及准备命令 |
| [`docs/SIM_DATA_PIPELINE.md`](./docs/SIM_DATA_PIPELINE.md) | 仿真数据从原始 HDF5 到训练输入的完整处理流程（第 3、4、8、9 节和 WAM 相关） |
| [`docs/action_policy_singlerighthand_edge_posttrain.md`](./docs/action_policy_singlerighthand_edge_posttrain.md) | 真机单右手数据合同（帧率、窗口、27 维动作）、视频帧缓存、DCP 转换、dry run 与冒烟测试 |
| [`docs/pointflow_data_pipeline.md`](./docs/pointflow_data_pipeline.md) | 真机视频帧 / VAE 窗口 latent 缓存的生成方法 |
| [`docs/pointflow_multinode_training.md`](./docs/pointflow_multinode_training.md) | 多机训练：torchrun 拓扑、HSDP（shard × replicate）、集群注入的环境变量与排障 |
| [`docs/training.md`](./docs/training.md)、[`docs/sft_config.md`](./docs/sft_config.md) | 上游 cosmos-framework 的训练指南和 TOML 配置字段说明 |
| [`docs/setup.md`](./docs/setup.md) | 环境安装、CUDA 版本选择 |

## Cosmos 3

**Cosmos 3** is our newest model family [[Report]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) [[Website]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/). It is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences within a unified Mixture-of-Transformers architecture. By supporting highly flexible input-output configurations, it seamlessly unifies critical modalities for Physical AI — effectively subsuming vision-language models, video generators, world simulators, and world-action models into a single framework. For a guided experience to test out Cosmos3, please visit [[Cosmos]](https://github.com/nvidia/cosmos).

## Framework Documentation

- [Quickstart](#setup)
- [Setup](./docs/setup.md)
- [Training (Supervised Fine-Tuning)](./docs/training.md)
  - [JSONL Dataset](./docs/dataset_jsonl.md)
- [Inference](./docs/inference.md)
- [Policy Server](./docs/action_policy_droid_server.md)
- [Agent Skills](#agent-skills)
- Reference
  - [Code Structure](./docs/code_structure.md)
  - [Environment Variables](./docs/environment_variables.md)
  - [FAQ](./docs/faq.md)
  - [AGENTS.md — repo map for coding agents](./AGENTS.md)

## Setup

For more details and alternative installation methods, see [Setup](./docs/setup.md#installation). Before installing, make sure your machine meets the [System Requirements](./docs/setup.md#system-requirements). If you want a curated PyTorch + CUDA environment, start from the [recommended NVIDIA NGC base image](./docs/setup.md#recommended-base-image).

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
```

Install the package with `uv` (pick the dependency group that matches your CUDA toolkit — see [CUDA Variants](./docs/setup.md#cuda-variants)):

```shell
# CUDA 13.0 (recommended)
uv sync --all-extras --group=cu130-train
# Or, for CUDA 12.8:
# uv sync --all-extras --group=cu128-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If you are starting from the recommended NGC image (`nvcr.io/nvidia/pytorch:26.06-py3`), see the [one-shot quickstart](./docs/setup.md#quickstart-from-the-recommended-base-image).

## Training

For the full guide (data preparation, base-checkpoint conversion, parallelism strategies, mixed precision, resuming), see [Training](./docs/training.md). The number of GPUs required depends on the recipe; the shipped recipes under [`examples/`](./examples/README.md) are 8-GPU configurations (tested on 8× H100 80 GB) launched via their paired launch shells, e.g.:

```shell
bash examples/launch_sft_vision_nano.sh
```

Users may adjust the GPU count to match their model and underlying hardware architecture — tune `NPROC_PER_NODE` and the parallelism degrees (DP/CP/FSDP shard) in the recipe accordingly.

## Inference

See [Inference](./docs/inference.md) for the full guide — launch commands, supported modes, parallelism presets, and troubleshooting.

Quick single-GPU launch:

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

## Policy Server

See [Policy Server](./docs/action_policy_droid_server.md) for the full guide.

## Agent Skills

Coding agents (Claude Code, Codex CLI, Cursor, and other AGENTS.md-aware
tools) can load task-specific instructions for this repo from the bundled
AgentSkills. Each skill is a self-contained `SKILL.md` that the agent
invokes automatically when the user's request matches its description.

Skills live in [`.agents/skills/`](./.agents/skills) (canonical) and are
mirrored under [`.claude/skills/`](./.claude/skills) for Claude Code:

| Skill                                                                            | When it activates                                                                    |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`cosmos3-setup`](./.agents/skills/cosmos3-setup/SKILL.md)                       | Installation, environment setup, checkpoint downloads, Docker.                       |
| [`cosmos3-codebase-nav`](./.agents/skills/cosmos3-codebase-nav/SKILL.md)         | "Where is X" / "where do I change parameter Y" questions across `cosmos_framework/`. |
| [`cosmos3-inference`](./.agents/skills/cosmos3-inference/SKILL.md)               | Running offline or online inference, parallelism, sampling parameters.               |
| [`cosmos3-post-training`](./.agents/skills/cosmos3-post-training/SKILL.md)       | SFT post-training end-to-end: data prep, DCP conversion, launch, export.             |
| [`cosmos3-env-troubleshoot`](./.agents/skills/cosmos3-env-troubleshoot/SKILL.md) | Diagnosing install/runtime errors (ImportError, CUDA, Docker, checkpoint failures).  |

See [`AGENTS.md`](./AGENTS.md) for the canonical repo map that agents load
first; the skills above are referenced from there.

## Reference

| Topic                                                        | What it covers                                                                                                                            |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [Setup](./docs/setup.md)                                     | Hardware/software prerequisites, `uv` install paths, CUDA variants, Docker base image, and base-checkpoint downloading.                   |
| [Code Structure](./docs/code_structure.md)                   | Repository layout and a per-subpackage tour of `cosmos_framework/` — where each concern lives and where to add new code.                  |
| [Training](./docs/training.md)                               | Launching multi-GPU and multi-node runs; parallelism strategies; mixed precision; resuming.                                               |
| [Inference (from a trained checkpoint)](./docs/inference.md) | Loading a trained checkpoint into one of the inference backends.                                                                          |
| [Policy Server](./docs/action_policy_droid_server.md)        | Running the server-client pipeline for Cosmos3-Policy-DROID.                                                                              |
| [FAQ](./docs/faq.md)                                         | Troubleshooting (OOM, NCCL hangs, slow training), environment variables, and common pitfalls.                                             |
| [AGENTS.md](./AGENTS.md) + [Agent Skills](#agent-skills)     | Repo map and task-specific `SKILL.md` files loaded automatically by AGENTS.md-aware coding agents (Claude Code, Codex CLI, Cursor, etc.). |
