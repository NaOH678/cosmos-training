# Cosmos 4B WAM × LiteGen × Ascend 910B

本项目用于在 Ascend 910B 上通过 LiteGen 运行 Cosmos 4B WAM 训练。按下文准备镜像和外部输入、安装环境、完成两步验收，再启动单机八卡或四节点 32 卡训练。

| 入口 | 用途 |
|---|---|
| `bash run.sh` | 单机八卡两步功能验收；也提供源码校验、输入检查和环境安装选项 |
| `bash train.sh` | 单机或多机正式训练，每节点执行一次 |
| `bash train.sh --smoke` | 指定拓扑的两步训练 |
| `bash scripts/submit_sensecore.sh` | 平台在各节点执行同一份命令，使用预装共享环境启动训练 |

环境和挂载写入本地 `config/cluster.env`，可选训练参数写入 `config/train.env`。模型权重、数据缓存、输出存储、Python 环境以及系统 CANN/驱动需单独准备。LiteGen 由独立仓库维护，运行时通过 `LITEGEN_ROOT` 指向外部 checkout。

补充操作说明：[LiteGen 衔接与版本升级](docs/litegen_integration.md)、[预装共享环境](docs/shared_environment.md)、[多节点训练与恢复](docs/distributed_training.md)。

## 1. 先准备目标节点

使用以下基础镜像：

```text
registry2.d.pjlab.org.cn/ccr-lepton-official-images/a2-cann:9.1.0-a2-ubuntu22.04-py3.12
```

节点与安装要求如下。

| 项目 | 要求 |
|---|---|
| CPU / 系统 | Linux aarch64，使用上述 Ubuntu 22.04 镜像 |
| 设备 | 每节点 8 × Ascend 910B，64 GiB HBM/卡，驱动设备映射完整 |
| CANN / ops | 9.1.0，**910B 对应的 ops** |
| 驱动 / 固件 | 由集群提供，与实际 910B 和镜像 CANN 匹配 |
| 原生 Python | 通过 `NPU_BOOTSTRAP_PYTHON` 指定镜像自身的 ARM Python 3.12 |
| PyTorch / TorchNPU | `2.10.0+cpu` ARM wheel / `2.10.0.post6` |
| 环境创建 | 镜像原生 ARM Python 3.12 + venv；146 个哈希锁定 Python 分发包 |
| 存储 | venv 安装预留至少 5 GiB；单个完整 DCP 至少 40 GiB 可用，正式输出建议预留 80 GiB 以上 |
| 其他 | bash、venv/pip、flock、可用的 Python 包源、HCCL 通信条件；共享卷用于保存输出 |

`torch` 版本中的 `+cpu` 是锁定 ARM 分发包的标识，TorchNPU 注册后训练运行在 NPU；不能仅凭版本后缀判断执行设备。验收会实际执行 NPU BF16 算子。

执行单机八卡验收时，平台应分配一台含八张卡的任务容器，只执行一次 `bash run.sh`。执行多机训练时，每个节点各执行一次 `bash train.sh`，由脚本创建本节点的训练 rank；不要按卡数重复启动 launcher。检查每节点的 `npu-smi info`、`/dev/davinci0` 至 `/dev/davinci7`、`/dev/davinci_manager`，以及平台要求的辅助设备和驱动挂载。

```bash
uname -m
cat /etc/os-release
ldd --version
npu-smi info
df -h . /tmp /dev/shm
```

宿主驱动、固件和匹配的 CANN/ops 由集群管理员或镜像提供。项目脚本不安装或升级这些系统组件。

## 2. 迁移源码和外部输入

从两个独立分支克隆，保持并列目录。以下 Git 命令只在其子进程中清理动态库路径，训练仍保留 CANN 环境：

```bash
export WORKSPACE_ROOT=/目标共享卷/工作目录
export NPU_BOOTSTRAP_PYTHON=python3.12  # 若不在 PATH，改为目标镜像中的实际解释器路径
"$NPU_BOOTSTRAP_PYTHON" --version
mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git clone --branch lmr-cosmos \
  https://github.com/NaOH678/cosmos-training.git cosmos-training-litegen-910b
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git clone --branch lmr-cosmos \
  https://github.com/DeepLink-org/LiteGen.dev.git LiteGen.dev
cd cosmos-training-litegen-910b
LITEGEN_COMMIT=$("$NPU_BOOTSTRAP_PYTHON" -c 'import json; print(json.load(open("config/litegen.lock.json"))["commit"])')
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git -C ../LiteGen.dev checkout --detach "$LITEGEN_COMMIT"
```

`WORKSPACE_ROOT` 由接收方选择，项目目录可重命名；脚本根据自身位置定位源码。Python路径按镜像实际位置调整。固定提交用于保证两个仓库的版本匹配；开发时可切回 LiteGen 的 `lmr-cosmos` 分支。具体链接与升级流程见[衔接说明](docs/litegen_integration.md)。如果使用源码压缩包，仍需单独克隆 LiteGen。压缩包不含本机 `config/cluster.env`；应在目标集群按实际挂载重新填写该文件。

```bash
cd "$WORKSPACE_ROOT/cosmos-training-litegen-910b"
cp config/cluster.env.example config/cluster.env
```

编辑 `config/cluster.env`，至少设置下面四项为实际挂载路径：

| 变量 | 需要的内容 |
|---|---|
| `EDGE_DROID_MODEL_PATH` | 模型 `config.json`、tokenizer、processor/preprocessor、special tokens、chat template 等配置 |
| `BASE_CHECKPOINT_PATH` | 原始预训练 DCP；包含 `model/.metadata` 和两份 `model/*.distcp`，约 6.28 GiB |
| `WAN_VAE_PATH` | `Wan2.2_VAE.pth` 文件，约 2.63 GiB |
| `SIM_POINTFK_BUNDLE` | WAM 训练所需的归一化统计、窗口索引、动作记录、VAE latent 和视频帧缓存 |

```text
<SIM_POINTFK_BUNDLE>/
  normalization_train_only.json
  datasets/cosmos-cache/
    window_index.json
    pointflow_fk_handguided1024/manifest.json
    pointflow_fk_handguided1024/*.pt
    vae_window_latents/window_manifest.json
    vae_window_latents/*.npy
    video_frames/*.npy
```

虽然目录名含 PointFlow，纯 WAM 也从这里读取动作与窗口记录。必须迁移全部被清单引用的文件，包含 75,640 份动作记录；latent 约 97.38 GiB、video frames 约 101.63 GiB。不要换成随机数据或重新生成的不同窗口来做 loss 对齐。复制 DCP 时必须保留隐藏的 `.metadata`。

[external_inputs.json](external_inputs.json) 记录原始路径、必要文件大小、配置/清单 SHA256 及缓存规模。初始权重和多 GiB payload 没有逐文件内容哈希，输入检查不等于这些 payload 的全量内容校验。

目标集群配置示例；只修改本地 `config/cluster.env`，该文件不提交 Git。以下模型/数据目录是通用布局，必须对应实际挂载的原始输入：

```bash
export NPU_BOOTSTRAP_PYTHON="${NPU_BOOTSTRAP_PYTHON:-python3.12}"
export COSMOS_NPU_ENV="${COSMOS_NPU_ENV:-.venv-npu}"
# LiteGen 默认使用项目同级的 ../LiteGen.dev；其他位置再显式设置 LITEGEN_ROOT。
# CANN 默认自动查找标准安装位置；其他位置再设置 CANN_ENV。
export OSS_ROOT="${OSS_ROOT:-/目标数据卷}"
export EDGE_DROID_MODEL_PATH="${EDGE_DROID_MODEL_PATH:-$OSS_ROOT/models/cosmos3-edge-droid}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$OSS_ROOT/models/cosmos3-edge-droid-dcp}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
export SIM_POINTFK_BUNDLE="${SIM_POINTFK_BUNDLE:-$OSS_ROOT/datasets/sim_task27_ball_wam}"
```

`NPU_BOOTSTRAP_PYTHON` 必须指向**镜像自身可用的 ARM Python 3.12**。如果 `python3.12` 不在 PATH，在本地配置中填写目标镜像的实际路径；安装器仍检查架构和版本。不要移动旧 venv 或从 x86 导出二进制环境。

| 路径 | 默认或设置方式 |
|---|---|
| 项目 | 接收方选择克隆目录；脚本自动定位项目根目录 |
| LiteGen | 项目同级 `../LiteGen.dev`，可用 `LITEGEN_ROOT` 指定其他绝对路径 |
| Python venv | 项目下 `.venv-npu`；`COSMOS_NPU_ENV` 的相对路径统一相对于项目根目录 |
| 环境与训练报告 | 项目下 `outputs/`；多节点通过 `OUTPUT_ROOT` 指定同一共享目录 |
| 模型与数据 | 本地配置中的四个输入变量，支持任意实际挂载位置 |

项目位于共享卷时，默认 venv 也在共享卷：按[共享环境说明](docs/shared_environment.md)**只由一个准备容器提前安装并封存**，各训练节点只读取。若选择每节点自行安装，显式将 `COSMOS_NPU_ENV` 设为节点本地路径，例如 `/tmp/cosmos-npu-env`；容器重建后需要重建。不要让多个节点同时安装或升级同一个 venv。

迁移的是源码与安装流程，venv 必须在目标最终位置重新创建；共享 venv 的各节点需要相同基础 Python 路径和挂载位置。`external_inputs.json` 和 `reference/` 中记录的来源路径不作为当前安装目录；输入检查使用本地配置覆盖后的路径。

## 3. 安装环境并检查输入

```bash
cd /目标卷/cosmos-training-litegen-910b
mkdir -p outputs/setup
set -o pipefail
bash run.sh --verify 2>&1 | tee outputs/setup/verify.log
bash run.sh --check-inputs --all-payloads 2>&1 | tee outputs/setup/inputs.log
bash run.sh --bootstrap-only 2>&1 | tee outputs/setup/bootstrap.log
```

- `--verify` 校验当前交付的 `MANIFEST.json`。
- `--check-inputs` 检查模型/数据清单哈希、权重大小、所有 frame/latent 文件及确定性抽样的动作文件；`--all-payloads` 额外检查全部动作文件是否存在。结果在 `outputs/input_check.json`。
- `--bootstrap-only` 使用原生 Python 创建独立 venv，从 [environment/requirements.lock](environment/requirements.lock) 安装依赖，先核对外部 LiteGen 的提交和运行时修改状态，再以 `--no-deps` 安装外部 LiteGen 与本项目 Cosmos 的 editable 源码，执行 `pip check` 和所有锁定版本检查。版本报告及 LiteGen 提交/路径写入 `outputs/environment/installed.json`。
- 对带 `.cosmos-shared` 标记的预建共享环境，bootstrap 只核对准备状态、依赖和源码，不执行安装。训练平台入口使用 `scripts/submit_sensecore.sh`；准备流程见[共享环境说明](docs/shared_environment.md)。
- 安装不改系统 Python、CANN 或驱动。首次安装需能访问 Python 包源；使用集群镜像源时保留锁文件的版本和哈希。
- 重复 bootstrap 可复用同一位置的环境；移动源码目录时应使用新的 venv 并重新 bootstrap，避免 editable 注册仍指向旧目录。

安装后激活并确认 Cosmos 来自本项目、LiteGen 来自外部 checkout：

```bash
source config/cluster.env
source scripts/activate.sh
"$TRAIN_PYTHON" -c 'import torch, torch_npu, litegen, cosmos_framework; print(torch.__version__, torch_npu.__version__); print(litegen.__file__); print(cosmos_framework.__file__); print(torch.npu.device_count())'
```

激活脚本会设置 `LITEGEN_DEVICE=npu`、`COSMOS_DEVICE=npu`、`COMPILE=false`、Cosmos 与外部 LiteGen 的 `PYTHONPATH` 和 CANN 库路径。训练进程必须保留 CANN 的 `LD_LIBRARY_PATH`。

## 4. 先做真实八卡两步功能验收

```bash
bash run.sh
```

默认按顺序完成安装/输入检查、八卡真实设备预检、五项 attention、两卡 FSDP2 恢复测试、八卡 4B WAM 两步训练和完整 DCP 检查。也可以指定一个尚不存在的报告目录：

```bash
bash run.sh --report-root "$PWD/outputs/acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
```

| 阶段 | 必须满足 |
|---|---|
| 设备预检 | 八张真实 910B、软件版本、逐卡 BF16 运算与同步通过 |
| attention | 五项真实 NPU 测试执行通过，**skip 不计通过** |
| 两卡 LiteGen | FSDP2、FP32 master AdamW、RNG/DCP 精确恢复及恢复后更新 |
| 八卡 WAM | 八个 NPU rank，最终 step=2，三类 loss 有限 |
| 完整检查点 | model、optim、scheduler、trainer 元数据/分片完整；小状态及 RNG 实际回读 |
| 最终状态 | `status.json` 中 `status=passed` 且 `wam_executed=true` |

报告位于 `outputs/wam-910b-job-*` 或指定的 `--report-root`。包括 `preflight.json`、`attention.xml`、`fsdp-dcp/result.json`、`wam_result.json`、`checkpoint_schema.json` 及各阶段日志。任一阶段失败会停止并返回非零。

两步验收使用 batch 1/卡、关闭 validation，不能代替 5000 步数值对齐。默认完整 smoke 检查点写 `/tmp`，检查完删除大分片、保留诊断报告。要保留权重，可在 `cluster.env` 中设置**尚不存在**的 `NPU_WAM_OUTPUT_ROOT=/持久卷/全新目录`，该文件系统至少有 40 GiB 空闲。`--checks-only` 只覆盖算子和两卡小模型，不执行真实 WAM。

## 5. 单机八卡原配置 5000 步训练

先完成上面的八卡验收。正式实验从原始 `BASE_CHECKPOINT_PATH` 开始，使用全新输出目录。训练参数为：batch 16/卡、global batch 128、梯度累积 1、seed 42、BF16、学习率周期 40000、warmup 50；5000 步结束不意味着把调度周期改成 5000。

```bash
cd /目标卷/cosmos-training-litegen-910b
source config/cluster.env
source scripts/activate.sh
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export MAX_ITER=5000 SAVE_ITER=5000 BATCH_SIZE=16 GLOBAL_BATCH_SIZE=128 GRAD_ACCUM_ITER=1
export CYCLE_LENGTH=40000 TRAIN_SEED=42 TRAIN_WORKERS=4
export COMPILE=false RUN_VALIDATION=true VALIDATION_ITER=2000 LOGGING_ITER=10
export BASELINE_MONITOR=1 BENCH_WARMUP_STEPS=50 ALLOW_RESUME=0
export JOB_NAME=sim_task27_wam_910b_align5k
export OUTPUT_ROOT="$PWD/outputs/wam-5000-$(date -u +%Y%m%dT%H%M%SZ)/output"
bash train.sh
```

`train.sh` 依次读取 `config/cluster.env`、可选的 `config/train.env`，激活环境后调用已有的 `launch_wam_litegen.sh`；后者复用原始 WAM launcher 与训练配置，不另写一套训练循环。`--dry-run` 只解析配置；正式训练需要去掉该选项。不会静默回退 CPU。`ALLOW_RESUME=1` 才允许从同一输出目录的已有检查点续训；有意续训前应确认检查点、拓扑和数值配置一致。

### 后台运行，关闭终端不影响任务

执行上一段的 `source` 和 `export`，**将最后一行训练命令替换为以下段落**。`OUTPUT_ROOT` 仍须为新路径：

```bash
export RUN_DIR="$(dirname "$OUTPUT_ROOT")/launcher-node-${NODE_RANK:-0}"
mkdir -p "$RUN_DIR"
nohup setsid bash -c '
  date -u +%FT%TZ > "$RUN_DIR/started_at.txt"
  printf "running\n" > "$RUN_DIR/status.txt"
  bash train.sh
  rc=$?
  printf "%s\n" "$rc" > "$RUN_DIR/exit_code.txt"
  date -u +%FT%TZ > "$RUN_DIR/finished_at.txt"
  if [ "$rc" -eq 0 ]; then
    printf "training_completed\n" > "$RUN_DIR/status.txt"
  else
    printf "failed\n" > "$RUN_DIR/status.txt"
  fi
  exit "$rc"
' > "$RUN_DIR/console.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_DIR/launcher.pid"
printf 'Run directory: %s\n' "$RUN_DIR"
```

`nohup + setsid` 脱离当前终端，但不能抵御容器回收、节点重启或平台任务超时；长期任务应保留平台作业。`training_completed` 只表示启动器正常退出，必须继续核对 step=5000、检查点与下节对齐结果。强制杀进程可能来不及写退出状态，不能只看 PID 或进程消失判断成功。

```bash
tail -n 50 "$RUN_DIR/console.log"
cat "$RUN_DIR/status.txt"
tail -n 3 "$OUTPUT_ROOT/metrics.csv"
"$COSMOS_NPU_ENV/bin/tensorboard" --logdir "$OUTPUT_ROOT/tensorboard" --host 127.0.0.1 --port 6006
```

正式输出包括 metrics、TensorBoard、性能报告、最终有效配置及 `cosmos3_action/action_sft/<job>/checkpoints/iter_000005000`。上述命令设置 `SAVE_ITER=5000`，因此常规中间保存不发生；同一任务的保留策略会清理较早检查点，跨不同输出目录不会互相清理。多个检查点同时存在时空间需求会超过 40 GiB。

## 6. 四节点 32 卡正式训练

### 提前准备

1. **申请资源与镜像。** 申请 4 个节点、每节点 8 张 910B，使用第 1 节的 A2 CANN 9.1.0 镜像。各节点具备匹配的原生 ARM Python 3.12、CANN/ops、驱动、设备映射和跨节点 HCCL 通信条件。
2. **准备共享挂载。** 所有节点在相同绝对路径访问项目、外部 LiteGen、Python 环境、模型和数据；输出目录也放在共享卷上，完整 DCP 所在文件系统至少预留 40 GiB。将实际路径写入 `config/cluster.env`，并按第 2 节完成源码、版本锁与输入检查。
3. **提前安装共享 venv。** 按[共享环境说明](docs/shared_environment.md)，在一个准备容器中将环境安装到最终共享路径，完成依赖检查后封存。其他节点使用相同的基础 Python 路径和系统软件栈，任务启动时只读取环境，不重复安装。
4. **完成提交前验证。** 按[多节点说明](docs/distributed_training.md)完成各节点本地预检和四节点两步验证。平台工作目录设置为共享卷上的项目根目录，并为本次正式训练选择一个新的共享输出目录名。

平台应在每个节点执行一次同一份命令，并自动提供以下变量：

| 变量 | 要求 |
|---|---|
| `WORLD_SIZE` | 节点总数，本例为 `4` |
| `RANK` | 当前节点编号，分别为 `0`、`1`、`2`、`3` |
| `MASTER_ADDR`、`MASTER_PORT` | 各节点一致且可达的主节点地址、端口 |

若平台提供 `SENSECORE_PYTORCH_NNODES` 和 `SENSECORE_PYTORCH_NODE_RANK`，入口也支持这组变量。以上节点编号由平台分配，无需逐节点修改命令。

### 平台提交命令

**将平台工作目录设为项目根目录，只提交下面一份命令。** 若平台不能配置工作目录，在最前面加 `cd /目标共享卷/workspace/cosmos-training-litegen-910b`。所有节点使用相同的 `OUTPUT_ROOT`，每次新实验更换目录名。

```bash
set -euo pipefail
# 平台工作目录设置为共享卷上的本项目根目录。
export TRAIN_ENV="$PWD/config/train.env.example"
export OUTPUT_ROOT="$PWD/outputs/wam-910b-32card-5000-run01"
export NPROC_PER_NODE=8
export GLOBAL_BATCH_SIZE=128 GRAD_ACCUM_ITER=1
export MAX_ITER=5000 SAVE_ITER=5000 CYCLE_LENGTH=40000
export TRAIN_SEED=42 TRAIN_WORKERS=4
export COMPILE=false RUN_VALIDATION=true VALIDATION_ITER=2000 LOGGING_ITER=10
export BASELINE_MONITOR=1 BENCH_WARMUP_STEPS=50 ALLOW_RESUME=0
export JOB_NAME=sim_task27_wam_task_baseline
unset BATCH_SIZE NNODES NODE_RANK
exec bash scripts/submit_sensecore.sh
```

该命令运行 **4 节点 × 8 卡、5000 步、global batch 128、每卡 batch 4**，梯度累积为 1，学习率周期为 40000。`TRAIN_ENV` 使用随仓库交付的默认模板，具体环境和输入路径由 `config/cluster.env` 提供。

平台负责在各节点执行命令，入口负责启动本节点训练进程；提交时直接使用上述脚本即可。环境与设备检查报告位于 `<OUTPUT_ROOT>/setup/<hostname>/`，训练日志位于 `<OUTPUT_ROOT>/logs/`。训练结束后的检查命令见[多节点训练说明](docs/distributed_training.md#正式训练与恢复)。

## 7. 检查训练输出

训练结束后，先确认所有节点正常退出，再核验最终步数、三类 loss、实际 NPU rank 数和完整检查点。单机八卡命令如下；四节点 32 卡将 `--world-size` 改为 `32`：

```bash
source config/cluster.env
source scripts/activate.sh
"$TRAIN_PYTHON" -m migration.npu.checkpoint_verify "$OUTPUT_ROOT" \
  --world-size 8 --steps 5000 --report "$OUTPUT_ROOT/verification.json"
```

核验成功时 `verification.json` 的 `status` 为 `passed`。两步功能验收与完整 5000 步训练分开检查，不能互相替代。

四组实验的测试结果见 [reference/comparison-four-runs/experiment-comparison.md](reference/comparison-four-runs/experiment-comparison.md)。

## 8. 目录说明

```text
run.sh                    两步验收 / 输入检查 / 环境安装统一入口
train.sh                  每节点执行一次的可配置训练入口
scripts/                  bootstrap、activate、输入和源码校验
config/cluster.env.example 目标集群配置模板
config/train.env.example   节点拓扑、全局 batch 和训练参数模板
external_inputs.json      外部权重和缓存清单
MANIFEST.json             本交付文件 SHA256
cosmos-training/          Cosmos 运行时、原配置与 LiteGen 桥接
migration/                WAM 启动、监控、检查点管理、结果比较与 NPU 验收工具
config/litegen.lock.json   外部 LiteGen 仓库、分支与固定提交
environment/              依赖声明 requirements.in 与安装锁文件 requirements.lock
reference/                四组 5000 步实验数据及 comparison-four-runs/ 对比报告，无历史权重
docs/                     跨仓库衔接、共享环境和多节点训练操作说明
outputs/                  本地新运行输出；不进入迁移压缩包
```

`environment/requirements.in` 保留直接依赖声明，供后续维护锁文件；交付安装仅使用 `environment/requirements.lock` 中锁定的版本和哈希。Python 环境由镜像原生 Python 3.12 创建为 venv。

`MANIFEST.json` 记录交付文件哈希，`config/litegen.lock.json` 固定平台版本。Cosmos 及附属运行时许可证保留在项目内；LiteGen 的许可证与平台修改说明由独立仓库维护。
