# Cosmos 4B WAM × LiteGen × Ascend 910B

将 Cosmos 4B WAM 训练任务接入 LiteGen，已在 **Linux aarch64 / Python 3.12 / 单机 8 × Ascend910B2 64 GiB** 上验证，并提供固定节点数的多机训练入口。数据、模型、loss、训练循环、scheduler 和检查点由 Cosmos 管理；LiteGen 提供设备与分布式接入、FSDP2 包装和 NPU 优化器支持。

**已完成真实八卡两步功能验收和完整 5000 步训练。** 相对原生 8 × A100 基线，5000 步的 20 项对齐检查全部通过；三类 loss 全程均值相对差为 0.073%、0.037%、0.318%。这是本任务的 loss 曲线工程对齐，不是逐位一致、所有 Cosmos 分支兼容或相同训练速度。

- [四组 5000 步实验对比报告](reference/comparison-four-runs/experiment_comparison.md)：完整 loss 曲线、相对基线差异图、rank 随机数机制说明、性能曲线与显存汇总，包含可重现绘图脚本。
- [历史国产卡适配实验说明](https://github.com/NaOH678/cosmos-training/blob/470fdd883dc28c850c30f553f08170096ac78da5/docs/experiment_910b.md)：原八卡实验记录；当前四组实验数据与趋势比较命令见下文 `reference/` 说明。
- [5000 步完整报告](https://github.com/NaOH678/cosmos-training/blob/d49e5f14bea617a6a5bd59ea210cc063bdee09f1/evidence/5000-step/full_comparison.json)、[loss 对比图](https://github.com/NaOH678/cosmos-training/blob/d49e5f14bea617a6a5bd59ea210cc063bdee09f1/evidence/5000-step/loss_comparison.png)。
- [跨仓库衔接与版本升级](docs/litegen_integration.md)：LiteGen 分支链接、固定提交、克隆、安装及升级流程。
- [LiteGen 分支修改说明](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/docs/cosmos_integration.md)：平台侧文件修改、接口契约、初始化修复与验证范围，由 LiteGen 仓库维护。
- [多节点、多卡训练](docs/distributed_training.md)：统一 `train.sh` 入口、四节点 32 卡配置、两步训练和结果检查；已完成实验的数据与比较边界见下文 `reference/` 说明。
- [预先安装的共享环境](docs/shared_environment.md)：一次安装、各节点只校验和读取，避免作业启动时重复访问包源；提供可直接提交的平台入口。

训练入口分为两类：`bash run.sh` 保留单机八卡验收；`bash train.sh` 用于可配置的单机/多机正式训练，`bash train.sh --smoke` 用于指定拓扑的两步训练。环境/挂载配置在 `config/cluster.env`，可选训练拓扑配置在 `config/train.env`（从 `config/train.env.example` 复制）。

本目录包含独立可安装源码、哈希锁定依赖、A100/910B 实验数据和验收脚本；**模型权重、约 200 GiB 以上的数据缓存、35 GiB 级训练检查点、CANN/驱动和 Python 环境须另外准备**。LiteGen 作为独立仓库维护，不随本项目或压缩包分发；运行时通过 `LITEGEN_ROOT` 引用外部 checkout。随附报告中的历史绝对路径用于追溯，不是运行依赖。

历史实验日志、报告附件和曲线已从交付目录移除，文档中的相关链接指向固定 Git 提交的归档。查看这些附件需要访问 GitHub；安装、训练、两步验收和新的 loss 对比不依赖这些历史附件。

## 1. 先准备目标节点

使用以下已验证的基础镜像：

```text
registry2.d.pjlab.org.cn/ccr-lepton-official-images/a2-cann:9.1.0-a2-ubuntu22.04-py3.12
```

验证环境与安装要求如下。

| 项目 | 实测环境 / 要求 |
|---|---|
| CPU / 系统 | Kunpeng-920，aarch64，Ubuntu 22.04.5 LTS，glibc 2.35 |
| 设备 | 8 × Ascend910B2，65536 MiB HBM/卡，驱动设备映射完整 |
| CANN / ops | 9.1.0，**910B 对应的 ops** |
| 驱动 / 固件 | 25.5.6 / 7.8.0.10.220 |
| 原生 Python | 实测 3.12.13；通过 `NPU_BOOTSTRAP_PYTHON` 指定目标镜像的 ARM Python 3.12 |
| PyTorch / TorchNPU | `2.10.0+cpu` ARM wheel / `2.10.0.post6` |
| 环境创建 | 镜像原生 ARM Python 3.12 + venv；146 个哈希锁定 Python 分发包 |
| 容器资源参考 | CPU quota 144 核，内存上限 1920 GiB，共享内存 128 GiB；这是本机配置，非最小资源结论 |
| 存储 | venv 安装预留至少 5 GiB；单个完整 DCP 至少 40 GiB 可用，正式输出建议预留 80 GiB 以上 |
| 其他 | bash、venv/pip、flock、可用的 Python 包源、HCCL 通信条件；共享卷用于保存输出 |

`torch` 版本中的 `+cpu` 是本次 ARM 分发包标识，TorchNPU 注册后训练运行在 NPU；不能仅凭版本后缀判断执行设备。验收会实际执行 NPU BF16 算子。

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

`WORKSPACE_ROOT` 由接收方选择，项目目录可重命名；脚本根据自身位置定位源码。Python路径按镜像实际位置调整。固定提交用于重现已验收状态；开发时可切回 LiteGen 的 `lmr-cosmos` 分支。具体链接与升级流程见[衔接说明](docs/litegen_integration.md)。如果使用源码压缩包，仍需单独克隆 LiteGen。压缩包不含本机 `config/cluster.env`；源码目录中的该文件只适用于当前集群。

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

迁移的是源码与安装流程，venv 必须在目标最终位置重新创建；共享 venv 的各节点需要相同基础 Python 路径和挂载位置。`external_inputs.json`、`reference/` 和历史实验报告中的来源绝对路径用于追溯，不是当前安装目录；输入检查使用本地配置覆盖后的路径。

## 3. 安装环境并检查输入

```bash
cd /目标卷/cosmos-training-litegen-910b
mkdir -p outputs/setup
set -o pipefail
bash run.sh --verify 2>&1 | tee outputs/setup/verify.log
bash run.sh --check-inputs --all-payloads 2>&1 | tee outputs/setup/inputs.log
bash run.sh --bootstrap-only 2>&1 | tee outputs/setup/bootstrap.log
```

- `--verify` 校验当前交付的 `MANIFEST.json`，不依赖旧迁移包的历史清单。
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

先完成上面的八卡验收。正式实验从原始 `BASE_CHECKPOINT_PATH` 开始，使用全新输出目录。下面覆盖的是已验证的实验参数：batch 16/卡、global batch 128、梯度累积 1、seed 42、BF16、学习率周期 40000、warmup 50；5000 步结束不意味着把调度周期改成 5000。

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

正式输出包括 metrics、TensorBoard、性能报告、最终有效配置及 `cosmos3_action/action_sft/<job>/checkpoints/iter_000005000`。本实验 `SAVE_ITER=5000`，因此常规中间保存不发生；同一任务的保留策略会清理较早检查点，跨不同输出目录不会互相清理。多个检查点同时存在时空间需求会超过 40 GiB。

### 扩展到四节点 32 卡

所有节点准备相同版本的两个仓库、环境和输入，输出必须是同一共享目录。先按[多节点说明](docs/distributed_training.md)完成各节点预检及 32 卡两步训练，再在四个节点各执行下面的正式训练命令。替换主节点地址和共享路径；`NODE_RANK` 分别设置为 0、1、2、3：

```bash
cd /目标卷/cosmos-training-litegen-910b
export NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29644
export OUTPUT_ROOT=/共享卷/outputs/wam-32card-5000-001
export GLOBAL_BATCH_SIZE=128 GRAD_ACCUM_ITER=1
export MAX_ITER=5000 SAVE_ITER=5000
unset BATCH_SIZE  # 根据全局 batch 自动得到每卡 batch 4
bash train.sh
```

多节点不会自动登录其他机器启动进程；上面的命令须由平台或使用者在每节点各执行一次。后台运行时沿用上面的包装，保持共享 `OUTPUT_ROOT`，外层 `RUN_DIR` 按节点编号区分，避免覆盖其他节点的日志和退出状态。

| 拓扑 | 默认正式训练每卡 batch | 全局 batch | 全局 FSDP2 分片数 |
|---|---:|---:|---:|
| 1 节点 × 8 卡 | 16 | 128 | 8 |
| 2 节点 × 8 卡 | 8 | 128 | 16 |
| 4 节点 × 8 卡 | 4 | 128 | 32 |

以上默认值适用于梯度累积 1、未显式指定每卡 batch 的正式训练。`--smoke` 默认每卡 batch 1；全局 batch 不能整除总卡数与梯度累积次数的乘积时会报错。详细参数、每节点配置及 `--world-size 32 --steps 2/5000` 结果检查命令见[多节点训练文档](docs/distributed_training.md)。

多节点入口此前完成 74 项回归、8/16/32 rank 配置解析和真实八卡两步验收，记录位于 `outputs/multinode-validation/validation_summary.json`。后续四节点 32 × 910B 的 `job-mrlv-sys5` 已完成 5000 步训练及输出检查，原始指标已收入下文的 `reference/`，可用于比较 loss 下降趋势和训练性能。

## 6. 与 A100 做完整对齐

```bash
"$TRAIN_PYTHON" cosmos-training/mrlv/compare_wam_cross_device.py \
  --baseline reference/wam-steps5k-8xA100 \
  --candidate "$OUTPUT_ROOT" \
  --report "$RUN_DIR/alignment.json"
```

保持原门槛：三类 loss 全程均值相对差 ≤5%，每 500 步窗口均值相对差 ≤10%；比较完整 5000 步、有效配置、动作归一化及运行元数据。只允许 `compile.enabled` 不同。性能单独报告，不把 910B 吞吐强行套用 A100 速度门槛。

`reference/` 保存以下四组实验数据：

| 目录 | 实验 |
|---|---|
| `wam-steps5k-8xA100/` | 原生 Cosmos，8 × A100，5000 步；上述命令的默认对照 |
| `wam-steps5k-8xA100-LiteGen/` | 接入 LiteGen 的 Cosmos，8 × A100，5000 步 |
| `wam-steps5k-8x910B-LiteGen/` | 接入 LiteGen 的 Cosmos，单节点 8 × 910B，5000 步；每卡 batch 16 |
| `wam-steps5k-32x910B-LiteGen/` | 接入 LiteGen 的 Cosmos，四节点 32 × 910B，5000 步；每卡 batch 4 |

四组 global batch 均为 128、梯度累积均为 1。每组保留以下五类原始文件，内容及 CSV 行尾不改写，不包含历史检查点或 TensorBoard event：

```text
<实验目录>/
├── action_stats.json
├── metrics.csv
├── performance.json
├── performance_windows.csv
└── cosmos3_action/action_sft/<原实验名>/config.yaml
```

`metrics.csv` 每 10 步记录一次，共 500 个点；性能汇总排除 warmup、validation 和 checkpoint 窗口，统计 4920 步。8 卡 NPU 数据来自 `wam-910b-8x-5000-20261105T131003Z/output`，32 卡数据来自 `job-mrlv-sys5` 的 `wam-910b-multinode-5000-run02`。配置中的历史绝对路径仅供追溯，读取这些指标不需要原模型、缓存或输出挂载。

需要以 LiteGen A100 实验为对照时，将基线目录改为 `reference/wam-steps5k-8xA100-LiteGen`。

[四组实验对比报告](reference/comparison-four-runs/experiment_comparison.md)展示三类 loss 的完整 5000 步曲线、逐窗口耗时与吞吐曲线，以及效率和显存汇总。报告直接读取上述四组数据，不需要历史检查点或原始模型挂载。使用已安装项目依赖的 Python 重新生成图表与统计：

```bash
"$TRAIN_PYTHON" reference/comparison-four-runs/generate_report.py
```

图表的 PNG、SVG 和 `summary.json` 写入 `reference/comparison-four-runs/`；该目录随源码交付，包含完整指标口径和配置差异说明。四组实验的三类 loss 均呈整体下降趋势。需要检查新训练的完整最终 DCP 时，仍使用本节开头的正式对齐命令。

此入口对应原八卡对齐实验，仍严格比较卡数、每卡 batch 和分片数。16/32 卡实验会产生这些配置差异，需要独立记录并审核跨规模比较条件；本次未放宽原比较规则。

**该严格工具还比较模型/数据路径。** 保持原挂载路径时可直接使用上述命令；迁移后路径不同，即使内容相同也会被判为配置不同。此时先核对输入内容，再对副本中的具体路径建立可审计映射，保留原始和映射后配置；不能关闭全部配置比较、改原基线或放宽 loss 门槛。两步输出不应送入该 5000 步对齐入口。

历史完整报告与曲线见本页顶部的归档链接，不随当前源码交付。新实验使用上述比较命令和保留的 `reference/` 基线生成自己的对齐报告；报告、TensorBoard 和检查点写入本地输出目录。

## 7. 文件组织与可追溯性

```text
run.sh                    两步验收 / 输入检查 / 环境安装统一入口
train.sh                  每节点执行一次的可配置训练入口
scripts/                  bootstrap、activate、输入和源码校验
config/cluster.env.example 目标集群配置模板
config/train.env.example   节点拓扑、全局 batch 和训练参数模板
external_inputs.json      外部权重和缓存清单
MANIFEST.json             本交付文件 SHA256
cosmos-training/          Cosmos 运行时、原配置、桥接、NPU 验收与训练脚本
config/litegen.lock.json   外部 LiteGen 仓库、分支和验收提交；不含平台源码
environment/              依赖声明 requirements.in 与安装锁文件 requirements.lock
reference/                四组 5000 步实验数据及 comparison-four-runs/ 对比报告，无历史权重
docs/                     跨仓库衔接、共享环境和多节点训练操作说明
outputs/                  本地新运行输出；不进入迁移压缩包
```

`environment/requirements.in` 保留直接依赖声明，供后续维护锁文件；交付安装仅使用 `environment/requirements.lock` 中锁定的版本和哈希。Python 环境由镜像原生 Python 3.12 创建为 venv。

当前交付由 Git 提交、`MANIFEST.json` 和 `config/litegen.lock.json` 记录版本与文件哈希。旧源码补丁、导出清单及复测报告保留在[历史提交](https://github.com/NaOH678/cosmos-training/tree/470fdd883dc28c850c30f553f08170096ac78da5)，不参与安装、训练或验收。Cosmos/附属运行时许可证保留在项目内，LiteGen 的许可证与平台修改说明保留在独立仓库。
