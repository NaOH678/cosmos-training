# Cosmos 4B WAM × LiteGen × Ascend 910B

将 Cosmos 的 task27 纯 WAM 4B 训练任务接入 LiteGen，在 **Linux aarch64 / Python 3.12 / 单机 8 × Ascend910B2 64 GiB** 上运行。数据、模型、loss、训练循环、scheduler 和检查点由 Cosmos 管理；LiteGen 提供设备与分布式接入、FSDP2 包装和 NPU 优化器支持。

**已完成真实八卡两步功能验收和完整 5000 步训练。** 相对原生 8 × A100 基线，5000 步的 20 项对齐检查全部通过；三类 loss 全程均值相对差为 0.073%、0.037%、0.318%。这是本任务的 loss 曲线工程对齐，不是逐位一致、所有 Cosmos 分支兼容或相同训练速度。

- [国产卡适配实验说明](docs/experiment_910b.md)：配置、接入职责、改动、完整 loss/性能对比及验证边界。
- [5000 步完整报告](evidence/5000-step/full_comparison.json)、[loss 对比图](evidence/5000-step/loss_comparison.png)。
- [交付目录复测记录](docs/delivery_validation.md)：独立环境、新路径验收与可追溯来源。
- [跨仓库衔接与版本升级](docs/litegen_integration.md)：LiteGen 分支链接、固定提交、克隆、安装及升级流程。
- [LiteGen 实际改动](docs/litegen_changes.md)：与原 LiteGen 目录逐文件比较。

本目录包含独立可安装源码、哈希锁定依赖、A100 基线、验收脚本和实验小型证据；**模型权重、约 200 GiB 以上的数据缓存、35 GiB 级训练检查点、CANN/驱动和 Python 环境须另外准备**。LiteGen 作为独立仓库维护，不随本项目或压缩包分发；运行时通过 `LITEGEN_ROOT` 引用外部 checkout。随附报告中的历史绝对路径用于追溯，不是运行依赖。

## 1. 先准备目标节点

平台任务使用以下基础镜像（由当前集群使用者指定）：

```text
registry2.d.pjlab.org.cn/ccr-lepton-official-images/a3-cann:9.2.0-a3-ubuntu22.04-py3.12
```

**镜像标签与容器内实际软件栈分别记录。** 当前容器内查询到的 CANN runtime 和 `Ascend-cann-910b-ops` 均为 **9.1.0**；已有两步与5000步结果基于这一实际安装组合。`9.2.0-a3` 标签本身不能证明镜像原装 ops 适用于910B，也不能作为“已验证原装CANN 9.2.0”的依据。新容器必须检查实际版本及芯片包；如仍是A3/910_93 ops，应由平台准备匹配910B的 CANN/ops 后再运行验收。Python bootstrap 不负责替换这些系统组件。

以下是已经实测的组合，不是对所有同系列版本的兼容承诺。

| 项目 | 实测环境 / 要求 |
|---|---|
| CPU / 系统 | Kunpeng-920，aarch64，Ubuntu 22.04.5 LTS，glibc 2.35 |
| 设备 | 8 × Ascend910B2，65536 MiB HBM/卡，驱动设备映射完整 |
| CANN / ops | 9.1.0，**910B 对应的 ops** |
| 驱动 / 固件 | 25.5.6 / 7.8.0.10.220 |
| 原生 Python | 3.12.13；本机 `/usr/local/python3.12.13/bin/python3.12` |
| PyTorch / TorchNPU | `2.10.0+cpu` ARM wheel / `2.10.0.post6` |
| 环境创建 | 镜像原生 ARM Python 3.12 + venv；146 个哈希锁定 Python 分发包 |
| 容器资源参考 | CPU quota 144 核，内存上限 1920 GiB，共享内存 128 GiB；这是本机配置，非最小资源结论 |
| 存储 | venv 安装预留至少 5 GiB；单个完整 DCP 至少 40 GiB 可用，正式输出建议预留 80 GiB 以上 |
| 其他 | bash、venv/pip、flock、可用的 Python 包源、HCCL 通信条件；共享卷用于保存输出 |

`torch` 版本中的 `+cpu` 是本次 ARM 分发包标识，TorchNPU 注册后训练运行在 NPU；不能仅凭版本后缀判断执行设备。验收会实际执行 NPU BF16 算子。

平台应分配一台含八张卡的任务容器，**只启动一次根目录的脚本**，由脚本创建训练 rank。不要让平台再启动八份同一脚本。检查 `npu-smi info`、`/dev/davinci0` 至 `/dev/davinci7`、`/dev/davinci_manager`，以及平台要求的辅助设备和驱动挂载。

```bash
uname -m
cat /etc/os-release
ldd --version
npu-smi info
df -h . /tmp /dev/shm
```

宿主驱动、固件和匹配的 CANN/ops 由集群管理员或镜像提供。项目脚本不会自动升级这些系统组件，也不会把 910_93/A3 ops 当成 910B ops。

## 2. 迁移源码和外部输入

从两个独立分支克隆，保持并列目录。以下 Git 命令只在其子进程中清理动态库路径，训练仍保留 CANN 环境：

```bash
cd /目标卷
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git clone --branch lmr-cosmos \
  https://github.com/NaOH678/cosmos-training.git cosmos-training-litegen-910b
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git clone --branch lmr-cosmos \
  https://github.com/DeepLink-org/LiteGen.dev.git LiteGen.dev
cd cosmos-training-litegen-910b
LITEGEN_COMMIT=$(/usr/local/python3.12.13/bin/python3.12 -c 'import json; print(json.load(open("config/litegen.lock.json"))["commit"])')
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git -C ../LiteGen.dev checkout --detach "$LITEGEN_COMMIT"
```

Python路径按镜像实际位置调整。固定提交用于重现已验收状态；开发时可切回 LiteGen 的 `lmr-cosmos` 分支。具体链接与升级流程见[衔接说明](docs/litegen_integration.md)。如果使用源码压缩包，仍需单独克隆 LiteGen。压缩包不含本机 `config/cluster.env`；源码目录中的该文件只适用于当前集群。

```bash
cd /目标卷/cosmos-training-litegen-910b
cp config/cluster.env.example config/cluster.env
```

编辑 `config/cluster.env`，至少设置下面四项为实际挂载路径：

| 变量 | 需要的内容 |
|---|---|
| `EDGE_DROID_MODEL_PATH` | 模型 `config.json`、tokenizer、processor/preprocessor、special tokens、chat template 等配置 |
| `BASE_CHECKPOINT_PATH` | 原始预训练 DCP；包含 `model/.metadata` 和两份 `model/*.distcp`，约 6.28 GiB |
| `WAN_VAE_PATH` | `Wan2.2_VAE.pth` 文件，约 2.63 GiB |
| `SIM_POINTFK_BUNDLE` | task27 原始训练归一化统计、窗口索引、动作记录、VAE latent 和视频帧缓存 |

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

当前集群配置示例；迁移到其他机器时修改右侧路径：

```bash
export NPU_BOOTSTRAP_PYTHON=/usr/local/python3.12.13/bin/python3.12
export COSMOS_NPU_ENV=/tmp/cosmos-training-litegen-910b-env
export LITEGEN_ROOT=/目标卷/LiteGen.dev
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export EDGE_DROID_MODEL_PATH=/oss/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/oss/shichaojian/models/cosmos3-edge-droid-dcp
export WAN_VAE_PATH=/oss/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth
export SIM_POINTFK_BUNDLE=/oss/shichaojian/wam_datasets_20261029/sim_task27_ball_wam
```

`NPU_BOOTSTRAP_PYTHON` 必须指向**镜像自身可用的 ARM Python 3.12**；本机裸命令 `python3.12` 不在 PATH，不能照搬默认命令。不要移动旧 venv 或从 x86 导出二进制环境。`COSMOS_NPU_ENV` 建议使用支持可靠文件锁的本地盘；本集群 DTFS 的 `flock` 曾阻塞，所以实测使用 `/tmp`。容器重建后需要重建这个环境；模型和正式输出仍放共享卷。

## 3. 安装环境并检查输入

```bash
cd /目标卷/cosmos-training-litegen-910b
mkdir -p outputs/setup
set -o pipefail
bash run.sh --verify 2>&1 | tee outputs/setup/verify.log
bash run.sh --check-inputs --all-payloads 2>&1 | tee outputs/setup/inputs.log
bash run.sh --bootstrap-only 2>&1 | tee outputs/setup/bootstrap.log
```

- `--verify` 校验本交付的 `MANIFEST.json`。原迁移包的历史清单另存于 `provenance/original-migration-manifest.json`，不要拿旧清单校验修复后的源码。
- `--check-inputs` 检查模型/数据清单哈希、权重大小、所有 frame/latent 文件及确定性抽样的动作文件；`--all-payloads` 额外检查全部动作文件是否存在。结果在 `outputs/input_check.json`。
- `--bootstrap-only` 使用原生 Python 创建独立 venv，从 [environment/requirements.lock](environment/requirements.lock) 安装依赖，先核对外部 LiteGen 的提交和运行时修改状态，再以 `--no-deps` 安装外部 LiteGen 与本项目 Cosmos 的 editable 源码，执行 `pip check` 和所有锁定版本检查。版本报告及 LiteGen 提交/路径写入 `outputs/environment/installed.json`。
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

## 5. 原配置 5000 步训练

先完成上面的八卡验收。正式实验从原始 `BASE_CHECKPOINT_PATH` 开始，使用全新输出目录。下面覆盖的是已验证的实验参数：batch 16/卡、global batch 128、梯度累积 1、seed 42、BF16、学习率周期 40000、warmup 50；5000 步结束不意味着把调度周期改成 5000。

```bash
cd /目标卷/cosmos-training-litegen-910b
source config/cluster.env
source scripts/activate.sh
export SENSECORE_ACCELERATE_DEVICE_COUNT=8
export SENSECORE_PYTORCH_NNODES=1 SENSECORE_PYTORCH_NODE_RANK=0
export MAX_ITER=5000 SAVE_ITER=5000 BATCH_SIZE=16 GRAD_ACCUM_ITER=1
export CYCLE_LENGTH=40000 TRAIN_SEED=42 TRAIN_WORKERS=4
export COMPILE=false RUN_VALIDATION=true VALIDATION_ITER=2000 LOGGING_ITER=10
export BASELINE_MONITOR=1 BENCH_WARMUP_STEPS=50 ALLOW_RESUME=0
export JOB_NAME=sim_task27_wam_910b_align5k
export OUTPUT_ROOT="$PWD/outputs/wam-5000-$(date -u +%Y%m%dT%H%M%SZ)/output"
bash cosmos-training/mrlv/launch_wam_litegen.sh
```

已有的 `launch_wam_litegen.sh` 复用原始 WAM launcher 与训练配置，不另写一套训练循环。`--dry-run` 只解析配置；正式训练需要去掉该选项。不会静默回退 CPU。`ALLOW_RESUME=1` 才允许从同一输出目录的已有检查点续训；有意续训前应确认检查点正确且配置一致。

### 后台运行，关闭终端不影响任务

执行上一段的 `source` 和 `export`，**将最后一行训练命令替换为以下段落**。`OUTPUT_ROOT` 仍须为新路径：

```bash
export RUN_DIR="$(dirname "$OUTPUT_ROOT")"
mkdir -p "$RUN_DIR"
nohup setsid bash -c '
  date -u +%FT%TZ > "$RUN_DIR/started_at.txt"
  printf "running\n" > "$RUN_DIR/status.txt"
  bash cosmos-training/mrlv/launch_wam_litegen.sh
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

## 6. 与 A100 做完整对齐

```bash
"$TRAIN_PYTHON" cosmos-training/mrlv/compare_wam_cross_device.py \
  --baseline reference/wam-task27-align5k-01 \
  --candidate "$OUTPUT_ROOT" \
  --report "$RUN_DIR/alignment.json"
```

保持原门槛：三类 loss 全程均值相对差 ≤5%，每 500 步窗口均值相对差 ≤10%；比较完整 5000 步、有效配置、动作归一化及运行元数据。只允许 `compile.enabled` 不同。性能单独报告，不把 910B 吞吐强行套用 A100 速度门槛。

**该严格工具还比较模型/数据路径。** 保持原挂载路径时可直接使用上述命令；迁移后路径不同，即使内容相同也会被判为配置不同。此时先核对输入内容，再对副本中的具体路径建立可审计映射，保留原始和映射后配置；不能关闭全部配置比较、改原基线或放宽 loss 门槛。两步输出不应送入该 5000 步对齐入口。

已有完整结果在 `evidence/5000-step/`，包含 PNG/PDF/SVG、逐点 CSV、十个窗口统计、NPU TensorBoard 快照、原始配置和性能数据。`plot_loss_comparison.py`、`build_overview.py` 可在安装好的环境中依据同目录快照重新绘图；`verify_full_run.py` 是历史完整检查点复核脚本，需要历史源路径，不能拿无权重的交付包假装再次验证全部分片。

## 7. 实际装配遇到的坑

| 问题 | 原因与本项目处理 |
|---|---|
| 910B 机器上的算子包不匹配 | 原镜像有 910_93 ops；换为同版本官方 910B ops，驱动未改。不能只凭镜像标签认定支持当前芯片。 |
| Git 报 glibc/库版本错误 | 仅 Git 子进程用 `env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git ...`；不对训练全局清空库路径。 |
| 共享盘创建 venv 卡住 | DTFS 上 `flock` 阻塞；改用本地 `/tmp` 环境。不要删除正在使用的锁或同时安装同一 venv。 |
| Python 版本/架构错误 | 显式指定镜像原生 ARM Python 3.12，bootstrap 验证架构和版本。环境不能跨架构或直接搬路径。 |
| CANN 的 Python 路径污染安装检查 | bootstrap 的安装子进程清理 `PYTHONPATH`；训练激活重新加载 CANN，保留其动态库路径。 |
| 全 wheel 安装失败 | `antlr4-python3-runtime==4.9.3`、`iopath==0.1.10` 使用锁定 sdist；仅这两项允许源码构建，安装固定构建工具并关闭隔离构建，保留哈希检查。 |
| 上游 extras 拉取 Apex/FlashAttention 等 CUDA 包 | 使用包内 NPU 安装元数据及统一锁文件；不要额外安装完整 CUDA extras。 |
| 无关 LeRobot 导入阻塞 WAM | 数据集按需导入；同时适配 wandb 的 runid 导入路径，补齐 iopath/portalocker/pytz。 |
| DCP 元数据含 `pathlib._local` | 加入 Python 3.12 的元数据兼容读取。检查点来源必须可信，DCP 元数据不是不可信输入解析接口。 |
| 初始 loss 与预训练起点不符 | 发现 FP32 master 在 DCP 加载前复制随机权重；修复为首次有梯度更新时创建 master，完整恢复保留已有 FP32 状态。详见实验说明。 |
| scheduler 七个空分片被误判丢失 | DCP 对复制状态去重，rank 0 保存实际内容；依据 metadata 引用、字节边界和回读结果判断，不能要求每个分片非空。 |
| 意外续训破坏起点 | 使用全新输出目录；launcher 默认拒绝已有可恢复检查点，显式 `ALLOW_RESUME=1` 才允许恢复。 |
| 更换源码目录却加载旧代码 | editable 安装与路径绑定；新位置重新创建环境并检查两份包的 `__file__`。 |

不要从旧的 `wam-checkpoint-20261105T092827Z` 或 `...T094018Z` 两步检查点继续正式训练：它们早于 master 初始化修复，仅是故障排查证据。本交付使用修复后完成 5000 步的源码。

## 8. 文件组织与可追溯性

```text
run.sh                    两步验收 / 输入检查 / 环境安装统一入口
scripts/                  bootstrap、activate、输入和源码校验
config/cluster.env.example 目标集群配置模板
external_inputs.json      外部权重和缓存清单
MANIFEST.json             本交付文件 SHA256
cosmos-training/          Cosmos 运行时、原配置、桥接、NPU 验收与训练脚本
config/litegen.lock.json   外部 LiteGen 仓库、分支和验收提交；不含平台源码
environment/              固定依赖和环境资料
reference/                原生 A100 5000 步基线，无历史权重
docs/                     实验、LiteGen 改动、新目录复测说明
evidence/                 已归档的报告、日志、曲线及小型原始数据
provenance/               来源提交、补丁、原清单和源码快照哈希
outputs/                  本地新运行输出；不进入迁移压缩包
```

本次拆分保持 Cosmos 训练运行时、数值配置和测试阈值不变；安装与激活入口改为校验并引用外部 LiteGen。原迁移仓库 HEAD 为 `60655e9c7bb8581bab95c2470816cf150e5bc57d`，其中包含未提交修复；真实来源以 `provenance/validated-source.patch` 和 `validated-source-sha256.json` 为准。复制时 750 个交付文件全部匹配 5000 步实验快照；这些是拆分前的历史快照证据；当前文件以本分支 MANIFEST 为准。Cosmos/附属运行时许可证保留在项目内，LiteGen 的许可证保留在独立平台仓库。
