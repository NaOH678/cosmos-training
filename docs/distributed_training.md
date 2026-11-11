# 多节点、多卡训练

## 入口与范围

`train.sh` 是可配置的训练入口，每节点执行一次。`run.sh` 保留原来的单节点八卡验收。两者复用现有 WAM launcher、模型、loss、训练循环和优化器；LiteGen 继续引用独立仓库的锁定提交，不需要修改平台源码。

```text
config/cluster.env          本机 Python、CANN、LiteGen、模型和数据路径
config/train.env            可选的本地训练配置，不提交 Git
config/train.env.example    训练拓扑和 batch 的模板
train.sh                   激活环境，启动当前节点的训练进程
run.sh                     安装、输入检查、原单机八卡验收
cosmos-training/mrlv/       现有启动器、监控、比较和 NPU 验证实现
```

采用固定节点数的全局 FSDP2：`shard=NNODES*NPROC_PER_NODE`、`replicate=1`，CP/CFGP 保持 1。不提供自动申请节点、SSH 启动、训练中动态扩缩容或 HSDP 切换。

已完成单节点 8 × 910B 和四节点 32 × 910B 的 5000 步实验，loss 与性能数据保留在 [`reference/`](../reference/)。新集群仍须按下文执行各节点预检和两步测试，再开始正式训练。

## 平台作业：推荐复现 sys5 的共享环境入口

`job-mrlv-sys5` 已完成 4 节点 × 8 张 910B、5000 步训练和最终产物检查。平台无法逐节点登录时，按 [README 第 6 节](../README.md#6-四节点-32-卡正式训练)使用预装共享 venv，只提交一份 `scripts/submit_sensecore.sh` 启动命令；平台提供的 `WORLD_SIZE=4` / `RANK=0..3` 在入口转为节点拓扑，随后 torchrun 创建总计 32 个训练 rank。

下文保留各节点独立环境和手动部署方式，供不使用该平台入口的集群参考。两种环境方式都要求相同的源码、输入和共享输出；不需要同时执行两种安装流程。

## 先准备每个节点可访问的环境

按 README 准备镜像、匹配 910B 的实际 CANN/ops、Python 和同一版本的两个仓库。模型/数据路径应一致；检查点输出必须是所有节点看到的同一共享目录，不能使用节点各自的 `/tmp`。共享文件系统须支持跨节点文件锁和 DCP 文件读写。

可选择[预先安装的共享 venv](shared_environment.md)，由 `scripts/submit_sensecore.sh` 在各节点校验后使用，全程不安装依赖。若选择各节点自行安装，则每节点使用本地 venv，不要并发修改一个共享 venv。在已有配置的环境中不要覆盖 `cluster.env`：

```bash
cd /目标卷/cosmos-training-litegen-910b
# 本段选择每节点独立安装；共享环境请使用上面链接中的单次准备流程。
export COSMOS_NPU_ENV=/tmp/cosmos-npu-env
bash run.sh --verify
bash run.sh --check-inputs
bash run.sh --bootstrap-only
cp config/train.env.example config/train.env
```

各节点需有相同软件栈、足够 CPU/内存和完整驱动设备映射。平台还须提供跨节点 HCCL 网络；主节点端口可连接不能代替 HCCL 集合通信测试。通信网卡环境变量按实际集群设置，本项目不猜测网卡名称。

平台在所有节点执行同一启动命令时，可将 `bash run.sh --bootstrap-only` 放在 `bash train.sh` 之前，自动在每节点创建本地环境。共享源码不代表共享 `/tmp`；新容器不能直接使用开发容器的 `/tmp/cosmos-npu-env`。设置 `NPU_BOOTSTRAP_REPORT_ROOT="$OUTPUT_ROOT/setup/$(hostname)"` 可将各节点的 `installed.json` 分开保存，安装日志也应写入该目录。安装失败必须停止训练；使用管道记录日志时启用 `set -o pipefail`。构建依赖临时清单位于各自 venv 内，不在共享目录并发写入。

本集群作业已实际观察到旧版平台变量：`WORLD_SIZE=4` 表示节点数，`RANK=0..3` 表示节点编号；在平台入口映射为 `NNODES` 和 `NODE_RANK` 后，再调用训练脚本。不要将该含义推广到 `torchrun` 创建的训练进程，其 `WORLD_SIZE` 和 `RANK` 表示全局训练进程数和编号。

## 参数与 batch

| 参数 | 含义 / 默认值 |
|---|---|
| `NNODES` | 节点数，默认 1 |
| `NPROC_PER_NODE` | 每节点卡数，默认 8；各节点相同 |
| `NODE_RANK` | 单机默认 0；多节点必须设置不同的 0…NNODES−1 |
| `MASTER_ADDR` | 多节点必须指定所有节点可访问的节点 0 地址，拒绝 loopback |
| `MASTER_PORT` | 所有节点相同，默认 29644 |
| `OUTPUT_ROOT` | 多节点必须指定相同的绝对共享路径，新实验使用新目录 |
| `GLOBAL_BATCH_SIZE` | 正式训练未指定每卡 batch 时，默认 128 |
| `BATCH_SIZE` | 每卡 batch，默认根据全局 batch 计算 |
| `GRAD_ACCUM_ITER` | 梯度累积，默认 1 |

标准拓扑变量优先于 `SENSECORE_ACCELERATE_DEVICE_COUNT`、`SENSECORE_PYTORCH_NNODES`、`SENSECORE_PYTORCH_NODE_RANK` 别名。模板用 `${变量:-默认值}` 保留调用环境的值；自定义配置不要无条件覆盖每个节点不同的 `NODE_RANK`。

```text
GLOBAL_BATCH_SIZE = NNODES × NPROC_PER_NODE × BATCH_SIZE × GRAD_ACCUM_ITER
```

默认全局 batch 必须能够整除总卡数与累积次数的乘积，否则启动前报错。显式设置 `BATCH_SIZE` 会保留该值；同时设置 `GLOBAL_BATCH_SIZE` 时，两者必须一致。超过 128 卡时须显式选择可行的全局 batch，不会自动放大。

| 拓扑 | 正式训练每卡 batch | 全局 batch | FSDP2 分片数 |
|---|---:|---:|---:|
| 1 × 8 | 16 | 128 | 8 |
| 2 × 8 | 8 | 128 | 16 |
| 4 × 8 | 4 | 128 | 32 |

`--smoke` 默认每卡 batch 1、两步、关闭 validation、逐步记录 loss 并在第 2 步保存。显式设置的 batch、步数等仍优先；避免继承正式训练参数。正式训练默认 5000 步、保存于 5000 步，学习率周期仍为 40000；BF16、seed 42、FP32 master AdamW 和 `compile=false` 沿用原配置。

## 四节点 32 卡：本地检查与跨节点两步验证

每节点先运行本地算子与两卡恢复检查。此命令不验证跨节点通信：

```bash
bash run.sh --checks-only --report-root "$PWD/outputs/local-check-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
```

使用平台单份命令时，工作目录设为共享项目根目录，以一个新的跨节点任务提交以下命令。清除正式训练覆盖值后，`--smoke` 才会采用两步、每卡 batch 1 和关闭 validation 的默认配置：

```bash
set -euo pipefail
export TRAIN_ENV="$PWD/config/train.env.example"
export OUTPUT_ROOT="$PWD/outputs/wam-910b-32card-smoke-run01"
export NPROC_PER_NODE=8 GRAD_ACCUM_ITER=1 ALLOW_RESUME=0
unset NNODES NODE_RANK BATCH_SIZE GLOBAL_BATCH_SIZE MAX_ITER SAVE_ITER RUN_VALIDATION LOGGING_ITER
exec bash scripts/submit_sensecore.sh --smoke
```

每节点的本地检查与跨节点 smoke 是不同任务；先完成前者，再统一启动后者。

以下是能够分别管理各节点 shell 时的手动方式。设置同一组公共值，替换 IP 和目录；节点编号分别设置为 0、1、2、3：

```bash
cd /目标卷/cosmos-training-litegen-910b
export NNODES=4 NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29644
export OUTPUT_ROOT=/共享卷/outputs/wam-32card-smoke-001
export NODE_RANK=0  # 此行仅用于节点 0；其他节点使用各自的编号
```

可先在一个节点解析正式训练配置。此命令不占用训练卡、不建立跨节点通信：

```bash
OUTPUT_ROOT=/共享卷/outputs/wam-32card-config-preview bash train.sh --dry-run
```

然后由平台在四个节点各执行一次：

```bash
bash train.sh --smoke
```

每个 launcher 启动本机八个 worker，总计 32 个 rank。`torchrun` 会等待其他节点，不会自动登录远端启动进程。所有节点均须正常退出，不能只看节点 0 的退出码。launcher 日志名含节点编号，保存在 `OUTPUT_ROOT/logs/`；TensorBoard 和训练汇总由全局 rank 0 写入。

全部结束后，在一个节点检查最终产物：

```bash
source config/cluster.env
source scripts/activate.sh
"$TRAIN_PYTHON" -m mrlv.npu.checkpoint_verify "$OUTPUT_ROOT" \
  --world-size 32 --steps 2 --report "$OUTPUT_ROOT/verification.json"
```

检查器核对 NPU rank 数、LiteGen FSDP2、完整有限 loss 序列、最终步数、model/optim/scheduler/trainer 四类 DCP 组件、metadata 引用与字节边界，并回读 scheduler/trainer 和全部 rank 的 RNG 状态。失败返回非零并写 `status=failed`。该报告只覆盖训练产物，不替代算子测试，也不是 5000 步 loss 对齐证明。

## 正式训练与恢复

四节点保留相同拓扑和各自的 `NODE_RANK`，使用全新的共享输出目录，去掉 `--smoke`：

```bash
export OUTPUT_ROOT=/共享卷/outputs/wam-32card-5000-001
export GLOBAL_BATCH_SIZE=128
bash train.sh
```

确认没有继承 `BATCH_SIZE=16` 或 smoke 的步数等覆盖值。如需后台运行，使用 README 的 `nohup setsid` 包装，每节点各执行一次。共享 `OUTPUT_ROOT` 保持相同，外层日志/退出状态目录 `RUN_DIR` 按 `NODE_RANK` 区分，避免互相覆盖。长期任务优先由平台管理；后台运行不能抵御容器回收或节点故障。

完成后使用同一个检查器，步数应能被日志间隔整除：

```bash
"$TRAIN_PYTHON" -m mrlv.npu.checkpoint_verify "$OUTPUT_ROOT" \
  --world-size 32 --steps 5000 --report "$OUTPUT_ROOT/verification.json"
```

默认拒绝含有可恢复检查点的输出目录。`ALLOW_RESUME=1` 仅用于相同拓扑、batch、代码和数值配置下的中断恢复；本次未实现或验证从 8 卡完整恢复为 32 卡的 RNG/数据进度语义。改变规模后，从相同原始 `BASE_CHECKPOINT_PATH` 开始新实验。

## 数值和性能比较的边界

全局 batch 128 保持每个 optimizer step 的样本数，但 rank 数、随机数流和归约顺序仍变化，不能预先承诺曲线逐点一致或四倍加速。全局 FSDP2 会跨节点收集参数、归约梯度，须实测集群网络上的性能。

交付中的四组实验数据、完整曲线、差异图及 rank 随机数机制说明见[实验对比报告](../reference/comparison-four-runs/experiment_comparison.md)，重现命令见 [README](../README.md)。跨规模对比时记录卡数、每卡 batch 和全局 batch，使用完整 loss 记录与同口径的性能窗口；正式验收工具保留原有检查条件。
