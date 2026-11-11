# 预先安装的共享 venv

适用于各节点使用相同 ARM Python 3.12 镜像、相同绝对挂载路径的任务。接收方自行选择共享卷和项目目录，不要求开发机的目录名。模型和数据由本地 `config/cluster.env` 指向实际挂载；共享 Python 环境不能替代每节点的 CANN、910B ops、驱动和 NPU 设备映射。

默认布局如下，`workspace` 可位于任意目标共享卷：

```text
workspace/
  LiteGen.dev/
  cosmos-training-litegen-910b/
    .venv-npu/          目标环境创建的 venv，不进入 Git
    config/cluster.env  本地配置，不进入 Git
    outputs/           节点报告与训练输出，不进入 Git
```

## 一次准备，之后只读取

只在一个准备容器安装一次，并使用最终路径。不要移动 `/tmp` 中已有的 venv；venv 依赖基础 Python 路径，其他节点也必须具有匹配的解释器。不要在已有训练任务使用的环境内升级软件，升级时使用新的版本目录。

```bash
cd /目标共享卷/workspace/cosmos-training-litegen-910b
source config/cluster.env
# 模板默认 COSMOS_NPU_ENV=.venv-npu，相对于项目根目录。
# 如需单独的版本目录，在本地配置中指定最终共享路径后再执行。
export NPU_BOOTSTRAP_REPORT_ROOT="$PWD/outputs/environment-shared-v1"
test ! -e "$COSMOS_NPU_ENV"
mkdir -p "$NPU_BOOTSTRAP_REPORT_ROOT"
set -o pipefail
bash scripts/bootstrap.sh 2>&1 | tee "$NPU_BOOTSTRAP_REPORT_ROOT/bootstrap.log"
source scripts/activate.sh
"$TRAIN_PYTHON" -m mrlv.npu.preflight --expected-devices 8 \
  --report "$NPU_BOOTSTRAP_REPORT_ROOT/preflight.json"
```

以上准备命令在项目根目录执行，原生 Python、LiteGen 和外部输入路径来自目标集群的本地配置。`source scripts/activate.sh` 会将相对 venv 路径解析成当前项目下的绝对路径。安装仍使用原有 `requirements.lock` 的版本和哈希；不为修复包源错误而更改依赖版本。锁文件中两份源码分发包仍由锁定构建工具安装。

完成依赖检查及硬件验证后，将安装报告复制为封存标记，再移除写权限：

```bash
cp "$NPU_BOOTSTRAP_REPORT_ROOT/installed.json" "$COSMOS_NPU_ENV/.cosmos-shared"
chmod -R a-w "$COSMOS_NPU_ENV"
```

`.cosmos-shared` 表示环境已准备好供任务共享使用；不是完整内容哈希证明。`.cosmos-npu-lock` 记录依赖锁哈希和环境绝对路径。对封存环境再次调用 bootstrap，只运行依赖校验和 `pip check`，不会执行安装、删除或升级。运行账号若是 root，文件权限并不能阻止其主动修改文件，因此仍必须遵守不更新活动环境的约定。

将 `config/cluster.env` 的环境行设为下列形式，允许调用方显式选择其他版本：

```bash
export COSMOS_NPU_ENV="${COSMOS_NPU_ENV:-.venv-npu}"
# 也可改为目标共享卷上的版本目录，例如 /目标共享卷/environments/cosmos-v1。
```

保留旧环境和报告，新版本验证通过后再切换配置。LiteGen 和 Cosmos 使用 editable 安装，训练期间也不能更新这两份源码。

## 平台只提交一份启动命令

已完成的 `job-mrlv-sys5` 使用四节点、每节点八张 910B，5000 步训练的配置、平台变量映射和完整提交命令集中在 [README 第 6 节](../README.md#6-四节点-32-卡正式训练)。先按本页完成共享环境准备，再使用该节的单份命令；平台在每个节点执行一次。

平台工作目录应为共享项目根目录，所有节点使用同一个新的输出目录名，不要在各节点分别生成时间戳。命令显式指定 5000 步、global batch 128、累积 1 和学习率周期 40000，每卡 batch 自动得到 4；使用 `config/train.env.example` 避免旧本地训练配置覆盖参数。

入口读取 `config/cluster.env` 并激活环境，核对准备标记、146个依赖、`pip check` 和本节点全部910B的BF16运算，然后调用原来的 `train.sh`。任何检查失败即停止，使用管道记录日志时也保留非零退出码。运行时不下载或安装 Python 包，不创建新的 venv，不运行外层 torchrun。

依赖检查仅在子进程中清理 `PYTHONPATH`，避免CANN随附分析工具的包元数据干扰venv检查；训练和硬件预检仍保留CANN的Python路径及 `LD_LIBRARY_PATH`，不通过给venv额外安装分析工具依赖来绕过错误。

拓扑优先级为：标准 `NPROC_PER_NODE`/`NNODES`/`NODE_RANK`，其次 `SENSECORE_*`。每节点卡数缺省8；旧平台的 `WORLD_SIZE`、`RANK` 分别作为节点数、节点编号的最后来源。此映射只适用于平台在每节点启动一次的入口，不适用于已经由 torchrun 创建的训练worker。`MASTER_ADDR`、`MASTER_PORT` 必须由平台提供。

README 的正式提交命令显式设置5000步、global batch128、累积1和学习率周期40000；四节点八卡时每卡batch4，并通过 `TRAIN_ENV=config/train.env.example` 避免旧本地配置覆盖。需要两步任务时使用[多节点说明](distributed_training.md)中的独立 smoke 命令，先清除正式训练参数；直接在保留 `MAX_ITER=5000` 的命令后添加 `--smoke` 不会自动覆盖该值。加 `--dry-run` 只校验Python环境并解析训练配置，跳过NPU预检与实际训练。

每节点报告位于：

```text
<OUTPUT_ROOT>/setup/<hostname>/
  installed.json
  environment.log
  pip-check.log
  preflight.json
  preflight.log
```

预先安装共享环境后，各节点启动时只读取和检查已有依赖，不需要同时访问在线包源。迁移到新集群时，仍须按[多节点训练说明](distributed_training.md)检查每个节点的设备、共享挂载和 HCCL 通信。
