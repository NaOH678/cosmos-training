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

平台在每个节点执行一次，所有节点使用相同的新输出目录：

```bash
cd /目标共享卷/workspace/cosmos-training-litegen-910b
export OUTPUT_ROOT="$PWD/outputs/wam-32card-5000-run01"
exec bash scripts/submit_sensecore.sh
```

若平台已将工作目录设为项目根目录，可省略 `cd`；目录名字不影响启动。所有节点必须使用同一输出目录名，不要各自用时间戳生成它。

入口读取 `config/cluster.env` 并激活环境，核对准备标记、146个依赖、`pip check` 和本节点全部910B的BF16运算，然后调用原来的 `train.sh`。任何检查失败即停止，使用管道记录日志时也保留非零退出码。运行时不下载或安装 Python 包，不创建新的 venv，不运行外层 torchrun。

依赖检查仅在子进程中清理 `PYTHONPATH`，避免CANN随附分析工具的包元数据干扰venv检查；训练和硬件预检仍保留CANN的Python路径及 `LD_LIBRARY_PATH`，不通过给venv额外安装分析工具依赖来绕过错误。

拓扑优先级为：标准 `NPROC_PER_NODE`/`NNODES`/`NODE_RANK`，其次 `SENSECORE_*`。每节点卡数缺省8；旧平台的 `WORLD_SIZE`、`RANK` 分别作为节点数、节点编号的最后来源。此映射只适用于平台在每节点启动一次的入口，不适用于已经由 torchrun 创建的训练worker。`MASTER_ADDR`、`MASTER_PORT` 必须由平台提供。

正式训练沿用5000步、global batch128、累积1和学习率周期40000。四节点八卡时每卡batch4。若 `config/train.env` 或调用环境有显式训练参数，以现有launcher规则为准；不要保留旧的 `BATCH_SIZE=16`。加 `--smoke` 可执行该拓扑的两步训练；加 `--dry-run` 只校验Python环境并解析训练配置，跳过NPU预检与实际训练。

每节点报告位于：

```text
<OUTPUT_ROOT>/setup/<hostname>/
  installed.json
  environment.log
  pip-check.log
  preflight.json
  preflight.log
```

## 历史安装失败及本方案的作用

历史任务 `job-mrlv-test4` 在依赖安装阶段失败，尚未进入WAM训练。master-0及worker-2无法解析到 `aioboto3==15.5.0`，worker-1无法解析到 `setuptools==84.2.1`；worker-0却下载到了这些相同版本。日志证明各节点当时的包获取结果不同，不能据此认定锁定版本不存在，也不足以区分包源同步、代理、网络或其他配置问题。

预先安装并验证共享环境，使训练节点不再依赖在线包源，是对本次启动失败的直接规避。它不证明跨节点HCCL、32卡训练或5000步loss已经验证通过。单机的共享卷读测试也不能代替多个物理节点的验证。
