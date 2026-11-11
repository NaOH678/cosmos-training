# Cosmos 与 LiteGen 的独立仓库衔接

## 1. 两个分支的职责

| 仓库 / 分支 | 作用 |
|---|---|
| [NaOH678/cosmos-training · lmr-cosmos](https://github.com/NaOH678/cosmos-training/tree/lmr-cosmos) | 可迁移WAM项目：Cosmos源码、910B适配、依赖锁、输入检查、验收、实验数据与操作文档 |
| [DeepLink-org/LiteGen.dev · lmr-cosmos](https://github.com/DeepLink-org/LiteGen.dev/tree/lmr-cosmos) | 平台代码及[分支修改说明](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/docs/cosmos_integration.md) |

Cosmos分支根目录沿用交付项目结构，训练源码位于 `cosmos-training/`。它基于原仓库main创建新分支，main保持不变。LiteGen分支保留完整平台开发仓库，不随Cosmos源码包分发。

```text
workspace/
  LiteGen.dev/                     独立Git仓库
  cosmos-training-litegen-910b/    Cosmos lmr-cosmos分支
    cosmos-training/              Cosmos训练源码
    config/litegen.lock.json       对应的平台版本
    scripts/                      安装与激活
    run.sh                        验收入口
```

当前锁定平台提交以 [`config/litegen.lock.json`](../config/litegen.lock.json) 为准。平台文件修改、设备接口、模型物化和优化器初始化契约集中在 LiteGen 的[分支修改说明](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/docs/cosmos_integration.md)，本项目仅保留安装与运行所需的衔接流程。

## 2. 克隆与固定版本

按根目录README第2节克隆两个 `lmr-cosmos` 分支，然后将LiteGen切换到 `config/litegen.lock.json` 的完整commit。使用HTTPS需要相应仓库读权限；已经配置SSH的环境可使用 `git@github.com:DeepLink-org/LiteGen.dev.git`。

`LITEGEN_ROOT` 默认是本项目的并列目录 `../LiteGen.dev`。非标准位置在本地 `config/cluster.env` 中设置绝对路径：

```bash
export LITEGEN_ROOT=/实际路径/LiteGen.dev
```

安装和激活均检查该目录为外部Git checkout、HEAD与锁定提交一致、运行时代码与安装元数据无未提交修改。分支名可以持续前进；可复现实验绑定完整commit，不隐式使用最新分支代码。校验失败时请检查路径和版本，不要绕过校验。

## 3. 安装与运行

使用已验证的镜像 `registry2.d.pjlab.org.cn/ccr-lepton-official-images/a2-cann:9.1.0-a2-ubuntu22.04-py3.12`，检查步骤与硬件前提见[README 第1节](../README.md#1-先准备目标节点)。

```bash
cd /实际路径/cosmos-training-litegen-910b
cp config/cluster.env.example config/cluster.env
# 编辑模型、数据、原生Python、外部LiteGen路径后执行：
bash run.sh --verify
bash run.sh --check-inputs --all-payloads
bash run.sh --bootstrap-only
bash run.sh
```

bootstrap使用本项目 `environment/requirements.lock` 的146个锁定分发包，然后安装外部LiteGen的核心包（`--no-deps --no-build-isolation -e`），不启用LiteGen的CUDA extras或默认uv开发环境。平台安装元数据由LiteGen仓库维护。

激活后 `LITEGEN_DEVICE=npu`、`COSMOS_DEVICE=npu`，训练继续由Cosmos原生循环驱动。检查导入位置：

```bash
source config/cluster.env
source scripts/activate.sh
"$TRAIN_PYTHON" -c 'import litegen, cosmos_framework; print(litegen.__file__); print(cosmos_framework.__file__)'
```

`outputs/environment/installed.json` 记录实际Python版本、146个分发包版本、LiteGen仓库、分支、提交与绝对路径。Git仅在子进程清理 `LD_LIBRARY_PATH` / `LD_PRELOAD`，训练保持CANN库路径。

## 4. 后续同步平台更新

1. 在独立LiteGen仓库切回 `lmr-cosmos`，获取并合并需要的平台更新；按平台的集成契约处理冲突。
2. 提交平台修改，将其完整commit更新到本项目 `config/litegen.lock.json`，同时更新MANIFEST中的文件哈希及 `external_litegen` 记录。
3. 若只有文档变化，核对平台运行时代码和安装依赖未变，并执行源码版本锁、现有环境和文档链接检查。若运行时或安装依赖变化，重新bootstrap并执行相关回归、五项真实NPU attention、两卡恢复和八卡两步验收。
4. 需要验证新运行时的5000步训练结果时，按README执行完整实验并比较原始指标；保留两个仓库及实验各自的版本记录。

正在运行的训练使用editable源码，勿在同一checkout中切换分支或修改平台；升级使用单独checkout和环境。项目不会在启动时自动git pull。

## 5. 验证范围

交付包含四组5000步实验数据，位置和比较命令见[README](../README.md)。历史实验、旧源码来源和仓库拆分复测保留在[固定 Git 提交](https://github.com/NaOH678/cosmos-training/tree/470fdd883dc28c850c30f553f08170096ac78da5)，不参与运行；平台自身的修改与测试说明由[LiteGen 文档](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/docs/cosmos_integration.md)维护。
