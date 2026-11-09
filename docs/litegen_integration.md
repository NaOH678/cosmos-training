# Cosmos 与 LiteGen 的独立仓库衔接

## 1. 两个分支的职责

| 仓库 / 分支 | 作用 |
|---|---|
| [NaOH678/cosmos-training · lmr-cosmos](https://github.com/NaOH678/cosmos-training/tree/lmr-cosmos) | 可迁移WAM项目：Cosmos源码、910B适配、依赖锁、输入检查、验收、实验记录 |
| [DeepLink-org/LiteGen.dev · lmr-cosmos](https://github.com/DeepLink-org/LiteGen.dev/tree/lmr-cosmos) | 从develop派生的平台适配：设备接口、延迟物化、FP32 master初始化修复 |

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

当前锁定平台提交：[84a251bacc38fd50f14f96a428ce06786ec12c37](https://github.com/DeepLink-org/LiteGen.dev/commit/84a251bacc38fd50f14f96a428ce06786ec12c37)。

## 2. 克隆与固定版本

按根目录README第2节克隆两个 `lmr-cosmos` 分支，然后将LiteGen切换到 `config/litegen.lock.json` 的完整commit。使用HTTPS需要相应仓库读权限；已经配置SSH的环境可使用 `git@github.com:DeepLink-org/LiteGen.dev.git`。

`LITEGEN_ROOT` 默认是本项目的并列目录 `../LiteGen.dev`。非标准位置在本地 `config/cluster.env` 中设置绝对路径：

```bash
export LITEGEN_ROOT=/实际路径/LiteGen.dev
```

安装和激活均检查该目录为外部Git checkout、HEAD与锁定提交一致、运行时代码与安装元数据无未提交修改。分支名可以持续前进；可复现实验绑定完整commit，不隐式使用最新分支代码。校验失败时请检查路径和版本，不要绕过校验。

## 3. 安装与运行

平台任务基础镜像为 `registry2.d.pjlab.org.cn/ccr-lepton-official-images/a3-cann:9.2.0-a3-ubuntu22.04-py3.12`。镜像标签由使用者提供；当前实测容器内CANN/910B ops为9.1.0，详细区别及硬件前提见README第1节。

```bash
cd /实际路径/cosmos-training-litegen-910b
cp config/cluster.env.example config/cluster.env
# 编辑模型、数据、原生Python、外部LiteGen路径后执行：
bash run.sh --verify
bash run.sh --check-inputs --all-payloads
bash run.sh --bootstrap-only
bash run.sh
```

bootstrap使用本项目 `environment/requirements.lock` 的146个锁定分发包，然后安装外部LiteGen的核心包（`--no-deps --no-build-isolation -e`），不启用LiteGen的CUDA extras或默认uv开发环境。平台完整 `pyproject.toml` 和 `uv.lock` 保留在平台仓库；PyYAML核心下界调整到已验证的6.0，以兼容本任务锁文件。

激活后 `LITEGEN_DEVICE=npu`、`COSMOS_DEVICE=npu`，训练继续由Cosmos原生循环驱动。检查导入位置：

```bash
source config/cluster.env
source scripts/activate.sh
"$TRAIN_PYTHON" -c 'import litegen, cosmos_framework; print(litegen.__file__); print(cosmos_framework.__file__)'
```

`outputs/environment/installed.json` 记录实际Python版本、146个分发包版本、LiteGen仓库、分支、提交与绝对路径。Git仅在子进程清理 `LD_LIBRARY_PATH` / `LD_PRELOAD`，训练保持CANN库路径。

## 4. 后续同步平台更新

1. 在独立LiteGen仓库切回 `lmr-cosmos`，获取develop的新提交并正常合并；解决冲突时保留本分支的设备接口、`materialize_meta=False` 及master延迟初始化契约。
2. 提交平台修改，执行平台单元测试；以新的commit更新本项目 `config/litegen.lock.json`，保留两个仓库各自的提交记录。
3. 重新bootstrap并执行回归、五项真实NPU attention、两卡恢复和八卡两步验收。更新本项目受版本管理文件时同时更新MANIFEST中的对应SHA256。
4. 如需声称新的平台版本仍通过5000步数值对齐，按README复跑并用原阈值比较；历史报告不能自动代表新版本。

正在运行的训练使用editable源码，勿在同一checkout中切换分支或修改平台；升级使用单独checkout和环境。项目不会在启动时自动git pull。

## 5. 验证范围

拆分前的完整5000步报告位于 `evidence/5000-step/`，平台逐文件差异记录位于 `evidence/litegen-diff/`，其中旧目录名是历史来源，不是当前安装路径。当前拆分的复测说明见 `docs/repository_split_validation.md`。
