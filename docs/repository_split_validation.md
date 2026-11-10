# 独立平台仓库拆分验证

日期：2026-11-09。当前项目通过外部LiteGen完成真实八卡两步验收，`status=passed`、`wam_executed=true`。

## 1. 仓库与版本

- LiteGen：[lmr-cosmos](https://github.com/DeepLink-org/LiteGen.dev/tree/lmr-cosmos)，提交 `84a251bacc38fd50f14f96a428ce06786ec12c37`；从develop提交 `ceff826ad729151fb385b1f4df787ebc4b5bd888` 派生，已推送。
- Cosmos：本地 `lmr-cosmos`，从main提交 `6b9705f202beb513cdefcbdcb0a9395987c5c22f` 派生。当前SSH账户 `mrlv0048` 被目标仓库拒绝写入，GitHub连接也只有读权限；远端发布待仓库Write权限。目标为 [NaOH678/cosmos-training](https://github.com/NaOH678/cosmos-training)。
- Cosmos的665个交付文件、外部LiteGen的55个Python运行时文件与此前验收快照内容一致。146包依赖锁、训练数值配置和阈值未改变。
- 项目不包含LiteGen运行时目录。安装与激活校验独立checkout及固定commit，版本信息记录在环境报告。

## 2. 本轮复测

| 检查 | 结果 | 证据 |
|---|---|---|
| LiteGen全部单元测试 | 459 passed，0失败、0跳过 | [XML](../evidence/repository-split/litegen-unit.xml) |
| 外部源码版本锁 | 6 passed | [XML](../evidence/repository-split/source-lock.xml) |
| Cosmos针对性回归 | 68 passed，0失败、0跳过 | [XML](../evidence/repository-split/cosmos-regression.xml) |
| 安装与版本 | pip check通过，146包匹配；LiteGen为外部路径 | [环境报告](../evidence/repository-split/installed.json)、[安装日志](../evidence/repository-split/bootstrap.log) |
| 输入检查 | 模型/缓存清单和payload检查通过 | [报告](../evidence/repository-split/input_check.json) |
| 八张真实910B预检 | BF16运算及同步通过 | [报告](../evidence/repository-split/acceptance/preflight.json) |
| NPU attention | 5项实际执行通过，0跳过 | [XML](../evidence/repository-split/acceptance/attention.xml) |
| 两卡FSDP2恢复 | FP32 master、RNG/DCP恢复及恢复后更新通过 | [报告](../evidence/repository-split/acceptance/fsdp-dcp/result.json) |
| 八卡真实4B WAM | 三类loss有限，最终step=2 | [报告](../evidence/repository-split/acceptance/wam_result.json) |
| 四类DCP组件 | model、optim、scheduler、trainer完整性检查通过 | [报告](../evidence/repository-split/acceptance/checkpoint_schema.json) |

[总状态](../evidence/repository-split/acceptance/status.json) · [复测汇总](../evidence/repository-split/summary.json) · [总日志](../evidence/repository-split/acceptance-console.log)。

初次平台测试的NPU替身缺少卷积HF32字段，补齐后通过；移出内嵌源码后旧editable注册失效，重新安装到外部目录后通过。未通过删除测试、跳过真实算子或放宽阈值处理这些问题。

## 3. 镜像和验证边界

本次拆分验收当时记录的镜像为使用者提供的 `registry2.d.pjlab.org.cn/ccr-lepton-official-images/a3-cann:9.2.0-a3-ubuntu22.04-py3.12`；该容器修复后的实际CANN runtime及910B ops均为9.1.0。[版本原始记录](../evidence/repository-split/image-and-cann.json)。当时没有拉取镜像重建新容器，也没有验证原装CANN 9.2.0。

2026-11-10更新：使用者确认原镜像选择有误，新建910B任务应使用 `registry2.d.pjlab.org.cn/ccr-lepton-official-images/a2-cann:9.1.0-a2-ubuntu22.04-py3.12`。配置模板和迁移说明已同步；上述历史证据中的镜像字段保留原记录，不据此声称已在新A2镜像中重跑实验。

本轮复用已存在的原生ARM Python 3.12 venv，并实际执行bootstrap更新两份editable注册和依赖校验；首次建环境的历史记录见 [交付验证](delivery_validation.md)。输入检查未对所有大型payload重新做内容哈希。

本轮只复跑两步功能验收，完整5000步对齐引用原归档实验。未重新测试CUDA硬件、多机、其他模型或compile训练；不声称平台未来更新自动继承本次结果。
