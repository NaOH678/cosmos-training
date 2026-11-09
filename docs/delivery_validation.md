# 新交付目录的独立复测

日期：2026-11-08。项目：`/mnt/lvminrui/workspace/cosmos-training-litegen-910b`。验证在独立环境 `/tmp/cosmos-training-litegen-910b-env` 中完成；没有借用旧环境的 editable 注册。

## 1. 最终结果

**真实八卡两步验收通过：`status=passed`、`wam_executed=true`。** 验收器运行时间为2026-11-08 12:10:58至12:16:49 UTC。源码运行时、数值配置和测试阈值未修改。

| 项目 | 结果 | 随包证据 |
|---|---|---|
| 原生ARM Python 3.12建环境、锁文件安装 | 146个分发包匹配，pip check通过 | [bootstrap.log](../evidence/delivery-validation/bootstrap.log)、[installed.json](../evidence/delivery-validation/installed.json) |
| 导入路径与可见设备 | 两个包均来自新项目，八张NPU | [imports.json](../evidence/delivery-validation/imports.json) |
| 模型与数据挂载 | 全部清单检查通过，含全部动作payload存在性检查 | [input-check-all.json](../evidence/delivery-validation/input-check-all.json) |
| 既有针对性回归 | 55 passed，0 failed，0 skipped | [regression.xml](../evidence/delivery-validation/regression.xml)、[日志](../evidence/delivery-validation/regression.log) |
| 5000步有效配置dry-run | 与已完成NPU配置相同；与A100仅compile不同 | [config-parity.json](../evidence/delivery-validation/config-parity.json) |
| 八卡预检 | 910B BF16与同步通过 | [preflight.json](../evidence/delivery-validation/acceptance/preflight.json) |
| NPU attention | 5项执行通过、0跳过 | [attention.xml](../evidence/delivery-validation/acceptance/attention.xml) |
| 两卡FSDP2恢复 | FP32 master、RNG/DCP及恢复后更新通过 | [result.json](../evidence/delivery-validation/acceptance/fsdp-dcp/result.json) |
| 八卡真实4B WAM | step=2，三类loss有限 | [wam_result.json](../evidence/delivery-validation/acceptance/wam_result.json) |
| 完整DCP | model/optim/scheduler/trainer通过；回读step2与八份RNG | [checkpoint_schema.json](../evidence/delivery-validation/acceptance/checkpoint_schema.json) |
| README后台命令 | 模拟退出码0及7的状态/日志处理正确 | [background-command-check.json](../evidence/delivery-validation/background-command-check.json) |

[总报告](../evidence/delivery-validation/summary.json) · [验收status.json](../evidence/delivery-validation/acceptance/status.json) · [验收总日志](../evidence/delivery-validation/acceptance-console.log) · [系统快照](../evidence/delivery-validation/system.txt) · [CANN包元数据](../evidence/delivery-validation/cann-packages.json)。

## 2. 本轮实际loss

| step | total loss | video loss | action loss |
|---|---:|---:|---:|
| 1 | 13.729534149169922 | 0.17178936302661896 | 1.2011641263961792 |
| 2 | 13.860824584960938 | 0.1785968840122223 | 1.2074856758117676 |

本次使用默认临时smoke输出，完整大分片经检查后由验收器清理。小型日志、有效配置、分片metadata和检查报告均已归档。此前5000步正式检查点仍保存在原共享卷，未移动或删除。

## 3. 复现命令

当前集群的 `config/cluster.env` 已配好四项输入和新环境位置：

```bash
cd /mnt/lvminrui/workspace/cosmos-training-litegen-910b
bash run.sh --verify
bash run.sh --check-inputs --all-payloads
bash run.sh --bootstrap-only
bash run.sh
```

新集群先按README修改配置和挂载，不能直接搬本机venv。正式5000步和后台命令见README；本轮只复跑两步，没有在新目录重复15小时以上的5000步实验。完整5000步结论来自随包源码快照对应的已完成实验。

## 4. 本次交付变化与边界

- 运行时代码、训练配置源码、依赖锁、验收脚本和阈值原样复制。
- 更新根README与`config/cluster.env.example`；新增实验、LiteGen差异、复测说明；归档历史和本轮证据；生成新的交付MANIFEST。
- 本机`config/cluster.env`、新输出、环境、外部模型/数据、最终大检查点不进入压缩包。
- 文档命令做了语法检查；后台监督片段用模拟launcher检查成功/失败退出，不把这个检查算作新一轮5000步。

包中原实验脚本/报告保留历史绝对路径以便追溯；实际训练入口根据当前checkout定位源码。未做其他芯片、多机或可选分支验证，未从5000步完整4B检查点恢复再更新。
