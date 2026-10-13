# Action 归一化：两套仓库实现对照（2026-10-05）

## 结论与当前选择

两仓库的新实验现统一采用 **Bench2Dex 的 action-only 分位数方案**。PointFlow-FK 当前仿真运行本来就使用该方案，无需切换。
先用10条数据（8训练、2验证）检查拟合与联合生成质量，通过后扩至100条。
扩容时沿用公式，按新的训练划分重算统计；验证和部署复用训练统计。
这是当前设计选择，尚无两套归一化的生成质量对照，不能声称某套效果一定更好。

**按仓库区分算法，不用“10条版/100条版”代指算法。** 两套代码都可以对不同规模
的数据计算统计。此前提到的 action/state 联合范围属于 WorldAct-bench2dex 的旧v1方案，
不属于后来提供的 Bench2Dex 代码；公式相同也不代表统计文件或训练样本相同。

## 2026-10-05 统一结果

WorldAct-bench2dex已改为与Bench2Dex/PointFlow-FK相同的公式和统计口径：

- 逐关节使用action.q01/q99，scale下限0.05 rad；无clip、无极值/5扩尺度。
- 训练episode内所有action_valid帧各计一次；state诊断统计使用同一掩码。
- 条件state和目标action均使用action的offset/scale；不分别归一化state，也不合并其分位数范围。
- 新生成文件采用v2 schema，含兼容Bench2Dex的source/joint_names/action/state字段。
  WorldAct额外校验manifest、训练分组、NPZ指纹和文件SHA256。
- 旧v1文件和旧运行副本未替换，旧模型继续按旧统计恢复/推理。切换算法必须使用新统计和新实验目录。
- PointFlow/FK位移std与本次action归一化统一无关，未修改。

新版统计：`/data/shichaojian/raw_data/bench2dex/task21/action_stats_q01q99_v2.json`。
这是当前100条数据池中90训练/10验证的统计，训练有效action帧57,182条；与PointFlow-FK现有8/2划分
使用相同公式，不是相同统计数值。本次没有更改两边的训练/验证划分。

对同一新版统计及全部训练/验证有效帧，WorldAct实现与`/mnt/afs/Bench2Dex/utils/bench2dex_action.py`
实际函数的offset/scale和归一化结果逐位一致，反变换最大误差1.2e-7 rad。
训练集中abs(y)>5的action元素比例约0.00209%，state元素比例约0.03134%；
验证集对应比例为0和0.05080%。比例分母为有效帧数×52，不是episode或窗口数量。
超过±5不会被裁剪；state长尾主要在右手指关节。尚未做新算法训练质量对照。
审计：`/data/shichaojian/raw_data/bench2dex/task21/action_normalization_unification_audit.json`。
19项CPU回归测试通过（另复查旧统计恢复保护测试）。

新运行示例：`/data/shichaojian/runs/bench2dex/cosmos_task21_bs16_actionq01q99_20k_flash2`。
启动命令见相邻仓库的[归一化文档](/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex/docs/bench2dex_single_task.md:142)。

## 统一前的代码与统计口径（保留作历史对照）

| 项目 | `/mnt/afs/Bench2Dex` | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex` |
| --- | --- | --- |
| 统计入口 | `tools/finalize_sim_batch.py:55` | `tools/compute_bench2dex_action_stats.py:28` |
| 应用入口 | `utils/bench2dex_action.py:11` | `cosmos_framework/utils/bench2dex_normalization.py:17` |
| action样本 | 训练episode中全部action_valid帧 | 训练有效窗口覆盖的action帧，重叠窗口去重 |
| state样本 | 同一action_valid掩码对应的state | 实际作为条件的窗口起始state |
| 参数参考 | 仅action的q01/q99 | action与state的q01/q99联合范围 |
| scale下限 | 每关节0.05 rad | 每关节0.05 rad |
| 极端值 | 不额外调整尺度 | 扩大scale，使拟合样本绝对归一化值不超过5 |
| 硬截断 | 无 | 无 |
| 逆变换 | `x = y * scale + offset` | 相同 |

代码链接：[Bench2Dex统计](/mnt/afs/Bench2Dex/tools/finalize_sim_batch.py:55)、
[Bench2Dex应用](/mnt/afs/Bench2Dex/utils/bench2dex_action.py:11)、
[WorldAct统计](/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex/tools/compute_bench2dex_action_stats.py:28)、
[WorldAct应用](/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex/cosmos_framework/utils/bench2dex_normalization.py:17)。
这些绝对路径指向本集群相邻仓库，未随本仓库分发。

### Bench2Dex公式（逐关节）

```text
lo = action.q01
hi = action.q99
offset = (lo + hi) / 2
scale = max((hi - lo) / 2, 0.05)
y = (action - offset) / scale
```

mean/std同时写入JSON，但不用于这套变换。0.05下限避免放大小幅关节噪声；
不裁剪目标，因此归一化值允许超出[-1,1]，也可能超过±5。
当前仓库在 `cosmos_framework/data/generator/action/datasets/sim_pointfk_dataset.py:56`
读取同一文件并构建相同的 `ActionAffineNormalization`。

### WorldAct-bench2dex旧v1公式（仅旧模型兼容）

```text
lo = min(action.q01, state.q01)
hi = max(action.q99, state.q99)
offset = (lo + hi) / 2
lower = min(action.min, state.min)
upper = max(action.max, state.max)
tail_scale = max(abs(lower - offset), abs(upper - offset)) / 5
scale = max((hi - lo) / 2, 0.05, tail_scale)
y = (action - offset) / scale
```

这里5是默认max_abs_target，约束拟合数据的幅度，不是clip，也不保证新数据不越界。
此方案同时覆盖条件state和目标action；少数极值可能扩大尺度，压缩正常动作的数值范围。
由于offset也可能变化，不能声称每个样本的绝对归一化值都比前者小。

## 文件路径与已核验来源

| 文件 | 用途 |
| --- | --- |
| `/data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2/normalization_train_only.json` | 当前仿真训练实际加载；数据包自带 |
| `/data/shichaojian/raw_data/bench2dex/task21/action_stats.json` | WorldAct旧v1联合范围统计，90训练/10验证，保持不变 |
| `/data/shichaojian/raw_data/bench2dex/task21/action_stats_q01q99_v2.json` | WorldAct新v2 action-only统计，同一90/10划分 |
| `/data/shichaojian/runs/bench2dex/cosmos_task21_bs16_normalized_20k_flash2/action_stats.json` | 上述WorldAct训练目录保存的统计副本 |

当前文件已用provenance中episode 0–7的5508条有效action复核：mean完全一致，
q01/q99最大差异小于3.4e-7。它不是完整100条的统计，也不是此次接入时重新生成的文件。
证据：`pointflow_outputs/sim_v2_preparation_20261004/action_normalization_origin_audit.json`。

旧的两种JSON契约不同：Bench2Dex读取action.q01/q99再计算参数；WorldAct旧v1直接保存offset/scale。
WorldAct新v2从action.q01/q99计算并核对保存的offset/scale，保留额外溯源字段。
迁移不能只改文件名，必须同时核对加载逻辑、关节顺序、单位和统计划分。

## PointFlow/FK尺度是另一件事

`/mnt/afs/Bench2Dex/tools/organize_sim_pointfk.py:91`在训练窗口的原始稠密点上统计
有效PointFlow位移；FK使用窗口中相对于首帧的位移。它输出全局std及32×3分帧分轴std。
当前 `tools/prepare_sim_pointfk_training.py:62` 在FK引导选出1024点后，对有效位移
重新统计全局标量std，训练加载该标量，并非action的分位数归一化。

| 尺度（当前10条数据包） | 原稠密点统计 | 选点后实际使用 |
| --- | ---: | ---: |
| PointFlow | 0.0410891 m | 0.0526976 m |
| FK | 0.0748352 m | 0.0748352 m |

两边全局std都按`sqrt(E[d²] - E[d]²)`计算；模型位移除以std，没有减去均值。
PointFlow变化来自选点分布变化。当前FK按valid掩码统计，原脚本直接计入全部FK位移；
本批数值基本一致。

## 扩至100条前

1. 统一episode级train/val划分，确认输入数据和关节顺序。
2. 修复当前选点准备脚本 `tools/prepare_sim_pointfk_training.py:23` 的`episode < 8`
   硬编码，改为读取manifest；Bench2Dex finalize的默认count=10及8/2描述也需更新。
3. 仅用新训练集计算action统计；检查归一化后超过±5的比例与对应动作，再判断是否需要换算法。
4. 按实际1024点选择重新统计PointFlow/FK scale，固定统计文件供训练、验证和部署共同使用。
5. 新统计配新实验目录；不要在当前运行中静默替换归一化文件。

训练入口和性能记录见[仿真实验文档](pointflow_sim_v2_data_20261004.md)，快捷启动见
[quickstart](pointflow_quickstart.md#当前仿真实验2026-10-05)。

## 2026-10-06 真机统计：sandwich / dropper mix

按用户要求减少验证数据，使用seed=42、5%验证：sandwich101为96/5，dropper mix71为67/4。
训练集action-only q01/q99统计已生成；state共用offset/scale，无clip。
sandwich当前缓存为末端xyz＋四元数＋手关节，四元数保持不变，xyz下限0.05米、手关节下限0.05弧度；
dropper mix为27维关节角，全部使用0.05弧度下限。
完整统计、划分、校验和复现命令见[真机归一化记录](../pointflow_outputs/real_action_normalization_20261006/README.md)。
E8归一化入口已接入真机训练/验证loader与joint预测物理动作导出；已有运行和源缓存未修改。
新入口见[实验队列E8](./pointflow_experiment_queue_20260930.md#e8--真机归一化重训系列等待中)。
