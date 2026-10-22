# Point/FK 架构实验：当前实现说明（更新于 2026-10-19）

本文描述当前 R4、A1 训练实现，替代 2026-10-17 的 v2 待实现规格。文件名保留，便于已有链接继续使用。配套[实验方案](pointfk_architecture_experiments_20261017.md)是历史设计背景，其中排期和未实现方案不能替代本文的实现状态。

**实际训练代码根目录**：`/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-arch`，下文源码路径均相对于此目录。用户原文档位于不带 `-arch` 的工作区；文档位置不代表训练使用该工作区的代码。

**本次更新范围**：R4 当前手–物关系编码、R5 绝对几何对照、A1 动作推导 FK 到 PointFlow 的输出适配器、robotframe 数据、优化器与 8 卡同步，以及当前运行契约。D1/D3/B/C 等历史提案不属于本次两个训练任务。

## 1. 当前实验与共同基线

R4 和 A1 是两个分别训练的实验，当前没有把两者叠加到同一个模型。

| 项目 | R4 | A1 |
| --- | --- | --- |
| 要检验的问题 | 当前手–物空间关系能否改善动作决策 | 预测动作的运动学信息能否改善物体点运动预测 |
| 新增信息通路 | 当前几何关系 → action 输入 embedding | 本次预测 action → command FK → PF 解码前 hidden |
| 输入缓存 | `*_training_robotframe_rel` | `*_training_robotframe` |
| 关系编码器 | `pointfk_relation="rel"` | 关闭 |
| 输出适配器 | 关闭 | `pointfk_output_adapter="action_to_pf"` |
| 未来 FK 设置 | 保留原预测与原 FK loss | 屏蔽未来 FK 对其他模态的读取，FK loss 权重为 0；采样状态仍保留 |
| 新增损失 | 无 | 无额外接触/刚性/command-FK 损失；使用 video/action/PF 原有损失 |

共同设置：robot-base 坐标、米制、M3 `physical_split` 基础隔离、nolocal 配置、seed=42、从基础模型重新训练 40k，不从旧冻结适配器短训恢复。当前作业每项单机 8 GPU，DP shard=8、DP replicate=1、CP=1，`sync_plain_param_grads=True`。

M3 隔离 PF↔action、PF↔FK、vision↔FK 的直接主干注意力。R4 在 action 输入新增当前几何通路；A1 在 PF 输出端新增 action-derived FK 通路，均是明确的结构变量。

实现配置入口：`tools/pointfk_revision_overrides.py:16`（`revision_overrides`）、`:66`（`revision_environment`）。该工具默认 8 卡；历史上仍支持的 6 卡选项不用于当前两个作业。

## 2. 数据、坐标与标签：实际已有内容

当前接口仅支持本轮双手关节命令配置：当前动作行 + 32 个未来物理步，共 33 行；有效关节命令 52 维，可 padding 到 64；每只手 21 个 FK 关键点，共 42 点。

| 字段 | 形状/语义 |
| --- | --- |
| `pointflow.inputs.anchor_xyz` | 当前可见点位置，拼接为 `[N,3]`，robot_base、米 |
| `normal` / `color` | 当前点法向 / RGB；关系编码中颜色除以 255 |
| `point_batch` | 每个点所属样本；样本间不得混合 |
| `point_instance` | 当前实例编号，R4 必需；`-1` 表示不属于已标注任务实例 |
| `fk.inputs.anchor_xyz` | 当前双手 `[B*42,3]`，robot_base、米 |
| 原 PF 未来位移与有效性 | 仍用于原有监督，不供 R4 选邻点或 A1 构造几何 |

**实际实例列表以 manifest 为准**：

- task27：`soccer, tennis_ball, golf_ball, table_tennis`，4 个球，编号 0–3。
- task21：`msg, soy, vinegar`，3 个瓶，编号 0–2。
- 当前这两份缓存没有额外容器类别。旧 v2 中“箱子已经作为额外实例加入”的要求尚未落实到当前缓存，不能据此把实例数设为 5。未归属任务实例的点为 `-1`，不进入 R4 候选。

实例标签来自当前 anchor 帧仿真 box3d 的几何归属，margin=0.005 m，属于前向可见的当前实例输入，不是未来标签。task27 manifest 中提及 camera optical frame 描述的是生成 membership 时的坐标约定；转换后实际点坐标由顶层 `frame="robot_base"` 声明。离散标签沿用，不再次计算或变换。

缓存根目录 `/data/shichaojian/sim_data/`：

| 用途 | task27 bundle | task21 bundle |
| --- | --- | --- |
| R4/R5 | `bench2dex_task27_ball_training_robotframe_rel` | `bench2dex_task21_v3_training_robotframe_rel` |
| A0/A1 | `bench2dex_task27_ball_training_robotframe` | `bench2dex_task21_v3_training_robotframe` |

manifest 相对路径：`datasets/cosmos-cache/pointflow_fk_handguided1024/manifest.json`。PF 和 FK 元数据均需与 `robot_base` 一致；launcher 与网络前向有检查，不静默回退相机缓存。

完整 R4 标签缓存已生成：task27 75,640 个窗口，task21 60,190 个窗口。task27 从已有 camera-rel 缓存转换，标签、点编号、UV、颜色、有效性保持一致，几何往返最大误差约 `4.77e-7 m`；task21 在已有 robotframe 样本上只增加 `point_instance`，逐窗口核对原字段不变。未重新跟踪或重新选择点。验证报告在 `/mnt/afs/experiments_20261018/arch_6gpu_81/` 下的 `task27_cache_report.json`、`task21_cache_report.json`、`task21_labels_report.json`；目录中的历史 `6gpu_81` 名称不代表当前训练卡数或节点约束。

推理部署必须复用相同的坐标、实例列表、membership 规则、margin 与关节归一化。本文不声称 4090 推理运行时已经完成同步部署或闭环验收；训练冒烟通过也不能替代该检查。

## 3. R4 / A1 的信息流区别

```text
R4：当前点云 + 当前实例 + 当前手部 FK
                   ↓
             手–物关系编码
                   ↓
         action 输入 embedding + 关系增量
                   ↓
                主干 → 动作预测

A1：主干 → 本次动作速度预测 → 干净动作估计
                                  ↓
                           反归一化 + URDF FK
                                  ↓
                         32步 command FK 位移
                                  ↓
      原 PF hidden + 局部交叉注意力适配器残差
                                  ↓
                           原 PF codec 解码
```

R4 给决策提供当前状态关系；A1 给运动预测提供预测动作的运动学条件。A1 同次前向没有把修正后的 PF 再送回动作头重新预测；但 PF 损失可沿可微通路回传到动作预测。

## 4. R4：当前手–物关系 → action 输入

代码：`cosmos_framework/model/generator/pointfk_relation.py:55`（`PointFKRelation`）、`:95`（`site_features`）；主网络接入位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1380`。

### 4.1 查询位置与每实例 kNN

左右手各用索引 `[0,4,8,12,16,20]`，即手掌与五个指尖，右手索引加 21，共 12 个查询位置。对每个查询位置、每个任务实例**分别取最近的 k=16 个当前可见点**。

这不是全场景竞争一个 16 点预算。task27 最多产生 `12×4×16` 个有效邻域条目，task21 最多 `12×3×16`；同一个原始点可被不同查询位置重复选中，因此条目数不等于独立点数。

选邻居不设硬半径，避免手尚未接近时没有信息。每实例不足 16 点时使用有效 mask，不复制点凑数；零点实例显式记录缺失。距离在 FP32 下按 `[B,12,I,Nmax]` 计算，随后 `topk`，不做 CPU kNN。

### 4.2 局部特征与距离衰减

每个邻点输入为：`(point-query)/0.05`、8 维距离 RBF、法向、RGB、I 维实例 one-hot、有效位。RBF 中心均匀覆盖 0–0.20 m，宽度实际为 `0.20/(8-1)` m，不能沿用旧文档的 0.025 m。

共享 MLP 为 `Linear→64→SiLU→64→SiLU`。邻点到查询位置距离为 d，其局部支持权重：

\[
w(d)=\exp\left[-\frac12(d/0.10)^2\right].
\]

先将 64 维点特征乘 w，再分别求有效点 max 与 mean。mean 除以**有效点数**，不除以权重总和，以免远处点被重新归一化成强信号；max 同样乘权重。空邻域的局部结果为零。

10、20、30 cm 的权重约为 0.607、0.135、0.011。0.10 m 是当前设定的软衰减尺度，不是硬接触半径，也没有实验保证它最优。

### 4.3 远处摘要与缺失语义

每个查询位置、每个实例另拼接：

- 最近**可见点**相对查询位置的向量 `/0.5`，保留远处方向；它不是物体中心或真实接触点。
- 最近距离的 8 维 RBF 与 `log1p(distance/0.05)`；后者在超过 20 cm 后仍能表达距离。
- `log1p(当前可见点数)`、实例缺失标记，以及平均局部支持度。

因此“局部贡献衰减”不意味着整个 R4 在远距离时输出零：方向、距离及可见性摘要仍可用于接近物体。缺失实例也不等价于距离为零。R4 无法凭空恢复没有进入点云的物体表面。

### 4.4 注入方式与当前边界

每个实例的局部 max/mean 与摘要拼接后，所有实例经 `Linear→128→SiLU` 形成每查询位置 128 维特征。12 个位置拼成 1536 维，经 `1536→512→H` 映射成每样本一个关系增量，广播加到该样本全部 33 行 action 输入 embedding，包含当前条件行。

原动作数值、条件/损失 mask 不变；不增加主干 token，不改 PF codec、RoPE 或主干注意力后端。当前实现**只注入 action 输入，没有旧提案中的可选 FK-anchor 注入**。最后投影权重和 bias 为零初始化，没有额外可学习门控。

R4 不读取未来动作、未来 PF/FK 或未来有效性监督。k=16 是局部预算选择，尚无 16/32/64 消融证明最优。点稀疏或物体复杂时仍可能不足；可见点距离也不能当作完整表面的精确接触距离。

## 5. R5 与其他历史方案的状态

R5 已有 `mode="abs"` 实现与预设，用作参数量匹配的绝对几何对照，但它不是当前两个平台任务之一。

R5 按**每实例可见点质心**选邻点，12 个位置使用相同的物体选择，不依赖手位置；输入绝对 robot-base XYZ、到原点的 RBF、物体质心摘要。该设计避免“仅替换坐标但仍按手选邻点”泄漏相对关系。RBF 上限为 1 m，绝对坐标尺度为 0.5 m；同样保留相应的有效性及支持度汇总。R4/R5 比较必须保持其他训练设置相同。

D1 掌坐标、D3 视频输入注入、B 动作输出修正、C 视频输出适配器均不属于当前 R4/A1 运行配置。`action_geometry.py` 中保留的 camera/C0 路径不代表 A1 使用相机拟合或当前启用了 C0。历史提案不作为已完成实现或效果证据。

## 6. A1：本次预测动作 → command FK → PointFlow

代码：`cosmos_framework/model/generator/action_geometry.py:24`（动作估计）、`:56`（`ActionGeometry`）、`:145`（`FKToPointAdapter`）；主网络 `_command_fk` 位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:385`，适配器接入位于 `:1659`。

### 6.1 恢复本次干净动作估计

主干运行后，先解码一次动作速度，再在 PF decode 前构造几何。动作生成行使用 FP32：

\[
\hat x_0=x_t-\sigma_{action}\,v_\theta.
\]

逐行动作 sigma 从实际动作 timesteps / num_train_timesteps 获取；不使用视频/PF sigma。当前条件行 sigma 必须为 0。条件动作直接取给定输入，忽略该行的 dummy 预测，因此全条件动作情形不会错误读取 dummy 零动作。

helper 只读当前前向已有的带噪动作、预测速度和噪声状态，不调用依赖训练 epsilon 或目标速度的 loss-only helper。不能将训练未来 GT 动作塞入 WAM 前向来替代预测动作。

### 6.2 反归一化与原生机器人 FK

仅支持 joint-command domain=28；要求 52 个互不重复的 joint_names。使用对应 bundle 的 `normalization_train_only.json` 中 q01/q99，中点为 offset，半范围下限为 0.05；offset/scale 存为模型持久 buffer。padding 后多出的 12 维不进入 URDF。

输入为 `[B,33,52或64]`，使用 URDF 原生 base_link FK 得到 `[B,33,42,3]`，返回未来 32 步相对当前的位移 `[B,32,42,3]`。**robot_base 路径不拟合 camera/world 刚体变换**；当前 FK 与当前 anchor 的逐点误差必须不超过 5 mm，否则报错。

该输出是预测关节命令对应的 **command FK**。它不保证机器人实际执行轨迹相同，也不等于数据中的 actual-state FK，更不包含物体动力学或接触求解。

### 6.3 四步一组的局部交叉注意力

32 个物理步完整计算后组成 8 个时间块。每个关键点的四步位移拼为 12 维 `/0.05`，再拼当前 anchor XYZ `/0.5`，经 MLP 与关键点身份 embedding 编成 128 维 key/value 输入。

PF 的 `point_hidden[8,K,H]` 作为 query 来源；query 同时包含点簇绝对位置，以及该簇到当前手部最近骨段的偏移和对数距离。手部骨段按现有 21 点拓扑构造，手掌对应退化线段。

每个样本、每个时间块内部做 2-head、128 维交叉注意力，读取双手 42 个关键点运动；不跨样本或时间块，不新增主干 token。点簇到当前各骨段距离 d 带来 attention bias：`-0.5*(d/0.10)^2`。

额外提供可学习 null key，其 value 恒为零，使点簇可以不读取手部运动。交叉注意力输出投影后还乘最近骨段距离的支持度 `exp(-0.5*(d_min/0.10)^2)`，连同投影 bias 一起衰减。这里是当前接近程度，不能解释为真实接触标签；也不是沿预测轨迹逐步判断未来接触。

### 6.4 动作可靠性门控与输出残差

每个四步块用最不可靠的一行动作 sigma：

\[
g_b=\left(1-\max_{t\in b}\sigma_{action,t}\right)^2.
\]

高噪声动作产生的几何修正弱，低噪声时更强，当前条件行不参与未来四步块门控。最终：

\[
h_{PF,new}=h_{PF,base}+g_b\,s(d_{min})\,W_{out}\,Attention(Q,K,V).
\]

此式中的 `W_out` 表示含 bias 的输出投影；权重与 bias 零初始化。原 PF codec 和独立预测通路保留，适配器不把 FK 位移直接加到物体目标、不强制物体刚性跟手，也没有额外手部/物体分类分流或混合成员加权实现。

动作估计→FK→适配器没有 detach，因此 PF 损失可沿该通路更新动作预测；零初始化首步的梯度分布受输出投影为零影响，不能要求所有上游参数首步都有非零梯度。同次前向动作只解码一次，PF 修正不会再回送动作头形成第二轮预测。

### 6.5 未来 FK：当前做了什么、尚未做什么

A0/A1 预设在 M3 基础上追加 `vision<-fk:noisy;action<-fk:noisy;pointflow<-fk:noisy` 的禁止规则，并设置 `pointfk_future_fk="action_derived"`、`fk_loss_weight=0`。

**当前没有把未来 FK 的 token、噪声/采样 state 或独立 decode 从整个系统删除。** 原独立 FK 输出仍可能存在，但它不是 action-derived command FK，不应拿其 ADE 证明 A1 的动作运动学预测准确，也不能宣称 A1 已节省未来 FK 对应的 token/计算。

因此当前 A1 是“屏蔽未来 FK 条件 + 零 FK loss + action→FK→PF 适配器”的实验。A0 预设保持同一屏蔽和 loss 设置，仅关闭适配器，是隔离适配器作用的直接对照；当前两个任务没有同时提交 A0。A1 与原 M3 的差异包含未来 FK 屏蔽/loss 改动，不能把全部收益都归因于适配器。彻底移除未来 FK 仍需另行修改噪声构造、state 拼拆、采样积分和评估接口。

## 7. 优化器与修正后的 8 卡同步

新增模块必须同时满足 `requires_grad=True` 与进入优化器 param_groups。当前配方将 `pointfk_relation`、`output_adapter` 纳入 optimizer allowlist，两者新模块 LR multiplier 均为 25，与已有新几何模块的设置一致。入口：`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py:254`。

曾发现旧六卡 A1 短训中 `output_adapter` 漏选，被冻结，50 步后输出头仍为零；该短训不能作学习证据，当前作业不恢复其 checkpoint。修正后启动时强制做优化器成员检查，缺失/冻结直接报错，而不是只看总 loss 是否下降。

同步覆盖 FSDP 外的普通非 DTensor 模块：

1. checkpoint 加载后、第一次前向前，通过 pre-hook 广播普通参数和 buffers；训练网络与 EMA 均覆盖。不能等第一次 backward 之后才同步初始化。
2. 每次 backward 后，普通可训练参数梯度在数据并行组内求和并除以组大小。部分 rank 未使用的参数贡献零；所有 rank 都未使用时保留 `grad=None`。
3. FSDP 的 DTensor 梯度仍走原同步流程。

源码：`cosmos_framework/model/generator/plain_param_sync.py:6`；`omni_mot_model.py:433` 的 pre-hook、`:1355` 的 `_sync_plain_param_grads`（均位于 `cosmos_framework/model/generator/`）。

审计实现：`cosmos_framework/model/generator/pointfk_training_audit.py:23`、`:54`。启动输出 `POINTFK_OPTIMIZER_AUDIT`；设置 `POINTFK_GRAD_AUDIT=1` 时，在指定 backward 次数输出各 rank 梯度与 head 更新量，达到检查点仍未更新则报错。当前流水线的 smoke 门槛要求第 50 次 backward 八个 rank 均有记录、head 更新非零且相互一致。

## 8. 当前配置与运行记录入口

完整预设应同时使用 environment 和 overrides，不能只复制某一个开关。以下命令只打印配置，不提交任务：

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-arch
python tools/pointfk_revision_overrides.py r4 --task 27 --gpus 8
python tools/pointfk_revision_overrides.py a1 --task 27 --gpus 8
# task21 将 --task 改为 21；A1 不需要 R4 的实例缓存。
```

两个平台任务：`arch-r4-rf8-any-1018`、`arch-a1-rf8-any-1018`。每项 8 GPU、112 CPU、1920 GiB 内存；不限制 hostname，在当前队列可用节点调度。用户已明确授权**本批次不限制总节点占用数**，不表示永久修改其他批次规则。

流水线目录 `/mnt/afs/experiments_20261018/arch_rf8_queue/`。每项依次执行：8-rank NCCL 小模块同步检查 → task27、task21 各 50 步 smoke → task27、task21 各从基础模型重新训练 40k。smoke 不计入正式 40k，task21 正式训练等待该项 task27 完成。2026-10-19 检查时四次 smoke 已成功，新增模块八卡更新一致；这不等于完成闭环效果验收。

正式输出目录模板：

```text
/data/shichaojian/runs/{r4|a1}_t{27|21}_rf8_optfix_scr40_20261018/
  cosmos3_action/action_sft/sim_task{27|21}_{r4|a1}_rf8_optfix/
    checkpoints/
    pointflow_eval/
```

实时训练日志模板：

```text
/mnt/afs/experiments_20261018/arch_rf8_queue/{r4|a1}/
  {r4|a1}_t{27|21}_rf8_optfix_scr40_20261018.log
```

**当前平台日志的可见性限制**：`run_pipeline.py` 将子训练进程 stdout/stderr 重定向到上述文件；平台控制台主要显示阶段开始/结束，停在 `starting` 不等于训练停住。应检查文件中最新 step、时间戳和异常堆栈。2026-10-19 排查确认 step 0 正式验证共 26 案例（2 train + 2 episodes×3 stages×4 windows），含联合采样和视频输出，耗时约 4 分半；此外启动前两次 smoke 约 17 分钟，因此不能直接拿另一项未做相同预检的任务步数比较启动速度。本文更新不修改这一日志行为。

## 9. 验证范围、限制与后续比较

已有测试覆盖关系近邻/稀疏与空实例/远距离衰减、动作逐行 sigma/FK/适配器、robotframe 转换、配置、优化器漏选保护；实际 8 卡 NCCL 预检检查初始化、参考平均梯度、可选/未用参数、AdamW 更新和 EMA buffers。四次实际 8 卡 50 步训练验证了前后向、保存以及新增模块学习与同步。

这些证据不保证训练到 40k 的稳定性或任务成功率提升。仍需按相同预算和推理协议比较：

- R4 对 robotframe + 修正同步的 M3；R5 用于区分显式相对关系与额外绝对几何容量。
- A1 对相同未来 FK 屏蔽/loss 设置的 A0，才能单独判断适配器作用。
- 训推坐标、membership、normalizer、URDF 和配置一致后，再做闭环 SR/LSCR 及配对检验；不以训练 ADE 或单次 smoke 替代。
- 当前两个任务没有同时完成/提交上述全部控制组，也没有 16/32/64 邻点或 0.10 m 尺度消融结果。

不宣称恢复了未观测物体、获得真实接触标签、移除了未来 FK 采样变量、实现物体刚性跟随或已经证明闭环收益。

## 10. 当前实现定位

下列行号对应 2026-10-19 核查版本；代码后续变动时以函数名为准。所有源码均在本文开头的 `-arch` 根目录。

| 源码入口（file:line） | 职责 |
| --- | --- |
| `cosmos_framework/model/generator/pointfk_relation.py:55` | R4/R5 模块与零初始化 |
| `cosmos_framework/model/generator/pointfk_relation.py:95` | 按实例 kNN、局部衰减与远处摘要 |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1380` | R4 当前几何计算并送入 action 编码 |
| `cosmos_framework/model/generator/action_geometry.py:24` | 本次逐行动作干净估计 |
| `cosmos_framework/model/generator/action_geometry.py:56` | 反归一化与 robot-base command FK |
| `cosmos_framework/model/generator/action_geometry.py:127` | 当前点到手部骨段的关系 |
| `cosmos_framework/model/generator/action_geometry.py:145` | A1 局部交叉注意力、null 与残差 |
| `cosmos_framework/model/generator/action_geometry.py:209` | 四步块动作可靠性门控 |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:385` | domain、当前行与 frame 检查；构造 command FK |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1659` | PF decode 前注入 A1 残差 |
| `cosmos_framework/model/generator/plain_param_sync.py:6` | 第一次前向前同步普通参数与 buffers |
| `cosmos_framework/model/generator/omni_mot_model.py:1355` | backward 后普通参数梯度平均 |
| `cosmos_framework/model/generator/pointfk_training_audit.py:23` | 优化器完整选入检查 |
| `cosmos_framework/model/generator/pointfk_training_audit.py:54` | 梯度及实际 head 更新审计 |
| `tools/pointfk_revision_overrides.py:16` | M3/R4/R5/A0/A1 当前配置预设 |

旧 v2 原文备份在 `/mnt/afs/experiments_20261019/architecture_doc_update/implementation.before.md`。本次更新仅修改文档，不修改模型、训练代码、运行进程或 checkpoint。
