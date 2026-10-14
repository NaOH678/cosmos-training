# task27改进实验记录（2026-10-13起）

目标：缓解video–action差距（不让dream video带偏action），把PF/FK相对原生WAM（raw）的闭环优势做大。本文记录实验、结果与产物位置，持续更新。耦合分析与开关见 [PointFK与action耦合分析](./pointflow_fk_action_coupling_20261012.md)，rollout失败分析见 [task27 rollout失败分析](./sim_task27_rollout_failure_analysis_20261013.md)。

## 实验台账（进行中与待补）

规则：实验启动即在此加一行（结果写“待补”），有数据就填，数据齐了把结论写入对应章节并把该行移到“已结”。闭环均为4090、binomial5、200局/点（参考场景同raw）；SR / LSCR 单位%。

### 进行中

| 实验 | 设置 | 位置 / 产物 | 启动（UTC） | 结果 | 结论去向 |
| --- | --- | --- | --- | --- | --- |
| guidance 2 | nolocal_s4_40000，`policy.g2.yaml` | 886，`runs/eval_200_8gpu_b5_g2` | 10-14 07:45 | 待补（155局时79成功） | §0.5 |
| guidance 1.5 | nolocal_s4_40000，`policy.g15.yaml` | 899，`runs/eval_200_8gpu_b5_g15` | 10-14 07:45 | 待补（200局96成功，汇总中） | §0.5 |
| WAM control 续训 g3 / g1 | `task27_cosmos_raw_ft40_wam_control` | 886 / 899，`runs/eval_200_8gpu_shared_b5{,_g1}` | 10-14 09:20 | 待补 | §2 |
| WAM control 离线eval | `xm40_eval.sh wam_control default` | 开发机，`/data/shichaojian/runs/xmodal_ft40_20261013/wam_control_default` | 训练结束后自动 | 待补 | §2 |
| 从头训 M1–M4 | §3 | `scr40-mask-m{1,2,3,4}` | 10-14 03:38–04:57 | 待补（训练中，约2.5天；之后25k–40k闭环g1） | §3 |
| 从头训 seed43 / indep | §3 | `scr40-{seed43,indep}` | 10-14 05:30 | 待补 | §3 |
| 从头训 cons / smooth | §3 | `scr40-{cons,smooth}` | 10-14 09:45提交，排队等节点 | 待补 | §3 |
| guidance 1 下PF/FK vs raw配对检验 | 30/35/40k合并，同种子配对，按扰动类 | CPU，读已有rollout | 未开始 | 待补 | §0.5 |

### guidance 1 同口径对比（30/35/40k合并，600局/模型，数据已齐，配对检验待补）

| 模型 | 总体 | none | cov_only | inv_only | inv_cov |
| --- | --- | --- | --- | --- | --- |
| raw | 52.3 | 76.7 | 34.7 | 51.3 | 46.7 |
| local | 54.7 | 72.0 | 46.0 | 48.7 | 52.0 |
| nolocal | 54.8 | 69.3 | 45.3 | 52.7 | 52.0 |

单点SR（g1）：raw 30k/35k/40k 52.0 / 54.0 / 51.0；local 54.5 / 54.5 / 55.0；nolocal 52.5 / 57.0 / 55.0。初步观察（待配对检验确认）：关CFG后PF/FK总体领先缩小到约2.5个点（g3约4个点）；cov_only仍领先约11个点；none上raw更高（76.7 vs 69–72）。

## 结论（截至2026-10-13 17:00 UTC）

- 训练步数比结构改动重要。4090上 `task27_cosmos_{raw,local,nolocal}_s4_*`（40k训练，binomial5，200局/点）：20k以内PF/FK不占优；25k–40k四点平均SR raw 43.8 / local 46.8 / nolocal 47.3，LSCR 76.0 / 79.9 / 80.4。单点200局SR标准误约3.5个点，比较以LSCR、多点合并和逐球漏斗为主。数值已逐项对过产物（42组SR/LSCR全部一致，均200局、审计passed、binomial5、execute_steps 32、ckpt步数匹配）。
- PF/FK优势来自外观泛化：30k/35k/40k合并（每模型600局），cov_only SR raw 28.7 → nolocal 44.7 / local 40.7，同种子配对nolocal-only 34 vs raw-only 10（p≈0.0004），是唯一显著的扰动类；none持平，inv_cov略差（40.7 vs 44.7）。逐球看tennis提升最大（物理放置80.8→89.5%）。
- 剩余瓶颈是golf：失败约70%在golf，“触碰→举起”约75%，各模型与raw无差别；其次golf“接近未触碰”。
- 20k上的微调（独立sigma、一致性损失、固定mask、平滑损失，各4000步）离线与闭环都无收益：离线action MAE 0.0166–0.0174（基线0.0171）；屏蔽action看video时dAct仍约0.052，action对video的依赖不变；闭环（binomial5）基线46.0%，control/indep/indep_cons 42.0/41.0/42.0%，在噪声内。
- 40k基础模型离线：MAE 0.0176，与20k相同；动作更平滑（lag-1自相关0.24→0.32，二阶/一阶1.19→1.12）；屏蔽video的dAct 0.0525，依赖不变。离线action误差不能预测闭环成功率。
- binomial5动作平滑影响大：同一20k nolocal，平均每局放置球数2.06→3.1。

## 0. 各组配置与目的

共同设置：从nonlocal基础模型（20k一轮用iter 20000，40k一轮用iter 40000）加载EMA权重续训4000步，其余配置与原训练相同。除control外都开独立sigma（indep），其余改动叠加在indep之上。mask记法 `q<-k` 表示q模态的token不能attend到k模态的token。

| 组 | 改动 | 目的 |
| --- | --- | --- |
| control | 无，原配置续训 | 对照：分离“多训4000步”本身的效果 |
| indep | `independent_{action,fk,pointflow}_schedule=true`，`shift_action=5`；四模态各自采样噪声水平 | 原配置四模态共用同一sigma，action去噪进度被video绑定；独立后训练中会出现“video噪声大、action噪声小”等组合，让action学会在video不可靠时少依赖video，缓解dream video带偏action |
| indep_cons | indep + `POINTFK_CONSISTENCY_WEIGHT=3`，`MODE=task` | 一致性损失：URDF正运动学FK(预测action)对齐FK(GT action)，权重(1−σ)²，梯度回到action；用机器人运动学把action约束在手部几何上 |
| cons_all | indep_cons + `POINTFK_PF_FK_WEIGHT=10` + `POINTFK_SMOOTH_WEIGHT=10` | 约束全开：再加PF与FK之间的一致性、action平滑损失，检验多约束叠加 |
| smooth_only | indep + `POINTFK_SMOOTH_WEIGHT=10` | 只加action平滑损失；对应binomial5后处理平滑的大收益，看能否在训练端学到 |
| M1 `geom_readonly` | indep + mask `action<-pointflow, action<-fk, vision<-pointflow, vision<-fk` | PF/FK只读：几何可以读video/action，但video和action都读不到几何。检验几何作为“辅助监督”而非输入时是否足够 |
| M2 `action_no_geom` | indep + mask `action<-pointflow, action<-fk` | 只切action读几何（video仍可读）：检验action是否直接依赖PF/FK |
| M3 `physical_split` | indep + mask切断 FK↔vision、PF↔action、PF↔FK | 按物理含义分组：FK（手部运动学）只与action交互，PF（物体/场景点流）只与video交互，去掉跨组捷径 |
| M4 `action_observed_vision` | indep + mask `action<-vision:noisy`（仅40k一轮） | action只看已观测的条件帧，看不到带噪的未来video；直接阻断dream video对action的影响 |

推理侧开关 `JOINT_VISION_LEAD=k`（仅离线eval）：video先去噪k步再让action跟上，目的同M4，但不改训练。mask实现在分支 `pointflow-fk-mask`，一致性/平滑损失在 `pointflow-fk-cons`，环境变量说明见 [PointFK与action耦合分析](./pointflow_fk_action_coupling_20261012.md) §5。

## 0.5 当前最重要的发现：推理关闭CFG（guidance 3→1）

只改 `policy.yaml` 的 `guidance`，其余条件不变。CFG只作用于文本：无条件分支把caption换成空串，图像/状态等条件两支相同；guidance 3 每步跑两次前向并外推 `v_uncond + 3(v_cond − v_uncond)`，作用到video、action、PF、FK全部输出；guidance 1 只跑条件分支。训练期eval（以及全部离线指标）一直是guidance 1，此前4090闭环都是3。

| 模型 | guidance 3 | guidance 1 | 配对（g1独赢 / g3独赢，p） |
| --- | --- | --- | --- |
| nolocal 30k | 50.0 | 52.5 | 29 / 24，0.58 |
| nolocal 35k | 47.0 | 57.0 | 35 / 15，0.007 |
| nolocal 40k | 47.5 | 55.0 | 36 / 21，0.063 |
| nolocal 三点合并（600局） | 48.2 | 54.8 | 100 / 60，0.002 |
| local 40k | 50.0 | 55.0 | 33 / 23，0.23 |
| raw 40k | 45.0 | 51.0 | 29 / 17，0.10 |

各模型都涨5–10个点，原生WAM也涨，说明CFG 3对本任务（单一固定caption）是过度外推，而非PF/FK特有；关CFG后nolocal仍高于raw（40k：55.0 vs 51.0）。raw的提升集中在无扰动（56→80），外观扰动不变（30→28），PF/FK在外观扰动上保持50。待补：raw/local 30k/35k的g1（同口径比较）、nolocal 40k的guidance 2与1.5。启动方法见Bench2Dex `deployment/cosmos/TASK27_INFERENCE.md`（使用文档，不含结果；结果以本文为准）。

## 1. 20k一轮

| 实验 | 设置 | 训练输出 |
| --- | --- | --- |
| control | 原配置续训4000步 | `/data/shichaojian/runs/ft_20261012/control` |
| indep | `independent_{action,fk,pointflow}_schedule=true`，`shift_action=5` | `.../indep` |
| indep_cons | indep + `POINTFK_CONSISTENCY_WEIGHT=3 MODE=task` | `.../indep_cons` |
| M1/M2/M3 | indep + `model.config.joint_attn_mask=geom_readonly/action_no_geom/physical_split` | `.../mask_*` |
| cons_all | indep + 一致性3 + `POINTFK_PF_FK_WEIGHT=10` + `POINTFK_SMOOTH_WEIGHT=10` | `.../cons_all` |
| smooth_only | indep + `POINTFK_SMOOTH_WEIGHT=10` | `.../smooth_only` |
| WAM control | 原生WAM 20k续训4000步 | `.../wam_control` |

均从 `sim_task27_pointfk_nonlocal_20k`（WAM：`sim_task27_wam_20k`）iter 20000加载EMA。mask代码在分支 `pointflow-fk-mask`（`f9b6c5b`），cons代码在 `pointflow-fk-cons`（`8efe5d4`）。

离线消融（24个验证窗口，EMA，默认eval已复现训练期指标；dAct为与同模型默认eval的action RMS差，噪声底约0.0017）：

| 模型 | 默认MAE | 去几何 dAct / MAE | 去video dAct / MAE | PF ADE |
| --- | --- | --- | --- | --- |
| 基线 nolocal 20k | 0.0171 | 0.0196 / 0.0195 | 0.0522 / 0.0315 | 11.14 |
| control | 0.0170 | 0.0194 / 0.0193 | 0.0513 / 0.0304 | 11.00 |
| indep | 0.0173 | 0.0173 / 0.0192 | 0.0516 / 0.0307 | 11.35 |
| indep_cons | 0.0166 | 0.0167 / 0.0184 | 0.0518 / 0.0307 | 11.14 |
| M1 | 0.0173 | — | — | 11.06 |
| M2 | 0.0174 | — | — | 11.22 |

mask模型不能再叠加 `JOINT_ATTN_BLOCK`，只做默认eval。脚本 `/mnt/afs/clusterx/xmodal-20261013/`，输出 `/data/shichaojian/runs/xmodal_ft4k_20261013/`。

闭环（4090，binomial5，200局，参考场景同raw）：`/oss-shichaojian/outputs/infer/task27/task27_cosmos_nolocal_{20000,ft4k_control,ft4k_indep,ft4k_indep_cons}/runs/eval_200_8gpu_b5/`；逐球漏斗 `/oss-shichaojian/outputs/infer/task27/analysis/gap_20261013/ft4k_b5/analysis.txt`。

## 2. 40k一轮（进行中）

起点 `sim_task27_ptfk_nonlocal_8gpu_40k_20261010`、`sim_task27_wam_8gpu_40k_20261010` iter 40000；与20k一轮命令逐条比对，只改起点、输出目录 `/data/shichaojian/runs/ft_40k_20261013/<name>` 和任务名 `ft40-<name>-4k`。脚本 `/mnt/afs/experiments_20261013/ft40/`。

- 同20k一轮的9组，另加M4 `action_observed_vision`（action只看已观测帧，不看带噪的未来video；规则 `action<-vision:noisy`，分支 `pointflow-fk-mask` `3788ac6`）。
- 推理开关 `JOINT_VISION_LEAD=k`（video提前k步去噪完，action再跟上；同一提交，仅离线eval）。
- 训完后自动：离线消融（`/data/shichaojian/runs/xmodal_ft40_20261013/`）；交接4090并跑200局binomial5闭环（`task27_cosmos_nolocal_ft40_<name>`，启动方法见Bench2Dex `deployment/cosmos/TASK27_INFERENCE.md`（使用文档，不含结果；结果以本文为准））。4090运行时原不支持 `joint_attn_mask`，已在worktree `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-mask`（分支 `runtime-joint-attn-mask`，`10cfcdc`）移植，mask模型从该worktree启动。
- 40k基础模型离线消融：`/data/shichaojian/runs/xmodal_40k_20261013/`。
- 推理侧：40k nolocal/raw的 `execute_steps` 16、8（原32），`policy.es{16,8}.yaml`，run-dir `eval_200_8gpu_b5_es{16,8}`（raw为 `eval_200_8gpu_shared_b5_es{16,8}`）。
- task21 40k离线消融（检验结论是否跨任务）。

execute_steps结果（40k@40000，binomial5，200局）：缩短执行步数明显变差，不再继续。

| 模型 | es32（原） | es16 | es8 |
| --- | --- | --- | --- |
| nolocal | 47.5（40k单点；30/35/40k合并48.2） | SR 26.5 / LSCR 69.8 | SR 5.0 / LSCR 30.8 |
| raw | 45.0（40k单点；合并44.2） | SR 26.0 / LSCR 65.5 | SR 4.5 / LSCR 20.3 |

es16各类SR：nolocal none/cov/inv/inv_cov 36/24/24/22，raw 38/20/22/24。产物 `task27_cosmos_{nolocal,raw}_s4_40000/runs/eval_200_8gpu*_b5_es16/`。

40k微调闭环（4090，binomial5，200局，起点nolocal 40k@40000：SR 47.5 / LSCR 80.4，none/cov/inv/inv_cov 60/50/44/36）：

| 微调 | SR | LSCR | none / cov / inv / inv_cov |
| --- | --- | --- | --- |
| control | 47.5 | 79.8 | 64 / 34 / 46 / 46 |
| indep | 36.0 | 72.5 | 48 / 20 / 40 / 36 |
| indep_cons | 45.0 | 77.4 | 56 / 32 / 50 / 42 |
| cons_all | 44.5 | 78.9 | 60 / 30 / 50 / 38 |
| M1 geom_readonly（video/action都不读PF/FK） | 22.5 | 62.1 | 26 / 22 / 20 / 22 |
| M2 action_no_geom | 26.5（g1 36.0） | 64.1 | 34 / 16 / 34 / 22 |
| M3 physical_split | 40.5（g1 41.5） | 72.6 | — |
| smooth_only | 40.0（g1 56.0） | 75.0 | 48 / 32 / 36 / 44 |
| M4 action_observed_vision | 14.0（g1 19.0） | 57.5 | 16 / 12 / 18 / 10 |
| WAM control | 闭环待交接（g3与g1各200局） | | |

小结（括号内为guidance 1，基础模型g1为55.0）：续训4000步（control）不变；独立sigma（indep）掉11个点，加一致性损失（indep_cons、cons_all）能拉回到基线附近但不超过；固定mask M1严重下降（action不读PF/FK后闭环崩，与离线MAE几乎不变形成反差，说明离线指标不可用于筛选）。产物 `/oss-shichaojian/outputs/infer/task27/task27_cosmos_nolocal_ft40_<name>/runs/eval_200_8gpu_b5/`。

逐球漏斗（golf“触碰→举起”，基础 g3 75.6% / g1 78.0%）：control 77.1%，与基础相同；indep 66.4%，indep_cons 回到79.1%；mask类主要掉在这一步（M1 55.7%、M2 61.2%、M4 37.7%，M3 72.7%），M1/M2“接近未触碰”也升到27–29%（基础19.6%）；smooth_only在g1下83.3%。乒乓与网球是mask/indep损失最多的其余球。漏斗 `/oss-shichaojian/outputs/infer/task27/analysis/gap_20261013/ft40/analysis.txt`，交接与结果汇总 `/oss-shichaojian/outputs/infer/task27/analysis/ft40_20261014/README.md`，provenance `sim_inference_assets/task27/provenance/handoff_20261014_cosmos_nolocal_ft40/readiness.json`。

推理参数（不训练，nolocal 40k@40000，binomial5，200局，2026-10-14 03:00 UTC起）：`num_steps` 4→8（886，run-dir `eval_200_8gpu_b5_ns8`），`guidance` 3→1（899，`eval_200_8gpu_b5_g1`）。策略文件为同目录 `policy.{ns8,g1}.yaml`，仅改该字段与 `output_dir`。

- num_steps 4→8：SR 47.5→43.5，LSCR 80.4→80.0，不改善，不再继续。产物 `task27_cosmos_nolocal_s4_40000/runs/eval_200_8gpu_b5_ns8/`。
- guidance 3→1（即关闭CFG）：SR 47.5→55.0，LSCR 80.4→84.9，none/cov/inv/inv_cov 72/50/52/46。同种子配对 g1独赢36、原独赢21，McNemar p=0.063，边缘显著。为确认，已排队 raw/local 40k 与 nolocal 35k/30k 的 g1（与原基线30/35/40k三点合并比较），后续每个ft40模型也同时跑 g1。启动方法见Bench2Dex `deployment/cosmos/TASK27_INFERENCE.md`（使用文档，不含结果；结果以本文为准）。

## 3. 从头训练的mask消融与一致性诊断（2026-10-14起）

续训从已收敛的40k出发、只训4000步，结构改动来不及适应，EMA又平均了适应初期，正负结果都会被压平，不能据此判断改动本身（见§2小结）。因此：

- M1–M4从头训40k：命令与 `sim-task27-ptfk-nonlocal-40k`（2026-10-10）逐字相同，只加 `model.config.joint_attn_mask=<preset>`，从mask worktree（`pointflow-fk-mask` `3788ac6`）启动，不开indep，以单独隔离mask的作用。任务名 `scr40-mask-m{1,2,3,4}`，输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_mask_<preset>_8gpu_40k_20261014`，提交脚本 `/mnt/afs/experiments_20261014/scratch_mask/submit_scratch.py`。约2.5天；按基础40k同样在25k–40k各点上4090闭环，与基础模型同步数比较。
- 对照（2026-10-14 05:30 UTC起，同一提交方式，脚本 `submit_scratch_base.py`）：`scr40-seed43` 基础配置只改 `trainer.seed=43`，用于估计“同配置不同种子”的闭环波动，M1–M4与基础模型的差异须超过这个波动才算数；`scr40-indep` 从头训开独立sigma（`independent_{action,fk,pointflow}_schedule=true`，`shift_action=5`），续训中indep掉11个点，从头训检验是否只是适应不足。输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_{seed43,indep}_8gpu_40k_20261014`。
- 损失类从头训（2026-10-14 09:45 UTC提交，脚本 `submit_scratch_loss.py`，从 `-cons` worktree启动，不开indep，以便与基础模型和seed43直接比较）：`scr40-cons` 只加action–FK一致性损失（`POINTFK_CONSISTENCY_WEIGHT=3 MODE=task`），`scr40-smooth` 只加action平滑损失（`POINTFK_SMOOTH_WEIGHT=10`）；两者均需 `POINTFK_CONSISTENCY_STATS`。输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_{cons,smooth}_8gpu_40k_20261014`。提交时无空闲节点，排队中。
- 一致性诊断（不训练）：在40k nonlocal/local与20k nonlocal上，用约150–250个验证窗口量FK头 vs FK(预测action)、手附近PF vs FK、各自对GT的误差，并区分抓取阶段，判断各模态预测是否已经一致、一致性损失是否有空间。输出 `/data/shichaojian/runs/consistency_diag_20261014/`。代码与说明在分支 `pointflow-fk-cons`（worktree `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-cons`）的 `tools/pointfk_consistency_diag/` 与 `docs/pointfk_consistency_diag_20261014.md`，未合并到本分支。
- 一致性诊断结果（2026-10-14，sanity通过）：各模态预测已相当一致，误差是“一起错”——FK头与预测action只差约14 mm，而各自对GT差26/31 mm；手附近PF–FK残差8 mm。抓取窗口误差高10–25%，但不一致只略增。即FK↔action、PF↔FK一致性损失可收紧的空间小，不针对主要误差；详见分支 `pointflow-fk-cons` 的 `docs/pointfk_consistency_diag_20261014.md`。
- 闭环一致性（2026-10-14，nolocal/local 30–40k，guidance 3与1，2000局、63118次规划）：FK头与FK(原始action)的相对差闭环18.9 mm（离线14.2），与阶段无关；成功局15.6、失败局19.1，按局均值预测失败AUC 0.82，失败局全程偏高而非结束前才升高；guidance 3比1高约20%（18.9 vs 15.7）。以实际轨迹为参照，FK头15.7 mm、FK(action) 10.2 mm，但实际轨迹本身由执行action产生，只说明FK头没跟上将要发生的运动。
- 同参照离线对比（同一批约150窗口）：对专家实际轨迹，FK头26.2、FK(预测action) 31.4 mm；对专家指令，FK头34.8、FK(预测action) 31.0 mm。两者各自更接近自己的训练目标，差距约等于专家指令与实际执行之间的偏差（约17 mm），并无优劣；第1帧FK头2.8 mm、第32帧两者都约36–38 mm，瓶颈在远期预测。
- 结论：模态间不一致主要来自FK头没跟上action；“把action拉向FK头”没有依据，“把FK头拉向action”只改善FK头，一致性损失提升成功率的潜力有限。不一致更适合作“模型困惑”信号（失败AUC 0.82），下一步是离线多采样测试：同一状态下矛盾最小的样本，action是否更准。详见分支 `pointflow-fk-cons` 的 `docs/pointfk_consistency_diag_20261014.md` §4“同参照对比”与§5，结果 `/data/shichaojian/runs/consistency_diag_20261014/same_ref/`、`/oss-shichaojian/outputs/infer/task27/analysis/consistency_closed_loop_20261014/`。
- 一致性损失尚未完整验证：现有 `task` 模式只是在FK空间算action损失，不涉及FK头；FK头↔action的跨模态版本（`pred_to_action(_rel)`）梯度只到FK头；PF↔FK只在cons_all中与其他项混用过；video↔PF未实现。诊断结果出来后再定从头训练的损失组合。

## 4. 待办

- 40k结果出来后按逐球漏斗与配对比较筛选，重点看golf“触碰→举起”和cov_only。
- golf抓取：待可视化确认失败形态（够不着、手指未合拢、抓后滑落）后设计针对性改动。
- 原生WAM离线eval无现成脚本，未做。
- 代码合并：mask、cons分支及4090运行时worktree待结果确定后再合并到主分支。
