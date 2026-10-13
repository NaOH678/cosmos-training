# PointFK与action耦合分析（2026-10-12）

## 结论

- 仿真task21/27的rollout中，nonlocal、local与原生WAM的差距大多处于统计噪声内；按现有数据，既不能说nonlocal优于local，也不能说其不如WAM。task21缺WAM rollout记录。
- 时间轴、条件与噪声契约检查无泄漏：训练中video/action/PointFlow/FK共用同一sigma，FK不会比action更干净；推理时PointFlow/FK从纯噪声生成，不读GT几何。
- 种子探针（task27 nonlocal/local iter 20000，加载EMA）：联合采样对初始噪声很不敏感，全部模态换噪声时action只变0.006（action自身RMS约0.8，不足1%）；只换PointFlow/FK噪声时action变化约为前者的31%，两种模式几乎相同。action与生成几何存在耦合，但几何本身随种子几乎不变，此探针无法回答“几何预测出错时action是否随之改变”。
- 模态间attention消融（§5）：屏蔽action看PointFlow/FK，action变化0.020、MAE 0.0171→0.0195；屏蔽action看video，变化0.052、MAE翻倍。反向屏蔽（几何/video不看action）几乎无影响。即action确实读取几何，但主要依赖video；几何不依赖action。
- 首轮探针（2026-10-12上午）未加载EMA，得出的“action不依赖PointFlow/FK、采样差异0.63”**作废**，原因见§3。

## 1. Rollout结果的统计口径

SR均为偶数、task21 LSCR为0.67的倍数、task27为0.5的倍数，推断每个场景50次rollout（task21三阶段、task27四阶段）；此为推断，未见rollout配置。SR约20%时：

| 口径 | n | 标准误 | 两方法显著差异约需 |
| --- | --- | --- | --- |
| 单场景（none/cov/inv/co） | 50 | ±5.7 | 16个点 |
| ave（四场景） | 200 | ±2.8 | 8个点 |

- task27 @20000：nonlocal 25 vs WAM 18，p≈0.09；同一nonlocal在10k/13k/17k/20k为17/15/21.5/25，checkpoint间波动与方法间差距相当。
- task27四个checkpoint平均：SR WAM 17.0、local 18.0、nonlocal 19.6；LSCR WAM 46.9、local 43.9、nonlocal 45.9。
- 唯一线索在inv（需改变动作）：task27 nonlocal平均26 vs WAM 18；但四个checkpoint来自同一训练、不独立，且co场景无此优势（14.5 vs 15.5）。

## 2. 单窗口时间轴与条件

窗口起点为episode帧s，k为相对步，20 Hz（0.05 s/步），33帧共1.6 s，无重采样。`FKTiming`中15 Hz为DROID默认值，仿真路径传入20。

| k | video | state | action `[33,D]` | PointFlow | FK |
| --- | --- | --- | --- | --- | --- |
| 0（帧s） | 首帧RGB，条件 | qpos[s] | 第0行=qpos[s]，条件 | anchor：帧s的1024点xyz | anchor：URDF(qpos[s]) |
| 1…32（帧s+k） | 生成 | — | 第k行=commanded[s+k−1] | xyz(s+k)−anchor | URDF(qpos[s+k])−anchor |

- Bench2Dex中 `robot/qpos` 为物理步进后的实测关节位置，`action/commanded` 为绝对关节目标；action[t]驱动qpos[t]→qpos[t+1]。FK由实测qpos经URDF计算（Bench2Dex `tools/export_robot_fk_episode.py:13`），等于“action执行结果”的3D表示，与action只差PD跟踪误差（未量化）。
- 条件：仿真数据集固定 `mode="wam"`（[sim_pointfk_dataset.py:126](../cosmos_framework/data/generator/action/datasets/sim_pointfk_dataset.py#L126)），vision只有latent帧0、action只有第0行为条件（[transforms.py:297](../cosmos_framework/data/generator/action/transforms.py#L297)、[sequence.py:387](../cosmos_framework/data/generator/sequence_packing/sequence.py#L387)）；条件位置sigma为0且不计loss；训练与推理一致，WAM/nonlocal/local相同。
- 噪声：`independent_action/fk/pointflow_schedule` 均为false，四模态共用sigma（[omni_mot_model.py:1972](../cosmos_framework/model/generator/omni_mot_model.py#L1972)）。推理时PointFlow/FK的condition mask与reference全为0（[omni_mot_model.py:2408](../cosmos_framework/model/generator/omni_mot_model.py#L2408)），与action仅通过attention耦合；推理不调用URDF，预测action与预测FK无一致性约束。
- 训练期eval的FK ADE是FK分支自身输出对GT，不是预测action经URDF的结果，不能用于评价action。

## 3. 联合采样种子探针

开关 `JOINT_SAMPLE_SEED_OFFSETS`，格式 `模态:偏移,...`，模态取 `vision/action/sound/pointflow/fk`，偏移加到该模态原有种子上（FK原种子已加 `MODALITY_SEED_STRIDE`）。默认为空，输出与原实现逐位一致；名称拼错直接报错。仅作用于 `generate_samples_from_batch` 的联合采样，训练loss和cond口径eval不受影响。实现见 [omni_mot_model.py:2206](../cosmos_framework/model/generator/omni_mot_model.py#L2206)，生效时rank0日志打印 `JOINT_SAMPLE_SEED_OFFSETS applied`。

每次探针为一次只做启动验证的离线eval，三组共用同一checkpoint和固定case，各用独立 `OUTPUT_ROOT`：

| 组 | `JOINT_SAMPLE_SEED_OFFSETS` | 作用 |
| --- | --- | --- |
| base | 空 | 基准 |
| geom | `pointflow:1,fk:1` | 只换PointFlow/FK初始噪声 |
| all | `vision:1,action:1,pointflow:1,fk:1` | 全部换，作为正常采样差异的参照 |

```bash
CKPT=/data/shichaojian/runs/sim_task27_pointfk_nonlocal_20k/cosmos3_action/action_sft/action_policy_sim_27_ball_nonlocal/checkpoints/iter_000020000
OUTPUT_ROOT=/data/shichaojian/runs/seed_probe/task27_nonlocal_20k_ema_geom \
JOINT_SAMPLE_SEED_OFFSETS=pointflow:1,fk:1 \
EXTRA_TAIL_OVERRIDES="checkpoint.load_path=$CKPT checkpoint.load_training_state=false checkpoint.strict_resume=true checkpoint.keys_to_skip_loading=[] trainer.max_iter=1" \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/examples/launch_sim_normalized.sh 27_ball nonlocal
```

产物为 `$OUTPUT_ROOT/cosmos3_action/action_sft/<job>/pointflow_eval/step_0000000/<case>_joint/joint_prediction.npz`，`action` 为归一化 `[33,D]`。三组跑完后用 [compare_joint_seed_probe.py](../tools/compare_joint_seed_probe.py) 按case汇总action第1–32行及FK/PointFlow/vision差异，并校验三组state行一致：

```bash
R=/data/shichaojian/runs/seed_probe/task27_nonlocal_20k_ema
python tools/compare_joint_seed_probe.py --base ${R}_base --geom ${R}_geom --all ${R}_all
```

注意：
- **必须加 `checkpoint.keys_to_skip_loading=[]`**。实验配置默认跳过 `net_ema.`，而validation在EMA scope内运行（[trainer/__init__.py:437](../cosmos_framework/trainer/__init__.py#L437)），EMA在构建时从未训练的主网络复制（[omni_mot_model.py:500](../cosmos_framework/model/generator/omni_mot_model.py#L500)）。不加此项时eval用的是未训练的PointFlow/FK头：联合PointFlow ADE约66 mm，比零预测（21.7 mm）还差，各模式输出相同。训练中EMA随训练更新，训练期eval不受影响，只有“加载checkpoint后直接eval”会中招。
- 校验方法：base组 `unipc4_all_ade_mm` 均值应与训练目录同checkpoint的 `step_0020000` 一致（同case同种子），action差异约0.0016。
- 结束后仍训练1步并保存checkpoint，占用新目录空间。

2026-10-12结果（task27 iter 20000，联合eval 4步UniPC，24个验证窗口，归一化单位RMS均值）：每组约7分钟；日志确认无 `Skipping loading of key: net_ema`、`max_iter=1`、偏移生效，三组state行一致。base组PointFlow ADE nonlocal 11.15 mm / local 11.17 mm，与训练期 `step_0020000`（11.15 / 11.18 mm）一致。

| 指标 | nonlocal geom−base | nonlocal all−base | local geom−base | local all−base |
| --- | --- | --- | --- | --- |
| action（第1–32行） | 0.0018 | 0.0060 | 0.0018 | 0.0058 |
| FK | 0.0008 | 0.0034 | 0.0007 | 0.0033 |
| PointFlow | 0.0025 | 0.0038 | 0.0024 | 0.0037 |
| vision | 0.0111 | — | 0.0108 | — |

- action的geom/all比值：nonlocal 0.31，local 0.32（中位数0.30/0.31）。
- val_00上base与geom的相关系数：PointFlow 0.92、FK 0.9997，即几何基本不随种子改变。PointFlow离群窗口为val_00（0.021）和val_12（0.013）。
- 解读：模型接近确定性采样，action的少量随机性中约三成来自几何噪声，耦合存在；但几何扰动只有千分之几，结论限于“小扰动下的敏感度”，不能推出action是否利用了几何信息。
- 产物在 `/data/shichaojian/runs/seed_probe/task27_{nonlocal,local}_20k_ema_{base,geom,all}`。未加载EMA的旧产物 `task27_{nonlocal,local}_20k_{base,geom,all}` 仅作反例保留。

## 4. 诊断（2026-10-12）

均为task27 nonlocal iter 20000离线eval（加载EMA，24个验证窗口，仅2个episode），不改代码。脚本与原始报告在服务器 `/root/claude_tmp/diag_20261012/REPORT_{A,B}.md`。

**采样步数与action锯齿**：`trainer.callbacks.pointflow_eval.sampling_steps=N trainer.callbacks.fk_eval.sampling_steps=N`，产物 `/data/shichaojian/runs/diag_20261012/task27_nonlocal_20k_steps{8,16}`。`joint_prediction.npz` 的 `action` 为弧度，第1–32行与 `state_action.npz["action"][s:s+32]` 对齐。

| 步数 | 一阶差分lag-1自相关 | 二阶/一阶差分RMS比 | PointFlow ADE (mm) | action MAE (rad) |
| --- | --- | --- | --- | --- |
| 4 | 0.19 | 1.24 | 11.15 | 0.0171 |
| 8 | 0.31 | 1.13 | 11.63 | 0.0169 |
| 16 | 0.41 | 1.04 | 11.87 | 0.0167 |
| GT | 0.84 | 0.54 | — | — |

步数增加只部分缓解，16步仍远离GT，锯齿主要来自学到的速度场本身。仿真rollout中同类锯齿可由Bench2Dex `ACTION_SMOOTHING=binomial5`（与真机 `trajectory_smoothing` 相同）在部署侧压低。

**PointFlow点在物体上的分布**：缓存 `pointflow_fk_handguided1024` 为512全局FPS+每手256近手点；按mask标签与 `rgbd.hdf5` 的box2d归属，每窗口均值：手502、篮子287、soccer 142、pingpong 69、golf 14（4–36）、tennis 9；无golf为0的窗口。

**rollout中“幻觉抓取”**（task27 local 20000，40个episode，CPU统计）：预测中静止球被抬起≥3 cm而仿真中未动，失败episode约1.3%、成功约0.4%；更常见的是真实抬起时预测只有约35%达到3 cm。

**可行性（读代码）**
- 模态间attention mask：现无机制，约60–100行（`mot/attention.py` 的 `SplitInfo` 加模态id，掩码路径用显式mask的SDPA）；条件token可保持可见。`POINTFLOW_ABLATE_MODES` 只改输入并只跑PointFlow单模态采样，不是模态间消融。
- 独立sigma：训练端 `rectified_flow_training_config.independent_{action,fk,pointflow}_schedule=true` 即可（iid采样，无先后约束）；联合推理只支持单一timestep，几何先于action需新采样器。
- 可微URDF：无torch实现（现有为numpy `KinematicTree`），需移植；FK目标为相机系位移/`fk_displacement_scale`。
- PointFlow点标签：磁盘 `track_label.npy` 仅区分手(2)/物体(3)，训练缓存不带标签。
- 一致性损失：`_compute_losses` 中可同时取各模态x0与sigma。

## 5. 模态间消融与新开关

以下开关均已合并，默认关闭；不设环境变量时eval与合并前一致（合并后默认eval与合并前action RMS差0.00166，等于合并前两次运行间的差0.00168，eval本身非逐位确定）。

| 开关 | 作用 | 位置 |
| --- | --- | --- |
| `JOINT_ATTN_BLOCK="q<-k,..."` | 推理：屏蔽带噪token之间的attention，模态为 `vision/action/pointflow/fk`；条件token始终可见；不支持local RoPE | `model/generator/joint_sample_switches.py` |
| `JOINT_GEOM_LEAD=k` | 推理：几何（PointFlow/FK）按自身(N−k)步UniPC网格去噪，提前k步到0，action/video仍为N步 | 同上 |
| `JOINT_SAMPLER_STEPS_OVERRIDE=N` | 推理：覆盖联合采样步数 | 同上 |
| `POINTFK_CONSISTENCY_WEIGHT` | 训练：>0启用URDF一致性损失，默认0 | `model/generator/pointfk_consistency.py`、`robot_fk_torch.py` |
| `POINTFK_CONSISTENCY_MODE` | `task`（默认，FK(预测action)对FK(GT action)，梯度到action，权重(1−σ)²）/ `gt` / `pred_to_action` / `task+pred` / `both` | 同上 |
| `POINTFK_CONSISTENCY_STATS` | action归一化统计，如 `/data/shichaojian/sim_data/bench2dex_task27_ball_training/normalization_train_only.json`；可选 `_URDF/_FK_JSON/_NORM/_EPS` | 同上 |

torch URDF FK与numpy `KinematicTree` 误差<1e-4 mm；相机旋转由对FK锚点的Kabsch拟合得到。不建议用 `gt` 模式：命令action与实际qpos的FK存在约16 mm的稳态跟踪偏差（各滞后中位数均≈16 mm），以GT FK为目标会引入此偏差。

> **更正（2026-10-13）**：上述约16 mm是把action[t]与qpos[t]比较（错开1步）的结果。action[t]驱动qpos[t]→qpos[t+1]，按qpos[t+1]对齐后，200局rollout的tracking差为7.6 mm（中位6.5 mm，p90 11.7 mm；1步错位口径为15.1–15.3 mm）。详见 [task27 rollout失败分析](./sim_task27_rollout_failure_analysis_20261013.md#3-三个差距lag-1对齐)。

**消融结果**：task27 nonlocal iter 20000，加载EMA，24个验证窗口；用法 `JOINT_ATTN_BLOCK="action<-pointflow,action<-fk" bash ev.sh NAME`，`ev.sh` 即§3探针命令加 `trainer.max_iter=1`，脚本在服务器 `/root/claude_tmp/diag_20261012/`。dAct为与默认eval的action RMS差（噪声底约0.0017）。

| 设置 | dAct | action MAE (rad) | PointFlow ADE (mm) | lag-1自相关 | 二阶/一阶 |
| --- | --- | --- | --- | --- | --- |
| 默认 | 0 | 0.0171 | 11.14 | 0.25 | 1.19 |
| `action<-pointflow,action<-fk` | 0.0196 | 0.0195 | 11.15 | −0.27 | 1.59 |
| `action<-vision` | 0.0522 | 0.0315 | 11.14 | −0.49 | 1.73 |
| `pointflow<-action,fk<-action` | 0.0017 | 0.0171 | 11.12 | 0.25 | 1.19 |
| `vision<-action` | 0.0026 | 0.0172 | 11.15 | 0.25 | 1.19 |
| `JOINT_GEOM_LEAD=1` | 0.0028 | 0.0172 | 10.75 | 0.19 | 1.24 |

- action读取几何：屏蔽后MAE升14%、锯齿加重；但依赖程度明显低于video。
- 几何与video不读取action（屏蔽后在噪声底附近）；信息流基本是video/几何→action单向。
- 几何领先1步使PointFlow ADE降0.4 mm，但action几乎不变：action对几何内容不敏感，与§3一致。
- 模型训练时无mask、共用sigma，以上均为分布外推理；“变化大”可说明存在依赖，“变化小”不能完全排除依赖。

## 6. 局限与待办

- 仅测task27单个checkpoint、2个验证episode；task21尚未测。
- §5开关已合并并通过单测（`cosmos_framework/model/generator/{joint_sample_switches,pointfk_consistency}_test.py`），一致性损失尚未实际训练。
- 计划微调（从nonlocal iter 20000，1500步）：独立sigma；对照（同设置不开独立sigma）；独立sigma+一致性损失。启动方式同§3探针，`EXTRA_TAIL_OVERRIDES` 增加 `trainer.max_iter=1500 scheduler.cycle_lengths=[1500]` 及对应开关（独立sigma：`model.config.rectified_flow_training_config.independent_{action,fk,pointflow}_schedule=true`；一致性：`POINTFK_CONSISTENCY_WEIGHT=0.1 POINTFK_CONSISTENCY_MODE=task POINTFK_CONSISTENCY_STATS=...`），输出到新的 `OUTPUT_ROOT`。
- 2026-10-13已启动（4000步，每500步存档与验证，输出 `/data/shichaojian/runs/ft_20261012/{control,indep,indep_cons}`）：control在开发机；indep、indep_cons经clusterx（`/mnt/afs/clusterx/submit-ft27-20261012.py`）提交。
- 独立action sigma须同时设 `model.config.rectified_flow_training_config.shift_action=5`（480分辨率的视频shift）；否则因全局 `shift` 为按分辨率的dict而报 `shift_action=None requires the global shift to be an int`。FK/PointFlow的独立sigma自动沿用视频shift。
- 一致性损失量级：未加权约0.03（总损失约1），权重0.1仅占约0.2%，实际使用权重3（约8%）。
- 推理时一致性引导（基于URDF梯度，非CFG）待独立sigma模型就绪后评估。
- 40k复现（待20k一轮全部结束后启动，与20k结果并列比较）。依据：4090上 `task27_cosmos_{raw,local,nolocal}_s4_*` 的200条binomial5评测，task27 nolocal在25k–40k平均SR 47.3%、LSCR 80.4%，raw为43.8%、76.0%；20k以内PF/FK不占优。单个checkpoint 200条SR标准误约3.5个点，比较以LSCR和逐球漏斗为主。起点：`/data/shichaojian/runs/sim_task27_ptfk_nonlocal_8gpu_40k_20261010`、`sim_task27_wam_8gpu_40k_20261010` 的 iter 40000（配置与20k相同，仅训练步数与学习率周期为40000）。20k做过的逐项在40k上重做：
  - 离线：§3种子探针、§5模态间消融（默认 / `action<-pointflow,action<-fk` / `action<-vision`，dAct、MAE、PF/FK ADE、lag-1自相关、二阶/一阶），先复现训练期验证指标。
  - 4000步微调：control、indep（独立sigma，`shift_action=5`）、indep_cons（+一致性损失，权重3）、固定mask M1 `geom_readonly` / M2 `action_no_geom` / M3 `physical_split`、cons_all（L1+L3+L4）、smooth_only（L4），以及原生WAM control；输出 `/data/shichaojian/runs/ft_40k_<日期>/`。
  - 微调后离线消融同上；交接到4090（`bench2dex-cloud-handoff` skill），按 `deployment/cosmos/TASK27_INFERENCE.md` 记录的命令做200条binomial5闭环评测，并跑逐球漏斗（接近/接触/抬起/搬运/放置、按真实球心的放置率）。
  - 20k一轮对应记录：微调 `/data/shichaojian/runs/ft_20261012/`；离线消融 `/data/shichaojian/runs/xmodal_ft4k_20261013/`（脚本 `/mnt/afs/clusterx/xmodal-20261013/`）；4090 rollout `/oss-shichaojian/outputs/infer/task27/task27_cosmos_nolocal_{20000,ft4k_*}/runs/eval_200_8gpu_b5/`。
