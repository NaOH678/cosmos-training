# task27改进实验记录（2026-10-13起）

目标：缓解video–action差距（不让dream video带偏action），把PF/FK相对原生WAM（raw）的闭环优势做大。本文记录实验、结果与产物位置，持续更新。耦合分析与开关见 [PointFK与action耦合分析](archive/pointflow_fk_action_coupling_20261012.md)，rollout失败分析见 [task27 rollout失败分析](./sim_task27_rollout_failure_analysis_20261013.md)。

## 当前状态与阅读说明（2026-10-21，先读这一节）

- **梯度同步问题**：10-19 发现 PF/FK 分支一直没有跨卡同步。分支是在 FSDP 包装之后才装上的，每张卡各训一份，检查点只存了 0 号卡那份。所以**本文 10-19 之前所有带 PF/FK 的结论只能作参考**，例如"M3 比 nolocal 高约 10"。修复已合入主分支 `1f9c737`：分支改在 FSDP 包装之前安装。raw 没有 PF/FK，不受影响。
- **同步版重训结果**：种子 42，g1 评测，200 局，20k / 25k / 30k 平均 SR：

  | 实验 | task27 | task21 |
  |---|---|---|
  | M3 放开 PF↔FK | 62.7 | 70.2 |
  | vw25 | 63.3（LSCR 只有约 69，失败局几乎一个球都没放进去） | 训练中 |
  | M3 | 54.5 | 68.5 |
  | 未同步 nolocal | 51.2 | 68.0 |
  | raw | 52.5 | 64.2 |

  PF↔FK 这座桥有用：放开 PF↔FK 比 M3 高约 8 个点。未同步版 M3 种子42 的 +10，同步后没有复现。
- **在跑的实验**：同步版 nolocal 基线、M3 放开 PF↔FK 种子43、组合（M3 放开 PF↔FK + objq32）、raw+vw25，均为 task27、task21 各一组。另一个会话在做 robotgeometry-v1（V1），task27 平均 68.7，task21 在训练中。V1 用了仿真里的真值手/物体分割作为输入，mask 也不彻底，所以它的提升来自哪里还要做消融。
- **评测波动**：同一个检查点重测会差 0–7 个点；换种子差约 5 个点。只有 20k、25k、30k 三个点都稳定，才下结论。
- **已删除的 worktree**：文中提到的 `-mask`、`-modemix`、`-fsdp`、`-cons` 训练 worktree，以及 4090 上的 `Bench2Dex-Pointflow-fk-{mask,p3,robotframe}` 推理 worktree，已于 10-21 删除，分支都还在。旧的 mask 检查点可以用 `Bench2Dex-Pointflow-fk-fsdp` 推理，已验证加载结果一致。
### 方法演变时间线（已尝试的方案，按时间）

除非另外说明，结果都是 g1 评测，200 局，20k / 25k / 30k 三个点的平均 SR，格式为 task27 / task21。标"未同步"的实验受 PF/FK 分支没跨卡同步的影响，只能作参考。

| 时间 | 方案 | 为什么做 | 结果 | 结论 |
|---|---|---|---|---|
| 10-10 | **raw / nolocal / local 基线**（从头训 40k） | raw 是只有视频和动作的原生 WAM；nolocal 加入 PF/FK 联合去噪；local 再加 PF/FK 的局部三维 RoPE | raw 52.5 / 64.2；nolocal 51.2 / 68.0（未同步）；local 54.8 / 66.7（未同步） | task27 上 PF/FK 没有拉开差距；20k 以内不占优，训练步数比结构改动更重要 |
| 10-12 | **推理 guidance 3→1** | 文本 CFG 可能带偏动作 | nolocal 30–40k：48.2→54.8（配对 p=0.002）；raw、local 同向 | 目前最大的单项收益，此后所有评测都用 g1 |
| 10-13 | 缩短开环执行步数（execute_steps 16 / 8） | 想让闭环修正更频繁 | 26.5 / 5.0，大幅变差 | 不用 |
| 10-14 | **nolocal 换种子 43** | 衡量训练随机性 | task27 56.5（种子 42 是 51.2） | 换种子会差约 5 个点；单个种子的小幅提升不可信 |
| 10-14 | indep：四个模态各自随机抽噪声强度 | 让各模态的去噪进度解耦 | task27 52.8（未同步） | 没有提升，越训越差 |
| 10-14 | **模态 mask 消融 M1–M4**（未同步） | 动机是"减少模态之间的干扰" | M1 只读 51.8、M2 动作不读几何 55.8、**M3 physical_split 62.0**、M4 动作不读带噪视频 54.5（task27） | M3 当时看起来涨了约 10 个点，后来被推翻（见 10-19 行） |
| 10-14 | 一致性诊断和一致性 loss（cons） | 怀疑 FK 头和"动作算出的 FK"不一致 | FK 头和 FK(action) 只差 14mm，两者各自离真值 26–31mm；多采样挑选只降 0.3mm；训练在 693 步停掉 | 各模态之间一致，但一起错；一致性 loss 改进空间有限，不再做 |
| 10-14 | **FDM / FDM+IDM 模式混合**（未同步） | 训练时混入前向和逆动力学模式，加强视频和动作的耦合 | FDM 57.5 / 63.0；FDM+IDM 58.7 / 65.3（重测 59.2）；FDM+IDM 种子43 task27 55.0；raw+FDM 52.2 | 换种子后提升没复现，task21 不涨，不再做 |
| 10-16 | **robotframe**：PF/FK 改用机器人底座坐标系（未同步） | 相机高度随场景变化，相机坐标下桌高要靠模型去推断 | 55.5 / 62.8 | 不如 nolocal；V1 后来沿用了这个坐标系 |
| 10-16 | **objq32**：训练时每个物体至少选 32 个点（未同步） | 小球上的点太少；之前只在推理时补点无效 | 57.3 / 69.2 | task27 小涨，task21 基本持平；放进组合实验 |
| 10-17 | **vw25**：视频 loss 权重 10→2.5（未同步） | 视频 loss 权重太大，可能挤占动作学习 | task27 61.2 | 有效果；需要 raw+vw25 对照，正在训 |
| 10-17 | M3 换种子 43、M3 task21（未同步） | 复现 M3 | 种子43 task27 56.2；task21 68.7 | 换种子后没复现 |
| 10-17 | P3：接近接触时提前重规划（只改推理） | 抓取时机不准 | 49.5（基线 55.0） | 没有改善 |
| 10-17 | 架构 R4 / A1（另一个会话） | R4：手–物关系编码；A1：用动作算出的 FK 引导 PF | 30k：A1 55.5、R4 51.5（task27） | 提升不明显 |
| 10-19 | **发现并修复梯度同步 bug** | PF/FK 分支在 FSDP 包装之后才安装，每张卡各训一份，检查点只存 0 号卡那份 | 修复：分支改在 FSDP 包装之前安装（`1f9c737`），加载一致性已验证 | 之前所有带 PF/FK 的结论都要重测 |
| 10-19 | **同步版重训**（种子 42） | 在正确的训练上重新判断 | **M3 放开 PF↔FK 62.7 / 70.2**；M3 54.5 / 68.5；vw25 63.3 / 训练中（LSCR 约 69，失败局几乎一个球都没放进去） | M3 本身基本无效；PF↔FK 这座桥有用（比 M3 高约 8 个点），符合项目动机 |
| 10-18 起 | **robotgeometry-v1**（另一个会话） | 不再单独预测 FK，FK 改由动作经 URDF 算出；手部 PF 和动作 FK 加距离 loss；物体 PF 读取动作 FK | task27 68.7 / task21 训练中 | 目前最高，但它用了真值手/物体分割作为输入、mask 也不彻底，提升来自哪里要做消融（见本地 TODO 文档） |
| 10-21 | 在跑：同步版 nolocal 基线、M3 放开 PF↔FK 种子43、组合（M3 放开 PF↔FK + objq32）、raw+vw25 | 公平基线、复现最好的方案、组合有效的方法、检验 vw25 是否依赖 PF/FK | 待补 | – |

**被推翻的旧结论**
- "M3 比 nolocal 高约 10 个点"：同步后和换种子后都没复现。
- "FDM+IDM 有提升"：换种子后没复现，task21 也不涨。
- 10-19 之前用未同步的 nolocal 当基线的比较，都要等同步版 nolocal 的结果出来重新比。

- **完整结果表**在本地 `paper_scj/实验_scr40_20261019*.xlsx`。

## 实验台账（进行中与待补）

规则：实验启动即在此加一行（结果写“待补”），有数据就填，数据齐了把结论写入对应章节并把该行移到“已结”。闭环均为4090、binomial5、200局/点（参考场景同raw）；SR / LSCR 单位%。

### 进行中

> **10-16 存储满中断**：约16:00 UTC `/data` 写满，M1–M4、task27 mix f25/f25i15、task21 mix f25/f25i15、robotframe×2、objq32×2 共12个训练失败（seed43、indep、开发机raw+mix已训完不受影响）。清理：删除784个 `optim/`（未训完的run保留 `latest_checkpoint.txt` 所指ckpt的optim）和56个iter≤50的冒烟/诊断ckpt，`model/` 未动；/data 100%→82%（可用9.4T）。脚本与清单 `/mnt/afs/experiments_20261016/cleanup/`。恢复：原命令逐字重交，任务名加 `-r1`（`/mnt/afs/experiments_20261016/resume_1016.py`；`submit_robotframe.py --suffix -r1 rf40 rf21 oq27 oq21`），从latest续训：M1/M3 39500、M2/M4 38000、task27 mix 36500/37000、task21 mix 32500/36000、robotframe 2000/1500；objq32无ckpt，从头训。随后按用户规则清理中间ckpt（`cleanup_ckpt_rule_1016.sh`，694个）：最大步数<10000的run只留最新；≥10000的run删去10000以下、只留5000倍数；另保留 `latest_checkpoint.txt` 所指及更新的目录。/data 82%→60%（可用21T）。后续闭环只能用 10k/15k/…/40k 这些步数。objq32 task27 改到开发机（10-17 03:27 UTC起，`/mnt/afs/experiments_20261016/robotframe/launch_objq32_t27_devbox.sh`，命令同preset oq27；集群上 `scr40-objq32-r1` 已停），确认读的是objq32 bundle（PF scale 0.035587）；task21 objq32 仍在集群排队。开发机上10-05起的task21 wam 20k进程早在19563步停止（只剩外壳），已清掉。
> **10-17 ckpt清理**（用户规则）：每个run保留latest及更新的iter（含optim）；其余 <10000 全删，≥10000 删500倍数但非1000倍数的iter，1000倍数只留model删optim。删61个iter目录、18个optim，约2.0T，/data 66%→62%（20T可用）。清单与脚本 `/mnt/afs/experiments_20261017/cleanup/{plan_1017.py,del_iter.txt,del_optim.txt,cleanup_1017.sh}`。
> **10-17 ckpt第二次清理**（同一规则）：删42个iter目录（robotframe×2、objq32×2、vw25×2的<10000旧iter，task21 mix_f25 的37500/38500/39500）和3个optim（task21 mix_f25 37k/38k/39k），约1.3T，/data 65%→63%。`/mnt/afs/experiments_20261017/cleanup_1017b/`。
> **10-17 规则改为≥10000只留5000倍数**（用户）：保留latest及更新的iter（含optim）；<10000全删；≥10000非5000倍数整删，5000倍数只留model。本次删21个iter目录（task27/task21 mix_f25、mix_f25i15，M2、M4的36k–39k等），约414G，/data 63%→61%。`/mnt/afs/experiments_20261017/cleanup_1017c/`。

| 实验 | 设置 | 位置 / 产物 | 启动（UTC） | 结果 | 结论去向 |
| --- | --- | --- | --- | --- | --- |
| guidance 2 | nolocal_s4_40000，`policy.g2.yaml` | 886，`runs/eval_200_8gpu_b5_g2` | 10-14 07:45 | 已完成：52.0 / 83.1（none/cov/inv/inv_cov 68/38/48/54） | §0.5 |
| guidance 1.5 | nolocal_s4_40000，`policy.g15.yaml` | 899，`runs/eval_200_8gpu_b5_g15` | 10-14 07:45 | 已完成：48.0 / 80.0（64/36/44/48） | §0.5 |
| 从头训 M1–M4 | §3 | `scr40-mask-m{1,2,3,4}` | 10-14 03:38–04:57 | 训练中：10-15 13:34 UTC iter M1 19980 / M2 19260 / M3 19640 / M4 19120（约6.0 s/it），无Traceback，预计10-17约01:00 UTC训完；之后25k–40k闭环g1 | §3 |
| 从头训 seed43 / indep | §3 | `scr40-{seed43,indep}` | 10-14 05:30 | 训练中：10-15 13:34 UTC iter seed43 20540 / indep 20520（约5.6 s/it），无Traceback，iter_000020000已存，预计10-16约19:30 UTC训完 | §3 |
| 从头训 cons / smooth | §3 | `scr40-{cons,smooth}` | 10-14 09:45提交 | 已取消（10-14 13:10，用户决定）：cons在开发机训到第693步后停掉（为FDM/IDM诊断让卡），不再恢复，输出保留在 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_cons_8gpu_40k_20261014`；smooth在集群排队时取消。节点改给模式混合训练 | — |
| 模式混合训练（wam+FDM / wam+FDM+IDM）从头训40k | nonlocal基础配置不变，统一噪声；`SIM_MODE_PROBS`：`scr40-mix-f25` wam 0.75 / FDM 0.25，`scr40-mix-f25i15` wam 0.6 / FDM 0.25 / IDM 0.15；FDM：第一帧+全部动作为条件，预测视频/PF/FK；IDM：全部视频帧为条件，预测动作/PF/FK；推理不变（wam，guidance 1）。注意：零损失样本仍计入batch均值，视频/动作损失有效权重约按混合比例缩小 | 分支 `pointflow-fk-modemix` `519b688`（worktree `-modemix`），单测5项通过、8卡冒烟通过（`/data/shichaojian/runs/modemix_smoke_20261014/`）；提交脚本 `/mnt/afs/experiments_20261014/modemix/submit_modemix.py`；输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_mix_{f25,f25i15}_8gpu_40k_20261014` | 10-14 13:35提交；10-14 23:01两者均在运行（iter约6030/6070，5.1 s/it，无Traceback），预计10-17约00:00 UTC训完 | 训练中：10-15 13:34 UTC iter f25 15420 / f25i15 15500（约5.6 s/it），无Traceback，预计10-17约03:30 UTC训完；之后25k–40k闭环g1，对照基础40k与seed43 | §3 |
| 模式混合训练 task21 对照（用户提交） | 与上一行 task27 模式混合同配置（nonlocal、`SIM_MODE_PROBS` f25 / f25i15），数据换成 task21 | 任务 `job-0-mix-f25-task21`、`job-0-mix-f25i15-task21`；输出 `/data/shichaojian/runs/sim_task21_ptfk_nonlocal_mix_{f25,f25i15}_8gpu_40k_20261014` | launcher 日志 f25i15 `20261014T152855`、f25 `20261014T202838`；10-15 03:03 UTC 查：f25i15 iter约7480，f25 iter约4200，无Traceback | 训练中：10-15 13:34 UTC iter f25 11000 / f25i15 14300，无Traceback | §3 |
| 模式混合 raw 对照（开发机） | raw/WAM 40k 基础配置不变 + `SIM_MODE_PROBS` wam 0.75 / FDM 0.25，从头训40k，用于判断混合增益是否依赖PF/FK | 启动脚本 `/mnt/afs/experiments_20261014/modemix/launch_raw_mix_f25_devbox.sh`；输出 `/data/shichaojian/runs/sim_task27_wam_mix_f25_8gpu_40k_20261015`；代码 `pointflow-fk-modemix` commit `519b688` | 2026-10-15 03:33 UTC | 训练中：10-15 13:34 UTC iter 9960（约3.6 s/it），无Traceback，预计10-16约20:00 UTC训完 | §3 |
| 示范精度（bad case跟进） | 训练示范中golf/tennis首次抓取指尖-球心水平偏差、掌心离桌高度随桌面偏移的补偿比例、是否有重抓；task21 soy/vinegar松手高度与倾斜（对照闭环：golf放进1.8 / 失败3.3 cm，补偿约40%） | 开发机CPU，脚本 `tools/badcase_demo_precision/`，结果 `/data/shichaojian/runs/demo_precision_20261015/` | 10-15 | **结论：策略的抓取偏差比示范更大，补偿也弱；task21示范无倒瓶。** 各50条独立示范（`_1`为同轨迹重渲染，已排除）。golf首次抓取指尖-球心水平偏差示范中位1.5 cm（p10/p90 1.1/2.4），仅1/50>3.3 cm；闭环成功1.8、失败3.3、全体p90 4.2 cm；tennis示范2.2（1.1/3.4）≈闭环抬起2.5（未抬起4.7）。golf指尖低于球心数示范中位3（闭环成功5/失败1）；掌心离桌高度示范15.5 cm（13.3/18.2），闭环14.6/16.4。补偿：golf掌心离桌高度对桌面偏移的斜率，示范-0.10（95%CI -0.31..0.12，即补偿约0.90），闭环同口径-0.43（补偿约0.57，非简报的40%）；低/高三分位掌心离桌高度差示范-0.7 cm、闭环-2.3 cm。重抓：golf 1/50、tennis 0/50。task21示范soy/vinegar均0次倒下；soy按闭环同规则（手离瓶心>5 cm）松手高度中位5.6 cm（2.3/9.8）、倾斜9.5°、下放速度-0.21 m/s（闭环不倒2.6 cm/8.6°/-0.19，倒下6.0 cm/16.5°），不比闭环不倒组更低；放置时最近物体距离示范终局10.6 cm（8.9/12.5），0/50<6 cm（闭环<6 cm时倒下率64%/89%）。vinegar按该规则不可靠（落在放置时刻），改用瓶轴规则示范松手高度2.2 cm、倾斜1.2°，但无闭环对照。口径注意：示范FK由fk21.npz按overhead相机外参换算到世界系（落地检查通过，未用URDF独立验证）；详见 `/data/shichaojian/runs/demo_precision_20261015/summary.md` | §3.5 |
| 缩短执行步数为何变差 | 排查es16/es8（26.5/5.0）与binomial5平滑、块衔接的关系；是否可做接近物体时提前重规划 | 4090只读排查，结果 `/oss-shichaojian/outputs/infer/task27/analysis/es_chunk_20261015/README.md` | 10-15 | es8不是流水线对齐bug而是闭环死锁：模型单帧无历史且episode内seed固定，抓取前后有"先停约10步再动"的chunk（es32的4.7% query前8行静止、首次运动在第10-13行，集中在动作步~64与224-288），es8只执行静止前缀、观测不变又得同一前缀，动作步~250后卡住（32步窗口净前进<0.006的占比 es32 0%/es16 1.8%/es8 51%，18/24 ep连续停滞>=200步；停滞前速度与es32相同）；代码里无隐含32的实现（contract.py:85取prediction[1:1+es]，平滑仅作用于返回的es行，无chunk拼接）。次要可修因素：binomial5每chunk重锚state，边界沿运动方向损失 1.07/1.11/1.45步(es32/16/8)。es16无死锁，下降原因(集中在golf末阶段，4阶段完成98->53)未解决。方案(未做)：默认es32，接近目标时提前重规划且间隔>=16+停滞保护，先50条none配对测；见README | §3.5 |
| 多采样预测落点分散度 | nolocal 40k每窗口8样本，抓取窗口预测指尖中心终点的样本间分散 vs 成败差距（预测偏差 2.5 vs 3.6 cm） | 开发机CPU，数据 `/data/shichaojian/runs/consistency_multisample_20261014/`，结果 `/data/shichaojian/runs/multisample_spread_20261015/` | 10-15 | **结论：样本间分散远小于闭环差距，不值得做测试时按对齐挑样本。** 设定：nolocal 40k、216窗口（抓取108/其他108），世界系水平xy，活动手指尖中心，t*=GT指尖中心离最近球（box3d）最近的帧，另报末帧；分散=8样本到均值的RMS径向偏差。抓取窗口t*处sd_xy：FK头0.20 cm、FK(预测action)0.31 cm（中位0.15/0.28，p90 0.41/0.45），最大两两距离0.56/0.89 cm，sd_xy>1 cm的窗口0%（末帧0.37/0.42 cm，仅1%>1 cm，最大两两约1.1 cm）；其他窗口相同量级。xy误差（对GT实际指尖）：best-of-8 1.52 / 随机1.74 / 中位1.74 / 均值位置1.73 cm（action 1.58/1.93/1.93/1.90），最好与随机仅差约0.2–0.35 cm；接触子集（t*距球<6 cm，n=60）0.16/0.28 cm，best 1.11 vs 随机1.29 cm。预测指尖到GT球的xy偏移，最优样本 vs 中位样本：抓取5.24 vs 5.45 cm（FK头）、4.89 vs 5.23 cm（action）；接触子集3.53 vs 3.68 / 3.40 vs 3.69 cm，对比成败差距1.1 cm（2.5 vs 3.6），样本内最多挪0.15–0.3 cm。无GT代理：到PF预测物体点的xy距离选样本，误差1.74（FK头，随机1.74）/1.88（action，随机1.93）cm，窗口内Spearman 0.03–0.19；medoid 1.73/1.90，均≈随机，远不及best-of-8；即便用GT球偏移挑最优，也只拿到约0.1–0.3 cm。结论：8个联合样本的落点几乎一样（噪声≪成败差距，也≪模型对GT的1–3 cm误差），差距来自状态/系统偏差而非采样随机性。脚本 `tools/pointfk_consistency_diag/analyze_spread.py`（分支 pointflow-fk-cons），逐窗口csv `windows_spread.csv` | §3.5 |
| PF/FK优势来源与球上点数（闭环） | raw/local/nolocal 30–40k：扰动类别×球交叉表；每次推理各球PF锚点数与预测/执行对准偏差、成败的关系 | 4090只读，结果 `/oss-shichaojian/outputs/infer/task27/analysis/pf_points_20261016/README.md` | 10-16 | **A 优势集中在 cov_only**：同 seed 配对（30/35/40k 合并，每 channel 150 对），cov_only 成功率 local-raw +12.0pp (p=.018)、nolocal-raw +16.0pp (p<.001)，g1 同为 +11.3/+10.7pp (p=.014/.011)；none/inv_only/inv_cov 差 -7 到 +5pp，均不显著。球上以 tennis 最一致（g3 全 channel +6.8/+8.0pp，p≤.001；cov_only +13pp），golf 在 cov_only 同样 +11 到 +17pp（其它 channel 无），pingpong 仅 latched 口径有（g1 cov_only +14pp，按结束时在箱内无差），soccer 无差；故"集中在 cov_only"成立，"只在 tennis/pingpong"不成立（golf 同样）。增益是抗 cov_only 退化（raw 该 channel 仅 28.7%），不是干净条件变强。**B 球上 PF 点数**（g3，local+nolocal，静止球，半径+0.5cm）：golf 手远(>10cm)仅 4-5 点(89% 落 3-5)，手<5cm 才 ~40；tennis 远 ~11、近 ~78；pingpong 19→86；soccer 60→230+（选择器=512全局FPS+每手5cm内256，与模型无关）。golf 接触前最后一次观测的点数与成败几乎无关（rho +0.08，控手距 +0.04 [-0.02,+0.09]，手距分层内 ≈0），与预测偏移也无关；cov-query 点数与成败 +0.15 主要是接近时机/手已在球上的间接指标。tennis 点数与预测偏移负相关（控手距 -0.37 到 -0.46）、与成败弱正相关（+0.13 到 +0.23，基线 92%），可能是遮挡/视角共因。相关性分析，无干预。详见 `/oss-shichaojian/outputs/infer/task27/analysis/pf_points_20261016/README.md` | §3.5 |
| 手-物体相对误差（离线） | nolocal 40k验证窗口：球表面真实锚点数（不含桌面点）；抓取窗口FK头指尖、PF球点的绝对误差 vs 指尖-球相对位置误差 | 开发机CPU，结果 `/data/shichaojian/runs/hand_obj_relative_20261016/` | 10-16 | 球面锚点（|d-r|≤0.7cm且高于桌面≥0.5cm，帧0）中位/p10/p90：足球67/52/254、网球10/4/15、乒乓球22/10/72、高尔夫5/0/31；0点窗口占比 足球0、网球5%、乒乓4%、高尔夫13%；手距球面<5cm时点数明显增多（高尔夫33、网球62）。抓取窗口（单样本a，FK头，t*）：E_hand/E_ball/E_rel(3D均值,cm)=3.11/2.08/2.67（接触窗<6cm: 2.17/2.10/1.20）；末帧接触窗5.6/7.4/4.1。手、球误差正相关（接触窗相关0.75，cos 0.64），相对误差反而更小（81%窗口E_rel<E_hand），说明主要是手和球一起偏的绝对位置误差，不是手-球相对关系；FK(预测动作)的手误差更大（E_rel 2.4）。球点数与E_ball几乎无关（Spearman -0.03），与E_rel在球内负相关（-0.5~-0.8，含手距离混杂；3-5点窗E_rel 3.3 vs >10点 2.1）。8样本均值与单样本几乎一致。详见 summary.md | §3.5 |
| task21 guidance 1 闭环 | raw/local/nolocal 30/35/40k，`policy.g1.yaml`（仅guidance 3→1），binomial5，827步，200局；与原g3同种子配对 | 4090：886 raw 40k/35k，899 nolocal/local 40/35/30k + raw 30k（`/root/launch_knob21.sh`）；run-dir `task21_cosmos_<m>_s4_<step>/runs/eval_200_8gpu{,_shared}_g1` | 10-16 04:15 | 已完成（9/9，各200局，同种子配对，30/35/40k合并600局）。g3→g1：raw 57.7→65.2（g3独赢60/g1独赢105，p=0.0006），local 62.0→69.8（57/104，p=0.0003），nolocal 64.5→69.0（75/102，p=0.050）。PF/FK vs raw：g3 local +4.3 pp（p=0.084）、nolocal +6.8 pp（p=0.004）；g1 local +4.7 pp（p=0.043）、nolocal +3.8 pp（p=0.090）。单点g1：raw 63.0/65.5/67.0，local 70.5/67.0/72.0，nolocal 68.5/68.0/70.5（30/35/40k）。 | §3.5 |
| task21放置的规划与点数（只读） | 倒下的放置：FK头预测的放置位置/倾斜/与已放物体间距是否已偏（规划 vs 执行）；箱内已有物体（msg、soy）PF点数；PF/FK优势按扰动类别×物体拆分 | 4090只读，结果 `/oss-shichaojian/outputs/infer/task21/analysis/placement_plan_20261016/README.md` | 10-16 | local+nolocal 30/35k/40k 共1636次放置(soy 1078倒下10%，vinegar 558倒下41%)：①规划≈执行(间距相关0.98，手xy误差中位0.7cm，倒下/保持无差别)，执行偏离不是主因；②松手时规划间距<6cm仅占倒下的soy 7%/vinegar 17%(保持0-1%)，<6cm则倒下78%/95%；旧“拥挤”口径一部分是倒下后的结果(soy松手后中位再靠近2.9cm)；③规划本身偏高才是主信号：规划松手高度>=6cm占倒下的soy 71%/vinegar 45%(保持5%)，倒下率62%/87%；执行松手倾斜>=25°倒下66-100%(tr含已在倒的，非纯因果)；④PF瓶身锚点PCA预测倾斜与执行倾斜相关仅0.44/0.59(AUC 0.61/0.70，真值0.91)；⑤箱内锚点很少(msg约20、soy约21/1024)，soy倒下时msg锚点更少(14 vs 20，<15个倒下24% vs 7%)，vinegar无关，锚点数不能判拥挤；⑥Q3配对600：SR 57.7→62.0/64.5(+26 p=.084/+41 p=.004)，增益来自“其它”类失败减少(−29/−32，msg抬起未进箱26→15/10、vinegar没抬起11→5/1，集中在inv_only)，soy倒下−8/−12不显著，vinegar倒下反而+12/+14 | §3.5 |
| 推理端选点干预（球保底点数） | nolocal 30/35/40k，guidance 1，仅推理时改选点：先在每个球（soccer/tennis/golf/乒乓）自身候选点上FPS各取至多32点，再全局FPS补到512，再左右手各至多256，总数仍1024；训练不变（存在训练/推理选点分布不一致）；与原g1（52.5/57.0/55.0）同种子配对 | 4090独立worktree+开关（默认关），结果 `task27_cosmos_nolocal_s4_<step>/runs/eval_200_8gpu_b5_g1_ballq32`，说明 `/oss-shichaojian/outputs/infer/task27/analysis/ballquota_20261016/README.md` | 10-16 | 已完成（worktree `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-ballquota`@7babc1f，开关 POINTFK_BALL_QUOTA=32；真实仿真中每球32点已核实）。同种子配对 g1→ballq32：30k 52.5→53.5、35k 57.0→50.5、40k 55.0→55.5；合并600局成功329→319（独赢71/61，p=0.43），golf放进332→323（70/61，p=0.49），tennis 551→551，乒乓537→536，soccer 591→593。无效果。 | §3.5 |
| 桌高跟随：raw vs PF/FK | task27 golf首次抓取时掌心离桌高度对桌面偏移的回归斜率→跟随比例，raw/local/nolocal 30–40k分别算（同一估计方法；示范0.90、local+nolocal合并0.57）；另离线验证窗口看预测手部轨迹高度是否随桌高变化 | 4090只读 `/oss-shichaojian/outputs/infer/task27/analysis/table_follow_20261016/README.md`；开发机CPU `/data/shichaojian/runs/table_follow_20261016/` | 10-16 | 已复现local+nolocal g3 斜率-0.434/follow 0.57(n=1020)。golf掌心follow(30/35/40k，簇bootstrap CI)：g3 raw 0.66[0.57,0.75]、local 0.54[0.43,0.65]、nolocal 0.59[0.47,0.70]；g1 三者0.62/0.62/0.63；PF/FK不比raw更跟随桌高(g3方向反而raw略好，仅local-raw CI不含0；g1无差)，三者都远低于示范0.90。指尖最低点follow PF/FK 0.91高于raw 0.83-0.85，方向相反。tennis同样无PF/FK优势。放置率低桌三分位最差，PF/FK增益主要在低桌。离线验证仅3个桌高(混杂)，无法判定计划本身是否欠跟随(计划误差斜率-0.16~-0.10，3点) | §3.5 |
| 箱子/球锚点与桌高 | PF选中1024锚点里箱子(obj_212_basket_1)与球的锚点世界z是否随桌高dz线性变化(箱空q0与后续query)，及相机光轴系下桌高是否平凡可读；local+nolocal s4 30/35/40k g3，1200 episode×每第4个query共10120 query，只读CPU | 4090只读 `/oss-shichaojian/outputs/infer/task27/analysis/box_table_20261016/README.md` | 10-16 | 箱锚点约318个(占1024的31%，q0候选中箱占50%)，floor 169/rim 49。世界系：箱底z=桌高+20.0mm(std 0.23mm)，对dz斜率1.000/R²1.0000；后续直立箱(倾角<5°，6517行)斜率1.000/R²0.997-0.999/残差0.8-1.4mm；24%行箱被碰歪≥5°(1.7%≥15°)，中位数失真(含倾倒floor中位R²0.61，最低点R²0.90)。球静止q0：球心z-桌高=常数(std 0.24mm)，球锚点z斜率0.99/R²≥0.996，每场景约95个球锚点，是第二通道。相机系：相机高度每场景随机(std 17.5mm，与dz无关)，箱底深度对dz斜率-1.03但R²仅0.68(残差18.7mm)；箱底深度减右手掌深度(q0机器人姿态固定)斜率-1.00/R²0.9987/残差0.95mm。结论：输入已含桌高(世界系精确，相机系需相对读法)，follow不足不是缺信息；未测模型是否使用 | §3.5 |
| 锚点改用机器人坐标系（robotframe） | PF锚点与位移、FK关键点从相机光轴系换到机器人根坐标系（仿真中机器人固定，等于世界系加常数；用每帧相机外参变换）；选点不变（刚体变换下FPS与手5 cm邻域不变）；重算归一化统计；nolocal，从头训40k，其余同基础nolocal；推理端同步变换 | 训练：主仓库（新增转换脚本与默认关闭的坐标系字段，新数据目录 `/data/shichaojian/sim_data/bench2dex_task27_ball_training_robotframe`）；推理：Bench2Dex worktree `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-robotframe`（开关 `POINTFK_GEOMETRY_FRAME`，默认camera）；方案 `/mnt/afs/experiments_20261016/robotframe/PLAN.md` | 10-16 | 实现完成，待冒烟：TR f9f8e4e/f6c8837/552736f/46f24f5，数据 /data/shichaojian/sim_data/bench2dex_task27_ball_training_robotframe，推理worktree /cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-robotframe@d1406a9；SIM_DATA_PIPELINE已更新（fffd39f）；50步冒烟已排队（clusterx `rf-smoke-1016`，10-16 08:40 UTC，提交脚本 `/mnt/afs/experiments_20261016/robotframe/submit_robotframe.py`），通过后提交 `scr40-robotframe` 40k（输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_robotframe_8gpu_40k_20261016`）；task21数据同样已转换（`/data/shichaojian/sim_data/bench2dex_task21_v3_training_robotframe`，xxx/xxx_1掌心差<1e-3 mm）；冒烟改为4卡 `rf-smoke4-1016`（host-10-140-108-12，08:43 UTC起）；4卡冒烟通过（50步、第25步验证、存iter 50，无报错；iter49 loss 13.6–15.4，基础run同步13.4；eval预览+0帧点与手/球/箱重合）。数据终检（独立于转换用的外参：仿真box3d球心经base_link逆变换）：落在球上的锚点中位 soccer 67/tennis 11/golf 5/乒乓23（相机系对照全为0）；运动球的平均位移与球心位移方向余弦中位0.996、长度比0.95（相机系对照残差331 mm）。10-16 09:02 UTC提交 `scr40-robotframe` 40k（排队等节点）；待补 | §3.5 |
| task21 锚点改用机器人坐标系 | 同task27 robotframe：nolocal从头训40k，种子42，仅换数据目录 `/data/shichaojian/sim_data/bench2dex_task21_v3_training_robotframe`（其余同 `sim_task21_ptfk_nonlocal_8gpu_40k_20261010`） | clusterx `scr40-robotframe-t21`，输出 `/data/shichaojian/runs/sim_task21_ptfk_nonlocal_robotframe_8gpu_40k_20261016`；4卡冒烟 `rf21-smoke4-1016`（108.12）；提交脚本 `/mnt/afs/experiments_20261016/robotframe/submit_robotframe.py` | 10-16 09:10 | 4卡冒烟通过（50步、验证、存iter 50，无报错；iter49 loss 13.98，基础run同步15.64；eval预览+0帧点与手/箱重合）；40k排队等节点；待补 | §3.5 |
| 训练时物体保底选点（objq32，task27/task21） | 相机系不变，仅改选点：每个任务物体（task27四个球；task21 msg/soy/vinegar）在自身候选点上FPS至多32点，再全局FPS补到512，左右手各至多256，补足1024；物体归属用仿真物体位姿几何判定，训练与推理同一函数；nolocal从头训40k，种子42；与坐标系实验分开（各自单独对照原基线） | 训练仓库 `977269b`（选点函数）+ `9dbd945`（`prepare_sim_pointfk_v3.py --object-quota`）；数据 `/data/shichaojian/sim_data/bench2dex_task27_ball_training_objq32`、`/data/shichaojian/sim_data/bench2dex_task21_v3_training_objq32`；推理 `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-ballquota`@`154b1d0`（分支 `ballquota-inference`），开关 `POINTFK_OBJECT_QUOTA=32`（可选 `POINTFK_OBJECT_TASK`） | 10-16 | 数据：task27 `/data/shichaojian/sim_data/bench2dex_task27_ball_training_objq32`（各球选中点中位 61/32/32/32，原68/11/5/22）、task21 `..._task21_v3_training_objq32`（msg/soy/vinegar 32/32/32，原23/27/22）；划分与动作统计逐字节同原数据，PointFlow scale 27: 0.035317→0.035587、21: 0.036563→0.036370。训练/推理选点在60个4090 query上逐索引一致，几何判定与渲染标签85,899点全一致。10-16 10:00 UTC提交4卡冒烟 `oq27-smoke4-1016`/`oq21-smoke4-1016` 与40k `scr40-objq32`/`scr40-objq32-t21`；两个4卡冒烟均通过（50步、验证、存iter 50，无报错；iter49 loss task27 15.41 / task21 13.99），40k排队等节点。推理需 `POINTFK_OBJECT_QUOTA=32`（ballquota worktree@154b1d0）；待补 | §3.5 |
| 从头训各组闭环（25k起） | guidance 1，binomial5，200局，同种子：task27 scr40-seed43/indep/M1–M4、mix f25/f25i15、raw+mix f25（开发机）各25k；对照 nolocal/raw s4_25000 g1（新跑）；task21 mix f25i15/f25 25k，对照 task21 nolocal s4_25000 g1。之后30k/35k/40k随训练进度补 | 4090 886/899；交接按bench2dex-cloud-handoff skill流程，模型目录 `task27_cosmos_nolocal_scr40_<tag>_<step>` 等，说明 `/oss-shichaojian/outputs/infer/task27/analysis/scr40_closedloop_20261016/README.md` | 10-16 | 25k基本完成（10-16 15:40 UTC），SR%（LSCR%）；none/cov/inv/inv_cov，同种子配对，Fisher p对nolocal基线：<br>task27：基线nolocal 50.5（80.88）70/36/52/44；raw 54.0；seed43 57.0（84.38）66/38/62/62，p=0.14；indep 55.0（83.38）82/44/50/44；mix_f25 58.0（86.12）68/60/58/46，p=0.06；mix_f25i15 57.5（83.75）74/44/54/58，p=0.08；raw+mix_f25 53.0（82.25）对raw 54.0，p=0.89；M1 geom_readonly 49.0（81.75）64/42/50/40（mask运行时，p=0.84）；M4 action_observed_vision 52.0（83.38）70/42/50/46（p=0.84）；M2 action_no_geom 58.5（85.50）82/52/48/52（mask运行时重跑，p=0.13，对seed43 p=0.84）；M3 physical_split 60.0（84.12）82/46/52/60（p=0.07，对seed43 p=0.61）。M2/M3高于基线但与seed43（57.0）相当，25k单点仍不能区分。<br>task21：基线nolocal 70.0；mix_f25 63.0（90.50）84/56/52/60，p=0.17；mix_f25i15 65.5（89.00）78/58/58/68，p=0.39。<br>判断：仅换种子（seed43）就比基线高6.5点，task27所有新组（mix、indep、M1/M4）都落在基线与seed43之间（49–58），25k单点区分不了；task21两组mix低于基线。需等30/35/40k合并后再下结论。<br>M1–M4首轮作废：`worldact_root` 指向主运行时（无mask实现），M1得0%。已改为 `Bench2Dex-Pointflow-fk-mask` 运行时重跑；M1、M4的有效结果为上面的数。M2、M3在OSS上的改名未生效，旧目录 `eval_200_8gpu_b5_g1` 仍是无效run（已放 `INVALID_nomask.txt` 标记），有效重跑写入 `eval_200_8gpu_b5_g1_maskrt`（`RN_SUFFIX=_maskrt bash /root/launch_scr40.sh ...`，10-16 15:30 UTC起：M2在886、M3在899）。mask生效的判据：worker日志出现 `Attention backend selected: flash2 (... causal=False, varlen=True ...)`；无效run中为 `varlen=False`，且日志里有主运行时绝对路径。交接与说明 `/oss-shichaojian/outputs/infer/task27/analysis/scr40_closedloop_20261016/README.md` | §3 |
| 从头训各组闭环 30k/35k/40k | guidance 1，binomial5，200局，同种子；task27：seed43、indep、M1–M4（mask运行时）、mix_f25、mix_f25i15、raw+mix_f25；task21：mix_f25、mix_f25i15。共33个run。raw/local/nolocal基线已有，不重测 | A800上传 `/root/claude_tmp/upload_scr40/up_step.sh`（链 `up_chain_30k40k.sh`），4090准备 `/root/prep_step.sh`（由25k模型目录只替换步数派生，worldact_root/mask/indep不变；先暂存到/cpfs再拷OSS，等分片mtime稳定），驱动 `/root/drive_30k40k.sh`（886取偶数项、899取奇数项），评测 `/root/launch_scr40.sh` | 10-17 07:00 UTC | **完成**（最后一个run 10-17 19:40 UTC结束，合并结果见§0.9）。seed43 30k首次评测因ossfs mtime竞态在暂存时中止（目录内 `ABORTED.txt`），有效结果在 `eval_200_8gpu_b5_g1_r2`。结果（SR%（LSCR%）；none/cov/inv/inv_cov；对照 g1 30k raw 52.0、local 54.5、nolocal 52.5）：30k M3 physical_split 63.5（85.88）66/60/66/62，对raw p=0.026、对nolocal p=0.033；M1 54.0（82.50）76/50/46/44；indep 52.5（81.50）68/44/54/44；M4 action_observed_vision 54.0（83.00）74/34/54/54。35k（对照 g1 raw 54.0、nolocal 57.0）：M3 65.5（87.88）74/58/66/64，对raw p=0.025、对nolocal p=0.10；indep 49.5（80.00）72/34/48/44。40k（对照 raw 51.0、nolocal 55.0）：mix_f25i15 66.0（87.88）76/56/68/64，对nolocal p=0.032、对raw p=0.003。task21 30k（对照 raw 63.0、nolocal 68.5）：mix_f25 67.0（90.67）82/62/60/64。M3 30k+35k合并400局 258 vs nolocal 219（p=0.006）、vs raw 212（p=0.001），cov_only两点均≥58（基线38–48），40k待出。10-17 12:40 UTC新增：indep 40k 47.0；M2 action_no_geom 30k 56.0；M4 35k 57.0；seed43 35k 55.0；task21 mix_f25 35k 54.0（对照nolocal 68.0）。注意：10-17冒烟发现PF/FK分支梯度未跨卡同步（见架构文档），以上PF/FK结果均带此问题，raw不受影响。排队事故：10-17 约04:20 UTC为P3插队停掉886上12个等待中的run（清单 `/root/claude_tmp/requeue_886.txt`），P3结束后漏了重新排队，886之后只跑驱动新提交的run；10-17 10:35 UTC用 `/root/claude_tmp/requeue_886_run.sh` 按原顺序重排（同run目录续跑）。10-17 19:40 UTC前新增：task27 mix_f25i15 30k 56.5（84.50）、35k 57.5；mix_f25 35k 55.5、40k 54.5；M2 35k 56.5、40k 55.5；M1 40k 51.0；raw+mix_f25 30k 53.0、40k 56.0；task21 mix_f25i15 30k 65.0、35k 65.5、40k 62.0；task21 mix_f25 40k 64.0。886/899之后被平台回收（队列跑完后空闲），换为4090 907/904 | §3 |
| PF/FK分支梯度未跨卡同步（问题与修复） | `pointflow_branch`、`fk_branch`（及后装模块）在FSDP包装之后安装，是普通参数，FSDP不做梯度all-reduce；各卡各练一份，checkpoint只存rank0那份。证据：已训checkpoint中这两组参数未分片（task21 local 40k：fk_branch 30、pointflow_branch 356个张量均未分片，主干与action/vision投影均分片）；4卡冒烟第20/40次前向4卡fk_branch参数和不同且发散（-0.011/-0.056/-0.064/-0.078）。影响：所有local/nolocal及其衍生PF/FK模型（M1–M4、mix、robotframe、objq32、vw25、indep、seed43）；raw不受影响。修复：`model.config.sync_plain_param_grads=true`（首次反向后从rank0广播net与EMA中的普通参数，此后每次反向后对其梯度做AVG all-reduce，在梯度裁剪之前），分支 `pointfk-arch` `4ca04c6`，2进程gloo单测通过；10-18 只移植同步部分到主分支 `047dd24`（默认关闭，单测通过），8卡冒烟未做 | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-arch` | 10-17 | 修复版用于架构v2六个模型（R3/R4/R5/A/R2/R1）；用户要求先提交这六个；是否补修复版nolocal/M3基线待定，与旧基线对比时需注明修复差异 | 架构文档 |
| 架构v2 第一批 R3 M3a（task27） | M3的6条规则中 `action<-pointflow` 改为 `action<-pointflow:noisy`（action可读当前PF anchor、不读带噪未来PF），从头训40k，种子42，nonlocal，其余同 `scr40-mask-m3`；对照task27 M3 | `/mnt/afs/experiments_20261017/arch/submit_batch1.py r3`（mask worktree），clusterx `scr40-m3a`，输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_mask_m3a_8gpu_40k_20261017`；4卡冒烟 `m3a-smoke27-1017` 通过（50步、验证、存iter 50，日志确认6条规则含`:noisy`） | 10-17 12:02 UTC（未修复版，12:55停止，输出保留不用） | 改用修复版重跑：`submit_arch_sync.py r3`，clusterx `scr40-m3a-sync`，输出 `..._mask_m3a_sync_8gpu_40k_20261017`；25k–40k闭环g1 | 架构文档§3.1；**10-17 15:10 UTC用户要求R系列全部暂停**（R3/R4/R5修复版已提交后停止，R1/R2未提交） |
| M3 task21 + M3/FDM+IDM 种子43复现（task27、task21） | 均为原run命令只改任务/种子/输出，不开梯度同步（与原M3、mix及raw/nolocal基线一致）：① M3 task21 种子42；② M3 task27 种子43；③ M3 task21 种子43；④ mix_f25i15 task27 种子43；⑤ mix_f25i15 task21 种子43。M3用mask worktree `3788ac6`，mix用modemix `519b688`。另做M3 task21 4卡冒烟 | `/mnt/afs/experiments_20261017/m3mix/submit_m3mix.py`；clusterx `scr40-mask-m3-t21`、`scr40-mask-m3-s43`、`scr40-mask-m3-t21-s43`、`scr40-mix-f25i15-s43`、`scr40-mix-f25i15-t21-s43`、冒烟 `m3t21-smoke4-1017`；输出 `/data/shichaojian/runs/sim_task{21,27}_ptfk_nonlocal_{mask_physical_split,mix_f25i15}[_seed43]_8gpu_40k_20261017` | 10-17 15:55 | 均已运行；M3 task21 4卡冒烟通过（日志确认physical_split 6条规则，训练到第25步并完成验证，无报错），16:06 UTC停止并删除冒烟输出；结果待补。**10-17腾卡调整**：⑤ mix_f25i15 task21 种子43（16:14 UTC）与 vw25 task21（16:20 UTC，停在iter 5402，latest 5000保留）被停止，非本会话操作；③ M3 task21 种子43 于16:36 UTC按用户要求停止并删除输出（未存ckpt）；④ mix_f25i15 task27 种子43 迁到108-126：先提交 `scr40-mix-f25i15-s43b`（钉108-126，命令同原任务，仅输出改为 `..._seed43_8gpu_40k_20261017b`），确认已开训后停止62-37上的原任务并删除其输出（仅iter 500）。现存：① `scr40-mask-m3-t21`、② `scr40-mask-m3-s43`、④ `scr40-mix-f25i15-s43b`。**10-18 已重新提交排队（用户）**：③ `scr40-mask-m3-t21-s43-r2`、⑤ `scr40-mix-f25i15-t21-s43-r2`（均从头训，命令同原任务，只任务名加 `-r2`，因原名仍以TERMINATED占用）；vw25 task21 见vw25行。原记录：**暂停（10-17，为同事腾卡，用户确认）**：③ M3 task21 种子43、⑤ mix_f25i15 task21 种子43（均无ckpt，恢复时从头重交）；vw25 task21见vw25行（25k–40k g1闭环，M3须mask运行时；对照本任务raw与nolocal，种子43另对照种子42原run） | §3 |
| robotframe 20k 闭环（task27、task21） | 机器人坐标系锚点模型 `scr40-robotframe-r1` / `scr40-robotframe-t21-r1` 的 iter 20000（用户要求，早于25k起评惯例），g1、binomial5、200局；对照同步数无（基线最早25k），参考 nolocal 25k g1（task27 50.5、task21 70.0） | A800上传 `/root/claude_tmp/upload_scr40/up_rf20k_finish.sh`（桶 `task{27,21}/cosmos_nolocal/scr40_robotframe`）；4090备料 `/cpfs/user/shichaojian/claude_4090_scripts/prep_robotframe20k.sh`：模型目录 `task{27,21}_cosmos_nolocal_scr40_robotframe_20000`，由 `*_nolocal_s4_25000` 只换路径派生，`worldact_root` 指 `Bench2Dex-Pointflow-fk-robotframe`（`d1406a9`），模型目录内 `model_environment.json` 加 `POINTFK_GEOMETRY_FRAME=robot_base` 并用本run的PF/FK缩放；评测 `bash /root/launch_scr40.sh <task> <dir> robotframe` | 10-18 | **完成**（10-18 08:40 UTC）。task27 57.0（none/cov/inv/inv_cov 39/21/29/25）；task21 64.0（LSCR 87.67，42/27/27/32）。参考：task27 nolocal 25k 50.5、4点合并53.8、seed43 57.1、raw 52.8；task21 nolocal 25k 70.0、合并69.2、raw 65.1。判断：单点20k，task27在run间波动内，task21低于nolocal约5–6 pp、与raw相当；需25–40k同步数比较。首次启动失败：推理worktree `pointfk_policy.py` 在 `super().__init__` 之前导入 `cosmos_framework`（运行时尚未加入sys.path），全部episode ModuleNotFoundError；修复 `7286fbd`（导入移到其后），失败run已改名 `*_FAILED_import` | §3 |
| 基线20k guidance 1（与robotframe 20k对比） | task27、task21 的 raw（wam）、nolocal、local 40k训练run的 iter 20000，g1、binomial5、200局，同协议同种子（用户要求） | 4090：task27在907、task21在904，各按 nolocal→local→raw 排队；模型目录 `task{27,21}_cosmos_{raw,nolocal,local}_s4_20000`（已有，与25k目录仅步数不同），`bash /root/launch_scr40.sh <task> <dir> main`；此前这些20k只有guidance 3结果（task27 raw 42.0/nolocal 38.5/local 48.5，task21 55.5/60.5/61.5） | 10-18 | **完成**（10-18 11:05 UTC）。g1 SR%（none/cov/inv/inv_cov 成功/50）：task27 raw 51.5（37/19/29/18）、nolocal 50.5（33/19/29/20）、local 51.5；task21 raw 64.5、nolocal 65.5（39/29/33/30）、local 68.5。对比robotframe 20k：task27 57.0（高5.5–6.5 pp）、task21 64.0（与raw/nolocal相当，低于local 4.5 pp）。均单run单点，差值在run间波动（3–6 pp）内，需25–40k合并 | §3 |
| objq32 20k 闭环（task27、task21） | 训练时物体保底选点32（`scr40-objq32-t21-r1`、开发机task27）的 iter 20000，g1、binomial5、200局 | 推理与训练一致的核对（10-18）：推理worktree `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-ballquota`@`154b1d0` 的objq选点块与训练 `cosmos_framework/data/pointflow_anchor_selection.py`（`977269b`）逐字相同（94行），`anchor_fps`/`hand_guided_fps` 亦相同；训练数据manifest：quota 32、margin 0.005 m、radius 0.05、1024点、按仿真box3d几何判定归属；推理用仿真位姿+网格包围盒做同一几何判定。开关在仿真进程：`launch_scr40.sh … objq32` 在该worktree下 `export POINTFK_OBJECT_QUOTA=32 POINTFK_OBJECT_TASK=<task>`；模型端运行时同主checkout，`model_environment.json` 用本run的PF缩放（task27 0.035587、task21 0.036370，与基线不同）。脚本：A800 `/root/claude_tmp/upload_scr40/up_first.sh`，4090 `prep_objq32_20k.sh`、`drive_1018.sh`（`/cpfs/user/shichaojian/claude_4090_scripts/`）；模型目录 `task{27,21}_cosmos_nolocal_scr40_objq32_20000` | 10-18 | 运行中。**task27选点核对通过**（68局/2091次query，`objq_check.py`）：渲染标签判为各球的点100%被几何判定归到同一球，几何判定多出约0.4%（5 mm margin），4个球每次均得32保底点；task21选点核对通过（200局/4306次query）：渲染标签点99.9–100%落在几何判定的同一瓶内，几何判定多出约6%（瓶体较大，5 mm margin覆盖面多）。**完成**（10-18）：task27 56.5（40/20/28/25），task21 67.0（41/31/29/33）。对照20k：task27 nolocal 50.5、raw 51.5；task21 nolocal 65.5、raw 64.5、local 68.5。单点，task27高约5–6 pp、task21与基线相当，需25–40k合并 | §3 |
| 从头训各组 20k 闭环 | task27：M3、FDM+IDM、seed43、M2、mix_f25、M4、M1、indep、raw+mix_f25；task21：FDM+IDM、mix_f25。g1、binomial5、200局，与基线20k、robotframe 20k同步数对比 | A800 `up_chain_1018.sh`（`up_step.sh` 20000）；4090 `drive_1018.sh` + `prep_step.sh`（由25k目录只换步数派生，mask模型仍用mask运行时）；907/904各一半 | 10-18 | **完成**（10-18）。g1 SR%（none/cov/inv/inv_cov）：task27 M3 62.5、FDM+IDM 62.0、M4 57.5、mix_f25 55.5（38/18/31/24）、seed43 54.5、M2 53.0、M1 52.5（37/25/23/20）、indep 51.0（32/21/29/20）、raw+mix_f25 50.5（36/18/22/25）；task21 FDM+IDM 65.5（39/31/30/31）、mix_f25 59.0（36/26/26/30）。基线20k：task27 raw 51.5、nolocal 50.5；task21 raw 64.5、nolocal 65.5。判断：20k的排序与25–40k合并一致——task27 M3、FDM+IDM领先（+11 pp于nolocal），M1/indep/raw+mix无增益；task21 FDM+IDM与nolocal持平、mix_f25低6.5 pp。单点，需种子复现 | §3 |
| robotframe 25k–40k 闭环 | 同robotframe 20k设置，训到即测 | A800 `up_chain_1018.sh` 第3路；4090 `prep_rf_step.sh`（由20k目录只换步数）；task27在907、task21在904 | 10-18 | 等训练；结果待补 | §3 |
| 在训各run训到即测（20k–40k，每5k） | objq32 task27/21（25–40k）；vw25 task27/21；M3 task21（种子42）；M3 种子43 task27/21；FDM+IDM 种子43 task27/21。g1、binomial5、200局（robotframe另见其行） | A800 `/root/claude_tmp/upload_scr40/up_auto_1018.sh`（每run一路：首步 `up_first.sh` 带训练元数据，其后 `up_step.sh`）；4090 `drive_auto.sh` + `prep_auto.sh`（`/cpfs/user/shichaojian/claude_4090_scripts/`）：模型目录 `<task>_cosmos_nolocal_<桶名>_<step>` 由 `<task>_cosmos_nolocal_s4_25000` 派生，PF/FK缩放、`joint_attn_mask`、`loss_scale` 取自该run训练配置，mask模型用mask运行时（不一致即拒绝），objq32用 `root=objq32`；907：objq32 t27、vw25 t27、M3 s43 t27、FDM+IDM s43 t21；904：objq32 t21、vw25 t21、M3 t21、M3 s43 t21、FDM+IDM s43 t27 | 10-18 | 自动进行中（截至10-19 03:00 UTC）。SR%（20k/25k/30k）：task27 robotframe 57.0/55.5/54.0、objq32 56.5/59.5/56.0、vw25 60.0/61.0/–、M3 s43 61.0、FDM+IDM s43 54.5；task21 robotframe 64.0/64.0/60.5、objq32 67.0/72.0/–、M3(s42) 69.0。同步数基线：task27 raw 51.5/54.0/52.0、nolocal 50.5/50.5/52.5、local 51.5/58.5/54.5、seed43 54.5/57.0/58.0、M3(s42) 62.5/60.0/63.5、FDM+IDM(s42) 62.0/57.5/56.5；task21 raw 64.5/65.0/63.0、nolocal 65.5/70.0/68.5、local 68.5/61.0/70.5、FDM+IDM 65.5/65.5/65.0。初步：M3 s43 20k复现s42（61.0 vs 62.5）；FDM+IDM s43 20k低于s42 7.5 pp；robotframe无增益；task21重交3个任务尚未开训 | §3 |
| M3 放开PF↔FK（从头训40k，task27、task21） | M3 基础上恢复 PF↔FK，只切 视频↔FK、PF↔动作（mask `fk<-vision;vision<-fk;pointflow<-action;action<-pointflow`），种子42，不开梯度同步；`scr40-mask-m3-pffk` / `scr40-mask-m3-pffk-t21`（用户要求，先做这一格） | `/mnt/afs/experiments_20261017/m3mix/submit_m3mix.py` preset `m3pffk27`/`m3pffk21`，输出 `sim_task{27,21}_ptfk_nonlocal_mask_m3_pffk_open_8gpu_40k_20261019`；A800 `up_auto_1019.sh`，4090 `drive_auto.sh <task> scr40_m3_pffk mask`；不限节点 | 10-19 | 排队中（10-19 06:10 UTC 重交；此前误删一次丢了排队位置）；结果待补 | §3 |
| task27 FDM+IDM 40k 重复评测 | `task27_cosmos_nolocal_scr40_mix_f25i15_40000` 原样重测一次（g1、binomial5、200局、同一协议与种子），看66.0%的单点随机性 | 4090 886，`RN_SUFFIX=_rep2 bash /root/launch_scr40.sh task27 task27_cosmos_nolocal_scr40_mix_f25i15_40000 main`，run `eval_200_8gpu_b5_g1_rep2` | 10-17 16:20 | **完成**（10-17 19:30 UTC）：64.0（86.75）78/52/66/60；原次66.0（87.88）76/56/68/64，两次相差2 pp，40k高点可复现，两次平均65.0 | §3 |
| 架构v2（梯度同步修复版） 第二批 R4/R5 E-rel（task27，M3上） | 12个查询点（双手掌心+指尖）各取最近16个任务物体点（无硬半径），相对坐标/RBF距离/法向/颜色/实例 + 每实例最近距离/点数 → 加到全部action输入行（零初始化）；R5同参数量但用相机系绝对坐标；从头训40k，M3，种子42；对照task27 M3 | 代码分支 `pointfk-arch`（`/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-arch`），`model.config.pointfk_relation=rel|abs`；数据 `/data/shichaojian/sim_data/bench2dex_task27_ball_training_rel`（与原数据包逐窗口一致，仅加 `point_instance`，`prepare_sim_pointfk_v3.py --save-object-instance --object-task task27`）；`submit_batch2.py r4 r5`，clusterx `scr40-m3-erel` / `scr40-m3-erelabs` | 冒烟中 | 单测14项通过；4卡冒烟 `erel-smoke27-1017` 进行中 | 架构文档§7、实现细则§4；**10-17 15:10 UTC用户要求R系列全部暂停**（R3/R4/R5修复版已提交后停止，R1/R2未提交） |
| 架构v2（梯度同步修复版） A1（task27，nonlocal，不用M3） | 网络内用 x_t−σv 估计干净action → URDF算指令FK（相机系）→ PF解码前cluster hidden交叉注意FK关键点（零初始化，门控(1−σ_a)²）；独立未来FK失效：mask `vision/action/pointflow<-fk:noisy` + `fk_loss_weight=0`，保留当前FK anchor；从头训40k，种子42；对照task27 nolocal | `model.config.pointfk_future_fk=action_derived pointfk_output_adapter=action_to_pf`；`submit_batch3.py a`，clusterx `scr40-a1` | 待冒烟 | 单测通过（指令FK与实际FK位移差14 mm，与离线诊断一致） | 实现细则§6 |
| 架构v2 C0（task27，nonlocal，不用M3） | 同A的指令FK，按PF内参与uv→视频仿射投到主视角patch，每个带噪视频token以局部偏置交叉注意本样本本时间块的42个关键点（含null键），零初始化加到llm2vae前，门控(1−σ_a)²；首版只用FK、不含PF点，不加局部video loss；独立未来FK保留；从头训40k，种子42；对照task27 nolocal | `model.config.pointfk_output_adapter=geometry_to_video`；`submit_batch3.py c`，clusterx `scr40-c0` | 暂缓（用户10-17决定） | 单测通过（投影与PF规则一致、局部性、门控） | 实现细则§6 |
| P3 接近接触时提前重规划（只改推理） | nolocal 40k g1，200局，同种子对g1基线55.0。每步看本块预测：预测指尖与预测球点距离<4.5 cm且本块已执行≥8步时截断本块重推理，每个球每局最多一次（来源 `/oss-shichaojian/outputs/infer/analysis/gap_levers_20261016/README.md` P3）。离线回放40局：每局每球各触发一次，触发步中位19.5，抓取时预测指尖–球距离中位0.27 cm（坐标系一致） | worktree `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-p3`（`p3-replan-inference` `ed49d6e`，开关 `POINTFK_P3_REPLAN=1`，`POINTFK_P3_DIST_CM`、`POINTFK_P3_MIN_STEPS`；不支持 `--async-inference`），启动 `RN_SUFFIX=_p3 bash /root/launch_scr40.sh task27 task27_cosmos_nolocal_s4_40000 p3`，run `eval_200_8gpu_b5_g1_p3`；用户要求最高优先级，886队列首位 | 10-17 07:40 UTC | 完成：SR 49.5（99/200）vs g1基线55.0（110/200），Fisher p=0.32，未改善、略差；分组与golf抓取对准待分析 | §3.5 |
| 降低video权重（vw25，从头训40k） | nolocal基础配置，种子42，相机系，仅 `model.config.rectified_flow_training_config.loss_scale` 10→2.5（action 10、PF/FK 1不变），video:action 1:1→1:4，检验降低video主导是否让action更多依赖几何；task27与task21各一 | `/mnt/afs/experiments_20261017/videoweight/submit_videoweight.py`（smoke27/smoke21/vw25/vw25t21）；推理用主运行时、无额外开关 | 10-17 08:20 UTC提交 | 冒烟通过（iter49 loss：task27 13.46、task21 12.80，config确认loss_scale 2.5、action_loss_weight 10）；40k已提交 `scr40-vw25` → `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_vw25_8gpu_40k_20261017`、`scr40-vw25-t21` → `/data/shichaojian/runs/sim_task21_ptfk_nonlocal_vw25_8gpu_40k_20261017`，排队中；结果：待填。**task21 10-18 已重新提交 `scr40-vw25-t21-r2`（原命令，从latest iter 5000续训）**；原记录：**task21 `scr40-vw25-t21` 暂停（10-17 16:20 UTC，为同事腾卡，用户确认）**：停在iter 5402，保留 latest iter 5000（含optim），恢复时原命令重交、从5000续训；task27 `scr40-vw25` 继续 | §3 |
| guidance 1下PF/FK vs raw配对检验 | 30/35/40k合并，同种子配对，按扰动类（g3同算作对照） | 886 CPU，`/oss-shichaojian/outputs/infer/task27/analysis/g1_paired_20261014/`（`paired.py`、`summary.txt`） | 10-14 11:15 | 已完成，结论见§0.5（g1：cov_only nolocal 45.3 vs raw 34.7，p=0.011；总体 p=0.27） | §0.5 |
| 离线多采样挑选 | nolocal 40k，每窗口8个联合采样，按FK头–FK(action)不一致挑选，对比随机/中位样本/oracle | 开发机8卡，`/data/shichaojian/runs/consistency_multisample_20261014/`，代码 `-cons` 分支 `tools/pointfk_consistency_diag/` | 10-14 11:30 | 已完成：N=8、216窗口，k=0与单采样一致（RMS 0.0015 rad）。argmin c_rel 对k0胜率64%（p=1.6e-4）、fkact_vs_fkact_gt 30.72→30.45 mm（-0.27，约1%），对实际轨迹58%（p=0.035）、动作MAE无提升（47%）；8动作均值-0.37 mm更好；窗口内Spearman 0.25；不值得部署挑选，见 `-cons` 文档§6 | §3 |
| FDM/IDM真值注入诊断 | nolocal 40k，约216窗口；IDM：每步注入带噪真实视频，看动作误差降幅；FDM：注入带噪真实动作及反事实动作（静止、换窗口），看视频/PF/FK是否随动作改变，部分窗口解码视频并排对比 | 开发机（scr40-cons已取消），`/data/shichaojian/runs/fdm_idm_probe_20261014/`，代码 `-cons` 分支 `tools/pointfk_consistency_diag/` | 10-14 12:00 | 已完成：216窗口，注入带噪GT。IDM：GT视频使FK(act)对指令FK误差 30.7→17.5 mm（-43%，h1 -7%、h32 -50%），动作MAE仅-17%，FK头/PF -72%/-57%；FDM：动作换成GT/静止/别窗口，视频latent变化仅为种子噪声的1%，FK/PF<0.5 mm（噪声3–4 mm）——视频不跟随动作，信息流基本单向视频→动作；见 `-cons` 文档§7 | §3 |

### guidance 1 同口径对比（30/35/40k合并，600局/模型，已完成，结论见§0.5）

| 模型 | 总体 | none | cov_only | inv_only | inv_cov |
| --- | --- | --- | --- | --- | --- |
| raw | 52.3 | 76.7 | 34.7 | 51.3 | 46.7 |
| local | 54.7 | 72.0 | 46.0 | 48.7 | 52.0 |
| nolocal | 54.8 | 69.3 | 45.3 | 52.7 | 52.0 |

单点SR（g1）：raw 30k/35k/40k 52.0 / 54.0 / 51.0；local 54.5 / 54.5 / 55.0；nolocal 52.5 / 57.0 / 55.0。初步观察（待配对检验确认）：关CFG后PF/FK总体领先缩小到约2.5个点（g3约4个点）；cov_only仍领先约11个点；none上raw更高（76.7 vs 69–72）。

## Baseline：guidance 1（2026-10-16起所有比较以此为准）

设置：4090闭环，binomial5，num_steps 4，execute_steps 32，guidance 1（`policy.g1.yaml`，仅改guidance），每点200局=四组（none/cov_only/inv_only/inv_cov）各50，同一批场景与种子（task27带 `--reference-run`、1190步；task21 827步、无reference run）。新实验与对应ckpt的这些run同种子配对比较。数值为SR%（LSCR%），四组为SR%。

task27（run-dir `/oss-shichaojian/outputs/infer/task27/task27_cosmos_<m>_s4_<step>/runs/eval_200_8gpu_b5_g1`，raw为 `eval_200_8gpu_shared_b5_g1`）：

| ckpt | raw | local | nolocal |
| --- | --- | --- | --- |
| 25k | 54.0（82.62）；74/42/52/48 | 58.5（86.00）；74/52/52/56 | 50.5（80.88）；70/36/52/44 |
| 30k | 52.0（82.38）；76/42/48/42 | 54.5（84.38）；72/44/52/50 | 52.5（82.62）；68/38/46/58 |
| 35k | 54.0（83.25）；74/34/56/52 | 54.5（84.25）；70/44/48/56 | 57.0（83.88）；68/48/60/52 |
| 40k | 51.0（82.12）；80/28/50/46 | 55.0（84.38）；74/50/52/44 | 55.0（84.88）；72/50/52/46 |
| 30–40k平均 | 52.3 | 54.7 | 54.8 |

task21（run-dir `/oss-shichaojian/outputs/infer/task21/task21_cosmos_<m>_s4_<step>/runs/eval_200_8gpu_g1`，raw为 `eval_200_8gpu_shared_g1`）：

| ckpt | raw | local | nolocal |
| --- | --- | --- | --- |
| 25k | 65.0（89.17）；82/58/60/60 | 61.0（90.33）；74/58/56/56 | 70.0（91.83）；76/64/68/72 |
| 30k | 63.0（88.00）；74/58/56/64 | 70.5（91.33）；80/66/70/66 | 68.5（89.50）；84/62/62/66 |
| 35k | 65.5（88.00）；88/58/62/54 | 67.0（93.00）；84/58/68/58 | 68.0（92.17）；82/56/68/66 |
| 40k | 67.0（88.67）；82/70/60/56 | 72.0（92.50）；90/60/64/74 | 70.5（91.83）；82/60/68/72 |
| 30–40k平均 | 65.2 | 69.8 | 69.0 |

30–40k合并配对（600局）PF/FK−raw：task27 nolocal +2.5 pp（p=0.27），仅cov_only显著（45.3 vs 34.7，p=0.011）；task21 local +4.7 pp（p=0.043）、nolocal +3.8 pp（p=0.090）。guidance 3旧数值见§0.5、§2及台账，仅作历史参考。

## 结论（截至2026-10-15 03:30 UTC）

10-14新增：

- 推理关闭CFG（guidance 3→1）是目前最大的单项收益：nolocal 30/35/40k合并SR 48.2→54.8，同种子配对g1独赢100 vs g3独赢60（p=0.002）；raw/local 40k同向（45→51、50→55）。guidance 2为52.0、1.5为48.0，1最好。num_steps 4→8（43.5）、execute_steps 16/8（26.5/5.0）都更差。详见§0.5。
- guidance 1下PF/FK相对raw总体不显著（nolocal 54.8 vs raw 52.3，p=0.27），唯一显著的仍是外观变化：cov_only nolocal 45.3 / local 46.0 vs raw 34.7（p≈0.01）；none类raw反而略高（76.7 vs 69.3，p=0.10）。PF/FK的作用是外观泛化，不是整体提升。
- 一致性诊断：各模态之间彼此一致但一起错（FK头 vs FK(action) 14 mm，各自对真值26–31 mm），一致性损失空间有限；离线多采样挑选只降0.3 mm，不值得部署。
- FDM/IDM真值注入：给真值视频，动作FK误差降43%（30.7→17.5 mm，越靠后的帧降得越多）；反过来视频/FK/PF对注入的动作几乎不敏感（≈种子噪声的1%）。信息流基本单向视频→动作，动作精度受梦出来的视频限制，这是video-action gap的直接证据。模式混合训练（§3，进行中）即针对此。

10-13及以前（guidance 3）：

- 训练步数比结构改动重要。4090上 `task27_cosmos_{raw,local,nolocal}_s4_*`（40k训练，binomial5，200局/点）：20k以内PF/FK不占优；25k–40k四点平均SR raw 43.8 / local 46.8 / nolocal 47.3，LSCR 76.0 / 79.9 / 80.4。单点200局SR标准误约3.5个点，比较以LSCR、多点合并和逐球漏斗为主。数值已逐项对过产物（42组SR/LSCR全部一致，均200局、审计passed、binomial5、execute_steps 32、ckpt步数匹配）。
- PF/FK优势来自外观泛化：30k/35k/40k合并（每模型600局），cov_only SR raw 28.7 → nolocal 44.7 / local 40.7，同种子配对nolocal-only 34 vs raw-only 10（p≈0.0004），是唯一显著的扰动类；none持平，inv_cov略差（40.7 vs 44.7）。逐球看tennis提升最大（物理放置80.8→89.5%）。
- 剩余瓶颈是golf：失败约70%在golf，“触碰→举起”约75%，各模型与raw无差别；其次golf“接近未触碰”。
- 40k基础模型离线：MAE 0.0176，与20k相同；动作更平滑（lag-1自相关0.24→0.32，二阶/一阶1.19→1.12）；屏蔽video的dAct 0.0525，依赖不变。离线action误差不能预测闭环成功率。
- binomial5动作平滑影响大：同一20k nolocal，平均每局放置球数2.06→3.1。

## 0. 结构改动的定义（从头训练各组使用）

mask记法 `q<-k` 表示q模态的token不能attend到k模态的token（整对模态，含条件token；`k:noisy` 只切k的带噪token）。从头训的M1–M4不开indep。

| 组 | 改动 | 目的 |
| --- | --- | --- |
| indep | `independent_{action,fk,pointflow}_schedule=true`，`shift_action=5`；四模态各自采样噪声水平 | 原配置四模态共用同一sigma，action去噪进度被video绑定；独立后让action学会在video不可靠时少依赖video |
| M1 `geom_readonly` | mask `action<-pointflow, action<-fk, vision<-pointflow, vision<-fk` | PF/FK只读：几何可以读video/action，但video和action都读不到几何 |
| M2 `action_no_geom` | mask `action<-pointflow, action<-fk` | 只切action读几何（video仍可读） |
| M3 `physical_split` | mask切断 FK↔vision、PF↔action、PF↔FK | 按物理含义分组：FK只与action交互，PF只与video交互 |
| M4 `action_observed_vision` | mask `action<-vision:noisy` | action只看已观测的条件帧，看不到带噪的未来video |

mask实现在分支 `pointflow-fk-mask`（`3788ac6`），4090推理需mask运行时 `/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk-mask`（分支 `runtime-joint-attn-mask`，`10cfcdc`）。

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

guidance扫描（nolocal 40k，单点200局）：1 → 55.0，1.5 → 48.0，2 → 52.0，3 → 47.5；1最好，1.5与2之间的倒挂在单点噪声（约±3.5）内，不再细扫。

### PF/FK vs raw 同种子配对（30/35/40k合并，每类150局，McNemar）

| guidance | 对比 | 总体 | none | cov_only | inv_only | inv_cov |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | nolocal vs raw | 54.8 vs 52.3（89/74，p=0.27） | 69.3 vs 76.7（13/24，p=0.10） | **45.3 vs 34.7（26/10，p=0.011）** | 52.7 vs 51.3（p=0.88） | 52.0 vs 46.7（p=0.29） |
| 1 | local vs raw | 54.7 vs 52.3（95/81，p=0.33） | 72.0 vs 76.7（p=0.32） | **46.0 vs 34.7（30/13，p=0.014）** | 50.7 vs 51.3（p=1.0） | 50.0 vs 46.7（p=0.56） |
| 3 | nolocal vs raw | 48.2 vs 44.2（111/87，p=0.10） | 58.7 vs 59.3（p=1.0） | **44.7 vs 28.7（34/10，p=0.0004）** | 48.7 vs 44.0（p=0.37） | 40.7 vs 44.7（p=0.50） |
| 3 | local vs raw | 48.7 vs 44.2（116/89，p=0.07） | 64.7 vs 59.3（p=0.30） | **40.7 vs 28.7（35/17，p=0.018）** | 46.0 vs 44.0（p=0.78） | 43.3 vs 44.7（p=0.89） |

括号内为“PF/FK独赢 / raw独赢”局数。脚本与输出 `/oss-shichaojian/outputs/infer/task27/analysis/g1_paired_20261014/`。

结论：两种guidance下，PF/FK（nolocal、local）相对raw唯一显著的优势都在外观扰动cov_only（g1约+11个点，p≈0.01；g3约+12–16个点），总体差异都不显著。关CFG后raw在无扰动场景提升最多（59→77），PF/FK在none上转为略低（69–72 vs 77，p=0.10–0.32，不显著）。即PF/FK的价值目前可归结为外观泛化，其余场景与raw持平。


## 0.9 现有闭环结果综合分析（10-18更新，含10-17全部完成的run；g1，binomial5，每点200局同种子）

数据：4090上全部已完成g1 run（62个），按训练run把25/30/35/40k合并（每点200局，4点800局）；对照为同ckpt的raw/nolocal。p为Fisher，把同一训练run的不同ckpt当独立样本，**偏乐观**。另：除raw外所有PF/FK模型都带“PF/FK分支梯度未跨卡同步”问题。

task27（合并SR%；四通道 none/cov/inv/inv_cov）：

| 模型 | ckpt | SR | −raw | −nolocal | 四通道 |
| --- | --- | --- | --- | --- | --- |
| raw | 25–40k | 52.8 | — | −1.0 | 76/36/52/47 |
| nolocal | 25–40k | 53.8 | +1.0 | — | 70/43/52/50 |
| local | 25–40k | 55.6 | +2.9 | +1.9 | 72/48/51/52 |
| seed43（nolocal只换种子） | 25–40k | 57.1 | +4.4 | +3.4 | 65/42/63/58 |
| M3 physical_split | 25–40k | 62.2 | +9.5 | +8.5 | 74/52/62/61 |
| M4 action_observed_vision | 25–40k | 55.2 | +2.5 | +1.5 | 72/41/53/54 |
| M2 action_no_geom | 25–40k | 56.6 | +3.8 | +2.8 | 76/42/54/53 |
| M1 geom_readonly | 25–40k | 50.6 | −2.2 | −3.2 | 66/43/48/46 |
| indep | 25–40k | 51.0 | −1.7 | −2.7 | 72/40/50/42（55→52.5→49.5→47，随训练下降） |
| mix_f25 | 25–40k | 56.8 | +4.0 | +3.0 | 74/51/54/48（58→59→55.5→54.5） |
| mix_f25i15 | 25–40k | 59.4 | +6.6 | +5.6 | 76/48/58/54（57.5→56.5→57.5→66.0；40k重测64.0） |
| raw+mix_f25 | 25–40k | 52.9 | +0.1 | −0.9 | 72/39/51/50 |

task21：raw 65.1、nolocal 69.2、local 67.6（25–40k）；mix_f25 62.0（25–40k：63/67/54/64），对nolocal −7.2、对raw −3.1；mix_f25i15 64.5（25–40k：65.5/65/65.5/62），对nolocal −4.7、对raw −0.6；四通道 mix_f25 80/55/52/60、mix_f25i15 79/61/56/62、nolocal 81/60/66/69。

结论（只基于上表）：
1. **训练run之间的波动是主要误差来源**：同配置只换种子，4点合并仍差3.4 pp（53.8 vs 57.1），单点最多差6.5 pp；同一run的多个ckpt高度相关，合并不能消除这部分。现有每个变体只有1个训练run，判断标准应是“超出两个nolocal run（53.8、57.1）的区间多少”，而不是p值。
2. **PF/FK相对raw的总体收益很小**：task27 nolocal +1.0、local +2.9、seed43 +4.4，均在run间波动内；task21 nolocal +4.1（p≈0.09）。收益集中在遮挡通道cov_only：raw 36、nolocal 43、local 48、seed43 42、M3 52；无扰动通道none反而raw最高（76 vs 70）。
3. **M3是唯一明显超出run间波动的结构**：62.2，高于nolocal两个run的上沿（57.1）5.1 pp，四个ckpt均≥60；增益分布在cov/inv/inv_cov三个扰动通道，none不变。但只有1个训练run、1个任务。
4. **mask的其他形式不支持“切得越多越好”**：M1（视频与动作都不读几何）低于基线；M4（动作只看观测帧）与基线持平；M2（动作不读PF/FK，视频仍读）4个ckpt合并56.6，落在两个nolocal run（53.8–57.1）之间。M3与M2的差别是M3还切了FK↔视频与PF↔FK，同时保留FK↔动作。成对比较（均单run、单任务，差值需与run间波动3.4 pp比）：M1−M2 = −6.0（差别只在视频能否读PF/FK），提示视频读几何有用，切掉有害；M3−M2 = +5.6（M3恢复动作读FK、切视频↔FK与PF↔FK、PF只与视频交互），这几处改动在M3里同时发生，现有数据分不开是哪一处起作用；M4≈基线，动作看带噪未来视频不是问题来源。M3的增益集中在扰动通道（cov 52 vs nolocal 43/seed43 42），none不变。待验证：M3 task21（训练中）与M3 task27 种子43（训练中）。
5. **模式混合在task27有弱正信号，在task21为负**：task27 mix_f25i15合并59.4，高于nolocal两run上沿2.3 pp，主要来自40k（66.0，重测64.0可复现），25–35k为56.5–57.5，在波动内；mix_f25 56.8在波动内。task21两组都低于nolocal（−7.2、−4.7），与raw持平或更低。raw+mix 不涨，说明task27的增益若成立则与PF/FK有关，但跨任务不一致。
6. **无效的改动**：indep（随训练下降）、推理期补球点、P3重规划、guidance≠1、execute_steps缩短。多数模型25k到40k没有持续上升。

## 2. 40k基础模型的推理参数与执行步数

推理参数（不训练，nolocal 40k@40000，binomial5，200局，2026-10-14 03:00 UTC起）：`num_steps` 4→8（886，run-dir `eval_200_8gpu_b5_ns8`），`guidance` 3→1（899，`eval_200_8gpu_b5_g1`）。策略文件为同目录 `policy.{ns8,g1}.yaml`，仅改该字段与 `output_dir`。

- num_steps 4→8：SR 47.5→43.5，LSCR 80.4→80.0，不改善，不再继续。产物 `task27_cosmos_nolocal_s4_40000/runs/eval_200_8gpu_b5_ns8/`。
- guidance 3→1（即关闭CFG）：SR 47.5→55.0，LSCR 80.4→84.9，none/cov/inv/inv_cov 72/50/52/46。同种子配对 g1独赢36、原独赢21，McNemar p=0.063；后续多点确认见§0.5。

execute_steps（40k@40000，binomial5，200局，guidance 3）：缩短执行步数明显变差，不再继续。`policy.es{16,8}.yaml`，run-dir `eval_200_8gpu_b5_es{16,8}`（raw为 `eval_200_8gpu_shared_b5_es{16,8}`）。

| 模型 | es32（原） | es16 | es8 |
| --- | --- | --- | --- |
| nolocal | 47.5（40k单点；30/35/40k合并48.2） | SR 26.5 / LSCR 69.8 | SR 5.0 / LSCR 30.8 |
| raw | 45.0（40k单点；合并44.2） | SR 26.0 / LSCR 65.5 | SR 4.5 / LSCR 20.3 |

es16各类SR：nolocal none/cov/inv/inv_cov 36/24/24/22，raw 38/20/22/24。产物 `task27_cosmos_{nolocal,raw}_s4_40000/runs/eval_200_8gpu*_b5_es16/`。

40k基础模型离线消融：`/data/shichaojian/runs/xmodal_40k_20261013/`。

## 3. 从头训练的mask消融与一致性诊断（2026-10-14起）

结构改动一律从头训40k，与基础模型同配置、同步数比较：

- M1–M4从头训40k：命令与 `sim-task27-ptfk-nonlocal-40k`（2026-10-10）逐字相同，只加 `model.config.joint_attn_mask=<preset>`，从mask worktree（`pointflow-fk-mask` `3788ac6`）启动，不开indep，以单独隔离mask的作用。任务名 `scr40-mask-m{1,2,3,4}`，输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_mask_<preset>_8gpu_40k_20261014`，提交脚本 `/mnt/afs/experiments_20261014/scratch_mask/submit_scratch.py`。约2.5天；按基础40k同样在25k–40k各点上4090闭环，与基础模型同步数比较。
- 对照（2026-10-14 05:30 UTC起，同一提交方式，脚本 `submit_scratch_base.py`）：`scr40-seed43` 基础配置只改 `trainer.seed=43`，用于估计“同配置不同种子”的闭环波动，M1–M4与基础模型的差异须超过这个波动才算数；`scr40-indep` 从头训开独立sigma（`independent_{action,fk,pointflow}_schedule=true`，`shift_action=5`）。输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_{seed43,indep}_8gpu_40k_20261014`。
- 损失类从头训（2026-10-14 09:45 UTC提交，脚本 `submit_scratch_loss.py`，从 `-cons` worktree启动，不开indep，以便与基础模型和seed43直接比较）：`scr40-cons` 只加action–FK一致性损失（`POINTFK_CONSISTENCY_WEIGHT=3 MODE=task`），`scr40-smooth` 只加action平滑损失（`POINTFK_SMOOTH_WEIGHT=10`）；两者均需 `POINTFK_CONSISTENCY_STATS`。输出 `/data/shichaojian/runs/sim_task27_ptfk_nonlocal_{cons,smooth}_8gpu_40k_20261014`。提交时无空闲节点，排队中。
- 一致性诊断（不训练）：在40k nonlocal/local与20k nonlocal上，用约150–250个验证窗口量FK头 vs FK(预测action)、手附近PF vs FK、各自对GT的误差，并区分抓取阶段，判断各模态预测是否已经一致、一致性损失是否有空间。输出 `/data/shichaojian/runs/consistency_diag_20261014/`。代码与说明在分支 `pointflow-fk-cons`（worktree `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk-cons`）的 `tools/pointfk_consistency_diag/` 与 `docs/pointfk_consistency_diag_20261014.md`，未合并到本分支。
- 一致性诊断结果（2026-10-14，sanity通过）：各模态预测已相当一致，误差是“一起错”——FK头与预测action只差约14 mm，而各自对GT差26/31 mm；手附近PF–FK残差8 mm。抓取窗口误差高10–25%，但不一致只略增。即FK↔action、PF↔FK一致性损失可收紧的空间小，不针对主要误差；详见分支 `pointflow-fk-cons` 的 `docs/pointfk_consistency_diag_20261014.md`。
- 闭环一致性（2026-10-14，nolocal/local 30–40k，guidance 3与1，2000局、63118次规划）：FK头与FK(原始action)的相对差闭环18.9 mm（离线14.2），与阶段无关；成功局15.6、失败局19.1，按局均值预测失败AUC 0.82，失败局全程偏高而非结束前才升高；guidance 3比1高约20%（18.9 vs 15.7）。以实际轨迹为参照，FK头15.7 mm、FK(action) 10.2 mm，但实际轨迹本身由执行action产生，只说明FK头没跟上将要发生的运动。
- 同参照离线对比（同一批约150窗口）：对专家实际轨迹，FK头26.2、FK(预测action) 31.4 mm；对专家指令，FK头34.8、FK(预测action) 31.0 mm。两者各自更接近自己的训练目标，差距约等于专家指令与实际执行之间的偏差（约17 mm），并无优劣；第1帧FK头2.8 mm、第32帧两者都约36–38 mm，瓶颈在远期预测。
- 结论：模态间不一致主要来自FK头没跟上action；“把action拉向FK头”没有依据，“把FK头拉向action”只改善FK头，一致性损失提升成功率的潜力有限。不一致更适合作“模型困惑”信号（失败AUC 0.82），下一步是离线多采样测试：同一状态下矛盾最小的样本，action是否更准。详见分支 `pointflow-fk-cons` 的 `docs/pointfk_consistency_diag_20261014.md` §4“同参照对比”与§5，结果 `/data/shichaojian/runs/consistency_diag_20261014/same_ref/`、`/oss-shichaojian/outputs/infer/task27/analysis/consistency_closed_loop_20261014/`。
- 一致性损失尚未完整验证：现有 `task` 模式只是在FK空间算action损失，不涉及FK头；FK头↔action的跨模态版本（`pred_to_action(_rel)`）梯度只到FK头；PF↔FK只在cons_all中与其他项混用过；video↔PF未实现。诊断结果出来后再定从头训练的损失组合。

## 3.5 bad case跟进：抓取/放置精度的来源（2026-10-15）

输入：4090侧bad case分析（`/oss-shichaojian/outputs/infer/analysis/{golf_env,badcase_30k40k}/README.md`，含Q1–Q6跟进）与本日三项分析（台账对应三行）。

- 失败是厘米级精度问题：golf只差一球的失败中约89%是第一次抓取对不准（指尖-球心水平偏差 放进1.8 / 失败3.3 cm）；task21主要是瓶子放进后倒下。
- 偏差在模型自己的规划里，不在执行：闭环每次推理保存的FK头预测指尖-球偏差 放进2.5 / 失败3.6 cm，预测与执行只差0.6 cm；失败中“预测<2 cm而执行≥3 cm”为0%。
- 偏差是系统性的，不是采样噪声：同一窗口8个样本的预测落点均方根离散0.2–0.3 cm（无窗口>1 cm），用真值挑最好的一个只改善约0.2 cm，不用真值的挑选法与随机相同。闭环挑样本不值得做。
- 模型比示范差：示范golf偏差中位1.5 cm（p10/p90 1.1/2.4），50条中只有1条>3.3 cm；掌心高度随桌面的跟随比例 示范约0.90、闭环约0.57（同一估计方法；此前“约40%”为误读）。示范桌面同样在±5 cm随机，不是分布外。task21示范0/50倒下，瓶子与最近物体间距均≥8.9 cm（闭环<6 cm时倒下率64–89%）。
- 数据量：每个task只有50条独立示范（`_1`为同一轨迹的重渲染），都是人工遥操作、几乎没有重抓（golf 1/50），不能脚本生成恢复数据。
- 缩短执行步数：es8崩溃是闭环卡死（单帧观测、无历史、整局固定种子，只执行“先停后动”块的停顿前缀，51%的步不动）；es16下降集中在golf阶段，原因未定；代码层面无错位、无写死32。报告 `/oss-shichaojian/outputs/infer/task27/analysis/es_chunk_20261015/README.md`。
- 综合：动作忠实执行模型想象的规划，规划的空间精度（小物体对准、桌面高度、放置间距）比示范差且为系统偏差；结合FDM/IDM诊断（给真值视频后FK头误差降72%、动作FK误差降43%），瓶颈在“想象的场景几何”，改进应针对生成的视频/几何在手与目标物体附近的精度，而不是动作跟随、挑样本或推理节奏。

10-16补充（台账“PF/FK优势来源与球上点数”“手-物体相对误差”“task21放置的规划与点数”三行）：

- task27中PF/FK相对raw的提升集中在cov_only（g3按球是否在箱内：local/nolocal +12/+16 pp，p≤0.018），且在cov_only下golf与tennis同样提升（各约+12–13 pp）；none/inv_only/inv_cov不显著。此前“提升在tennis/乒乓球、不在golf”的说法不成立（来自失败结构表，未按扰动类别拆分），以此为准。PF/FK的作用是外观扰动下的鲁棒性，干净场景无提升。
- 球上点数：选点为512全局FPS+每手5 cm内至多256点，手远时golf恒为4–5点（离线16%窗口为0点），tennis约10、乒乓约20、soccer约60；手进5 cm后才上升。控制手距后golf点数与放进、预测偏差均无关（rho≈0），但因手远时点数几乎不变，相关分析无法检验“点数不足”的因果；只有改选点才能检验。
- 手-物体相对误差（离线，接触窗口）：手2.2 cm、球2.1 cm、相对1.2 cm，手与球误差同向（相关0.75），相对位置比各自绝对位置准；相对位置损失空间小。
- task21：倒下主要来自规划的松手高度过高与瓶身倾斜，不是太挤也不是执行偏差（规划与执行手位差约0.7 cm；规划高度≥6 cm覆盖soy倒下71%、vinegar 45%，未倒仅5%；规划间距<6 cm只占倒下的7%/17%）。此前“间距<6 cm倒下率64–89%”量的是倒下前0.5 s的距离，部分是倒下的结果。箱内物体点数与倒下无关。PF/FK在task21的提升（nolocal +41局，p=0.004）来自抓取/搬运类失败减少（主要inv_only），倒下比例不变（raw 18.7%、nolocal 17.5%）。报告 `/oss-shichaojian/outputs/infer/task21/analysis/placement_plan_20261016/README.md`、`/oss-shichaojian/outputs/infer/task27/analysis/pf_points_20261016/README.md`。

关键数据（详表见各README）：

| 项 | 数据 | 来源 |
| --- | --- | --- |
| golf对准偏差（执行） | 示范中位1.5 cm（p10/p90 1.1/2.4，50条中1条>3.3）；闭环放进1.8 / 失败3.3 | `/data/shichaojian/runs/demo_precision_20261015/summary.md`、`golf_env/README.md` |
| golf对准偏差（规划，FK头） | 放进2.5 / 失败3.6；规划vs执行0.6 cm；规划<2 cm时放进78%，≥4 cm时27% | `golf_env/README.md` Q4 |
| 掌心随桌面跟随比例 | 示范0.90、闭环0.57（同一回归口径） | `demo_precision_20261015/summary.md` |
| 8样本落点离散 | 均方根0.2–0.3 cm；真值挑最优比随机好约0.2 cm；免真值挑选=随机 | `/data/shichaojian/runs/multisample_spread_20261015/summary.md` |
| 球上点数（手远，中位） | golf 4–5（16%为0）、tennis 10、乒乓20、soccer 60；手<5 cm后golf约33–40 | `hand_obj_relative_20261016/summary.md`、`task27/analysis/pf_points_20261016/README.md` |
| 接触窗口误差 | 手2.2、球2.1、相对1.2 cm，手-球误差相关0.75 | `hand_obj_relative_20261016/summary.md` |
| PF/FK vs raw（cov_only，g3按球在箱内） | 成功 +12/+16 pp；golf +12/+13，tennis +13/+13（local/nolocal） | `pf_points_20261016/partA_tables.md` |
| task21倒下 | 规划松手高度 倒下soy 8.3 / vinegar 5.4 cm vs 未倒3.7 / 4.1；规划间距<6 cm仅占倒下7%/17% | `task21/analysis/placement_plan_20261016/README.md` |
| es8卡死 | 51%步不动；es32的“先停后动”块第10–13行才动 | `task27/analysis/es_chunk_20261015/README.md` |

更正记录（以本节为准）：①“桌面跟随约40%”为误读，同口径为57%；②“PF/FK提升在tennis/乒乓、不在golf”不成立，cov_only下golf同样提升；③task21“间距<6 cm倒下64–89%”量的是倒下前0.5 s，部分是倒下的结果。④推理时PF锚点只来自可见的机器人+物体（含箱子）mask，桌面不在mask中，锚点里没有桌面点；桌高只能通过球/箱子等物体点的z间接得到（4090 `pf_points_20261016/README.md`）。

判断：两个任务的瓶颈都是规划的空间精度（小物体对准、桌面高度、松手高度），raw/local/nolocal在此相同，现有PF/FK结构不改善它；PF/FK的已知作用是外观扰动鲁棒。已排除/降级：闭环挑样本（离散太小）、两阶段视频→IDM与一致性挑选（规划本身偏）、手-物体相对损失（相对误差小于绝对误差）、缩短执行步数（es8卡死、es16原因未定）。“给目标物体保底点数”既未被支持也未被否定（手远时golf点数几乎不变，相关分析无法检验）。

桌高跟随 raw vs PF/FK（10-16，台账“桌高跟随”行；4090 `task27/analysis/table_follow_20261016/README.md`）：golf首次抓取掌心离桌高度的跟随比例（1+斜率，30/35/40k合并，场景bootstrap 95% CI）g3：raw 0.66 [0.57,0.75]、local 0.54 [0.43,0.65]、nolocal 0.59 [0.47,0.70]；g1：三者均0.62–0.63。示范0.90。PF/FK不比raw跟随得更好（g3下local略差，CI刚排除0；g1无差别）；最低指尖高度口径相反（PF/FK 0.91 vs raw 0.83–0.85），两种口径不一致。tennis同样无PF/FK优势。三模型都在低桌面档最差（golf放进 g3：raw 21.4%、local 31.8%、nolocal 27.4%，中/高档55–63%），PF/FK在低桌面档的放进优势不能用手高跟随解释（未检验显著性）。离线验证窗口仅3种桌高、混杂严重，无法判断规划是否欠跟随。结论：物体锚点里的三维高度信息没有让手高更好地跟随桌面，桌高问题不是“raw缺深度、PF/FK有深度”的差别。

箱子/球锚点与桌高（10-16，4090 `task27/analysis/box_table_20261016/README.md`）：每次推理1024点中约324点在箱子上（首帧：箱底169、箱壁148含箱沿49），筛选前箱子占候选点50%；mask只含机器人与物体，无桌面。世界坐标下箱底点z−桌面=20.0 mm（std 0.23 mm），对桌高斜率1.000、R²=1.000；之后箱子直立时R² 0.997–0.999，球被放入遮挡后依然成立（箱子倾斜≥5°的23.7%行除外）；静止球点同样精确（R²≥0.996）。但模型看到的是相机坐标：相机高度每个场景随机（std 17.5 mm，范围64 mm，与桌高无关），箱底点绝对深度对桌高只有R² 0.68；需与机器人点做相对比较才能读出（首帧箱底深度−右掌锚点深度：R² 0.9987），而手一动这个参照就变。结论：桌高信息在输入里且充足，无需加桌面点；但它被相机位姿随机化混在相机坐标里，要靠模型自己做“物体点−机器人点”的相对推断才能得到，而动作是关节空间（机器人坐标）。未验证模型是否用到这一线索。相机随机化的来源：每局场景采样 `scene_generalization_sample.spatial.camera`（profile `train_default_camera`，与 `table_height`、`object_pose` 同属spatial），与四组扰动类别无关（none/cov_only/inv_only/inv_cov的cam_z分布相同，std 17–19 mm）；训练示范同样有（`rgbd.hdf5` cam_overhead外参，100条z std 18.5 mm、范围66 mm，`cameras.yaml`只记标称值），不是分布外。

推理端选点干预ballq32（10-16，台账同名行）：推理时给每个球保底32点（golf由4–5点增至32点），nolocal 30/35/40k g1合并600局：成功329→319（p=0.43），golf放进332→323（p=0.49），其余球不变。不改训练时，给golf更多点不改善抓取；训练/推理选点分布不一致使它不能否定“训练时加点”，但已无支持该方向的正证据。

task21 guidance 1（10-16，台账同名行）：三模型g1都显著好于g3（600局合并：raw +7.5 pp p=0.0006，local +7.8 pp p=0.0003，nolocal +4.5 pp p=0.050），与task27一致，g1应作为task21默认。g1下PF/FK相对raw的优势缩小但仍在（local +4.7 pp p=0.043，nolocal +3.8 pp p=0.090；g3下nolocal +6.8 pp p=0.004），因为raw从g1获益更多。之后PF/FK与raw的比较一律在g1下报告。

## 4. 待办

- 40k结果出来后按逐球漏斗与配对比较筛选，重点看golf“触碰→举起”和cov_only。
- golf抓取：待可视化确认失败形态（够不着、手指未合拢、抓后滑落）后设计针对性改动。
- 原生WAM离线eval无现成脚本，未做。
- 待决定（2026-10-16提出）：①推理端选点干预——nolocal 40k推理时从全局点中划32–64点给待抓球，200局与原结果配对（需改4090代码副本，存在训练/推理选点分布不一致；golf变好=强正证据，无变化=不能下结论），排在task21 g1之后；②通用方法——手/目标区域加大视频损失权重（对raw同样有效，不区分PF/FK），等模式混合与从头训结果再定。
- 待决定（2026-10-16提出）：③锚点改用机器人基座/世界坐标（PF/FK特有）：现锚点在相机坐标，相机高度随场景随机（std 17.5 mm），桌高与物体高度要靠模型做相对推断；改到机器人坐标后箱底/球点z直接等于桌高+常数，并与关节空间动作同一参照。需重新预处理PF/FK缓存并从头训（约8节点×2天），推理端同步改坐标变换（仿真外参已知）。④4090会话提出的P3：接近球时（预测指尖距预测球点4–5 cm、块内≥8步）提前重查一次，每球一次；建议先nolocal 40k g1 none类50局配对小测。P1（训练时小球保底点数）、P2（加桌面点）：P2已不需要（箱子点充足），P1依据弱（ballq32推理端600局无效）。③已确认并在实施（robotframe行）。
- task21示范松手高度需用新松手定义（手离瓶轴>R+2.5 cm持续3帧）在开发机重算后再与闭环比较；旧示范数5.6 cm口径不同。
- 从头训各组20k闭环（seed43/indep/M1–M4 checkpoint已存）：用户说到时通知。
- 代码合并：mask、cons分支及4090运行时worktree待结果确定后再合并到主分支。
- （10-19 提案，待用户批准）M3 三刀的 2³ 析因消融：因子为 视频↔FK、PF↔动作、PF↔FK 三对是否双向切断，视频↔动作、视频↔PF、FK↔动作始终保留。8 格中 nolocal（全开）与 M3（全切）已有，新训 6 格 × task27/task21 = 12 个从头 40k（种子42、不开梯度同步，与 nolocal/M3 一致），每 5k 自动闭环 g1。主效应各由 4 vs 4 格估计。用户问到的"链式"（只切视频↔FK、PF↔动作）是其中一格；切视频↔动作的纯链式不在本设计内。
- （10-19 用户要求）task27 FDM+IDM 种子42 的 20k/25k/30k 原样重测（同配置、同场景种子），run `eval_200_8gpu_b5_g1_rep2`；20k/25k 在907、30k 在904；用于检验种子42相对种子43的差距是否来自评测随机性。**完成**：重测 20k/25k/30k = 58.0/61.0/58.5（首测 62.0/57.5/56.5；各类 38/23/30/25、36/33/28/25、34/26/34/23）。同ckpt两次评测差2–4 pp；两次均比同种子 nolocal（50.5/50.5/52.5）高 6–10 pp；种子43 FDM+IDM（54.5/55.5）与同种子 nolocal（54.5/57.0）持平——差距来自训练种子而非评测随机性。
- （10-19 用户要求）raw + vw25 从头训40k（task27 `scr40-raw-vw25`、task21 `scr40-raw-vw25-t21`）：raw（wam，无PF/FK）只改 `loss_scale` 10→2.5，其余同 raw 基线；用于判断 vw25 的增益是否依赖几何信息。`/mnt/afs/experiments_20261017/videoweight/submit_videoweight.py` preset `rawvw25`/`rawvw25t21`，输出 `sim_task{27,21}_wam_vw25_8gpu_40k_20261019`，不限节点。排队中；闭环（20k/25k/30k）的 raw 推理目录待 20k 时由 raw_s4 派生。注意 task21 的 nolocal vw25 尚未训练。
- （10-19 10:50 UTC）25k 新结果：M3 种子43 task27 53.5（LSCR 84.5；38/19/27/23），20k 为 61.0；同步数 nolocal 种子43 57.0、nolocal 种子42 50.5、raw 54.0，M3 种子42 60.0。M3 task21 25k 68.0（LSCR 89.7；43/25/36/32），同步数 nolocal 70.0、raw 65.0。M3 的复现在 25k 转弱，需 30k 再判断。
- （10-19 用户要求）M3 种子43 task27 20k/25k 原样重测、M3 种子42 task27 20k–40k 原样重测（mask 运行时，run `eval_200_8gpu_b5_g1_rep2`），区分评测随机性与训练种子差异。结果待补。
- （10-19 用户要求，必须同步）M3 + PF/FK 梯度同步，从头训（40k 学习率周期，30k 手动停）：point-fk 主分支合并 mask 分支（`3e992be`，测试 35 passed）后同时具备 `joint_attn_mask` 与 `sync_plain_param_grads`；任务 `scr40-m3-sync`、`-s43`、`-t21`、`-t21-s43`（`submit_m3mix.py` preset `m3sync27/m3sync27s43/m3sync21/m3sync21s43`，排除 108.81/108.193），输出 `sim_task{27,21}_ptfk_nonlocal_mask_physical_split_sync[_seed43]_8gpu_40k_20261019`；自动闭环 20k/25k/30k（A800 `up_auto_m3sync.sh`，4090 `drive_auto.sh <task> scr40_m3_sync[_s43] mask`）。排队中。注意：此前所有 M3/M3放开PF↔FK/FDM+IDM 训练均未同步。
- （10-19）撤销 4 个 M3+同步任务（旧手动同步写法，且 \`sync_plain_param_grads\` 误放在 RectifiedFlowTrainingConfig，\`model.config.\` 覆盖会启动即失败），并停掉其上传/评测流。方案 A（FSDP 原生管理 PF/FK 分支）已实现：训练分支 \`pointfk-fsdp-native\` \`4ea316e\`（worktree \`-fsdp\`），推理 \`Bench2Dex-Pointflow-fk-fsdp\` \`a7993ce\`；8 卡 smoke：0 个未受管参数，5.58 vs 5.53 s/it，存档/续训/Sonata 预训练恢复/单卡加载均通过；待 4090 真机推理验证后合入主分支。
- （10-19 待用户定）规划：此前所有 PF/FK 结果均为分支未同步训练；重建对比需同步版 nolocal 基线 + M3 + vw25（两任务、关键项两种子），训到30k、测20/25/30k。
- （10-19 用户定）同步版重训清单（先不换种子）：nolocal、M3、vw25、M3放开PF↔FK，task27/task21 各一，共 8 个；M3 与用户 motivation（PF↔FK 作为桥）冲突，故 M3放开PF↔FK 同等优先验证。raw、raw+vw25 不受影响不重训。待 4090 推理验证与合入主分支后提交。
- （10-19 13:40 UTC）同步版重训已提交（point-fk 主分支 \`1f9c737\` 合入 \`pointfk-fsdp-native\`：PF/FK 分支在 FSDP 前安装，FSDP 原生切分/同步；8 卡 smoke 5.58 vs 5.53 s/it；4090 新运行时 \`Bench2Dex-Pointflow-fk-fsdp\` \`a7993ce\` 加载 M3 s42 25k 与旧 mask 运行时 937 个张量逐一相同；新格式 ckpt 推理 8 局 0 失败）。种子42、40k 周期 30k 手动停、训练内评测每 2000 步（用户要求）、排除 108.81/108.193。集群排队顺序：\`scr40-sync-m3\`、\`-m3-t21\`、\`-m3-pffk-t21\`、\`-vw25\`、\`-vw25-t21\`，之后 \`scr40-raw-vw25\`、\`-t21\`（raw 不受同步问题影响，作 vw25 对照，排在后面）；task27 M3放开PF↔FK 在开发机（\`/mnt/afs/experiments_20261019/launch_sync_m3pffk_t27_devbox.sh\`）。输出 \`sim_task{27,21}_ptfk_nonlocal_{mask_physical_split,mask_m3_pffk,vw25}_sync_8gpu_40k_20261019\`。自动闭环 20/25/30k：A800 \`up_auto_sync.sh\`，4090 \`drive_auto.sh <task> scr40_sync_{m3,m3_pffk,vw25} {mask,mask,main}\`；raw+vw25 的推理目录待 20k 前单独准备。未同步的 \`scr40-mask-m3-pffk-t21\` 用户同意停止，但停止命令被权限检查拦截，仍在运行。
- （10-21 结果，g1 200 局，SR / LSCR）同步版：task27 M3 20k 49.0（重测 54.0）、25k 56.0 / 82.4；M3放开PF↔FK 20k 61.5 / 85.9、25k 64.0 / 87.4；vw25 20k 60.0 / 66.5、25k 66.5 / 70.9（LSCR 异常低：失败局几乎一个球都没放进去）。task21 M3 20k 68.0、25k 71.5；M3放开PF↔FK 20k 71.5、25k 69.0（25k 在新目录 \`eval_200_8gpu_g1_r2\` 重测：原目录被 908 停掉的进程预先写过文件，OSS 不能覆盖，导致无限重跑，原目录作废）。对照：task27 raw 51.5/54.0，nolocal 50.5/50.5；task21 raw 64.5/65.0，nolocal 65.5/70.0。判断：未同步 M3 种子42 的约 +10 同步后不复现；M3放开PF↔FK 比同步版 M3 高 8–12（PF↔FK 桥有用），比 raw 高约 10。评测机改为 4090 922（task27）/ 908（task21，原 907/904 被回收）；\`prep_auto.sh\` 新增 \`fsdp\` 选项，同步版一律用 \`Bench2Dex-Pointflow-fk-fsdp\` 推理。
- （10-21 用户定、已提交）同步版 nolocal 基线与同步版 M3放开PF↔FK 种子43，两任务各一：\`scr40-sync-nolocal\`、\`-t21\`、\`scr40-sync-m3-pffk-s43\`、\`-t21\`（\`submit_m3mix.py\` preset \`syncnolocal{27,21}\` / \`syncm3pffks43{27,21}\`，主分支 \`1f9c737\` 之后，40k 周期 30k 手动停，训练内评测每 2000 步，排除 108.81/108.193），输出 \`sim_task{27,21}_ptfk_nonlocal_{sync,mask_m3_pffk_sync_seed43}_8gpu_40k_20261021\`；自动闭环 20/25/30k：A800 \`up_auto_sync1021.sh\`，4090 \`drive_auto.sh <task> scr40_sync_{nolocal,m3_pffk_s43} fsdp\`。结果待补。raw+vw25 推理：4090 \`prep_raw_auto.sh\` / \`drive_raw_auto.sh\`（以 raw_s4 25k 为模板，只改 ckpt 与 loss_scale；用 raw_s4 ckpt 冒烟、dry-run 通过；两任务 action_stats 与 raw_s4 md5 相同），A800 \`up_auto_rawvw25.sh\`。下一批（代码待准备）：M3放开PF↔FK + objq32 组合（需把 ballquota 仿真选点合入 fsdp 推理 worktree）。
- （10-19）按保留规则清理 /data：删 205 个中间 iter 目录（约 6.0T）与 22 个 5000 倍数 iter 的 optim（约 238G），每个 run 的 latest（含 optim）保留；其他会话的 a1_/r4_/robotgeometry 在训 run（21 个目录）未动。/data 86%→74%（可用 7.2T→14T）。清单与脚本 `/mnt/afs/experiments_20261019/cleanup_1019/`。
