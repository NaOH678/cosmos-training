# PointFlow 实验队列(2026-09-30 建立,持续更新)

状态图例:**进行中** / **等待中** / **已完成** / **已停止**(结论已出,提前终止)

通用约定:

- 所有 run 用 `examples/launch_pointflow_sandwich101.sh`,新实验一律**新 OUTPUT_ROOT**;
- 标量 scale 当前有效值 **0.0528**(101 集分层 500+守卫重扫),启动命令里显式带上;
- 改目标参数化(标量↔向量 scale、token mode、点数)→ 不能 resume,必须从头训;
- 判据统一看 eval **ADE/zero**(gt-cond 为主nt 为辅),loss 读数跨参数化不可比;
- 后台数据准备(CPU)与 GPU 训练可并行。

## 总表

| # | 实验 | 假设/目的 | 状态 | 判据 |
| ---- | ---- | ---- | ---- | ---- |
| A | per_point-500(v2,标量 scale) | 现行最好形态,对照曲线 | **已停止**(9200 iter,平台化后轻微回退) | —(对照) |
| B | cluster-origin(stage1, 500 点) | 簇 token 原始 decode 基线 | **已完成**(10000 iter,平台高于 A) | — |
| E1 | per-frame scale × per_point-500 | 前段帧学不出来是归一化问题 | **已完成:证伪**(停 @6312;cond 打平 v2、val_08 略差、joint 破位不稳回退) | ~~t≤2 比值下压~~ 假设不成立:前段帧非瓶颈 |
| E2s | 方案 C 单机8卡 smoke | PointDecoder 新路径从未上卡 | **已完成**(停 @6800,烟雾通过;相对16卡 E2为半 batch,作为 E2-stage0 的同拓扑对照) | 几十 iter 无 NaN/OOM,loss 正常降 |
| E2 | cluster + skip(1,2) + 点级 block×4 | 在接近 v2 生成质量的前提下降低开销,作为后续主线底座 | **决定停止**(最后核查 @5852,完整eval @5800;待确认作业退出) | @5500视觉质量接近v2,训练步耗时约减少55%;保留5500 checkpoint |
| E2-stage0 | stage0 cluster + skip(1,2) + 点级 block×4 | 更细簇能否以小幅计算增量缩小 per-point 差距 | **决定收尾**(完整eval/checkpoint @2000;待确认作业退出) | @2000视觉抽查与E2s接近,未见稳定收益;保留2000 checkpoint,后续优先stage1融合对照 |
| E2-fusion | stage1 + 池化前几何—运动融合 + skip(1,2) + pb4 | 保留点的几何与运动配对能否改善生成质量 | **已完成**(实验结论归档;视觉@9700,作业核查@9911) | 与E2s质量接近、步耗时相同,未证明稳定增益;主线暂保留原stage1+skip/pb |
| E3a | N=4000 数据准备(缓存+scale) | 密集版前置,CPU 后台 | **等待中** | 缓存 101 集齐备 + 标量/向量 scale 产出 |
| E3b | 密集版 N=4000 训练 | 一点一 token 放不下时的形态验证 | **等待中**(依赖 E3a + E1/E2 结果定形态) | 等 iter 对比 A;token 预算可控 |
| E4 | per-frame scale × cluster(方案 C) | 两个修复叠加 | **已取消**(E1 证伪,framescale 不进任何主配方) | — |
| E5a | dropper 数据准备(缓存+scale) | dropper 迁移前置,CPU 后台 | **等待中** | 缓存 101 集齐备 + strat500 scale 产出 |
| E5b | dropper per-point-500 训练 | sandwich 结论在 dropper 复现 | **等待中**(依赖 E5a) | 与 A 同口径对比 cond ADE/漂移 |
| E5c | dropper cluster 形态(可选) | E2 结论跨数据集验证 | **等待中**(依赖 sandwich E2) | 与 E5b 对比 |

## A — per_point-500 基线(v2)【已停止 @9200】

```
OUTPUT_ROOT=/data/shichaojian/runs/pointflow_sandwich101_strat500_v2_20260929 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 16 卡 HSDP(shard=8 replicate=2),max_samples_per_batch=16,~11.1 s/iter。
- 产物:
  - OUTPUT_ROOT `/data/shichaojian/runs/pointflow_sandwich101_strat500_v2_20260929`
  - eval `…/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_0000100~0009200/`
  - 日志 `…/logs/action_policy_singlerighthand_edge_sft_node{0,1}.log`
- 轨迹:5000 iter 进入平台(val_00/02/03 cond ADE ≈ 8.6/8.0/7.0),5400~7500 在
  7.4~7.9 之间横盘,7600 起轻微回退(8.2/7.8/7.7 @9200)→ 用户手动停。
- 结论:标量 scale 的 per-point 形态在 101 集上 ~5000 iter 即饱和,继续训只有过拟合
  风险;最终 val cond ADE 7.7~8.2mm(ratio ~0.30),joint 14.9/48.0/16.8mm,
  静态漂移 2.3~8.2mm。**val_02_joint(48mm)显著差于 zero 基线(26mm),是全 run
  的固定短板 case,后续实验重点观察对象。**

## B — cluster-origin baseline【已完成 @10000】

- 配置:同 A 的 scale 0.0528、同 strat500 缓存,仅 token mode=cluster(原始池化 decode)。
- 产物:OUTPUT_ROOT `/data/shichaojian/runs/cosmos/cluster_500p_origin_baseline`,
  eval `…/pointflow_eval/step_0000500~0010000/`,日志 `…/logs/action_policy_singlerighthand_edge_sft.log`;
  5000 iter 后减速,9000~10000 平台(cond 9.1/8.5/7.7)。
- 结论(对比 A):
  - **等 iter 全面落后**:@5000 cond 12.1/8.6/6.9 vs A 8.6/8.0/7.0;
  - **平台更高**:终值 9.1/8.5/7.7 vs A 平台期 7.4~7.9 → 池化丢信息的上限更低;
  - joint 终值 14.5/38.6/18.2 与 A 同量级(val_02_joint 同样是短板);
  - 簇内多点共用一个 token 的 decode 瓶颈实锤 → E2(skip + 点级 block)要治的就是这个。

## E1 — per-frame scale × per_point-500【已完成:证伪,停 @6312(2026-10-01)】

验证归一化诊断的最便宜实验:数据/缓存全复用,只改目标参数化。

```
OUTPUT_ROOT=/data/shichaojian/runs/perpoint_strat500_framescale_20260930 POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930_frame_scales_env.json NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 向量来源:`pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930_frame_scales_env.json`
  (96 值 std,逐帧×逐通道,101 集 × 256 窗口;收敛序列验证 w256/w64 偏差 ≤1%,
  w8/w64 旧版已弃用);
- resume 必须带同一 env;原理见 scale 文档 §14。
- 产物:OUTPUT_ROOT `/data/shichaojian/runs/perpoint_strat500_framescale_20260930`,
  eval `…/pointflow_eval/step_0000100~/`,日志 `…/logs/action_policy_singlerighthand_edge_sft_node{0,1}.log`。
- 中期读数(@5100):cond 7.6/8.1/8.2,4400 步进平台,与 v2 平台(7.4~7.9)打平
  (val_00 略好、val_03 略差)——**cond 口径无明显收益**;
  **但 val_02_joint 短板被治好:4100~4300 步打到 13.3mm(首次优于 zero 26.3,
  v2 全程 48mm)**,4400 后振荡回 ~33 不稳;val_00/03_joint 与 v2 同量级。
  framescale 的价值体现在 joint 口径,继续观察是否回落稳定。
- **终局判读(@6000~6312,2026-10-01 停)**:
  - cond 与 v2 完全打平,后段设计目标落空(val_08 cond 35.4 vs v2 32.2,反而略差);
  - val_02_joint 破位是 @4100 前后的甜区(最佳 16.6),之后缓慢回退到 36,
    不是收敛方向,**不能算 framescale 的稳定收益**;
  - 可视化逐帧对比(val_02_joint / val_08 cond)与 v2 无可见差别,
    大运动 case 两者同为"欠采样式跟随";
  - **结论:framescale 证伪。** 它解决的是"早晚帧 loss 权重不均"的假设,
    但前段帧本来就不是瓶颈,真正短板(后段大运动)是信息/表达问题,
    不是 loss 权重问题。对照 PointWorld:per-timestep 归一化在他们那也是
    数值卫生,精度来自全点架构+稠密监督(见观测文档 §5.8);
  - **代码路径保留**(env 门控、默认关),仅作 joint 不稳定机制排查的消融开关,
    主配方禁用;framescale 向量的派生实验(E4、E5b-fs、E3b 叠加项)全部取消;
  - 保留产物:`iter_000004000` checkpoint(joint 甜区)及全 500 间隔 ckpt。

## E2s — 方案 C 单机8卡 smoke【已完成,停 @6800】

PointDecoder(skip + 点级 block)代码路径首次上卡,本地 8 卡运行。

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_skip_pb4_smoke_20260930 POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=1 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 结果:4.9 s/iter,35 GB/卡,无 NaN/OOM;cond ADE 200→600 步 95→75→46→**28.9mm**
  (ratio 0.53),等 iter 追平 per-point 形态;漂移 81→10.2mm。**烟雾通过。**
- 后续:未及时停,跑到 6800 步(cond 10.7/7.5/7.8,joint 16.0/28.6/14.0);
  因 8 卡半 batch 口径被污染、且 16 卡 E2 同配置已起,用户确认停掉。
  **附带发现:cluster 系 val_00 cond 明显偏差(卡 ~10.7 vs per-point 平台 7.4~7.6),
  val_02/03 正常——E2 到 5000 步后若 val_00 仍压不下去,需单查。**
- 产物:OUTPUT_ROOT `/data/shichaojian/runs/cluster_500p_skip_pb4_smoke_20260930`,
  eval `…/pointflow_eval/step_0000100~0006800/`。

## E2 — cluster + skip + 点级 block 全量【决定停止,待确认作业退出】

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_skip_pb4_20260930 POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 配置:16卡 HSDP(shard=8 replicate=2),stage1、strat500、scale 0.0528、skip(1,2)、pb4。
  对照 B 与 v2(同 scale、同缓存);方案细节见 cluster decode 文档。
- 产物:OUTPUT_ROOT `/data/shichaojian/runs/cluster_500p_skip_pb4_20260930`,
  eval `…/pointflow_eval/step_0000100~/`,日志 `…/logs/action_policy_singlerighthand_edge_sft_node{0,1}.log`。
- **停止决定(2026-10-01)**:用户同意不再为小幅指标变化继续占用16卡,腾卡验证
  几何—运动融合并推进FK主线。最后核查日志为08:58 UTC、5852步,完整eval到5800;
  本次仅更新实验记录,未执行停止命令,最终退出状态/步数待确认。
- **判据修订**:目标是生成质量接近v2且成本显著降低,不要求逐项指标超越v2。
  以joint dream的手物交互、形态保持、预测点与生成表面的对应为主,ADE/漂移辅助诊断。
  旧的“2.5/3指标未达标 ⇒ decode修补救不了池化”仅作历史判断保留,不足以证明结构根因,
  也不应据此否定该形态作为主线底座。
- **视觉对比(@5500,同为16卡)**:抽查全部12个val窗口的预测点叠加帧,并复核6个重点
  窗口的无标注dream帧;E2与v2操作类型和画面质量接近,没有看到整体质量下降一个档次。
  val_04/08原始画面接近;val_10中段手部位置变化较大;val_11拿起后左移的时序不同;
  局部点偏离表面的现象两者都有。结论限于固定窗口抽帧,不代表已验证连续视频的全部
  瞬时伪影、多seed稳健性或生成action的正确性。
- 对比视频:`/data/shichaojian/e2_vs_v2_step5500_joint_dream/val_XX_points.mp4`
  (12个case,**左E2/右v2,仅带预测点面板**,每边都是各自模型生成的视频与点);
  `comparison_info.json`记录源路径、模型和布局。
- 效率:5200~5490步日志中的iter_speed中位数 **5.00 vs v2 11.08 s/iter**,
  同16卡训练步约快2.2倍、耗时减少55%;不含完整eval墙钟成本,不是推理延迟测量。

下表为val case等权平均(mm):ADE覆盖12个case,drift覆盖有静态点的8个case。
E2/v2 @5500已核对episode、seed、有效点数与zero基线一致。

| run / step | cond ADE | joint ADE | cond drift | joint drift |
| ---- | ---- | ---- | ---- | ---- |
| E2 @4000 | 14.29 | 22.76 | 5.42 | 5.27 |
| E2 @5500 | 15.38 | 22.50 | 6.65 | 5.76 |
| v2 @5500 | 13.23 | 24.40 | 4.34 | 4.47 |
| E2 @5800 | 15.45 | 22.58 | 7.20 | 6.23 |

- 趋势:4000→5800的joint均值基本相当,cond及静态漂移反而上升;4900~5300的joint
  曾恶化至29~31mm,5400后恢复到22~23mm。没有持续收益的证据,不必凑满10000步。
- **保留基线checkpoint**:
  `/data/shichaojian/runs/cluster_500p_skip_pb4_20260930/cosmos3_action/action_sft/action_policy_singlerighthand_edge/checkpoints/iter_000005500`
  (最后核查latest_checkpoint.txt指向此目录),与已保存的视觉对比对应。
- 后续:以E2为候选主线底座;融合对照先用8卡stage1+skip/pb,与8卡E2s比较,
  默认关闭开关、全新OUTPUT_ROOT。实现与命令见
  [几何—运动融合文档 §8](./pointflow_geometry_motion_encoding_20261001.md#8-实现与启用)。

## E2-stage0 — 更细簇 + skip + 点级 block【决定收尾,待确认作业退出】

检验 stage1 是否压缩过于激进:适当增加主干 point token,能否以较小成本缩小与
per-point 的差距。历史 24 窗口扫描中,500 点在 stage1 约 30 簇(4cm)、stage0
约 60 簇(2cm),对应主干 point token 约 270→540,仍远少于 per-point 的 4500。
这是假设依据,不是本 run 已测得的簇数;见 [cluster decode 文档 §4](./pointflow_cluster_decode_20260929.md#4-关键事实认知修正别再用旧印象)。

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_stage0_skip_pb4_20261001 POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=0 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 EXTRA_TAIL_OVERRIDES="trainer.grad_accum_iter=1" NPROC_PER_NODE=8 NNODES=1 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 配置:8 卡 FSDP(shard=8 replicate=1),grad_accum_iter=1,同 strat500 缓存、
  scale 0.0528、skip(1,2)、pb4;改变 stage 不改变选点及逐点监督,沿用同一 scale。
- **主要对照 E2s**(同8卡、无累积,已跑到6800),16卡 E2/v2/E1 作为辅助参照,
  不把不同有效 batch 的等 iter 比较当作严格对照。保留 skip/pb 是为了比较 stage;
  level1/2 skip 在 stage0 下提供更深层几何上下文,不是更细分辨率。
- 边界:stage1→0 同时改变簇粒度、Sonata 特征层级与维度(64→32),不是纯 K 消融;
  即使改善,也不能直接证明差距全部来自簇内运动混合。cap=128 只是 packing 预算,
  不控制实际簇数。代码依据:`pointflow_geometry.py:111`、`pointflow_batch.py:56`
  (分别位于 `cosmos_framework/model/generator/`、`cosmos_framework/data/`)。
- 产物:OUTPUT_ROOT `/data/shichaojian/runs/cluster_500p_stage0_skip_pb4_20261001`,
  eval `…/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_*/`,
  日志 `…/logs/action_policy_singlerighthand_edge_sft_node0.log`。
- **早期快照(2026-10-01 06:16 UTC)**:日志到386步,最新完整 eval 为300步;
  与 E2s 的 episode/seed/有效点数/zero 基线一致。下表 ADE 为12个 val case 等权平均,
  drift 为有静态点的8个 val case 等权平均,单位 mm;不与旧段落的单 case 读数混用。

| run @300 | cond ADE | joint ADE | cond drift | joint drift |
| ---- | ---- | ---- | ---- | ---- |
| E2-stage0(8卡) | **58.22** | **62.43** | **46.76** | **44.55** |
| E2s stage1(8卡) | 72.22 | 77.76 | 61.72 | 59.27 |
| E2 stage1(16卡) | 70.22 | 81.13 | 60.31 | 57.86 |
| v2 per-point(16卡) | 26.10 | 43.45 | 11.86 | 9.26 |
| E1 per-point framescale(16卡) | 25.30 | 36.77 | 13.03 | 8.81 |

- 相对 E2s:cond/joint ADE 分别降低 **19.4%/19.7%**,两口径均12/12 case 改善;
  cond/joint drift 分别降低24.2%/24.8%。200~370步 iter_speed 中位数
  **5.35 vs 4.95 s/iter(+8%)**,不含完整训练+eval的墙钟成本。
- 轨迹:stage0 @100/200/300 的 cond/joint 为104.65/108.56、93.98/100.57、
  58.22/62.43;E2s 同期为104.63/108.16、95.52/95.33、72.22/77.76。
  **目前只支持更早进入快速收敛阶段,不能判定平台更好**。
- 上述为早期历史快照,收尾判断以下面的2000步评估为准。原始数据可按上述 run 的
  `pointflow_eval/step_0000300/val_XX[_joint]/metrics.json` 复核。

### 2000步收尾记录(2026-10-01)

- **决定**:完成本轮stage0对照,保留2000步checkpoint;后续优先stage1的几何—运动
  池化前融合实验,与同8卡、无累积的E2s比较。本次归档视频并更新文档,未执行停止
  命令,作业退出状态及最终训练步数待确认,不能将评估步2000记作实际停止步数。
- **视觉结论**:抽查val_00/02/05/08/09/11各第6/14/23/32帧,stage0与E2s的动作过程
  和点随手移动表现总体接近;05/08没有明确提升,11主要是动作时序差异。未见更细簇
  带来稳定、肉眼明确的质量收益。此结论限于关键帧抽查,不等同于连续播放的抖动评估。
- **边界**:stage0仍在改善,不据此宣布收敛或否定最终上限;收尾依据是当前质量收益
  不足以支持继续占卡,优先推进主线。也不能用V2单个指标高低代替生成质量判断。

下表与早期快照同口径:ADE为12个val case等权平均,drift为有静态点的8个case等权
平均,单位mm。2000步三组的episode、seed、有效点数与zero基线已核对一致;
V2为16卡,仅作辅助参照。

| run / step | cond ADE | joint ADE | cond drift | joint drift |
| ---- | ---- | ---- | ---- | ---- |
| E2-stage0 @1900 | 17.18 | 27.00 | 6.83 | 6.78 |
| E2-stage0 @2000 | 16.88 | 26.03 | 6.76 | 5.97 |
| E2s stage1 @2000 | 17.14 | 25.51 | 6.36 | 7.23 |
| v2 per-point @2000 | 15.37 | 27.62 | 6.14 | 6.73 |

- 纯预测点对比视频归档:
  `/data/shichaojian/stage0_vs_e2s_vs_v2_step2000_joint_dream/`
  内含`val_05_points.mp4`、`val_08_points.mp4`、`val_11_points.mp4`、
  `comparison_info.json`及`metrics.csv`(含1900/2000步明细)。
  **左stage0 / 中E2s / 右V2**,各自生成的joint dream叠加各自预测点;
  保留完整预测面板,不附加无点的原始视频行。
- **保留checkpoint**:
  `/data/shichaojian/runs/cluster_500p_stage0_skip_pb4_20261001/cosmos3_action/action_sft/action_policy_singlerighthand_edge/checkpoints/iter_000002000`
  (收尾核查时`latest_checkpoint.txt`指向此目录)。
- 下一轮实现与启动命令见
  [几何—运动融合文档 §8](./pointflow_geometry_motion_encoding_20261001.md#8-实现与启用),
  使用全新OUTPUT_ROOT,其余配置与E2s保持一致。

## E2-fusion — 池化前几何—运动融合【已完成】

检验先将逐点带噪运动、level0几何特征及相对XYZ融合,再做簇内池化,是否优于
原来的运动编码后池化。实现与推导见
[几何—运动融合文档 §8](./pointflow_geometry_motion_encoding_20261001.md#8-实现与启用)。

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_stage1_geomotion_skip_pb4_20261001 POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_GEOMETRY_MOTION_FUSION=true POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 POINTFLOW_DISPLACEMENT_FRAME_SCALES='' POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE='' EXTRA_TAIL_OVERRIDES="trainer.grad_accum_iter=1" NPROC_PER_NODE=8 NNODES=1 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 配置:8卡、无梯度累积、stage1、cap128、skip(1,2)、pb4、strat500、标量scale
  0.0528;从原基座开始,不接E2/stage0 checkpoint。日志确认`geometry_motion_fusion=True`。
- **状态(2026-10-02)**:按用户要求标记实验**已完成**,结论和产物已归档。
  归档视频对应融合9700步;更新文档时日志到9911步(04:32 UTC),训练上限10000,
  latest checkpoint为9500。此状态表示实验收尾,不声称作业已经退出或10000步已经完成;
  本次未执行停止命令。
- **视觉结论**:最新对照抽查val_00/02/05/08/09/11的第6/19/32帧,另复查6800
  同步数的05/08/11。融合与E2s的动作过程、点随手移动表现总体接近;局部时序有差异,
  未见融合稳定改善点与物体贴合。关键帧抽查不等同于连续播放的抖动评估。
- **效率**:6000~6490步iter_speed中位数,融合与E2s均为**4.96 s/iter**(同8卡),
  未观察到训练步耗时增加;不代表已测量完整墙钟成本、显存或推理延迟。
- **结论**:融合可正常训练,但未证明稳定质量增益。主线暂保留原stage1+skip/pb,
  融合保留为消融开关;不延长本轮,后续优先推进FK主线。此实验不足以支持
  “池化前缺少几何配对就是主要瓶颈”,也不据此否定所有几何—运动融合方法。

下表单位mm:ADE为12个val case等权平均,drift为有静态点的8个case等权平均。
6800同步数及最新各步数组合均已核对episode、seed、有效点数与zero基线一致。
严格结构对照以融合/E2s同8卡同6800步为主;最新步数不同,V2另为16卡。

| run / step | cond ADE | joint ADE | cond drift | joint drift |
| ---- | ---- | ---- | ---- | ---- |
| E2-fusion @6800 | 15.03 | 22.82 | 6.17 | 6.23 |
| E2s @6800 | 14.55 | 22.88 | 5.38 | 4.91 |
| E2-fusion @8000 | 14.17 | 22.55 | 5.54 | 5.50 |
| E2-fusion @9700 | 14.16 | 22.25 | 5.79 | 5.84 |
| v2 @9200 | 12.76 | 24.28 | 5.23 | 4.05 |

- 近期趋势:8000→9700的joint ADE仅22.55→22.25,cond基本不变,静态漂移没有同步
  改善;不能把多训练后的微小收益归因于融合结构。
- OUTPUT_ROOT:`/data/shichaojian/runs/cluster_500p_stage1_geomotion_skip_pb4_20261001`。
  eval位于`cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/`;
  核查时保留checkpoint为同级`checkpoints/iter_000009500`,9700是eval步而非checkpoint步。
- **纯预测点视频归档**:
  `/data/shichaojian/fusion_vs_e2s_vs_v2_step9700_joint_dream/`
  包含`val_05_points.mp4`、`val_08_points.mp4`、`val_11_points.mp4`、`metrics.csv`
  及`comparison_info.json`(来源、步数、布局、视频SHA256)。
  **左融合@9700 / 中E2s@6800 / 右V2@9200**,各自dream叠加各自预测点,
  完整预测面板,无额外无点视频行。

## E3a — N=4000 数据准备【等待中,CPU 后台】

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/build_pointflow_window_cache.py --manifest pointflow_outputs/sandwich_924_20260928/manifest.json --output /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 --episode-allowlist examples/pointflow_sandwich924_all_101_episodes.txt --top-n 4000 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --workers 16
```

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/scan_pointflow_selection.py --manifest pointflow_outputs/sandwich_924_20260928/manifest.json --cache-root /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache --episode-allowlist examples/pointflow_sandwich924_all_101_episodes.txt --top-n 4000 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --windows-per-episode 256 --workers 64 --window-cache-root /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 --output pointflow_outputs/scale_scan_924_strat4000_20260930.json
```

- 4000 点的配额比例(2:0.40,3:0.45,4:0.15)是否要随密度调整,启动前过一眼幻影扫描。

## E3b — 密集版 N=4000 训练【等待中,依赖 E3a + E2/E2-stage0】

形态由 E2/E2-stage0 结果决定(cluster 是必然;E1 已证伪,**不带 per-frame scale**):

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_n4000_202610XX POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=256 POINTFLOW_SELECT_TOP_N=4000 POINTFLOW_WINDOW_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 POINTFLOW_DISPLACEMENT_SCALE=<E3a扫描值> NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

## E4 — per-frame scale × cluster【已取消】

~~E1 若成立,把向量叠加到 E2 的配置上。~~ E1 已证伪(2026-10-01),framescale
不进任何主配方,本实验取消;framescale 代码路径仅保留作消融开关。

## E5 — dropper 数据集迁移系列【等待中】

sandwich 上的形态结论(per-point vs cluster、scale 参数化)需要在 dropper 上复现验证。
dropper 侧已有资产:

- raw `/data/shichaojian/raw_data/singlerighthand_dropper_100`
- VAE latent 缓存 `/data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache/vae_window_latents/`(101 集已齐)
- manifest `pointflow_outputs/dropper_924_20260928/manifest.json`
- allowlist `examples/pointflow_dropper_all_101_episodes.txt`(101 集)
- 启动脚本 `examples/launch_pointflow_dropper101.sh`(**默认路径还指着旧集群 gpfs 和旧
  manifest `dropper_101_20260923`,跑之前必须用 env 覆盖:**
  `SINGLERIGHTHAND_RAW_ROOT` / `SINGLERIGHTHAND_CACHE_ROOT` / `POINTFLOW_MANIFEST`)
- 注意:dropper 的 `arm_action_space="joint"`(脚本从缓存 manifest 读,toml 不用改);
  幻影漂移画像与 sandwich 相反——23.4% 选中点、集中在手部轨迹(L2 30.2%),见
  `pointflow_outputs/dropper_101_20260923/phantom_drift_scan.json`,配额可能要调。

### E5a — dropper 数据准备【等待中,CPU 后台】

缺:strat500 窗口缓存 + scale 扫描(sandwich 配方:top-n 500、mvm3、mvs16、
quotas 2:0.40,3:0.45,4:0.15、phantom-guard;旧 scale 0.0482 是 labeled29 时代
top-300 扫的,不可沿用):

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/build_pointflow_window_cache.py --manifest pointflow_outputs/dropper_924_20260928/manifest.json --output /data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache/pointflow_windows --episode-allowlist examples/pointflow_dropper_all_101_episodes.txt --top-n 500 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --workers 16
```

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/scan_pointflow_selection.py --manifest pointflow_outputs/dropper_924_20260928/manifest.json --cache-root /data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache --episode-allowlist examples/pointflow_dropper_all_101_episodes.txt --top-n 500 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --windows-per-episode 256 --workers 64 --window-cache-root /data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache/pointflow_windows --output pointflow_outputs/scale_scan_dropper_924_strat500_20260930.json
```

- 启动前先过一眼新 phantom 扫描结果,确认 quotas 是否按 dropper 画像调整。

### E5b — dropper per-point-500 训练【等待中,依赖 E5a】

```
OUTPUT_ROOT=/data/shichaojian/runs/pointflow_dropper101_strat500_202610XX SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/singlerighthand_dropper_100 SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache POINTFLOW_MANIFEST=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/dropper_924_20260928/manifest.json POINTFLOW_WINDOW_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache/pointflow_windows POINTFLOW_DISPLACEMENT_SCALE=<E5a扫描值> NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_dropper101.sh
```

- 判据:与 sandwich A 同口径对比(cond ADE/ratio、漂移、val_02_joint 类短板 case)。
- ~~若 sandwich E1 成立,可并行加一档 framescale 变体(E5b-fs)。~~ E1 已证伪,不加。

### E5c — dropper cluster 形态(可选)【等待中,依赖 sandwich E2 结论】

sandwich E2 若证明 skip+点级 block 能救 cluster,再在 dropper 上复现;否则跳过。

## 共享数据产物(sandwich,所有 A/B/E1/E2/E2-stage0 实验共用)

- manifest `pointflow_outputs/sandwich_924_20260928/manifest.json`
- allowlist `examples/pointflow_sandwich924_all_101_episodes.txt`
- VAE latent 缓存 `/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/vae_window_latents/`(101 集)
- strat500 窗口缓存 `/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows/`
  (top-n 500、quotas 2:0.40,3:0.45,4:0.15、mvm3、mvs16、phantom-guard,见其中
  `cache_manifest.json` / `build_report.json`)
- 标量 scale 0.0528 来源 `pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930.json`
- per-frame 向量 `pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930_frame_scales_env.json`(E1 用)

## 变更记录

| 日期 | 变更 |
| ---- | ---- |
| 2026-09-30 | 建立队列;A/B 出结论(B 停止);per-frame scale 与方案 C 代码就绪(未 commit) |
| 2026-09-30 晚 | A(v2)停 @9200(平台化+轻微回退,结论写入);B 跑满 10000 补齐终值,结论修订为"等 iter 落后 + 平台更高";E1 进行中(@2000 等 iter 领先 A);E2s 烟雾通过(600 iter),E2 可启动;新增 E5 dropper 迁移系列(E5a 数据准备 / E5b per-point 训练 / E5c cluster 可选);A/B/E1/E2s 产物路径与 sandwich 共享数据产物路径落档 |
| 2026-10-01 | E1 @5100:cond 平台与 v2 打平,但 val_02_joint 短板治好(13.3mm 首优 zero,后振荡回 ~33);E2 全量已起 @2500,等 iter 追平 E2s;E2s 停 @6800(8卡半 batch 口径污染,被 E2 取代),发现 cluster 系 val_00 cond 偏差现象待查;**全 case 对比(cond/joint/漂移)+ 架构事实 + 未解问题落档 `docs/pointflow_form_comparison_observations_20261001.md`** |
| 2026-10-01 06:16 UTC 快照 | 新增 E2-stage0(8卡无累积),记录更细簇假设、命令、产物及@300同iter对比:相对 E2s cond/joint ADE改善19.4%/19.7%,训练步耗时+8%;尚未判断平台。更正 E2s 的“单卡”称谓为“单机8卡”,明确其作为 stage0 同拓扑对照的用途。 |
| 2026-10-01 07:31 UTC | **E1 停 @6312,终局证伪**:cond 与 v2 打平、val_08 反略差、val_02_joint 破位为 @4100 甜区后回退、可视化与 v2 无别;framescale 不进主配方,代码路径仅留作消融开关,E4/E5b-fs/E3b 叠加项全部取消;`iter_000004000` 甜区 ckpt 保留。E1 复盘教训:事前应先摆 PointWorld 对照证据(归一化=数值卫生)并写证伪条件,可省机时 |
| 2026-10-01 07:50 UTC | **E2 @5000 三条验收判读:(a) val_00 cond 9.93 未达 B 的 9.1(介于 B 与 E2s 之间);(b) train_00 漂移 13.7 钉在 cluster 基线(v2 5.2),val_09_joint 漂移 3.15 有改善但未达 v2 的 1.85;(c) val_02_joint 37.8 未复现 E2s 破位(28.6,E2s 本身也在 13.8~33 振荡)。2.5/3 不成立 ⇒ "decode 修补救不了池化"基本坐实,重心转向 token 形成阶段(E2b stage0 / E3 密集)。E2 亮点:val_09_joint ade 9.61 全形态最优, val_04_joint 保持 cluster 形态优势** |
| 2026-10-01 08:58 UTC 快照后决定 | **E2决定停止,作业退出待确认**(日志5852步,完整eval5800)。按生成质量/成本重新定性:5500步视觉质量接近v2、同16卡训练步耗时5.00 vs 11.08s,足以作为候选主线底座;4000~5800无持续收益,保留iter_000005500并腾卡做融合/FK。补齐纯预测点并排视频目录与指标口径,修订此前仅凭指标判定池化根因的过强结论。 |
| 2026-10-01 2000步评估后 | **E2-stage0决定收尾,作业退出待确认**。归档stage0/E2s/V2的2000步纯预测点视频、指标与来源说明;视觉抽查未见stage0稳定收益,保留iter_000002000,后续优先stage1池化前几何—运动融合。未执行停止命令。 |
| 2026-10-01 09:50 UTC | **两个在跑作业先后停止**:E2(16卡)日志止于 @5939(09:07),E2b(本地8卡)止于 @2116(09:32),均无 traceback(正常 kill 特征);E2b 与本文档"2000步收尾"决定一致,E2 停止原因待用户确认(若是平台侧动作则忽略)。E2b 逐 case 同 step 判读补充进观测文档 §5.9(均值表掩盖了"运动侧胜/漂移侧负"的结构);checkpoint 保留:E2 `iter_000005500`、E2b `iter_000002000`。下一步:几何—运动联合编码实验(`pointflow_geometry_motion_encoding_20261001.md` §8) |
| 2026-10-02 | **E2-fusion已完成**(按用户要求归档;视觉@9700,作业核查@9911,未执行停止)。同8卡6800步及最新joint dream抽查未见融合稳定质量收益,步耗时与E2s同为4.96s;归档三组对比MP4、指标和来源说明,主线暂保留原stage1+skip/pb,后续优先FK。 |
