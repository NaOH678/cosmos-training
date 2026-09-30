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
| A | per_point-500(基线) | 现行最好形态,对照曲线 | **进行中**(~5900 iter) | —(对照) |
| B | cluster-origin(stage1, 500 点) | 簇 token 原始 decode 基线 | **已停止**(plateau,结论:全败于 A,病灶在泛化) | — |
| E1 | per-frame scale × per_point-500 | 前段帧学不出来是归一化问题 | **等待中** | t≤2 比值从 ~0.6 明显下压;等 iter val ADE 低于 A |
| E2s | 方案 C 单卡 smoke | PointDecoder 新路径从未上卡 | **等待中** | 几十 iter 无 NaN/OOM,loss 正常降 |
| E2 | cluster + skip(1,2) + 点级 block×4 | 治 cluster 泛化病灶,为密集点铺路 | **等待中**(依赖 E2s) | 等 iter val ADE 低于 B;静态漂移改善 |
| E3a | N=4000 数据准备(缓存+scale) | 密集版前置,CPU 后台 | **等待中** | 缓存 101 集齐备 + 标量/向量 scale 产出 |
| E3b | 密集版 N=4000 训练 | 一点一 token 放不下时的形态验证 | **等待中**(依赖 E3a + E1/E2 结果定形态) | 等 iter 对比 A;token 预算可控 |
| E4 | per-frame scale × cluster(方案 C) | 两个修复叠加(可选,看 E1 结果) | **等待中** | 同 E2,且前段帧比值下压 |

## A — per_point-500 基线【进行中】

```
OUTPUT_ROOT=/data/shichaojian/runs/pointflow_sandwich101_strat500_v2_20260929 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 记录:A/B 对比见主线文档 9-30 节;E1 出前 500~1000 步 eval 后可停。

## B — cluster-origin baseline【已停止】

- OUTPUT_ROOT `/data/shichaojian/runs/cosmos/cluster_500p_origin_baseline`,9500 iter plateau;
- 结论:等 iter / 等 wall-clock 双口径全败于 A;train ADE 相当、泛化差、静态漂移重 →
  池化丢信息,靠记忆训练集。这正是 E2 要治的。

## E1 — per-frame scale × per_point-500【等待中】

验证归一化诊断的最便宜实验:数据/缓存全复用,只改目标参数化。

```
OUTPUT_ROOT=/data/shichaojian/runs/perpoint_strat500_framescale_20260930 POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930_frame_scales_env.json NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 向量来源:`pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930_frame_scales_env.json`
  (96 值 std,逐帧×逐通道,101 集 × 256 窗口;收敛序列验证 w256/w64 偏差 ≤1%,
  w8/w64 旧版已弃用);
- resume 必须带同一 env;原理见 scale 文档 §14。

## E2s — 方案 C 单卡 smoke【等待中】

PointDecoder(skip + 点级 block)代码路径首次上卡,先几十 iter。

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_skip_pb4_smoke POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=1 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 看:显存峰值、loss/grad 是否有限、iter 速度;没问题再进 E2。

## E2 — cluster + skip + 点级 block 全量【等待中,依赖 E2s】

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_skip_pb4_20260930 POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_DISPLACEMENT_SCALE=0.0528 NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- 对照 B(同 scale 0.0528、同缓存);方案细节见 cluster decode 文档。

## E3a — N=4000 数据准备【等待中,CPU 后台】

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/build_pointflow_window_cache.py --manifest pointflow_outputs/sandwich_924_20260928/manifest.json --output /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 --episode-allowlist examples/pointflow_sandwich924_all_101_episodes.txt --top-n 4000 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --workers 16
```

```
PYTHONPATH=$PWD /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/scan_pointflow_selection.py --manifest pointflow_outputs/sandwich_924_20260928/manifest.json --cache-root /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache --episode-allowlist examples/pointflow_sandwich924_all_101_episodes.txt --top-n 4000 --min-voxel-members 3 --min-valid-steps 16 --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard --windows-per-episode 256 --workers 64 --window-cache-root /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 --output pointflow_outputs/scale_scan_924_strat4000_20260930.json
```

- 4000 点的配额比例(2:0.40,3:0.45,4:0.15)是否要随密度调整,启动前过一眼幻影扫描。

## E3b — 密集版 N=4000 训练【等待中,依赖 E3a + E1/E2】

形态由 E1/E2 结果决定(cluster 是必然,是否带 per-frame scale 看 E1):

```
OUTPUT_ROOT=/data/shichaojian/runs/cluster_n4000_202610XX POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=256 POINTFLOW_SELECT_TOP_N=4000 POINTFLOW_WINDOW_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows_n4000 POINTFLOW_DISPLACEMENT_SCALE=<E3a扫描值> NNODES=2 bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

## E4 — per-frame scale × cluster(可选)【等待中】

E1 若成立,把向量叠加到 E2 的配置上;向量需按 cluster 选点重扫(E3a 的 scan 会直接产出)。

## 变更记录

| 日期 | 变更 |
| ---- | ---- |
| 2026-09-30 | 建立队列;A/B 出结论(B 停止);per-frame scale 与方案 C 代码就绪(未 commit) |
