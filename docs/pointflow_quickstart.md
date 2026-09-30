# PointFlow 开箱使用手册(新集群)

> 面向刚拿到这套代码的机器/人:从 clone 到跑起 pointflow 训练的最短路径。
> 本文档只做索引和关键操作,细节全部链接到现有文档。
>
> 分支:`WorldAct-cosmos3-edge-droid-sft-pointflow`(基于 `cosmos3-edge-droid-sft`,
> 包含此前全部训练代码 + pointflow 全部工作)。

## 0. 这是什么

在 cosmos3 edge 的 action policy SFT 里接入 pointflow 模态:逐点 token
(per_point,300 点),与 video/action 在主循环里联合去噪。数据管线和实验
记录见[文档地图](#文档地图)。

## 1. 拿代码 + 装环境

```bash
git clone https://github.com/liuhangxu-robin/WorldAct.git
cd WorldAct && git checkout WorldAct-cosmos3-edge-droid-sft-pointflow
```

- 集群公网慢/不通 → 按 [docs/setup_offline.md](./setup_offline.md):有网机
  `uv sync` 填满 cache → 拷 cache → 离线机 `uv sync --offline --locked`。
- 有网 → 按 [docs/setup.md](./setup.md):`uv sync --all-extras --group=cu130-train`。
- 驱动 CUDA 13.2 兼容 cu130 wheel(驱动 ≥ 构建版本即可)。
- **每次开 shell 必做**:`source .venv/bin/activate && export LD_LIBRARY_PATH=''`。

**pointflow 追加依赖**(sonata 编码器需要,不在 base 环境/lock 里):

```bash
uv pip install --python .venv/bin/python --default-index https://pypi.org/simple \
  --only-binary :all: 'addict==2.4.0' 'spconv-cu126==2.3.8'
uv pip install --python .venv/bin/python --no-deps --no-index \
  --find-links "https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html" \
  --only-binary :all: torch-scatter
```

- spconv 没有 cu130 构建,`spconv-cu126` 实测可在 torch 2.10.0+cu130 /
  A800(sm80)上正常跑 GPU kernel(2026-09-28 验证);torch-scatter 按当前
  torch 版本选 PyG 官方 wheel(本集群为 `2.1.2+pt210cu130`)。
- 缺这三个包的典型症状:训练启动即 `ModuleNotFoundError: No module named
  'addict'`(`pointflow_geometry.py` → `auxiliary/sonata`)。

## 2. 数据摆放

以下都不在 git 里,需从旧集群拷贝(大小为参考值):

| 内容 | 旧集群路径 | 谁引用它 |
| ---- | ---------- | -------- |
| 模型包(VAE/tokenizer/transformer) | `models/cosmos3-edge-droid/`(11G) | `EDGE_DROID_MODEL_PATH` |
| 训练基座 DCP | `models/cosmos3-edge-droid-dcp/`(6.3G) | `BASE_CHECKPOINT_PATH` |
| sonata 权重 | `checkpoints/ptv3/sonata_small.pth`(148M) | `POINTFLOW_SONATA_CHECKPOINT` |
| 原始数据 | `raw_data/singlerighthand_{sandwich,dropper}_100/` | `SINGLERIGHTHAND_RAW_ROOT` |
| 缓存(manifest+video+window latents) | `datasets/singlerighthand-*-cosmos-cache/`(每套百 G 级) | `SINGLERIGHTHAND_CACHE_ROOT` |
| pointflow 轨迹(9.24 转换产物) | `pf_out/9.24/<dataset>/labeled/` | manifest 内的绝对路径 |

**路径不一致时的两处修改**:

1. 启动脚本顶部 env(`examples/launch_sft_action_policy_singlerighthand_edge.sh:10-20`)
   全部支持同名环境变量覆盖,不用改脚本;
2. pointflow manifest 里嵌了轨迹的绝对路径——挂载点变了就 sed 换前缀,或按
   [数据 pipeline 步骤 2](./pointflow_data_pipeline.md) 重建。

### 本集群实例(2026-09-28 起,inspur / A800 sm80)

- 仓库:`/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow`(pointflow 分支 checkout);
- 共享 venv:`/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv`(在 base checkout 下,
  其 editable 安装指向 base 代码——**手动跑脚本必须 `cd` 到本仓库并用
  `python -m cosmos_framework.scripts.xxx`**,否则 import 到 base 的旧代码;
  训练启动脚本自身会 `cd` + `-m`,不受影响);
- 数据根:`/data/shichaojian/`,布局与上表一致(`models/`、`checkpoints/ptv3/`、
  `raw_data/`、`datasets/`、`pf_out/9.24/`、`runs/`);
- 9.24 labeled 已转换:sandwich/dropper 各 101 集,`video` 字段已补;
- manifest 已按新路径重建:`pointflow_outputs/sandwich_924_20260928/manifest.json`、
  `pointflow_outputs/dropper_924_20260928/manifest.json`(含本机绝对路径,不入 git)。

本集群启动训练前的 env 覆盖(复制即用):

```bash
export LD_LIBRARY_PATH=''
export PYTHON_BIN=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export POINTFLOW_SONATA_CHECKPOINT=/data/shichaojian/checkpoints/ptv3/sonata_small.pth
# sandwich:
export SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/singlerighthand_sandwich_100
export SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
export POINTFLOW_MANIFEST=$PWD/pointflow_outputs/sandwich_924_20260928/manifest.json
export OUTPUT_ROOT=/data/shichaojian/runs/<每次必换>
bash examples/launch_pointflow_labeled29_sandwich.sh
# dropper 换:raw_data/singlerighthand_dropper_100、
#   datasets/singlerighthand-dropper-100-cosmos-cache、
#   pointflow_outputs/dropper_924_20260928/manifest.json,
#   启动脚本换 launch_pointflow_dropper101.sh
```

## 3. 启动训练

```bash
# 本集群(A800/SenseCore)全量 101 集:一切配置已固化,只给 OUTPUT_ROOT
OUTPUT_ROOT=/data/shichaojian/runs/<新目录,每次必换> bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

`launch_pointflow_sandwich101.sh` 固化了两层默认,均可从外部同名 env 覆盖:

- **集群布局**:PYTHON_BIN(共享 venv)、模型/DCP/sonata 权重、sandwich raw+cache、
  101 集 manifest 与 allowlist;
- **训练配置**:per_point 分层语义选点(N=500,手/物/台面=40/45/15,组内 GT 运动
  排名+缺额回补)、幻影守卫(幽灵点占比 7.5%→0.7%)、`POINTFLOW_EVAL_JOINT=true`
  (联合去噪 dream eval)、`COSMOS_FLASH2_VARLEN=1`、expandable_segments、
  batch 16/卡、val-on-start、scale 0.0528(101 集分层重扫)、窗口缓存、
  HSDP(节点内分片、跨节点复制,分片/复制度数自动等于 SenseCore 注入的每节点
  卡数/节点数)。

通用/旧集群的裸启动方式(全 env 手写)见第 2 节"本集群实例";其它wrapper:
`launch_pointflow_labeled29_sandwich.sh`(29 集)、`launch_pointflow_dropper101.sh`(dropper)。

- 脚本开头的 `Checking inputs...` 会核对 TOML、数据集、checkpoint、缓存,缺什么会直接报;
- `OUTPUT_ROOT` **每次显式换**(wrapper 未设置会直接报错);resume = 同一 `OUTPUT_ROOT` 重跑同一命令;
- WANDB 默认 offline;更多开关见脚本内注释和
  [docs/pointflow_data_pipeline.md 步骤 5](./pointflow_data_pipeline.md)。

### 关键 env / override 速查(训练常用)

| 开关 | 默认 | 作用 |
| ---- | ---- | ---- |
| `POINTFLOW_EVAL_JOINT=true` | false | eval 加跑部署形态联合 rollout:首帧+state action 干净,未来 video/action/point 联合去噪,解码梦视频做画布,产出 `<case>_joint/`(含 dream 拼接)。费显存费时长,第一轮 eval 盯 OOM |
| `POINTFLOW_SIGMA_SCAN=true` | false | eval 诊断:固定 sigma 网格上扫 x0_hat 的 ADE |
| `trainer.run_validation_on_start=true`(EXTRA_TAIL_OVERRIDES) | false | 开局先跑一轮 val(fresh 在 iter 0,resume 在断点 iter);默认等 `trainer.validation_iter`(=100) |
| `model.config.parallelism.data_parallel_shard_degree=8` + `..._replicate_degree=1`(EXTRA_TAIL_OVERRIDES) | 1/-1(纯 DDP) | FSDP 全分片,80GB 卡必须;语义同 DP,全局 batch 不变 |
| `COSMOS_FLASH2_VARLEN=1` | 关 | 解除 flash2 varlen 禁令(sm80 上 varlen 唯一替代是 natten,慢 ~4.8x/kernel、~2.2x/step);数值验证 `tools/check_flash2_varlen.py` |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | 关 | 碎片回收;另:DCP save 前已内置 `empty_cache()`(修偶发 NCCL "unhandled cuda error") |
| `NNODES=2 NODE_RANK=0/1 MASTER_ADDR=<node0>` | 单机 | 多机训练;16 卡配 `shard=8 replicate=2`(HSDP:分片留节点内 NVLink,节点间只梯度 allreduce);shard×replicate×CP 必须等于 WORLD_SIZE。细节见 [pointflow_multinode_training.md](./pointflow_multinode_training.md) |
| `POINTFLOW_SELECT_REGION_QUOTAS` | 空(扁平排名) | 分层语义配额,如 `2:0.40,3:0.45,4:0.15`(手/物/台面);需配合 `SELECT_TOP_N>0`,组内仍按 GT 运动排名,缺额全局回补 |
| `POINTFLOW_SELECT_PHANTOM_GUARD=true` | false | 幻影点(uv 冻结但 3D 漂移)降级到排名末尾;101 集实测选中点幻影占比 7.5%→0.7% |
| `POINTFLOW_WINDOW_CACHE_ROOT` | 空(在线算) | 预计算窗口缓存目录(数据管线步骤 3b);缓存后每样本读 ~120KB 而非 ~205MB |

### resume 语义(查证结论)

- **自动 resume**:同一 `OUTPUT_ROOT` 重跑同一命令 → 从最新 DCP 续训,模型/优化器/调度器/dataloader 状态全恢复;dataloader 有 per-rank pickle 状态,**续训不重复取数**;
- **skip 只有 EMA**:`keys_to_skip_loading=["net_ema."]`(experiment config:146),EMA 从主网络热启动。**pointflow 权重 resume 时完整恢复**;首次从 base DCP 加载时 pointflow 键不存在于 base,走 partial load 保持随机初始化(日志里的 "Skipping loading of key: net_ema.pointflow_branch..." 是 EMA 跳过,属正常);
- **eval case 固定**:`fixed_cases.json` 钉在 OUTPUT_ROOT 下,resume 时校验——改 case 数量/窗口数会报 "Existing fixed_cases.json has different case counts",此时必须换新 OUTPUT_ROOT;
- **改 FSDP 拓扑**(shard 度数)后 resume 旧 checkpoint 由 DCP resharding 支持,但同一 run 内别改。

## 4. 看训练效果

- eval 产物:`<OUTPUT_ROOT>/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_*/`,
  含逐 case mp4、`validation_stages.html`(浏览器打开看网格);
- 缺拼接视频时离线补:`tools/stitch_pointflow_canvas.py --eval-dir <上述目录>`;
- 指标判据看 eval ADE/zero,跨 scale 的 loss 读数不可比(见数据 pipeline 运维规则);
- 可视化规则与投影约定:[docs/pointflow_eval_visualization.md](./pointflow_eval_visualization.md)。

## 5. 接入新数据 / 换数据集

完整操作手册:[docs/pointflow_data_pipeline.md](./pointflow_data_pipeline.md)
(cache 骨架 → allowlist → **9.24 efep 转换(步骤 1b)** → manifest → window
latents → scale 标定 + 幻影扫描 + 可视化验证 → 训练)。

## 6. 当前状态(2026-09-28,新集群已贯通)

- sandwich/dropper 的 cosmos 缓存(manifest/video/window latents)**已就绪**;
- 9.24 labeled 转换**已完成**(sandwich/dropper 各 101 集,`video` 字段已补),
  manifest 已按新路径重建(见第 2 节"本集群实例"),旧集群的 manifest 仅
  供查历史扫描结论;
- 冒烟已过:1×A800 3 步训练 loss 正常下降、点云数据路 8/8 样本各 300 点、
  checkpoint 正常保存;sonata 依赖(addict/spconv-cu126/torch-scatter)已在
  共享 venv 装好(第 1 节);
- 主干文档 `docs/pointflow_per_point_tokens_20260919.md` 有全部实验结论与
  待办(幻影守卫、scale 重标定、位置编码、FK 模态)。

### 6.1 最新实验状态(2026-09-30)

- **在跑**:per_point-500(`pointflow_sandwich101_strat500_v2_20260929`)与
  cluster-origin baseline(`cluster_500p_origin_baseline`);A/B 结论 per_point 全胜,
  cluster 已 plateau(主线文档 9-30 节);
- **per-frame scale 已落地**:`POINTFLOW_DISPLACEMENT_FRAME_SCALES`(32 值向量,
  101 集实测 0.00506→0.07583)治"前段帧学不出来";启用需从头训 + 新 OUTPUT_ROOT,
  用法见 scale 文档 §14;
- **待启动**:cluster+skip+点级 block(方案 C,先单卡 smoke)、per-frame scale 版
  per_point run、密集版 N=4000(需先重建窗口缓存 + 重扫 scale)。

## 7. 排障速查

| 症状 | 看哪里 |
| ---- | ------ |
| `torch._C` import 报错 | 没 `export LD_LIBRARY_PATH=''`([setup.md](./setup.md#pytorch-import-issue)) |
| OOM / NCCL / 训练慢 | [docs/faq.md](./faq.md) |
| push 被 pre-push hook 拦 | 机器缺 git-lfs;conda base 里有,或确认无 LFS 文件后 `git push --no-verify` |
| 训练报 labeled 目录不存在 | manifest 死链,见第 2/6 节 |
| "labeled uv outside the tracker grid" | 相机 rig 变了,仿射常数重测(数据 pipeline 步骤 2) |
| `No module named 'addict'` / `spconv` / `torch_scatter` | sonata 依赖没装,见第 1 节"pointflow 追加依赖" |
| "Cannot open head video: None" | 转换时漏了 `--video`,补 labeled report.json 的 `video` 字段(数据 pipeline 步骤 1b) |
| 手动跑脚本行为怪异/签名对不上 | 用了脚本路径直接跑,被 base checkout 的 editable 安装劫持;改 `python -m` 从本仓库根跑(第 2 节) |

## 文档地图

| 文档 | 内容 |
| ---- | ---- |
| [AGENTS.md](../AGENTS.md) | 仓库总地图(目录结构、命令、规则) |
| [docs/setup.md](./setup.md) / [setup_offline.md](./setup_offline.md) | 在线 / 离线(内网预下载)环境安装 |
| [docs/pointflow_data_pipeline.md](./pointflow_data_pipeline.md) | 数据处理操作手册(接新数据必读) |
| [docs/pointflow_multinode_training.md](./pointflow_multinode_training.md) | 多机训练:SenseCore 注入兼容、HSDP 配置、提交命令、排障 |
| [docs/pointflow_per_point_tokens_20260919.md](./pointflow_per_point_tokens_20260919.md) | 主线实验记录与结论(最长最全,含 9-30 A/B 对比与误差三分解) |
| [docs/pointflow_experiment_queue_20260930.md](./pointflow_experiment_queue_20260930.md) | **实验队列**:状态(进行中/等待中/已完成)+ 各实验启动命令 |
| [docs/pointflow_displacement_scale_20260914.md](./pointflow_displacement_scale_20260914.md) | scale 根因与标定;§14 per-frame scale(逐帧归一化,env 向量) |
| [docs/pointflow_cluster_decode_20260929.md](./pointflow_cluster_decode_20260929.md) | 簇 token + 点级 decode:方案、实现、env 清单、实验计划 |
| [docs/pointworld_implementation_20260929.md](./pointworld_implementation_20260929.md) | PointWorld/EgoWAM 点处理机制调研(方案对照) |
| [docs/pointflow_eval_visualization.md](./pointflow_eval_visualization.md) | eval 可视化与 stage viewer 用法 |
| [docs/pointflow_position_encoding.md](./pointflow_position_encoding.md) | 位置编码分析((t,h,w,z) 方案) |
| [docs/training.md](./training.md) / [docs/inference.md](./inference.md) | 通用训练 / 推理 |
