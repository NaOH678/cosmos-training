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

## 3. 启动训练

```bash
export LD_LIBRARY_PATH=''
OUTPUT_ROOT=<新目录,每次必换> bash examples/launch_pointflow_labeled29_sandwich.sh   # sandwich
OUTPUT_ROOT=<新目录>           bash examples/launch_pointflow_dropper101.sh            # dropper
```

- 脚本开头的 `Checking inputs...` 会核对 TOML、数据集、checkpoint、缓存,缺什么会直接报;
- `NPROC_PER_NODE` 默认 8,按可见 GPU 数覆盖;
- `OUTPUT_ROOT` **每次显式换**(防 resume 混淆);resume = 同一 `OUTPUT_ROOT` 重跑同一命令;
- WANDB 默认 offline;更多开关见脚本内注释和
  [docs/pointflow_data_pipeline.md 步骤 5](./pointflow_data_pipeline.md)。

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

## 6. 当前状态(2026-09-28)

- sandwich/dropper 的 cosmos 缓存(manifest/video/window latents)**已就绪**;
- 9.24 新版轨迹的 flat labeled 转换**尚未跑**(`pf_out/9.24/*/efep_seg_v61/` 各
  101 集待 `tools/convert_efep_labeled.py` 转换),旧 manifest 是死链——
  **直接起训练会读不到轨迹而报错**,先按步骤 1b 转换再重建 manifest;
- 主干文档 `docs/pointflow_per_point_tokens_20260919.md` 有全部实验结论与
  待办(幻影守卫、scale 重标定、位置编码、FK 模态)。

## 7. 排障速查

| 症状 | 看哪里 |
| ---- | ------ |
| `torch._C` import 报错 | 没 `export LD_LIBRARY_PATH=''`([setup.md](./setup.md#pytorch-import-issue)) |
| OOM / NCCL / 训练慢 | [docs/faq.md](./faq.md) |
| push 被 pre-push hook 拦 | 机器缺 git-lfs;conda base 里有,或确认无 LFS 文件后 `git push --no-verify` |
| 训练报 labeled 目录不存在 | manifest 死链,见第 2/6 节 |
| "labeled uv outside the tracker grid" | 相机 rig 变了,仿射常数重测(数据 pipeline 步骤 2) |

## 文档地图

| 文档 | 内容 |
| ---- | ---- |
| [AGENTS.md](../AGENTS.md) | 仓库总地图(目录结构、命令、规则) |
| [docs/setup.md](./setup.md) / [setup_offline.md](./setup_offline.md) | 在线 / 离线(内网预下载)环境安装 |
| [docs/pointflow_data_pipeline.md](./pointflow_data_pipeline.md) | 数据处理操作手册(接新数据必读) |
| [docs/pointflow_per_point_tokens_20260919.md](./pointflow_per_point_tokens_20260919.md) | 主线实验记录与结论(最长最全) |
| [docs/pointflow_eval_visualization.md](./pointflow_eval_visualization.md) | eval 可视化与 stage viewer 用法 |
| [docs/pointflow_position_encoding.md](./pointflow_position_encoding.md) | 位置编码分析((t,h,w,z) 方案) |
| [docs/training.md](./training.md) / [docs/inference.md](./inference.md) | 通用训练 / 推理 |
