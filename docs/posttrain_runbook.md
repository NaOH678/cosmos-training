# Cosmos 训练 Runbook & 检查清单(自用)

> 目的:每次启动新训练前**一条条过**,避免重复踩坑浪费时间。
> 背景环境:HPC 集群(A800-80GB ×8/节点,driver 550 / CUDA 12.4),Apptainer SIF + `.venv-cosmos`,**无 HF 外网**。
> 关联:`docs/training.md`(官方 SFT)、`docs/custom_dataset.md`(自定义数据)、agent 记忆 `franka-xhand-dual-action-posttrain`、`cu128-env-on-apptainer`。

---

## 0. 启动前 30 秒检查清单(逐条打勾)

```
[ ] 节点 GPU 空闲?  squeue -u $USER 没残留;目标卡 nvidia-smi 干净
[ ] 上次崩的进程清了?  pkill -u $USER -f cosmos_framework.scripts.train
[ ] 命令里【没有】 export LD_LIBRARY_PATH=     ← 会清掉 --nv 的驱动库 → "no NVIDIA driver"
[ ] 命令里【没有】 --quotatype=spot/reserved   ← 默认配额才分到能用的 GPU 节点
[ ] tokenizer 指向本地?  model.config.vlm_config.tokenizer.pretrained_model_name=<本地 text_tokenizer>
[ ] WAN_VAE_PATH 是原版 .pth(不是 diffusers safetensors)
[ ] BASE_CHECKPOINT_PATH 是 DCP 目录(model/ + checkpoint.json),不是 HF 目录
[ ] IMAGINAIRE_OUTPUT_ROOT 指到你想存的盘(默认会写 /tmp)
[ ] max_iter 和 cycle_lengths 【相等】       ← 否则 lr 提前归零,后面白训
[ ] max_iter 按【这次的卡数/有效 batch】定过 ← 8卡和32卡同样步数 ≠ 同样训练量
[ ] save_iter × ckpt大小 不会爆盘            ← 16B+EMA 每个 ~80GB
[ ] 并行:多机别用 full-shard,用 HSDP        ← shard=每节点卡数, replicate=节点数
[ ] wandb_mode 设对(无外网用 "offline")
[ ] 先 --dryrun 校验配置,再 max_iter=20 smoke,最后正式跑
```

---

## 1. 环境 / 启动命令(集群相关,最容易卡)

**单机基本姿势**(你文档里验证过的,照抄别加料):
```
srun -p eailab_os --gres=gpu:N apptainer exec --nv --bind /mnt:/mnt \
  --workdir <repo> <sif> \
  bash -c "source .venv-cosmos/bin/activate && <路径env> torchrun --nproc_per_node=N ..."
```

**坑(本次实际踩过)**:
- ❌ **`export LD_LIBRARY_PATH=`**:`apptainer --nv` 把宿主驱动库路径注入容器的 `LD_LIBRARY_PATH`,清空它 → torch 找不到 libcuda → `RuntimeError: Found no NVIDIA driver`。**绝对不要加这句**(它只在某些纯 CPU 构建场景才需要)。
- ❌ **`--quotatype=spot/reserved`**:本集群加了反而分到没驱动的节点(`cuda: False`)。**默认配额**(不加 quotatype)才稳。
- ⚠️ 残留进程占显存:任务崩了 torchrun 的 worker 可能不退、占满显存 → 下个任务 OOM。**重跑前先 `pkill -u $USER -f cosmos_framework.scripts.train`**。

**多机(N 节点 × 8 卡)**:
```
srun -p eailab_os -N<节点数> --ntasks-per-node=1 --gres=gpu:8 bash -c '
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  export MASTER_PORT=29500; export NODE_RANK=$SLURM_NODEID
  apptainer exec --nv --bind /mnt:/mnt --workdir <repo> <sif> bash -c "source .venv-cosmos/bin/activate && <路径env> \
    torchrun --nnodes=<节点数> --nproc_per_node=8 --node_rank=\$NODE_RANK \
      --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT -m cosmos_framework.scripts.train ..."'
```
- `MASTER_ADDR` 在**宿主机**算(容器内可能没 `scontrol`);`\$` 转义让容器内 bash 取透传的环境变量。
- 多机常见警告 `[c10d] hostname of the client socket cannot be retrieved err=-3` → **无害**,反向 DNS 噪音。
- 若卡在 rendezvous 5 分钟没动 → master_addr 解析/透传问题,改 `APPTAINERENV_MASTER_ADDR` 或 `--rdzv-endpoint`。

---

## 2. 离线依赖(无 HF 外网,本集群必踩)

官方 ckpt 是 **HF Diffusers 格式**(`Cosmos3OmniDiffusersPipeline`),里面打包了 tokenizer / ViT / diffusers-VAE。训练前要转 DCP,而 `convert_model_to_dcp` 会**实例化整个模型** → 连带需要 tokenizer + Wan VAE 在本地。

- **Qwen tokenizer**:在 `<ckpt>/text_tokenizer/`。训练/转换都指过去;config_variant 用 `"hf"`(`"gcp"` 会去 s3/HF 下载)。
- **Wan VAE**:框架要**原版 `Wan2.2_VAE.pth`**(`WanVAE_` 的 196 个 key),**不是** diffusers 的 `AutoencoderKLWan` safetensors。两者 key 命名不同,不能直接塞。
- **DCP 转换可纯 CPU 跑**(不抢 GPU 节点):在**任何 cosmos import 之前** `os.environ["COSMOS_DEVICE"]="cpu"`(`flags.py` 在 import 时读它)。转换脚本里 monkeypatch:① `create_qwen2_tokenizer_with_download` 读本地;② `download_checkpoint_v2` 对 wan2pt2 URI 返回本地 .pth。
- 转出的 DCP:`<out>/model/*.distcp + .metadata` + `<out>/checkpoint.json`。训练 `BASE_CHECKPOINT_PATH` 指 `<out>`。

---

## 3. 训练基本配置(数值)

| 项 | 含义 / 坑 |
|---|---|
| `max_iter` | **优化器步数**。和卡数无关,但**有效 batch 随卡数变** → 见下方"步数换算"。 |
| `cycle_lengths` | LambdaCosine 周期,**必须 = max_iter**。否则 lr 在 cycle_lengths 步就降到 0,后面全白训。 |
| `warm_up_steps` | 线性 warmup;按 max_iter 比例设(~1–5%)。 |
| `save_iter` | 存盘频率。**算一下 ckpt 数 × 单个大小**(16B+EMA ≈ 80GB)别爆盘。 |
| `grad_accum_iter` | 每步累计几个 micro-batch 再更新。 |
| `max_batch_size` | **每卡** micro-batch 上限(`max_tokens` 设很大时由它定)。 |
| `lr` | 大 batch 可考虑线性/根号上调;微调一般先保守。 |

**有效 batch / 步数换算(关键,本次踩过)**:
```
有效 batch/步 = 卡数 × max_batch_size × grad_accum_iter
```
- **8 卡 ≠ 32 卡**:同样 max_iter,32 卡每步喂 4 倍数据。
- 要把某个旧 run 的训练量搬到新卡数:**max_iter 与有效 batch 成反比**缩放。
  例:8卡@50000步(batch64) = 3.2M 样本 → 32卡(batch256)要 12500 步才等量。
- 一个 epoch 步数 = 数据集样本数 / 有效 batch。

---

## 4. 并行 / 性能(本次重点:不要无脑 full-shard)

- **多机别用 fully-sharded**(`data_parallel_shard_degree = world_size`):每层都跨**慢的节点间网络** all-gather 参数 → 通信瓶颈。
  症状:`nvidia-smi` GPU-util 100% 但**功耗忽高忽低**(部分卡 ~100W),显存只用一点点。
- **用 HSDP(Hybrid Sharded)**:
  ```
  [model.parallelism]
  data_parallel_shard_degree     = <每节点卡数, 如 8>   # 节点内 NVLink 分片
  data_parallel_replicate_degree = <节点数, 如 4>       # 跨节点只 all-reduce 梯度
  ```
  约束:`shard × replicate × context_parallel = world_size`。有效 batch 不变。
  - ⚠️ 若用 CLI override `…data_parallel_shard_degree=-1`,会盖掉 TOML 的 HSDP → 记得**去掉那个 override**。

  **实测(dual-dex,32 卡 = 4 节点 × 8,16B,batch4/卡)**:
  | | full-shard(shard=32)<br>AC=full | **HSDP(shard8×repl4)<br>AC=full** | HSDP<br>AC=none(关梯度检查) |
  |---|---|---|---|
  | **单步耗时** | ~30s | **~9–10s(快 3×)** | **~18.3s(反而更慢!)** |
  | 功耗 | 98–282W 乱跳(等通信) | **382–424W 顶上限** | 383–422W 顶上限 |
  | 显存/卡 | ~20GB | ~33.6GB(剩 ~46GB) | ~63.7GB(剩 ~16GB) |
  | ETA(12500 步) | ~4.3 天 | **~35 小时** | — |

  **两条经验(都实测过)**:
  1. **full-shard → HSDP = 3× 提速**(30s→10s):去掉跨节点 all-gather 的通信瓶颈。判断通信瓶颈看 `nvidia-smi` 功耗——**忽高忽低=在等网络(该换 HSDP),顶上限=算力/访存饱和**。
  2. **⚠️ 关梯度检查(AC=none)在这套上反而更慢**(10s→18.3s)+ 显存翻倍(33→64GB)。注意:none 时**功耗也顶上限、util 100%**,说明不是等通信,而是 **存全部激活 → 反向写/读巨量激活(显存带宽密集)+ 63.7/80GB 近满载的分配器碎片同步**;A800 算力富余、带宽相对吃紧,**"重算(多 FLOPs)" 比 "存+读巨量激活(吃带宽)" 更便宜** → AC=full 又省显存又更快。**结论:多机大模型保持 `AC=full`;想再快试 `selective`(中间档),别用 `none`。** `100% util + 功耗满 ≠ 高效`。
- **activation_checkpointing**(`model.config.activation_checkpointing.mode`):
  - `full`:省显存、慢(重算激活)。**默认、稳**。
  - `none`:快、吃显存(存全部激活)。显存有富余时可换速度。
  - **判断**:开 HSDP 后先 `nvidia-smi`;剩余 > ~35GB 再试 `mode=none`,**用 CLI override 试、OOM 就撤**(第 1 步会崩,零成本回退)。
- 显存利用低 = 有提速空间;但**改一项、测一项**,别 HSDP + 关 AC 一起上以免 OOM。

---

## 5. 日志 / 监控

- **wandb_mode**:`"disabled"`(不记)/ `"offline"`(本地记,之后 `wandb sync` 上传,**无外网用这个**)/ `"online"`(要 `WANDB_API_KEY` + 通 wandb.ai)。
  - 改了**下次启动才生效**(运行中的任务不热更新)。
- **wandb 自动分项记 loss**:`wandb_log.py` 会把 `output_batch` 里所有含 "loss" 的 key 各打一条曲线 → `flow_matching_loss_vision / _action / _sound` 自动分开,**不用额外写**。
- **stdout 看分项 loss**:`iter_speed` 默认只打总 loss;本仓库已改它额外打 `L_vid / L_act`(改动在 `callbacks/iter_speed.py`)。
- **loss 怎么读**:`policy` 模式是**视频+动作联合**预测,不是只有 action。
  ```
  总 loss = L_vid × 1.0 + L_act × action_loss_weight(默认 10)
  ```
  显示的 L_vid/L_act 是**加权前**的原始值;盯 **L_act 是否下降**(策略在不在学)。
  - 实测(dual-dex 早期):`L_vid≈0.1, L_act≈1.5 → 总≈0.1×1 + 1.5×10 = 15.1`。**总 loss 几乎全是动作**(视频基座本来就会、loss 已很低);所以**只看 L_act 的下降趋势**即可。
- **无害警告**(别慌):
  - `OFUCallback: no nvidia-smi samples` → 容器内采不到 GPU 利用率,只影响 MFU 数字。
  - `Guards may run slower on Python 3.13.0` → torch.compile 提示。
  - `[c10d] hostname ... err=-3` → 多机反向 DNS 噪音。
  - 一大堆 `Skipping loading of key: net_ema.*` → 正常,基座加载时按 `keys_to_skip_loading` 跳过 EMA / 新初始化的适配器。

---

## 6. Checkpoint

- 训练存 **DCP**(`model/*.distcp + .metadata`,`scheduler/`,`dataloader/`)。
  - **不是每卡一个文件、也不用手动合并**:`.metadata` 记全局布局,加载时**自动重分片**,可用任意卡数 resume。
  - 含 EMA(`ema.enabled=True`)→ 体积 ~2 倍(16B ≈ 80GB)。**做 policy 一般用不到 EMA,想省盘可 `ema.enabled=False`**。
- **续训**:`load_path` 指 DCP;resume 会按 `dataloader/rank_*.pkl` 快进数据位置(精确续训)。
- **部署**:`export_model` 把某个 `iter_N` 转成单文件 HF safetensors → 喂 `action_policy_server_*`。

---

## 7. 数据适配(换新数据/新本体时)

- **先读真实数据,别假设**:动作语义(关节 vs 末端位姿 vs 增量)、图像存法(mp4 vs parquet 内嵌字节)、meta 格式(jsonl vs parquet)。本次就因为没读差点写错。
- 步骤:① `domain_utils.py` 注册本体(domain_id + raw_action_dim);② 算归一化统计 JSON;③ 写 map-style Dataset;④ 写 `DataPacker`;⑤ 写 experiment 并在 `configs/base/config.py` 的 `make_config()` 里**显式 import 注册**;⑥ 配 TOML。
- **batch 嵌套契约**(踩过):`video` / `action` / `text_token_ids` 都要 **`[[每样本 tensor]]`** 双层;`image_size` 堆叠成 `[B,4]`;其余字段扁平 per-sample 列表。
  - text_token_ids 给扁平 `[tensor,...]` → `TypeError: 'int' object is not iterable`(模型按 `[t for x in batch for t in x]` 解析)。
- **验证三连**:`--dryrun`(配置)→ `max_iter=20` smoke(端到端、看 loss 有限)→ 正式跑。

---

## 8. 常见报错速查

| 报错 | 原因 | 处理 |
|---|---|---|
| `Found no NVIDIA driver` | 命令里 `export LD_LIBRARY_PATH=` 清掉了 --nv 驱动库;或落到坏节点/纯CPU节点 | 去掉那句;不加 --quotatype;确认 `--nv` + `--gres=gpu` |
| `uvx hf download Qwen/...` 失败 | 无 HF 外网,tokenizer config_variant=gcp 去下载 | tokenizer 指本地 text_tokenizer + config_variant=hf |
| VAE `to_empty` / `no NVIDIA driver`(转换时) | `COSMOS_DEVICE` 没在 import 前设 cpu | 转换脚本顶部 `os.environ["COSMOS_DEVICE"]="cpu"` |
| `CUDA out of memory`(刚启动) | 别的/残留进程占着卡 | `pkill -u $USER -f scripts.train`;`nvidia-smi` 看占用 |
| `TypeError: 'int' object is not iterable`(_pack_text_tokens) | text_token_ids 嵌套不对 | collate 用 `[[tensor]]` 双层 |
| lr 早早变 0 / loss 不降 | `cycle_lengths != max_iter` | 两者设相等 |
| 多机 util 100% 但慢、功耗抖 | full-shard 跨节点通信瓶颈 | 换 HSDP(shard=节点内卡数, replicate=节点数) |

---

---

## 9. 本次 dual-dex 改了哪些代码、为什么(详解)

> 目标:把一台**新机器人本体**(dual-arm Franka + XHand,38 维动作,3 相机,parquet 内嵌图)接进 Cosmos3 action policy 训练。下面是**完整改动 + 每处的原理**,后续接别的本体照此模板。

### 9.1 改动的 3 个仓库文件(框架文件,小改)

**(a) `data/vfm/action/domain_utils.py`** —— 注册本体身份
```python
EMBODIMENT_TO_DOMAIN_ID["franka_xhand_dual"]   = 25   # 身份编号(选 25:未占用且 < num_embodiment_domains=32)
EMBODIMENT_TO_RAW_ACTION_DIM["franka_xhand_dual"] = 38 # 真实动作维度(用于补零掩码,见 9.4)
```
- `domain_id` 是模型选"动作接口槽"的索引(第 10 节);`raw_action_dim` 告诉模型"补零到 64 后,前 38 维才是真的"。
- 约束:`domain_id < num_embodiment_domains`(32);别和官方已用的号冲突(0,1,2,3,4,5,6,7,8,12,13,15,20 已占)。

**(b) `configs/base/config.py`** —— 让实验被 Hydra 发现
```python
def make_config():
    ...
    import cosmos_framework.configs.base.experiment.action.posttrain_config.franka_xhand_dual_policy  # noqa: F401
```
- 实验文件用 `cs.store(group="experiment", ...)` 在 **import 时**注册进 Hydra ConfigStore;**不显式 import 就不会执行 `cs.store`** → `experiment=franka_xhand_dual_policy` 找不到。这是 OSS 的注册方式(`vision_sft_nano` 等也是这样显式 import 的)。

**(c) `callbacks/iter_speed.py`** —— 日志打分项 loss
- 默认只打**合并总 loss**;policy 训练里总 loss 几乎被动作主导(`L_act × 10`),要单独看 video/action 趋势。
- 改动:从 `output_batch` 取 `flow_matching_loss_vision / _action / _sound`(模型已存在那里),拼进日志行。两处都改了:前 50 步的详细行 + 之后的 `every_n_impl` 周期行。
- 注:wandb 其实已**自动**把 `output_batch` 里所有含 `loss` 的 key 各记一条曲线(`wandb_log.py`),所以 wandb 端不用改;这里只是补 stdout。

### 9.2 新增的 6 个文件(训练配方)

| 文件 | 职责 | 关键点 |
|---|---|---|
| `…/datasets/franka_xhand_dual_lerobot_dataset.py` | map-style Dataset | 读 parquet **内嵌 PNG 字节**(非 mp4)、读 `episodes.jsonl/tasks.jsonl`(非 parquet meta)、按 episode 切窗口(`chunk_length=16`)、源 ~30fps **降采样到 15**、3 相机拼成近方形、`actions` 列直接 quantile 归一化(无位姿数学)。`__getitem__` 返回 `{video[C,T,H,W] uint8, action[16,38], ai_caption, mode, domain_id=25, conditioning_fps, viewpoint}` |
| `…/action/franka_xhand_dual_packer.py` | `ActionDataPacker(DataPacker)` | `sft_process_sample` 调 `ActionTransformPipeline`(resize/pad 视频、tokenize 文本、**动作补零到 64**、建 sequence_plan);`compute_num_tokens`;`sft_collate_fn` 拼成模型要的 batch(`video/action/text_token_ids` 都是 `[[tensor]]` 双层,`image_size` 堆叠 `[B,4]`,其余 per-sample 列表) |
| `…/datasets/compute_action_stats.py` | 归一化统计脚本 | 扫全 202 个 parquet 的 `actions` 列,算逐维 `q01/q99/mean/std/min/max` |
| `…/datasets/franka_xhand_dual_normalization.json` | 上面跑出的统计 | 38 维;2 个恒零手指 DOF → 归一化成常数 -1(无害) |
| `…/experiment/action/posttrain_config/franka_xhand_dual_policy.py` | experiment | 镜像 `vision_sft_nano` 但换 `DataPackerDataLoader` + `DataLoaderStateCallback`;HSDP;优化器训 `moe_gen + 动作适配器`;`keys_to_skip_loading` + `strict_resume=False`(见 9.5) |
| `examples/toml/sft_config/franka_xhand_dual_policy.toml` | 启动 TOML | `[job].experiment` 指向上面实验,叠加 HSDP/max_iter/save_iter/wandb 等覆盖 |

### 9.2.1 dual-dex 数据到底怎么切(以及和官方 DROID 的区别)

先说结论:当前 dual-dex **不是** "202 条轨迹 = 202 个训练样本"。  
它是:

1. 每条轨迹(episode / parquet)先按时间切成很多个**窗口**
2. 每个窗口产出 1 个训练样本
3. `__len__` 统计的是**窗口数**,不是轨迹数

#### A. 当前 dual-dex 的切分逻辑

`FrankaXHandDualLeRobotDataset` 里这几个参数是关键:

- `fps=15.0`
- `chunk_length=16`
- 源数据 `info.json` 里 `fps ≈ 29.9734`
- `step = round(source_fps / target_fps) = round(29.9734 / 15) = 2`
- `window_stride` 默认 `None`,于是 `stride = chunk_length = 16`

代码等价于:

```python
source_fps = 29.9734
target_fps = 15.0
step = 2
chunk_length = 16
stride = 16

rows = [s + i * step for i in range(chunk_length + 1)]   # 17 帧观测
action_rows = rows[:chunk_length]                        # 前 16 帧对应 16 步 action
```

含义:

- **30Hz 原始数据先降到 15Hz**:原始每隔 2 行取 1 行
- 每个 sample 用 **17 帧观测 + 16 步 action**
- 一个 sample 覆盖的原始跨度是 `chunk_length * step = 32` 行
- 但下一个窗口起点只前进 `stride = 16` 行
- 所以当前是**50% overlap 的滑窗**

#### B. dual-dex 的完整例子

假设某条轨迹原始 row 编号是:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
```

原始频率约 30Hz,相邻 2 行之间约 `66.7ms`,也就是训练时的 15Hz 采样间隔。

**窗口 1** 令 `start = 0`:

```text
obs rows:    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
action rows: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
```

- 这是 **17 帧观测**
- 配 **16 个 action**
- 覆盖真实时长约 `32 / 30 ≈ 1.07s`

**窗口 2** 令 `start = 16`:

```text
obs rows:    16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48
action rows: 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
```

能看到:

- 窗口 2 不是从 32 开始,而是从 16 开始
- 所以窗口 1 和窗口 2 在时间上**重叠了一半**

#### C. 为什么 202 条轨迹会变成很多 iter

当前这份数据(202 条轨迹)按上面的逻辑展开后,实际约是:

- `202` 个 episode
- `13720` 个窗口样本
- 平均每条轨迹约 `67.9` 个窗口

如果训练配置是:

- 8 卡
- `max_batch_size=6`
- `grad_accum_iter=1`

那么每个 optimizer step 的全局 batch 是:

```text
8 * 6 * 1 = 48 个窗口 / iter
```

所以 1 个 epoch 大约是:

```text
13720 / 48 ≈ 286 iter
```

所以日志里看到"epoch=0 但已经很多 iter"是**正常现象**,因为 epoch 统计的是**窗口**,不是原始轨迹数。

#### D. 和官方 DROIDLeRobotDataset 的对比

两者**高层目标一致**:

- 都用 `fps=15.0`
- 都用 `chunk_length=16`
- 都是 **17 帧观测 + 16 步 action**

但**具体切分逻辑不一样**:

| 项 | dual-dex(当前) | 官方 `DROIDLeRobotDataset` |
|---|---|---|
| 原始频率处理 | 从 `~30Hz` **降采样到 15Hz**(`step=2`) | 代码里**不显式做 step 降采样**,直接取连续 row |
| 单个 sample 的观测行 | `s, s+2, s+4, ..., s+32` | `idx, idx+1, ..., idx+16` |
| 窗口起点 stride | `16` | `1` |
| overlap | 约 **50%** | **极密滑窗**(每次只挪 1 行) |

官方 DROID 的核心代码更像:

```python
observation_rows = self._rows[idx : idx + 17]
action_rows = observation_rows[:16]
```

也就是说:

- **官方 DROID**:连续时间序列上的**超密滑窗**
- **当前 dual-dex**:先降采样,再做**较稀疏的重叠滑窗**

#### E. 官方 DROID 的完整例子

还是假设原始 row 编号是:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
```

官方 DROID 的 `__getitem__(idx)` 是直接连续切:

**窗口 1** 令 `idx = 0`:

```text
obs rows:    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
action rows: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

**窗口 2** 令 `idx = 1`:

```text
obs rows:    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
action rows: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
```

能看到:

- 它**不跳行**
- 下一个窗口只前进 1 行
- 所以 overlap 非常大

如果原始数据本身就是 15Hz,那这就是"严格按 15Hz 连续滑窗"。  
而 dual-dex 当前这套更像"原始 30Hz 数据先稀疏到 15Hz,再按较大的 stride 切窗口"。

#### F. 为什么当前 dual-dex 要先降采样到 15Hz

因为我们希望在 `chunk_length=16` 不变时,每个 chunk 覆盖**更长的真实时间**。

- 若保持 30Hz 且 `chunk_length=16`,一个 chunk 只覆盖 `16 / 30 ≈ 0.53s`
- 降到 15Hz 后,同样 `chunk_length=16`,一个 chunk 覆盖 `16 / 15 ≈ 1.07s`

等价于:

- **固定 token 数**
- 换来**更长时间视野**
- 同时减少相邻帧冗余

代价是:

- 时间分辨率变粗
- 快速瞬时动作细节会少一些

如果后面想做对照实验,最值得改的 3 个旋钮就是:

1. `fps`: 15 vs 30
2. `chunk_length`: 16 vs 32
3. `window_stride`: 16 vs 32 vs 1

### 9.3 动作的完整数据流(38 → 64 → 4096 → 64 → 38)
```
Dataset: actions[16,38] 原始值
  → 归一化 quantile → [16,38] ∈ [-1,1]
Packer/ActionTransformPipeline: pad_action_to_max_dim → [16,64](第 38..63 补 0),记 raw_action_dim=38
collate: action = [[tensor[16,64]]],domain_id=[tensor(25)],raw_action_dim=[tensor(38)]
模型前向(cosmos3_vfm_network.py):
  pack_action → 把每个 action token 打上该样本的 domain_id(per-token 展开)
  action2llm(64→4096, 按 domain_id 取 25 号槽) → [T,4096]
  + action_pos_embed(3D RoPE) + action_modality_embed(可学习"我是动作"标记)
  → 进 36 层 MoT 主干(diffusion/gen 那半,full/noise 注意力)
  → llm2action(4096→64, 25 号槽) → 预测 [T,64] 的速度场
```

### 9.4 38 维和基座"不匹配"为什么不用改模型(补零 + 掩码)
模型动作投影固定是 `max_action_dim=64`,所有本体进模型前**都补零到 64**。多出的 **38..63 通道在 4 个地方被置零/排除**,所以它们**不携带任何信号、不参与训练**:
| 位置 | 代码 | 作用 |
|---|---|---|
| 加噪后的 x_t | `omni_mot_model.py:1462` `xt_action[i][:, raw_dim:]=0` | 补零通道不被加噪 |
| 噪声本身 | `omni_mot_model.py:1709` `noise_action_i[:, raw_dim:]=0` | 不给补零通道注噪声 |
| 速度目标 v | `omni_mot_model.py:1952` `v[:, raw_dim:]=0` | flow-matching 目标在补零通道为 0 |
| loss | `algorithm/loss/flow_matching.py:75-76` `sqerr_i = sqerr_i[:, :raw_dim]` | 平方误差只切前 38 维,补零通道不进 loss |
→ 等价于"模型实际只学这 38 个真实维度",10 维(DROID)、38 维(你)共用同一套 64 维投影**毫无冲突**。

### 9.5 新本体权重为什么"从零初始化 + 非严格 resume"
基座 `Cosmos3-Nano` 里**没有 domain 25 的动作权重**(且 nano-base 未必经过动作 mid-train),硬 load 会 shape/缺键。实验里:
```python
keys_to_skip_loading = ["net_ema.", "action2llm", "llm2action",
                        "action_modality_embed", "action_pos_embed"]
strict_resume = False
```
→ 加载基座时**跳过这些动作适配器**(`trunc_normal` 随机初始化,见第 10 节),其余主干(Qwen und 专家、diffusion gen 专家、`vae2llm`/`llm2vae` 视频投影、`time_embedder`…)**全部正常复用**。日志会刷 `Skipping loading of key: net.action2llm...` 印证。这也是为什么 `L_act` 初期高(~1.5)、降得快——动作头确实从零在学你这个本体。

---

## 10. DomainAwareLinear / 多本体"槽"机制(详解)

**作用**:让**一个模型支持多种机器人本体(cross-embodiment)**——视觉理解、动力学、推理等**主干共享**,只有**动作的输入/输出投影**按本体分开。实现借鉴 X-VLA(代码注释标注)。`action2llm` / `llm2action` 就是它。

### 10.1 "槽"的本质:两张 nn.Embedding 权重表
`DomainAwareLinear` **不是** 32 个独立 `nn.Linear`,而是把 32 套权重压进 embedding 表,按 `domain_id` 查表:
```python
class DomainAwareLinear(nn.Module):
    def __init__(self, input_size, output_size, num_domains):
        self.fc   = nn.Embedding(num_domains, output_size * input_size)  # 权重表:每行=一套【展平的】权重矩阵
        self.bias = nn.Embedding(num_domains, output_size)              # 偏置表:每行=一套 bias

    def forward(self, x, domain_id):                 # x:[B,T,I], domain_id:[B] (每个样本一个本体号)
        W = self.fc(domain_id).view(B, I, O)         # 取该样本本体的那行 → 还原成 [B,I,O] 矩阵
        b = self.bias(domain_id).view(B, O)
        return torch.bmm(x, W) + b.unsqueeze(1)      # 逐样本(batched)线性: [B,T,I]×[B,I,O]=[B,T,O]
```
- 以 `action2llm`(I=64, O=4096, num_domains=32)为例:
  - `fc.weight` 形状 **`[32, 64*4096]=[32, 262144]`** —— 第 25 行 reshape 成 `[64,4096]` 就是**你这台机器人的权重矩阵**;
  - `bias.weight` 形状 **`[32, 4096]`** —— 第 25 行是你的 bias。
- 数学上**等价于"32 个 `Linear(64→4096)`,按 domain_id 选一个用"**;用 `bmm` 实现的好处:**一个 batch 里不同样本可以是不同本体**(各取各的行),不用拆 batch。

### 10.2 两个方向 + 在前向里的位置
| 模块 | I→O | 每槽权重 | 在前向里 |
|---|---|---|---|
| `action2llm` | 64→4096 | `[64,4096]` + bias`[4096]` | `cosmos3_vfm_network.py:755` 把补零后的动作投进隐空间 |
| `llm2action` | 4096→64 | `[4096,64]` + bias`[64]` | `:847-848` 把主干输出的动作隐状态投回 64 维(预测速度场) |

前向链(`cosmos3_vfm_network.py`):
```
pack_action(:752) → 每个 action token 打上样本的 domain_id(d_id.expand(T),per-token)
action2llm(:755, 取 25 号槽) → + action_pos_embed(3D RoPE,:758) + action_modality_embed(:768)
→ 36 层 MoT 主干
→ llm2action(:847, 25 号槽) → 预测 [T,64] 速度(再按 9.4 只取前 38 维算 loss)
```
- **domain_id 是 per-token 展开的**:同一样本的所有 action token 都贴同一个本体号,所以 `DomainAwareLinear` 能对每个 token 索引正确的槽。

### 10.3 初始化与参数量
- 网络 `set_up` 里对动作适配器用 **`trunc_normal_`**(`cosmos3_vfm_network.py:242-248`),bias 置零(注意:`DomainAwareLinear.__init__` 默认是 xavier,但网络这里**覆盖**成 trunc_normal)。
- 参数量:每槽 ≈ `64×4096 + 4096 ≈ 266K`;32 槽 ≈ **8.5M**(单个 `action2llm`),加 `llm2action` 也才 ~17M —— 相对 16B 主干**微不足道**,所以"动作头从零训"很轻、收敛快。

### 10.4 选槽看"本体"不看"任务"(易混点)
- `domain_id`/槽 = **机器人身份**(哪台机器人、什么构型/动作空间);
- **任务**("capture the tube" / "倒水" …)由**文本指令**表达,走共享主干的语言理解,**与 domain_id 无关**;

| 场景 | 槽 | 指令 |
|---|---|---|
| 同一机器人,不同任务 | **同槽** | 不同指令 |
| 不同机器人,同一任务 | **不同槽** | 同一指令 |

类比:**槽 = "哪副身体/手脚接口",指令 = "现在让它干什么"**。共享一个"大脑"(主干),每台机器人一套专属"接口"。

### 10.5 官方槽 vs 新槽、跨本体、部署一致性
- **官方槽**(`droid=8`、`libero=5`、`bridge=7`…):NVIDIA 训过 → 若**不跳过加载**,槽里带预训练动作权重,可借用/迁移;
- **空槽**(你的 25):随机初始化、从零学;
- **本次跑**:`keys_to_skip_loading` 跳过整个 `action2llm/llm2action` → **32 个槽全部新初始化**,只用/训 25 号(数据全是 `domain_id=25`),其它槽是没用上的随机权重;
- **跨本体混训**:不跳过加载、保留官方槽,多本体数据混在一起(每条样本带自己 `domain_id` 自动走各自槽),主干共享、互相促进;
- **⚠️ 部署一致性**:训练时数据标 `domain_id=25`,**部署时 `action_policy_server` 必须 `--domain-name franka_xhand_dual`(→ 同样解析成 25)**,否则会去读别的(没训过的)槽,动作全错。

---

---

## 11. 常用启动命令(dual-dex 项目,复制即用)

**固定路径**(写在这省得每次找):
```
SIF      = /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif
REPO     = /mnt/hwfile/shichaojian/cosmos-framework
DATASET  = /mnt/petrelfs/shichaojian/lingbot-va/wan_va/dataset/dex-dual-capture-tube-combined-lerobot
WAN_VAE  = /mnt/inspurfs/eailab_os/scj_ckpt/wan2.2_VAE.pth/Wan2.2_VAE.pth
NANO_DCP = /mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano-dcp                  # 训练基座(DCP)
TOKENIZER= /mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano/text_tokenizer
OUTPUT   = /mnt/inspurfs/eailab_os/scj_ckpt/cosmos-dual-dex                   # IMAGINAIRE_OUTPUT_ROOT
RUNDIR   = $OUTPUT/cosmos3/action/franka_xhand_dual_policy                    # 产物/checkpoint/wandb 都在这下
TOML     = examples/toml/sft_config/franka_xhand_dual_policy.toml
```

**① GPU 自检(1 卡,确认 cuda 可用)**
```
srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt --workdir /mnt/hwfile/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && nvidia-smi -L && python -c 'import torch;print(torch.cuda.is_available())'"
```

**② 清残留进程(崩溃后必做,否则占显存 OOM)**
```
pkill -u $USER -f cosmos_framework.scripts.train
```

**③ 32 卡训练 · 从基座开始(HSDP,AC=full)**
```
srun -p eailab_os -N4 --ntasks-per-node=1 --gres=gpu:8 bash -c 'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); export MASTER_PORT=29500; export NODE_RANK=$SLURM_NODEID; apptainer exec --nv --bind /mnt:/mnt --workdir /mnt/hwfile/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && DATASET_PATH=/mnt/petrelfs/shichaojian/lingbot-va/wan_va/dataset/dex-dual-capture-tube-combined-lerobot WAN_VAE_PATH=/mnt/inspurfs/eailab_os/scj_ckpt/wan2.2_VAE.pth/Wan2.2_VAE.pth BASE_CHECKPOINT_PATH=/mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano-dcp IMAGINAIRE_OUTPUT_ROOT=/mnt/inspurfs/eailab_os/scj_ckpt/cosmos-dual-dex PYTHONPATH=. torchrun --nnodes=4 --nproc_per_node=8 --node_rank=\$NODE_RANK --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/franka_xhand_dual_policy.toml -- model.config.vlm_config.tokenizer.pretrained_model_name=/mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano/text_tokenizer"'
```

**④ 从某个 checkpoint 续跑**:把 ③ 里的 `BASE_CHECKPOINT_PATH` 改成
`/mnt/inspurfs/eailab_os/scj_ckpt/cosmos-dual-dex/cosmos3/action/franka_xhand_dual_policy/checkpoints/iter_000012500`
（注:当前 `strict_resume=False / load_training_state=False`，是"接权重重新计步/优化器"，不是严格断点续训；要严格续训需把 `checkpoint.load_training_state=true` 一并 override。）

**⑤ 关梯度检查提速**:在 ③/④ 命令末尾 `--` 那段再加一项(其它不变):
`model.config.activation_checkpointing.mode=selective`（或 `=none` 最快但最吃显存，盯第 1 步别 OOM）

**⑥ 8 卡单机 smoke(20 步,验证能否跑通)**:用 ③ 的内层但去掉多机部分:
```
srun -p eailab_os --gres=gpu:8 apptainer exec --nv --bind /mnt:/mnt --workdir /mnt/hwfile/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && DATASET_PATH=/mnt/petrelfs/shichaojian/lingbot-va/wan_va/dataset/dex-dual-capture-tube-combined-lerobot WAN_VAE_PATH=/mnt/inspurfs/eailab_os/scj_ckpt/wan2.2_VAE.pth/Wan2.2_VAE.pth BASE_CHECKPOINT_PATH=/mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano-dcp IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/franka_xhand_dual_policy.toml -- model.config.vlm_config.tokenizer.pretrained_model_name=/mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano/text_tokenizer model.config.parallelism.data_parallel_shard_degree=-1 trainer.max_iter=20"
```
（单机 8 卡用 `data_parallel_shard_degree=-1` 自动分片；多机才用 TOML 里的 HSDP。）

**⑦ DCP 转换(CPU 离线,基座只需转一次)**:见 `.tmp_convert_local.py`（已含 tokenizer/VAE 指本地 + `COSMOS_DEVICE=cpu` 三补丁），跑:
```
srun -p eailab_os --cpus-per-task=16 --mem=200G apptainer exec --bind /mnt:/mnt --workdir /mnt/hwfile/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && export LD_LIBRARY_PATH= PYTHONPATH=. HF_HUB_OFFLINE=1 COSMOS_DEVICE=cpu && python .tmp_convert_local.py"
```

**⑧ 保存 stdout 日志**:任意训练命令末尾加 `2>&1 | tee ~/train_$(date +%m%d_%H%M).log`（框架不自动存训练 stdout）。

**⑨ wandb 离线曲线上传**（避开 `cosmos3` 项目路由问题,指自己的 project）:
```
wandb sync -p dual_dex_action /mnt/inspurfs/eailab_os/scj_ckpt/cosmos-dual-dex/cosmos3/action/franka_xhand_dual_policy/wandb/offline-run-*
```

---

_最后更新:基于 dual-dex(franka_xhand_dual)action policy 首次 32 卡训练通跑的经验整理(已完成 12500 步)。新方法/新模型时按本清单逐条核对。_
