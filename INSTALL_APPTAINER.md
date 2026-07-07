# Cosmos-Framework · Apptainer 环境安装记录

在 HPC 集群上,基于已有的 Apptainer SIF 镜像配置 cosmos-framework 的运行环境(推理 + 后训练)。
本文档记录**实际安装过程**、**对官方文档的偏离**、**踩过的坑**,以及 **GPU 验证**。

环境验证日期:2026-06-09。`torch 2.10.0+cu128` 在本集群可装可跑。

## 0. 背景与约束

- **GPU**:NVIDIA A800-SXM4-80GB(sm_80)
- **宿主驱动**:`550.90.07`,`nvidia-smi` 报 CUDA Version 12.4
- **关键**:驱动 12.4 通过 **CUDA 12.x minor 向前兼容**可以跑 **cu128**(CUDA 12.8)的 wheel(实测 5 项 GPU 算子全过)。但**跑不了 cu130**(CUDA 13 需要 R580+ 驱动)→ 只能用 `cu128` / `cu128-train` 组,**别碰 `cu130*`**。
- **登录/计算节点**:CentOS 7,glibc 2.17 —— **太旧**,torch 2.10 的 manylinux_2_28 wheel 在宿主直接跑会崩。必须在 SIF 内运行。
- **SIF 镜像**:`/mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif`(Ubuntu 22.04 + CUDA 12.4 + uv 0.9.22,**不带 python**),glibc 够新。
- **网络**:
  - 集群代理 `closeai-proxy.pjlab.org.cn:23128` 可达外网(`download.pytorch.org` / `github.com` / `nvidia-cosmos.github.io`)→ 外网包**保留代理**。
  - 内网 PyPI 镜像 `pkg.pjlab.org.cn` 必须**绕过代理**(进 `no_proxy`,而不是 unset 全部代理,否则外网也断)。
  - 集群**无 GitHub 出口给 uv 拉 python**(`python-build-standalone` 会 403)→ 必须 `UV_PYTHON_DOWNLOADS=never` + 自带 python。
- **Python**:由 conda env `cosmos` 提供(Python 3.13.0):`/mnt/petrelfs/shichaojian/miniconda3/envs/cosmos/bin/python`。
- **路径**:`/mnt/petrelfs/shichaojian/cosmos-framework` 是 `/mnt/hwfile/shichaojian/cosmos-framework` 的符号链接,同一份仓库。

## 1. 总体设计

SIF 不带 python、uv 又不能从 GitHub 拉 python,沿用 lingbot 的方案:**conda 提供解释器 + SIF 内 uv 装包**,但这次走仓库自带的 `uv.lock` 做 **frozen 同步**:

```
conda env (cosmos)             ── 只提供 Python 3.13.0
        ↓ --python
SIF (uv + cuda12.4 toolchain)  ── uv 在 venv 里按 uv.lock 装包
        ↓ UV_PROJECT_ENVIRONMENT
.venv-cosmos (项目目录下)       ── 实际运行/训练用的虚拟环境
```

venv 路径:`/mnt/petrelfs/shichaojian/cosmos-framework/.venv-cosmos`

## 2. 安装步骤(实际执行顺序)

所有命令模板:`srun -p eailab_os [--gres=gpu:1] apptainer exec [--nv] --bind /mnt:/mnt --workdir $(pwd) <SIF> bash -c '...'`。
`--nv` 仅 GPU 任务用;`--quotatype=spot` 走 spot 配额。

### 2.0 连通性 + 驱动 + 轮子可用性体检(GPU 节点)

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c 'echo "=== 驱动 ==="; nvidia-smi --query-gpu=name,driver_version --format=csv; uv --version; curl -4 -fsS https://download.pytorch.org/whl/cu128/torch/ --max-time 40 | grep -oE "torch-2\.10[^\"<]*\.whl" | grep cp313 | grep linux | sort -u'
```

预期:驱动 `550.90.07`;索引列出 `torch-2.10.0+cu128-cp313-...manylinux_2_28_x86_64.whl`。
> **注意**:体检 `download.pytorch.org` 这种外网时**不要 unset 代理**(外网靠代理)。

### 2.1 用 conda 的 python 建 venv

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os apptainer exec --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif uv venv .venv-cosmos --python /mnt/petrelfs/shichaojian/miniconda3/envs/cosmos/bin/python
```

### 2.2 装推理环境(`cu128` 组,frozen 同步)

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os apptainer exec --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c 'export UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=.venv-cosmos UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4; uv sync --frozen --group cu128 --python /mnt/petrelfs/shichaojian/miniconda3/envs/cosmos/bin/python'
```

- `--frozen`:用仓库自带 `uv.lock`,**跳过解析**。一旦不加 frozen 或改默认 index,lock 会失效 → uv 做全宇宙 universal 解析(把 vllm/cu130 所有冲突组 × x86_64+aarch64 全解一遍,还会为读 metadata 而 streaming 整个 flash-attn/natten 轮子),极慢。
- `UV_PYTHON_DOWNLOADS=never`:禁止 uv 偷偷去 GitHub 拉 python。
- `UV_PROJECT_ENVIRONMENT=.venv-cosmos`:让 sync 装进这个 venv(而非默认 `.venv`)。

### 2.3 升级到训练环境(`cu128-train` 组 + `train` extra)

> **关键**:`--group cu128-train` 只给 CUDA 训练工具链(transformer-engine / torchao / triton)。
> megatron-core / lerobot / datasets 在 **`train` extra**(`[project.optional-dependencies]`,不是 dependency-group),由 `--all-extras` 引入。
> 官方后训练装法(`docs/setup.md`)= 两个一起上:

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os apptainer exec --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c 'export UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=.venv-cosmos UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4; uv sync --all-extras --group cu128-train --python /mnt/petrelfs/shichaojian/miniconda3/envs/cosmos/bin/python'
```

- megatron-core / lerobot 是 **git 源**(`pyproject.toml:302-303`,Megatron-LM rev `de56227`),会 clone + 现场 build,经代理,慢。
- 只要训练、不要 serve/guardrail/libero/policy-server,可把 `--all-extras` 换成 `--extra train`。

### 2.4 (可选)音频模态补 torchaudio

cosmos 支持 audio 模态,`data/vfm/augmentors/audio_parsing.py` 里**惰性 import torchaudio** 做重采样。但 torchaudio **不在 lock 里**,sync 每次都会把它剪掉。只有用音频时才补:

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os apptainer exec --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c 'source .venv-cosmos/bin/activate; uv pip install torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128'
```

> **顺序**:必须放在所有 `uv sync` 之后,否则会被再次剪掉。

### 2.5 GPU 总验证

```bash
cd /mnt/petrelfs/shichaojian/cosmos-framework && srun -p eailab_os --quotatype=spot --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt --workdir $(pwd) /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c 'source .venv-cosmos/bin/activate; python -c "import torch,transformer_engine,torchao,flash_attn,natten; from megatron.core import parallel_state; import cosmos_framework; print(torch.__version__, torch.cuda.is_available())"'
```

实测输出:`2.10.0+cu128 True | all training imports OK`。

## 3. 对官方文档的偏离点

| 官方说法 | 实际做法 | 原因 |
|---|---|---|
| `uv sync --all-extras --group=cu130-train`(默认 CUDA 13) | 用 `cu128-train` | 驱动 550.90.07 只支持 CUDA 12.4,跑不了 cu130 |
| 直接 `uv sync`(自管 `.venv`) | `UV_PROJECT_ENVIRONMENT=.venv-cosmos` + SIF 内执行 | SIF 无 python,宿主 glibc 太旧 |
| 不说 python 哪来 | conda env `cosmos` 提供 + `UV_PYTHON_DOWNLOADS=never` | SIF 无 python,uv 拉 python 走 GitHub 被墙 |
| 默认走 PyPI/官方 index | 外网保留代理,pjlab 进 `no_proxy` | HPC 网络:外网要代理,内网镜像要绕代理 |

## 4. 踩过的坑(重要)

1. **代理一刀切 unset 会断外网**:pjlab 内网包才需绕代理;`download.pytorch.org` / cosmos 私有 index / github 都是外网,**要留代理**。正确做法是把 `pkg.pjlab.org.cn` 放进 `no_proxy`,而不是 unset 全部。
2. **"cache 秒装"有前提**:用临时 `uv pip install --index-url cu128` 随手装的 torch,缓存键按那个 URL 建;而 `uv sync --frozen` 走 lock 里 hash 钉死的源(nvidia/triton 是 `pypi.org`,torch 是 `download.pytorch.org/whl`),**键对不上就重下**。真正的复用是**让 sync 装进一个已装好 torch/cuXX 的 venv**(`UV_PROJECT_ENVIRONMENT`),uv 认已安装且版本匹配就跳过——实测 108 包 18s、271 包 35s 完成,大头不重下。
3. **Lustre 幽灵锁**:uv 缓存在 `/mnt/petrelfs/.../.cache/uv`(网络 FS)。**强杀**正在 build 的 uv 会留下不被任何进程持有、却让后来者等 300s 超时的死锁文件(`Failed to acquire lock on the distribution cache`)。解法:确认无 uv 在跑(`squeue -u $USER`)后 `find /mnt/petrelfs/shichaojian/.cache/uv -name '.lock' -delete`。
4. **别两个 `uv sync` 并行**:它们抢同一个 editable build 锁,必然一个超时失败。
5. **torch+cu128 不在 pjlab**(只在 download.pytorch.org);但 `nvidia-*` CUDA 库在 pypi,pjlab 能镜像 → 想加速这批可设 `UV_DEFAULT_INDEX=pjlab`,但那会让 lock 失效触发重解析,与 `--frozen` 冲突,权衡后 frozen 一律走代理。

## 5. 不需要的东西

- **torchaudio** —— 仅 audio 模态用,无音频任务不装(sync 也会自动剪掉)。
- **vllm** —— 和 `cu128*` 组**互斥**(`pyproject.toml:274-282`),只为 `packages/vllm-cosmos3` 里的 Qwen3-VL(prompt 上采样/VLM 服务)用。需要时**另建独立 venv**(`--group vllm`),不能塞进 `.venv-cosmos`。

## 6. 最终包清单关键版本

| 类别 | 包 | 版本 |
|---|---|---|
| Python | (conda env `cosmos`) | 3.13.0 |
| torch | torch / torchvision / torchcodec | 2.10.0+cu128 / 0.25.0+cu128 / 0.10.0+cu128 |
| CUDA(bundled) | nvidia-cudnn-cu12 / nvidia-cublas-cu12 / nvidia-nccl-cu12 | 9.10.2.21 / 12.8.4.1 / 2.27.5 |
| 训练工具链 | transformer-engine / torchao / triton | 2.12+cu128.torch210 / 0.16.0+cu128 / 3.6.0 |
| 注意力 | flash-attn / flash-attn-3-nv / natten | 2.7.4.post1+cu128.torch210 / 1.0.3+cu128.torch210 / 0.21.6.dev6+cu128.torch210 |
| 训练栈 | megatron-core / lerobot / datasets | git(Megatron-LM de56227)/ git / 已装 |
| 推理 | diffusers / transformers / accelerate | 0.37.0 / 4.57.6 / 1.13.0 |
| 数值 | numpy | 2.2.6(override 钉 `<2.3`) |

## 7. 日常激活模板

```bash
# CPU 任务
srun -p eailab_os apptainer exec --bind /mnt:/mnt --workdir /mnt/petrelfs/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && <你的命令>"

# GPU 任务(注意 --nv)
srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt --workdir /mnt/petrelfs/shichaojian/cosmos-framework /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif bash -c "source .venv-cosmos/bin/activate && <你的命令>"
```

> 提醒:**不要再对 `.venv-cosmos` 跑 `uv sync`**,否则会按 lock 重新对齐(把手动补的 torchaudio 等剪掉)。需要改依赖组时再 sync。
