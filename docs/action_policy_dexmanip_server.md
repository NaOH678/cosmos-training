# DexManip Action Policy Server

Use this server when the robot-side client is
`dexmanip_tool/dexhand_deployment/deployment.py` and you want to keep that local
stack unchanged.

The server:

- loads the `franka_xhand_dual_policy` checkpoint directly from a DCP iteration
  directory;
- accepts DexManip's existing pickle + ZMQ observation protocol;
- tiles the three RealSense views into the same image layout used during
  training;
- runs Cosmos policy inference on the GPU node;
- denormalizes the predicted 38-D action chunk with
  `franka_xhand_dual_normalization.json`;
- returns one absolute action step per request in the exact field layout that
  `deployment.py` already understands.

## 1. Launch on the GPU node

Follow the existing cluster runtime pattern documented in
`INSTALL_APPTAINER.md` and `docs/posttrain_runbook.md`:

```shell
srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt \
  --workdir /mnt/petrelfs/shichaojian/cosmos-framework \
  /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif \
  bash -c "source .venv-cosmos/bin/activate && <your command>"
```

If this is the same `.venv-cosmos` you already used for training, you usually
do not need to rebuild the full environment. Check the required imports first:

```shell
srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt \
  --workdir /mnt/petrelfs/shichaojian/cosmos-framework \
  /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif \
  bash -c "source .venv-cosmos/bin/activate && python -c 'import torch,zmq,cosmos_framework; print(torch.cuda.is_available())'"
```

If `zmq` is missing, sync the serving group into that existing `cu128` env:

```shell
srun -p eailab_os apptainer exec --bind /mnt:/mnt \
  --workdir /mnt/petrelfs/shichaojian/cosmos-framework \
  /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif \
  bash -c "export UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=.venv-cosmos UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4; \
    uv sync --all-extras --group cu128-train --group policy-server --python /mnt/petrelfs/shichaojian/miniconda3/envs/cosmos/bin/python"
```

Start the server:

```shell
srun -p eailab_os --gres=gpu:1 apptainer exec --nv --bind /mnt:/mnt \
  --workdir /mnt/petrelfs/shichaojian/cosmos-framework \
  /mnt/hwfile/shichaojian/ubuntu2204-cuda124-uv.sif \
  bash -c "source .venv-cosmos/bin/activate && PYTHONPATH=. \
    python -m cosmos_framework.scripts.action_policy_server_dexmanip \
      --checkpoint-path /mnt/inspurfs/eailab_os/scj_ckpt/cosmos-dual-dex/cosmos3/action/franka_xhand_dual_policy/checkpoints/iter_000012500 \
      --port 25555 \
      --prompt 'capture the tube'"
```

Notes:

- `--checkpoint-path` may point at the DCP iteration directory directly. The
  loader infers the `franka_xhand_dual_policy` experiment and resolves the
  underlying `/model` directory automatically.
- Do not add `export LD_LIBRARY_PATH=` before the launch command. The existing
  Apptainer runbook explicitly warns that this breaks CUDA driver injection.
- Follow the existing runbook and use the default quota pattern shown above
  unless you already know a different quota mode is safe for your partition.
- The default camera order is:
  `241122305539,236422070385,115222071207`
  which must match training `cam_1/cam_2/cam_3`.
- The default prompt is `capture the tube`, which matches the training task
  caption.

## 2. Connect from DexManip

Keep `deployment.py` and `policyclient.py` unchanged. On the robot machine,
point the policy server IP to the GPU node and use a lower local control rate
that matches training:

```yaml
policy_server:
  ip: "<gpu-node-ip>"
  port: 25555
  timeout_ms: 10000

control_hz: 15
camera_hz: 30
obs_hz: 15

arms: ["left", "right"]
```

Why `15 Hz`:

- the raw dataset was recorded at about `29.97 Hz`;
- `FrankaXHandDualLeRobotDataset` downsamples it to `15 Hz` before training;
- each predicted chunk contains `16` action steps at that cadence.

Driving the local controller at `50 Hz` without interpolation burns through a
16-step chunk in about `0.32 s`, which defeats the chunked policy design.

## 3. Safety and behavior

- The server returns **absolute** Franka targets:
  `ee_pos[3] + ee_quat_xyzw[4]`
- XHand targets are returned in **degrees**
- On bootstrap, the first request blocks until the first chunk is generated
- If a later chunk is not ready in time, the server falls back to
  **hold-current-pose/current-hand**

## 4. Recommended local DexManip settings

Start from the LingBot dual-arm config and change:

- `policy_server.ip`
- `policy_server.port: 25555`
- `control_hz: 15`
- `obs_hz: 15`
- keep the RealSense serial order identical to training

The Franka absolute-mode safety knobs from
`dexhand_deployment/configs/lingbot_va_dual.yaml` remain a good starting point.
