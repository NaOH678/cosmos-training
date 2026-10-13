# 真机 action 归一化统计（2026-10-06）

用户要求减少验证集，当前采用 seed=42、split_val_ratio=0.05，复用 SingleRightHandRawDataset 按 manifest 顺序和 allowlist 划分的实际逻辑。

| 数据 | 动作表示 | 训练/验证 | 统计action帧 | 当前统计 |
| --- | --- | --- | --- | --- |
| sandwich101 | xyz(m)＋quat_xyzw＋20维手关节(rad) | 96/5 | 101570 | [action_stats.json](sandwich101_val5pct/action_stats.json) |
| dropper mix71 | 7维臂关节＋20维手关节(rad) | 67/4 | 68946 | [action_stats.json](dropper_mix71_val5pct/action_stats.json) |

每个子目录同时有完整split.json和validation.json。旧dropper_mix71/action_stats.json是先前64/7划分的中间产物，不用于新的5%配方。

计算：只用训练集action的q01/q99，offset=(q01+q99)/2，scale=max((q99-q01)/2,0.05)，变换为(action-offset)/scale；state复用同一变换。无clip，无action/state合并分位数，无极值扩scale。源帧各统计一次，不因重叠窗口重复加权；NPZ有action_valid则使用，没有则使用全部有限缓存帧。本次不额外去尾或裁剪。

sandwich四元数维度3..6保持offset=0、scale=1；位置scale下限0.05米、关节下限0.05弧度。此为保留现有旋转表示的选择，并非已经证明最优的尺度。dropper mix全27维均为关节角，可直接使用0.05弧度下限。

检查全部172个episode的action/state：变换后均有限，float32归一化/反归一化最大误差1.1921e-7，sandwich四元数逐元素不变，train/val无episode交集。统计文件保存manifest和各训练NPZ的SHA256。

**归一化重训接入已新增（E8系列），不自动改变已有训练。** `examples/launch_real_normalized.sh`使用REAL_ACTION_STATS_PATH及SHA256快照，训练与验证loader校验划分和源数据后应用变换。joint_prediction.npz额外保存action_physical（物理单位）；部署/其他离线评估应显式使用RealActionNormalizer.denormalize，不能直接把模型输出送到机器人。使用新OUTPUT_ROOT，不继续旧未归一化checkpoint。PointFlow/FK几何及其位移scale不随action统计改变，FK计算仍使用反归一化后的物理量。原NPZ和latent未改写。

复现（输出路径必须尚不存在，防止覆盖）：

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/compute_real_action_stats.py \
  --cache-root /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache \
  --raw-root /data/shichaojian/raw_data/singlerighthand_sandwich_100 \
  --episode-allowlist examples/pointflow_sandwich924_all_101_episodes.txt \
  --split-val-ratio 0.05 --preserve-quaternion --output <新目录>/sandwich_action_stats.json

LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/compute_real_action_stats.py \
  --cache-root /data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache \
  --raw-root /data/shichaojian/raw_data/dropper_dagger_mix \
  --episode-allowlist examples/pointflow_dropper_dagger_mix_71_episodes.txt \
  --split-val-ratio 0.05 --output <新目录>/dropper_mix_action_stats.json
```
