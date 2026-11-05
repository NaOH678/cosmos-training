# 真机 CoTracker 替代实验（2026-11-04）

入口：`tools/run_real_cotracker_pipeline.py`；批处理：`tools/run_real_cotracker_batch.py`。
输出：`pointflow_outputs/cotracker_real_20261104/`。原始数据只读；随机种子 0，每类一条完整视频，共 5,991 帧。

|任务|episode|帧数|URDF|
|---|---|---:|---|
|bottle|episode_0077_20261102_133352|739|dagger|
|scissors|episode_0066_20261103_144300|987|dagger|
|dropper_dagger_new_new|episode_0029_20261103_181615|1039|dagger|
|dualhand_micropipette|episode_0112_20260825_105714|1675|complete|
|singlerighthand_sandwich_100|episode_0009_20260731_133239|1551|complete|

URDF 的完整路径：

- complete：`/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf`
- dagger：仓库内 `assets/dagger_fk/marvin_wuji_d435_dagger.urdf`

已将两个候选 URDF 的真实手表面投影叠加到五条录制首帧核对，见每个任务的 `urdf_candidates.jpg`。不按视频拟合外参。逐条读取录制的 RGB 内参和 SDK depth-to-color 外参，不使用 URDF 型号来更改内参。

## 处理规则

1. **SAM2.1-large 全部物体持续跟踪**。没有复用静态 mask 的优化。物体提示与当前任务一致；只在首次初始化时使用搜索区域，移动物体后续不受该搜索框约束。用当前 qpos 和正确 URDF 去掉手臂；黑手材质和 3px opening 去掉细线缆；移液枪架、移液枪沿用原基线的白色、蓝白材质门限。
2. 训练数据在前 0、5、10、15、20、30、45、60 帧按时间顺序补充未初始化目标。每次只读当前图像，已初始化的对象继续 SAM2 跟踪；不反向传播、不把后面发现的 mask 写回早先帧。部署按用户指定的无遮挡首帧只执行第 0 帧初始化。同一组文本/搜索区域/过滤规则可以复用，具体框由 DINO 生成。这是用户允许的初始化条件差异，并非严格相同的初始化输入。
3. **CoTracker3 只替换二维点跟踪**。每次输入 RGB[t-1]、RGB[t]，以旧 UV 为 query 重新初始化该图像对，外部维护整条 episode 的轨迹 ID。在当前手/物体 mask 的未覆盖网格补点，grid_step=3。物体合并成 label 3；手 label 2，不要求持续识别面包/芝士等类别。丢失的 UV 轨迹终止，再出现分配新 ID。深度失败可以使观测无效，但不会伪造有效三维标签。
4. 使用官方 CoTracker3 `scaled_online.pth`，BF16；预训练 16 帧时间嵌入截取前 2 帧，与仿真代码中的短窗口方案一致。真实瓶子视频六个静止/运动图像对的短窗口对比记录在 `context_audit.json`：共同可见点的差异 P95 0.109–0.280px，可见性一致率 100%。这不是相对真实轨迹的精度测量。
5. **DA3 来自当前 Track4World**：原 DA3 Nested Metric 权重，加 `track4world_da3.pth` 的 AnyView 权重，复用 metric scaling、depth alignment、sky 处理。输入同样是 640×448，再转到 630×448 的 DA3 网络网格；米制缩放使用记录的真实 RGB 焦距。当前试验使用单个当前图像，避免旧缓存的 128 帧未来上下文及最后 chunk 标度；因此不是旧 Track4World 全片缓存的逐值复现。DA3 confidence >2 沿用原门限。
6. **D435 校正**：SDK 标定到 RGB 的点云使用最近 Z 可见性及同实例局部邻域检查。手 mask 内距边界至少 3px 且位于内部距离最高四分位作为核心；用 5mm 直方图、10mm 平滑取主峰，要求 ±3cm 支持占比 ≥55%、至少 30 个可靠像素，差值 MAD 筛选。每只手独立检查观测历史，取有效手的平移量中位数，应用到全部手和物体。
7. **历史突变门限**：复用 `HandDepthGuard`，只保存最近 4 个已接受的手核心 Z；中位数速度裁到 ±0.6m/s；当前残差大于 40mm + 0.20m/s×min(age,0.25s) 时拒绝。最多预测 2 次观测、最长 0.5s，预测值不回灌历史；不可靠时不使用该 D435 修正。两次传感器更新之间短时延续最近校正量，过期丢弃。
8. **仅做 Z 平移**：先用原始 DA3 深度和真实 K 得到 XYZ，然后 `xyz[:,2] += delta`。X、Y 逐值不变；原始观测 UV 独立保存，不由修正后 XYZ 重新投影替换。所有物体与手共用同一 delta，不单独拟合物体。
9. **飞点检查**：同实例像素邻域深度断层门限 max(20mm,4%Z)；三维 16NN 的稳健离群点筛选，最小 8mm 门限；相同轨迹的深度变化减去该实例的中位数运动，再用 max(25mm,5×1.4826×MAD) 去掉残差异常。只使用当前/过去观测。没有未来插值、双向平滑或按 URDF 拟合。实验阶段额外的“整物体 DA3 中位深度门限”曾误删真实拿起动作，已删除并对缓存重跑最终过滤。

## 模型数据与产物

每个任务目录内：

- `segmentation_h264.mp4`：完整 SAM2 mask 叠加。
- `pointflow_rgb_h264.mp4`：彩色轨迹 ID、最近 10 帧轨迹线；只连接相同 ID 的连续有效观测。
- `cotracker_da3_interactive.html`：沿用之前 Track4World 的 Viser 离线打包。固定初始视角，可手动旋转、缩放、暂停和拖时间轴。默认校正后手/物体，手橙色、物体 RGB；可切原始 DA3、被剔除候选和真实 URDF 手表面。HTML 展示每 3 帧一次、稳定 ID 取一半点，不影响完整视频和数据导出。
- `pointflow/obs_pos.npy`、`obs_uv.npy`、`obs_track.npy`、`obs_valid.npy`、`obs_unique.npy`、`frame_offsets.npy`、`track_birth.npy`、`track_label.npy`：与旧 Track4World ragged 观测具有相同用途。
- `pointflow/timestamps_sec.npy` 使用原视频 FPS 的时间轴，与旧模型窗口约定相同；真实相机时间另存在 `sensor_timestamps_sec.npy`，供历史门限使用，不覆盖原始记录。
- `pointflow/frames/*.npz` 保存每帧 UV、校正前后 XYZ、有效性、实例、颜色和置信度，便于区分分割、跟踪与深度错误。
- `pointflow/model_window_*.npz`、`verification.json`：开头和中段的原有 PointFlow 窗口，以及实际模型 batch 的形状验证。

适配器 `cosmos_framework/data/ragged_pointflow.py`：

```python
from cosmos_framework.data.ragged_pointflow import RaggedPointFlow
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.data.pointflow_batch import build_pointflow_batch

source = RaggedPointFlow(".../bottle/pointflow")
sample = source.prepare_sample(0, timing=PointFlowTiming(fps=15, steps=32, steps_per_token=4), seed=0)
batch = build_pointflow_batch([sample])
```

保留原有 normals、RGB 采样、2cm voxel、9维 PTV3 特征和相机坐标米制位移标签。UV 到原 RGB 的像素中心 affine 记录在 sample metadata 中。未修改 Cosmos 主干、既有训练配置及 PointFlow 选点预算。适配器用于接入新 ragged 数据；原来只会读取 `position.npy` 的生产数据来源尚未自动切换为该适配器。

## 检查边界

本地无法读取用户电脑的 `file:///C:/.../sandwich_episode_0013_v2_dense3d.html`；参考的是本地同名旧 HTML 及同一套 Viser 打包工具。原生 `vis_3d_efep` 的未来补洞和双向平滑不作为在线过滤；本次不使用未来帧。

看起来没有孤立飞点不等于真实深度准确：透明瓶、细管及遮挡区域仍可能出现连片错误或变形。被剔除点保留红色诊断图层，数据中的 `obs_valid` 与展示一致。后续报告会分别列出缺失、剔除、手表面距离和延迟；手表面距离不作为物体深度真值，也不会用它反过来修正数据。

## 本轮检查后的补充

- sandwich 的初始“蛋/芝士”检测框几乎相同，两个初始 mask IoU=0.9847。已改为**相对当前 DINO 框**的正负点提示：蛋在框内 (0.44,0.92) 为正、(0.40,0.42) 为负；芝士反过来。SAM2 使用这些点分开可见蛋白和芝士，不使用旧 episode 的固定 mask/框。模板可以在部署首帧复用；目标的具体图像坐标仍由当前 DINO 框决定。全片重新跟踪。
- 为抑制三维点整步跳远，增加每个连续可见 ID 的 3D 位移门限 `max(80mm,1.5m/s×dt)`。超过门限只标无效，不改 XYZ、UV，不补成训练真值。与误删真实拿起动作的“整物体中位深度永久门限”不同；不会因为物体离开初始位置而永久丢弃它。规则在训练预处理和部署中相同。速度门限引入时版本为2，最终统一过滤版本号为 `filter_revision=3`。
- bottle 独立运行 20 帧与全片前 20 帧对比，ID、UV、原始 DA3 XYZ 逐值完全一致，记录在 `bottle/prefix_invariance.json`。这验证了该前缀没有受到后续图像影响。
- `RaggedPointFlow.prepare_sample()` 已补齐当前模型 batch 所需的 UV→RGB 像素中心 affine 和视频尺寸，`build_pointflow_batch` 已在实际数据上验证。它保留原始观测 UV，并不把修正 XYZ 重投影后的 UV 换进模型。

- 最终版本3增加当前帧18mm半径三维连通分量：较小分量若离主体最近距离>60mm，且Z中位数差>60mm，才剔除。它针对普通SOR删不掉的整片错误深度；不会修改保留点的坐标。
