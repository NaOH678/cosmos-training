# 真机选点深度来源对比（A~E）任务书

目标：选出在线选点用的深度方案。全部离线，不改训练和部署代码。

| 方案 | 深度 | 网络相机条件 | 尺度 |
|---|---|---|---|
| A | D435 过滤版 | 无 | 传感器 |
| B | DA3 | 无 | DA3 自身 |
| C | DA3 | 真实内参+固定位姿 | DA3 自身 |
| D | DA3 | 无 | 每帧 D435/DA3 比值中位数（mask 内） |
| E | DA3 | 真实内参+固定位姿 | 同 D |

## 术语定义

**尺度**：把 DA3 深度换算成真实米数的系数，真实深度 ≈ 尺度 × DA3 深度。内参决定点在哪条射线上，尺度决定点在射线上走多远，所以真实内参修不了尺度误差。尺度错了，整个点云会一起放大或缩小。

各方案的尺度来源：
- A（传感器）：D435 直接测距，不需要换算。
- B、C（DA3 自身）：用 DA3 metric 分支对当前这一帧给出的米制深度，不做末段尺度恢复，不用其他帧。
- D、E（D435 中位数）：每帧计算 s = median(z_D435 / z_DA3)，再令 z = s × z_DA3，整帧共用一个 s。

D、E 的计算细则：
- 像素范围：当前帧手和物体 mask 的并集，且 D435 深度在 0.15–2.5 m 内、DA3 深度有效。
- D435 深度先配准到 RGB 画布，比较的是同一像素的 z；此处不使用 D435 过滤结果，靠中位数抵抗飘移点。
- 最终点的位置全部来自 DA3，D435 只贡献 s，不进入点云。
- 有效像素少于 200 时沿用上一帧的 s，并记录这些帧。阈值 200 是初值，若回退频繁需在报告中说明。
- 只用当前帧和更早的数据，不读未来帧；不做时间平滑。

**A 的过滤版**：沿用 tools/render_d435_point_accuracy.py 的规则，即 0.15–2.5 m 范围和 3×3 邻域支持，不加因果跨帧过滤。

**真实内参 + 固定位姿（C、E）**：向 DA3 网络输入按输入分辨率缩放后的 RGB 真实内参和单位外参（头部相机固定），写法见 tools/run_da3_surface_comparison.py。

硬性要求：DA3 必须单帧在线运行，不用 128 帧上下文和末段尺度；所有方案都用真实内参反投影；同帧用同一份 mask。

数据（只评有 D435 的帧）：
- sandwich: episode_0013_20260731_133649，mask 在 pointflow_outputs/sandwich_bread_variants_20261030_v2/separate
- dropper: episode_0000_20260830_155455，mask 在 pointflow_outputs/dropper_micro_requested_20261030/dropper_mix
- micro: episode_0007_20260824_174548，mask 在 pointflow_outputs/dropper_micro_requested_20261030/micropipette

可复用：
- DA3 加载与条件输入：tools/run_da3_surface_comparison.py
- D435 反投影与过滤：tools/render_d435_point_accuracy.py
- URDF 表面距离：tools/analyze_da3_urdf_conditions.py
- 选点：tools/validate_real_foreground_selection.py 的 sim_recipe（右手版，1024 点，半径 0.05 m）
- 环境：/data/shichaojian/Track4World_portable/Track4World/.venv，LD_LIBRARY_PATH 置空

指标（手部点到 URDF 表面，mm；全部点和 FPS 选中点各算一遍，以选中点为准）：
每帧 P90 的中位数/均值/最大值、P90>30 mm 的帧数、每帧中位距离的中位数。
另报 DA3 单帧耗时 p50/p95（丢弃热身）和 D/E 每帧尺度的波动。物体无 3D 真值，只报选中点中离群点比例并做目检。

额外检查：sandwich 上 D435 的 P90>30 mm 有 69 帧（基线 d435_point_accuracy_20261102），目检确认是深度飘移还是 mask 漏到背景。

交付：结果写到 pointflow_outputs/depth_source_comparison_20261103/（新目录，含 README、per_frame、summary），给出 A~E 对比表和推荐方案。每个任务只有一条数据，结论需注明这一局限。
