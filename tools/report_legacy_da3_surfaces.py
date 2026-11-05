"""Human-readable historical-method URDF comparison and video index."""

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    root = Path("pointflow_outputs/legacy_da3_urdf_errors_20261104")
    methods = json.loads((root / "methods_and_videos.json").read_text())
    common = json.loads((root / "common239_same_uv_summary.json").read_text())
    native = json.loads((root / "native_support_summary.json").read_text())
    full_paired = json.loads((root / "same_uv_full_available_summary.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    rows = json.loads((root / "common239_same_uv_per_frame.json").read_text())

    def table(summary):
        lines = [
            "| 方法 | 帧数 | 平均距离 mm | 帧中位距离的平均 mm | 帧P90的平均 mm | P90>30mm帧数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key, item in summary.items():
            lines.append(
                f"| {methods[key]['name']} | {item['frames']} | {item['mean_frame_mean_mm']:.2f} | {item['mean_frame_median_mm']:.2f} | {item['mean_frame_p90_mm']:.2f} | {item['frames_p90_gt30']} |"
            )
        return "\n".join(lines)

    text = (
        """# 旧DA3方法重新统计：以URDF手表面为距离标准

数据：Sandwich `episode_0013_20260731_133649`。本报告只评价旧方法，不混入新的单帧A–E方案。
按此前五类修正方法展开为八个实现配置，另加未修正原始PointFlow参考。

## 主对比：同帧、同手部像素

统一239个有D435记录的帧，因为两种D435核心校准只有这239帧。
每帧统一原始有效手部UV，按旧四列实验的规则等间隔取最多2500个；所有九种方法使用完全相同的UV和点数。
方法之间只比较其旧几何变换／旧预测输出，不重新选1024点、不额外过滤、不重新估计校准参数。
每个DA3手点到同帧完整URDF手表面点集求最近欧氏距离，单位mm。
“平均距离”先逐帧平均，再对帧平均；“帧中位距离的平均”先逐帧取中位数，再对帧平均；P90列同理。

"""
        + table(common)
        + """

上表是公平横向比较的主要依据；不能拿它与旧21个FK关节到点云的误差数值混比，也不能直接与新单帧／1024选点实验混比。

## 各方法原有支持与完整帧数

旧缓存三组使用全部有效手部观测；四列重跑各组沿用最多2500点；核心校准沿用相同最多2500点、239帧。
各方法在自身原有帧范围的结果如下，不应把不同帧数的平均值直接用于排名：

"""
        + table(native)
        + """

`same_uv_full_available_summary.json` 还保存统一每帧最多2500像素、按各方法完整可用帧数的统计。

## 标准确实来自URDF网格

- 重新加载机器人URDF的右手visual三角网格，按面积以1.5mm间距参数、seed=0采样，每帧 """
        + str(verification["surface_points_per_frame"])
        + """ 个手表面点。
- 使用原始记录qpos逐帧正向运动学，使用原视频已校验的Link_Base→RGB相机变换；不做ICP、尺度拟合或平移拟合。
- 重建21关节只作FK链一致性检查，最大差 """
        + f"{verification['max_fk_keypoint_reconstruction_error_m']:.3g}"
        + """ m；**没有用这些关节作为距离目标**。
- 距离目标是整只手的完整URDF表面样本，排除手臂；没有根据DA3点来挑选或移动参考表面。采样表面不是精确的点到三角形距离，存在约毫米级采样离散误差。
- URDF不是实际手套扫描真值，模型形状、标定和记录同步误差仍会计入距离。
- 本次重算的URDF表面保存在`urdf_surface_camera.npy`，形状1192×32464×3；并核对与之前URDF表面缓存的一致性，见`verification.json`。

## 精确复现旧方法

1. **历史B＋末段**：原始缓存Z乘历史比例1.099358947872622，原UV按真实RGB K反投影。没有重新运行网络。
2. **历史B＋逐段**：按旧chunks_all.json每段estimated norm和Metric ratio，逆掉末段恢复影响后进行真实K反投影；不是四列重跑第二种的另一名称。
3. **四列第一／二种**：读取旧unconditioned缓存；第一种乘末段norm／当前段(norm+1e-6)，第二种保留本段米制几何。最终都用真实K。
4. **四列第三／四种**：读取旧conditioned缓存，真实K＋固定相机位姿已参与网络预测；尺度恢复分别对应末段／自身段。
5. **Z平移／视线缩放**：严格使用旧`calibration.json`中已保存的delta／scale，在旧四列第一种点云上应用。未通过旧可靠性筛选的10帧保持原值，共229帧校正；未剔除失败帧，也没有根据URDF重新拟合校正参数。

四列重跑是128帧上下文实验，不能解释成单帧在线结果。历史缓存B与四列第一种虽然配置相近，输入几何来源不同，因此分开报告。
旧第900帧校准失败被原样保留，避免按URDF结果挑掉不利帧。

## 每个旧方法对应的视频路径

| 方法 | 之前视频（绝对路径） | 视频内位置 |
|---|---|---|
"""
    )
    for key, info in methods.items():
        text += f"| {info['name']} | [{Path(info['video']).name}]({info['video']}) | {info['video_position']} |\n"
    text += """
视频仍是原来生成的文件，很多显示的是FK骨架；本次误差已独立改用URDF手表面重算，不是从旧视频标题／旧FK误差表抄数值。

之前已把第一／第三种替换成URDF手表面并显示DA3物体点的四个视频，可用于对应这两种方法目检：

"""
    for view in ["front", "side", "opposite", "overhead"]:
        p = (
            Path.cwd()
            / "pointflow_outputs/da3_urdf_two_conditions_20261103"
            / f"episode_0013_20260731_133649_conditions_1_3_urdf_objects_{view}.mp4"
        )
        assert p.exists()
        text += f"- [{view}]({p})\n"
    text += """
## 文件与验证

- `comparison.csv`：统一239帧主表，包含完整绝对视频路径。
- `common239_same_uv_per_frame.json`：逐帧主对比记录。
- `native_support_per_frame.json`：各方法原有点集和完整可用帧记录。
- `methods_and_videos.json`：方法定义与视频路径。
- `verification.json`：URDF重建、相同UV、历史已存点云重建误差等校验。
- `frame_median_errors.png`：统一239帧的中位距离随时间曲线。
- `tools/evaluate_legacy_da3_surfaces.py` 与 `tools/report_legacy_da3_surfaces.py`：复现脚本；只用CPU，无需GPU推理。

本次只有Sandwich一条episode；主表只评手，不能推出物体深度精度或其他任务的优劣。
"""
    text += '''
## 本次结论与失败尾部

- 统一239帧、相同手UV下，平均距离最小的是旧D435核心区域视线缩放（10.13mm），其次是Z平移（10.73mm）。
- 不借用D435测距的旧配置中，四列第四种（相机条件＋各段自身尺度）平均距离最低，为15.30mm。
- 核心校准仍有严重失败：视线缩放最坏帧P90为271.18mm，Z平移为243.00mm；第900帧的平均距离分别163.68mm和140.56mm。不能把平均改善理解成每帧可靠。
- 距离指标只描述已有可见手点的位置，不衡量漏点／缺失手指；即使距离小，也不能据此说明分割覆盖更完整。
'''
    (root / "README.md").write_text(text)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time = np.array([r["frame"] / 30 for r in rows])
    groups = [
        ["original_cached", "historical_B_last", "historical_B_own"],
        ["rerun_1", "rerun_2", "rerun_3", "rerun_4"],
        ["rerun_1", "core_Z_translation", "core_ray_scale"],
    ]
    for ax, group in zip(axes, groups):
        for key in group:
            ax.plot(time, [r["methods"][key]["median_mm"] for r in rows], label=key, linewidth=0.9)
        ax.set_ylabel("Frame median distance (mm)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Source time (s)")
    fig.suptitle("Historical DA3 methods | same 239 frames / same UV | distance to full posed URDF surface; no fitting")
    fig.tight_layout()
    fig.savefig(root / "frame_median_errors.png", dpi=150)
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    manifest = {}
    for name in [
        "tools/evaluate_legacy_da3_surfaces.py",
        "tools/report_legacy_da3_surfaces.py",
        "tools/render_urdf_surface_da3_d435.py",
    ]:
        p = Path(name)
        shutil.copyfile(p, snapshot / p.name)
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(table(common), flush=True)


if __name__ == "__main__":
    main()
