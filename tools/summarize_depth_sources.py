"""Aggregate A-E full/selected hand distances, plots and reproducible tables."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    root = Path("pointflow_outputs/depth_source_comparison_20261103")
    summary = {
        task: json.loads((root / task / "summary.json").read_text()) for task in ["sandwich", "dropper", "micro"]
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "source",
                "group",
                "valid_frames",
                "missing_frames",
                "median_frame_P90_mm",
                "mean_frame_P90_mm",
                "max_frame_P90_mm",
                "frames_P90_gt30mm",
                "median_frame_median_mm",
                "mean_hand_points",
            ]
        )
        for task, s in summary.items():
            for mode in "ABCDE":
                for group in ["all_hand", "selected_hand"]:
                    r = s[mode][group]
                    writer.writerow(
                        [task, mode, group]
                        + [
                            r[k]
                            for k in [
                                "valid_frames",
                                "missing_frames",
                                "p90_median_mm",
                                "p90_mean_mm",
                                "p90_max_mm",
                                "frames_p90_gt30",
                                "frame_p50_median_mm",
                                "selected_hand_count_mean",
                            ]
                        ]
                    )
    colors = dict(A="#168ed0", B="#ff7f0e", C="#239541", D="#b427a8", E="#9b5b24")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    scale_fig, scale_axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, sax, (task, s) in zip(axes, scale_axes, summary.items()):
        rows = json.loads((root / task / "per_frame.json").read_text())
        t = np.array([r["time"] for r in rows])
        t -= t[0]
        for mode in "ABCDE":
            ax.plot(
                t, [r[mode]["selected_hand"]["p90_mm"] for r in rows], label=mode, color=colors[mode], linewidth=0.8
            )
        ax.axhline(30, color="black", linestyle="--", linewidth=0.8)
        ax.set(title=f"{task}: selected hand points", ylabel="P90 distance to URDF (mm)", xlabel="Time (s)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=5)
        for mode in "DE":
            sax.plot(t, [r[mode]["scale"] for r in rows], label=mode, color=colors[mode])
        sax.set(title=task, ylabel="Sensor / DA3 scale", xlabel="Time (s)")
        sax.grid(alpha=0.2)
        sax.legend()
    fig.tight_layout()
    fig.savefig(root / "selected_hand_p90_over_time.png", dpi=140)
    scale_fig.tight_layout()
    scale_fig.savefig(root / "sensor_scale_over_time.png", dpi=140)
    table = ["| 任务 | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|"]
    for task, s in summary.items():
        table.append(
            "| " + task + " | " + " | ".join(f"{s[m]['selected_hand']['p90_median_mm']:.2f}" for m in "ABCDE") + " |"
        )
    badtable = ["| 任务 | 总帧数 | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, s in summary.items():
        badtable.append(
            "| "
            + task
            + " | "
            + str(s["metadata"]["frames"])
            + " | "
            + " | ".join(str(s[m]["selected_hand"]["frames_p90_gt30"]) for m in "ABCDE")
            + " |"
        )
    timings = (
        json.loads((root / "standalone_single_frame_timing.json").read_text())
        if (root / "standalone_single_frame_timing.json").exists()
        else None
    )
    timingtable = ["| 任务 | B p50 / p95 ms | C p50 / p95 ms |", "|---|---:|---:|"]
    if timings:
        for task in summary:
            r = timings["tasks"][task]["summary"]
            timingtable.append(
                "| " + task + " | " + " | ".join(f"{r[m]['p50_ms']:.1f} / {r[m]['p95_ms']:.1f}" for m in "BC") + " |"
            )
    audit = json.loads((root / "sandwich/failure_audit/audit.json").read_text())
    core = sum(r["far_points"] * r["far_inside_eroded_mask_fraction"] for r in audit) / sum(
        r["far_points"] for r in audit
    )
    text = (
        """# 单帧深度来源 A–E：URDF 手表面距离对比

本次只做离线对照，不修改训练缓存或部署代码。三个任务各一条完整数据，只评实际记录到D435的帧。

## 主要结果

以下是每帧选中手点到URDF表面的P90，再对帧取中位数，单位mm，越小越好。不是全片所有点混在一起的P90。

"""
        + "\n".join(table)
        + """

每帧选中手点P90超过30mm的帧数：

"""
        + "\n".join(badtable)
        + """

完整全点／选中点指标、极值、缺失帧、点数见 `comparison.csv` 和 `summary.json`。若某帧未选中任何手点，则记为缺失，不能当成零误差。
`selected_hand_p90_over_time.png` 显示全片长尾；`sensor_scale_over_time.png` 显示D/E尺度波动。

## 方法

- A：原始D435通过标定转换到RGB相机系，与同帧SAM2 mask关联；0.15–2.5m范围，3×3至少2个同实例邻居，深度差≤max(8mm,1%深度)。投影z-buffer容差3mm。没有SOR、分离团块或跨帧过滤，没有补点。
- B：单张RGB输入AnyView；当前帧Metric分支提供米制参考，真实RGB焦距用于Metric换算；AnyView深度对齐到Metric。最终使用真实RGB内参反投影。
- C：同B，但额外向AnyView输入按分辨率缩放的真实K及单位相机位姿。
- D/E：分别对B/C深度乘s。s是在手＋物体mask并集、D435范围0.15–2.5m且两者深度有效的同像素上计算median(D435/DA3)。使用未过滤的RGB配准D435，最终点全部来自DA3。
- D/E有效像素<200时继承上一帧s，不做平滑；若首次就不足200，明确初始化s=1。实际回退数、尺度均值／std／范围与相邻变化见各任务summary。没有读取未来帧。
- DA3使用与旧实验相同的Track4World AnyView权重＋DA3原始Metric分支。每次时间维N=1，没有128帧上下文、没有末段／逐段归一化恢复。RGB预处理为640×480→640×448→630×448，深度回到640×480后用真实K反投影。
- 所有方案使用同帧同一份已生成SAM2-large实例mask；不重新分割。
- 全部前景进入候选池，随机池上限8192，固定seed=源帧号；沿用sim_recipe的右手版本，显式预算1024：512全局FPS，再以剩余预算在右手21关节5cm内选点，最后全局补足。点类别不参与选取，但用于选后统计。
- Micro有双手，URDF距离统计包括双手表面，选点按任务书仍为右手局部优先，左手只能通过全局部分或回填进入。不能把本结果解释为双手各256点配额的测试。
- URDF visual网格按面积采样约1.5mm间距，seed=0，通过记录qpos逐帧FK及相机坐标变换。指标为测得手点到完整URDF手表面样本的最近距离，无ICP、位移拟合或URDF删点。显示为相机自可见表面。
- Dropper使用dagger URDF，其他任务使用完整D435机器人URDF。URDF并非手套扫描真值，仍有模型形状、标定和曝光同步误差，物体没有三维真值。

## 单帧耗时

以下是全片并发计算结束后的单进程复测：A800，模型常驻GPU，FP16，N=1，每任务5轮预热、12个均匀分布帧各测2次、B/C交替执行。
范围为Metric＋AnyView＋米制对齐＋深度resize，不包括视频解码、RGB预处理/H2D、SAM2、点提取、选点、Cosmos/Point/FK。D/E复用B/C结果，另有CPU标量比值中位数计算；这里不把D/E说成完整端到端时延。

"""
        + "\n".join(timingtable)
        + """

详见`standalone_single_frame_timing.json`。全片过程中各任务的da3_ms受同卡并发和CPU负载影响，保留供审计，不作为部署延迟。
本轮为真正单帧推理，不能和之前128帧offload批处理折算约250ms直接等同。

## 69帧D435异常复查

Sandwich全手点P90>30mm恰为69帧，复现原基线；1024点选后为33帧，表明选点降低了部分长尾，但没有消除飞点。
已导出并查看全部69帧，见`sandwich/failure_audit/contact_sheet_1.jpg`至`contact_sheet_6.jpg`；逐帧图和数值也保留。
左图橙色轮廓为SAM手mask，青色为URDF投影；右图红色为距离URDF>30mm的D435手点。
目检见大量异常点位于黑手套的内部区域，例如121、126、890、895、900、1114、1119帧，不是仅落在mask边缘或远处背景。
按7×7腐蚀定义的mask内部统计，全部异常点中约 """
        + f"{core:.1%}"
        + """ 位于内部。这个比例是空间位置证据，不是深度错误归因的真值标签。
因此与“手套处深度异常／RGB-depth时差”更一致，不能仅通过缩小mask解决；局部mask漏边、URDF形状和同步误差仍可能同时存在，没有把69帧全部武断归因给传感器。

## 物体

选中物体点的离群比例定义：在该帧选中物体点内做16邻居平均距离，超过max(均值+2std,8mm)的比例；只作为可复现的稀疏／离群描述，不用于删点，不等于物体准确率，也可能把合法的小物体判成离群。
各任务summary提供该比例均值，视频保留所有选中点，未用URDF过滤物体。

## 视频与产物

每任务目录下有四条固定视角完整视频：`<task>_A_to_E_front.mp4`、`side.mp4`、`opposite.mp4`、`overhead.mp4`。
蓝色为URDF手表面，橙色为选中的手点，物体为原RGB颜色。五种方案同一坐标范围，范围包含全片所有选中飞点，不通过裁掉飞点美化。
30FPS容器按记录时间持有到下一实际深度帧（约6Hz），无深度插值、未来补点或自动转动；每任务视频来自全片239／449／423个实际深度帧。
首次RGB画面相机/关节时间对应沿用原录制帧编号，没有另行校准毫秒级曝光偏差。

## 校验与复现

- 数据脚本：`tools/compare_depth_sources.py`；独立延迟：`tools/benchmark_depth_sources_single.py`；视频：`tools/render_depth_sources.py`。
- CPU选点瓶颈用C++标量double实现等价FPS加速，禁止fast-math和FMA，保持候选池、初始点、距离、去重和tie规则；随机／退化／实际输出案例核对见`verification.json`。没有修改训练／部署选点模块。
- 中断恢复只跳过已保存帧，继承上一完成帧s，并重新预热；不把不完整帧纳入统计。
- 三任务在同一A800上并发提高吞吐；视频在CPU并行编码。统计和独立延迟均有原始逐帧记录。
- 每个任务只有一条数据，不能推广为所有物体、光照和位置的结论。最终推荐需同时看尾部飞点、手点覆盖、物体目检和实际部署延迟。
"""
    )
    text += """
## 本次选择建议

- 按任务书主指标（选中手点的逐帧P90中位数），A在三个episode都最低，建议优先以A作为当前深度基线。
- Sandwich的A典型偏移小，但最坏帧P90达249.6mm；C最坏为54.3mm。A仍需要坏深度检测／异常帧处理，不能据此认为直接上线已足够可靠。
- C相对B仅在Sandwich改善，在Dropper和Micro未改善；真实相机条件不保证深度更准。
- D/E对Micro的尺度有明显改善，但其余任务没有稳定收益，尤其Sandwich变差。当前“全mask共同估一个尺度”的版本不建议直接作为统一默认方案。
- 上述结论只针对这三条数据及所用Track4World AnyView权重，未验证其他模型权重、所有物体或完整机器人闭环任务。

## 四视角视频链接

| 任务 | 原视角 | 侧面 | 反向 | 俯视 |
|---|---|---|---|---|
"""
    for task in summary:
        text += (
            "| "
            + task
            + " | "
            + " | ".join(
                f"[{view}]({task}/{task}_A_to_E_{view}.mp4)" for view in ["front", "side", "opposite", "overhead"]
            )
            + " |\n"
        )
    (root / "README.md").write_text(text)
    print("\n".join(table), flush=True)


if __name__ == "__main__":
    main()
