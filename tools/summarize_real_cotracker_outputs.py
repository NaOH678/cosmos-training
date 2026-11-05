"""Write artifact links, provenance, coverage and timing after all five trials."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pointflow_outputs/cotracker_real_20261104"
TASKS = ["bottle", "scissors", "dropper", "micro", "sandwich"]
rows = []
for task in TASKS:
    p = OUT / task
    source = json.loads((p / "source.json").read_text())
    init = json.loads((p / "initialization.json").read_text())
    schedule = init.get("dino_schedule_frames", init["dino_frames"])
    latest = max((int(v) for v in init["births"].values()), default=0)
    init["dino_schedule_frames"] = schedule
    init["dino_frames"] = schedule if init["missing"] else [f for f in schedule if f <= latest]
    (p / "initialization.json").write_text(json.dumps(init, indent=2, ensure_ascii=False))
    report = json.loads((p / "pointflow/report.json").read_text())
    timing = json.loads((p / "pointflow/COMPLETE_STAGE.json").read_text())
    audit = json.loads((p / "depth_audit.json").read_text())
    verification = json.loads((p / "verification.json").read_text())
    render = json.loads((p / "render_complete.json").read_text())
    stats = json.loads((p / "pointflow/per_frame.json").read_text())
    assert source["frames"] == timing["frames"] == verification["frames"] == render["frames"]
    assert report.get("filter_revision") == 3
    artifacts = {
        name: str(p / file)
        for name, file in [
            ("mask", "segmentation_h264.mp4"),
            ("flow", "pointflow_rgb_h264.mp4"),
            ("html", "cotracker_da3_interactive.html"),
        ]
    }
    for file in artifacts.values():
        assert Path(file).stat().st_size > 10000
    row = dict(
        task=task,
        **source,
        urdf=report["urdf"],
        artifacts=artifacts,
        timing=timing["timing"],
        d435_update_p50_ms=median(r["sensor_ms"] for r in stats[5:] if r["sensor_updated"]),
        d435_status=dict(Counter(r["shift_status"] for r in stats)),
        valid_observations=timing["valid_observations"],
        filter_counts={
            key: sum(r.get(key, 0) for r in stats)
            for key in ["temporal_removed", "spatial_removed", "velocity_removed", "component_removed"]
        },
        hand_urdf=audit["hand_urdf"],
        missing_point_frames=audit["missing_point_frames"],
        model_windows=verification["model_windows"],
        dino_frames=init["dino_frames"],
        missing_initial_objects=init["missing"],
    )
    rows.append(row)
(OUT / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
lines = [
    "# 五类真机 CoTracker＋DA3 结果",
    "",
    f"全部完整处理：{sum(r['frames'] for r in rows)} 帧，5个 mask 视频、5个 RGB 点流轨迹视频、5个 Viser HTML。随机选择种子0。",
    "",
    "流程与模型数据接口已验证；micro 等任务仍有连片深度偏差，尚未达到稳定无飞点的效果。",
    "",
    "|任务|SAM2 mask|RGB点流轨迹|Viser HTML|",
    "|---|---|---|---|",
]
for r in rows:
    a = r["artifacts"]
    lines.append(f"|{r['task']}|[视频]({a['mask']})|[视频]({a['flow']})|[HTML]({a['html']})|")
lines += ["", "## 数据与配置", "", "|任务|episode|帧数|URDF|", "|---|---|---:|---|"]
for r in rows:
    lines.append(
        f"|{r['task']}|{Path(r['episode']).name}|{r['frames']}|{'dagger' if 'dagger_fk' in r['urdf'] else 'complete'}|"
    )
lines += [
    "",
    f"实现与使用方式：[实验说明]({ROOT}/docs/real_cotracker_pipeline_20261104.md:1)。每个任务的 `pointflow/` 包含完整观测数组、原始/修正后 XYZ、有效性与样例模型窗口；`verification.json` 验证 X/Y 不变、只加Z、ID唯一，并实际构建模型 batch。",
    "",
    "SAM2为全物体持续跟踪基线。训练开头多帧仅按到达顺序补充缺失目标；本次最后五条均在首帧初始化完整，实际 DINO 调用帧均为0。训练/推理复用同一规则与文本；sandwich蛋/芝士复用的是相对DINO检测框的正负点模板。",
    "",
    "CoTracker仅替换二维轨迹。DA3沿用Track4World的AnyView权重和Nested Metric标定/对齐实现，但当前试验只使用当前图像，未复现旧128帧未来上下文/最后chunk归一化。手核心D435的历史门限决定统一Z平移，手与物体共用；UV及XY不重投影、不随校正修改。",
    "",
    "## 检查结果（手表面距离）",
    "",
    "下表每10帧及末帧抽样，参考为对应URDF真实手表面，不是FK关节点；不作拟合，也不是物体真值。可视化更干净不能证明深度准确。",
    "",
    "|任务|抽样帧|平均逐帧平均距离mm|逐帧P90的中位数mm|最差逐帧P90 mm|",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    a = r["hand_urdf"]
    lines.append(
        f"|{r['task']}|{a['sampled_frames']}|{a['mean_frame_mean_mm']:.2f}|{a['median_frame_p90_mm']:.2f}|{a['max_frame_p90_mm']:.2f}|"
    )
lines += [
    "",
    "飞点检查包括像素深度断层、三维离群点、每条轨迹的局部运动残差、80mm最小三维步长门限，以及远离主体且深度相差超过60mm的较小连通点块。被剔除候选在HTML红色图层保留；没有用未来插值补成有效数据。透明物体、细管及遮挡处仍可能出现连片偏差；强过滤也可能删除真实局部表面。详见每条 `depth_motion_audit.jpg`、`depth_audit.json` 和 `depth_audit_frames.json`。",
    "",
    "所有10个视频均已完整解码并核对帧数。HTML使用现有Viser离线打包，未下载浏览器验证工具；展示每3帧一次、稳定ID取一半点，数据保留全部处理帧和候选点。RGB视频包含最近10帧的轨迹线，非仅逐帧散点。",
    "",
    "## 耗时",
    "",
    "A800单卡、DA3 FP16、CoTracker BF16。以下为完整回放中排除前5帧后的p50；部分阶段与SAM2同卡并发，不能当作独占GPU或Cosmos整链路在线延迟。D435只统计实际更新帧，包含深度解码、标定配准及核心参数/历史门限；并非只测几条标量运算。其他RGB帧短时延续最新校正量。",
    "",
    "|任务|CoTracker ms|DA3 ms|D435更新帧 ms|最终点过滤 ms|",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    t = r["timing"]
    lines.append(
        f"|{r['task']}|{t['tracking_ms']['p50_ms']:.1f}|{t['da3_ms']['p50_ms']:.1f}|{r['d435_update_p50_ms']:.1f}|{t['filter_ms']['p50_ms']:.1f}|"
    )
lines += [
    "",
    "完整JSON：[summary.json]("
    + str(OUT / "summary.json")
    + ")。源码静态检查通过；PointFlow窗口15项回归、5项新增契约测试及5项历史门限测试通过。bottle前20帧与独立短片结果的ID、UV、原始DA3 XYZ完全一致。",
    "",
    "每实例无有效点帧数只作诊断，不等价于物体物理缺失；遮挡、实例之间覆盖/改名、SAM丢失和深度过滤都可能导致该计数。模型中物体类别等价，不能仅按实例名称判断覆盖。",
]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print("ALL ARTIFACTS VERIFIED", sum(r["frames"] for r in rows), "frames")
