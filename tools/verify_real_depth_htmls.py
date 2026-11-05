"""Validate generated v1/v2 artifacts against cached RGB tracks, without models."""

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

import msgspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "pointflow_outputs/cotracker_real_20261104"
TASKS = ["bottle", "scissors", "micro", "sandwich"]


def verify(task, version):
    name = "v1_offline_da3" if version == "v1" else "v2_d435"
    dest = ROOT / f"pointflow_outputs/cotracker_real_{name}_20261104" / task
    if not (dest / "html_complete.json").exists():
        return None
    prior = dest / "verification.json"
    target = dest / f"{task}_{name}.html"
    if prior.exists():
        cached = json.loads(prior.read_text())
        if cached.get("version") == version and cached.get("html_bytes") == target.stat().st_size:
            print(task, version, "verification already passed", flush=True)
            return cached
    source = json.loads((V0 / task / "source.json").read_text())
    config = json.loads((V0 / task / "task_config.json").read_text())
    expected = "marvin_wuji_d435_dagger.urdf" if task in ["bottle", "scissors"] else "marvin_wuji_d435_complete.urdf"
    assert Path(config["urdf"]).name == expected
    total = source["frames"]
    meta = json.loads((dest / "points_complete.json").read_text())
    k = meta.get("real_intrinsics")
    counts = []
    captures = 0
    assert len(list((dest / "frames").glob("*.npz"))) == total
    for frame in range(total):
        with (
            np.load(dest / "frames" / f"{frame:06d}.npz") as n,
            np.load(V0 / task / "pointflow/frames" / f"{frame:06d}.npz") as old,
        ):
            for field in ["ids", "uv", "labels", "instances"]:
                np.testing.assert_array_equal(n[field], old[field])
            valid = n["valid"]
            p = n["xyz"][valid]
            uv = n["uv"][valid]
            assert np.isfinite(p).all()
            counts.append(int(valid.sum()))
            if version == "v2":
                captures += bool(n["sensor_update"])
                np.testing.assert_array_equal(valid, n["provenance"] > 0)
                if not n["sensor_update"]:
                    assert not (n["provenance"] == 1).any()
                proj = np.column_stack((p[:, 0] / p[:, 2] * k["fx"] + k["ppx"], p[:, 1] / p[:, 2] * k["fy"] + k["ppy"]))
                np.testing.assert_allclose(proj, uv, atol=1e-4)
            else:
                assert float(n["shift"]) == 0
    target = dest / f"{task}_{name}.html"
    s = target.read_text()
    embedded = json.loads(s.split("window.__VISER_EXPORT_METADATA__ = ", 1)[1].split(";\nconst offlineHdri", 1)[0])
    assert embedded["frames"] == total and embedded["urdf"] == config["urdf"]
    if version == "v2":
        assert not embedded["da3_used"] and captures == meta["depth_frames"]
    else:
        assert not embedded["d435_used"] and embedded["uses_future_frames"]
    encoded = re.search(r'<script id="viser-scene-data"[^>]*>(.*?)</script>', s, re.S).group(1)
    recording = msgspec.msgpack.decode(gzip.decompress(base64.b64decode(encoded)))
    assert abs(recording["durationSeconds"] - total / source["fps"]) < 1e-6
    assert len(embedded["frame_ids"]) == len(range(0, total, 2))
    report = dict(
        task=task,
        version=version,
        frames=total,
        actual_depth_frames=captures if version == "v2" else None,
        ids_uv_instances_equal_v0=True,
        finite_valid_xyz=True,
        xyz_reprojects_to_uv=True if version == "v2" else None,
        zero_d435_shift=True if version == "v1" else None,
        urdf=config["urdf"],
        serialized_scene_decoded=True,
        valid_points_min=min(counts),
        valid_points_median=float(np.median(counts)),
        valid_points_max=max(counts),
        html=str(target),
        html_bytes=target.stat().st_size,
    )
    (dest / "verification.json").write_text(json.dumps(report, indent=2))
    print(
        task,
        version,
        total,
        "frames",
        round(target.stat().st_size / 1e6, 1),
        "MB",
        "min/median valid",
        min(counts),
        np.median(counts),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    reports = []
    for task in TASKS:
        for version in ["v1", "v2"]:
            result = verify(task, version)
            if result:
                reports.append(result)
    if args.require_all:
        assert len(reports) == 8, f"Only {len(reports)}/8 completed"
    output = ROOT / "pointflow_outputs/cotracker_real_v1_v2_comparison_20261104"
    output.mkdir(exist_ok=True)
    (output / "verification.json").write_text(json.dumps(reports, indent=2))
    lines = [
        "# 其他四条数据的 v1 / v2 HTML",
        "",
        "v1：旧 Track4World 多帧 DA3（含未来帧），无 D435 校正。v2：纯 D435，空间过滤、历史突跳检测和有限时长补点。两者复用相同 v0 SAM2 mask / CoTracker UV 与 ID。",
        "",
        "| 数据 | 版本 | 帧数 | HTML |",
        "|---|---|---:|---|",
    ]
    for r in reports:
        lines.append(f"| {r['task']} | {r['version']} | {r['frames']} | [{Path(r['html']).name}]({r['html']}) |")
    lines += [
        "",
        "数据和 URDF 配置来自原 v0 source.json / task_config.json：bottle、scissors 使用 dagger URDF；micro、sandwich 使用 complete URDF。",
        "",
        "验证覆盖全片 IDs/UV/实例与 v0 一致、有效 XYZ 有限、v2 投影回原 UV、v1 无 D435 平移、HTML 内嵌场景可解码且时长正确。深度精度仍需观测或独立真值验证，平稳不等于准确。",
        "",
        "各任务目录 points_complete.json / depth_complete.json / chunks/*.json 留有耗时；运行日志分别为两个版本目录中的 remaining_run.log。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
