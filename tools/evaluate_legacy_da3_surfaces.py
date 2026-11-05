"""Reconstruct historical DA3 variants and evaluate against posed URDF meshes.

No inference, calibration refit, new filtering, or FK-joint distance metric.
Primary comparison uses exactly the same 239 frames and up-to-2500 hand UVs.
Native historical support/full frame ranges are reported separately.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import render_urdf_surface_da3_d435 as surface_tools
from scipy.spatial import cKDTree

BASE = Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base")
sys.path.insert(0, str(BASE / "tools"))
import render_fk_vs_pointflow as core  # noqa: E402

EP = "episode_0013_20260731_133649"
OUT = Path("pointflow_outputs/legacy_da3_urdf_errors_20261104")
METHODS = {
    "original_cached": (
        "原始PointFlow（未修正参考）",
        "原始缓存XYZ",
        Path("/data/shichaojian/renders/fk_vs_pointflow") / f"{EP}_fk_vs_pointflow_d435.mp4",
        "DA3原始点云",
    ),
    "historical_B_last": (
        "历史B：真实内参校准＋末段尺度",
        "旧缓存Z×1.099358947872622，真实K反投影",
        BASE / "outputs/fk_vs_pointflow_B" / f"{EP}_fk_vs_pointflow_B_continuous.mp4",
        "B方案",
    ),
    "historical_B_own": (
        "历史B：逐段尺度修正",
        "旧缓存按各段estimated norm与Metric ratio恢复",
        BASE / "outputs/fk_vs_pointflow_scale_fix" / f"{EP}_three_continuous.mp4",
        "第三组：B＋逐段尺度",
    ),
    "rerun_1": (
        "四列第一种：无相机条件＋末段尺度",
        "重跑AnyView，所有段用末段norm恢复",
        BASE / "outputs/fk_vs_pointflow_camera_conditions" / f"{EP}_four_continuous.mp4",
        "第一列",
    ),
    "rerun_2": (
        "四列第二种：无相机条件＋自身尺度",
        "重跑AnyView，使用各段自身米制几何",
        BASE / "outputs/fk_vs_pointflow_camera_conditions" / f"{EP}_four_continuous.mp4",
        "第二列",
    ),
    "rerun_3": (
        "四列第三种：相机条件＋末段尺度",
        "真实K＋单位位姿输入网络，末段恢复",
        BASE / "outputs/fk_vs_pointflow_camera_conditions" / f"{EP}_four_continuous.mp4",
        "第三列",
    ),
    "rerun_4": (
        "四列第四种：相机条件＋自身尺度",
        "真实K＋单位位姿输入网络，各段自身米制几何",
        BASE / "outputs/fk_vs_pointflow_camera_conditions" / f"{EP}_four_continuous.mp4",
        "第四列",
    ),
    "core_Z_translation": (
        "D435核心区域：Z方向平移",
        "四列第一种＋历史保存的[0,0,delta]",
        BASE / "outputs/da3_d435_core_calibration" / f"{EP}_da3_sensor_calibration_comparison.mp4",
        "左侧校正后曲线／点云",
    ),
    "core_ray_scale": (
        "D435核心区域：沿视线缩放",
        "四列第一种×历史保存的scale",
        BASE / "outputs/da3_d435_core_calibration" / f"{EP}_da3_sensor_calibration_comparison.mp4",
        "右侧校正后曲线／点云",
    ),
}


def metric(points, tree):
    if not len(points):
        raise ValueError("Empty historical hand observation")
    d, nearest = tree.query(points)
    d = d * 1000
    return dict(
        points=len(points),
        mean_mm=float(d.mean()),
        median_mm=float(np.median(d)),
        p90_mm=float(np.percentile(d, 90)),
        p95_mm=float(np.percentile(d, 95)),
        within20mm=float(np.mean(d <= 20)),
    )


def aggregate(rows):
    result = {}
    for method in METHODS:
        selected = [r["methods"][method] for r in rows if method in r["methods"]]
        result[method] = dict(
            frames=len(selected),
            mean_frame_mean_mm=float(np.mean([r["mean_mm"] for r in selected])),
            mean_frame_median_mm=float(np.mean([r["median_mm"] for r in selected])),
            median_frame_median_mm=float(np.median([r["median_mm"] for r in selected])),
            mean_frame_p90_mm=float(np.mean([r["p90_mm"] for r in selected])),
            median_frame_p90_mm=float(np.median([r["p90_mm"] for r in selected])),
            max_frame_p90_mm=float(np.max([r["p90_mm"] for r in selected])),
            frames_p90_gt30=sum(r["p90_mm"] > 30 for r in selected),
            mean_within20mm=float(np.mean([r["within20mm"] for r in selected])),
        )
    return result


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    work = BASE / "outputs/.da3_camera_four_work"
    b_report = json.loads((BASE / "outputs/fk_vs_pointflow_B/scale_report.json").read_text())
    chunks = json.loads((BASE / "outputs/fk_vs_pointflow_scale_fix/chunks_all.json").read_text())["chunks"]
    anchors_report = json.loads((BASE / "outputs/da3_d435_core_calibration/calibration.json").read_text())
    anchors = {int(r["frame"]): r for r in anchors_report["anchors"]}
    common_frames = set(anchors)
    assert len(common_frames) == 239
    obs = core.load_pointflow(EP)
    K = core.colour_K(core.colour_intrinsics(EP))
    rotation, translation = core.base_to_camera()
    fk, _ = core.load_fk(EP, rotation, translation)
    hand = surface_tools.HandSurface(EP, 0.0015, seed=0)
    surface_store = np.lib.format.open_memmap(
        OUT / "urdf_surface_camera.npy", mode="w+", dtype=np.float32, shape=(len(fk), len(hand.pts), 3)
    )
    last_original = max(chunks.values(), key=lambda r: r["start"])["norms"]["estimated"]
    last_rerun = {}
    for mode in ["unconditioned", "conditioned"]:
        with np.load(sorted(work.glob(mode + "_*.npz"))[-1]) as n:
            last_rerun[mode] = float(n["norm"])

    def unproject(uv, z):
        u, v = core.canvas_uv_to_colour(uv)
        return np.column_stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z))

    common_rows = []
    native_rows = []
    paired_full_rows = []
    files = {}
    maximum_fk_error = 0.0
    maximum_previous_surface_difference = 0.0
    previous_surface = np.load(
        "pointflow_outputs/da3_urdf_two_conditions_20261103/urdf_surface_camera.npy", mmap_mode="r"
    )
    max_native_reconstruction_difference = 0.0
    validation_points = np.load(BASE / "outputs/fk_vs_pointflow_scale_fix/comparison_points.npz")
    for frame in range(len(fk)):
        surface = hand.surface_base(frame) @ rotation.T + translation
        surface_store[frame] = surface
        reconstructed_fk = hand.keypoints_base(frame) @ rotation.T + translation
        maximum_fk_error = max(maximum_fk_error, float(np.max(np.abs(reconstructed_fk - fk[frame]))))
        assert maximum_fk_error < 1e-6
        maximum_previous_surface_difference = max(
            maximum_previous_surface_difference, float(np.max(np.abs(surface - previous_surface[frame])))
        )
        tree = cKDTree(surface)
        points, uv = core.hand_observations(obs, frame)
        take = np.linspace(0, len(uv) - 1, min(2500, len(uv))).astype(int)
        native = {"original_cached": points}
        native["historical_B_last"] = unproject(uv, points[:, 2] * b_report["ratio_B_over_original"])
        chunk = chunks[str(frame // 128 * 128)]
        factor = (chunk["norms"]["estimated"] + 1e-6) / last_original * chunk["metric_ratio"]
        native["historical_B_own"] = unproject(uv, points[:, 2] * factor)
        paired = {key: value[take] for key, value in native.items()}
        for mode in ["unconditioned", "conditioned"]:
            start = frame // 128 * 128
            if (mode, start) not in files:
                for key in list(files):
                    if key[0] == mode:
                        files.pop(key).close()
                files[(mode, start)] = np.load(work / f"{mode}_{start:04d}.npz")
            n = files[(mode, start)]
            index = frame - start
            np.testing.assert_array_equal(n[f"uv_{index}"], uv[take].astype(np.float32))
            p = unproject(n[f"uv_{index}"].astype(float), n[f"z_{index}"].astype(float))
            legacy = p * (last_rerun[mode] / (float(n["norm"]) + 1e-6))
            names = ("rerun_1", "rerun_2") if mode == "unconditioned" else ("rerun_3", "rerun_4")
            for name, value in zip(names, [legacy, p]):
                native[name] = value
                paired[name] = value
        if frame in anchors:
            info = anchors[frame]
            delta = float(info.get("offset_m", 0)) if info["accepted"] else 0.0
            scale = float(info.get("scale", 1)) if info["accepted"] else 1.0
            for name, value in [
                ("core_Z_translation", paired["rerun_1"] + [0, 0, delta]),
                ("core_ray_scale", paired["rerun_1"] * scale),
            ]:
                paired[name] = value
                native[name] = value
        for key, p in [
            ("original", native["original_cached"]),
            ("B", native["historical_B_last"]),
            ("corrected", native["historical_B_own"]),
        ]:
            stored = f"{key}_{frame}"
            if stored in validation_points.files:
                difference = float(np.max(np.abs(validation_points[stored] - p)))
                assert difference < 1e-6, (stored, difference)
                max_native_reconstruction_difference = max(max_native_reconstruction_difference, difference)
        native_stats = {key: metric(value, tree) for key, value in native.items()}
        pair_stats = {
            key: (native_stats[key] if key.startswith("rerun") or key.startswith("core") else metric(value, tree))
            for key, value in paired.items()
        }
        native_rows.append(dict(frame=frame, methods=native_stats))
        paired_full_rows.append(dict(frame=frame, methods=pair_stats))
        if frame in common_frames:
            assert all(len(p) == len(paired["rerun_1"]) for p in paired.values())
            common_rows.append(dict(frame=frame, methods=pair_stats))
        if frame % 100 == 0:
            print("FRAME", frame, "/", len(fk), flush=True)
    surface_store.flush()
    for n in files.values():
        n.close()
    for name, rows in [
        ("common239_same_uv", common_rows),
        ("native_support", native_rows),
        ("same_uv_full_available", paired_full_rows),
    ]:
        (OUT / f"{name}_per_frame.json").write_text(json.dumps(rows, indent=2))
        (OUT / f"{name}_summary.json").write_text(json.dumps(aggregate(rows), indent=2))
    methods = {
        key: dict(name=value[0], method=value[1], video=str(value[2]), video_position=value[3])
        for key, value in METHODS.items()
    }
    for value in methods.values():
        assert Path(value["video"]).exists(), value["video"]
    (OUT / "methods_and_videos.json").write_text(json.dumps(methods, ensure_ascii=False, indent=2))
    verification = dict(
        episode=EP,
        urdf=str(core.URDF),
        urdf_md5=hashlib.md5(core.URDF.read_bytes()).hexdigest(),
        surface_points_per_frame=len(hand.pts),
        surface_sample_spacing_m=0.0015,
        seed=0,
        frames=len(fk),
        common_frames=len(common_frames),
        urdf_rebuilt_from_qpos=True,
        max_fk_keypoint_reconstruction_error_m=maximum_fk_error,
        max_difference_vs_previous_urdf_surface_m=maximum_previous_surface_difference,
        max_reconstruction_difference_vs_historical_clouds_m=max_native_reconstruction_difference,
        all_shared_hand_uvs_exactly_match=True,
        distance_target="all URDF hand visual mesh surface samples, not FK21",
        fitting=False,
        calibration_reestimated=False,
        accepted_core_frames=anchors_report["accepted_frames"],
    )
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2))
    with (OUT / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method_id",
                "method",
                "frames",
                "mean_frame_mean_mm",
                "mean_frame_median_mm",
                "mean_frame_p90_mm",
                "max_frame_p90_mm",
                "frames_p90_gt30",
                "video",
                "video_position",
            ]
        )
        for method, row in aggregate(common_rows).items():
            writer.writerow(
                [method, methods[method]["name"]]
                + [
                    row[k]
                    for k in [
                        "frames",
                        "mean_frame_mean_mm",
                        "mean_frame_median_mm",
                        "mean_frame_p90_mm",
                        "max_frame_p90_mm",
                        "frames_p90_gt30",
                    ]
                ]
                + [methods[method]["video"], methods[method]["video_position"]]
            )
    print("COMPLETE", json.dumps(aggregate(common_rows)), flush=True)


if __name__ == "__main__":
    main()
