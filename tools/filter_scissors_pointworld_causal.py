"""PointWorld spatial rules plus explicitly causal history/fusion; reuse cached stereo."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pointflow_outputs/scissors_foundationstereo_20261105"
OUT = ROOT / "pointflow_outputs/scissors_foundationstereo_pointworld_causal_20261105"


def noise_mask(xyz, eps_values=(0.02, 0.05), min_points=5):
    """Exact DBSCAN noise membership via core support; no cluster labels needed."""
    if not len(xyz):
        return np.zeros(0, bool)
    tree = cKDTree(xyz)
    noise = np.zeros(len(xyz), bool)
    for eps in eps_values:
        core = tree.query_ball_point(xyz, eps, return_length=True) >= min_points
        if not core.any():
            noise[:] = True
        else:
            distance = cKDTree(xyz[core]).query(xyz, workers=1)[0]
            noise |= distance > eps
    return noise


def run(limit=None, destination=OUT):
    destination.mkdir(exist_ok=True)
    (destination / "frames").mkdir(exist_ok=True)
    for name in ["source.json", "task_config.json", "video.json"]:
        (destination / name).write_bytes((SRC / name).read_bytes())
    history = {}
    anchors = {}
    seen_candidates = set()
    rows = []
    state_instances = {}
    K = json.loads((SRC / "points_complete.json").read_text())["real_intrinsics"]
    all_paths = sorted((SRC / "frames").glob("*.npz"))
    for frame, path in enumerate(all_paths[:limit]):
        with np.load(path) as n:
            d = {k: n[k] for k in n.files}
        start = time.perf_counter()
        ids, inst, uv = d["ids"], d["instances"], d["uv"]
        now = float(d["timestamp"])
        update = bool(d["sensor_update"])
        candidate = d["valid"].copy() & np.isfinite(d["xyz"]).all(1)
        input_valid = int(candidate.sum())
        xyz = d["xyz"].copy()
        spatial = np.zeros(len(ids), bool)
        temporal = np.zeros(len(ids), bool)
        fused = np.zeros(len(ids), bool)
        repaired = np.zeros(len(ids), bool)
        if update:
            good = np.flatnonzero(candidate)
            spatial[good] = noise_mask(xyz[good])
            candidate &= ~spatial
            for i in np.flatnonzero(spatial):
                history.pop(int(ids[i]), None)
            for instance in np.unique(inst):
                group = np.flatnonzero((inst == instance) & candidate)
                paired = [i for i in group if int(ids[i]) in history and now - history[int(ids[i])]["time"] <= 0.2]
                motion, common_dz = 1e6, 0.0
                if len(paired) >= 5:
                    motion = float(np.median([np.linalg.norm(uv[i] - history[int(ids[i])]["uv"]) for i in paired]))
                    common_dz = float(np.median([xyz[i, 2] - history[int(ids[i])]["measured_z"] for i in paired]))
                old_instance = state_instances.get(int(instance))
                # Reject coherent depth teleports too; a normal moving surface remains supported.
                if old_instance and len(group) >= 5 and motion < 5 and abs(common_dz) > 0.08:
                    temporal[group] = True
                stationary = motion < 0.8 and abs(common_dz) < 0.015
                for i in group:
                    key = int(ids[i])
                    old = history.get(key)
                    if old and now - old["time"] <= 0.2:
                        residual = xyz[i, 2] - old["measured_z"] - common_dz
                        if abs(residual) > 0.03:
                            temporal[i] = True
                    if temporal[i]:
                        candidate[i] = False
                        # No filling at currently unsupported or spatially rejected locations.
                        # Current support exists here; short Z hold uses only previous accepted data.
                        if old and now - old["time"] <= 0.10:
                            xyz[i, 2] = old["z"]
                            candidate[i] = True
                            repaired[i] = True
                        continue
                    raw_z = float(xyz[i, 2])
                    if old and now - old["time"] <= 0.2:
                        # Causal extension, not part of PointWorld original: static surface fusion.
                        alpha = 0.12 if stationary else 0.7
                        delta = raw_z - old["z"]
                        xyz[i, 2] = old["z"] if stationary and abs(delta) < 0.002 else old["z"] + alpha * delta
                        fused[i] = True
                    history[key] = dict(z=float(xyz[i, 2]), measured_z=raw_z, uv=uv[i].copy(), time=now)
                if len(group) >= 5:
                    state_instances[int(instance)] = True
        else:
            # Consume accepted history, never the baseline's unfiltered intermediate values.
            candidate[:] = False
            for i, key in enumerate(ids):
                old = history.get(int(key))
                if old and now - old["time"] <= 0.10:
                    xyz[i, 2] = old["z"]
                    candidate[i] = True
                    repaired[i] = True
        history = {key: value for key, value in history.items() if now - value["time"] <= 0.2}
        # Preserve UV exactly; re-lift XY using the same intrinsics after filtering Z.
        xyz[:, 0] = (uv[:, 0] - K["ppx"]) / K["fx"] * xyz[:, 2]
        xyz[:, 1] = (uv[:, 1] - K["ppy"]) / K["fy"] * xyz[:, 2]
        pre_voxel = candidate.copy()
        # First-frame representative IDs persist. Newly visible tracks enter only free voxels.
        present = {int(key) for key in ids}
        anchors = {key: stamp for key, stamp in anchors.items() if key in present or now - stamp <= 0.2}
        selected = np.array([int(key) in anchors for key in ids]) & candidate
        occupied = {tuple(q) for q in np.floor(xyz[selected] / 0.015).astype(np.int64)}
        for i in np.flatnonzero(candidate & ~selected):
            if int(ids[i]) in seen_candidates:
                continue
            voxel = tuple(np.floor(xyz[i] / 0.015).astype(np.int64))
            if voxel not in occupied:
                selected[i] = True
                occupied.add(voxel)
                anchors[int(ids[i])] = now
        seen_candidates.update(int(ids[i]) for i in np.flatnonzero(candidate))
        for i in np.flatnonzero(selected):
            anchors[int(ids[i])] = now
        d["xyz"] = xyz
        d["valid"] = selected
        d["pre_voxel_valid"] = pre_voxel
        d["spatial_rejected"] = spatial
        d["temporal_rejected"] = temporal
        d["provenance"] = np.where(selected, np.where(repaired, 2, np.where(fused, 4, 1)), 0).astype(np.uint8)
        filter_ms = (time.perf_counter() - start) * 1000
        np.savez_compressed(destination / "frames" / path.name, **d)
        rows.append(
            dict(
                frame=frame,
                sensor_update=update,
                input_valid=input_valid,
                spatial_rejected=int(spatial.sum()),
                temporal_rejected=int(temporal.sum()),
                before_voxel=int(pre_voxel.sum()),
                measured=int((selected & ~repaired).sum()),
                repaired=int((selected & repaired).sum()),
                interframe=0,
                selected=int(selected.sum()),
                filter_ms=filter_ms,
            )
        )
        if frame % 100 == 0:
            print(
                "FILTER",
                frame,
                "selected",
                int(selected.sum()),
                "spatial",
                int(spatial.sum()),
                "temporal",
                int(temporal.sum()),
                flush=True,
            )
    (destination / "per_frame.json").write_text(json.dumps(rows, indent=2))
    real = [r["filter_ms"] for r in rows if r["sensor_update"]]
    summary = dict(
        frames=len(rows),
        uses_future_frames=False,
        depth_model_rerun=False,
        grid_m=0.015,
        dbscan_eps_m=[0.02, 0.05],
        dbscan_min_points=5,
        real_update_filter_p50_ms=float(np.median(real)),
        real_update_filter_p95_ms=float(np.percentile(real, 95)),
        selected_points_p50=float(np.median([r["selected"] for r in rows])),
        notes="PointWorld spatial DBSCAN + persistent voxel IDs; causal history/Z fusion is an adaptation, not exact offline PointWorld.",
    )
    (destination / "filter_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


def export_html():
    import run_dropper_d435_v2 as export

    export.TASK = "scissors"
    export.VERSION = "foundationstereo_pointworld_causal"
    export.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
    export.OUT = OUT
    legend = '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:350px;font:13px/1.5 system-ui"><b>Scissors · PointWorld规则＋因果稳定</b><br>01：过滤及体素选点后；02：原始深度点；03：历史融合/短时沿用；04：真实URDF手表面；05：上一版未过滤FoundationStereo。<br>PointWorld：2/5cm空间DBSCAN，至少5点，1.5cm首帧体素代表ID。新显露点仅进入空体素。<br>因果改写：仅历史深度跳变检查、静止融合、最多0.1秒深度沿用；不用未来帧，不重新运行深度。缺少可信观测就隐藏，不长期冻结。<br>蓝色URDF，橙色手点，物体RGB；可旋转、暂停、拖时间轴。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
    export.html(
        comparison=SRC,
        show_urdf=True,
        surface_stride=4,
        depth_label="FoundationStereo",
        legend_text=legend,
        display_hold_steps=0,
        highlight_interframe=True,
        comparison_label="05_unfiltered_FoundationStereo",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["filter", "html", "all"])
    args = parser.parse_args()
    if args.stage in ["filter", "all"]:
        run()
    if args.stage in ["html", "all"]:
        export_html()
