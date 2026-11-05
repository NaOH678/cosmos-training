"""Apply the earlier D435 V3 filters to cached metric FoundationStereo depth."""

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import run_dropper_d435_v2 as old
from filter_scissors_pointworld_causal import noise_mask

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pointflow_outputs/scissors_foundationstereo_20261105"
PREVIOUS = ROOT / "pointflow_outputs/scissors_foundationstereo_pointworld_causal_20261105"
OUT = ROOT / "pointflow_outputs/scissors_foundationstereo_d435_filters_20261105"
V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
Fusion = old.load_module(
    "fs_surface_fusion", ROOT / "cosmos_framework/inference/robot_policy/causal_surface_fusion.py"
).CausalSurfaceFusion


class BoundaryFusion(Fusion):
    """V3 contact gate plus explicit mask-edge depth-discontinuity gate."""

    def gate_buffer(self, sensor):
        result = super().gate_buffer(sensor)
        inside_distance = np.zeros(self.labels.shape, np.float32)
        for instance in np.unique(self.instances):
            if instance:
                mask = (self.instances == instance).astype(np.uint8)
                distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                inside_distance[mask > 0] = distance[mask > 0]
        # Do not uniformly erode thin geometry: require a measured depth break too.
        self.hard_pixels = (inside_distance <= 2) & (self.instances > 0) & self.edge
        self.hard_pixels |= self.near_hand & self.edge & (self.labels == 3)
        result[self.hard_pixels] = np.inf
        self.boundary_removed = int((self.hard_pixels & np.isfinite(sensor)).sum())
        return result

    def blocked(self, uv, labels):
        q = np.rint(uv).astype(int)
        q[:, 0] = q[:, 0].clip(0, self.labels.shape[1] - 1)
        q[:, 1] = q[:, 1].clip(0, self.labels.shape[0] - 1)
        return self.hard_pixels[q[:, 1], q[:, 0]] if hasattr(self, "hard_pixels") else np.zeros(len(q), bool)


def process(limit=None, destination=OUT):
    destination.mkdir(exist_ok=True)
    (destination / "frames").mkdir(exist_ok=True)
    for name in ["source.json", "task_config.json", "video.json"]:
        (destination / name).write_bytes((SRC / name).read_bytes())
    calibration = json.loads((SRC / "calibration.json").read_text())
    streams = copy.deepcopy(calibration["streams"])
    # FoundationStereo depth is left-IR optical Z in meters, NOT SDK depth units.
    streams["depth"] = copy.deepcopy(streams["infrared_left"])
    streams["depth"]["depth_scale_m"] = 1.0
    K = json.loads((SRC / "points_complete.json").read_text())["real_intrinsics"]
    pairs = json.loads((SRC / "ir_pair_index.json").read_text())
    by_frame = {}
    for pair in pairs:
        by_frame[pair["available_rgb_step"]] = pair
    maps = np.load(V0 / "instances.npy", mmap_mode="r")
    semantics = np.load(V0 / "labels.npy", mmap_mode="r")
    stabilizer = old.Stabilizer(old.Guard)
    geometry = old.Geometry()
    fusion = BoundaryFusion()
    anchors = {}
    seen = set()
    rows = []
    count = json.loads((SRC / "source.json").read_text())["frames"]
    for frame in range(min(count, limit or count)):
        with np.load(SRC / "frames" / f"{frame:06d}.npz") as n:
            d = {
                k: n[k] for k in ["ids", "uv", "labels", "instances", "colors", "timestamp", "sensor_update", "raw_xyz"]
            }
        uv, ids, labels, instances = d["uv"], d["ids"], d["labels"], d["instances"]
        t = float(d["timestamp"])
        update = bool(d["sensor_update"])
        image = cv2.imread(str(V0 / "frames" / f"{frame:05d}.jpg"))
        imap = cv2.resize(maps[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        semantic = cv2.resize(semantics[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        tic = time.perf_counter()
        fusion.prepare_frame(semantic, imap, image)
        z = np.full(len(ids), np.nan, np.float32)
        valid = np.zeros(len(ids), bool)
        cores = {}
        hard = np.zeros(len(ids), bool)
        geometry_stats = {}
        if update:
            native = np.load(SRC / "native_depth" / f"{by_frame[frame]['sequence']:06d}.npy")
            points, _, _, _, raw, kept, pix = old.measured_cloud(
                native, streams, maps[frame], semantics[frame], image, "cpu"
            )
            kept &= ~fusion.occlusion(points, pix, native, streams)
            buffer = np.full((480, 640), np.inf, np.float32)
            np.minimum.at(buffer, (pix[kept, 1], pix[kept, 0]), points[kept, 2].astype(np.float32))
            buffer = cv2.resize(buffer, (640, 448), interpolation=cv2.INTER_NEAREST)
            buffer = fusion.gate_buffer(buffer)
            z = old.sample_same_instance(uv, instances, buffer, imap, radius=1.5)
            cores = old.core_depths(buffer, imap, semantic)
            valid = np.isfinite(z)
            xyz = np.column_stack(((uv[:, 0] - K["ppx"]) / K["fx"] * z, (uv[:, 1] - K["ppy"]) / K["fy"] * z, z)).astype(
                np.float32
            )
            before = valid.copy()
            valid, geometry_stats = geometry.apply(ids, xyz, valid, instances, t)
            hard |= before & ~valid
        blocked = fusion.blocked(uv, labels)
        q = np.rint(uv).astype(int)
        q[:, 0] = q[:, 0].clip(0, 639)
        q[:, 1] = q[:, 1].clip(0, 447)
        mismatch = imap[q[:, 1], q[:, 0]] != instances
        hard |= blocked | mismatch
        # On a real observation, an unsupported sample must not be recreated from history.
        if update:
            hard |= ~np.isfinite(z)
        valid &= ~hard
        for i in np.flatnonzero(hard):
            stabilizer.samples.pop(int(ids[i]), None)
            cell = np.floor(uv[i] / 2).astype(int)
            fusion.cells.pop((int(instances[i]), int(cell[1] * 320 + cell[0])), None)
        out, p, stats = stabilizer.update(ids, instances, z, valid, t, update, cores)
        before_surface = p.copy()
        out, p, extra = fusion.update(ids, uv, instances, labels, z, out, p, t, update, hard)
        p[hard] = 0
        out[hard] = np.nan
        # Never retain a surface-rejected measurement in trusted history.
        for i in np.flatnonzero((before_surface == 1) & (p == 0)):
            stabilizer.samples.pop(int(ids[i]), None)
        trusted = p == 1
        if update:
            geometry.last_ids = ids[trusted].copy()
            geometry.last_points = np.column_stack(
                (
                    (uv[trusted, 0] - K["ppx"]) / K["fx"] * z[trusted],
                    (uv[trusted, 1] - K["ppy"]) / K["fy"] * z[trusted],
                    z[trusted],
                )
            ).astype(np.float32)
        xyz = np.column_stack(
            ((uv[:, 0] - K["ppx"]) / K["fx"] * out, (uv[:, 1] - K["ppy"]) / K["fy"] * out, out)
        ).astype(np.float32)
        accepted = (p > 0) & np.isfinite(xyz).all(1)
        spatial = np.zeros(len(ids), bool)
        if update:
            a = np.flatnonzero(accepted)
            spatial[a] = noise_mask(xyz[a])
            accepted &= ~spatial
            for i in np.flatnonzero(spatial):
                stabilizer.samples.pop(int(ids[i]), None)
                cell = np.floor(uv[i] / 2).astype(int)
                fusion.cells.pop((int(instances[i]), int(cell[1] * 320 + cell[0])), None)
        p[~accepted] = 0
        xyz[~accepted] = np.nan
        if update:
            trusted = p == 1
            geometry.last_ids = ids[trusted].copy()
            geometry.last_points = np.column_stack(
                (
                    (uv[trusted, 0] - K["ppx"]) / K["fx"] * z[trusted],
                    (uv[trusted, 1] - K["ppy"]) / K["fy"] * z[trusted],
                    z[trusted],
                )
            ).astype(np.float32)
        # Same persistent voxel sampling as previous version, without double temporal smoothing.
        present = set(map(int, ids))
        anchors = {key: stamp for key, stamp in anchors.items() if key in present or t - stamp <= 0.2}
        selected = np.array([int(i) in anchors for i in ids]) & accepted
        occupied = {tuple(v) for v in np.floor(xyz[selected] / 0.015).astype(np.int64)}
        for i in np.flatnonzero(accepted & ~selected):
            if int(ids[i]) in seen:
                continue
            voxel = tuple(np.floor(xyz[i] / 0.015).astype(np.int64))
            if voxel not in occupied:
                selected[i] = True
                occupied.add(voxel)
                anchors[int(ids[i])] = t
        seen.update(map(int, ids[accepted]))
        for key in ids[selected]:
            anchors[int(key)] = t
        filter_ms = (time.perf_counter() - tic) * 1000
        d.update(
            xyz=xyz,
            valid=selected,
            pre_voxel_valid=accepted,
            provenance=np.where(selected, p, 0).astype(np.uint8),
            hard_blocked=hard,
            spatial_rejected=spatial,
        )
        np.savez_compressed(destination / "frames" / f"{frame:06d}.npz", **d)
        rows.append(
            dict(
                frame=frame,
                sensor_update=update,
                filter_ms=filter_ms,
                measured=int((selected & (p == 1)).sum()),
                repaired=int((selected & ((p == 2) | (p == 4))).sum()),
                interframe=int((selected & (p == 3)).sum()),
                hard_blocked=int(hard.sum()),
                spatial_rejected=int(spatial.sum()),
                before_voxel=int(accepted.sum()),
                selected=int(selected.sum()),
                geometry=geometry_stats,
                **extra,
            )
        )
        if frame % 100 == 0:
            print("FILTER", frame, "selected", int(selected.sum()), "hard", int(hard.sum()), flush=True)
    old.save(destination / "per_frame.json", rows)
    real = [r["filter_ms"] for r in rows if r["sensor_update"]]
    old.save(
        destination / "filter_summary.json",
        dict(
            frames=len(rows),
            uses_future_frames=False,
            depth_model_rerun=False,
            depth_source="FoundationStereo left IR metric Z",
            d435_sdk_depth_used=False,
            filter_p50_ms=float(np.median(real)),
            filter_p95_ms=float(np.percentile(real, 95)),
            selected_p50=float(np.median([r["selected"] for r in rows])),
        ),
    )


def strengthen_static(destination=OUT, limit=None):
    """Preserve strong static fusion after V3 gates, without feeding sensor guards."""
    state = {}
    modes = {}
    k = json.loads((SRC / "points_complete.json").read_text())["real_intrinsics"]
    timings = []
    for path in sorted((destination / "frames").glob("*.npz"))[:limit]:
        with np.load(path) as n:
            d = {key: n[key] for key in n.files}
        tic = time.perf_counter()
        base = d.get("before_static_xyz", d["xyz"]).copy()
        xyz = base.copy()
        t = float(d["timestamp"])
        for i in np.flatnonzero(d["hard_blocked"]):
            state.pop(int(d["ids"][i]), None)
        for instance in np.unique(d["instances"]):
            group = np.flatnonzero(d["valid"] & (d["instances"] == instance))
            pairs = [i for i in group if int(d["ids"][i]) in state and t - state[int(d["ids"][i])]["time"] <= 0.2]
            if bool(d["sensor_update"]):
                trusted = [i for i in pairs if d["provenance"][i] == 1]
                mode = False
                if len(trusted) >= 5:
                    displacement = np.array(
                        [np.linalg.norm(d["uv"][i] - state[int(d["ids"][i])]["uv"]) for i in trusted]
                    )
                    dz = np.array([base[i, 2] - state[int(d["ids"][i])]["measured"] for i in trusted])
                    mode = bool(
                        np.median(displacement) < 0.8
                        and np.percentile(displacement, 90) < 1.5
                        and abs(np.median(dz)) < 0.015
                    )
                modes[int(instance)] = mode
            mode = modes.get(int(instance), False)
            for i in group:
                key = int(d["ids"][i])
                previous = state.get(key)
                if mode and previous and t - previous["time"] <= 0.2:
                    if bool(d["sensor_update"]) and d["provenance"][i] == 1:
                        delta = float(base[i, 2]) - previous["z"]
                        xyz[i, 2] = previous["z"] if abs(delta) < 0.002 else previous["z"] + 0.12 * delta
                    else:
                        xyz[i, 2] = previous["z"]
                if bool(d["sensor_update"]) and d["provenance"][i] == 1:
                    state[key] = dict(z=float(xyz[i, 2]), measured=float(base[i, 2]), uv=d["uv"][i].copy(), time=t)
        xyz[:, 0] = (d["uv"][:, 0] - k["ppx"]) / k["fx"] * xyz[:, 2]
        xyz[:, 1] = (d["uv"][:, 1] - k["ppy"]) / k["fy"] * xyz[:, 2]
        state = {key: value for key, value in state.items() if t - value["time"] <= 0.2}
        d["before_static_xyz"] = base
        d["xyz"] = xyz
        elapsed = (time.perf_counter() - tic) * 1000
        timings.append(elapsed)
        np.savez_compressed(path, **d)
    old.save(
        destination / "static_fusion_summary.json",
        dict(
            frames=len(timings),
            uses_future_frames=False,
            p50_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)),
            static_alpha=0.12,
        ),
    )
    # The recording must be regenerated after point coordinates change.
    (destination / "html_complete.json").unlink(missing_ok=True)


def html():
    old.TASK = "scissors"
    old.VERSION = "foundationstereo_d435_filters"
    old.V0 = V0
    old.OUT = OUT
    legend = '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:355px;font:13px/1.5 system-ui"><b>Scissors · FoundationStereo＋D435 V3过滤</b><br>深度仍为FoundationStereo，不使用D435 SDK深度。<br>加入原D435规则：原生邻域支持、IR→RGB遮挡过滤、接触边界深度跳变门控、核心深度历史检查、三维离群过滤、静止表面融合。边界拒绝点禁止被历史补回。<br>保留PointWorld空间过滤和1.5cm体素选点。全部过滤仅使用当前/历史帧。<br>01：新版；02：原始FoundationStereo取样；03：历史补点/融合；04：真实URDF；05：上一个PointWorld过滤版。蓝色URDF、橙色手点。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
    old.html(
        comparison=PREVIOUS,
        show_urdf=True,
        surface_stride=4,
        depth_label="FoundationStereo",
        legend_text=legend,
        display_hold_steps=0,
        highlight_interframe=True,
        comparison_label="05_previous_PointWorld_filter",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["filter", "stabilize", "html", "all"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage in ["filter", "all"]:
        process(args.limit)
    if args.stage in ["stabilize", "all"]:
        strengthen_static(limit=args.limit)
    if args.stage in ["html", "all"]:
        html()
