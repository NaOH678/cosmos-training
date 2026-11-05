"""Depth/tracking/export stages for the five real CoTracker trials."""

import base64
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/co-tracker"))
from render_d435_point_accuracy import load_module

_core = load_module(
    "real_pointflow_core", str(ROOT / "cosmos_framework/inference/robot_policy/causal_rgb_pointflow.py")
)
_guard = load_module("real_pointflow_guard", str(ROOT / "cosmos_framework/inference/robot_policy/hand_depth_guard.py"))
CausalPointTracks, CausalGeometryFilter, translate_z = (
    _core.CausalPointTracks,
    _core.CausalGeometryFilter,
    _core.translate_z,
)
HandDepthGuard = _guard.HandDepthGuard
from run_real_cotracker_pipeline import save


class DepthSource:
    """Current-frame AnyView + Metric with Track4World AnyView weights.

    Same nested DA3 scaling/alignment/sky routines; calibrated focal scaling;
    no whole-video last-chunk normalization and no future frames.
    """

    def __init__(self, intr):
        import torch
        from compare_depth_sources import load_model

        self.torch = torch
        self.model = load_model()
        torch.cuda.set_per_process_memory_fraction(0.85)
        self.intr = intr
        self.K = torch.tensor(
            [
                [intr["fx"] * 630 / 640, 0, (intr["ppx"] + 0.5) * 630 / 640 - 0.5],
                [0, intr["fy"] * 448 / 480, (intr["ppy"] + 0.5) * 448 / 480 - 0.5],
                [0, 0, 1],
            ],
            device="cuda",
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]

    def predict(self, image):
        import torch.nn.functional as F
        from addict import Dict

        torch = self.torch
        x = (
            torch.from_numpy(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 448)))
            .permute(2, 0, 1)[None]
            .float()
            .cuda()
            / 255
        )
        x = F.interpolate((x - self.mean) / self.std, (448, 630), mode="bilinear", align_corners=False, antialias=True)[
            None
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            metric = self.model.da3_metric(x)
            output = self.model.da3(
                x, export_feat_layers=[], infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"
            )
        with torch.inference_mode():
            p = Dict({k: output[k].float() for k in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
            m = Dict({k: metric[k].float().clone() for k in ["depth", "sky"]})
            predK = p.intrinsics
            p.intrinsics = self.K[None, None]
            self.model._apply_metric_scaling(p, m)
            p.intrinsics = predK
            self.model._apply_depth_alignment(p, m)
            self.model._handle_sky_regions(p, m)
            z = F.interpolate(p.depth[0, None], (448, 640), mode="bilinear", align_corners=False)[0, 0]
            conf = F.interpolate(p.depth_conf[0, None], (448, 640), mode="bilinear", align_corners=False)[0, 0]
        return z.cpu().numpy(), conf.cpu().numpy()


def sensor_buffer(depth, streams, instances, labels, image):
    from render_d435_point_accuracy import measured_cloud

    points, _, _, _, raw, filtered, uv = measured_cloud(depth, streams, instances, labels, image, "cpu")
    # Preserve calibrated nearest-depth visibility and same-instance neighbour support.
    zbuf = np.full(image.shape[:2], np.inf, np.float32)
    use = filtered & np.isfinite(points).all(1)
    np.minimum.at(zbuf, (uv[use, 1], uv[use, 0]), points[use, 2].astype(np.float32))
    return cv2.resize(zbuf, (640, 448), interpolation=cv2.INTER_NEAREST)


def hand_shift(sensor, depth, instances, labels, guards, timestamp):
    details = []
    measured = []
    predicted = []
    for instance in np.unique(instances[labels == 2]):
        if instance == 0:
            continue
        mask = ((instances == instance) & (labels == 2)).astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        inner = (mask > 0) & (distance >= max(3.0, float(np.percentile(distance[mask > 0], 75))))
        samples = sensor[inner & np.isfinite(sensor) & (sensor > 0.1) & (sensor < 2.5)]
        core = inner & np.isfinite(sensor) & np.isfinite(depth) & (depth > 0.1)
        observed = None
        valid = False
        mode_fraction = None
        if len(samples) >= 30:
            edges = np.arange(0.1, 2.505, 0.005)
            hist, _ = np.histogram(samples, edges)
            idx = int(np.argmax(gaussian_filter1d(hist.astype(float), 2)))
            observed = float((edges[idx] + edges[idx + 1]) / 2)
            mode_fraction = float(np.mean(abs(samples - observed) <= 0.03))
            core &= abs(sensor - observed) <= 0.03
            valid = mode_fraction >= 0.55 and int(core.sum()) >= 30
        guard = guards.setdefault(int(instance), HandDepthGuard())
        result = guard.update(timestamp, observed, valid)
        delta = None
        if result["status"] == "measured":
            differences = sensor[core] - depth[core]
            median = np.median(differences)
            mad = np.median(abs(differences - median))
            robust = abs(differences - median) <= max(0.01, 3 * 1.4826 * mad)
            if robust.sum() >= 30:
                delta = float(np.median(differences[robust]))
                measured.append(delta)
        elif result["status"] == "predicted":
            target = depth[inner & np.isfinite(depth) & (depth > 0.1)]
            if len(target) >= 30:
                delta = float(result["z"] - np.median(target))
                predicted.append(delta)
        details.append(
            dict(instance=int(instance), observed_z_m=observed, mode_fraction=mode_fraction, offset_m=delta, **result)
        )
    chosen = measured or predicted
    return (
        (float(np.median(chosen)) if chosen else 0.0),
        ("measured" if measured else "predicted" if predicted else "unavailable"),
        details,
    )


def points(out, limit=None):
    import torch
    from cotracker.predictor import CoTrackerOnlinePredictor

    source = json.loads((out / "source.json").read_text())
    ep = Path(source["episode"])
    info = json.loads((out / "video.json").read_text())
    T = min(info["frames"], limit or info["frames"])
    dest = out / ("pointflow_smoke" if limit else "pointflow")
    dest.mkdir(exist_ok=True)
    frames = dest / "frames"
    frames.mkdir(exist_ok=True)
    if (dest / "COMPLETE_STAGE.json").exists():
        return
    streams = json.loads((ep / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"][
        "streams"
    ]
    intr = streams["color"]["intrinsics"]
    labels = np.load(out / "labels.npy", mmap_mode="r")
    instances = np.load(out / "instances.npy", mmap_mode="r")
    assert len(labels) >= T
    local = Path("/tmp/cotracker_real_depth_20261104") / out.name
    local.mkdir(parents=True, exist_ok=True)
    original = ep / "auxiliary_camera/depth.lmdb/data.mdb"
    if not (local / "data.mdb").exists() or (local / "data.mdb").stat().st_size != original.stat().st_size:
        shutil.copyfile(original, local / "data.mdb")
    env = lmdb.open(str(local), readonly=True, lock=False)
    with env.begin() as txn:
        depth_frames = {
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        }
    sync = json.loads((ep / "sync_timestamps.json").read_text())
    timestamps = np.array([s.get("camera_head", s["anchor"]) for s in sync[:T]], float)
    timestamps -= timestamps[0]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Non-increasing camera timestamps")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    depth_model = DepthSource(intr)
    tracker = (
        CoTrackerOnlinePredictor(
            checkpoint=str(ROOT / "third_party/co-tracker/scaled_online.pth"), v2=False, window_len=16
        )
        .cuda()
        .eval()
    )
    # Identical causal pair protocol to the simulation pipeline; short embedding is explicit.
    tracker.model.window_len = 2
    tracker.model.time_emb = tracker.model.time_emb[:, :2].clone()
    tracks = CausalPointTracks(grid_step=3)
    geometry = CausalGeometryFilter()
    guards = {}
    previous = None
    last_shift_time = -1.0
    shift = 0.0
    shift_status = "unavailable"
    details = []
    rows = []
    for frame in range(T):
        image = cv2.imread(str(out / "frames" / f"{frame:05d}.jpg"))
        rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 448))
        semantics = cv2.resize(labels[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        semantics = np.where(semantics == 2, 2, np.where(semantics > 0, 3, 0)).astype(np.uint8)
        imap = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        if frame and len(tracks.ids):
            video = torch.from_numpy(np.stack([previous, rgb])).permute(0, 3, 1, 2)[None].float().cuda()
            pred = []
            visible = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(tracks.ids), 8192):
                    q = (
                        torch.from_numpy(
                            np.c_[np.zeros(len(tracks.uv[start : start + 8192])), tracks.uv[start : start + 8192]]
                        )
                        .float()[None]
                        .cuda()
                    )
                    tracker(video_chunk=video[:, :1], is_first_step=True, queries=q, add_support_grid=True)
                    uv, ok = tracker(video_chunk=video, add_support_grid=True)
                    pred.append(uv[0, 1].float().cpu().numpy())
                    visible.append(ok[0, 1].cpu().numpy())
            born = tracks.advance(semantics, np.concatenate(pred), np.concatenate(visible))
            del video, uv, ok, q
        else:
            born = tracks.advance(semantics)
        tracks.record_births(frame, born)
        torch.cuda.synchronize()
        tracking_ms = (time.perf_counter() - tic) * 1000
        tic = time.perf_counter()
        depth, confidence = depth_model.predict(image)
        torch.cuda.synchronize()
        da3_ms = (time.perf_counter() - tic) * 1000
        tic = time.perf_counter()
        if frame in depth_frames:
            with env.begin() as txn:
                blob = txn.get(f"depth/head/{frame:06d}".encode())
            native = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
            sensor = sensor_buffer(native, streams, instances[frame], labels[frame], image)
            shift, shift_status, details = hand_shift(sensor, depth, imap, semantics, guards, float(timestamps[frame]))
            last_shift_time = float(timestamps[frame])
        elif timestamps[frame] - last_shift_time > 0.5:
            shift = 0.0
            shift_status = "expired"
        sensor_ms = (time.perf_counter() - tic) * 1000
        tic = time.perf_counter()
        uv = tracks.uv.copy()
        q = np.rint(uv).astype(int)
        q[:, 0] = q[:, 0].clip(0, 639)
        q[:, 1] = q[:, 1].clip(0, 447)
        z = cv2.remap(depth, uv[:, 0, None], uv[:, 1, None], cv2.INTER_LINEAR).ravel()
        conf = cv2.remap(confidence, uv[:, 0, None], uv[:, 1, None], cv2.INTER_LINEAR).ravel()
        fy = intr["fy"] * 448 / 480
        cy = (intr["ppy"] + 0.5) * 448 / 480 - 0.5
        raw = np.column_stack(((uv[:, 0] - intr["ppx"]) / intr["fx"] * z, (uv[:, 1] - cy) / fy * z, z)).astype(
            np.float32
        )
        corrected = translate_z(raw, shift)
        assigned = imap[q[:, 1], q[:, 0]]
        valid = _core.depth_sample_valid(uv, z, conf, assigned)
        kept, filterstats = geometry.apply(tracks.ids, corrected, valid, assigned, float(timestamps[frame]))
        # Invalid entries are retained for audit, but never exposed as valid targets.
        color = rgb[q[:, 1], q[:, 0]].copy()
        filter_ms = (time.perf_counter() - tic) * 1000
        np.savez_compressed(
            frames / f"{frame:06d}.npz",
            ids=tracks.ids,
            uv=uv,
            xyz=corrected,
            raw_xyz=raw,
            valid=kept,
            labels=tracks.labels,
            instances=assigned,
            colors=color,
            depth_conf=conf,
            shift=np.float32(shift),
        )
        row = dict(
            frame=frame,
            time=float(timestamps[frame]),
            points=len(tracks.ids),
            valid_points=int(kept.sum()),
            new_points=len(born),
            shift_m=shift,
            shift_status=shift_status,
            hand_guard=details,
            sensor_updated=frame in depth_frames,
            tracking_ms=tracking_ms,
            da3_ms=da3_ms,
            sensor_ms=sensor_ms,
            filter_ms=filter_ms,
            **filterstats,
        )
        rows.append(row)
        previous = rgb
        if frame % 50 == 0:
            save(dest / "progress.json", row)
            print(
                "POINT",
                out.name,
                frame,
                "/",
                T,
                "points",
                len(tracks.ids),
                "valid",
                int(kept.sum()),
                "ms",
                round(tracking_ms),
                round(da3_ms),
                flush=True,
            )
    env.close()
    lengths = []
    for frame in range(T):
        with np.load(frames / f"{frame:06d}.npz") as n:
            lengths.append(len(n["ids"]))
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    M = int(offsets[-1])
    fields = {
        "obs_track": ("ids", np.int64, (M,)),
        "obs_pos": ("xyz", np.float32, (M, 3)),
        "obs_uv": ("uv", np.float32, (M, 2)),
        "obs_valid": ("valid", bool, (M,)),
        "obs_unique": (None, bool, (M,)),
    }
    for name, (key, dtype, shape) in fields.items():
        array = np.lib.format.open_memmap(dest / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        for frame in range(T):
            with np.load(frames / f"{frame:06d}.npz") as n:
                array[offsets[frame] : offsets[frame + 1]] = n[key] if key else True
        array.flush()
        del array
    K = np.array(
        [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"] * 448 / 480, (intr["ppy"] + 0.5) * 448 / 480 - 0.5], [0, 0, 1]]
    )
    for name, value in dict(
        frame_offsets=offsets,
        track_label=np.asarray(tracks.track_labels, np.int32),
        track_birth=np.asarray(tracks.births, np.int32),
        frame_indices=np.arange(T),
        timestamps_sec=np.arange(T) / info["fps"],
        sensor_timestamps_sec=timestamps,
        intrinsics=np.broadcast_to(np.diag([1 / 640, 1 / 448, 1]) @ K, (T, 3, 3)),
    ).items():
        np.save(dest / (name + ".npy"), value)
    report = dict(
        schema_version="rgbd_pointflow_v1",
        filter_revision=getattr(_core, "FILTER_REVISION", 1),
        coordinate="fixed_optical_camera",
        metric_scale=True,
        video=str(ep / "videos/head.mp4"),
        image_size_wh=[640, 448],
        source_image_size_wh=[info["width"], info["height"]],
        frames=T,
        tracks=len(tracks.births),
        observations=M,
        tracker="CoTracker3 online architecture, causal adjacent-pair reinitialization; pretrained embedding prefix window_len=2",
        uses_future_frames=False,
        depth="Track4World AnyView weights + original DA3 Metric, current-frame nested alignment, calibrated focal; confidence>2, same-instance pixel depth discontinuities",
        correction="Hand-core D435 offset; guarded last4 accepted observations; same delta added only to Z of ALL hand/object points. XY and original observed UV unchanged.",
        filter="Per-instance 16NN robust outliers, per-track depth-change residual vs same-instance median; no future fill or temporal Gaussian",
        urdf=json.loads((out / "task_config.json").read_text())["urdf"],
        data_adapter="cosmos_framework.data.ragged_pointflow.RaggedPointFlow",
        grid_step=3,
        point_ids="episode-global; lost UV trajectories terminate, newly exposed surfaces get new IDs",
        timeline="timestamps_sec follows source video FPS, as legacy exports; sensor_timestamps_sec keeps actual camera timestamps for D435 gating",
    )
    save(dest / "report.json", report)
    save(dest / "per_frame.json", rows)
    save(
        dest / "COMPLETE_STAGE.json",
        dict(
            frames=T,
            valid_observations=sum(r["valid_points"] for r in rows),
            timing={
                key: dict(
                    p50_ms=float(np.median([r[key] for r in rows[5:]])),
                    p95_ms=float(np.percentile([r[key] for r in rows[5:]], 95)),
                )
                for key in ["tracking_ms", "da3_ms", "sensor_ms", "filter_ms"]
            },
        ),
    )
    print("POINT COMPLETE", out.name, T, flush=True)


def render(out):
    import subprocess

    import viser
    from render_d435_point_accuracy import Surface

    sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
    from export_efep_official_html import package

    pointdir = out / "pointflow"
    report = json.loads((pointdir / "report.json").read_text())
    rows = json.loads((pointdir / "per_frame.json").read_text())
    T = report["frames"]
    info = json.loads((out / "video.json").read_text())
    ep = Path(json.loads((out / "source.json").read_text())["episode"])
    recipe = json.loads((out / "task_config.json").read_text())
    reference = Surface(ep, recipe)
    fps = info["fps"]
    writer = cv2.VideoWriter(str(out / "pointflow_rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    assert writer.isOpened()
    server = viser.ViserServer(host="127.0.0.1", port=9182)
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    empty = np.empty((0, 3), np.float32)
    ec = np.empty((0, 3), np.uint8)
    nodes = {
        name: server.scene.add_point_cloud("/" + name, points=empty, colors=ec, point_size=0.002, visible=visible)
        for name, visible in [
            ("01_corrected_points", True),
            ("02_original_DA3", False),
            ("03_removed_candidates", False),
            ("04_URDF_surface", False),
        ]
    }
    line = server.scene.add_line_segments(
        "/05_flow_trails",
        points=np.empty((0, 2, 3), np.float32),
        colors=np.empty((0, 2, 3), np.uint8),
        line_width=1.0,
        visible=True,
    )
    server._run_garbage_collector(force=True)
    record = server.get_scene_serializer()
    history = deque(maxlen=11)
    centers = []
    thumbs = []
    display_frames = []
    for frame in range(T):
        with np.load(pointdir / "frames" / f"{frame:06d}.npz") as n:
            data = {k: n[k] for k in n.files}
        image = cv2.imread(str(out / "frames" / f"{frame:05d}.jpg"))
        valid = data["valid"]
        ids = data["ids"][valid]
        uv = data["uv"][valid] * [1, 480 / 448]
        xyz = data["xyz"][valid]
        color = data["colors"][valid].copy()
        color[data["labels"][valid] == 2] = [255, 127, 14]
        hsv = np.zeros((len(ids), 1, 3), np.uint8)
        hsv[:, 0, 0] = (ids * 47 % 180).astype(np.uint8)
        hsv[:, 0, 1:] = [220, 255]
        trace_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0] if len(ids) else np.empty((0, 3), np.uint8)
        selected = (ids % 5) == 0
        segments2 = []
        segments3 = []
        segmentcolors = []
        for old in history:
            _, ii, jj = np.intersect1d(ids[selected], old["ids"], assume_unique=True, return_indices=True)
            # Adjacent trail edges only; never connect unrelated IDs or gaps.
            current_indices = np.flatnonzero(selected)[ii]
            ok = (
                (frame - old["frame"] == 1)
                & (np.linalg.norm(uv[current_indices] - old["uv"][jj], axis=1) < 60)
                & (np.linalg.norm(xyz[current_indices] - old["xyz"][jj], axis=1) < 0.08)
            )
            if np.any(ok):
                segments2.extend(np.stack([old["uv"][jj[ok]], uv[current_indices[ok]]], axis=1))
                segments3.extend(np.stack([old["xyz"][jj[ok]], xyz[current_indices[ok]]], axis=1))
                segmentcolors.extend(trace_color[current_indices[ok]])
        # Carry the last 10 measured consecutive edges for each currently alive ID.
        current_edges = {int(i): (u, p, c) for i, u, p, c in zip(ids, uv, xyz, color)}
        for old in history:
            for track, a, b, c in old.get("edges", []):
                if track in current_edges and frame - old["frame"] < 10:
                    cv2.line(
                        image,
                        tuple(np.rint(a).astype(int)),
                        tuple(np.rint(b).astype(int)),
                        tuple(map(int, c[::-1])),
                        1,
                        cv2.LINE_AA,
                    )
        edges = []
        for seg, c in zip(segments2, segmentcolors):
            cv2.line(
                image,
                tuple(np.rint(seg[0]).astype(int)),
                tuple(np.rint(seg[1]).astype(int)),
                tuple(map(int, c[::-1])),
                1,
                cv2.LINE_AA,
            )
        # Save edge identity, for causal RGB trail rendering on following frames.
        if history:
            old = history[-1]
            _, ii, jj = np.intersect1d(ids[selected], old["ids"], assume_unique=True, return_indices=True)
            ci = np.flatnonzero(selected)[ii]
            for j, k in zip(ci, jj):
                if np.linalg.norm(uv[j] - old["uv"][k]) < 60 and np.linalg.norm(xyz[j] - old["xyz"][k]) < 0.08:
                    edges.append((int(ids[j]), old["uv"][k], uv[j], trace_color[j]))
        for j in np.flatnonzero(selected):
            cv2.circle(image, tuple(np.rint(uv[j]).astype(int)), 1, tuple(map(int, trace_color[j, ::-1])), -1)
        cv2.putText(
            image,
            f"{out.name} {frame}/{T - 1} | measured {int(valid.sum())} | Z {rows[frame]['shift_m'] * 1000:+.1f}mm",
            (5, 22),
            0,
            0.45,
            (255, 255, 255),
            1,
        )
        writer.write(image)
        if frame in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
            cv2.imwrite(str(out / f"flow_preview_{frame:05d}.jpg"), image)
        if frame % 3 == 0 or frame == T - 1:
            shown = (ids % 2) == 0
            nodes["01_corrected_points"].points = xyz[shown]
            nodes["01_corrected_points"].colors = color[shown]
            raw = data["raw_xyz"][valid]
            nodes["02_original_DA3"].points = raw[shown]
            nodes["02_original_DA3"].colors = color[shown]
            removed = (~valid) & np.isfinite(data["xyz"]).all(1) & (data["ids"] % 2 == 0)
            nodes["03_removed_candidates"].points = data["xyz"][removed]
            nodes["03_removed_candidates"].colors = np.tile([255, 40, 40], (int(removed.sum()), 1)).astype(np.uint8)
            surface = reference.at(frame)[::4].astype(np.float32)
            nodes["04_URDF_surface"].points = surface
            nodes["04_URDF_surface"].colors = np.tile([30, 160, 235], (len(surface), 1)).astype(np.uint8)
            seg3 = np.asarray(segments3, np.float32).reshape(-1, 2, 3)
            sc = np.asarray(segmentcolors, np.uint8).reshape(-1, 3)
            line.points = seg3[::6]
            line.colors = np.repeat(sc[::6, None, :], 2, axis=1)
            dt = (1 if frame == T - 1 else min(frame + 3, T - 1) - frame) / fps
            record.insert_sleep(dt)
            display_frames.append(frame)
            ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 65])
            assert ok
            thumbs.append(base64.b64encode(jpg).decode())
            if len(xyz):
                centers.append(np.median(xyz, 0))
        history.append(dict(frame=frame, ids=ids.copy(), uv=uv.copy(), xyz=xyz.copy(), edges=edges))
        if frame % 200 == 0:
            print("RENDER", out.name, frame, T, flush=True)
    writer.release()
    recording = record.serialize()
    server.stop()
    center = np.median(centers, 0)
    camera = dict(position=(center + [0.55, -0.2, -0.15]).tolist(), look_at=center.tolist(), up=[0, -1, 0])
    meta = dict(
        duration=T / fps,
        frame_ids=display_frames,
        frame_times=[f / fps for f in display_frames],
        rgb_thumbnails=thumbs,
        source=report,
    )
    legend = (
        '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202be8;color:#eee;padding:12px;width:325px;font:13px/1.5 system-ui"><b>'
        + out.name
        + ' · CoTracker3 + DA3</b><br>固定视角，可拖动旋转、缩放、暂停及拖动时间轴。<br>01：校正点云；02：原始DA3；03：被过滤候选（红）；04：真实URDF手表面（蓝）；05：点轨迹。右侧Scene切换图层。<br>只改变Z，XY与UV保留。D435只计算手部核心平移量，手物共用。过滤只读当前/过去帧。<br>展示每3帧一次、稳定ID抽取一半点；导出数据保留全部点。<div id="source"></div><img id="rgb" style="width:100%"></aside>'
    )
    behavior = """<script>let paused=false;const m=window.__VISER_EXPORT_METADATA__;setInterval(()=>{const icon=document.querySelector('svg.tabler-icon-player-pause-filled');if(!paused&&icon){icon.closest('button').click();paused=true;}const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-m.duration)<.15);const t=slider?Number(slider.getAttribute('aria-valuenow')):0;let i=0;while(i+1<m.frame_times.length&&m.frame_times[i+1]<=t)i++;document.getElementById('source').textContent=`源帧 ${m.frame_ids[i]} · ${t.toFixed(2)}s`;const image=document.getElementById('rgb');if(image.dataset.i!==String(i)){image.src='data:image/jpeg;base64,'+m.rgb_thumbnails[i];image.dataset.i=String(i);}},150);</script>"""
    (out / "cotracker_da3_interactive.html").write_text(
        package(recording, meta, camera, out.name + " CoTracker3 DA3", legend).replace("</body>", behavior + "</body>")
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(out / "pointflow_rgb.mp4"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "pointflow_rgb_h264.mp4"),
        ],
        check=True,
    )
    save(
        out / "render_complete.json",
        dict(
            frames=T,
            html_display_frames=len(display_frames),
            html_bytes=(out / "cotracker_da3_interactive.html").stat().st_size,
        ),
    )


def verify(out):
    from cosmos_framework.data.pointflow_batch import build_pointflow_batch
    from cosmos_framework.data.pointflow_window import PointFlowTiming
    from cosmos_framework.data.ragged_pointflow import RaggedPointFlow

    dest = out / "pointflow"
    report = json.loads((dest / "report.json").read_text())
    if report.get("filter_revision", 1) < _core.FILTER_REVISION:
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "tools/refilter_real_pointflow.py"), str(out)], check=True)
        report = json.loads((dest / "report.json").read_text())
    if "source_image_size_wh" not in report:
        info = json.loads((out / "video.json").read_text())
        report["source_image_size_wh"] = [info["width"], info["height"]]
        save(dest / "report.json", report)
    T = report["frames"]
    checks = []
    for frame in range(T):
        with np.load(dest / "frames" / f"{frame:06d}.npz") as n:
            assert np.array_equal(n["xyz"][:, :2], n["raw_xyz"][:, :2])
            assert len(np.unique(n["ids"])) == len(n["ids"])
            assert np.isfinite(n["xyz"][n["valid"]]).all()
            np.testing.assert_allclose(n["xyz"][:, 2], n["raw_xyz"][:, 2] + n["shift"], rtol=1e-6, atol=1e-7)
    adapter = RaggedPointFlow(dest)
    for start in [0, T // 2]:
        sample = adapter.prepare_window(
            start, timing=PointFlowTiming(fps=15.0, steps=32, steps_per_token=4), max_points=8192, seed=0
        )
        path = dest / f"model_window_{start:05d}.npz"
        np.savez_compressed(path, **{k: v for k, v in sample.items() if isinstance(v, np.ndarray)})
        checks.append(
            dict(
                start=start,
                anchor_points=len(sample["point_ids"]),
                ptv3_voxels=len(sample["coord"]),
                targets=list(sample["target_displacement"].shape),
            )
        )
        model_sample = adapter.prepare_sample(start, timing=PointFlowTiming(), max_points=8192, seed=0)
        batch = build_pointflow_batch([model_sample])
        checks[-1]["model_batch_xyz_shape"] = list(batch.inputs["anchor_xyz"].shape)
        checks[-1]["model_batch_uv_shape"] = list(batch.inputs["anchor_uv"].shape)
    save(
        out / "verification.json", dict(frames=T, xy_unchanged=True, z_only=True, unique_ids=True, model_windows=checks)
    )
    print("VERIFIED", out.name, checks, flush=True)


def run_stage(stage, out, limit=None):
    if stage == "points":
        points(out, limit)
    elif stage == "render":
        render(out)
    elif stage == "verify":
        verify(out)
