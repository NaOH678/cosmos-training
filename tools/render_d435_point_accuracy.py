"""Multi-view D435/URDF comparison from actual depth frames; no point tracker or depth model."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.spatial import cKDTree

TASKS = {
    "sandwich": (
        "singlerighthand_sandwich_100/episode_0013_20260731_133649",
        "pointflow_outputs/sandwich_bread_variants_20261030_v2/separate",
    ),
    "dropper": (
        "dropper_dagger_mix/episode_0000_20260830_155455",
        "pointflow_outputs/dropper_micro_requested_20261030/dropper_mix",
    ),
    "micro": (
        "dualhand_micropipette/episode_0007_20260824_174548",
        "pointflow_outputs/dropper_micro_requested_20261030/micropipette",
    ),
}
CYAN = np.array([50, 215, 245], np.uint8)  # RGB
ORANGE = np.array([255, 135, 30], np.uint8)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Surface:
    def __init__(self, episode, config):
        sys.path.append("/data/shichaojian/wuji-mjlab/.venv/lib/python3.10/site-packages")
        import mujoco

        self.mj = mujoco
        replay = load_module("d435_replay", "/data/shichaojian/wuji-mjlab/scripts/replay/replay_teleop.py")
        urdf = Path(
            config.get(
                "urdf", "/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
            )
        )
        self.model = m = mujoco.MjModel.from_xml_string(replay.resolve_urdf_resources(urdf))
        self.data = mujoco.MjData(m)
        self.traj, meta = replay.load_episode(episode)
        self.addr, _, _ = replay.model_joint_map(m, replay.dataset_joint_names(meta))
        self.camera = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_d435_link_optical_frame")
        rng = np.random.default_rng(0)
        points, geoms = [], []
        sides = config.get("hand_sides", ["right"])
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if "_visual_" not in name or not any(body.startswith(s + "_hand_") for s in sides):
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            faces = m.mesh_face[m.mesh_faceadr[mesh] : m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
            tri = verts[faces]
            area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
            n = int(np.ceil(area.sum() / 0.0015**2))
            chosen = rng.choice(len(tri), n, p=area / area.sum())
            a, b = np.sqrt(rng.random(n)), rng.random(n)
            points.append(
                (1 - a[:, None]) * tri[chosen, 0]
                + (a * (1 - b))[:, None] * tri[chosen, 1]
                + (a * b)[:, None] * tri[chosen, 2]
            )
            geoms.append(np.full(n, g))
        self.points, self.geoms = np.concatenate(points), np.concatenate(geoms)
        self.urdf = str(urdf)

    def at(self, frame):
        assert frame < len(self.traj)
        self.data.qpos[self.addr] = self.traj[frame]
        self.mj.mj_forward(self.model, self.data)
        world = (
            np.einsum("nij,nj->ni", self.data.geom_xmat.reshape(-1, 3, 3)[self.geoms], self.points)
            + self.data.geom_xpos[self.geoms]
        )
        # Same explicit optical-axis roll as the existing SAM2 URDF hand gate.
        return ((world - self.data.xpos[self.camera]) @ self.data.xmat[self.camera].reshape(3, 3)) * [-1, -1, 1]


def project(points, intr):
    z = points[:, 2]
    valid = np.isfinite(points).all(1) & (z > 0.05)
    uv = np.zeros((len(points), 2), np.int32)
    uv[valid, 0] = np.rint(points[valid, 0] / z[valid] * intr["fx"] + intr["ppx"]).astype(int)
    uv[valid, 1] = np.rint(points[valid, 1] / z[valid] * intr["fy"] + intr["ppy"]).astype(int)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < intr["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < intr["height"])
    return uv, valid


def visible_surface(points, intr):
    uv, good = project(points, intr)
    zbuf = np.full((intr["height"], intr["width"]), np.inf, np.float32)
    np.minimum.at(zbuf, (uv[good, 1], uv[good, 0]), points[good, 2].astype(np.float32))
    zbuf = cv2.erode(zbuf, np.ones((3, 3), np.uint8))
    good[good] &= points[good, 2] <= zbuf[uv[good, 1], uv[good, 0]] + 0.004
    return points[good]


@lru_cache(maxsize=4)
def gpu_grid(height, width, device):
    import torch

    return torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float64),
        torch.arange(width, device=device, dtype=torch.float64),
        indexing="ij",
    )


def gpu_projection(depth, streams, instance, labels, device):
    """Dense backprojection, cross-camera transform and RGB-view z-buffer on CUDA."""
    import torch

    di, ci = streams["depth"]["intrinsics"], streams["color"]["intrinsics"]
    with torch.inference_mode():
        z = torch.as_tensor(depth.astype(np.float32), device=device) * streams["depth"]["depth_scale_m"]
        y, x = gpu_grid(*depth.shape, device)
        xyz = torch.stack(((x - di["ppx"]) / di["fx"] * z, (y - di["ppy"]) / di["fy"] * z, z), -1).reshape(-1, 3)
        extr = streams["depth"]["extrinsics_to_color"]
        rotation = torch.tensor(
            np.asarray(extr["rotation"]).reshape(3, 3, order="F").copy(), device=device, dtype=torch.float64
        )
        points = xyz @ rotation.T + torch.tensor(extr["translation"], device=device, dtype=torch.float64)
        good = torch.isfinite(points).all(1) & (points[:, 2] > 0.05) & (z.flatten() > 0.05)
        safe_z = points[:, 2].clamp_min(0.001)
        u = torch.round(points[:, 0] / safe_z * ci["fx"] + ci["ppx"]).long()
        v = torch.round(points[:, 1] / safe_z * ci["fy"] + ci["ppy"]).long()
        good &= (u >= 0) & (u < ci["width"]) & (v >= 0) & (v < ci["height"])
        pix = v[good] * ci["width"] + u[good]
        zbuf = torch.full((ci["height"] * ci["width"],), float("inf"), device=device)
        zbuf.scatter_reduce_(0, pix, points[good, 2].float(), reduce="amin", include_self=True)
        visible = good.clone()
        visible[good] &= points[good, 2] <= zbuf[pix] + 0.003
        ids = torch.zeros(len(points), device=device, dtype=torch.uint8)
        cls = torch.zeros_like(ids)
        ids[good] = torch.as_tensor(np.array(instance), device=device)[v[good], u[good]]
        cls[good] = torch.as_tensor(np.array(labels), device=device)[v[good], u[good]]
        raw = visible & (ids > 0)
        return tuple(a.cpu().numpy() for a in (points, torch.stack([u, v], 1), visible, ids, cls, raw))


def measured_cloud(depth, streams, instance, labels, image, device="cpu"):
    di, ci = streams["depth"]["intrinsics"], streams["color"]["intrinsics"]
    # All current recordings have zero lens-distortion coefficients.
    assert not np.any(di["coeffs"]) and not np.any(ci["coeffs"]), (
        "Implement calibrated undistortion before using this capture"
    )
    z = depth.astype(np.float32) * streams["depth"]["depth_scale_m"]
    if device.startswith("cuda"):
        points, uv, good, ids, classes, raw = gpu_projection(depth, streams, instance, labels, device)
    else:
        y, x = np.indices(z.shape)
        xyz = np.stack(((x - di["ppx"]) / di["fx"] * z, (y - di["ppy"]) / di["fy"] * z, z), -1).reshape(-1, 3)
        extr = streams["depth"]["extrinsics_to_color"]
        rotation = np.asarray(extr["rotation"]).reshape(3, 3, order="F")
        points = xyz @ rotation.T + np.asarray(extr["translation"])
        uv, good = project(points, ci)
        good &= z.ravel() > 0.05
        ids = np.zeros(len(points), np.uint8)
        classes = np.zeros(len(points), np.uint8)
        ids[good] = instance[uv[good, 1], uv[good, 0]]
        classes[good] = labels[uv[good, 1], uv[good, 0]]
        zbuf = np.full(instance.shape, np.inf, np.float32)
        np.minimum.at(zbuf, (uv[good, 1], uv[good, 0]), points[good, 2].astype(np.float32))
        good[good] &= points[good, 2] <= zbuf[uv[good, 1], uv[good, 0]] + 0.003
        raw = good & (ids > 0)
    # Keep measured values unchanged. Require two same-instance, similar-depth
    # neighbours in a 3x3 native-depth window; preserves a one-pixel-wide line.
    native_ids = ids.reshape(z.shape)
    candidate = raw.reshape(z.shape)
    supports = np.zeros(z.shape, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            near_z = np.roll(z, (dy, dx), axis=(0, 1))
            near_ids = np.roll(native_ids, (dy, dx), axis=(0, 1))
            ok = np.roll(candidate, (dy, dx), axis=(0, 1)) & (near_ids == native_ids)
            ok &= np.abs(near_z - z) <= np.maximum(0.008, 0.01 * z)
            if dy < 0:
                ok[dy:] = False
            if dy > 0:
                ok[:dy] = False
            if dx < 0:
                ok[:, dx:] = False
            if dx > 0:
                ok[:, :dx] = False
            supports += ok
    kept = raw & (z.ravel() >= 0.15) & (z.ravel() <= 2.5) & (supports.ravel() >= 2)
    colors = np.zeros((len(points), 3), np.uint8)
    colors[good] = image[uv[good, 1], uv[good, 0], ::-1]
    object_pixels = good & (classes != 2)
    colors[object_pixels] = (0.82 * colors[object_pixels] + 40).clip(0, 255).astype(np.uint8)
    colors[classes == 2] = ORANGE
    return points, colors, classes, ids, raw, kept, uv


def display_xyz(points):
    return points[:, [0, 2, 1]] * [1, 1, -1]  # right, depth, up


def panel(points, colors, center, span, azimuth, title, reference):
    canvas = np.full((480, 640, 3), (25, 23, 20), np.uint8)
    a, e = np.deg2rad([azimuth, 22])
    direction = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
    forward = -direction
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, up, forward], axis=1)
    scale = 365 / span

    def pixels(p):
        q = (display_xyz(p) - center) @ basis
        return np.column_stack([320 + q[:, 0] * scale, 258 - q[:, 1] * scale]), q[:, 2]

    # A fixed metric reference grid (0.1m spacing), not a fitted tabletop.
    grid_z = center[2] - span / 2
    for axis in (0, 1):
        for offset in np.arange(-span / 2, span / 2 + 0.001, 0.1):
            p = np.tile(center, (2, 1))
            p[:, 2] = grid_z
            p[:, axis] += offset
            p[:, 1 - axis] += [-span / 2, span / 2]
            q = (p - center) @ basis
            v = np.column_stack([320 + q[:, 0] * scale, 258 - q[:, 1] * scale]).astype(int)
            cv2.line(canvas, tuple(v[0]), tuple(v[1]), (48, 45, 40), 1, cv2.LINE_AA)
    # Shared z-buffer ordering across measured and reference point clouds.
    pts = np.concatenate([reference, points])
    rgb = np.concatenate([np.tile(CYAN, (len(reference), 1)), colors])
    xy, depth = pixels(pts)
    valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 1) & (xy[:, 0] < 639) & (xy[:, 1] >= 38) & (xy[:, 1] < 478)
    select = np.flatnonzero(valid)
    select = select[np.argsort(-depth[select])]
    xy = xy[select].astype(int)
    bgr = rgb[select, ::-1]
    # Vectorized 2x2 splats; no depth fitting, smoothing, or track identities.
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        canvas[xy[:, 1] + dy, xy[:, 0] + dx] = bgr
    cv2.putText(canvas, title, (10, 23), 0, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "cyan: URDF | orange: D435 hand | RGB: objects", (10, 461), 0, 0.4, (210, 210, 210), 1)
    cv2.putText(canvas, "grid: 0.10 m", (10, 440), 0, 0.4, (180, 180, 180), 1)
    return canvas


def sampled(points, colors, ids, cap=1800):
    selections = []
    for i in np.unique(ids):
        members = np.flatnonzero(ids == i)
        selections.extend(members[np.linspace(0, len(members) - 1, min(cap, len(members)), dtype=int)])
    return points[selections], colors[selections]


def run(args):
    episode_path, mask_path = TASKS[args.task]
    episode = Path("/data/shichaojian/raw_data") / episode_path
    mask_dir = Path(mask_path)
    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]
    streams = metadata["streams"]
    source_init = json.loads((mask_dir / "initialization.json").read_text())
    config = (
        json.loads((mask_dir / "task_config.json").read_text())
        if (mask_dir / "task_config.json").exists()
        else source_init["config"]
    )
    surface = Surface(episode, config)
    labels = np.load(mask_dir / "labels.npy", mmap_mode="r")
    instances = np.load(mask_dir / "instances.npy", mmap_mode="r")
    local = Path("/tmp/d435_point_accuracy_20261102") / args.task
    local.mkdir(parents=True, exist_ok=True)
    src = episode / "auxiliary_camera/depth.lmdb/data.mdb"
    if not (local / "data.mdb").exists() or (local / "data.mdb").stat().st_size != src.stat().st_size:
        shutil.copyfile(src, local / "data.mdb")
    env = lmdb.open(str(local), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        )
    frames = [f for f in frames if f < len(labels)]
    if args.limit:
        frames = frames[: args.limit]
    capture = cv2.VideoCapture(str(episode / "videos/head.mp4"))
    fps_rgb = capture.get(cv2.CAP_PROP_FPS)
    ci = streams["color"]["intrinsics"]
    # A fixed box inferred from several frames, never independently re-centred.
    bounds = []
    projection_check = None
    for f in frames[:: max(1, len(frames) // 12)]:
        bounds.append(surface.at(f)[::20])
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{f:06d}".encode())
        d = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, image = capture.read()
        assert ok
        p, _, _, _, raw, _, _ = measured_cloud(d, streams, instances[f], labels[f], image, args.device)
        if projection_check is None and args.device.startswith("cuda"):
            cpu_p, _, _, _, cpu_raw, _, _ = measured_cloud(d, streams, instances[f], labels[f], image)
            max_delta = float(np.max(np.abs(p - cpu_p)))
            changed = int(np.count_nonzero(raw != cpu_raw))
            assert max_delta < 1e-6 and changed < 32, (max_delta, changed)
            projection_check = dict(max_coordinate_difference_metres=max_delta, differing_roi_membership_pixels=changed)
        q = p[raw]
        q = q[(q[:, 2] >= 0.15) & (q[:, 2] <= 2.5)]
        if len(q):
            bounds.append(q[::20])
    bound = display_xyz(np.concatenate(bounds))
    lo, hi = np.percentile(bound, [1, 99], axis=0)
    center = (lo + hi) / 2
    span = float(np.max(hi - lo) * 1.25 + 0.08)
    writer = cv2.VideoWriter(
        str(out / "multiview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), streams["depth"]["fps"], (1280, 960)
    )
    assert writer.isOpened()
    rows = []
    samples = set([0, len(frames) // 5, len(frames) // 2, 4 * len(frames) // 5, len(frames) - 1])
    for index, f in enumerate(frames):
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{f:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, image = capture.read()
        assert ok
        points, colors, classes, ids, raw, kept, uv = measured_cloud(
            depth, streams, instances[f], labels[f], image, args.device
        )
        ref = surface.at(f)
        shown_ref = visible_surface(ref, ci)[::2]
        tree = cKDTree(ref)
        stats = dict(
            frame=f,
            time_sec=f / fps_rgb,
            raw_hand=int((raw & (classes == 2)).sum()),
            filtered_hand=int((kept & (classes == 2)).sum()),
            raw_objects=int((raw & (classes != 2)).sum()),
            filtered_objects=int((kept & (classes != 2)).sum()),
        )
        for name, gate in [("raw", raw), ("filtered", kept)]:
            hand = points[gate & (classes == 2)]
            distances = tree.query(hand)[0] * 1000 if len(hand) else np.array([])
            stats[name + "_hand_reference_mm"] = (
                dict(p50=float(np.median(distances)), p90=float(np.percentile(distances, 90)))
                if len(distances)
                else None
            )
        rows.append(stats)
        rgb = image.copy()
        for cls, color in [(2, (30, 135, 255)), (3, (60, 220, 60))]:
            mask = labels[f] == 2 if cls == 2 else (labels[f] > 0) & (labels[f] != 2)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, color, 1)
        original = cv2.resize(rgb, (512, 384))
        rgb = np.full((480, 640, 3), 15, np.uint8)
        rgb[92:476, 64:576] = original
        cv2.putText(
            rgb,
            f"{args.task} | frame {f} | t={f / fps_rgb:.2f}s | measured depth frames only",
            (8, 20),
            0,
            0.44,
            (245, 245, 245),
            1,
        )
        cv2.putText(
            rgb,
            f"points raw -> filtered: hand {stats['raw_hand']} -> {stats['filtered_hand']}",
            (8, 41),
            0,
            0.45,
            (30, 135, 255),
            1,
        )
        cv2.putText(
            rgb, f"objects {stats['raw_objects']} -> {stats['filtered_objects']}", (8, 61), 0, 0.45, (60, 220, 60), 1
        )
        raw_mm, filtered_mm = stats["raw_hand_reference_mm"], stats["filtered_hand_reference_mm"]
        if raw_mm and filtered_mm:
            cv2.putText(
                rgb,
                f"hand-to-URDF median (reference): {raw_mm['p50']:.1f} -> {filtered_mm['p50']:.1f} mm",
                (8, 82),
                0,
                0.43,
                (235, 220, 140),
                1,
            )
        angle = -95 + 230 * index / max(len(frames) - 1, 1)
        rp, rc = sampled(points[raw], colors[raw], ids[raw])
        fp, fc = sampled(points[kept], colors[kept], ids[kept])
        raw_panel = panel(rp, rc, center, span, angle, "RAW ROI depth + URDF | rotating view", shown_ref)
        filtered = panel(fp, fc, center, span, angle, "FILTERED + URDF | same rotating view", shown_ref)
        side = panel(fp, fc, center, span, 10, "FILTERED + URDF | fixed side view", shown_ref)
        frame = np.vstack([np.hstack([rgb, raw_panel]), np.hstack([filtered, side])])
        writer.write(frame)
        if index in samples:
            cv2.imwrite(str(out / f"preview_{f:04d}.jpg"), frame)
            cloud = dict(
                urdf_surface=ref.astype(np.float32),
                urdf_visible=shown_ref.astype(np.float32),
                raw=points[raw].astype(np.float32),
                filtered=points[kept].astype(np.float32),
                raw_colors=colors[raw],
                filtered_colors=colors[kept],
                raw_class=classes[raw],
                filtered_class=classes[kept],
            )
            np.savez_compressed(out / f"cloud_{f:04d}.npz", **cloud)
        if index % 30 == 0:
            print(args.task, index, "/", len(frames), stats, flush=True)
    writer.release()
    capture.release()
    env.close()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(out / "multiview.mp4"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "multiview_h264.mp4"),
        ],
        check=True,
    )
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    summary = dict(
        device=args.device,
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        projection_cpu_cuda_check=projection_check,
        task=args.task,
        episode=str(episode),
        mask_source=str(mask_dir.resolve()),
        frames=len(frames),
        frame_ids=frames,
        output_fps=streams["depth"]["fps"],
        rgb_fps=fps_rgb,
        urdf=surface.urdf,
        filters=[
            "finite native depth >0.05m; SDK column-major depth-to-RGB extrinsics; RGB z-buffer visibility and original SAM2-large masks",
            "0.15..2.5m; at least 2 same-instance 3x3 neighbours within max(8mm,1% depth)",
            "no filled holes, no temporal averaging, no mask erosion, no URDF-distance gating; measured XYZ unchanged",
        ],
        bounds=dict(center=center.tolist(), span=span),
        limitations=[
            "URDF is a kinematic reference, not measured ground truth; gloves, calibration and recorded-state timing can differ",
            "Object geometry has no ground truth; missing and through-glass depth is not reconstructed",
            "Stored RGB/depth matching uses recorded LMDB frame IDs; no stale depth repeated to 30Hz",
            "Display subsampling per instance only; statistics use all associated depth points",
            "Depth extrinsics serialization is interpreted as the standard rs2_extrinsics column-major layout",
        ],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE", args.task, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, default=Path("pointflow_outputs/d435_point_accuracy_20261102"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
