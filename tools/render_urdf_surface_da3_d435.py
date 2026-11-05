#!/usr/bin/env python3
"""URDF hand surface vs PointFlow (DA3) vs D435 hand points, in head-camera metres, as one video.

Successor of ``WorldAct-cosmos3-edge-droid-sft_base/tools/render_fk_vs_pointflow_video.py``
(same episode, shared axis box, 6 fps = real time on the depth frames) with three changes:

  reference  the right Wuji hand's *mesh surface* from the URDF, posed by MuJoCo FK from
             the recorded qpos -- not the FK-21 skeleton.  The FK chain is the one that
             wrote ``raw_data/sandwich_fk21`` (checked: the 21 keypoints are reproduced to
             <1e-6 m before anything is drawn) and the Link_Base -> colour-camera transform
             is the verified ``fk_camera_extrinsic.py`` (URDF md5 checked).  Drawn: the
             camera-visible part of the surface (z-buffer, the only part a depth sensor can
             return).  Distances use the whole surface.
  DA3        the training points exactly as stored: ``efep_seg_v2`` label 2,
             ``obs_valid & obs_unique``.  DA3's own camera (estimated intrinsics, metric
             scale) is kept; nothing is replaced by the real camera.
  D435       head depth through the same per-frame gates the DA3 points went through:
             SAM2 v2 hand mask (arm cut) on the 640x448 canvas, 1 px ellipse erosion,
             no depth edge (utils3d depth_edge rtol 0.04, as ``run_efep_export.cleaned_masks``),
             0.1 < z < 6 m, then the exporter's own ``drop_detached.detached_mask``
             (voxel 0.015, min_frac 0.25, touch 2, gap 0.1 m).  Not applicable without
             tracks: the model confidence gate (> 0.3) and the track-based gap fill /
             despike.  The depth-edge runs on the native sensor grid, as in
             ``render_d435_pipeline_comparison.py``.

Title numbers: median distance from each cloud's points to the nearest URDF surface point.

    .venv/bin/python tools/render_urdf_surface_da3_d435.py --out <new dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CORE_DIR = Path('/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools')
sys.path.insert(0, str(CORE_DIR))
import render_fk_vs_pointflow as core  # noqa: E402

sys.path.insert(0, '/data/shichaojian/Track4World_portable')
import utils3d  # noqa: E402

REPLAY_DIR = Path('/data/shichaojian/wuji-mjlab/scripts/replay')
DETACHED_PY = Path('/data/shichaojian/pf_out/9.24/code/pf_out/drop_detached.py')
MASK_ROOT = core.DATA / 'pf_out/9.24/sandwich/sam2_masks_v2'
SURFACE_BLUE = core.FK_BLUE


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HandSurface:
    """Area-uniform samples on the right Wuji hand's visual meshes, posed per frame."""

    def __init__(self, episode, spacing_m, seed=0):
        import mujoco

        self.mujoco = mujoco
        self.replay = load_file('replay_teleop', REPLAY_DIR / 'replay_teleop.py')
        self.fk21 = load_file('export_wuji_fk21', REPLAY_DIR / 'export_wuji_fk21.py')
        if self.fk21.DEFAULT_URDF.resolve() != core.URDF.resolve():
            raise SystemExit(f'FK URDF {self.fk21.DEFAULT_URDF} is not the extrinsic URDF {core.URDF}')
        spec = mujoco.MjSpec.from_string(self.replay.resolve_urdf_resources(core.URDF))
        for mesh in spec.meshes:  # MuJoCo 3.3 rejects tiny closed-volume meshes; inertia only, not kinematics
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        self.model = m = spec.compile()
        self.data = mujoco.MjData(m)
        self.traj, meta = self.replay.load_episode(core.RAW_ROOT / episode)
        self.addr, _, _ = self.replay.model_joint_map(m, self.replay.dataset_joint_names(meta))
        self.base = self.fk21._body_id(m, 'Link_Base')
        self.sources = self.fk21.fk21_sources(m, 'right')

        rng = np.random.default_rng(seed)
        pts, gidx = [], []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or '_visual_' not in name or not body.startswith('right_hand_'):
                continue
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            f = m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
            tri = v[f]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            n = int(np.ceil(area.sum() / spacing_m ** 2))
            face = rng.choice(len(tri), n, p=area / area.sum())
            r1, r2 = rng.random(n), rng.random(n)
            s = np.sqrt(r1)
            p = ((1 - s)[:, None] * tri[face, 0] + (s * (1 - r2))[:, None] * tri[face, 1]
                 + (s * r2)[:, None] * tri[face, 2])
            pts.append(p)
            gidx.append(np.full(n, g, np.int64))
        if not pts:
            raise SystemExit('no right-hand visual meshes in the URDF')
        self.pts, self.gidx = np.concatenate(pts), np.concatenate(gidx)

    def _forward(self, frame):
        self.data.qpos[self.addr] = self.traj[min(frame, len(self.traj) - 1)]
        self.mujoco.mj_forward(self.model, self.data)

    def surface_base(self, frame):
        """[N,3] surface samples in Link_Base metres."""
        self._forward(frame)
        R = self.data.geom_xmat.reshape(-1, 3, 3)[self.gidx]
        world = np.einsum('nij,nj->ni', R, self.pts) + self.data.geom_xpos[self.gidx]
        Rb = self.data.xmat[self.base].reshape(3, 3)
        return (world - self.data.xpos[self.base]) @ Rb

    def keypoints_base(self, frame):
        self._forward(frame)
        return self.fk21.positions_in_base(self.model, self.data, self.base, self.sources)


def visible(cam, K, size=(640, 480), window=5, tol_m=0.006):
    """Which camera-frame surface samples the camera sees: z-buffer + min filter over the splat gaps."""
    import cv2

    W, H = size
    z = cam[:, 2]
    u = np.rint(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(np.int64)
    inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zbuf = np.full((H, W), np.inf, np.float32)
    np.minimum.at(zbuf, (v[inside], u[inside]), z[inside].astype(np.float32))
    zbuf = cv2.erode(zbuf, np.ones((window, window), np.uint8))  # erode = min filter
    out = np.zeros(len(cam), bool)
    out[inside] = z[inside] <= zbuf[v[inside], u[inside]] + tol_m
    return out


def d435_gated(blob, meta, K, mask_v2, detached, label=core.HAND_LABEL, with_pixels=False):
    """D435 points of one SAM2 class through the exporter's per-frame gates, plus the count after each gate.

    ``with_pixels`` also returns each kept point's native colour-image pixel (for its RGB).
    """
    import cv2

    raw_depth = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * meta[3]
    points = core.depth_cloud(blob, meta)
    hom = points @ K.T
    valid = np.isfinite(points).all(1) & (hom[:, 2] > 0)
    uv = np.zeros((len(points), 2), np.int64)
    canvas_uv = (hom[valid, :2] / hom[valid, 2:] + .5) * np.array([1., core.CANVAS_H / 480]) - .5
    uv[valid] = np.rint(canvas_uv).astype(np.int64)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < core.CANVAS_W) & (uv[:, 1] >= 0) & (uv[:, 1] < core.CANVAS_H)
    hand = cv2.resize((mask_v2 == label).astype(np.uint8), (core.CANVAS_W, core.CANVAS_H),
                      interpolation=cv2.INTER_NEAREST)
    eroded = cv2.erode(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    roi, kept = np.zeros(len(points), bool), np.zeros(len(points), bool)
    roi[inside] = hand[uv[inside, 1], uv[inside, 0]] > 0
    kept[inside] = eroded[uv[inside, 1], uv[inside, 0]] > 0
    counts = dict(in_mask=int(roi.sum()), after_erosion=int(kept.sum()))
    with np.errstate(divide='ignore', invalid='ignore'):
        edge = utils3d.numpy.depth_edge(raw_depth, rtol=.04).ravel()
    kept &= ~edge
    counts['after_depth_edge'] = int(kept.sum())
    kept &= (points[:, 2] > .1) & (points[:, 2] < 6.)
    counts['after_range'] = int(kept.sum())
    ids = np.flatnonzero(kept)
    if len(ids) > 20:  # the exporter's own threshold
        ids = ids[~detached.detached_mask(points[ids], .015, .25, 2, .1)]
    counts['after_detached'] = len(ids)
    if with_pixels:
        pixels = np.rint(hom[ids, :2] / hom[ids, 2:]).astype(np.int64).clip([0, 0], [639, 479])
        return points[ids], counts, pixels
    return points[ids], counts


def median_to_surface(tree, pts):
    return float(np.median(tree.query(pts)[0]) * 1000) if len(pts) else float('nan')


def parse_range(spec):
    if '-' in spec:
        lo, hi = (int(x) for x in spec.split('-', 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--frames', default='0-1191', help='intersected with the frames that have head depth')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=float, default=6.0, help='6 is real time: depth frames are 5 video frames apart')
    ap.add_argument('--spacing-mm', type=float, default=1.5, help='URDF surface sample spacing')
    ap.add_argument('--subsample', type=int, default=2500, help='max points drawn per cloud and frame')
    ap.add_argument('--size', type=int, default=900)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scratch', type=Path, default=Path('/root/claude_tmp/urdf_surface/depth'),
                    help='local lmdb copy (GPFS cannot mmap lmdb)')
    ap.add_argument('--out', type=Path, required=True, help='new directory')
    ap.add_argument('--dump', action='store_true',
                    help='also write clouds.npz (per frame: visible URDF surface, D435 hand) for the Viser builder')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--dump-classes', default='',
                    help='with --dump: also store D435 points + RGB of these SAM2 classes, e.g. 2,3,4')
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; write a new directory instead of overwriting')

    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    detached = load_file('original_drop_detached', DETACHED_PY)
    R, t = core.base_to_camera()
    surface = HandSurface(args.episode, args.spacing_mm / 1000, args.seed)
    stored, _ = core.load_fk(args.episode, R, t)
    for f in range(0, len(stored), 50):
        gap = np.abs(surface.keypoints_base(f) @ R.T + t - stored[f]).max()
        if gap > 1e-6:
            raise SystemExit(f'URDF FK does not reproduce sandwich_fk21 at frame {f}: {gap:.2e} m')
    obs = core.load_pointflow(args.episode)
    K = core.colour_K(core.colour_intrinsics(args.episode))
    meta = core.depth_metadata(args.episode)
    depth_dir = core.local_depth_dir(args.episode, args.scratch)
    masks = np.load(MASK_ROOT / args.episode / 'labels.npy', mmap_mode='r')
    keys = set(core.depth_keys(depth_dir))
    wanted = parse_range(args.frames)
    frames = [f for f in wanted if f in keys and f < len(masks)][::args.stride]
    if not frames:
        raise SystemExit('no requested frame has head depth')
    print(f'{len(frames)} depth frames {frames[0]}..{frames[-1]} at {args.fps:g} fps -> {len(frames) / args.fps:.1f} s; '
          f'{len(surface.pts):,} surface samples', flush=True)

    data, rows = {}, []
    for i, f in enumerate(frames):
        surf = surface.surface_base(f) @ R.T + t
        seen = visible(surf, K)
        tree = cKDTree(surf)
        da3, _ = core.hand_observations(obs, f)
        d435, counts = d435_gated(core.fetch_depth(depth_dir, f), meta, K, np.asarray(masks[f]), detached)
        data[f] = (surf[seen], da3, d435)
        rows.append(dict(frame=f, surface=len(surf), surface_visible=int(seen.sum()), da3=len(da3),
                         da3_to_surface_mm=median_to_surface(tree, da3),
                         d435_to_surface_mm=median_to_surface(tree, d435), d435_gates=counts))
        if i % 40 == 0:
            print(f'  {i + 1}/{len(frames)} {rows[-1]}', flush=True)

    sample = []
    for f in frames[::max(1, len(frames) // 24)]:
        sample += [c[::max(1, len(c) // 400)] for c in data[f] if len(c)]
    pts = np.concatenate(sample)
    # One box for the whole video (a per-frame box hides motion); percentiles so a few stray
    # points cannot shrink the hand to a corner of it.
    lo, hi = np.percentile(pts, .5, axis=0) - .03, np.percentile(pts, 99.5, axis=0) + .03
    print(f'shared box: X {lo[0]:.3f}..{hi[0]:.3f}  Y {lo[1]:.3f}..{hi[1]:.3f}  Z {lo[2]:.3f}..{hi[2]:.3f} m', flush=True)

    args.out.mkdir(parents=True)
    if args.dump:
        clouds = {'frames': np.array(frames)}
        classes = [int(c) for c in args.dump_classes.split(',') if c.strip()]
        video = cv2.VideoCapture(str(core.RAW_ROOT / args.episode / 'videos/head.mp4')) if classes else None
        for f in frames:
            clouds[f'surface_{f}'] = data[f][0].astype(np.float32)
            clouds[f'd435_{f}'] = data[f][2].astype(np.float32)
            if not classes:
                continue
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok_frame, bgr = video.read()
            if not ok_frame:
                raise SystemExit(f'cannot read video frame {f}')
            blob = core.fetch_depth(depth_dir, f)
            for c in classes:
                p, counts, pix = d435_gated(blob, meta, K, np.asarray(masks[f]), detached, label=c, with_pixels=True)
                clouds[f'd435c{c}_{f}'] = p.astype(np.float32)
                clouds[f'd435c{c}_rgb_{f}'] = bgr[pix[:, 1], pix[:, 0], ::-1].copy()
                clouds[f'd435c{c}_gates_{f}'] = np.array(list(counts.values()))
        np.savez_compressed(args.out / 'clouds.npz', **clouds)
    path = args.out / f'{args.episode}_urdf_surface_da3_d435.mp4'
    if args.no_video:
        (args.out / 'per_frame.json').write_text(json.dumps(rows, indent=1) + '\n')
        print(f'wrote {args.out} (no video)')
        return
    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.size, args.size))
    if not writer.isOpened():
        raise SystemExit(f'VideoWriter failed to open {path}')
    rng = np.random.default_rng(args.seed)
    for i, f in enumerate(frames):
        surf, da3, d435 = data[f]
        ax.clear()
        for cloud, color, label, alpha, size in ((surf, SURFACE_BLUE, 'URDF hand surface (visible)', .35, 2),
                                                 (da3, core.PF_ORANGE, 'PointFlow (DA3, training points)', .45, 3),
                                                 (d435, core.D435_GREEN, 'D435 (same gates)', .75, 3)):
            if not len(cloud):
                continue
            draw = cloud[rng.choice(len(cloud), args.subsample, replace=False)] if len(cloud) > args.subsample else cloud
            ax.scatter(*draw.T, s=size, c=color, alpha=alpha, depthshade=False, label=f'{label} [{len(cloud)}]')
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel='Camera X (m)', ylabel='Camera Y (m)', zlabel='Camera Z (m)')
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=22, azim=-62)
        r = rows[i]
        ax.set_title(f'frame {f}   median to URDF surface: D435 {r["d435_to_surface_mm"]:5.1f} mm   '
                     f'DA3 {r["da3_to_surface_mm"]:5.1f} mm', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        fig.canvas.draw()
        writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR))
    writer.release()
    plt.close(fig)

    def stats(key):
        x = np.array([r[key] for r in rows], float)
        return dict(median=round(float(np.nanmedian(x)), 1), p10=round(float(np.nanpercentile(x, 10)), 1),
                    p90=round(float(np.nanpercentile(x, 90)), 1), frames=int(np.isfinite(x).sum()))
    summary = dict(episode=args.episode, frames=len(frames), fps=args.fps, video=str(path),
                   metric='per frame: median distance from the cloud points to the nearest URDF hand surface sample (mm)',
                   d435_to_surface_mm=stats('d435_to_surface_mm'), da3_to_surface_mm=stats('da3_to_surface_mm'),
                   d435_gates_applied=['SAM2 v2 hand mask on 640x448 canvas', '1 px ellipse erosion',
                                       'not a depth edge (rtol 0.04, native sensor grid)', '0.1 < z < 6 m',
                                       'drop_detached.detached_mask(voxel .015, min_frac .25, touch 2, gap .1)'],
                   d435_gates_not_applicable=['model confidence > 0.3', 'track gap fill', 'track despike'],
                   urdf=str(core.URDF), surface_spacing_mm=args.spacing_mm, surface_samples=len(surface.pts))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (args.out / 'per_frame.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
