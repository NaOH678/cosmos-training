"""Same-pixel DA3 hand distance to posed URDF surfaces; no alignment fitting."""

import argparse
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


def reference(root):
    target = root / "urdf_surface_camera.npy"
    display = root / "urdf_visible_display.npz"
    if target.exists() and display.exists():
        return
    hand = surface_tools.HandSurface(EP, 0.0015, seed=0)
    rotation, translation = core.base_to_camera()
    fk, _ = core.load_fk(EP, rotation, translation)
    K = core.colour_K(core.colour_intrinsics(EP))
    array = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(fk), len(hand.pts), 3))
    shown = {}
    max_fk_error = 0.0
    for frame in range(len(fk)):
        points = hand.surface_base(frame) @ rotation.T + translation
        reconstructed = hand.keypoints_base(frame) @ rotation.T + translation
        max_fk_error = max(max_fk_error, float(np.max(np.abs(reconstructed - fk[frame]))))
        assert max_fk_error < 1e-6, max_fk_error
        array[frame] = points
        visible = points[surface_tools.visible(points, K)]
        shown[f"frame_{frame}"] = visible[np.linspace(0, len(visible) - 1, min(3500, len(visible))).astype(int)].astype(
            np.float32
        )
        if frame % 200 == 0:
            print("URDF", frame, "/", len(fk), "surface", len(points), flush=True)
    array.flush()
    np.savez_compressed(display, **shown)
    (root / "urdf_reference.json").write_text(
        json.dumps(
            dict(
                urdf=str(core.URDF),
                samples_per_frame=len(hand.pts),
                frames=len(fk),
                spacing_m=0.0015,
                max_fk_reconstruction_error_m=max_fk_error,
                fitting=False,
                coordinate_frame="verified Link_Base to RGB optical camera, metres",
                surface="right hand visual meshes; no forearm; glove not represented",
            ),
            indent=2,
        )
    )


def clouds(root, historical=False):
    work = BASE / "outputs/.da3_camera_four_work" if historical else root / "cache_all_objects"
    K = core.colour_K(core.colour_intrinsics(EP))
    result = {}
    for mode in ["unconditioned", "conditioned"]:
        files = sorted(work.glob(f"{mode}_*.npz"))
        assert len(files) == 10
        with np.load(files[-1]) as payload:
            last = float(payload["norm"])
        frames = []
        for path in files:
            with np.load(path) as payload:
                start, end = int(payload["start"]), int(payload["end"])
                factor = last / (float(payload["norm"]) + 1e-6)
                assert start == len(frames)
                for index in range(end - start):
                    row = {}
                    for name in ["hand"] if historical else ["hand", "object"]:
                        prefix = "" if historical else name + "_"
                        uv, z = payload[f"{prefix}uv_{index}"], payload[f"{prefix}z_{index}"]
                        u, v = core.canvas_uv_to_colour(uv)
                        points = np.column_stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z)) * factor
                        assert np.isfinite(points).all()
                        row[name] = points.astype(np.float32)
                    frames.append(row)
        result[mode] = frames
    assert len(result["unconditioned"]) == len(result["conditioned"]) == 1192
    return result


def analyze(root, historical=False):
    data = clouds(root, historical)
    surfaces = np.load(root / "urdf_surface_camera.npy", mmap_mode="r")
    rows = []
    for frame in range(len(surfaces)):
        tree = cKDTree(surfaces[frame])
        row = dict(frame=frame)
        for mode in data:
            points = data[mode][frame]["hand"]
            distances, nearest = tree.query(points)
            delta = points - surfaces[frame][nearest]
            row[mode] = dict(
                count=len(points),
                mean_mm=float(np.mean(distances) * 1000),
                median_mm=float(np.median(distances) * 1000),
                p90_mm=float(np.percentile(distances, 90) * 1000),
                within_10mm=float(np.mean(distances <= 0.01)),
                within_20mm=float(np.mean(distances <= 0.02)),
                mean_delta_xyz_mm=(delta.mean(0) * 1000).tolist(),
            )
        rows.append(row)
        if frame % 200 == 0:
            print("METRICS", frame, flush=True)
    summary = {}
    for mode in data:
        summary[mode] = {
            name: float(np.mean([row[mode][name] for row in rows]))
            for name in ["mean_mm", "median_mm", "p90_mm", "within_10mm", "within_20mm"]
        }
        summary[mode]["mean_nearest_surface_delta_xyz_mm"] = np.mean(
            [row[mode]["mean_delta_xyz_mm"] for row in rows], axis=0
        ).tolist()
    difference = np.array([row["unconditioned"]["median_mm"] - row["conditioned"]["median_mm"] for row in rows])
    summary["paired_frames"] = dict(
        condition3_better_by_gt_1mm=int((difference > 1).sum()),
        condition1_better_by_gt_1mm=int((difference < -1).sum()),
        within_1mm=int((np.abs(difference) <= 1).sum()),
        mean_improvement_3_vs_1_mm=float(difference.mean()),
    )
    name = "historical_metrics" if historical else "metrics"
    (root / f"{name}_per_frame.json").write_text(json.dumps(rows, indent=2))
    (root / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    if not historical:
        lo, hi = surfaces.min(axis=(0, 1)), surfaces.max(axis=(0, 1))
        for frames in data.values():
            for row in frames:
                for points in row.values():
                    if len(points):
                        lo, hi = np.minimum(lo, points.min(0)), np.maximum(hi, points.max(0))
        (root / "render_bounds.json").write_text(
            json.dumps(dict(lo=(lo - 0.03).tolist(), hi=(hi + 0.03).tolist()), indent=2)
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pointflow_outputs/da3_urdf_two_conditions_20261103"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    reference(args.root)
    if not args.reference_only:
        analyze(args.root, args.historical)
