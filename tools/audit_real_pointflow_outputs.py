"""Audit exported measured points; no audit statistic modifies the data."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_d435_point_accuracy import Surface
from run_real_cotracker_pipeline import save


def audit(out):
    dest = out / "pointflow"
    report = json.loads((dest / "report.json").read_text())
    rows = json.loads((dest / "per_frame.json").read_text())
    T = report["frames"]
    ep = Path(json.loads((out / "source.json").read_text())["episode"])
    config = json.loads((out / "task_config.json").read_text())
    ref = Surface(ep, config)
    measured = []
    previous = None
    gaps = {i + 1: 0 for i in range(len(config["objects"]))}
    geometry_rows = []
    for frame in range(T):
        with np.load(dest / "frames" / f"{frame:06d}.npz") as n:
            d = {k: n[k] for k in n.files}
        valid = d["valid"]
        points = d["xyz"][valid]
        ids = d["ids"][valid]
        instances = d["instances"][valid]
        for instance in gaps:
            if not np.any(instances == instance):
                gaps[instance] += 1
        diag = dict(frame=frame, valid_points=int(valid.sum()), invalid_points=int((~valid).sum()), per_instance={})
        for instance in np.unique(instances):
            p = points[instances == instance]
            diag["per_instance"][int(instance)] = dict(
                n=len(p), z_p10=float(np.percentile(p[:, 2], 10)), z_p90=float(np.percentile(p[:, 2], 90))
            )
        if previous is not None:
            _, ii, jj = np.intersect1d(ids, previous["ids"], assume_unique=True, return_indices=True)
            delta = points[ii] - previous["xyz"][jj]
            diag["shared_ids"] = len(ii)
            diag["tracks_displacement_gt80mm"] = int((np.linalg.norm(delta, axis=1) > 0.08).sum())
            diag["max_displacement_mm"] = float(np.linalg.norm(delta, axis=1).max() * 1000) if len(ii) else None
        if frame % 10 == 0 or frame == T - 1:
            hand = d["xyz"][valid & (d["labels"] == 2)]
            surface = ref.at(frame)
            dist = cKDTree(surface).query(hand, workers=1)[0] * 1000 if len(hand) else np.empty(0)
            record = dict(
                frame=frame,
                hand_points=len(hand),
                mean_mm=float(dist.mean()) if len(dist) else None,
                p90_mm=float(np.percentile(dist, 90)) if len(dist) else None,
            )
            measured.append(record)
        geometry_rows.append(diag)
        previous = dict(ids=ids, xyz=points)
    # Select both poor-depth frames and high-motion frames, rather than only pretty frames.
    worst = [
        r["frame"] for r in sorted([r for r in measured if r["p90_mm"] is not None], key=lambda r: -r["p90_mm"])[:3]
    ]
    motion = [r["frame"] for r in sorted(geometry_rows, key=lambda r: -(r.get("max_displacement_mm") or 0))[:3]]
    selected = sorted(set([0, T // 2, T - 1] + worst + motion))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 3.3 * len(selected)))
    for row, frame in enumerate(selected):
        with np.load(dest / "frames" / f"{frame:06d}.npz") as n:
            d = {k: n[k] for k in n.files}
        image = cv2.imread(str(out / "frames" / f"{frame:05d}.jpg"))
        a = fig.add_subplot(len(selected), 3, row * 3 + 1)
        a.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title(f"frame {frame}")
        a.axis("off")
        valid = d["valid"]
        xyz = d["xyz"][valid]
        c = d["colors"][valid] / 255.0
        c[d["labels"][valid] == 2] = [1, 0.5, 0.05]
        for column, (axis, name) in enumerate([(0, "X"), (1, "Y")], 1):
            a = fig.add_subplot(len(selected), 3, row * 3 + column + 1)
            a.scatter(xyz[:, axis], xyz[:, 2], s=1, c=c)
            a.set(xlabel=name + " m", ylabel="Z m", title=f"{int(valid.sum())}/{len(valid)} valid")
    fig.tight_layout()
    fig.savefig(out / "depth_motion_audit.jpg", dpi=100)
    plt.close(fig)
    values = [r for r in measured if r["p90_mm"] is not None]
    result = dict(
        frames=T,
        urdf=report["urdf"],
        audit_frames=selected,
        hand_urdf=dict(
            sampled_frames=len(values),
            mean_frame_mean_mm=float(np.mean([r["mean_mm"] for r in values])) if values else None,
            median_frame_p90_mm=float(np.median([r["p90_mm"] for r in values])) if values else None,
            max_frame_p90_mm=float(np.max([r["p90_mm"] for r in values])) if values else None,
        ),
        missing_point_frames={config["objects"][i - 1]["name"]: count for i, count in gaps.items()},
        fast_track_step_frames=sum(r.get("tracks_displacement_gt80mm", 0) > 0 for r in geometry_rows),
        note="URDF hand surface is a reference only, not object ground truth or a fitted correction. Statistics do not alter data; missing points may be occluded, mask-lost, or filtered.",
    )
    save(out / "depth_audit.json", result)
    save(out / "depth_audit_frames.json", geometry_rows)
    save(out / "hand_urdf_audit.json", measured)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    for video in ["segmentation_h264.mp4", "pointflow_rgb_h264.mp4"]:
        cap = cv2.VideoCapture(str(out / video))
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == T, (video, count, T)
    print("FULL VIDEO DECODE OK", out.name, T, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    audit(parser.parse_args().task)
