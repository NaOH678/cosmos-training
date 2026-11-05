"""Validate exported full clips and make RGB/3D spot-check plots without a browser."""

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import cv2
import msgspec
import numpy as np
from render_d435_point_accuracy import Surface

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "pointflow_outputs/fast_foundationstereo_v4_20261105"


def validate(task):
    p = BASE / task
    v0 = ROOT / "pointflow_outputs/cotracker_real_20261104" / task
    source = json.loads((p / "source.json").read_text())
    config = json.loads((p / "task_config.json").read_text())
    T = source["frames"]
    pairs = json.loads((p / "ir_pair_index.json").read_text())
    cal = json.loads((p / "calibration.json").read_text())
    rows = json.loads((p / "per_frame.json").read_text())
    assert len(rows) == T and sum(r["sensor_update"] for r in rows) == len(pairs)
    assert len(set(x["right_video_index"] for x in pairs)) == len(pairs)
    for pair in pairs:
        assert pair["capture_time"] <= pair["rgb_time"] + 1e-6 and abs(pair["left_time"] - pair["right_time"]) <= 0.001
    coverage = []
    counts = []
    for f in range(T):
        with np.load(p / "frames" / f"{f:06d}.npz") as n, np.load(v0 / "pointflow/frames" / f"{f:06d}.npz") as old:
            for key in ["ids", "uv", "instances", "labels", "colors"]:
                np.testing.assert_array_equal(n[key], old[key])
            xyz = n["xyz"][n["valid"]]
            assert np.isfinite(xyz).all() and ((xyz[:, 2] >= 0.15) & (xyz[:, 2] <= 2.5)).all()
            counts.append(len(xyz))
            coverage.append(float(n["valid"].mean()))
    for pair in pairs:
        depth = np.load(p / "native_depth" / f"{pair['sequence']:06d}.npy")
        assert depth.shape == (480, 640) and np.isfinite(depth).all()
    h = (p / f"{task}_fast_foundationstereo_v4.html").read_text()
    m = json.JSONDecoder().raw_decode(h.split("window.__VISER_EXPORT_METADATA__ = ", 1)[1])[0]
    rec = msgspec.msgpack.decode(
        gzip.decompress(
            base64.b64decode(re.search(r'<script[^>]*id="viser-scene-data"[^>]*>(.*?)</script>', h, re.S).group(1))
        )
    )
    assert m["frames"] == T and len(m["frame_ids"]) == len(range(0, T, 2)) and m["urdf_visible_by_default"]
    assert abs(rec["durationSeconds"] - T / source["fps"]) < 1e-6
    method = json.loads((p / "method.json").read_text())
    assert method["host"] == "dev-fb1f64cc-0" and not method["sdk_depth_used"] and method["V4_preserved"]
    bench = json.loads((p / "benchmark.json").read_text())
    actual = [r["sensor_ms"] + r["filter_ms"] for r in rows if r["sensor_update"]]
    inter = [r["filter_ms"] for r in rows if not r["sensor_update"]]
    summary = dict(
        task=task,
        frames=T,
        stereo_pairs=len(pairs),
        max_stereo_dt_ms=cal["max_pair_dt_ms"],
        counter_offsets=sorted(set(x["sequence"] - x["right_sequence"] for x in pairs)),
        ids_uv_labels_instances_colors_equal=True,
        valid_xyz_finite=True,
        native_depth_meters_range=[0.15, 2.5],
        urdf=config["urdf"],
        urdf_visible=True,
        display_frames=len(m["frame_ids"]),
        duration_seconds=rec["durationSeconds"],
        median_valid_points=float(np.median(counts)),
        median_valid_fraction=float(np.median(coverage)),
        api_p50_ms=bench["api_p50_ms"],
        api_p95_ms=bench["api_p95_ms"],
        cpu_depth_update_post_p50_ms=float(np.median(actual)),
        cpu_depth_update_post_p95_ms=float(np.percentile(actual, 95)),
        cpu_intermediate_post_p50_ms=float(np.median(inter)),
        full_model_plus_post_estimate_ms=bench["api_p50_ms"] + float(np.median(actual)),
        latency_note="Model and postprocessing profiled separately; sum is an estimate, excludes RGB/IR IO, SAM2/CoTracker and output IO; no full pipeline real-time claim.",
        v4_surface_sha256=hashlib.sha256(
            (ROOT / "cosmos_framework/inference/robot_policy/causal_object_surface.py").read_bytes()
        ).hexdigest(),
    )
    (p / "validation.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = Surface(Path(source["episode"]), config)
    fig = plt.figure(figsize=(16, 8))
    for col, f in enumerate([0, T // 3, 2 * T // 3, T - 1]):
        ax = fig.add_subplot(2, 4, col + 1)
        ax.imshow(cv2.cvtColor(cv2.imread(str(v0 / "frames" / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{task} RGB {f}")
        with np.load(p / "frames" / f"{f:06d}.npz") as n:
            x = n["xyz"][n["valid"]]
            c = n["colors"][n["valid"]] / 255.0
            c[n["labels"][n["valid"]] == 2] = [1, 0.5, 0.05]
        ref = surface.at(f)[::4]
        assert len(ref) > 0 and np.isfinite(ref).all()
        ax = fig.add_subplot(2, 4, col + 5, projection="3d")
        ax.scatter(x[:, 0], x[:, 2], -x[:, 1], c=c, s=1)
        ax.scatter(ref[:, 0], ref[:, 2], -ref[:, 1], c="#1ea0eb", s=0.1, alpha=0.5)
        ax.view_init(elev=15, azim=-65)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.3, 1.25)
        ax.set_zlim(-0.4, 0.4)
        ax.set_xlabel("X m")
        ax.set_ylabel("Z m")
        ax.set_title(f"Fast + V4 ({len(x)} points)")
    fig.tight_layout()
    fig.savefig(p / "quality_check.png", dpi=140)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    result = [validate(task) for task in ["scissors", "bottle"]]
    (BASE / "validation_summary.json").write_text(json.dumps(result, indent=2))
