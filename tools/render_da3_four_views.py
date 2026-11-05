import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

BASE = Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base")
sys.path.insert(0, str(BASE / "tools"))
import render_fk_vs_pointflow as core

work = BASE / "outputs/.da3_camera_four_work"
parser = argparse.ArgumentParser()
parser.add_argument("--view", choices=["side", "opposite", "overhead"], required=True)
args = parser.parse_args()
angles = {"side": (22, 28), "opposite": (22, 118), "overhead": (75, -62)}
elev, azim = angles[args.view]
root = Path("pointflow_outputs/da3_four_views_20261103")
out = root / f"episode_0013_20260731_133649_four_{args.view}_continuous.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
EP = "episode_0013_20260731_133649"
R, t = core.base_to_camera()
fk, _ = core.load_fk(EP, R, t)
K = core.colour_K(core.colour_intrinsics(EP))
T = len(fk)
data = [[] for _ in range(T)]
for mode in ["unconditioned", "conditioned"]:
    files = sorted(work.glob(f"{mode}_*.npz"))
    assert len(files) == 10
    with np.load(files[-1]) as n:
        last = float(n["norm"])
    expected = 0
    for path in files:
        with np.load(path) as n:
            start, end = int(n["start"]), int(n["end"])
            norm = float(n["norm"])
            assert start == expected and norm > 0
            expected = end
            for i, f in enumerate(range(start, end)):
                uv = n[f"uv_{i}"].astype(float)
                z = n[f"z_{i}"].astype(float)
                assert np.isfinite(z).all() and (z > 0).all()
                u, v = core.canvas_uv_to_colour(uv)
                p = np.column_stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z))
                data[f].extend([p * (last / (norm + 1e-6)), p])
        print("Loaded", path.name, flush=True)
    assert expected == T
for f in range(T):
    assert len(data[f]) == 4
    assert all(p.shape == data[f][0].shape for p in data[f])
lo = fk.min(axis=(0, 1))
hi = fk.max(axis=(0, 1))
for row in data:
    for p in row:
        lo = np.minimum(lo, p.min(0))
        hi = np.maximum(hi, p.max(0))
lo -= 0.03
hi += 0.03
print("Bounds", lo, hi, flush=True)
W, H = 3200, 850
fig = plt.figure(figsize=(32, 8.5), dpi=100)
heading = fig.suptitle("", fontsize=18)
fig.text(
    0.5,
    0.025,
    "DA3 depth | same RGB, weights, hand pixels and true-K scaling/reprojection | FK reference only | no D435 depth",
    ha="center",
    fontsize=13,
)
labels = [
    "No camera condition | Last-chunk scale",
    "No camera condition | Own-chunk scale",
    "True K + static pose condition | Last-chunk scale",
    "True K + static pose condition | Own-chunk scale",
]
artists = []
for col, label in enumerate(labels):
    ax = fig.add_subplot(1, 4, col + 1, projection="3d")
    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        zlim=(lo[2], hi[2]),
        xlabel="Camera X (m)",
        ylabel="Camera Y (m)",
        zlabel="Camera Z (m)",
    )
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    points = ax.scatter([], [], [], s=3, c=core.PF_ORANGE, alpha=0.4, depthshade=False, label="DA3 hand")
    joints = ax.scatter(
        [], [], [], s=28, c=core.FK_BLUE, edgecolors="white", linewidths=0.5, depthshade=False, label="FK-21"
    )
    bones = Line3DCollection([fk[0, [a, b]] for a, b in core.EDGES], colors=core.FK_BLUE, linewidths=2)
    ax.add_collection3d(bones)
    ax.set_title(label, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    artists.append((points, joints, bones))
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.13, top=0.86, wspace=0.08)
cmd = [
    "ffmpeg",
    "-y",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    f"{W}x{H}",
    "-r",
    "30",
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-threads",
    "4",
    "-crf",
    "18",
    "-preset",
    "fast",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(out),
]
with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
    try:
        for f in range(T):
            heading.set_text(f"{EP} | {args.view} view | frame {f}/{T - 1} | {f / 30:.2f} s")
            for col, (points, joints, bones) in enumerate(artists):
                points._offsets3d = tuple(data[f][col].T)
                joints._offsets3d = tuple(fk[f].T)
                bones.set_segments([fk[f, [a, b]] for a, b in core.EDGES])
            fig.canvas.draw()
            encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
            if f in (300, 650):
                fig.savefig(root / f"{args.view}_frame_{f:04d}.png", dpi=100)
            if f % 100 == 0:
                print("Rendered", f, "/", T, flush=True)
    finally:
        encoder.stdin.close()
    assert encoder.wait() == 0
plt.close(fig)
print("VIDEO COMPLETE", out, flush=True)

(root / f"{args.view}_metadata.json").write_text(
    json.dumps(
        dict(
            frames=T,
            fps=30,
            resolution=[W, H],
            elev=elev,
            azim=azim,
            bounds=[lo.tolist(), hi.tolist()],
            conditions=labels,
            cache=str(work),
            data_changed=False,
        ),
        indent=2,
    )
)
