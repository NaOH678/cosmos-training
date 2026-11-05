"""Two DA3 camera conditions, posed URDF hand surfaces and DA3 object points."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze_da3_urdf_conditions import EP, clouds, core  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--view", choices=["front", "side", "opposite", "overhead"], required=True)
parser.add_argument("--root", type=Path, default=Path("pointflow_outputs/da3_urdf_two_conditions_20261103"))
args = parser.parse_args()
angles = dict(front=(22, -62), side=(22, 28), opposite=(22, 118), overhead=(75, -62))
elev, azim = angles[args.view]
data = clouds(args.root)
reference = np.load(args.root / "urdf_surface_camera.npy", mmap_mode="r")
metrics = json.loads((args.root / "metrics_per_frame.json").read_text())
K = core.colour_K(core.colour_intrinsics(EP))
cap = cv2.VideoCapture(str(core.RAW_ROOT / EP / "videos/head.mp4"))
colors = []
for frame in range(len(reference)):
    ok, image = cap.read()
    assert ok
    p = data["unconditioned"][frame]["object"]
    u = np.rint(p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]).astype(int).clip(0, 639)
    v = np.rint(p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]).astype(int).clip(0, 479)
    colors.append(np.minimum(image[v, u, ::-1].astype(float) * 0.82 + 35, 255) / 255)
cap.release()
bounds = json.loads((args.root / "render_bounds.json").read_text())
lo, hi = np.array(bounds["lo"]), np.array(bounds["hi"])
width, height = 2000, 1000
fig = plt.figure(figsize=(20, 10), dpi=100)
heading = fig.suptitle("", fontsize=15)
fig.text(
    0.5,
    0.025,
    "Blue: posed URDF hand surface | Orange: DA3 hand | RGB: DA3 objects | camera metres; no fitting",
    ha="center",
    fontsize=12,
)
artists = []
labels = dict(
    unconditioned="Condition 1: no camera condition + last-chunk scale",
    conditioned="Condition 3: true K + static pose + last-chunk scale",
)
for column, mode in enumerate(data):
    ax = fig.add_subplot(1, 2, column + 1, projection="3d")
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
    surface = ax.scatter([], [], [], s=2, c="#168ed0", alpha=0.22, depthshade=False, label="URDF hand surface")
    objects = ax.scatter([], [], [], s=3, alpha=0.8, depthshade=False, label="DA3 objects (RGB)")
    hand = ax.scatter([], [], [], s=3, c="#ff7f0e", alpha=0.7, depthshade=False, label="DA3 hand")
    ax.legend(loc="upper right", fontsize=10)
    artists.append((ax, surface, hand, objects, mode))
fig.subplots_adjust(left=0.035, right=0.965, bottom=0.09, top=0.9, wspace=0.06)
output = args.root / f"{EP}_conditions_1_3_urdf_objects_{args.view}.mp4"
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
    f"{width}x{height}",
    "-r",
    "30",
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-threads",
    "3",
    "-crf",
    "18",
    "-preset",
    "fast",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(output),
]
surface_indices = np.linspace(0, reference.shape[1] - 1, min(4500, reference.shape[1])).astype(int)
with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
    try:
        for frame in range(len(reference)):
            heading.set_text(f"{EP} | {args.view} | frame {frame}/{len(reference) - 1} | {frame / 30:.2f} s")
            for ax, surface, hand, objects, mode in artists:
                surface._offsets3d = tuple(reference[frame, surface_indices].T)
                hand._offsets3d = tuple(data[mode][frame]["hand"].T)
                objects._offsets3d = tuple(data[mode][frame]["object"].T)
                objects.set_facecolors(colors[frame])
                value = metrics[frame][mode]
                ax.set_title(
                    f"{labels[mode]}\nHand to URDF: median {value['median_mm']:.1f} mm; p90 {value['p90_mm']:.1f} mm",
                    fontsize=12,
                )
            fig.canvas.draw()
            encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
            if frame in (0, 300, 650, 1000):
                fig.savefig(args.root / f"{args.view}_frame_{frame:04d}.png", dpi=100)
            if frame % 200 == 0:
                print(args.view, "RENDER", frame, "/", len(reference), flush=True)
    finally:
        encoder.stdin.close()
    assert encoder.wait() == 0
plt.close(fig)
(args.root / f"{args.view}_video_metadata.json").write_text(
    json.dumps(
        dict(
            frames=len(reference),
            fps=30,
            width=width,
            height=height,
            view=[elev, azim],
            conditions=[1, 3],
            bounds=bounds,
        ),
        indent=2,
    )
)
print("COMPLETE", output, flush=True)
