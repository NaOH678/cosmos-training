"""Fixed-camera A-E videos from evaluated points, preserving sensor timestamps."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VIEWS = dict(front=(22, -62), side=(22, 28), opposite=(22, 118), overhead=(75, -62))
LABELS = dict(
    A="A: filtered D435",
    B="B: DA3 single image",
    C="C: DA3 + true K / static pose",
    D="D: DA3 + sensor scale",
    E="E: DA3 + camera condition + sensor scale",
)


def main(args):
    root = args.root / args.task
    rows = json.loads((root / "per_frame.json").read_text())
    paths = sorted((root / "frames").glob("*.npz"))
    assert len(paths) == len(rows)
    # Shared full-data bounds: failures are deliberately not clipped away.
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for path in paths:
        with np.load(path) as n:
            for key in ["urdf"] + [m + "_points" for m in LABELS]:
                p = n[key]
                if len(p):
                    low, high = np.minimum(low, p.min(0)), np.maximum(high, p.max(0))
    low -= 0.03
    high += 0.03
    fig = plt.figure(figsize=(24, 14), dpi=100)
    heading = fig.suptitle("", fontsize=18)
    rgbax = fig.add_subplot(2, 3, 1)
    rgbax.axis("off")
    rgbax.set_title("Same RGB / SAM2 mask; selected 1024 points", fontsize=13)
    rgbimage = rgbax.imshow(np.zeros((240, 320, 3), np.uint8))
    artists = []
    for idx, mode in enumerate(LABELS):
        ax = fig.add_subplot(2, 3, idx + 2, projection="3d")
        ax.set(
            xlim=(low[0], high[0]),
            ylim=(low[1], high[1]),
            zlim=(low[2], high[2]),
            xlabel="Camera X (m)",
            ylabel="Camera Y (m)",
            zlabel="Camera Z (m)",
        )
        ax.set_box_aspect(high - low)
        ax.view_init(elev=VIEWS[args.view][0], azim=VIEWS[args.view][1])
        surface = ax.scatter([], [], [], s=2, c="#168ed0", alpha=0.30, depthshade=False)
        cloud = ax.scatter([], [], [], s=5, alpha=0.85, depthshade=False)
        artists.append((mode, ax, surface, cloud))
    fig.text(
        0.5,
        0.025,
        "Blue: URDF hand surface | Orange: selected hand points | RGB: selected objects | no fitting; shared fixed axes",
        ha="center",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.075, top=0.92, wspace=0.12, hspace=0.2)
    fps = 30
    times = np.array([r["time"] for r in rows])
    times -= times[0]
    duration = float(times[-1] + np.median(np.diff(times)))
    count = int(round(duration * fps))
    lookup = np.searchsorted(times, np.arange(count) / fps, side="right") - 1
    lookup = np.clip(lookup, 0, len(rows) - 1)
    repeats = np.bincount(lookup, minlength=len(rows))
    output = root / f"{args.task}_A_to_E_{args.view}.mp4"
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
        "2400x1400",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        try:
            for i, path in enumerate(paths):
                with np.load(path) as n:
                    rgbimage.set_data(n["rgb"][..., ::-1])
                    heading.set_text(
                        f"{args.task} | {args.view} | source frame {rows[i]['frame']} | {times[i]:.2f}s | causal A-E depth comparison"
                    )
                    reference = n["urdf"]
                    take = np.linspace(0, len(reference) - 1, min(3500, len(reference))).astype(int)
                    for mode, ax, surface, cloud in artists:
                        surface._offsets3d = tuple(reference[take].T)
                        cloud._offsets3d = tuple(n[mode + "_points"].T)
                        cloud.set_facecolors(n[mode + "_colors"] / 255)
                        r = rows[i][mode]
                        p = r["selected_hand"]["p90_mm"]
                        ptext = f"{p:.1f} mm" if p is not None else "no hand points"
                        extra = f" | scale {r['scale']:.3f}" if mode in ["D", "E"] else ""
                        ax.set_title(
                            f"{LABELS[mode]}\nSelected hand P90: {ptext} | n={r['selected_hand']['n']}{extra}",
                            fontsize=12,
                        )
                    fig.canvas.draw()
                    pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes()
                    for _ in range(int(repeats[i])):
                        encoder.stdin.write(pixels)
                    if i in {0, len(paths) // 2, len(paths) - 1}:
                        fig.savefig(root / f"{args.view}_preview_{rows[i]['frame']:04d}.png", dpi=100)
                    if i % 80 == 0:
                        print(args.task, args.view, i, "/", len(paths), flush=True)
        finally:
            encoder.stdin.close()
        assert encoder.wait() == 0
    plt.close(fig)
    metadata = dict(
        task=args.task,
        view=args.view,
        angles=VIEWS[args.view],
        depth_observations=len(paths),
        video_frames=count,
        fps=fps,
        duration=count / fps,
        bounds=[low.tolist(), high.tolist()],
        source_frame_ids=[r["frame"] for r in rows],
        note="All sources held to next actual depth observation, causal display; no interpolation. Same bounds include all selected outliers.",
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pointflow_outputs/depth_source_comparison_20261103"))
    parser.add_argument("--task", choices=["sandwich", "dropper", "micro"], required=True)
    parser.add_argument("--view", choices=VIEWS, default="front")
    main(parser.parse_args())
