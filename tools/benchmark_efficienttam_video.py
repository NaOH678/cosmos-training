"""EfficientTAM video benchmark: first-frame mask init + full-video propagate.

One GPU per task, all objects share a single predictor state (same口径 as the
SAM2-large baseline in tools/segment_sandwich_video.py). A timed lazy JPEG
loader keeps per-frame loop timing inclusive of image decode; pure propagate
cost is loop time minus the loader's own read timing.

Usage:
  python tools/benchmark_efficienttam_video.py --task-dir <dir> --out <dir> [--verify]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/data/shichaojian/seg_model_compare/repos/EfficientTAM")
sys.path.insert(0, str(REPO))

CHECKPOINT = "/data/shichaojian/checkpoints/efficienttam_ti_512x512.pt"
MODEL_CFG = "configs/efficienttam/efficienttam_ti_512x512.yaml"
MODEL_NAME = "efficienttam_ti_512x512"

# sandwich task config carries no color_bgr; reuse the palette of the SAM2 pipeline.
SANDWICH_COLORS = {
    "robot_hand": (255, 145, 35),
    "lettuce": (50, 210, 70),
    "bread": (200, 75, 210),
    "egg": (30, 145, 250),
    "cheese": (20, 220, 240),
    "plate": (210, 180, 70),
    "tray": (180, 180, 180),
    "bread_rack_group": (215, 70, 215),
    "bread_slice_1": (215, 70, 215),
    "bread_slice_2": (70, 225, 230),
}


class TimedLazyLoader:
    """Sequential lazy JPEG source mirroring the SAM2 baseline CurrentFrameLoader."""

    def __init__(self, files, image_size):
        self.files = files
        self.image_size = image_size
        self.allowed = 0
        self.cached = {}
        self.load_ms = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import torch
        from efficient_track_anything.utils.misc import _load_img_as_tensor

        if index > self.allowed:
            raise RuntimeError(f"Future frame requested: {index}>{self.allowed}")
        if index not in self.cached:
            started = time.perf_counter()
            image, _, _ = _load_img_as_tensor(self.files[index], self.image_size)
            image = image.float()
            image -= torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            image /= torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            self.load_ms[index] = (time.perf_counter() - started) * 1000
            self.cached = {index: image}
        return self.cached[index]


def task_objects(task_dir):
    summary = json.loads((task_dir / "summary.json").read_text())
    names = list(summary["initialized_objects"])
    init = json.loads((task_dir / "initialization.json").read_text())
    colors = {}
    for obj in init.get("config", {}).get("objects", []):
        if obj.get("color_bgr"):
            colors[obj["name"]] = tuple(obj["color_bgr"])
    for name in names:
        colors.setdefault(name, SANDWICH_COLORS.get(name, (180, 180, 180)))
    return names, [colors[n] for n in names]


def render_frame(image, masks, names, colors, frame):
    out = image.copy()
    for idx in range(masks.shape[0]):
        color = np.array(colors[idx])
        m = masks[idx]
        out[m] = (0.45 * out[m] + 0.55 * color).astype(np.uint8)
    cv2.putText(out, f"frame {frame} | EfficientTAM {MODEL_NAME}", (8, 22), 0, 0.55, (255, 255, 255), 1)
    for row, (name, color) in enumerate(zip(names, colors)):
        cv2.putText(out, f"{row}: {name}", (8, 44 + 18 * row), 0, 0.45, color, 1)
    return out


def run(args):
    task_dir = Path(args.task_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names, colors = task_objects(task_dir)
    init_masks = np.load(task_dir / "initial_masks.npy")
    assert init_masks.shape[0] == len(names), (init_masks.shape, names)

    frames_dir = task_dir / "frames"
    files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    num_frames = len(files)
    height, width = init_masks.shape[1:]

    import torch
    import efficient_track_anything.efficienttam_video_predictor as module
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

    os.chdir(REPO)  # hydra compose resolves configs relative to the repo root

    if args.verify:
        image = cv2.imread(str(files[0]))
        for idx, name in enumerate(names):
            ys, xs = np.nonzero(init_masks[idx])
            area = len(ys)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if area else None
            print(f"channel {idx}: {name} area={area} bbox={box}", flush=True)
        cv2.imwrite(str(out / "mapping_preview.png"), render_frame(image, init_masks, names, colors, 0))
        print("wrote", out / "mapping_preview.png", flush=True)
        return

    loader = None

    def load(video_path, image_size, **kwargs):
        nonlocal loader
        loader = TimedLazyLoader(files, image_size)
        return loader, height, width

    module.load_video_frames = load

    started = time.perf_counter()
    predictor = build_efficienttam_video_predictor(
        MODEL_CFG,
        CHECKPOINT,
        device="cuda",
        # torch.compile needs Python.h (python3.10-dev), absent on this box; the
        # SAM2 baseline also ran with an uncompiled image encoder, so keep parity.
        hydra_overrides_extra=["++model.compile_image_encoder=False"],
    )
    build_s = time.perf_counter() - started

    started = time.perf_counter()
    state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=False)
    init_s = time.perf_counter() - started

    masks = np.lib.format.open_memmap(
        out / "masks.npy", mode="w+", dtype=bool, shape=(num_frames, len(names), height, width)
    )
    loop_ms, read_ms = [], []
    with torch.inference_mode():
        started = time.perf_counter()
        for idx in range(len(names)):
            predictor.add_new_mask(state, frame_idx=0, obj_id=idx + 1, mask=init_masks[idx])
        add_s = time.perf_counter() - started
        stream = predictor.propagate_in_video(state, start_frame_idx=0, reverse=False)
        for frame in range(num_frames):
            loader.allowed = frame
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fi, obj_ids, video_res_masks = next(stream)
            torch.cuda.synchronize()
            loop_ms.append((time.perf_counter() - t0) * 1000)
            read_ms.append(loader.load_ms.get(frame, 0.0))
            assert fi == frame
            for j in range(len(names)):
                masks[frame, j] = (video_res_masks[obj_ids.index(j + 1), 0] > 0).cpu().numpy()
            if frame % 200 == 0:
                print(f"frame {frame} loop_ms={loop_ms[-1]:.1f} read_ms={read_ms[-1]:.1f}", flush=True)
    propagate_s = time.perf_counter() - started
    masks.flush()

    prop_ms = [max(l - r, 0.0) for l, r in zip(loop_ms, read_ms)]
    tail = slice(5, None)
    per_frame = dict(
        p50=float(np.median(np.asarray(loop_ms)[tail])),
        p90=float(np.percentile(np.asarray(loop_ms)[tail], 90)),
        mean=float(np.mean(np.asarray(loop_ms)[tail])),
    )
    prop_only = dict(
        p50=float(np.median(np.asarray(prop_ms)[tail])),
        p90=float(np.percentile(np.asarray(prop_ms)[tail], 90)),
    )

    # Render video + keyframe previews.
    writer = cv2.VideoWriter(str(out / "segmentation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    assert writer.isOpened()
    preview_frames = sorted({0, num_frames // 5, 2 * num_frames // 5, 3 * num_frames // 5, 4 * num_frames // 5, num_frames - 1})
    render_s = time.perf_counter()
    for frame in range(num_frames):
        image = cv2.imread(str(files[frame]))
        writer.write(render_frame(image, masks[frame], names, colors, frame))
        if frame in preview_frames:
            cv2.imwrite(str(out / f"preview_{frame:04d}.png"), render_frame(image, masks[frame], names, colors, frame))
    writer.release()
    render_s = time.perf_counter() - render_s
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(out / "segmentation.mp4"), "-c:v", "libx264",
         "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out / "segmentation_h264.mp4")],
        check=True,
    )
    (out / "segmentation.mp4").unlink()

    summary = dict(
        model=MODEL_NAME,
        checkpoint=CHECKPOINT,
        task=out.name,
        frames=num_frames,
        num_objects=len(names),
        objects=names,
        per_frame_ms=per_frame,
        propagate_only_ms=prop_only,
        total_seconds=float(propagate_s),
        timing=dict(
            model_build_s=float(build_s),
            init_state_s=float(init_s),
            add_masks_s=float(add_s),
            propagate_loop_s=float(propagate_s),
            render_s=float(render_s),
            jpeg_read_ms_mean=float(np.mean(read_ms)),
        ),
        input_resolution=f"{width}x{height}",
        notes=(
            "Single GPU, all objects in one predictor state; first-frame masks via add_new_mask; "
            "official 512x512 config (inputs resized/squashed to 512x512 internally). "
            "per_frame_ms includes lazy JPEG decode inside the timed loop (SAM2 baseline口径); "
            "propagate_only_ms subtracts the loader's measured JPEG read time. First 5 frames excluded. "
            "Official defaults: fp32 inference, no autocast, postprocessing enabled."
        ),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
