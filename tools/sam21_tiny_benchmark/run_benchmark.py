"""SAM2.1-tiny video benchmark: first-frame mask init + full-video tracking.

Single GPU, all objects in one predictor state (same protocol as the SAM2-large
baseline in tools/segment_sandwich_video.py). Lazy per-frame JPEG loader mirrors
the baseline so per-frame times include JPEG decode; loader time is recorded
separately to also report pure propagate time.

Usage: python run_benchmark.py <task>
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _task_defs import (
    BASELINE_P50_MS,
    CHECKPOINT,
    MODEL,
    OUT_ROOT,
    PYTHON,
    SAM2_CONFIG,
    SAM2_PYLIBS,
    TASKS,
    task_names_colors,
)

sys.path.insert(0, SAM2_PYLIBS)


class TimedLazyLoader:
    """Sequential lazy JPEG loader; records per-frame load+preprocess ms."""

    def __init__(self, files, size):
        self.files, self.size, self.allowed, self.cached = files, size, 0, {}
        self.load_ms = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import torch
        from sam2.utils.misc import _load_img_as_tensor

        if index > self.allowed:
            raise RuntimeError(f"Future frame requested: {index}>{self.allowed}")
        if index not in self.cached:
            started = time.perf_counter()
            image, _, _ = _load_img_as_tensor(self.files[index], self.size)
            image = image.float()
            image -= torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            image /= torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            self.load_ms[index] = (time.perf_counter() - started) * 1000
            self.cached = {index: image}
        return self.cached[index]


def visualize(task, task_dir, out_dir, names, colors, masks):
    info = json.loads((task_dir / "video.json").read_text())
    frames, height, width = info["frames"], info["height"], info["width"]
    n = len(names)
    keyframes = sorted({0, frames // 5, 2 * frames // 5, 3 * frames // 5, 4 * frames // 5, frames - 1})
    writer = cv2.VideoWriter(
        str(out_dir / "segmentation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    assert writer.isOpened()
    legend = [(name, tuple(int(c) for c in color)) for name, color in zip(names, colors)]
    for frame in range(frames):
        image = cv2.imread(str(task_dir / "frames" / f"{frame:05d}.jpg"))
        assert image is not None
        for i, color in enumerate(colors):
            m = masks[frame, i]
            if not m.any():
                continue
            image[m] = (0.45 * image[m].astype(np.float32) + 0.55 * np.asarray(color)).astype(np.uint8)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, tuple(int(c) for c in color), 1)
        for row, (name, color) in enumerate(legend):
            y = 20 + 19 * row
            cv2.rectangle(image, (5, y - 11), (20, y + 2), color, -1)
            cv2.putText(image, name, (25, y), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"{MODEL} frame {frame}/{frames - 1}", (5, height - 8), 0, 0.5, (255, 255, 255), 1)
        writer.write(image)
        if frame in keyframes:
            cv2.imwrite(str(out_dir / f"preview_{frame:04d}.png"), image)
    writer.release()


def main(task):
    task_dir = TASKS[task]
    out_dir = OUT_ROOT / MODEL / task
    out_dir.mkdir(parents=True, exist_ok=False)
    names, colors = task_names_colors(task_dir)
    info = json.loads((task_dir / "video.json").read_text())
    frames_total, height, width = info["frames"], info["height"], info["width"]
    n = len(names)
    initial = np.load(task_dir / "initial_masks.npy")
    assert initial.shape == (n, height, width), initial.shape

    import sam2.sam2_video_predictor as module
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    files = sorted((task_dir / "frames").glob("*.jpg"))
    assert len(files) == frames_total, (len(files), frames_total)
    loader = None

    def load(video_path, image_size, **kwargs):
        nonlocal loader
        loader = TimedLazyLoader(files, image_size)
        return loader, height, width

    module.load_video_frames = load
    predictor = build_sam2_video_predictor(SAM2_CONFIG, CHECKPOINT, device="cuda")

    masks = np.lib.format.open_memmap(
        out_dir / "masks.npy", mode="w+", dtype=bool, shape=(frames_total, n, height, width)
    )
    times, load_times = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(task_dir / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for idx in range(n):
            predictor.add_new_mask(state, frame_idx=0, obj_id=idx + 1, mask=initial[idx])
        stream = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=frames_total - 1, reverse=False)
        loop_started = time.perf_counter()
        for frame in range(frames_total):
            loader.allowed = frame
            torch.cuda.synchronize()
            started = time.perf_counter()
            fi, ids, logits = next(stream)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
            load_times.append(loader.load_ms.get(frame, 0.0))
            assert fi == frame
            for j in range(n):
                masks[frame, j] = (logits[ids.index(j + 1), 0] > 0).cpu().numpy()
            if frame % 200 == 0:
                print("track", task, frame, f"{times[-1]:.1f} ms", flush=True)
        total_seconds = time.perf_counter() - loop_started
    masks.flush()
    del masks

    body = slice(5, None)
    full = np.asarray(times)[body]
    propagate = np.asarray(times)[body] - np.asarray(load_times)[body]
    summary = dict(
        model=MODEL,
        checkpoint=CHECKPOINT,
        config=SAM2_CONFIG,
        task=task,
        frames=frames_total,
        num_objects=n,
        objects=names,
        per_frame_ms=dict(
            p50=float(np.median(full)), p90=float(np.percentile(full, 90)), mean=float(full.mean())
        ),
        propagate_only_ms=dict(p50=float(np.median(propagate)), p90=float(np.percentile(propagate, 90))),
        baseline_sam2_large_p50_ms=BASELINE_P50_MS[task],
        total_seconds=float(total_seconds),
        input_resolution=f"{width}x{height}",
        sam2_image_size=1024,
        notes=(
            "Single A800, all objects in one predictor state, full video, first 5 frames excluded. "
            "per_frame_ms includes lazy JPEG decode+ImageNet normalization (matches baseline protocol); "
            "propagate_only_ms subtracts the measured loader time. "
            "SAM2 resizes the 640px-longest-side input to its internal image_size=1024 by default. "
            "init/add_new_mask time excluded; total_seconds covers the propagate loop only."
        ),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_frame_times.json").write_text(json.dumps(dict(full=times, jpeg_load=load_times)))
    print(json.dumps(summary, indent=2), flush=True)

    masks = np.load(out_dir / "masks.npy", mmap_mode="r")
    visualize(task, task_dir, out_dir, names, colors, masks)

    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(out_dir / "segmentation.mp4"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_dir / "segmentation_h264.mp4"),
        ],
        check=True,
    )
    (out_dir / "segmentation.mp4").unlink()
    print("done", task, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
