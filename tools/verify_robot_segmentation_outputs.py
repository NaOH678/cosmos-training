"""Check complete video decoding, mask constraints and sampled tracking consistency."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def verify(out):
    info = json.loads((out / "video.json").read_text())
    init = json.loads((out / "initialization.json").read_text())
    config = json.loads((out / "task_config.json").read_text())
    instances = np.load(out / "instances.npy", mmap_mode="r")
    labels = np.load(out / "labels.npy", mmap_mode="r")
    arms = np.load(out / "arm_core.npy", mmap_mode="r")
    sides = np.load(out / "hand_side.npy", mmap_mode="r") if (out / "hand_side.npy").exists() else None
    hands = np.load(out / "hand_allowed.npy", mmap_mode="r")
    initial = np.load(out / "initial_masks.npy", mmap_mode="r")
    expected = (info["frames"], info["height"], info["width"])
    assert instances.shape == labels.shape == expected
    x0, y0, x1, y1 = config["workspace_xyxy"]
    outside = np.ones(expected[1:], bool)
    outside[y0:y1, x0:x1] = False
    violations = dict(workspace=0, hand_side=0, object_arm=0, static_support=0)
    dynamic = config.get("dynamic_objects")
    counts = []
    for frame in range(expected[0]):
        inst = instances[frame]
        violations["workspace"] += int(np.count_nonzero(inst[outside]))
        frame_counts = []
        for idx, item in enumerate(init["detections"]):
            mask = inst == idx + 1
            obj = item["object"]
            frame_counts.append(int(mask.sum()))
            assert np.all(labels[frame][mask] == obj["label"])
            if obj["label"] == 2:
                allowed = hands[frame]
                if sides is not None and obj.get("hand_side"):
                    allowed = sides[frame] == (2 if obj["hand_side"] == "right" else 3)
                violations["hand_side"] += int(np.count_nonzero(mask & ~allowed))
            else:
                violations["object_arm"] += int(np.count_nonzero(mask & arms[frame]))
            if obj.get("static_support") or (dynamic is not None and obj["name"] not in dynamic):
                violations["static_support"] += int(np.count_nonzero(mask & ~initial[idx]))
        counts.append(frame_counts)
    assert not any(violations.values()), violations
    capture = cv2.VideoCapture(str(out / "segmentation_h264.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    selected = set([0, 60, expected[0] // 5, expected[0] // 2, 4 * expected[0] // 5, expected[0] - 1])
    tiles = []
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if decoded in selected:
            tiles.append(image)
        decoded += 1
    capture.release()
    assert decoded == expected[0], (decoded, expected[0])
    assert abs(fps - info["fps"]) < 0.001
    cv2.imwrite(str(out / "full_video_contact_sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    benchmark = np.load(out / "benchmark_masks.npy", mmap_mode="r")
    benchmark_indices = json.loads((out / "benchmark.json").read_text())["indices"]
    if dynamic is not None:
        expected_indices = [i for i, item in enumerate(init["detections"]) if item["object"]["name"] in dynamic]
        assert benchmark_indices == expected_indices
    worker_indices = []
    comparisons = []
    for worker in range(8):
        details = json.loads((out / f"worker{worker}.json").read_text())
        worker_indices.extend(details["indices"])
        if not details["indices"]:
            continue
        raw = np.load(out / f"worker{worker}_masks.npy", mmap_mode="r")
        for pos, idx in enumerate(details["indices"]):
            for frame in (0, 30, 60, 119):
                a, b = raw[frame, pos], benchmark[frame, benchmark_indices.index(idx)]
                union = np.count_nonzero(a | b)
                comparisons.append(float(np.count_nonzero(a & b) / union) if union else 1.0)
    assert sorted(worker_indices) == benchmark_indices, (worker_indices, benchmark_indices)
    counts = np.array(counts)
    report = dict(
        source=json.loads((out / "source.json").read_text()),
        decoded_frames=decoded,
        fps=fps,
        constraint_violations=violations,
        sampled_single_vs_eight_gpu_iou=dict(mean=float(np.mean(comparisons)), minimum=float(np.min(comparisons))),
        objects=[
            dict(name=item["object"]["name"], zero_mask_frames=int(np.count_nonzero(counts[:, idx] == 0)))
            for idx, item in enumerate(init["detections"])
        ],
        note="Constraint checks are not ground-truth accuracy measurements. A zero mask may reflect complete occlusion.",
    )
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    verify(parser.parse_args().out)
