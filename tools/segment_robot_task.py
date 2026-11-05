"""Task-specific current-frame DINO initialization for the shared SAM2 video engine."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from segment_sandwich_dynamic import DynamicSegmenter
from segment_sandwich_video import CHECKPOINTS, appearance_gate, object_color, object_hand_gate


def prepare(args):
    recipe = json.loads(args.config.read_text())
    source = json.loads((args.out / "source.json").read_text())
    args.episode = Path(source["episode"])
    (args.out / "task_config.json").write_text(json.dumps(recipe, indent=2))
    folder = args.out / "frames"
    folder.mkdir(exist_ok=False)
    cap = cv2.VideoCapture(str(args.episode / "videos/head.mp4"))
    info = dict(frames=int(cap.get(7)), fps=cap.get(5), width=int(cap.get(3)), height=int(cap.get(4)))
    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        cv2.imwrite(str(folder / f"{frame:05d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame += 1
    cap.release()
    assert frame == info["frames"]
    (args.out / "video.json").write_text(json.dumps(info))
    subprocess.run(
        [
            sys.executable,
            "-u",
            "tools/segment_sandwich_video.py",
            "geometry",
            "--episode",
            str(args.episode),
            "--out",
            str(args.out),
        ],
        check=True,
    )
    initialize(args)


def initialize(args):
    import torch

    recipe = json.loads((args.out / "task_config.json").read_text())
    image = cv2.imread(str(args.out / "frames/00000.jpg"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand = np.load(args.out / "hand_allowed.npy", mmap_mode="r")[0]
    sides = np.load(args.out / "hand_side.npy", mmap_mode="r")[0]
    arms = np.load(args.out / "arm_core.npy", mmap_mode="r")[0]
    base = dict(box_threshold=0.12, text_threshold=0.12, nms_iou=0.5, objects=[])
    model = DynamicSegmenter(base, str(CHECKPOINTS / "grounding-dino-base"), str(CHECKPOINTS / "sam2.1_hiera_large.pt"))
    model.sam.set_image(rgb)
    masks = []
    items = []
    rejected = []
    all_candidates = []
    dino_ms = 0
    workspace = np.zeros(rgb.shape[:2], bool)
    wx0, wy0, wx1, wy1 = recipe["workspace_xyxy"]
    workspace[wy0:wy1, wx0:wx1] = True
    for obj in recipe["objects"]:
        if obj.get("hand_side"):
            expected = object_hand_gate(dict(object=obj), hand, sides)
            ys, xs = np.where(expected)
            if not len(xs):
                rejected.append(dict(name=obj["name"], reason="no_URDF_hand_projection"))
                continue
            x0, y0, x1, y1 = (
                max(int(xs.min()) - 25, 0),
                max(int(ys.min()) - 25, 0),
                min(int(xs.max()) + 26, rgb.shape[1]),
                min(int(ys.max()) + 26, rgb.shape[0]),
            )
        else:
            x0, y0, x1, y1 = obj["search_xyxy"]
        local = copy.deepcopy(obj)
        local["max_instances"] = 8
        model.config = dict(base, objects=[local], box_threshold=obj["threshold"], text_threshold=obj["threshold"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            candidates, _ = model.detect(rgb[y0:y1, x0:x1])
        torch.cuda.synchronize()
        dino_ms += (time.perf_counter() - started) * 1000
        chosen = None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for item in candidates:
                box = np.asarray(item["box"]) + [x0, y0, x0, y0]
                options, scores, _ = model.sam.predict(box=box, multimask_output=True)
                mask = options[int(np.argmax(scores))].astype(bool) & workspace
                before = int(mask.sum())
                arm_fraction = float((mask & arms).sum() / max(before, 1))
                if obj["label"] == 2:
                    mask &= expected
                    if mask.sum() < 50:
                        continue
                else:
                    if arm_fraction > 0.4:
                        rejected.append(dict(name=obj["name"], reason="mostly_arm", box=box.tolist()))
                        continue
                    mask &= ~arms
                    hand_union = np.zeros(mask.shape, bool)
                    for m, it in zip(masks, items):
                        if it["object"]["label"] == 2:
                            hand_union |= m
                    if (mask & hand_union).sum() > obj.get("max_initial_hand_overlap", 1.01) * max(int(mask.sum()), 1):
                        rejected.append(dict(name=obj["name"], reason="hand_misdetected_as_object", box=box.tolist()))
                        continue
                    for m, it in zip(masks, items):
                        if it["object"]["label"] == 2:
                            mask &= ~m
                    region = np.zeros(mask.shape, bool)
                    region[max(y0 - 5, 0) : min(y1 + 5, mask.shape[0]), max(x0 - 5, 0) : min(x1 + 5, mask.shape[1])] = (
                        True
                    )
                    mask &= region
                    mask &= appearance_gate(image, obj)
                    if mask.sum() < 25 or mask.sum() > 0.9 * (x1 - x0) * (y1 - y0):
                        rejected.append(dict(name=obj["name"], reason="empty_or_crop_filling_mask", box=box.tolist()))
                        continue
                item.update(
                    box=box.tolist(),
                    sam_score=float(np.max(scores)),
                    mask_pixels=int(mask.sum()),
                    search_xyxy=[x0, y0, x1, y1],
                )
                all_candidates.append(item)
                merit = float(item["score"]) + 0.1 * item["sam_score"]
                if chosen is None or merit > chosen[0]:
                    chosen = (merit, item, mask)
        if chosen:
            _, item, mask = chosen
            item["object"] = obj
            item["instance_id"] = len(items) + 1
            items.append(item)
            masks.append(mask)
    initialized = {x["object"]["name"] for x in items}
    for group, definition in recipe.get("group_definitions", {}).items():
        indices = [i for i, item in enumerate(items) if item["object"].get("merge_group") == group]
        if not indices:
            continue
        group_mask = np.logical_or.reduce([masks[i] for i in indices])
        ys, xs = np.where(group_mask)
        group_item = dict(
            object=definition,
            box=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            score=min(items[i]["score"] for i in indices),
            query="union of independently prompted visible parts",
            parts=[items[i] for i in indices],
            mask_pixels=int(group_mask.sum()),
        )
        items = [item for i, item in enumerate(items) if i not in indices] + [group_item]
        masks = [m for i, m in enumerate(masks) if i not in indices] + [group_mask]
    for i, item in enumerate(items):
        item["instance_id"] = i + 1
    np.save(args.out / "initial_masks.npy", np.asarray(masks))
    init = dict(
        config=recipe,
        detections=items,
        rejected=rejected,
        all_candidates=all_candidates,
        missing=[x["name"] for x in recipe["objects"] if x["name"] not in initialized],
        dino_ms=dino_ms,
        dino_frames=[0],
        note="Task-specific search ROIs and URDF side association; no future-frame initialization.",
    )
    (args.out / "initialization.json").write_text(json.dumps(init, indent=2))
    preview = image.copy()
    ids = np.zeros(image.shape[:2], np.uint8)
    for i in sorted(range(len(items)), key=lambda i: (items[i]["object"]["label"] == 2, -int(masks[i].sum()))):
        ids[masks[i]] = i + 1
    for i, item in enumerate(items):
        color = np.array(object_color(item))
        m = ids == i + 1
        preview[m] = (preview[m] * 0.45 + color * 0.55).astype(np.uint8)
        cv2.putText(preview, item["object"]["name"], (5, 20 + i * 18), 0, 0.45, color.tolist(), 1)
    cv2.imwrite(str(args.out / "initialization_preview.jpg"), preview)
    print(
        json.dumps(
            dict(task=recipe["task"], initialized=list(initialized), missing=init["missing"], dino_ms=dino_ms), indent=2
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "initialize"])
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    globals()[a.mode](a)


if __name__ == "__main__":
    main()
