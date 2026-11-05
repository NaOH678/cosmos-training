"""Inspect saved EdgeTAM masks against the actual initialization and shared SAM2 output gates."""

import json
from pathlib import Path

import cv2
import numpy as np
from segment_sandwich_video import appearance_gate, object_color, object_hand_gate

ROOT = Path("pointflow_outputs/edgetam_mask_audit_20261102")
CASES = {
    "sandwich": ("pointflow_outputs/sandwich_bread_variants_20261030_v2/separate", [0, 238, 476, 820, 953, 1191]),
    "dropper": ("pointflow_outputs/dropper_micro_requested_20261030/dropper_mix", [0, 450, 900, 1125, 1800, 2250]),
    "micro": ("pointflow_outputs/dropper_micro_requested_20261030/micropipette", [0, 60, 421, 843, 1264, 1686, 2107]),
}


def overlay(rgb, masks, detections, title):
    image = rgb.copy()
    order = sorted(range(len(masks)), key=lambda i: (detections[i]["object"]["label"] == 2, -int(masks[i].sum())))
    ids = np.zeros(rgb.shape[:2], np.uint8)
    for i in order:
        ids[masks[i]] = i + 1
    for i, item in enumerate(detections):
        mask = ids == i + 1
        image[mask] = (0.45 * image[mask] + 0.55 * np.array(object_color(item))).astype(np.uint8)
    cv2.rectangle(image, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 21), 0, 0.5, (255, 255, 255), 1)
    return image


def main():
    ROOT.mkdir(exist_ok=True)
    report = {}
    for task, (source, frames) in CASES.items():
        source = Path(source)
        out = ROOT / task
        out.mkdir(exist_ok=True)
        edge = Path("pointflow_outputs/seg_model_compare_20261031/EdgeTAM") / task
        masks = np.load(edge / "masks.npy", mmap_mode="r")
        initial = np.load(source / "initial_masks.npy", mmap_mode="r")
        init = json.loads((source / "initialization.json").read_text())
        detections = init["detections"]
        names = [item["object"]["name"] for item in detections]
        assert names == json.loads((edge / "meta.json").read_text())["names"]
        config = (
            json.loads((source / "task_config.json").read_text())
            if (source / "task_config.json").exists()
            else init["config"]
        )
        allowed = np.load(source / "hand_allowed.npy", mmap_mode="r")
        arms = np.load(source / "arm_core.npy", mmap_mode="r")
        sides = np.load(source / "hand_side.npy", mmap_mode="r") if (source / "hand_side.npy").exists() else None
        hand_ids = [i for i, item in enumerate(detections) if item["object"]["label"] == 2]
        large_ids = np.load(source / "instances.npy", mmap_mode="r")
        first_overlap = np.any(initial[hand_ids], axis=0)
        rows = []
        for frame in frames:
            image = cv2.imread(str(source / "frames" / f"{frame:05d}.jpg"))
            raw = np.array(masks[frame])
            gated = raw.copy()
            workspace = np.zeros(image.shape[:2], bool)
            x0, y0, x1, y1 = config["workspace_xyxy"]
            workspace[y0:y1, x0:x1] = True
            gated &= workspace
            hand_union = np.any(raw[hand_ids], axis=0)
            stats = []
            for i, item in enumerate(detections):
                obj = item["object"]
                if obj["label"] == 2:
                    gated[i] &= object_hand_gate(item, allowed[frame], sides[frame] if sides is not None else None)
                else:
                    gated[i] &= ~arms[frame]
                    gated[i] &= appearance_gate(image, obj)
                    if obj.get("static_support"):
                        gated[i] &= initial[i]
                    if gated[i].sum() > 0.19 * gated[i].size:
                        gated[i] = False
                union = np.count_nonzero(raw[i] | initial[i])
                stats.append(
                    dict(
                        name=names[i],
                        initial_pixels=int(initial[i].sum()),
                        raw_pixels=int(raw[i].sum()),
                        gated_pixels=int(gated[i].sum()),
                        initial_hand_overlap_pixels=int(np.count_nonzero(initial[i] & first_overlap))
                        if i not in hand_ids
                        else None,
                        raw_hand_overlap_pixels=int(np.count_nonzero(raw[i] & hand_union))
                        if i not in hand_ids
                        else None,
                        raw_arm_core_pixels=int(np.count_nonzero(raw[i] & arms[frame])),
                        first_frame_iou=float(np.count_nonzero(raw[i] & initial[i]) / union)
                        if frame == 0 and union
                        else None,
                    )
                )
            large = np.stack([large_ids[frame] == i + 1 for i in range(len(names))])
            panels = [
                image,
                overlay(image, raw, detections, f"{frame}: Edge raw; exclusive colors"),
                overlay(image, gated, detections, f"{frame}: Edge + shared output gates"),
                overlay(image, large, detections, f"{frame}: previous SAM2-large final"),
            ]
            cv2.imwrite(
                str(out / f"comparison_{frame:04d}.jpg"), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            )
            # Isolate the object mask: shared coloring can otherwise hide overlap with the hand.
            tool = "micropipette" if task == "micro" else "dropper" if task == "dropper" else "bread_slice_2"
            idx = names.index(tool)
            isolated = [
                image,
                overlay(image, [raw[idx]], [detections[idx]], f"{frame}: {tool} raw ONLY"),
                overlay(image, [gated[idx]], [detections[idx]], f"{frame}: {tool} gated ONLY"),
            ]
            cv2.imwrite(str(out / f"tool_{frame:04d}.jpg"), np.hstack(isolated))
            rows.append(dict(frame=frame, objects=stats))
        initial_overlaps = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                intersection = int(np.count_nonzero(initial[i] & initial[j]))
                if intersection:
                    initial_overlaps.append(dict(objects=[names[i], names[j]], pixels=intersection))
        report[task] = dict(
            source=str(source), initialization=init.get("note"), initial_pairwise_overlap=initial_overlaps, frames=rows
        )
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2))
    print(ROOT, flush=True)


if __name__ == "__main__":
    main()
