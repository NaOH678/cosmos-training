"""Build two first-frame-only prompt variants from automatic DINO/SAM2 proposals.

A rack-local search ROI and a brown-appearance gate are task configuration, not
manual object masks. The partially visible second slice may have low confidence.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from segment_sandwich_video import COLORS


def build(base, probe, root):
    original = json.loads((base / "initialization.json").read_text())
    original_masks = np.load(base / "initial_masks.npy")
    proposals = json.loads((probe / "candidates.json").read_text())
    candidate_masks = np.load(probe / "candidate_masks.npy")
    image = cv2.imread(str(base / "frames/00000.jpg"))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brown = (hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 25)
    hand = original_masks[
        next(i for i, item in enumerate(original["detections"]) if item["object"]["name"] == "robot_hand")
    ]
    arms = np.load(base / "arm_core.npy", mmap_mode="r")[0]
    # Use the actual SAM hand mask here. A dilated hand search gate would erase
    # the nearby, genuinely visible back slice before it could be initialized.
    visible = ~hand & ~arms
    selected = []
    audit = []
    for i, item in sorted(enumerate(proposals["detections"]), key=lambda pair: -pair[1]["score"]):
        if "rack" in item["query"] or "holder" in item["query"]:
            continue
        mask = candidate_masks[i] & visible
        fraction = float((mask & brown).sum() / max(mask.sum(), 1))
        overlap = max([float((mask & m).sum() / max(min(mask.sum(), m.sum()), 1)) for _, _, m in selected], default=0.0)
        accepted = bool(mask.sum() >= 80 and fraction >= 0.70 and overlap < 0.5)
        audit.append(
            dict(
                candidate_index=i,
                score=item["score"],
                area=int(mask.sum()),
                brown_fraction=fraction,
                overlap=overlap,
                accepted=accepted,
            )
        )
        if accepted:
            selected.append((i, item, mask))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError(f"Only {len(selected)} distinct visible bread proposals passed; no fabricated second slice")
    rack_candidates = [
        (i, item, candidate_masks[i] & visible)
        for i, item in enumerate(proposals["detections"])
        if "rack" in item["query"] or "holder" in item["query"]
    ]
    rack_candidates = [entry for entry in rack_candidates if entry[2].sum() >= 100 and np.mean(brown[entry[2]]) < 0.65]
    rack_index, rack_item, rack_mask = max(rack_candidates, key=lambda entry: entry[1]["score"])
    outputs = []
    for variant in ["group", "separate"]:
        out = root / variant
        out.mkdir(parents=True, exist_ok=False)
        for name in ["frames", "video.json", "hand_allowed.npy", "arm_core.npy", "geometry_timing.json"]:
            (out / name).symlink_to((base / name).resolve(), target_is_directory=name == "frames")
        info = copy.deepcopy(original)
        masks = [m.copy() for m in original_masks]
        items = info["detections"]

        def append(name, mask, source):
            idx = len(items) + 1
            ys, xs = np.where(mask)
            item = dict(
                instance_id=idx,
                object=dict(id=100 + idx, name=name, label=4),
                box=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                score=min(x["score"] for x in source),
                query=" | ".join(x["query"] for x in source),
                source_frame=0,
                automatic_prompt=True,
                sources=source,
            )
            items.append(item)
            masks.append(mask)

        if variant == "group":
            append(
                "bread_rack_group",
                rack_mask | selected[0][2] | selected[1][2],
                [rack_item, selected[0][1], selected[1][1]],
            )
        else:
            for number, (_, item, mask) in enumerate(selected, 1):
                append(f"bread_slice_{number}", mask, [item])
        info["config"].update(
            bread_variant=variant,
            bread_search_roi=proposals["search_roi"],
            bread_brown_gate=dict(hue_max=35, saturation_min=60, value_min=25, min_fraction=0.70, min_pixels=80),
            bread_max_overlap=0.5,
            bread_query_source=str(probe.resolve()),
        )
        info["config"]["objects"] = [x for x in info["config"]["objects"] if x["name"] != "bread"]
        info["config"]["objects"] += [x["object"] for x in items if x["object"]["name"].startswith("bread_")]
        info.update(
            missing=[],
            bread_candidates=selected and [entry[0] for entry in selected],
            rack_candidate=rack_index,
            bread_selection_audit=audit,
            dino_frames=[0],
            baseline_initialization_dino_ms=original["dino_ms"],
            dino_ms=None,
            initialization_timing_note="Combined baseline and bread prompt initialization was not benchmarked.",
            source_base=str(base.resolve()),
            note="Only visible first-frame surfaces initialized. Group has one ID; separate has two IDs. No later DINO.",
        )
        np.save(out / "initial_masks.npy", np.asarray(masks))
        (out / "initialization.json").write_text(json.dumps(info, indent=2))
        ids = np.zeros(image.shape[:2], np.uint8)
        for i in sorted(
            range(len(items)), key=lambda i: (items[i]["object"]["name"] == "robot_hand", -int(masks[i].sum()))
        ):
            ids[masks[i]] = i + 1
        preview = image.copy()
        for i, item in enumerate(items):
            color = np.array(COLORS[item["object"]["name"]])
            m = ids == i + 1
            preview[m] = (preview[m] * 0.45 + color * 0.55).astype(np.uint8)
        cv2.imwrite(str(out / "initialization_preview.jpg"), preview)
        outputs.append(str(out))
    print(
        json.dumps(
            dict(outputs=outputs, bread_candidates=[entry[0] for entry in selected], rack_candidate=rack_index),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("pointflow_outputs/sandwich_firstframe_track_20261030"))
    p.add_argument("--probe", type=Path, default=Path("pointflow_outputs/sandwich_bread_prompt_probe_20261030"))
    p.add_argument("--out", type=Path, default=Path("pointflow_outputs/sandwich_bread_variants_20261030"))
    a = p.parse_args()
    build(a.base, a.probe, a.out)


if __name__ == "__main__":
    main()
