"""First-frame-only prompt probe for bread and its rack; all boxes are detected."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_sandwich_dynamic import DynamicSegmenter
from segment_sandwich_video import CHECKPOINTS

QUERIES = [
    "brown toast slice.",
    "upright slice of bread.",
    "partially occluded toast slice.",
    "slice of bread.",
    "bread.",
    "toast.",
    "brown bread.",
    "two slices of bread.",
    "toast rack with bread.",
    "bread rack.",
    "bread holder.",
    "metal toast rack.",
    "a rack holding two upright brown bread slices.",
]
SEARCH = (470, 275, 565, 395)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--search", nargs=4, type=int, default=SEARCH)
    p.add_argument("--queries", nargs="+", default=QUERIES)
    p.add_argument(
        "--frame", type=Path, default=Path("pointflow_outputs/sandwich_firstframe_track_20261030/frames/00000.jpg")
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(
        box_threshold=0.12,
        text_threshold=0.12,
        nms_iou=0.7,
        objects=[dict(id=i + 100, name=q, label=4, texts=[q], max_instances=8) for i, q in enumerate(args.queries)],
    )
    model = DynamicSegmenter(
        config, str(CHECKPOINTS / "grounding-dino-base"), str(CHECKPOINTS / "sam2.1_hiera_large.pt")
    )
    image = cv2.cvtColor(cv2.imread(str(args.frame)), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = args.search
    with torch.inference_mode():
        items, missing = model.detect(image[y0:y1, x0:x1])
    model.sam.set_image(image)
    masks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in items:
            box = np.asarray(item["box"]) + [x0, y0, x0, y0]
            choices, scores, _ = model.sam.predict(box=box, multimask_output=True)
            item["box"] = box.tolist()
            item["sam_score"] = float(np.max(scores))
            masks.append(choices[int(np.argmax(scores))].astype(bool))
    np.save(args.out / "candidate_masks.npy", np.asarray(masks))
    (args.out / "candidates.json").write_text(
        json.dumps(dict(frame=0, search_roi=args.search, detections=items, missing=missing), indent=2)
    )
    for i, item in enumerate(items):
        preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        m = masks[i]
        preview[m] = (preview[m] * 0.4 + np.array([50, 160, 250]) * 0.6).astype(np.uint8)
        x, y, a, b = map(int, item["box"])
        cv2.rectangle(preview, (x, y), (a, b), (0, 255, 0), 1)
        cv2.putText(preview, f"{i}: {item['query']} {item['score']:.3f}", (5, 22), 0, 0.48, (0, 0, 255), 1)
        cv2.imwrite(str(args.out / f"candidate_{i:03d}.jpg"), preview)
        print(i, item["query"], item["score"], item["box"], "area", int(m.sum()), flush=True)


if __name__ == "__main__":
    main()
