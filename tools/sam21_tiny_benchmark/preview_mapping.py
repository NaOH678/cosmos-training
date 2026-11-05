"""Overlay each initial-mask channel on frame 0 with a name legend for eyeballing.

Usage: python preview_mapping.py <task>
Writes <task_dir>/mapping_preview_<task>.jpg and <task_dir>/mapping_check.json.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _task_defs import TASKS, task_names_colors


def main(task):
    task_dir = TASKS[task]
    names, colors = task_names_colors(task_dir)
    masks = np.load(task_dir / "initial_masks.npy")
    assert masks.shape[0] == len(names), (masks.shape, names)
    image = cv2.imread(str(task_dir / "frames" / "00000.jpg"))
    assert image is not None and masks.shape[1:] == image.shape[:2]

    viz = image.copy()
    for i, color in enumerate(colors):
        m = masks[i]
        viz[m] = (0.45 * viz[m].astype(np.float32) + 0.55 * np.array(color)).astype(np.uint8)
        # contour outline to make tiny objects visible
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, contours, -1, tuple(int(c) for c in color), 1)
    for row, (name, color) in enumerate(zip(names, colors)):
        y = 22 + 20 * row
        cv2.rectangle(viz, (6, y - 12), (22, y + 2), tuple(int(c) for c in color), -1)
        cv2.putText(viz, f"{row}: {name} area={int(masks[row].sum())}", (28, y), 0, 0.55, (255, 255, 255), 1)

    out = task_dir / f"mapping_preview_{task}.jpg"
    cv2.imwrite(str(out), viz, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (task_dir / f"mapping_check_{task}.json").write_text(
        json.dumps(dict(task=task, names=names, colors_bgr=[list(c) for c in colors]))
    )
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
