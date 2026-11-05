"""RGB/mask/URDF overlays for every severe D435 frame, without fitting geometry."""

import json
from pathlib import Path

import cv2
import lmdb
import numpy as np
from render_d435_point_accuracy import TASKS, Surface, measured_cloud, project, visible_surface
from scipy.spatial import cKDTree

root = Path("pointflow_outputs/depth_source_comparison_20261103")
task = "sandwich"
out = root / task / "failure_audit"
out.mkdir(exist_ok=True)
rows = json.loads((root / task / "per_frame.json").read_text())
bad = [r for r in rows if r["A"]["all_hand"]["p90_mm"] > 30]
ep, mask = TASKS[task]
episode = Path("/data/shichaojian/raw_data") / ep
mask = Path(mask)
info = json.loads((mask / "initialization.json").read_text())
config = json.loads((mask / "task_config.json").read_text()) if (mask / "task_config.json").exists() else info["config"]
streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"][
    "streams"
]
intr = streams["color"]["intrinsics"]
reference = Surface(episode, config)
labels = np.load(mask / "labels.npy", mmap_mode="r")
instances = np.load(mask / "instances.npy", mmap_mode="r")
env = lmdb.open("/tmp/d435_point_accuracy_20261102/sandwich", readonly=True, lock=False)
cap = cv2.VideoCapture(str(episode / "videos/head.mp4"))
tiles = []
audit = []
for row in bad:
    f = row["frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, image = cap.read()
    assert ok
    with env.begin() as txn:
        blob = txn.get(f"depth/head/{f:06d}".encode())
    depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
    p, c, cls, ids, raw, kept, uv = measured_cloud(depth, streams, instances[f], labels[f], image, "cpu")
    surface = reference.at(f)
    selected = np.flatnonzero(kept & (cls == 2))
    distances = cKDTree(surface).query(p[selected])[0]
    far = selected[distances > 0.03]
    hand = (labels[f] == 2).astype(np.uint8)
    interior = cv2.erode(hand, np.ones((7, 7), np.uint8)) > 0
    xy = uv[far]
    fraction = float(interior[xy[:, 1], xy[:, 0]].mean()) if len(xy) else 0
    left = image.copy()
    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, (0, 150, 255), 1)
    su, valid = project(visible_surface(surface, intr), intr)
    left[su[valid, 1], su[valid, 0]] = (255, 220, 20)
    right = image.copy()
    right[xy[:, 1], xy[:, 0]] = (40, 40, 255)
    tile = np.hstack((cv2.resize(left, (240, 180)), cv2.resize(right, (240, 180))))
    canvas = np.full((220, 480, 3), 245, np.uint8)
    canvas[40:] = tile
    cv2.putText(
        canvas,
        f"f{f} P90={row['A']['all_hand']['p90_mm']:.0f}mm far-core={fraction:.0%}",
        (5, 17),
        0,
        0.5,
        (0, 0, 0),
        1,
    )
    cv2.putText(canvas, "L: mask/orange URDF/cyan | R: >30mm/red", (5, 34), 0, 0.42, (0, 0, 0), 1)
    tiles.append(canvas)
    cv2.imwrite(str(out / f"{f:04d}.jpg"), canvas)
    audit.append(
        dict(
            frame=f,
            p90_mm=row["A"]["all_hand"]["p90_mm"],
            far_points=len(far),
            far_inside_eroded_mask_fraction=fraction,
        )
    )
for page in range((len(tiles) + 11) // 12):
    sheet = np.full((220 * 3, 480 * 4, 3), 245, np.uint8)
    for i, tile in enumerate(tiles[page * 12 : (page + 1) * 12]):
        y, x = divmod(i, 4)
        sheet[y * 220 : (y + 1) * 220, x * 480 : (x + 1) * 480] = tile
    cv2.imwrite(str(out / f"contact_sheet_{page + 1}.jpg"), sheet)
(out / "audit.json").write_text(json.dumps(audit, indent=2))
cap.release()
env.close()
print("Audited", len(bad), "frames", flush=True)
