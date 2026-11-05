"""Reapply the final causal validity policy to cached geometry without GPU inference."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_cotracker_outputs import _core, save

out = Path(sys.argv[1])
dest = out / "pointflow"
rows = json.loads((dest / "per_frame.json").read_text())
T = len(rows)
filter = _core.CausalGeometryFilter()
offsets = np.load(dest / "frame_offsets.npy")
valid = np.lib.format.open_memmap(dest / "obs_valid.npy", mode="r+")
for frame, row in enumerate(rows):
    p = dest / "frames" / f"{frame:06d}.npz"
    with np.load(p) as n:
        data = {k: n[k] for k in n.files}
    tic = time.perf_counter()
    base = _core.depth_sample_valid(data["uv"], data["raw_xyz"][:, 2], data["depth_conf"], data["instances"])
    keep, stats = filter.apply(data["ids"], data["xyz"], base, data["instances"], row["time"])
    row.update(stats, valid_points=int(keep.sum()), filter_ms=(time.perf_counter() - tic) * 1000)
    row.pop("group_rejected", None)
    data["valid"] = keep
    np.savez_compressed(p, **data)
    valid[offsets[frame] : offsets[frame + 1]] = keep
valid.flush()
save(dest / "per_frame.json", rows)
info = json.loads((out / "video.json").read_text())
timestamps = np.array([row["time"] for row in rows])
np.save(dest / "sensor_timestamps_sec.npy", timestamps)
np.save(dest / "timestamps_sec.npy", np.arange(T) / info["fps"])
report = json.loads((dest / "report.json").read_text())
report["filter_revision"] = _core.FILTER_REVISION
report["filter"] = (
    "Per-instance 16NN robust outliers; smaller disconnected 18mm-radius components more than60mm from the main component and >60mm median-Z separation rejected; per-track depth-change residual vs same-instance median; per-ID measured 3D step > max(80mm,1.5m/s*dt) rejected; no future fill or temporal Gaussian"
)
report["depth"] = (
    "Track4World AnyView weights + original DA3 Metric, current-frame nested alignment, calibrated focal; confidence>2, same-instance pixel depth discontinuities"
)
report["timeline"] = (
    "timestamps_sec follows source video FPS, as legacy exports; sensor_timestamps_sec keeps actual camera timestamps for D435 gating"
)
save(dest / "report.json", report)
summary = json.loads((dest / "COMPLETE_STAGE.json").read_text())
summary["valid_observations"] = sum(r["valid_points"] for r in rows)
summary["timing"]["filter_ms"] = {
    "p50_ms": float(np.median([r["filter_ms"] for r in rows[5:]])),
    "p95_ms": float(np.percentile([r["filter_ms"] for r in rows[5:]], 95)),
}
save(dest / "COMPLETE_STAGE.json", summary)
print("REFILTERED", out, T, summary["valid_observations"])
