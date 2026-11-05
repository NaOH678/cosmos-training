"""Read-only matched-ID DA3/D435/FoundationStereo stability and display audit."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "pointflow_outputs"
OUT = BASE / "scissors_three_depth_stability_20261105"
PATHS = dict(
    DA3=BASE / "cotracker_real_v1_offline_da3_20261104/scissors",
    D435=BASE / "cotracker_real_v3_d435_20261105/scissors",
    FoundationStereo=BASE / "scissors_foundationstereo_v3_parity_20261105",
)
V0 = BASE / "cotracker_real_20261104/scissors"


def describe(values):
    a = np.asarray(values, dtype=float)
    return (
        dict(
            n=len(a),
            p50_mm=float(np.median(a) * 1000),
            p95_mm=float(np.percentile(a, 95) * 1000),
            gt20mm_fraction=float((a > 0.02).mean()),
        )
        if len(a)
        else dict(n=0)
    )


def main():
    OUT.mkdir(exist_ok=True)
    updates = {name: json.loads((p / "per_frame.json").read_text()) for name, p in PATHS.items() if name != "DA3"}
    frames = sorted(
        set(r["frame"] for r in updates["D435"] if r["sensor_update"])
        & set(r["frame"] for r in updates["FoundationStereo"] if r["sensor_update"])
    )
    maps = np.load(V0 / "instances.npy", mmap_mode="r")
    names = {x["id"]: x["name"] for x in json.loads((V0 / "task_config.json").read_text())["objects"]}
    values = {}
    previous = None
    previous_frame = None
    frame_rows = []
    for frame in frames:
        data = {}
        for name, path in PATHS.items():
            with np.load(path / "frames" / f"{frame:06d}.npz") as f:
                data[name] = {k: f[k] for k in ["ids", "uv", "xyz", "valid", "instances"]}
        a = data["DA3"]
        for d in data.values():
            np.testing.assert_array_equal(a["ids"], d["ids"])
            np.testing.assert_array_equal(a["uv"], d["uv"])
        imap = cv2.resize(maps[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        q = np.rint(a["uv"]).astype(int)
        q[:, 0] = q[:, 0].clip(0, 639)
        q[:, 1] = q[:, 1].clip(0, 447)
        distance = np.zeros(len(q), np.float32)
        for inst in np.unique(a["instances"]):
            image = cv2.distanceTransform((imap == inst).astype(np.uint8), cv2.DIST_L2, 5)
            m = a["instances"] == inst
            distance[m] = image[q[m, 1], q[m, 0]]
        for d in data.values():
            d["core"] = distance >= 3
        if previous is not None and frame - previous_frame <= 4:
            _, ii, jj = np.intersect1d(previous["DA3"]["ids"], a["ids"], return_indices=True)
            common = np.ones(len(ii), bool)
            for name in PATHS:
                common &= previous[name]["valid"][ii] & data[name]["valid"][jj]
            common &= previous["DA3"]["instances"][ii] == a["instances"][jj]
            for inst in np.unique(a["instances"][jj[common]]):
                for region in ["all", "core", "boundary"]:
                    use = common & (a["instances"][jj] == inst)
                    if region == "core":
                        use &= a["core"][jj] & previous["DA3"]["core"][ii]
                    if region == "boundary":
                        use &= ~a["core"][jj] | ~previous["DA3"]["core"][ii]
                    ia, ib = ii[use], jj[use]
                    if len(ia) < 8:
                        continue
                    for name in PATHS:
                        now = data[name]["xyz"][ib]
                        before = previous[name]["xyz"][ia]
                        delta = now[:, 2] - before[:, 2]
                        shared = float(np.median(delta))
                        v = values.setdefault(
                            (int(inst), region, name),
                            dict(step=[], residual=[], object_shift=[], relative_residual=[], pair_distance=[]),
                        )
                        v["step"].extend(abs(delta).tolist())
                        v["residual"].extend(abs(delta - shared).tolist())
                        v["object_shift"].append(abs(shared))
                        v["relative_residual"].extend(
                            (abs(delta - shared) / np.maximum(0.1, np.median(before[:, 2]))).tolist()
                        )
                        # Pair distances are invariant to shared rigid translation/rotation.
                        rng = np.random.default_rng(frame * 17 + int(inst))
                        n = min(128, len(ia) * 2)
                        i = rng.integers(0, len(ia), n)
                        j = rng.integers(0, len(ia), n)
                        different = i != j
                        i, j = i[different], j[different]
                        v["pair_distance"].extend(
                            abs(
                                np.linalg.norm(now[i] - now[j], axis=1) - np.linalg.norm(before[i] - before[j], axis=1)
                            ).tolist()
                        )
        frame_rows.append(dict(frame=frame, counts={name: int(d["valid"].sum()) for name, d in data.items()}))
        previous, previous_frame = data, frame
    result = {}
    for (inst, region, name), v in values.items():
        result.setdefault(names[inst], {}).setdefault(region, {})[name] = {
            k: describe(x) for k, x in v.items() if k != "relative_residual"
        } | dict(relative_residual_p50_per_mille=float(np.median(v["relative_residual"]) * 1000))
    mode_stats = {}
    for name, rows in updates.items():
        mode_stats[name] = {}
        for inst in [3, 5, 6, 7]:
            states = [r.get("surface_modes", {}).get(str(inst), {}) for r in rows if r["sensor_update"]]
            mode_stats[name][names[inst]] = dict(
                modes=dict(Counter(s.get("mode", "missing") for s in states)),
                uv_median_px=float(np.median([s["uv_median_px"] for s in states if "uv_median_px" in s])),
            )
    report = dict(
        scope="Matched IDs valid in all three at both actual-depth-update times; <=4 RGB frames apart; observed movement not absolute depth error. Pair distance still changes for articulated scissors/hands.",
        paired_depth_update_frames=len(frames),
        instances=result,
        static_modes=mode_stats,
    )
    (OUT / "statistics.json").write_text(json.dumps(report, indent=2))
    (OUT / "frame_counts.json").write_text(json.dumps(frame_rows, indent=2))
    for obj in ["cup", "wooden_block_left", "wooden_block_right", "scissors"]:
        print(
            obj,
            {
                name: {
                    k: round(result[obj]["all"][name][k]["p50_mm"], 3) for k in ["step", "residual", "pair_distance"]
                }
                for name in PATHS
            },
            flush=True,
        )
    print("paired depth updates", len(frames), flush=True)


if __name__ == "__main__":
    main()
