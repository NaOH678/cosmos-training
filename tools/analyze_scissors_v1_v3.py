"""Read-only V1/V3 stability, coverage and thin/occlusion disagreement diagnostics."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "pointflow_outputs"
V0 = BASE / "cotracker_real_20261104/scissors"
V1 = BASE / "cotracker_real_v1_offline_da3_20261104/scissors"
V3 = BASE / "cotracker_real_v3_d435_20261105/scissors"
OUT = BASE / "scissors_v1_v3_diagnosis_20261105"


def describe(x):
    x = np.asarray(x, dtype=float)
    return (
        dict(
            n=len(x),
            median=float(np.median(x)),
            p90=float(np.percentile(x, 90)),
            p95=float(np.percentile(x, 95)),
            p99=float(np.percentile(x, 99)),
            fraction_gt20mm=float(np.mean(x > 0.02)),
            fraction_gt50mm=float(np.mean(x > 0.05)),
        )
        if len(x)
        else dict(n=0)
    )


def regions(data, canvas, semantics):
    q = np.rint(data["uv"]).astype(int)
    q[:, 0] = q[:, 0].clip(0, 639)
    q[:, 1] = q[:, 1].clip(0, 447)
    d = cv2.distanceTransform((canvas == 3).astype(np.uint8), cv2.DIST_L2, 5)
    h = cv2.distanceTransform((semantics != 2).astype(np.uint8), cv2.DIST_L2, 5)
    scissors = canvas == 3
    yy, xx = np.nonzero(scissors)
    long = np.full(len(q), np.nan)
    if len(xx) > 10:
        pixels = np.column_stack((xx, yy))
        center = pixels.mean(0)
        _, _, v = np.linalg.svd(pixels - center, full_matrices=False)
        projection = (pixels - center) @ v[0]
        low, high = np.percentile(projection, [1, 99])
        long = ((data["uv"] - center) @ v[0] - low) / max(high - low, 1)
    obj = data["labels"] == 3
    s = data["instances"] == 3
    masks = dict(
        all_objects=obj,
        hands=data["labels"] == 2,
        object_near_hand=obj & (h[q[:, 1], q[:, 0]] <= 6),
        object_far_hand=obj & (h[q[:, 1], q[:, 0]] >= 10),
        scissors=s,
        scissors_thin=s & (d[q[:, 1], q[:, 0]] <= 2),
        scissors_endcaps=s & ((long < 0.15) | (long > 0.85)),
        scissors_middle=s & (long > 0.3) & (long < 0.7),
        cup=data["instances"] == 5,
        right_block=data["instances"] == 7,
    )
    return masks, long


def main():
    OUT.mkdir(exist_ok=True)
    instances = np.load(V0 / "instances.npy", mmap_mode="r")
    labels = np.load(V0 / "labels.npy", mmap_mode="r")
    acc = {}
    coverage = {}
    previous = None
    frames = []
    uvchanges = []
    rows = json.loads((V3 / "per_frame.json").read_text())
    for frame in range(987):
        with np.load(V1 / "frames" / f"{frame:06d}.npz") as n:
            a = {k: n[k] for k in ["ids", "uv", "instances", "labels", "xyz", "valid"]}
        with np.load(V3 / "frames" / f"{frame:06d}.npz") as n:
            b = {k: n[k] for k in ["ids", "uv", "instances", "labels", "xyz", "valid", "sensor_update", "provenance"]}
        assert np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["uv"], b["uv"])
        if not b["sensor_update"]:
            continue
        canvas = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        sem = cv2.resize(labels[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        groups, long = regions(a, canvas, sem)
        common = a["valid"] & b["valid"]
        delta = a["xyz"][:, 2] - b["xyz"][:, 2]
        for name, g in groups.items():
            c = coverage.setdefault(name, dict(candidates=0, v1_valid=0, v3_valid=0, common=0))
            c["candidates"] += int(g.sum())
            c["v1_valid"] += int((g & a["valid"]).sum())
            c["v3_valid"] += int((g & b["valid"]).sum())
            c["common"] += int((g & common).sum())
            acc.setdefault(("disagreement", name), []).append(abs(delta[g & common]))
        core = groups["scissors_middle"] & common
        if core.sum() >= 5:
            offset = float(np.median(delta[core]))
            end = groups["scissors_endcaps"] & common
            thin = groups["scissors_thin"] & common
            for name, g in [("scissors_endcaps", end), ("scissors_thin", thin)]:
                acc.setdefault(("local_disagreement_after_middle_offset", name), []).append(abs(delta[g] - offset))
            frames.append(
                dict(
                    frame=frame,
                    scissors_candidates=int(groups["scissors"].sum()),
                    scissors_v1_valid=int((groups["scissors"] & a["valid"]).sum()),
                    scissors_v3_valid=int((groups["scissors"] & b["valid"]).sum()),
                    middle_da3_minus_d435_mm=offset * 1000,
                    end_pairs=int(end.sum()),
                    end_local_abs_median_mm=float(np.median(abs(delta[end] - offset)) * 1000) if end.sum() else None,
                    thin_pairs=int(thin.sum()),
                )
            )
        if previous:
            olda, oldb = previous
            _, ii, jj = np.intersect1d(olda["ids"], a["ids"], return_indices=True)
            both = (
                olda["valid"][ii]
                & oldb["valid"][ii]
                & a["valid"][jj]
                & b["valid"][jj]
                & (olda["instances"][ii] == a["instances"][jj])
            )
            stationary = np.linalg.norm(a["uv"][jj] - olda["uv"][ii], axis=1) < 0.5
            uvchanges.append(
                np.linalg.norm(
                    a["uv"][jj[both & (a["instances"][jj] == 5)]] - olda["uv"][ii[both & (a["instances"][jj] == 5)]],
                    axis=1,
                )
            )
            for name, g in groups.items():
                use = both & g[jj] & stationary
                for variant, old, new in [("v1", olda, a), ("v3", oldb, b)]:
                    change = abs(new["xyz"][jj[use], 2] - old["xyz"][ii[use], 2])
                    acc.setdefault((variant + "_sameid_abs_dz", name), []).append(change)
        previous = (a, b)
        if frame % 200 == 0:
            print("ANALYZE", frame, flush=True)
    result = dict(
        frames=987,
        actual_depth_frames=sum(r["sensor_update"] for r in rows),
        units="meters for depth statistics; UV pixels separately",
        cautions=[
            "V1 uses future128-frame context; V3 is causal. Their stability is not a fair online-method comparison.",
            "No object ground truth; D435/DA3 disagreement is not an accuracy error.",
            "Endcaps are PCA endpoint regions of SAM mask, not manually labeled true blade tips. Thin region is mask interior distance<=2 UV pixels.",
            "Temporal tail metrics pool individual same-ID points, common valid in both variants, same instance and UV change<0.5px; large changes may include real rotation.",
        ],
        coverage=coverage,
        statistics=[
            dict(metric=metric, region=name, **describe(np.concatenate(chunks) if chunks else []))
            for (metric, name), chunks in acc.items()
        ],
        cup_uv_displacement_pixels={
            k: v for k, v in describe(np.concatenate(uvchanges)).items() if not k.startswith("fraction_gt")
        },
        static_fraction={
            str(i): sum(x["surface_modes"].get(str(i), {}).get("mode") == "static" for x in rows) / len(rows)
            for i in range(3, 8)
        },
    )
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "per_frame_scissors.json").write_text(json.dumps(frames, indent=2))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
