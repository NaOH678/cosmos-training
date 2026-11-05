"""Does the online geometric foreground reproduce what SAM2-masked training candidates look like?

Training candidates (``pf_out/9.24/sandwich/labeled``) are tracked points inside SAM2 masks: label 2 hand,
3 manipulated object, 4 table / static object.  Online (``cosmos_framework/inference/robot_policy/pointfk.py``)
there is no mask, only the depth cloud, so ``foreground_cloud`` + ``select_foreground_ids`` stand in for it.
This script applies that stand-in rule (same constants; arm removal omitted -- the labelled set holds no arm
points) to the *labelled* first-frame points of real training episodes and reports

  1. frame sanity: label-2 points vs. FK keypoints (checks the camera frame and the frame indexing),
  2. recall / purity of the geometric rule against the labels,
  3. what three recipes select from the labelled points: the sim recipe (global 1/2 + hand-local FPS),
     the geometric foreground with hand 0.40 / object 0.60, and their overlap,
  4. track validity of the selected points over the next 32 steps (15 Hz = every 2nd source frame),
     which is what decides how much supervision an anchor-first selection (option A) keeps.

CPU only; reads one frame row plus 32 future rows per window from the flat labelled arrays.

    python tools/validate_real_foreground_selection.py --episodes 12 --windows 8 --out <dir>/report.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
REPO = Path(__file__).resolve().parents[1]
LABELED = ROOT / "pf_out/9.24/sandwich/labeled"
FK_ROOT = Path("/data/shichaojian/raw_data/sandwich_fk21")

TABLE_Z, MIN_H, MAX_H = 0.58, 0.015, 0.45
X_RANGE, Y_RANGE = (0.30, 0.95), (-0.65, 0.65)
HAND_RADIUS, HAND_MIN_H, HAND_FRACTION, BUDGET, POOL = 0.05, 0.008, 0.40, 500, 8192
STEPS, SOURCE_STRIDE = 32, 2  # 15 Hz steps on 30 Hz source frames


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load_file("pf_sel", REPO / "cosmos_framework/data/pointflow_anchor_selection.py")
extrinsic = load_file("fk_ext", REPO / "cosmos_framework/data/fk_camera_extrinsic.py")
CAM_R, CAM_T = np.asarray(extrinsic.R, dtype=np.float64), np.asarray(extrinsic.t, dtype=np.float64)  # p_cam = R p_base + t
fps = selection.anchor_fps


def to_base(xyz):
    return (xyz - CAM_T) @ CAM_R


def pool(members, rng):
    return np.sort(rng.choice(members, POOL, replace=False)) if len(members) > POOL else members


def sim_recipe(xyz, fk, rng):
    """Right-hand version of hand_guided_fps over a random pool, like the online path."""
    ids = pool(np.arange(len(xyz)), rng)
    glob = BUDGET // 2
    chosen = fps(xyz[ids], glob).tolist()
    near = cKDTree(fk).query(xyz[ids])[0] <= HAND_RADIUS
    near[chosen] = False
    cand = np.flatnonzero(near)
    if len(cand):
        chosen.extend(cand[fps(xyz[ids][cand], BUDGET - glob)].tolist())
    chosen.extend(fps(xyz[ids], BUDGET - len(chosen), initial=chosen).tolist())
    return ids[np.asarray(chosen)]


def geometric(xyz, fk):
    base = to_base(xyz)
    height = base[:, 2] - TABLE_Z
    window = (base[:, 0] >= X_RANGE[0]) & (base[:, 0] <= X_RANGE[1]) & (base[:, 1] >= Y_RANGE[0]) & (base[:, 1] <= Y_RANGE[1])
    hand_d = cKDTree(fk).query(xyz)[0]
    is_hand = (hand_d <= HAND_RADIUS) & (height >= HAND_MIN_H)
    is_obj = window & (height >= MIN_H) & (height <= MAX_H)
    return is_hand | is_obj, is_hand, height, hand_d


def foreground_recipe(xyz, keep, is_hand, rng):
    groups = []
    for mask in (is_hand & keep, keep & ~is_hand):
        groups.append(pool(np.flatnonzero(mask), rng))
    hand_n = min(len(groups[0]), int(round(BUDGET * HAND_FRACTION)))
    obj_n = min(len(groups[1]), BUDGET - hand_n)
    hand_n = min(len(groups[0]), BUDGET - obj_n)
    out = [g[fps(xyz[g], n)] for g, n in zip(groups, (hand_n, obj_n)) if n > 0]
    return np.concatenate(out) if out else np.empty(0, np.int64)


def composition(labels, ids):
    return {int(k): round(float((labels[ids] == k).mean()), 4) for k in (2, 3, 4)} if len(ids) else {}


def chamfer(a, b):
    return float(cKDTree(b).query(a)[0].mean()) if len(a) and len(b) else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write a new file instead of overwriting")

    names = sorted(p.name for p in LABELED.iterdir() if p.name.startswith("episode_"))
    names = [names[i] for i in np.linspace(0, len(names) - 1, args.episodes).astype(int)]
    rng = np.random.default_rng(0)
    rows = []
    for name in names:
        d = LABELED / name
        labels = np.load(d / "region_labels.npy")
        position = np.load(d / "position.npy", mmap_mode="r")
        valid = np.load(d / "valid.npy", mmap_mode="r")
        with np.load(FK_ROOT / name / "annotations/wuji_fk21.npz", allow_pickle=True) as z:
            fk_base = np.asarray(z["positions"])[:, 1]  # right hand, Link_Base, [T,21,3]
        frames = position.shape[0]
        last = frames - 1 - STEPS * SOURCE_STRIDE
        for frame in np.linspace(0, last, args.windows).astype(int):
            xyz_all = np.asarray(position[frame], dtype=np.float64)
            ok = np.asarray(valid[frame]) & np.isfinite(xyz_all).all(1) & (xyz_all[:, 2] > 0)
            idx = np.flatnonzero(ok)
            xyz, lab = xyz_all[idx], labels[idx]
            fk = fk_base[frame] @ CAM_R.T + CAM_T
            keep, is_hand, height, hand_d = geometric(xyz, fk)

            sim = sim_recipe(xyz, fk, rng)
            geo = foreground_recipe(xyz, keep, is_hand, rng)

            future = np.stack([np.asarray(valid[frame + SOURCE_STRIDE * k])[idx] for k in range(1, STEPS + 1)])  # [32, N]

            def validity(ids):
                return {
                    "valid_at_step32": round(float(future[-1][ids].mean()), 4),
                    "valid_all_32": round(float(future[:, ids].all(0).mean()), 4),
                    "mean_valid_fraction": round(float(future[:, ids].mean()), 4),
                } if len(ids) else {}

            rows.append(dict(
                episode=name, frame=int(frame), candidates=int(len(idx)),
                label_counts={int(k): int((lab == k).sum()) for k in (2, 3, 4)},
                label2_within_5cm_of_fk=round(float((hand_d[lab == 2] <= 0.05).mean()), 4) if (lab == 2).any() else None,
                label2_median_dist_to_fk_m=round(float(np.median(hand_d[lab == 2])), 4) if (lab == 2).any() else None,
                pass_rate_by_label={int(k): round(float(keep[lab == k].mean()), 4) for k in (2, 3, 4) if (lab == k).any()},
                is_hand_vs_label2={"hand_pred_that_are_label2": round(float((lab[is_hand] == 2).mean()), 4) if is_hand.any() else None,
                                   "label2_recalled_as_hand": round(float(is_hand[lab == 2].mean()), 4) if (lab == 2).any() else None},
                label4_below_min_height=round(float((height[lab == 4] < MIN_H).mean()), 4) if (lab == 4).any() else None,
                kept_composition=composition(lab, np.flatnonzero(keep)),
                selected_sim=dict(n=int(len(sim)), composition=composition(lab, sim), **validity(sim)),
                selected_geo=dict(n=int(len(geo)), composition=composition(lab, geo), **validity(geo)),
                all_candidates_validity=validity(np.arange(len(idx))),
                chamfer_geo_to_sim_m=round(chamfer(xyz[geo], xyz[sim]), 4),
                chamfer_sim_to_geo_m=round(chamfer(xyz[sim], xyz[geo]), 4),
            ))
        print(name, "done", flush=True)

    def mean(path):
        values = []
        for r in rows:
            v = r
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and np.isfinite(v):
                values.append(v)
        return round(float(np.mean(values)), 4) if values else None

    summary = {
        "windows": len(rows),
        "label2_within_5cm_of_fk": mean(["label2_within_5cm_of_fk"]),
        "label2_median_dist_to_fk_m": mean(["label2_median_dist_to_fk_m"]),
        "pass_rate_label2": round(float(np.nanmean([r["pass_rate_by_label"].get(2, np.nan) for r in rows])), 4),
        "pass_rate_label3": round(float(np.nanmean([r["pass_rate_by_label"].get(3, np.nan) for r in rows])), 4),
        "pass_rate_label4": round(float(np.nanmean([r["pass_rate_by_label"].get(4, np.nan) for r in rows])), 4),
        "label4_below_min_height": mean(["label4_below_min_height"]),
        "hand_pred_are_label2": mean(["is_hand_vs_label2", "hand_pred_that_are_label2"]),
        "label2_recalled_as_hand": mean(["is_hand_vs_label2", "label2_recalled_as_hand"]),
        "sim_composition": {k: round(float(np.mean([r["selected_sim"]["composition"].get(k, 0) for r in rows])), 4) for k in (2, 3, 4)},
        "geo_composition": {k: round(float(np.mean([r["selected_geo"]["composition"].get(k, 0) for r in rows])), 4) for k in (2, 3, 4)},
        "validity_all_candidates": {k: mean(["all_candidates_validity", k]) for k in ("valid_at_step32", "valid_all_32", "mean_valid_fraction")},
        "validity_sim": {k: mean(["selected_sim", k]) for k in ("valid_at_step32", "valid_all_32", "mean_valid_fraction")},
        "validity_geo": {k: mean(["selected_geo", k]) for k in ("valid_at_step32", "valid_all_32", "mean_valid_fraction")},
        "chamfer_geo_to_sim_m": mean(["chamfer_geo_to_sim_m"]),
        "chamfer_sim_to_geo_m": mean(["chamfer_sim_to_geo_m"]),
    }
    args.out.write_text(json.dumps({"summary": summary, "windows": rows}, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
