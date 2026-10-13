"""Fit training-only action q01/q99 statistics for real single-right-hand caches.

Writes statistics, not a second normalized dataset. Frame weighting matches the
simulation recipe: each valid source action frame once, without window repeats.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import SingleRightHandRawDataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return {
        "count": len(values),
        **{
            name: fn(values, axis=0).tolist()
            for name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]
        },
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute(args):
    root = Path(args.cache_root)
    common = dict(
        root=args.raw_root,
        cache_root=str(root),
        episode_allowlist=args.episode_allowlist,
        split_seed=args.split_seed,
        split_val_ratio=args.split_val_ratio,
    )
    train = SingleRightHandRawDataset(split="train", **common)
    val = SingleRightHandRawDataset(split="val", **common)
    if train.arm_action_space != "joint" and not args.preserve_quaternion:
        raise ValueError("EEF cache requires explicit --preserve-quaternion; do not scale rotations silently")
    actions, states, fingerprints, counts = [], [], {}, {}
    for ep in train._episodes:
        path = root / "episodes" / f"{ep.name}.npz"
        with np.load(path, allow_pickle=False) as data:
            a, s = data["action"], data["state"]
            valid = data["action_valid"].astype(bool) if "action_valid" in data else np.ones(len(a), bool)
            if a.shape != (ep.num_frames, 27) or s.shape != a.shape or valid.shape != (len(a),):
                raise ValueError(f"Invalid shapes: {ep.name}")
            if not np.isfinite(a[valid]).all() or not np.isfinite(s[valid]).all():
                raise ValueError(f"Nonfinite valid frame: {ep.name}")
            actions.append(a[valid])
            states.append(s[valid])
            counts[ep.name] = int(valid.sum())
        fingerprints[ep.name] = digest(path)
    a, s = (np.concatenate(x).astype(np.float64) for x in (actions, states))
    astats, sstats = summarize(a), summarize(s)
    low, high = (np.asarray(astats[k], np.float32) for k in ("q01", "q99"))
    offset, scale = (low + high) / 2, np.maximum((high - low) / 2, np.float32(0.05))
    preserved = []
    if train.arm_action_space == "eef":
        preserved = list(range(3, 7))
        offset[3:7] = 0
        scale[3:7] = 1
    output = {
        "schema": "singlerighthand_action_quantile_v1",
        "method": "action_quantile_scale_floor",
        "arm_action_space": train.arm_action_space,
        "action_names": train.action_names,
        "units": ["radian"] * 27
        if train.arm_action_space == "joint"
        else ["meter"] * 3 + ["unitless_quaternion"] * 4 + ["radian"] * 20,
        "preserved_dimensions": preserved,
        "fit_split": "train",
        "split_seed": args.split_seed,
        "split_val_ratio": args.split_val_ratio,
        "frame_weighting": "all_action_valid_frames_once",
        "validity_note": "Uses action_valid when present; otherwise all finite cached rows. No extra tail trimming.",
        "state_transform": "same_as_action",
        "min_scale": 0.05,
        "forward_clamp": None,
        "offset": offset.tolist(),
        "scale": scale.tolist(),
        "action": astats,
        "state": sstats,
        "cache_root": str(root),
        "manifest_sha256": digest(root / "manifest.json"),
        "episode_allowlist": args.episode_allowlist,
        "train_episodes": [ep.name for ep in train._episodes],
        "val_episodes": [ep.name for ep in val._episodes],
        "episode_sha256": fingerprints,
        "valid_frame_counts": counts,
        "train_windows": len(train),
        "val_windows": len(val),
        "normalized_abs_max": {
            "action": float(np.abs((a - offset) / scale).max()),
            "state": float(np.abs((s - offset) / scale).max()),
        },
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "path": str(dest),
                "train_episodes": len(train._episodes),
                "val_episodes": len(val._episodes),
                "frames": len(a),
                "normalized_abs_max": output["normalized_abs_max"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--episode-allowlist")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-val-ratio", type=float, required=True)
    p.add_argument("--preserve-quaternion", action="store_true")
    p.add_argument("--output", required=True)
    compute(p.parse_args())
