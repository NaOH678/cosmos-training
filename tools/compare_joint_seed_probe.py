"""Compare joint samples from JOINT_SAMPLE_SEED_OFFSETS probe runs (docs/pointflow_fk_action_coupling_20261012.md §3).

Each run is an OUTPUT_ROOT of an eval-only launch; cases are matched by name in
pointflow_eval/step_<N>/<case>_joint/joint_prediction.npz.  Action row 0 is the clean
state condition and must agree across runs; rows 1..32 are the generated actions.
"""

import argparse
import statistics
from pathlib import Path

import numpy as np


def cases(run, step):
    roots = sorted(Path(run).glob(f"cosmos3_action/action_sft/*/pointflow_eval/step_{step:07d}"))
    if len(roots) != 1:
        raise FileNotFoundError(f"{run}: expected one pointflow_eval/step_{step:07d}, found {len(roots)}")
    return {path.parent.name.removesuffix("_joint"): path for path in roots[0].glob("*_joint/joint_prediction.npz")}


def rms(value):
    return float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="run without offsets")
    parser.add_argument("--geom", type=Path, required=True, help="run with pointflow/fk offsets")
    parser.add_argument("--all", type=Path, required=True, help="run with every modality offset")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    runs = {name: cases(getattr(args, name), args.step) for name in ("base", "geom", "all")}
    shared = sorted(set.intersection(*(set(found) for found in runs.values())))
    if not shared:
        raise ValueError("No case is present in all three runs")
    columns = ["action_geom", "action_all", "ratio", "fk_geom", "fk_all", "pointflow_geom", "pointflow_all", "vision_geom"]
    print("case " + " ".join(columns))
    val_rows = []
    for case in shared:
        base, geom, full = (np.load(runs[name][case]) for name in ("base", "geom", "all"))
        if not (np.array_equal(base["action"][0], geom["action"][0]) and np.array_equal(base["action"][0], full["action"][0])):
            raise ValueError(f"{case}: state row differs between runs")
        diff = lambda key, other, rows=slice(None): rms(other[key][rows] - base[key][rows])
        action_geom, action_all = diff("action", geom, slice(1, None)), diff("action", full, slice(1, None))
        row = [
            action_geom,
            action_all,
            action_geom / action_all,
            diff("fk", geom),
            diff("fk", full),
            diff("pointflow", geom),
            diff("pointflow", full),
            diff("vision", geom),
        ]
        print(case + " " + " ".join(f"{value:.4f}" for value in row))
        if case.startswith("val"):
            val_rows.append(row)
    if val_rows:
        print(f"val n={len(val_rows)}")
        for label, reduce in (("mean", statistics.mean), ("median", statistics.median)):
            print(label + " " + " ".join(f"{reduce(column):.4f}" for column in zip(*val_rows)))


if __name__ == "__main__":
    main()
