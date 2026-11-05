"""Three HTMLs: one causal hand-derived Z shift applied to the whole foreground."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from build_track4world_core_html import export
from render_d435_point_accuracy import load_module

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    source = Path("pointflow_outputs/track4world_instance_core_html_20261104") / args.task
    old = json.loads((source / "manifest.json").read_text())
    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    module = load_module("hand_depth_guard", str(ROOT / "cosmos_framework/inference/robot_policy/hand_depth_guard.py"))
    hand_names = [name for name in old["instances"].values() if "hand" in name]
    guards = {name: module.HandDepthGuard() for name in hand_names}
    rows = []
    for oldrow, frame in zip(old["stats"], old["frames"]):
        with np.load(source / "frames" / f"{frame:06d}.npz") as n:
            payload = {k: n[k] for k in n.files if k != "scaled"}
        start = time.perf_counter()
        measured = []
        predicted = []
        hand_details = []
        for p in oldrow["parameters"]:
            if p["name"] not in guards:
                continue
            observed = p.get("mode_Z_m")
            result = guards[p["name"]].update(oldrow["time"], observed, p["accepted"])
            if result["status"] == "measured":
                delta = p["offset_m"]
                measured.append(delta)
            elif result["status"] == "predicted":
                # If an old current-frame fit exists, preserve its DA3 pairing;
                # replace only the bad sensor Z with the historical prediction.
                if p["accepted"] and observed is not None:
                    delta = p["offset_m"] + result["z"] - observed
                else:
                    hand_id = p["instance"]
                    points = payload["points"][payload["instances"] == hand_id]
                    if not len(points):
                        delta = None
                    else:
                        delta = result["z"] - float(np.median(points[:, 2]))
                if delta is not None:
                    predicted.append(delta)
            else:
                delta = None
            hand_details.append(
                dict(name=p["name"], observed_z_m=observed, raw_offset_m=p["offset_m"], used_offset_m=delta, **result)
            )
        if measured:
            shift = float(np.median(measured))
            status = "measured"
        elif predicted:
            shift = float(np.median(predicted))
            status = "predicted"
        else:
            shift = 0.0
            status = "no_sensor_correction"
        guard_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        payload["translated"] = payload["points"].copy()
        payload["translated"][:, 2] += shift
        apply_ms = (time.perf_counter() - start) * 1000
        np.savez_compressed(out / "frames" / f"{frame:06d}.npz", **payload)
        row = dict(oldrow)
        row.update(
            shared_shift_m=shift,
            shared_shift_status=status,
            hand_guard=hand_details,
            guard_ms=guard_ms,
            translation_ms=apply_ms,
        )
        row["parameters"] = [
            dict(p, offset_m=shift, accepted=status != "no_sensor_correction", reason="共用手部Z平移：" + status)
            for p in oldrow["parameters"]
        ]
        rows.append(row)
    manifest = dict(
        old,
        stats=rows,
        per_instance=False,
        shared_hand_shift=True,
        guard_rule="last 4 accepted hand-core Z values; median velocity clipped ±0.6m/s; reject residual >40mm+0.20m/s*min(age,0.25s); predict <=2 frames and <=0.5s; do not update history with rejected/predicted depths; then discard D435 correction and show original DA3",
        shared_rule="Prefer accepted observed hands; median hand delta if >1. Otherwise use short predicted hand delta. Apply identical delta to every hand/object point; never estimate object delta.",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = dict(
        task=args.task,
        frames=len(rows),
        status_counts={
            s: sum(r["shared_shift_status"] == s for r in rows)
            for s in ["measured", "predicted", "no_sensor_correction"]
        },
        guard_p50_ms=float(np.median([r["guard_ms"] for r in rows])),
        guard_p95_ms=float(np.percentile([r["guard_ms"] for r in rows], 95)),
        apply_p50_ms=float(np.median([r["translation_ms"] for r in rows])),
        hand_status_counts={
            name: {
                s: sum(h["status"] == s for r in rows for h in r["hand_guard"] if h["name"] == name)
                for s in ["measured", "predicted", "unavailable"]
            }
            for name in hand_names
        },
    )
    (out / "shared_shift_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    export(out, manifest, methods=[("z_translation", "translated")])
    print("COMPLETE SHARED", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["sandwich", "dropper", "micro"], required=True)
    parser.add_argument("--out", type=Path, default=Path("pointflow_outputs/track4world_shared_hand_shift_20261104"))
    main(parser.parse_args())
