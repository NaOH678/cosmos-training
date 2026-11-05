"""Generate only scissors v3, using reusable strictly streaming surface filters."""

import argparse
from pathlib import Path

import run_dropper_d435_v2 as pipeline

ROOT = Path(__file__).resolve().parents[1]
Fusion = pipeline.load_module(
    "causal_surface_v3", ROOT / "cosmos_framework/inference/robot_policy/causal_surface_fusion.py"
).CausalSurfaceFusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pipeline.cv2.setNumThreads(2)
    pipeline.TASK = "scissors"
    pipeline.VERSION = "v3_d435"
    pipeline.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
    pipeline.OUT = ROOT / "pointflow_outputs/cotracker_real_v3_d435_20261105/scissors"
    if args.limit:
        pipeline.OUT = pipeline.OUT.parent / "scissors_smoke"
    pipeline.build(enhancement=Fusion(), limit=args.limit)
    if not args.limit:
        pipeline.html(comparison=ROOT / "pointflow_outputs/cotracker_real_v2_d435_20261104/scissors")
