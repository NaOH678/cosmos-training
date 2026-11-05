"""Two format-review HTMLs: existing scissors V1/V2 with visible URDF hand mesh points."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pointflow_outputs/scissors_v1_v2_urdf_review_20261105"


def prepare(version):
    name = "v1_offline_da3" if version == "v1" else "v2_d435"
    source = ROOT / f"pointflow_outputs/cotracker_real_{name}_20261104/scissors"
    dest = OUT / version / "scissors"
    dest.mkdir(parents=True, exist_ok=True)
    for item in ["frames", "source.json", "task_config.json", "points_complete.json", "per_frame.json"]:
        target = dest / item
        if not target.exists():
            target.symlink_to(source / item, target_is_directory=item == "frames")
    return source, dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()
    source, dest = prepare(args.version)
    if args.version == "v1":
        import run_cotracker_offline_da3 as export

        export.OUT = OUT / "v1"
        export.html("scissors", show_urdf=True, surface_stride=4)
    else:
        import run_dropper_d435_v2 as export

        export.cv2.setNumThreads(2)
        export.TASK = "scissors"
        export.VERSION = "v2_d435"
        export.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
        export.OUT = dest
        export.html(show_urdf=True, surface_stride=4)
    (dest / "review.json").write_text(
        json.dumps(
            dict(
                original=str(source),
                frames_reused=True,
                depth_recomputed=False,
                urdf_visible=True,
                urdf="assets/dagger_fk/marvin_wuji_d435_dagger.urdf",
                hand_sides=["left", "right"],
                surface_stride=4,
            ),
            indent=2,
        )
    )
