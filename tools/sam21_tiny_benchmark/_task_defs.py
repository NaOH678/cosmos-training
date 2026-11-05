"""Shared task definitions for the SAM2.1-tiny video segmentation benchmark."""

import json
from pathlib import Path

ROOT = Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk")
OUT_ROOT = ROOT / "pointflow_outputs/seg_model_compare_20261031"
MODEL = "sam2.1_hiera_t"
CHECKPOINT = "/data/shichaojian/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_PYLIBS = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pylibs_sam2"
PYTHON = "/data/shichaojian/Track4World_portable/Track4World/.venv/bin/python"

# tools/segment_sandwich_video.py COLORS fallback for objects without explicit color_bgr.
SANDWICH_COLORS = {
    "robot_hand": (255, 145, 35),
    "lettuce": (50, 210, 70),
    "bread": (200, 75, 210),
    "egg": (30, 145, 250),
    "cheese": (20, 220, 240),
    "plate": (210, 180, 70),
    "tray": (180, 180, 180),
    "bread_slice_1": (215, 70, 215),
    "bread_slice_2": (70, 225, 230),
}

TASKS = {
    "sandwich": ROOT / "pointflow_outputs/sandwich_bread_variants_20261030_v2/separate",
    "dropper": ROOT / "pointflow_outputs/dropper_micro_requested_20261030/dropper_mix",
    "micro": ROOT / "pointflow_outputs/dropper_micro_requested_20261030/micropipette",
}

BASELINE_P50_MS = {"sandwich": 143.0, "dropper": 99.0, "micro": 136.0}


def task_names_colors(task_dir):
    """Object names (mask channel order) and BGR colors for a task."""
    summary = json.loads((task_dir / "summary.json").read_text())
    init = json.loads((task_dir / "initialization.json").read_text())
    names = list(summary["initialized_objects"])
    color_by_name = {}
    for obj in init.get("config", {}).get("objects", []):
        if obj.get("color_bgr") is not None:
            color_by_name[obj["name"]] = tuple(obj["color_bgr"])
    colors = [color_by_name.get(name, SANDWICH_COLORS.get(name, (180, 180, 180))) for name in names]
    return names, colors
