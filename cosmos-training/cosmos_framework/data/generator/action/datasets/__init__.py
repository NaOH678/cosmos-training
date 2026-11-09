# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action dataset wrappers for Cosmos Action.

All concrete datasets inherit from :class:`ActionBaseDataset` and expose a
``load_action_stats()`` classmethod for retrieving pre-computed normalization
statistics without instantiating the dataset.
"""

from importlib import import_module

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset

# Optional dataset backends are loaded only when requested.
_DATASETS = {
    "AgiBotWorldBetaLeRobotDataset": "agibotworld_beta_lerobot_dataset",
    "BridgeOrigLeRobotDataset": "bridge_orig_lerobot_dataset",
    "DROIDLeRobotDataset": "droid_lerobot_dataset",
    "DROIDMergedLeRobotDataset": "droid_merged_lerobot_dataset",
    "FractalLeRobotDataset": "fractal_lerobot_dataset",
    "HumanHandPoseLeRobotDataset": "human_hand_pose_lerobot_dataset",
    "LIBEROLeRobotDataset": "libero_lerobot_dataset",
    "RoboMINDFrankaDataset": "robomind_franka_dataset",
    "RoboMINDURDataset": "robomind_ur_dataset",
    "UMILeRobotDataset": "umi_lerobot_dataset",
}

__all__ = ["ActionBaseDataset", *_DATASETS]


def __getattr__(name):
    if name not in _DATASETS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_DATASETS[name]}", __name__), name)
    globals()[name] = value
    return value
