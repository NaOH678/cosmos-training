"""Timing contract for FK windows.

The counterpart of ``pointflow_window.PointFlowTiming``, and a separate type on
purpose: FK runs in a worktree that contains no PointFlow module at all (v1 is
"FK only"), so importing the PointFlow one would drag the whole point-cloud data
path in for a four-field dataclass.  The fields and their validation are the same
because the two modalities must land on the same 15 Hz / 32-step / 4-step-per-token
lattice as the video -- that agreement is checked in ``from_cosmos``, which reads
the same dataset and tokenizer config the video path does.

If PointFlow and FK are ever loaded together (v2), compare the *fields*, not the
types: these are two distinct classes and ``==`` between them is always False.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FKTiming:
    """Physical timing copied from the resolved Cosmos dataset/tokenizer configuration."""

    fps: float = 15.0
    steps: int = 32
    steps_per_token: int = 4

    def __post_init__(self):
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if type(self.steps) is not int or type(self.steps_per_token) is not int:
            raise ValueError("steps and steps_per_token must be integers")
        if self.steps < 1 or self.steps_per_token < 1 or self.steps % self.steps_per_token:
            raise ValueError("Future steps must be positive and divisible by the tokenizer temporal compression")

    @property
    def blocks(self) -> int:
        """Number of motion blocks: one noisy token per keypoint per block."""
        return self.steps // self.steps_per_token

    @classmethod
    def from_cosmos(cls, dataset_config, tokenizer_config):
        timing = cls(
            fps=float(dataset_config["fps"]),
            steps=dataset_config["chunk_length"],
            steps_per_token=tokenizer_config["temporal_compression_factor"],
        )
        # The dataset's window length and the tokenizer's expected clip length have
        # to be the same number: if they disagree the video and the FK labels are
        # built for different durations, and the mRoPE time axis would drift by a
        # whole block without any shape error to catch it.
        durations = tokenizer_config.get("encode_exact_durations")
        if durations is not None and timing.steps + 1 not in durations:
            raise ValueError("FK states do not match Cosmos encode_exact_durations")
        return timing


def fk_window_sample(archive, frame_ids, episode, fps=20):
    """Pack existing dual-hand camera-space FK arrays; no robot-specific FK solving."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    rows = np.searchsorted(archive["source_frame_ids"], frame_ids)
    np.testing.assert_array_equal(archive["source_frame_ids"][rows], frame_ids)
    timing = FKTiming(fps=fps, steps=len(frame_ids) - 1, steps_per_token=4)
    positions = archive["positions_camera"][rows].reshape(len(rows), 42, 3).astype(np.float32)
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite FK positions")
    anchor = positions[0]
    projected = anchor @ archive["intrinsics"].T
    uv = projected[:, :2] / projected[:, 2:]
    width, height = archive["image_size_wh"]
    names = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]
    for finger in ("index", "middle", "ring", "pinky"):
        names += [f"{finger}_{joint}" for joint in ("mcp", "pip", "dip", "tip")]
    return dict(
        inputs=dict(
            anchor_xyz=anchor.copy(),
            point_ids=np.arange(42, dtype=np.int64),
            anchor_uv=uv.astype(np.float32),
            image_size_wh=np.array([width, height]),
        ),
        targets=dict(displacement=positions[1:] - anchor[None], valid=np.ones((timing.steps, 42), bool)),
        metadata=dict(
            episode=episode,
            start_frame=int(frame_ids[0]),
            raw_frame_ids=frame_ids.copy(),
            timing=timing,
            hand="both",
            hand_order=["left", "right"],
            coordinate="fixed_optical_camera",
            point_block_seconds=np.arange(4, timing.steps + 1, 4) / fps,
            geometry_source="observed qpos + URDF; future qpos only supplies target labels",
            validity_semantics="FK geometry known from proprioception, independent of camera visibility",
            keypoint_names=[f"{side}_{name}" for side in ("left", "right") for name in names],
        ),
    )
