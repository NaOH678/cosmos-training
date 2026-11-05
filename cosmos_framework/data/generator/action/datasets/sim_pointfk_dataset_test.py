# SPDX-License-Identifier: OpenMDW-1.1
"""Mode-mixing (SIM_MODE_PROBS) tests for the sim PointFlow-FK dataset: CPU only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.sim_pointfk_dataset import draw_mode, parse_mode_probs
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder

LATENT_T, ACTION_T = 9, 32  # cached sim window: 33 frames -> 9 latent frames, 32 action rows


def _plan(mode):
    pipe = ActionTransformPipeline(tokenizer_config=None, max_action_dim=4, video_temporal_downsample=4)
    d = {
        "ai_caption": "x",
        "video": torch.zeros(3, 1, 64, 64),  # cached_video_only: anchor frame only
        "cached_video_num_frames": 33,
        "action": torch.zeros(ACTION_T, 2),
        "conditioning_fps": torch.tensor(20),
        "mode": mode,
        "domain_id": torch.tensor(0),
        "viewpoint": "concat_view",
    }
    plan = pipe(d, resolution="256")["sequence_plan"]
    plan.has_point = True  # the dataset sets this from the PointFlow payload; PF/FK carry no condition mask
    return plan


def _pack(plan):
    b = PackedSequenceBuilder()
    b.ensure_vision(), b.ensure_action(), b.begin_sample(0)
    b.pack_vision_tokens(torch.zeros(1, 4, LATENT_T, 2, 2), plan.condition_frame_indexes_vision, 0.5)
    b.pack_action_tokens(
        torch.zeros(ACTION_T, 4), plan.condition_frame_indexes_action, 0.5,
        action_start_frame_offset=plan.action_start_frame_offset,
    )
    vis_mask = b.vision.condition_mask[0].flatten()  # [T] 1 = clean
    act_mask = b.action.condition_mask[0].flatten()  # [T]
    return vis_mask, act_mask, len(b.vision.mse_loss_indexes), len(b.action.mse_loss_indexes)


@pytest.mark.L0
def test_wam_unchanged():
    plan = _plan("wam")
    vis, act, nv, na = _pack(plan)
    assert vis.tolist() == [1] + [0] * 8 and act.sum() == 0
    assert nv == 8 * 4 and na == ACTION_T
    assert plan.has_point


@pytest.mark.L0
def test_forward_dynamics_masks():
    plan = _plan("forward_dynamics")
    vis, act, nv, na = _pack(plan)
    assert vis.tolist() == [1] + [0] * 8  # first frame clean, rest noisy + loss
    assert act.sum() == ACTION_T and na == 0  # all action rows clean, no action loss
    assert nv == 8 * 4
    assert plan.has_point  # PF/FK tokens carry no condition mask: always noisy + loss


@pytest.mark.L0
def test_inverse_dynamics_masks():
    plan = _plan("inverse_dynamics")
    vis, act, nv, na = _pack(plan)
    assert vis.sum() == LATENT_T and nv == 0  # all video frames clean, no video loss
    assert act.sum() == 0 and na == ACTION_T  # action is the target
    assert plan.has_point  # PF/FK remain targets


@pytest.mark.L0
def test_parse_default_off_and_errors():
    assert parse_mode_probs("") is None and parse_mode_probs(None) is None and parse_mode_probs("wam:1") is None
    modes, probs = parse_mode_probs("wam:3,forward_dynamics:1")
    assert modes == ["wam", "forward_dynamics"] and probs == [0.75, 0.25]
    for bad in ("foo:1", "wam", "wam:1,wam:1", "wam:-1,forward_dynamics:2"):
        with pytest.raises(ValueError):
            parse_mode_probs(bad)


@pytest.mark.L0
def test_mode_frequencies_determinism_independence():
    mp = parse_mode_probs("wam:0.6,forward_dynamics:0.25,inverse_dynamics:0.15")
    draws = [draw_mode(mp, 42, i, 0, 0, 0) for i in range(20000)]
    freq = {m: draws.count(m) / len(draws) for m in mp[0]}
    assert abs(freq["wam"] - 0.6) < 0.02 and abs(freq["forward_dynamics"] - 0.25) < 0.02
    assert abs(freq["inverse_dynamics"] - 0.15) < 0.02
    assert draws == [draw_mode(mp, 42, i, 0, 0, 0) for i in range(20000)]  # reproducible
    other = [draw_mode(mp, 42, i, 1, 0, 0) for i in range(2000)]  # other rank
    worker = [draw_mode(mp, 42, i, 0, 3, 0) for i in range(2000)]  # other worker
    revisit = [draw_mode(mp, 42, i, 0, 0, 1) for i in range(2000)]  # next epoch
    for alt in (other, worker, revisit):
        assert 0.2 < np.mean([a != b for a, b in zip(alt, draws[:2000])]) < 0.8
