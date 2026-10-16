import numpy as np
import pytest

from cosmos_framework.data.pointflow_anchor_selection import (
    hand_guided_fps,
    match_quota_objects,
    object_membership,
    object_quota_hand_guided_fps,
    quota_object_in_camera,
)


def _scene(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([-0.5, -0.4, 0.5], [0.5, 0.4, 1.2], size=(n, 3))
    fk = np.concatenate([rng.uniform(-0.1, 0.1, (21, 3)) + [-0.2, 0, 0.8], rng.uniform(-0.1, 0.1, (21, 3)) + [0.2, 0, 0.8]])
    return xyz, fk


def test_no_object_points_reproduces_hand_guided_fps():
    xyz, fk = _scene()
    old, _ = hand_guided_fps(xyz, fk, 1024, 0.05)
    new, detail = object_quota_hand_guided_fps(xyz, fk, 1024, 0.05, np.full(len(xyz), -1), 32)
    assert np.array_equal(old, new) and detail["object_points"] == {}


def test_object_quota_counts_unique_and_order():
    xyz, fk = _scene()
    index = np.full(len(xyz), -1)
    index[:500], index[500:520], index[520:1000] = 0, 1, 2  # object 1 has fewer than 32 candidates
    keep, detail = object_quota_hand_guided_fps(xyz, fk, 1024, 0.05, index, 32)
    assert len(keep) == 1024 == len(set(keep.tolist()))
    assert detail["object_points"] == {0: 32, 1: 20, 2: 32}
    assert [int((index[keep[:84]] == i).sum()) for i in range(3)] == [32, 20, 32]  # objects come first
    assert detail["global_points"] == 512


def test_membership_sphere_and_box():
    identity = np.eye(4)
    sphere = quota_object_in_camera("sphere", [0, 0, 1], [0, 0, 0, 1], [0.1, 0.1, 0.1], identity)
    box = quota_object_in_camera("box", [1, 0, 1], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], [0.2, 0.04, 0.1], identity)
    pts = np.array([[0.054, 0, 1], [0.056, 0, 1], [1, 0.1, 1], [1, 0.106, 1], [1.02, 0, 1], [1.06, 0, 1], [5, 5, 5]])
    # box is rotated 90 deg about z: its local x (length 0.2) points along world y
    assert object_membership(pts, [sphere, box]).tolist() == [0, -1, 1, -1, 1, -1, -1]


def test_match_quota_objects():
    names = ["obj_053_mini_soccer_ball_2", "obj_056_tennis_ball_3", "obj_058_golf_ball_4", "obj_144_table_tennis_5", "distractor_00_clutter_056_tennis_ball"]
    assert [n for n, _ in match_quota_objects(names, "task27")] == names[:4]
    with pytest.raises(ValueError):
        match_quota_objects(names[:3], "task27")
