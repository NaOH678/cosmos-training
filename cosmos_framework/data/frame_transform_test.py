import numpy as np
import pytest

from cosmos_framework.data import frame_transform as ft


def _transform():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # a proper rotation
    extrinsic[:3, 3] = [0.3, -0.2, 1.1]
    base = np.eye(4)
    base[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    base[:3, 3] = [0.5, -0.43, 0.75]
    return ft.robot_from_optical(extrinsic, base), extrinsic, base


def test_transform_is_rigid_and_uses_isaac_convention():
    t, extrinsic, base = _transform()
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-12)
    # optical +z (forward) is the Isaac +x axis of the camera
    forward_world = base[:3, :3] @ t[:3, :3] @ [0, 0, 1]
    np.testing.assert_allclose(forward_world, extrinsic[:3, :3] @ [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(ft.invert_rigid(t) @ t, np.eye(4), atol=1e-12)


def test_round_trip_and_metadata_helpers():
    t, _, _ = _transform()
    x = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    meta = {"cam_from_frame": ft.invert_rigid(t).astype(np.float32)}
    back = ft.points_to_camera(ft.apply_points(t, x), meta)
    assert back.dtype == np.float32 and np.abs(back - x).max() < 1e-5
    v = ft.vectors_to_camera(ft.apply_vectors(t, x), meta)
    assert np.abs(v - x).max() < 1e-5
    assert ft.points_to_camera(x, {}) is x and ft.points_to_camera(x, None) is x


def _sample(n=64):
    rng = np.random.default_rng(1)
    p = (rng.normal(size=(n, 3)) * 0.1 + [0, 0, 1]).astype(np.float32)
    color = rng.integers(0, 255, (n, 3), dtype=np.uint8)
    normal = rng.normal(size=(n, 3)).astype(np.float32)
    inputs = dict(anchor_xyz=p, normal=normal, color=color, anchor_uv=rng.normal(size=(n, 2)).astype(np.float32))
    inputs.update(ft.voxelize_anchor(p, color, normal))
    fk = dict(inputs=dict(anchor_xyz=p[:42 // 2]), targets=dict(displacement=rng.normal(size=(4, 21, 3)).astype(np.float32)), metadata={})
    pf = dict(inputs=inputs, targets=dict(displacement=rng.normal(size=(4, n, 3)).astype(np.float32)), metadata={})
    return dict(pointflow=pf, fk=fk)


def test_convert_sample_recomputes_voxels_and_keeps_uv():
    t, _, _ = _transform()
    sample = _sample()
    new = ft.convert_sample_to_robot(sample, t)
    pf = new["pointflow"]
    np.testing.assert_array_equal(pf["inputs"]["anchor_uv"], sample["pointflow"]["inputs"]["anchor_uv"])
    again = ft.voxelize_anchor(pf["inputs"]["anchor_xyz"], pf["inputs"]["color"], pf["inputs"]["normal"])
    for key, value in again.items():
        np.testing.assert_array_equal(pf["inputs"][key], value)
    assert pf["metadata"]["frame"] == "robot_base" and new["fk"]["metadata"]["coordinate"] == "robot_base_link"
    back = ft.points_to_camera(pf["inputs"]["anchor_xyz"], pf["metadata"])
    assert np.abs(back - sample["pointflow"]["inputs"]["anchor_xyz"]).max() < 1e-5
    assert "frame" not in sample["pointflow"]["metadata"]  # input untouched
    with pytest.raises(ValueError):
        ft.convert_sample_to_robot(new, t)


def test_frame_env(monkeypatch):
    monkeypatch.delenv(ft.FRAME_ENV, raising=False)
    assert ft.frame_from_env() == "camera"
    monkeypatch.setenv(ft.FRAME_ENV, "robot_base")
    assert ft.frame_from_env() == "robot_base"
    monkeypatch.setenv(ft.FRAME_ENV, "world")
    with pytest.raises(ValueError):
        ft.frame_from_env()
