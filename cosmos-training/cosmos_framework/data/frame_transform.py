"""Rigid frame changes for PointFlow/FK geometry: optical camera <-> URDF ``base_link``.

The cache and the model inputs are in the fixed optical camera frame (x right, y down,
z forward) by default.  ``robot_base`` is the root link of the robot URDF.  Only
positions, displacements and normals are frame dependent; image coordinates
(``anchor_uv``, ``uv_to_video``, intrinsics) always stay in the camera image.

Isaac ``extrinsic_world_from_cam`` is NOT an optical-frame pose: it uses the Isaac/ROS
camera convention (+X forward, +Y left, +Z up), so ``R_world_from_optical = R_E @ C``.
"""

import json
import os

import numpy as np

CAMERA = "camera"
ROBOT_BASE = "robot_base"
FRAMES = (CAMERA, ROBOT_BASE)
FRAME_ENV = "POINTFK_GEOMETRY_FRAME"
# Isaac camera axes (x fwd, y left, z up) expressed in optical axes: R_world_from_optical = R_E @ C.
ISAAC_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def frame_from_env(default=CAMERA):
    frame = os.environ.get(FRAME_ENV) or default
    if frame not in FRAMES:
        raise ValueError(f"{FRAME_ENV} must be one of {FRAMES}, got {frame!r}")
    return frame


def world_from_optical(extrinsic_world_from_cam):
    result = np.array(extrinsic_world_from_cam, dtype=np.float64)
    result[:3, :3] = result[:3, :3] @ ISAAC_TO_OPTICAL
    return result


def invert_rigid(transform):
    transform = np.asarray(transform, dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def robot_from_optical(extrinsic_world_from_cam, world_from_robot_base):
    """T_robot_from_optical = inv(W_base) @ (E with R <- R_E @ C), float64."""
    return invert_rigid(world_from_robot_base) @ world_from_optical(extrinsic_world_from_cam)


def apply_points(transform, points):
    """Apply a 4x4 rigid transform to [...,3] points (float64 arithmetic, input dtype out)."""
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    moved = points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    return moved.astype(points.dtype) if np.issubdtype(points.dtype, np.floating) else moved


def apply_vectors(transform, vectors):
    """Rotate [...,3] displacements/normals (no translation)."""
    vectors = np.asarray(vectors)
    transform = np.asarray(transform, dtype=np.float64)
    moved = vectors.astype(np.float64) @ transform[:3, :3].T
    return moved.astype(vectors.dtype) if np.issubdtype(vectors.dtype, np.floating) else moved


def points_to_camera(points, metadata):
    """Back to the camera frame for projection; identity when the sample is already camera frame."""
    cam_from_frame = (metadata or {}).get("cam_from_frame")
    if cam_from_frame is None:
        return points
    return apply_points(np.asarray(_numpy(cam_from_frame), dtype=np.float64), _numpy(points))


def vectors_to_camera(vectors, metadata):
    cam_from_frame = (metadata or {}).get("cam_from_frame")
    if cam_from_frame is None:
        return vectors
    return apply_vectors(np.asarray(_numpy(cam_from_frame), dtype=np.float64), _numpy(vectors))


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def voxelize_anchor(point, color, normal, voxel_size=0.02):
    """Sonata voxel inputs for one anchor set (the exact recipe of tools/prepare_sim_pointfk_v3.py)."""
    shift = (point.min(0) + point.max(0)) / 2
    shift[2] = point[:, 2].min()
    center = point - shift
    grid = np.floor(center / voxel_size).astype(np.int64)
    grid -= grid.min(0)
    _, reps, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    feat = np.concatenate([center, color.astype(np.float32) / 255, normal], axis=1)
    return dict(
        coord_shift=shift,
        coord=center[reps],
        feat=feat[reps],
        grid_coord=grid[reps],
        original_to_voxel=inverse,
        voxel_representatives=reps,
    )


def read_episode_transform(episode_root, frame_index=0):
    """robot_from_optical for one exported episode directory (rgbd.hdf5 + fk21.json)."""
    import h5py
    from pathlib import Path

    root = Path(episode_root)
    base = np.array(json.loads((root / "fk21.json").read_text())["world_from_robot_base"], dtype=np.float64)
    with h5py.File(root / "rgbd.hdf5", "r") as handle:
        extrinsics = handle["cameras/cam_overhead/extrinsic_world_from_cam"][:]
    return robot_from_optical(extrinsics[frame_index], base), extrinsics, base


def _add_frame_metadata(metadata, robot_from_cam):
    metadata["frame"] = ROBOT_BASE
    metadata["cam_from_frame"] = invert_rigid(robot_from_cam).astype(np.float32)
    metadata["frame_from_cam"] = np.asarray(robot_from_cam, dtype=np.float64).astype(np.float32)


def convert_sample_to_robot(sample, robot_from_cam):
    """Return a copy of a camera-frame cache sample with positions in the robot base frame.

    Rotated: anchor_xyz (+translation), normal, target displacements.  Unchanged: uv,
    color, point_ids, intrinsics, validity.  Sonata coord/feat/grid/voxel mapping are
    recomputed because the voxel grid is axis-aligned in the new frame.
    """
    if sample["pointflow"]["metadata"].get("frame", CAMERA) != CAMERA or sample["fk"]["metadata"].get("frame", CAMERA) != CAMERA:
        raise ValueError("Sample is not in the camera frame")
    out = dict(sample)
    pointflow, fk = dict(sample["pointflow"]), dict(sample["fk"])
    inputs = dict(pointflow["inputs"])
    inputs["anchor_xyz"] = apply_points(robot_from_cam, inputs["anchor_xyz"])
    inputs["normal"] = apply_vectors(robot_from_cam, inputs["normal"])
    if len(inputs["anchor_xyz"]):
        inputs.update(voxelize_anchor(inputs["anchor_xyz"], inputs["color"], inputs["normal"]))
    pointflow["inputs"] = inputs
    pointflow["targets"] = dict(pointflow["targets"], displacement=apply_vectors(robot_from_cam, pointflow["targets"]["displacement"]))
    pointflow["metadata"] = dict(pointflow["metadata"])
    _add_frame_metadata(pointflow["metadata"], robot_from_cam)
    fk["inputs"] = dict(fk["inputs"], anchor_xyz=apply_points(robot_from_cam, fk["inputs"]["anchor_xyz"]))
    fk["targets"] = dict(fk["targets"], displacement=apply_vectors(robot_from_cam, fk["targets"]["displacement"]))
    fk["metadata"] = dict(fk["metadata"], coordinate="robot_base_link")
    _add_frame_metadata(fk["metadata"], robot_from_cam)
    out.update(pointflow=pointflow, fk=fk)
    return out
