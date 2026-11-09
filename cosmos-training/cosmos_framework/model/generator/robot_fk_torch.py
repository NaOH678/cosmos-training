"""Batched, differentiable URDF forward kinematics in torch (fp32).

A port of the numpy reference ``Bench2Dex utils/sim_surface_gt.py::KinematicTree`` +
``utils/sim_robot_profile.py::positions`` (the code that produced the stored
``fk21.npz`` keypoints).  Same conventions: column vectors, metres,
``T_child = T_parent @ T_origin(xyz, rpy) @ T_motion(q)``, revolute motion =
``Rotation.from_rotvec(axis * q)``, prismatic motion = ``axis * q``.

The module has no learnable parameters; it exists so the cross-modal consistency
loss can push gradients from camera-space keypoints back into joint angles.
Joint values are matched to URDF joints BY NAME (``joint_names``), never by position.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import torch


def _rpy_to_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (== scipy 'xyz')."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_urdf(path: str) -> dict:
    """Same parse as ``KinematicTree.from_urdf``: {joint_name: dict(parent, child, before, axis, type)}."""
    joints = {}
    for j in ET.parse(path).getroot().findall("joint"):
        origin, axis = j.find("origin"), j.find("axis")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        before = np.eye(4)
        before[:3, :3] = _rpy_to_matrix(rpy)
        before[:3, 3] = xyz
        joints[j.get("name")] = dict(
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            before=before,
            axis=np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1.0, 0, 0]),
            type=j.get("type"),
        )
    roots = {j["parent"] for j in joints.values()} - {j["child"] for j in joints.values()}
    if len(roots) != 1:
        raise ValueError(f"Expected one kinematic root, got {roots}")
    return dict(joints=joints, root=next(iter(roots)))


class TorchRobotFK(torch.nn.Module):
    """``q[..., J] (joint_names order) -> keypoints[..., K, 3]`` in the frame given by ``world_from_base``.

    ``mapping`` is the flat list of keypoint specs ``{"link": name}`` or
    ``{"link": a, "midpoint_to": b}`` (midpoint of two link origins).
    """

    def __init__(self, urdf_path: str, joint_names, mapping, world_from_base=None):
        super().__init__()
        tree = parse_urdf(urdf_path)
        joints, root = tree["joints"], tree["root"]
        self.joint_names = list(joint_names)
        column = {n: i for i, n in enumerate(self.joint_names)}
        child_to_joint = {j["child"]: n for n, j in joints.items()}

        needed_links = set()
        for spec in mapping:
            needed_links.add(spec["link"])
            if "midpoint_to" in spec:
                needed_links.add(spec["midpoint_to"])
        needed_joints = set()
        for link in needed_links:
            while link != root:
                if link not in child_to_joint:
                    raise ValueError(f"Link {link!r} is not connected to root {root!r}")
                name = child_to_joint[link]
                needed_joints.add(name)
                link = joints[name]["parent"]
        # Topological order (parents before children), restricted to joints that feed a keypoint.
        order, placed, pending = [], {root}, dict.fromkeys(sorted(needed_joints))
        while pending:
            progressed = False
            for name in list(pending):
                if joints[name]["parent"] in placed:
                    order.append(name)
                    placed.add(joints[name]["child"])
                    del pending[name]
                    progressed = True
            if not progressed:
                raise ValueError(f"Disconnected joints: {list(pending)}")
        missing = [n for n in order if joints[n]["type"] != "fixed" and n not in column]
        if missing:
            raise ValueError(f"Action joint names do not cover URDF joints needed for the keypoints: {missing}")
        for n in order:
            if joints[n]["type"] not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported joint type {joints[n]['type']!r} for {n}")

        self.root = root
        self._order = [(joints[n]["parent"], joints[n]["child"], joints[n]["type"], column.get(n, -1)) for n in order]
        self.register_buffer("_before", torch.tensor(np.stack([joints[n]["before"] for n in order]), dtype=torch.float32), persistent=False)
        axes = np.stack([joints[n]["axis"] for n in order])
        norm = np.linalg.norm(axes, axis=-1, keepdims=True)
        self.register_buffer("_axis_norm", torch.tensor(norm[:, 0], dtype=torch.float32), persistent=False)
        self.register_buffer("_axis_unit", torch.tensor(axes / np.where(norm > 0, norm, 1.0), dtype=torch.float32), persistent=False)
        self._mapping = [(s["link"], s.get("midpoint_to")) for s in mapping]
        base = np.eye(4) if world_from_base is None else np.asarray(world_from_base, dtype=np.float64)
        self.register_buffer("_base_R", torch.tensor(base[:3, :3], dtype=torch.float32), persistent=False)
        self.register_buffer("_base_t", torch.tensor(base[:3, 3], dtype=torch.float32), persistent=False)
        self.num_keypoints = len(self._mapping)

    @classmethod
    def from_json(cls, urdf_path: str, fk_json_path: str, joint_names):
        """``fk_json_path`` is a per-episode ``fk21.json`` (mapping + world_from_robot_base; both constant)."""
        spec = json.load(open(fk_json_path))
        mapping = [entry for hand in spec["mapping"] for entry in hand]  # [left 21, right 21] -> 42
        return cls(urdf_path, joint_names, mapping, spec["world_from_robot_base"])

    @staticmethod
    def _motion(kind, q, axis_unit, axis_norm):
        """[N,4,4] motion transform for joint values q [N]."""
        n = q.shape[0]
        out = torch.eye(4, dtype=q.dtype, device=q.device).repeat(n, 1, 1)
        if kind == "prismatic":
            out[:, :3, 3] = (axis_unit * axis_norm) * q[:, None]
            return out
        phi = (q * axis_norm)[:, None, None]  # Rotation.from_rotvec(axis*q): angle |axis|*q about axis/|axis|
        k = axis_unit
        zero = torch.zeros((), dtype=q.dtype, device=q.device)
        kx = torch.stack([zero, -k[2], k[1], k[2], zero, -k[0], -k[1], k[0], zero]).reshape(3, 3)
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        out[:, :3, :3] = eye + torch.sin(phi) * kx + (1.0 - torch.cos(phi)) * (kx @ kx)
        return out

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} joints, got {q.shape[-1]}")
        lead = q.shape[:-1]
        with torch.autocast(device_type=q.device.type, enabled=False):
            flat = q.reshape(-1, q.shape[-1]).to(torch.float32)
            eye = torch.eye(4, dtype=torch.float32, device=flat.device)
            nodes = {self.root: eye.expand(flat.shape[0], 4, 4)}
            for i, (parent, child, kind, col) in enumerate(self._order):
                before = self._before[i]
                if kind == "fixed":
                    nodes[child] = nodes[parent] @ before
                else:
                    motion = self._motion(kind, flat[:, col], self._axis_unit[i], self._axis_norm[i])
                    nodes[child] = nodes[parent] @ (before @ motion)
            pts = []
            for link, other in self._mapping:
                p = nodes[link][:, :3, 3]
                if other is not None:
                    p = (p + nodes[other][:, :3, 3]) / 2
                pts.append(p)
            points = torch.stack(pts, dim=1)  # [N,K,3] robot-base frame
            points = points @ self._base_R.T + self._base_t
        return points.reshape(*lead, self.num_keypoints, 3)


def estimate_rigid_transform(src: torch.Tensor, dst: torch.Tensor):
    """Kabsch: R, t with ``dst ~= src @ R.T + t`` for ``src, dst [B,M,3]`` (float64 SVD, no grad)."""
    with torch.no_grad():
        s, d = src.detach().double(), dst.detach().double()
        sc, dc = s.mean(1, keepdim=True), d.mean(1, keepdim=True)
        h = (s - sc).transpose(1, 2) @ (d - dc)  # [B,3,3]
        u, _, vh = torch.linalg.svd(h)
        sign = torch.sign(torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
        diag = torch.ones(h.shape[0], 3, dtype=h.dtype, device=h.device)
        diag[:, 2] = sign
        r = vh.transpose(1, 2) @ torch.diag_embed(diag) @ u.transpose(1, 2)
        t = dc[:, 0] - torch.einsum("bij,bj->bi", r, sc[:, 0])
    return r.float(), t.float()
