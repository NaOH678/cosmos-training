"""Causal point budgets shared by preprocessing and selection previews."""

import numpy as np


def anchor_fps(xyz, count, initial=()):
    """Deterministic XYZ FPS; ties use candidate order, never future labels."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite anchors")
    initial = np.asarray(initial, dtype=np.int64)
    count = min(count, len(xyz) - len(initial))
    selected = np.empty(count, dtype=np.int64)
    if count == 0:
        return selected
    index = int(np.argmin(np.square(xyz - xyz.mean(0)).sum(1)))
    nearest = np.full(len(xyz), np.inf)
    used = np.zeros(len(xyz), dtype=bool)
    for prior in initial:
        nearest = np.minimum(nearest, np.square(xyz - xyz[prior]).sum(1))
        used[prior] = True
    if len(initial):
        nearest[used] = -np.inf
        index = int(nearest.argmax())
    for k in range(count):
        selected[k] = index
        used[index] = True
        nearest = np.minimum(nearest, np.square(xyz - xyz[index]).sum(1))
        nearest[used] = -np.inf
        index = int(nearest.argmax())
    return selected


def hand_guided_fps(xyz, fk, count, radius):
    from scipy.spatial import cKDTree

    global_count = count // 2
    per_hand = (count - global_count) // 2
    selected = anchor_fps(xyz, global_count).tolist()
    distances = np.stack([cKDTree(fk[h * 21 : (h + 1) * 21]).query(xyz)[0] for h in range(2)], axis=1)
    nearest_hand = distances.argmin(1)
    details = dict(global_points=len(selected), radius_metres=radius, local_points=[])
    for hand in range(2):
        eligible = (nearest_hand == hand) & (distances[:, hand] <= radius)
        eligible[selected] = False
        candidates = np.flatnonzero(eligible)
        extra = candidates[anchor_fps(xyz[candidates], per_hand)]
        selected.extend(extra.tolist())
        details["local_points"].append(len(extra))
    fill = anchor_fps(xyz, count - len(selected), initial=selected)
    details["global_fallback_points"] = len(fill)
    selected.extend(fill.tolist())
    return np.asarray(selected, dtype=np.int64), details


# --- objq: training-time object point quota (opt-in; everything above is unchanged) -------------------------
# Shared byte-for-byte by training preprocessing (tools/prepare_sim_pointfk_v3.py --object-quota) and the
# Bench2Dex inference copy (policy/Cosmos/anchor_selection.py).  Everything from this marker to EOF is copied verbatim to the inference copy (compare the sha256 of that text).
OBJECT_QUOTA_TASKS = {
    "task27": (("soccer", "sphere"), ("tennis_ball", "sphere"), ("golf_ball", "sphere"), ("table_tennis", "sphere")),
    "task21": (("msg", "box"), ("soy", "box"), ("vinegar", "box")),
}
OBJECT_MARGIN_METRES = 0.005


def match_quota_objects(names, task):
    """[(object name, kind)] in the task's fixed order; each fragment must match exactly one ``obj_*`` name."""
    result = []
    for part, kind in OBJECT_QUOTA_TASKS[task]:
        hits = sorted(n for n in names if n.startswith("obj_") and part in n)
        if len(hits) != 1:
            raise ValueError(f"object quota {task}: expected exactly one obj_* matching {part!r}, got {hits}")
        result.append((hits[0], kind))
    return result


def quota_object_in_camera(kind, center_world, quat_xyzw, size_lwh, world_from_optical):
    """Box3d (world centre, xyzw quaternion, local size) -> descriptor in the optical camera frame."""
    from scipy.spatial.transform import Rotation

    world_from_optical = np.asarray(world_from_optical, dtype=np.float64)
    rotation = world_from_optical[:3, :3]
    size = np.asarray(size_lwh, dtype=np.float64)
    box_rotation = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    return dict(
        kind=kind,
        center=rotation.T @ (np.asarray(center_world, dtype=np.float64) - world_from_optical[:3, 3]),
        rotation=rotation.T @ box_rotation,
        half=size / 2,
        radius=float(size.max() / 2),
    )


def object_membership(xyz, objects, margin=OBJECT_MARGIN_METRES):
    """Per-point object id (-1: none).  Sphere: distance to centre <= radius + margin; box: inside the
    oriented box expanded by margin.  A point inside several objects goes to the smallest normalised score."""
    xyz = np.asarray(xyz, dtype=np.float64)
    best = np.full(len(xyz), np.inf)
    index = np.full(len(xyz), -1, dtype=np.int64)
    for i, obj in enumerate(objects):
        offset = xyz - np.asarray(obj["center"], dtype=np.float64)
        if obj["kind"] == "sphere":
            score = np.sqrt(np.square(offset).sum(1)) / (obj["radius"] + margin)
        else:
            local = offset @ np.asarray(obj["rotation"], dtype=np.float64)
            score = (np.abs(local) / (np.asarray(obj["half"], dtype=np.float64) + margin)).max(1)
        take = (score <= 1.0) & (score < best)
        best[take] = score[take]
        index[take] = i
    return index


def object_quota_hand_guided_fps(xyz, fk, count, radius, object_index, quota):
    """hand_guided_fps with a guaranteed per-object quota.

    object_index[i] is the object id (0..K-1) of candidate i, -1 for none.  Order: per-object FPS (up to
    ``quota`` each, all if fewer) -> global FPS until the global part is count//2 (those points as initial
    set) -> per-hand local FPS exactly as hand_guided_fps -> global fill to count.  Unique, no future info.
    With no object candidates it reproduces hand_guided_fps exactly.
    """
    from scipy.spatial import cKDTree

    object_index = np.asarray(object_index)
    selected = []
    object_points = {}
    for obj in np.unique(object_index[object_index >= 0]).tolist():
        candidates = np.flatnonzero(object_index == obj)
        chosen = candidates[anchor_fps(xyz[candidates], quota)]
        selected.extend(chosen.tolist())
        object_points[int(obj)] = len(chosen)
    global_count = count // 2
    selected.extend(anchor_fps(xyz, max(global_count - len(selected), 0), initial=selected).tolist())
    per_hand = (count - global_count) // 2
    distances = np.stack([cKDTree(fk[h * 21 : (h + 1) * 21]).query(xyz)[0] for h in range(2)], axis=1)
    nearest_hand = distances.argmin(1)
    details = dict(
        global_points=len(selected), radius_metres=radius, local_points=[], object_quota=int(quota), object_points=object_points
    )
    for hand in range(2):
        eligible = (nearest_hand == hand) & (distances[:, hand] <= radius)
        eligible[selected] = False
        candidates = np.flatnonzero(eligible)
        extra = candidates[anchor_fps(xyz[candidates], per_hand)]
        selected.extend(extra.tolist())
        details["local_points"].append(len(extra))
    fill = anchor_fps(xyz, count - len(selected), initial=selected)
    details["global_fallback_points"] = len(fill)
    selected.extend(fill.tolist())
    return np.asarray(selected, dtype=np.int64), details
