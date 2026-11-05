"""Causal UV identity management and depth-only correction for real PointFlow.

The tracker consumes RGB[t-1:t+1], with external episode-global IDs. No future
observations, object names, FK fitting, or repaired-depth feedback are used.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

FILTER_REVISION = 3


class CausalPointTracks:
    def __init__(self, grid_step=3):
        self.grid_step = grid_step
        self.ids = np.empty(0, np.int64)
        self.uv = np.empty((0, 2), np.float32)
        self.labels = np.empty(0, np.uint8)
        self.births = []
        self.track_labels = []

    def advance(self, labels, propagated_uv=None, visible=None):
        h, w = labels.shape
        if len(self.ids):
            uv = np.asarray(propagated_uv, np.float32)
            finite = np.isfinite(uv).all(1)
            q = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(np.int32)
            inside = finite & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            q[:, 0] = q[:, 0].clip(0, w - 1)
            q[:, 1] = q[:, 1].clip(0, h - 1)
            keep = inside & np.asarray(visible, bool) & (labels[q[:, 1], q[:, 0]] == self.labels)
            slots = np.flatnonzero(keep)
            _, first = np.unique(q[slots, 1] * w + q[slots, 0], return_index=True)
            slots = slots[np.sort(first)]
            self.ids, self.uv, self.labels = self.ids[slots], uv[slots], self.labels[slots]
        yy, xx = np.mgrid[0 : h : self.grid_step, 0 : w : self.grid_step]
        valid = labels[yy, xx] > 0
        new = np.column_stack((xx[valid], yy[valid])).astype(np.float32)
        regions = labels[yy[valid], xx[valid]]
        keep = np.ones(len(new), bool)
        for label in np.unique(regions):
            old = self.uv[self.labels == label]
            which = np.flatnonzero(regions == label)
            if len(old):
                distance = cKDTree(old).query(new[which], workers=1)[0]
                keep[which] = distance > self.grid_step
        new, regions = new[keep], regions[keep]
        born = np.arange(len(self.births), len(self.births) + len(new), dtype=np.int64)
        self.ids = np.r_[self.ids, born]
        self.uv = np.concatenate((self.uv, new))
        self.labels = np.r_[self.labels, regions]
        self.track_labels.extend(regions.tolist())
        return born

    def record_births(self, frame, born):
        if len(self.births) != len(self.track_labels) - len(born):
            raise RuntimeError("Births must be recorded once per frame")
        self.births.extend([frame] * len(born))


def translate_z(points, delta):
    """Rigid camera-Z shift; X/Y (and the independently stored UV) never change."""
    result = points.copy()
    result[:, 2] += delta
    return result


def depth_sample_valid(uv, depth, confidence, instances):
    """DA3 confidence plus local same-instance depth discontinuity rejection."""
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 2.5) & (confidence > 2)
    for instance in np.unique(instances):
        which = np.flatnonzero((instances == instance) & valid)
        if len(which) < 3:
            continue
        distances, nearest = cKDTree(uv[which]).query(uv[which], k=min(5, len(which)), workers=1)
        difference = abs(depth[which][nearest] - depth[which, None])
        abrupt = (distances <= 5.0) & (difference > np.maximum(0.02, 0.04 * depth[which, None]))
        valid[which[np.any(abrupt, axis=1)]] = False
    return valid


class CausalGeometryFilter:
    """Current spatial support and per-ID temporal depth residuals.

    Only accepted measured points enter history. Rejected points remain invalid;
    a display carry-forward is never exported as a measured training target.
    """

    def __init__(self):
        self.last_ids = np.empty(0, np.int64)
        self.last_points = np.empty((0, 3), np.float32)
        self.last_time = None

    def apply(self, ids, points, valid, instances, timestamp):
        keep = valid.copy() & np.isfinite(points).all(1) & (points[:, 2] > 0.1) & (points[:, 2] < 2.5)
        temporal_removed = spatial_removed = 0
        velocity_removed = 0
        component_removed = 0
        if self.last_time is not None and timestamp - self.last_time <= 0.2:
            _, ii, jj = np.intersect1d(ids, self.last_ids, assume_unique=True, return_indices=True)
            good = keep[ii]
            ii, jj = ii[good], jj[good]
            change = points[ii, 2] - self.last_points[jj, 2]
            step = np.linalg.norm(points[ii] - self.last_points[jj], axis=1)
            too_fast = step > max(0.08, 1.5 * (timestamp - self.last_time))
            keep[ii[too_fast]] = False
            velocity_removed = int(too_fast.sum())
            for instance in np.unique(instances[ii]):
                which = np.flatnonzero(instances[ii] == instance)
                if len(which) < 12:
                    continue
                delta = change[which]
                median = np.median(delta)
                mad = np.median(abs(delta - median))
                bad = abs(delta - median) > max(0.025, 5 * 1.4826 * mad)
                keep[ii[which[bad]]] = False
                temporal_removed += int(bad.sum())
        for instance in np.unique(instances[keep]):
            which = np.flatnonzero(keep & (instances == instance))
            if len(which) < 16:
                continue
            if len(which) >= 30:
                local = points[which]
                pairs = cKDTree(local).query_pairs(0.018, output_type="ndarray")
                graph = csr_matrix(
                    (np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(local), len(local))
                )
                n_components, components = connected_components(graph, directed=False)
                if n_components > 1:
                    sizes = np.bincount(components)
                    main = int(np.argmax(sizes))
                    main_points = local[components == main]
                    nearest = cKDTree(main_points).query(local, workers=1)[0]
                    main_z = np.median(main_points[:, 2])
                    for label in range(n_components):
                        part = components == label
                        if (
                            sizes[label] < 0.9 * sizes[main]
                            and nearest[part].min() > 0.06
                            and abs(np.median(local[part, 2]) - main_z) > 0.06
                        ):
                            keep[which[part]] = False
                            component_removed += int(part.sum())
                    which = which[keep[which]]
                    if len(which) < 16:
                        continue
            distances = cKDTree(points[which]).query(points[which], k=min(17, len(which)), workers=1)[0][:, 1:].mean(1)
            median = np.median(distances)
            mad = np.median(abs(distances - median))
            bad = distances > max(0.008, median + 5 * 1.4826 * mad)
            keep[which[bad]] = False
            spatial_removed += int(bad.sum())
        self.last_ids = ids[keep].copy()
        self.last_points = points[keep].copy()
        self.last_time = timestamp
        return keep, dict(
            temporal_removed=temporal_removed,
            spatial_removed=spatial_removed,
            velocity_removed=velocity_removed,
            component_removed=component_removed,
        )
