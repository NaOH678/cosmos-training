"""Stateful RGB-D rejection and one-observation repair; no future-frame access.

Input XYZ is in the RGB optical frame, in metres. Call once per actual depth
observation, in timestamp order, with masks produced by the same online pipeline.
Optical flow is computed only between the previous and current available images.
"""

import time

import cv2
import numpy as np


class CausalDepthFilter:
    """Reject inconsistent depth with motion correspondence; never recycle repairs."""

    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.previous = None
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.flow.setUseSpatialPropagation(True)

    @staticmethod
    def raster(points, uv, ids, selected, shape):
        xyz = np.zeros((*shape, 3), np.float32)
        inst = np.zeros(shape, np.uint8)
        index = np.flatnonzero(selected)
        # Assign far-to-near so that the nearest measured surface wins.
        index = index[np.argsort(-points[index, 2], kind="stable")]
        x, y = uv[index].T
        xyz[y, x] = points[index]
        inst[y, x] = ids[index]
        return xyz, inst

    def update(self, image, instance_mask, points, uv, ids, spatial, timestamp):
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        accepted = spatial.copy()
        repairs = np.empty((0, 3), np.float32)
        repair_ids = np.empty(0, np.uint8)
        repair_uv = np.empty((0, 2), np.int32)
        stats = dict(matched=0, temporal_rejected=0, repaired=0, instance_motion={})
        previous = self.previous
        if previous is not None:
            dt = float(timestamp - previous["time"])
            if dt <= 0:
                raise ValueError("Depth timestamps must strictly increase")
            # Do not bridge a sensor outage. This is a time limit, not a frame count.
            if dt <= 0.30:
                forward = self.flow.calc(previous["gray"], gray, None)
                backward = self.flow.calc(gray, previous["gray"], None)
                yy, xx = np.indices(gray.shape, dtype=np.float32)
                map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
                warp = lambda value, mode: cv2.remap(  # noqa: E731
                    value, map_x, map_y, mode, borderMode=cv2.BORDER_CONSTANT
                )
                prior_xyz = warp(previous["xyz"], cv2.INTER_NEAREST)
                prior_ids = warp(previous["ids"], cv2.INTER_NEAREST)
                residual = np.linalg.norm(backward + warp(forward, cv2.INTER_LINEAR), axis=2)
                photometric = np.abs(gray.astype(float) - warp(previous["gray"], cv2.INTER_LINEAR))
                correspondence = (
                    (map_x >= 0)
                    & (map_x < width - 1)
                    & (map_y >= 0)
                    & (map_y < height - 1)
                    & (residual <= 1.5)
                    & (photometric <= 30)
                    & (prior_ids > 0)
                    & (prior_ids == instance_mask)
                )
                current_xyz, current_ids = self.raster(points, uv, ids, spatial, gray.shape)
                for instance in np.unique(ids[spatial]):
                    region = correspondence & (instance_mask == instance)
                    support = region & (current_ids == instance)
                    if support.sum() < 40:
                        continue
                    delta = current_xyz[..., 2][support] - prior_xyz[..., 2][support]
                    median = float(np.median(delta))
                    mad = float(np.median(np.abs(delta - median)))
                    # Compensate coherent modest axial motion. A large step is
                    # untrusted rather than silently redefining the expected depth.
                    max_step = 0.020 + 0.30 * dt
                    motion_trusted = abs(median) <= max_step
                    motion = median if motion_trusted else 0.0
                    tolerance = min(0.045, max(0.020, 3 * 1.4826 * mad))
                    index = np.flatnonzero(spatial & (ids == instance))
                    x, y = uv[index].T
                    matched = region[y, x]
                    error = np.abs(points[index, 2] - prior_xyz[y, x, 2] - motion)
                    reject = matched & (error > tolerance)
                    accepted[index[reject]] = False
                    stats["matched"] += int(matched.sum())
                    stats["temporal_rejected"] += int(reject.sum())
                    stats["instance_motion"][str(int(instance))] = dict(
                        delta_z_m=median,
                        mad_m=mad,
                        tolerance_m=tolerance,
                        trusted=motion_trusted,
                        support=int(support.sum()),
                    )
                    # Repair only using reliable motion and an interior pixel;
                    # never extrapolate an unobserved/new/disoccluded surface.
                    if not motion_trusted or mad > 0.012:
                        continue
                    interior = cv2.erode((instance_mask == instance).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    valid_xyz, valid_ids = self.raster(points, uv, ids, accepted, gray.shape)
                    repair = region & interior & (valid_ids == 0)
                    pred_z = prior_xyz[..., 2] + motion
                    # Do not put a repaired surface behind a closer current measurement.
                    raw_xyz, raw_ids = self.raster(points, uv, ids, ids > 0, gray.shape)
                    repair &= (raw_ids == 0) | (raw_xyz[..., 2] >= pred_z - 0.02)
                    repair &= (pred_z >= 0.15) & (pred_z <= 2.5)
                    ry, rx = np.nonzero(repair)
                    z = pred_z[ry, rx]
                    k = self.intrinsics
                    p = np.column_stack(((rx - k["ppx"]) / k["fx"] * z, (ry - k["ppy"]) / k["fy"] * z, z))
                    repairs = np.concatenate((repairs, p.astype(np.float32)))
                    repair_ids = np.concatenate((repair_ids, np.full(len(p), instance, np.uint8)))
                    repair_uv = np.concatenate((repair_uv, np.column_stack((rx, ry)).astype(np.int32)))
        # State contains ONLY current accepted sensor observations, never repairs.
        xyz, raster_ids = self.raster(points, uv, ids, accepted, gray.shape)
        self.previous = dict(gray=gray.copy(), xyz=xyz, ids=raster_ids, time=float(timestamp))
        stats["repaired"] = len(repairs)
        stats["milliseconds"] = (time.perf_counter() - start) * 1000
        return accepted, repairs, repair_ids, repair_uv, stats
