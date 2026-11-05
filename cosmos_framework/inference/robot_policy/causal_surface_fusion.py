"""Current/past-only D435 surface fusion and hand-contact boundary rejection.

This module is independent of dataset readers and rendering. Raw accepted
measurements, filtered surface state and synthetic repairs remain separate.
"""

import cv2
import numpy as np


class CausalSurfaceFusion:
    def __init__(self):
        self.previous = {}
        self.modes = {}
        self.runs = {}
        self.reference = {}
        self.cells = {}
        self.near_hand = None
        self.edge = None
        self.projection_removed = 0
        self.boundary_removed = 0

    def prepare_frame(self, labels, instances, image=None):
        self.gray = (
            cv2.cvtColor(cv2.resize(image, (labels.shape[1], labels.shape[0])), cv2.COLOR_BGR2GRAY)
            if image is not None
            else None
        )
        self.labels = labels
        self.instances = instances
        self.near_hand = cv2.distanceTransform((labels != 2).astype(np.uint8), cv2.DIST_L2, 5) <= 4
        if not np.any(labels == 2):
            self.near_hand[:] = False
        self.projection_removed = 0
        self.boundary_removed = 0

    def occlusion(self, points, pixels, native, streams):
        """Invalidate far samples behind near samples in native epipolar order."""
        shape = native.shape
        uv = pixels.reshape(*shape, 2)
        z = points[:, 2].reshape(shape)
        valid = (native > 0) & np.isfinite(z) & (z > 0.15) & (z < 2.5)
        translation = streams["depth"]["extrinsics_to_color"]["translation"]
        axis = 1 if abs(translation[0]) >= abs(translation[1]) else 0
        coordinate = uv[..., 0 if axis == 1 else 1]
        direction = 1 if translation[0 if axis == 1 else 1] >= 0 else -1
        bad = np.zeros(shape, bool)
        for offset in range(1, 21):
            shift = direction * offset
            prior_u = np.roll(coordinate, shift, axis=axis)
            prior_z = np.roll(z, shift, axis=axis)
            prior_valid = np.roll(valid, shift, axis=axis)
            candidate = valid & prior_valid & (direction * (prior_u - coordinate) > 0.5) & (prior_z + 0.02 < z)
            if axis == 1:
                candidate[:, :offset] = False if direction > 0 else candidate[:, :offset]
                if direction < 0:
                    candidate[:, -offset:] = False
            else:
                candidate[:offset] = False if direction > 0 else candidate[:offset]
                if direction < 0:
                    candidate[-offset:] = False
            bad |= candidate
        self.projection_removed = int(bad.sum())
        return bad.ravel()

    def gate_buffer(self, sensor):
        finite = np.isfinite(sensor)
        kernel = np.ones((3, 3), np.uint8)
        high = cv2.dilate(np.where(finite, sensor, 0).astype(np.float32), kernel)
        low = cv2.erode(np.where(finite, sensor, 1e6).astype(np.float32), kernel)
        support = cv2.filter2D(finite.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        self.edge = (support >= 2) & ((high - low) > np.maximum(0.012, 0.015 * np.where(finite, sensor, 1)))
        block = self.near_hand & self.edge & (self.labels == 3)
        result = sensor.copy()
        result[block] = np.inf
        self.boundary_removed = int((block & finite).sum())
        return result

    def blocked(self, uv, labels):
        q = np.rint(uv).astype(int)
        q[:, 0] = q[:, 0].clip(0, self.labels.shape[1] - 1)
        q[:, 1] = q[:, 1].clip(0, self.labels.shape[0] - 1)
        # A short-lived prior edge may only reject points, never supply depth.
        edge = self.edge if self.edge is not None else np.zeros(self.labels.shape, bool)
        return (labels == 3) & self.near_hand[q[:, 1], q[:, 0]] & edge[q[:, 1], q[:, 0]]

    def image_displacement(self, old, positions, instance):
        if getattr(self, "gray", None) is None or old.get("gray") is None or len(positions) < 12:
            return None
        positions = (
            positions[np.linspace(0, len(positions) - 1, min(96, len(positions))).astype(int)]
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )
        kwargs = dict(
            winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
        )
        target, status, error = cv2.calcOpticalFlowPyrLK(old["gray"], self.gray, positions, None, **kwargs)
        if target is None:
            return None
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(self.gray, old["gray"], target, None, **kwargs)
        if back is None:
            return None
        q = np.rint(target[:, 0]).astype(int)
        inside = (
            (q[:, 0] >= 0) & (q[:, 0] < self.instances.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < self.instances.shape[0])
        )
        q[:, 0] = q[:, 0].clip(0, self.instances.shape[1] - 1)
        q[:, 1] = q[:, 1].clip(0, self.instances.shape[0] - 1)
        good = (
            status.ravel().astype(bool)
            & reverse.ravel().astype(bool)
            & inside
            & (error.ravel() < 20)
            & (np.linalg.norm(back[:, 0] - positions[:, 0], axis=1) < 0.75)
        )
        good &= self.instances[q[:, 1], q[:, 0]] == instance
        return np.linalg.norm(target[good, 0] - positions[good, 0], axis=1) if good.sum() >= 12 else None

    def update(self, ids, uv, instances, labels, raw_z, z, provenance, timestamp, sensor_update, blocked):
        t = float(timestamp)
        out = z.copy()
        p = provenance.copy()
        details = {}
        measured = (p == 1) & np.isfinite(raw_z) & ~blocked
        for instance in np.unique(instances[labels == 3]):
            instance = int(instance)
            which = np.flatnonzero((instances == instance) & (labels == 3))
            old = self.previous.get(instance)
            mode = self.modes.get(instance, "unknown")
            current = which[measured[which]]
            candidate = False
            moving = False
            motion = {}
            if old is not None and t - old["time"] <= 0.3:
                _, a, b = np.intersect1d(old["ids"], ids[which], return_indices=True)
                current_indices = which[b]
                if len(a) >= 12:
                    displacement = np.linalg.norm(uv[current_indices] - old["uv"][a], axis=1)
                    image_motion = self.image_displacement(old, old["uv"][a], instance)
                    if image_motion is not None:
                        displacement = image_motion
                    moving = bool(np.median(displacement) > 0.6 or np.percentile(displacement, 90) > 1.5)
                    good = measured[current_indices]
                    if good.sum() >= 12:
                        dz = raw_z[current_indices[good]] - old["z"][a[good]]
                        median = float(np.median(dz))
                        mad = float(np.median(abs(dz - median)))
                        moving |= abs(median) > 0.012
                        candidate = bool(
                            np.median(displacement) < 0.35
                            and np.percentile(displacement, 90) < 1.2
                            and abs(median) < 0.008
                            and mad < 0.008
                        )
                        motion = dict(uv_median_px=float(np.median(displacement)), delta_z_m=median, mad_m=mad)
            reference = self.reference.get(instance)
            if mode == "static" and reference is not None:
                _, a, b = np.intersect1d(reference["ids"], ids[which], return_indices=True)
                if len(a) >= 12:
                    drift = np.linalg.norm(uv[which[b]] - reference["uv"][a], axis=1)
                    image_motion = self.image_displacement(reference, reference["uv"][a], instance)
                    if image_motion is not None:
                        drift = image_motion
                    if np.median(drift) > 1.0 or np.percentile(drift, 90) > 2.5:
                        moving = True
                    good = measured[which[b]]
                    if good.sum() >= 12 and abs(np.median(raw_z[which[b[good]]] - reference["z"][a[good]])) > 0.012:
                        moving = True
                elif sensor_update:
                    moving = True
            if moving:
                mode = "moving"
                self.runs[instance] = 0
                self.reference.pop(instance, None)
                self.cells = {k: v for k, v in self.cells.items() if k[0] != instance}
            elif sensor_update:
                self.runs[instance] = self.runs.get(instance, 0) + 1 if candidate else 0
                if self.runs[instance] >= 3:
                    mode = "static"
                    if instance not in self.reference:
                        self.reference[instance] = dict(
                            ids=ids[current].copy(),
                            uv=uv[current].copy(),
                            z=raw_z[current].copy(),
                            gray=getattr(self, "gray", None),
                        )
                elif mode != "static":
                    mode = "unknown"
                elif not candidate:
                    mode = "unknown"
                    self.reference.pop(instance, None)
            self.modes[instance] = mode
            # One update per 2x2 UV cell per actual capture, not once per point.
            if sensor_update and (candidate or mode == "static"):
                if len(current):
                    cell = np.floor(uv[current] / 2).astype(int)
                    keys = cell[:, 1] * 320 + cell[:, 0]
                    for key in np.unique(keys):
                        select = current[keys == key]
                        cache_key = (instance, int(key))
                        if np.ptp(raw_z[select]) > 0.015:
                            self.cells.pop(cache_key, None)
                            continue
                        value = float(np.median(raw_z[select]))
                        prior = self.cells.get(cache_key)
                        if mode == "static" and prior and t - prior[1] <= 0.2 and abs(value - 1 / prior[0]) >= 0.015:
                            continue
                        if prior and t - prior[1] <= 0.2 and abs(value - 1 / prior[0]) < 0.015:
                            alpha = float(np.clip(1 - np.exp(-(t - prior[1]) / 0.35), 0.08, 0.4))
                            inverse = (1 - alpha) * prior[0] + alpha / value
                        else:
                            inverse = 1 / value
                        self.cells[cache_key] = (inverse, t)
            if mode == "static":
                for i in which:
                    if blocked[i]:
                        continue
                    cell = np.floor(uv[i] / 2).astype(int)
                    entry = self.cells.get((instance, int(cell[1] * 320 + cell[0])))
                    if entry and t - entry[1] <= 0.2:
                        value = 1 / entry[0]
                        if p[i] == 1 and abs(float(raw_z[i]) - value) > 0.015:
                            p[i] = 0
                            out[i] = np.nan
                        else:
                            out[i] = value
                            if not p[i]:
                                p[i] = 4  # known surface estimate, not a measurement
                    elif not sensor_update and p[i]:
                        # No reliable surface at this location: never extrapolate noise.
                        p[i] = 0
                        out[i] = np.nan
            if sensor_update:
                current = current[p[current] == 1]
                self.previous[instance] = dict(
                    ids=ids[current].copy(),
                    uv=uv[current].copy(),
                    z=raw_z[current].copy(),
                    time=t,
                    gray=getattr(self, "gray", None),
                )
            details[instance] = dict(mode=mode, static_run=self.runs.get(instance, 0), **motion)
        p[blocked] = 0
        out[blocked] = np.nan
        good = np.isfinite(out) & (out >= 0.15) & (out <= 2.5)
        p[~good] = 0
        out[p == 0] = np.nan
        self.cells = {k: v for k, v in self.cells.items() if t - v[1] <= 0.5}
        return (
            out,
            p,
            dict(
                surface_modes=details,
                surface_estimates=int((p == 4).sum()),
                projection_occlusion_removed=self.projection_removed,
                boundary_pixels_removed=self.boundary_removed,
                boundary_points_blocked=int(blocked.sum()),
            ),
        )
