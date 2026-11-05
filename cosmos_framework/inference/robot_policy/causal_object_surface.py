"""Causal per-track surface fusion and continuous, edge-aware depth sampling."""

import numpy as np
from scipy.spatial import cKDTree


def sample_continuous(uv, instances, sensor, mask, radius=2.5):
    """Same-instance inverse-distance sampling; never blend depth discontinuities."""
    result = np.full(len(uv), np.nan, np.float32)
    for instance in np.unique(instances):
        if not instance:
            continue
        yy, xx = np.nonzero(np.isfinite(sensor) & (mask == instance))
        if not len(xx):
            continue
        target = np.flatnonzero(instances == instance)
        count = min(4, len(xx))
        distance, index = cKDTree(np.column_stack((xx, yy))).query(uv[target], k=count, workers=1)
        distance, index = distance.reshape(len(target), count), index.reshape(len(target), count)
        values = sensor[yy[index], xx[index]]
        # Select the nearest depth layer before interpolating in image space.
        same_layer = abs(values - values[:, :1]) <= 0.012
        usable = same_layer & (distance <= radius)
        weights = np.where(usable, 1 / np.maximum(distance, 0.05) ** 2, 0)
        weight_sum = weights.sum(1)
        good = (distance[:, 0] <= radius) & (weight_sum > 0)
        output = (values * weights).sum(1) / np.maximum(weight_sum, 1e-8)
        result[target[good]] = output[good]
    return result


class CausalObjectSurfaceState:
    """Motion evidence and accepted measurements are separate from fused output."""

    def __init__(self):
        self.previous = {}
        self.reference = {}
        self.states = {}
        self.runs = {}
        self.modes = {}
        self.motion_runs = {}

    def update(
        self,
        ids,
        uv,
        instances,
        labels,
        raw_z,
        z,
        provenance,
        timestamp,
        sensor_update,
        blocked,
        motion_estimator=None,
        gray=None,
    ):
        t = float(timestamp)
        out, p = z.copy(), provenance.copy()
        measured = (p == 1) & np.isfinite(raw_z) & ~blocked
        details = {}
        for i in np.flatnonzero(blocked):
            self.states.pop(int(ids[i]), None)
        for instance in np.unique(instances[labels == 3]):
            instance = int(instance)
            which = np.flatnonzero((instances == instance) & (labels == 3))
            current = which[measured[which]]
            old = self.previous.get(instance)
            mode = self.modes.get(instance, "unknown")
            motion = {}
            if sensor_update:
                low, moving = False, False
                if old is not None and t - old["time"] <= 0.3:
                    _, a, b = np.intersect1d(old["ids"], ids[which], return_indices=True)
                    if len(a) >= 12:
                        locations = which[b]
                        displacement = np.linalg.norm(uv[locations] - old["uv"][a], axis=1)
                        flow = motion_estimator(old, old["uv"][a], instance) if motion_estimator else None
                        if flow is not None:
                            displacement = flow
                        med, p90 = float(np.median(displacement)), float(np.percentile(displacement, 90))
                        good = measured[locations]
                        dz = float(np.median(raw_z[locations[good]] - old["z"][a[good]])) if good.sum() >= 12 else 0.0
                        # Depth noise no longer prevents static fusion; only a large coherent Z change exits it.
                        low = med < 0.8 and p90 < 2.5 and abs(dz) < 0.03
                        moving = med > 1.2 or p90 > 3.0 or abs(dz) > 0.03
                        motion = dict(uv_median_px=med, uv_p90_px=p90, delta_z_m=dz, image_motion_used=flow is not None)
                reference = self.reference.get(instance)
                if mode == "static" and reference is not None:
                    _, a, b = np.intersect1d(reference["ids"], ids[which], return_indices=True)
                    if len(a) >= 12:
                        flow = motion_estimator(reference, reference["uv"][a], instance) if motion_estimator else None
                        drift = flow if flow is not None else np.linalg.norm(uv[which[b]] - reference["uv"][a], axis=1)
                        moving |= bool(np.median(drift) > 2.5 or np.percentile(drift, 90) > 5.0)
                self.motion_runs[instance] = self.motion_runs.get(instance, 0) + 1 if moving else 0
                if self.motion_runs[instance] >= 2:
                    mode = "moving"
                    self.runs[instance] = 0
                    self.reference.pop(instance, None)
                elif low:
                    self.runs[instance] = self.runs.get(instance, 0) + 1
                    if self.runs[instance] >= 3:
                        mode = "static"
                        if instance not in self.reference:
                            self.reference[instance] = dict(
                                ids=ids[current].copy(), uv=uv[current].copy(), z=raw_z[current].copy(), gray=gray
                            )
                elif mode != "static":
                    self.runs[instance] = 0
                    mode = "unknown" if not moving else "moving"
                self.modes[instance] = mode
            for i in which:
                key = int(ids[i])
                prior = self.states.get(key)
                if prior and (prior["instance"] != instance or t - prior["time"] > 0.2):
                    prior = None
                if measured[i] and sensor_update:
                    value = float(raw_z[i])
                    if mode == "static" and prior:
                        difference = value - prior["z"]
                        # Unexplained static depth discontinuity is invalid, not new surface state.
                        if abs(difference) > 0.03:
                            out[i] = np.nan
                            p[i] = 0
                            continue
                        alpha = float(np.clip(1 - np.exp(-(t - prior["time"]) / 0.45), 0.06, 0.25))
                        value = prior["z"] if abs(difference) < 0.0015 else prior["z"] + alpha * difference
                    out[i] = value
                    self.states[key] = dict(z=value, time=t, instance=instance)
                elif p[i] and prior:
                    # Objects use bounded Z hold on intermediate frames, with no velocity extrapolation.
                    if t - prior["time"] <= 0.10:
                        out[i] = prior["z"]
                    else:
                        out[i] = np.nan
                        p[i] = 0
            if sensor_update:
                current = current[p[current] == 1]
                self.previous[instance] = dict(
                    ids=ids[current].copy(), uv=uv[current].copy(), z=raw_z[current].copy(), time=t, gray=gray
                )
            details[instance] = dict(mode=mode, static_run=self.runs.get(instance, 0), **motion)
        p[blocked] = 0
        out[blocked] = np.nan
        self.states = {k: v for k, v in self.states.items() if t - v["time"] <= 0.3}
        return (
            out,
            p,
            dict(
                surface_modes=details, surface_estimates=int((p == 4).sum()), boundary_points_blocked=int(blocked.sum())
            ),
        )
