"""Tiny causal gate on one hand's scalar core depth; no flow or point matching."""

from collections import deque

import numpy as np


class HandDepthGuard:
    """Compare current Z to a trend of accepted measurements; never learn from repairs."""

    def __init__(
        self,
        history=4,
        base_tolerance_m=0.04,
        tolerance_speed_mps=0.20,
        max_velocity_mps=0.60,
        prediction_seconds=0.5,
        prediction_frames=2,
    ):
        self.history = deque(maxlen=history)
        self.base = base_tolerance_m
        self.speed = tolerance_speed_mps
        self.max_velocity = max_velocity_mps
        self.horizon = prediction_seconds
        self.max_prediction_frames = prediction_frames
        self.rejected = 0
        self.last_time = None

    def update(self, timestamp, observed_z, valid=True):
        timestamp = float(timestamp)
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Hand depth observations must have increasing timestamps")
        self.last_time = timestamp
        valid = bool(valid and observed_z is not None and np.isfinite(observed_z) and 0.1 < float(observed_z) < 6)
        if not self.history:
            if valid:
                self.history.append((timestamp, float(observed_z)))
                return dict(status="measured", z=float(observed_z), velocity_mps=0.0, residual_m=0.0, threshold_m=None)
            return dict(status="unavailable", z=None, velocity_mps=0.0, residual_m=None, threshold_m=None)
        last_t, last_z = self.history[-1]
        age = timestamp - last_t
        velocities = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(self.history, list(self.history)[1:])]
        velocity = float(np.clip(np.median(velocities), -self.max_velocity, self.max_velocity)) if velocities else 0.0
        predicted = last_z + velocity * min(age, self.horizon)
        threshold = self.base + self.speed * min(age, 0.25)
        residual = abs(float(observed_z) - predicted) if valid else None
        if valid and residual <= threshold:
            self.history.append((timestamp, float(observed_z)))
            self.rejected = 0
            return dict(
                status="measured",
                z=float(observed_z),
                velocity_mps=velocity,
                residual_m=residual,
                threshold_m=threshold,
            )
        self.rejected += 1
        if age <= self.horizon and self.rejected <= self.max_prediction_frames:
            return dict(
                status="predicted", z=predicted, velocity_mps=velocity, residual_m=residual, threshold_m=threshold
            )
        return dict(status="unavailable", z=None, velocity_mps=velocity, residual_m=residual, threshold_m=threshold)
