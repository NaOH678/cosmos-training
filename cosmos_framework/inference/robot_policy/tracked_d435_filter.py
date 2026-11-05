"""Causal D435 depth stabilization at existing RGB track IDs; no depth model."""

from collections import deque

import numpy as np


class TrackedD435Filter:
    """Measured history is independent of repaired/smoothed output.

    Call for every RGB frame. sensor_update distinguishes actual D435 captures
    from intermediate RGB frames. ID identity must also agree with the instance.
    """

    def __init__(self, guard_factory):
        self.guard_factory = guard_factory
        self.guards = {}
        self.samples = {}
        self.anchors = {}
        self.observation = 0
        self.last_time = None

    def update(self, ids, instances, z, valid, timestamp, sensor_update, cores=None):
        timestamp = float(timestamp)
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("RGB timestamps must increase")
        self.last_time = timestamp
        ids, instances = np.asarray(ids), np.asarray(instances)
        z = np.asarray(z, np.float32)
        accepted = np.asarray(valid, bool).copy() & np.isfinite(z) & (z >= 0.15) & (z <= 2.5)
        if not sensor_update:
            accepted[:] = False
        output = np.full(len(ids), np.nan, np.float32)
        provenance = np.zeros(len(ids), np.uint8)  # 1 measured, 2 short repair, 3 RGB interframe
        details = {}
        if sensor_update:
            self.observation += 1
            for instance in np.unique(instances):
                if not instance:
                    continue
                mask = instances == instance
                observed = (cores or {}).get(int(instance))
                if int(instance) not in (cores or {}) and accepted[mask].sum() >= 30:
                    observed = float(np.median(z[mask & accepted]))
                guard = self.guards.setdefault(int(instance), self.guard_factory())
                result = guard.update(timestamp, observed, observed is not None)
                details[int(instance)] = result
                if result["status"] != "measured":
                    accepted[mask] = False
                if result["status"] == "measured":
                    self.anchors[int(instance)] = (timestamp, float(result["z"]), result["velocity_mps"])
        # For each actual measurement, compensate modest instance motion before
        # detecting local jumps. This catches patches while preserving coherent motion.
        for instance in np.unique(instances):
            if not instance:
                continue
            result = details.get(int(instance))
            anchor = self.anchors.get(int(instance))
            which = np.flatnonzero(instances == instance)
            matches = []
            for i in which:
                history = self.samples.get(int(ids[i]))
                if history and history[-1][3] == int(instance) and timestamp - history[-1][0] <= 0.3:
                    matches.append((i, history))
            if sensor_update and result and result["status"] == "measured":
                delta = [float(z[i]) - hist[-1][1] for i, hist in matches if accepted[i]]
                motion = float(np.median(delta)) if len(delta) >= 12 else 0.0
                mad = float(np.median(abs(np.asarray(delta) - motion))) if delta else 0.0
                tolerance = min(0.045, max(0.020, 5 * 1.4826 * mad))
                for i, hist in matches:
                    if accepted[i] and abs(float(z[i]) - hist[-1][1] - motion) > tolerance:
                        accepted[i] = False
            for i in which:
                history = self.samples.get(int(ids[i]))
                previous = history[-1] if history and history[-1][3] == int(instance) else None
                if accepted[i]:
                    value = float(z[i])
                    output[i] = value
                    # Small two-measurement jitter only; no accumulated EMA lag.
                    if previous and timestamp - previous[0] <= 0.15 and abs(value - previous[1]) < 0.015:
                        output[i] = 0.7 * value + 0.3 * previous[1]
                    provenance[i] = 1
                    if history is None or (history and history[-1][3] != int(instance)):
                        history = self.samples[int(ids[i])] = deque(maxlen=4)
                    history.append((timestamp, value, self.observation, int(instance)))
                elif previous and anchor:
                    age = timestamp - previous[0]
                    observations_old = self.observation - previous[2]
                    if age > 0.2 or observations_old > 2:
                        continue
                    # A whole-hand failure is repaired only while its scalar guard
                    # permits prediction. Objects use the same bounded protection.
                    if sensor_update and result and result["status"] == "unavailable":
                        continue
                    velocity = float(anchor[2])
                    if len(history) >= 2:
                        velocities = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(history, list(history)[1:])]
                        velocity = float(np.clip(np.median(velocities), -0.6, 0.6))
                    output[i] = previous[1] + velocity * min(age, 0.2)
                    provenance[i] = 2 if sensor_update else 3
        stale = [key for key, hist in self.samples.items() if timestamp - hist[-1][0] > 0.5]
        for key in stale:
            del self.samples[key]
        return (
            output,
            provenance,
            dict(
                measured=int((provenance == 1).sum()),
                repaired=int((provenance == 2).sum()),
                interframe=int((provenance == 3).sum()),
                invalid=int((provenance == 0).sum()),
                guards=details,
            ),
        )
