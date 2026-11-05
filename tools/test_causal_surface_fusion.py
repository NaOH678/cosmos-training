import unittest
from pathlib import Path

import numpy as np
from render_d435_point_accuracy import load_module

Fusion = load_module(
    "surface_fusion_test",
    Path(__file__).resolve().parents[1] / "cosmos_framework/inference/robot_policy/causal_surface_fusion.py",
).CausalSurfaceFusion


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.filter = Fusion()
        x, y = np.meshgrid(np.arange(10, 30, 2), np.arange(10, 20, 2))
        self.uv = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        self.ids = np.arange(len(self.uv))
        self.inst = np.ones(len(self.ids), int) * 3
        self.labels = np.ones(len(self.ids), np.uint8) * 3

    def step(self, z, frame, sensor=True, base=None, uv=None):
        z = np.full(len(self.ids), z, np.float32)
        return self.filter.update(
            self.ids,
            self.uv if uv is None else uv,
            self.inst,
            self.labels,
            z,
            z.copy() if base is None else np.full(len(z), base, np.float32),
            np.ones(len(z), np.uint8) * (1 if sensor else 3),
            frame / 15,
            sensor,
            np.zeros(len(z), bool),
        )

    def test_stationary_fusion_reduces_jitter(self):
        values = []
        raw = []
        for f in range(45):
            value = 1 + (0.003 if f % 2 else -0.003)
            out, p, info = self.step(value, f)
            if f > 10:
                values.append(out[0])
                raw.append(value)
        self.assertEqual(info["surface_modes"][3]["mode"], "static")
        self.assertLess(np.std(values), np.std(raw) * 0.5)

    def test_motion_exits_static_without_long_lag(self):
        for f in range(8):
            self.step(1, f)
        out, p, info = self.step(1.03, 8, uv=self.uv + [2, 0])
        self.assertEqual(info["surface_modes"][3]["mode"], "moving")
        np.testing.assert_allclose(out, 1.03)

    def test_stationary_interframe_ignores_noisy_velocity(self):
        for f in range(8):
            self.step(1, f)
        history = self.filter.previous[3]["time"]
        cells = dict(self.filter.cells)
        out, p, info = self.step(float("nan"), 7.5, False, base=1.04)
        np.testing.assert_allclose(out, 1.0)
        self.assertEqual(history, self.filter.previous[3]["time"])
        self.assertEqual(cells, self.filter.cells)

    def test_blocked_points_cannot_be_repaired(self):
        for f in range(8):
            self.step(1, f)
        z = np.ones(len(self.ids), np.float32)
        blocked = np.ones(len(z), bool)
        out, p, info = self.filter.update(
            self.ids, self.uv, self.inst, self.labels, z, z, np.ones(len(z), np.uint8), 8 / 15, True, blocked
        )
        self.assertFalse(p.any())
        self.assertTrue(np.isnan(out).all())

    def test_prefix_does_not_depend_on_future(self):
        outputs = []
        for f in range(12):
            outputs.append(self.step(1 + 0.001 * (f % 3), f)[0].copy())
        self.step(1.3, 12)
        self.filter = Fusion()
        for f in range(12):
            np.testing.assert_array_equal(outputs[f], self.step(1 + 0.001 * (f % 3), f)[0])

    def test_hand_boundary_is_not_smoothed_across(self):
        labels = np.ones((40, 40), np.uint8) * 3
        labels[:, :20] = 2
        self.filter.prepare_frame(labels, labels)
        depth = np.ones((40, 40), np.float32)
        depth[:, :20] = 0.5
        result = self.filter.gate_buffer(depth)
        self.assertTrue(np.isinf(result[20, 20]))
        self.assertTrue(np.isfinite(result[20, 25]))

    def test_rgb_static_detection_tolerates_tracker_drift(self):
        import cv2

        rng = np.random.default_rng(3)
        gray = rng.integers(0, 255, (40, 40), dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        labels = np.ones((40, 40), np.uint8) * 3
        for f in range(14):
            self.filter.prepare_frame(labels, labels, image)
            out, p, info = self.step(1.0, f, uv=self.uv + [f * 0.12, 0])
        self.assertEqual(info["surface_modes"][3]["mode"], "static")
        moved = cv2.warpAffine(image, np.array([[1.0, 0, 2], [0, 1.0, 0]], np.float32), (40, 40))
        self.filter.prepare_frame(labels, labels, moved)
        out, p, info = self.step(1.03, 14, uv=self.uv + [3.6, 0])
        self.assertEqual(info["surface_modes"][3]["mode"], "moving")

    def test_static_surface_expires_without_real_measurement(self):
        for f in range(8):
            self.step(1.0, f)
        out, p, _ = self.step(float("nan"), 12, False, base=1.0)
        self.assertFalse(p.any())
        self.assertTrue(np.isnan(out).all())

    def test_native_occlusion_marks_far_reversed_projection(self):
        y, x = np.indices((2, 30))
        pix = np.stack((x, y), axis=-1)
        pix[:, 12:15, 0] -= 8
        points = np.zeros((2, 30, 3), np.float32)
        points[..., 2] = 0.5
        points[:, 12:15, 2] = 1.0
        streams = {"depth": {"extrinsics_to_color": {"translation": [0.015, 0, 0]}}}
        bad = self.filter.occlusion(
            points.reshape(-1, 3), pix.reshape(-1, 2), np.ones((2, 30), np.uint16), streams
        ).reshape(2, 30)
        self.assertTrue(bad[:, 12].all())
        self.assertFalse(bad[:, 0].any())


if __name__ == "__main__":
    unittest.main()
