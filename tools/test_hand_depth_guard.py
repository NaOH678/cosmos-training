"""Synthetic causal scalar-depth regression checks."""

import unittest
from pathlib import Path

from render_d435_point_accuracy import load_module

module = load_module(
    "guard", str(Path(__file__).resolve().parents[1] / "cosmos_framework/inference/robot_policy/hand_depth_guard.py")
)


class Tests(unittest.TestCase):
    def test_velocity_prediction_and_no_repair_feedback(self):
        g = module.HandDepthGuard()
        self.assertEqual(g.update(0, 0.70)["status"], "measured")
        self.assertEqual(g.update(0.167, 0.71)["status"], "measured")
        r = g.update(0.334, 1.05)
        self.assertEqual(r["status"], "predicted")
        self.assertAlmostEqual(r["z"], 0.72, places=5)
        self.assertEqual(len(g.history), 2)
        self.assertEqual(g.update(0.501, 1.06)["status"], "predicted")
        self.assertEqual(g.update(0.668, 1.07)["status"], "unavailable")
        self.assertEqual(len(g.history), 2)
        self.assertEqual(g.update(0.835, 0.75)["status"], "measured")

    def test_missing_support_uses_short_history(self):
        g = module.HandDepthGuard()
        self.assertEqual(g.update(0, None, False)["status"], "unavailable")
        g.update(0.167, 0.7)
        self.assertEqual(g.update(0.334, None, False)["status"], "predicted")

    def test_normal_constant_motion(self):
        g = module.HandDepthGuard()
        for i in range(20):
            self.assertEqual(g.update(i / 6, 0.65 + 0.01 * i)["status"], "measured")

    def test_large_gap_no_prediction(self):
        g = module.HandDepthGuard()
        g.update(0, 0.7)
        self.assertEqual(g.update(1, 1.2)["status"], "unavailable")

    def test_time_must_increase(self):
        g = module.HandDepthGuard()
        g.update(0, 0.7)
        with self.assertRaises(ValueError):
            g.update(0, 0.8)


if __name__ == "__main__":
    unittest.main()
