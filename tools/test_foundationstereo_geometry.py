import unittest

import numpy as np
from run_scissors_foundationstereo import aligned_samples, prepare


class StereoGeometry(unittest.TestCase):
    def test_saved_calibration_and_pairing(self):
        _, calibration, pairs, rgb = prepare()
        self.assertAlmostEqual(calibration["baseline_m"], 0.05009937582848547, places=9)
        self.assertEqual(len(pairs), 493)
        self.assertEqual(len(set(p["sequence"] for p in pairs)), 493)
        self.assertLess(calibration["lr_time_max_ms"], 1.0)
        for p in pairs:
            self.assertLessEqual(p["capture_time"], rgb[p["available_rgb_step"]])
            self.assertGreaterEqual(p["available_rgb_step"], p["saved_rgb_step"])

    def test_identity_projection_preserves_metric_depth(self):
        k = dict(fx=500.0, fy=500.0, ppx=320.0, ppy=240.0)
        streams = {
            "infrared_left": {
                "intrinsics": k,
                "extrinsics_to_color": {"rotation": np.eye(3).ravel().tolist(), "translation": [0.0, 0.0, 0.0]},
            },
            "color": {"intrinsics": k},
        }
        uv = np.array([[100.0, 100.0], [300.0, 200.0], [500.0, 350.0]], np.float32)
        z = aligned_samples(
            np.ones((480, 640), np.float32), streams, uv, np.array([3, 3, 3]), np.ones((480, 640), np.uint8) * 3
        )
        np.testing.assert_allclose(z, 1.0)
        z = aligned_samples(
            np.ones((480, 640), np.float32), streams, uv, np.array([4, 4, 4]), np.ones((480, 640), np.uint8) * 3
        )
        self.assertTrue(np.isnan(z).all())


if __name__ == "__main__":
    unittest.main()
