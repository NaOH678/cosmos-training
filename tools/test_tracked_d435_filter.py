import unittest
from pathlib import Path

import numpy as np
from render_d435_point_accuracy import load_module

CORE = Path(__file__).resolve().parents[1] / "cosmos_framework/inference/robot_policy"
Guard = load_module("guard_test_v2", CORE / "hand_depth_guard.py").HandDepthGuard
Filter = load_module("tracked_test_v2", CORE / "tracked_d435_filter.py").TrackedD435Filter


class Tests(unittest.TestCase):
    def setUp(self):
        self.filter = Filter(Guard)
        self.ids = np.arange(50)
        self.instances = np.ones(50, int)

    def step(self, z, t, update=True, ids=None, valid=None):
        return self.filter.update(
            self.ids if ids is None else ids,
            self.instances,
            np.full(50, z),
            np.ones(50, bool) if valid is None else valid,
            t,
            update,
        )

    def test_whole_hand_jump_predicts_and_expires(self):
        self.step(1.0, 0)
        a, p, _ = self.step(1.3, 0.067)
        self.assertTrue((p == 2).all())
        np.testing.assert_allclose(a, 1.0)
        self.step(1.3, 0.134)
        a, p, _ = self.step(1.3, 0.201)
        self.assertTrue((p == 0).all())
        self.assertTrue(np.isnan(a).all())
        self.assertEqual(len(self.filter.guards[1].history), 1)

    def test_intermediate_rgb_does_not_ingest_measurement(self):
        self.step(1.0, 0)
        a, p, _ = self.step(1.8, 0.033, False)
        self.assertTrue((p == 3).all())
        np.testing.assert_allclose(a, 1.0)
        self.assertEqual(len(self.filter.guards[1].history), 1)

    def test_new_ids_do_not_get_invented_depth(self):
        self.step(1.0, 0)
        a, p, _ = self.step(float("nan"), 0.033, False, ids=self.ids + 100)
        self.assertTrue((p == 0).all())

    def test_smooth_small_jitter_and_keep_coherent_motion(self):
        self.step(1.0, 0)
        a, p, _ = self.step(1.01, 0.067)
        np.testing.assert_allclose(a, 1.007, atol=1e-6)
        a, p, _ = self.step(1.04, 0.134)
        np.testing.assert_allclose(a, 1.04, atol=1e-6)
        self.assertTrue((p == 1).all())

    def test_prefix_unchanged_by_future(self):
        first = self.step(1.0, 0)
        second = self.step(1.01, 0.067)
        self.step(1.8, 0.134)
        self.filter = Filter(Guard)
        prefix_first = self.step(1.0, 0)
        prefix_second = self.step(1.01, 0.067)
        for full, prefix in [(first, prefix_first), (second, prefix_second)]:
            np.testing.assert_array_equal(full[0], prefix[0])
            np.testing.assert_array_equal(full[1], prefix[1])

    def test_patch_fly_repaired_from_raw_history(self):
        self.step(1.0, 0)
        z = np.ones(50)
        z[:5] = 1.15
        a, p, _ = self.filter.update(self.ids, self.instances, z, np.ones(50, bool), 0.067, True)
        self.assertTrue((p[:5] == 2).all())
        np.testing.assert_allclose(a, 1.0)
        self.assertEqual(len(self.filter.samples[0]), 1)


if __name__ == "__main__":
    unittest.main()
