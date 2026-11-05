"""Synthetic motion/depth faults and streaming-prefix invariance checks."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "cosmos_framework/inference/robot_policy/causal_depth_filter.py"
spec = importlib.util.spec_from_file_location("causal_depth", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ZeroFlow:
    def calc(self, first, second, initial):
        return np.zeros((*first.shape, 2), np.float32)


class CausalFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = module.CausalDepthFilter(dict(fx=100, fy=100, ppx=20, ppy=20))
        self.filter.flow = ZeroFlow()
        self.image = np.zeros((40, 40, 3), np.uint8)
        self.mask = np.ones((40, 40), np.uint8)
        y, x = np.indices((40, 40))
        self.uv = np.column_stack((x.ravel(), y.ravel()))
        self.points = np.column_stack(((x.ravel() - 20) / 100, (y.ravel() - 20) / 100, np.ones(1600)))
        self.ids = np.ones(1600, np.uint8)
        self.valid = np.ones(1600, bool)

    def update(self, z, t, valid=None):
        points = self.points * np.asarray(z).reshape(-1, 1)
        return self.filter.update(
            self.image,
            self.mask,
            points,
            self.uv,
            self.ids,
            self.valid if valid is None else valid,
            t,
        )

    def test_depth_patch_rejected_and_repair_expires(self):
        self.update(np.ones(1600), 0)
        z = np.ones((40, 40))
        z[10:20, 10:20] = 1.25
        result = self.update(z.ravel(), 1 / 6)
        self.assertEqual(result[-1]["temporal_rejected"], 100)
        self.assertEqual(len(result[1]), 100)
        self.assertTrue(np.allclose(result[1][:, 2], 1.0))
        valid = self.valid.copy().reshape(40, 40)
        valid[10:20, 10:20] = False
        result = self.update(np.ones(1600), 2 / 6, valid.ravel())
        self.assertEqual(len(result[1]), 0, "Repairs must never propagate as measured state")

    def test_coherent_axial_motion_preserved(self):
        self.update(np.ones(1600), 0)
        result = self.update(np.full(1600, 1.03), 1 / 6)
        self.assertEqual(result[-1]["temporal_rejected"], 0)

    def test_current_mask_blocks_old_surface(self):
        self.update(np.ones(1600), 0)
        self.mask[:, :] = 0
        result = self.update(np.ones(1600), 1 / 6, np.zeros(1600, bool))
        self.assertEqual(len(result[1]), 0)

    def test_long_gap_resets_matching(self):
        self.update(np.ones(1600), 0)
        result = self.update(np.full(1600, 1.25), 1.0)
        self.assertEqual(result[-1]["matched"], 0)
        self.assertEqual(len(result[1]), 0)

    def test_prefix_independent_of_future(self):
        first = self.update(np.ones(1600), 0)
        second = self.update(np.full(1600, 1.02), 1 / 6)
        self.update(np.full(1600, 1.50), 2 / 6)
        self.assertTrue(first[0].all())
        self.assertTrue(second[0].all())
        self.assertFalse(np.shares_memory(second[0], self.filter.previous["xyz"]))
        self.filter = module.CausalDepthFilter(dict(fx=100, fy=100, ppx=20, ppy=20))
        self.filter.flow = ZeroFlow()
        prefix_first = self.update(np.ones(1600), 0)
        prefix_second = self.update(np.full(1600, 1.02), 1 / 6)
        for full, prefix in [(first, prefix_first), (second, prefix_second)]:
            for index in range(4):
                np.testing.assert_array_equal(full[index], prefix[index])

    def test_large_coherent_depth_jump_rejected(self):
        self.update(np.ones(1600), 0)
        result = self.update(np.full(1600, 1.25), 1 / 6)
        self.assertGreater(result[-1]["temporal_rejected"], 1500)
        self.assertEqual(len(result[1]), 0, "Do not repair from untrusted object motion")

    def test_lateral_motion_uses_corresponding_pixel(self):
        class TranslationFlow:
            count = 0

            def calc(self, first, second, initial):
                flow = np.zeros((*first.shape, 2), np.float32)
                flow[..., 0] = 2 if self.count % 2 == 0 else -2
                self.count += 1
                return flow

        z = 1 + self.uv[:, 0] * 0.015
        self.update(z, 0)
        self.filter.flow = TranslationFlow()
        result = self.update(1 + np.maximum(self.uv[:, 0] - 2, 0) * 0.015, 1 / 6)
        self.assertEqual(result[-1]["temporal_rejected"], 0)


if __name__ == "__main__":
    unittest.main()
