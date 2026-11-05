"""Offline legacy filtering and projection invariants, without model inference."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cotracker_offline_da3 import temporal_filter


class OfflineFilters(unittest.TestCase):
    def test_gap_readmits_existing_prediction_without_moving_it(self):
        points = np.array([[0, 0, 1], [0.01, 0, 1], [0.02, 0, 1]], np.float32)
        before = points.copy()
        valid, info = temporal_filter(
            np.zeros(3, int), np.arange(3), points, np.array([1, 0, 1], bool), np.ones(3, bool)
        )
        self.assertTrue(valid.all())
        self.assertEqual(info["readmitted"], 1)
        np.testing.assert_array_equal(points, before)

    def test_out_and_back_spike_is_removed(self):
        points = np.array([[0, 0, 1], [0, 0, 1.2], [0, 0, 1.001]], np.float32)
        valid, info = temporal_filter(np.zeros(3, int), np.arange(3), points, np.ones(3, bool), np.ones(3, bool))
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(info["despiked"], 1)

    def test_different_ids_do_not_fill_each_other(self):
        valid, info = temporal_filter(
            np.array([0, 1, 2]), np.arange(3), np.zeros((3, 3), np.float32), np.array([1, 0, 1], bool), np.ones(3, bool)
        )
        self.assertFalse(valid[1])
        self.assertEqual(info["readmitted"], 0)

    def test_long_gap_and_wrong_depth_remain_invalid(self):
        xyz = np.tile([0, 0, 1.0], (7, 1))
        v = np.array([1, 0, 0, 0, 0, 0, 1], bool)
        out, _ = temporal_filter(np.zeros(7, int), np.arange(7), xyz, v, np.ones(7, bool))
        np.testing.assert_array_equal(out, v)
        xyz = np.array([[0, 0, 1], [0, 0, 1.3], [0, 0, 1]], float)
        out, _ = temporal_filter(np.zeros(3, int), np.arange(3), xyz, np.array([1, 0, 1], bool), np.ones(3, bool))
        self.assertFalse(out[1])

    def test_legacy_projection_matches_original_depth_to_points(self):
        import torch
        import utils3d

        width, height, focal = 640, 448, 530.0
        z = np.full((height, width), 0.8, np.float32)
        K = np.array([[focal / width, 0, 0.5], [0, focal / height, 0.5], [0, 0, 1]], np.float32)
        expected = utils3d.torch.depth_to_points(
            torch.from_numpy(z), intrinsics=torch.from_numpy(K), use_ray=False
        ).numpy()
        uv = np.array([[0, 0], [319, 223], [600, 400]], np.float32)
        actual = np.c_[
            (uv[:, 0] + 0.5 - width / 2) / focal * 0.8, (uv[:, 1] + 0.5 - height / 2) / focal * 0.8, np.ones(3) * 0.8
        ]
        np.testing.assert_allclose(actual, expected[uv[:, 1].astype(int), uv[:, 0].astype(int)], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
