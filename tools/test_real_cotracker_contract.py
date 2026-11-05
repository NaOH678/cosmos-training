"""Contract checks for causal real tracking, Z-only correction and ragged windows."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from render_d435_point_accuracy import load_module

core = load_module(
    "causal_rgb_pointflow", str(ROOT / "cosmos_framework/inference/robot_policy/causal_rgb_pointflow.py")
)
from cosmos_framework.data.pointflow_window import _episode_metadata
from cosmos_framework.data.ragged_pointflow import RaggedPointFlow


class Contracts(unittest.TestCase):
    def test_coherent_detached_depth_patch_is_not_accepted_as_a_hand_surface(self):
        rng = np.random.default_rng(7)
        main = np.c_[rng.random((120, 2)) * 0.03, np.ones(120)]
        detached = np.c_[rng.random((40, 2)) * 0.01, np.ones(40) * 0.78]
        points = np.r_[main, detached]
        kept, stats = core.CausalGeometryFilter().apply(
            np.arange(160), points, np.ones(160, bool), np.ones(160, int), 0.0
        )
        self.assertGreater(int(kept[:120].sum()), 110)
        self.assertFalse(kept[120:].any())
        self.assertEqual(stats["component_removed"], 40)

    def test_z_shift_keeps_xy_and_uv(self):
        rng = np.random.default_rng(0)
        xyz = rng.random((100, 3)).astype(np.float32)
        uv = rng.random((100, 2)).astype(np.float32)
        old = uv.copy()
        result = core.translate_z(xyz, 0.12)
        np.testing.assert_array_equal(result[:, :2], xyz[:, :2])
        np.testing.assert_array_equal(old, uv)
        np.testing.assert_allclose(result[:, 2], xyz[:, 2] + 0.12)

    def test_newly_visible_surfaces_and_lost_ids(self):
        tracks = core.CausalPointTracks(3)
        labels = np.zeros((20, 20), np.uint8)
        labels[:8, :8] = 2
        born = tracks.advance(labels)
        tracks.record_births(0, born)
        old = tracks.ids.copy()
        born = tracks.advance(labels, tracks.uv, np.ones(len(tracks.ids), bool))
        tracks.record_births(1, born)
        np.testing.assert_array_equal(tracks.ids, old)
        labels[12:, 12:] = 3
        born = tracks.advance(labels, tracks.uv, np.zeros(len(tracks.ids), bool))
        tracks.record_births(2, born)
        self.assertFalse(np.intersect1d(old, tracks.ids).size)
        self.assertTrue(len(born) > 0)

    def test_temporal_filter_removes_depth_fly_and_keeps_motion(self):
        rng = np.random.default_rng(0)
        xyz = np.c_[rng.random((100, 2)) * 0.05, np.ones(100)].astype(np.float32)
        ids = np.arange(100)
        good = np.ones(100, bool)
        inst = np.ones(100, int)
        f = core.CausalGeometryFilter()
        f.apply(ids, xyz, good, inst, 0.0)
        moved = xyz + [0.001, 0, 0.003]
        moved[17, 2] += 0.15
        kept, stats = f.apply(ids, moved, good, inst, 1 / 30)
        self.assertFalse(kept[17])
        self.assertGreater(kept.sum(), 90)
        self.assertGreater(stats["temporal_removed"], 0)

    def test_ragged_alignment_and_missing_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            values = dict(
                frame_offsets=[0, 2, 4, 5],
                obs_track=[0, 1, 0, 2, 2],
                obs_pos=[[0, 0, 1], [1, 0, 1], [0, 0, 1.1], [2, 0, 1], [2, 0, 1.1]],
                obs_uv=[[1, 1], [2, 2], [1, 1], [3, 3], [3, 3]],
                obs_valid=[1, 1, 1, 1, 1],
                track_label=[2, 3, 3],
                frame_indices=[0, 1, 2],
                timestamps_sec=[0, 0.1, 0.2],
            )
            for key, value in values.items():
                np.save(path / (key + ".npy"), np.array(value))
            w = RaggedPointFlow(path).window(np.arange(3))
            self.assertEqual(w["point_ids"].tolist(), [0, 1])
            self.assertFalse(w["valid"][1, 1])
            self.assertFalse(w["valid"][2].any())
            self.assertEqual(w["position"][1, 0, 2], np.float32(1.1))
            (path / "report.json").write_text(
                json.dumps(
                    dict(
                        schema_version="rgbd_pointflow_v1",
                        coordinate="fixed_optical_camera",
                        metric_scale=True,
                        image_size_wh=[640, 448],
                        video="head.mp4",
                    )
                )
            )
            self.assertEqual(_episode_metadata(path), ("head.mp4", 640, 448))


if __name__ == "__main__":
    unittest.main()
