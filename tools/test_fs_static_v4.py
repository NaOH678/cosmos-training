import unittest

import numpy as np
from run_fs_static_v4 import NoObjectExtrapolation, core, pipeline


class StaticSurface(unittest.TestCase):
    def test_continuous_sampling_reduces_pixel_switch(self):
        sensor = np.tile(0.8 + 0.004 * np.arange(8), (8, 1)).astype(np.float32)
        z = core.sample_continuous(np.array([[3.49, 3.0], [3.51, 3.0]]), np.array([3, 3]), sensor, np.full((8, 8), 3))
        self.assertLess(abs(float(z[1] - z[0])), 0.001)

    def test_interpolation_does_not_blend_depth_layers(self):
        sensor = np.full((8, 8), 0.7, np.float32)
        sensor[:, 4:] = 1.0
        z = core.sample_continuous(np.array([[3.49, 3.0], [3.51, 3.0]]), np.array([3, 3]), sensor, np.full((8, 8), 3))
        self.assertTrue(np.all(np.minimum(abs(z - 0.7), abs(z - 1.0)) < 1e-5))

    def sequence(self, number):
        state = core.CausalObjectSurfaceState()
        n = 30
        ids = np.arange(n)
        uv = np.column_stack((np.arange(n), np.zeros(n))).astype(float)
        outputs = []
        raw = []
        for f in range(number):
            points = uv + [0.55 * np.sin(f), 0]
            depth = 0.8 + 0.006 * np.sin(f * 1.3 + np.arange(n) * 0.9)
            out, p, stats = state.update(
                ids,
                points,
                np.full(n, 3),
                np.full(n, 3),
                depth,
                depth,
                np.ones(n, np.uint8),
                f / 15,
                True,
                np.zeros(n, bool),
            )
            outputs.append(out)
            raw.append(depth)
        return state, np.array(outputs), np.array(raw), uv

    def test_noise_does_not_prevent_static_mode(self):
        state, out, raw, _ = self.sequence(30)
        self.assertEqual(state.modes[3], "static")
        self.assertLess(np.mean(np.std(out[10:], axis=0)), np.mean(np.std(raw[10:], axis=0)) * 0.5)

    def test_real_motion_exits_static_mode(self):
        state, _, _, uv = self.sequence(10)
        n = len(uv)
        for f in range(10, 13):
            z = np.full(n, 0.8)
            state.update(
                np.arange(n),
                uv + [2 * (f - 9), 0],
                np.full(n, 3),
                np.full(n, 3),
                z,
                z,
                np.ones(n, np.uint8),
                f / 15,
                True,
                np.zeros(n, bool),
            )
        self.assertEqual(state.modes[3], "moving")

    def test_prefix_is_independent_of_future(self):
        _, a, _, _ = self.sequence(8)
        _, b, _, _ = self.sequence(20)
        np.testing.assert_array_equal(a, b[:8])

    def test_object_interframe_has_no_velocity_extrapolation(self):
        filt = NoObjectExtrapolation(pipeline.Guard)
        ids = np.arange(30)
        instances = np.full(30, 3)
        valid = np.ones(30, bool)
        filt.update(ids, instances, np.full(30, 0.8), valid, 0.0, True)
        filt.update(ids, instances, np.full(30, 0.806), valid, 1 / 15, True)
        z, p, _ = filt.update(ids, instances, np.full(30, np.nan), np.zeros(30, bool), 0.1, False)
        np.testing.assert_allclose(z, 0.806, atol=1e-6)
        self.assertTrue((p == 3).all())

    def test_scalar_object_guard_has_no_velocity_extrapolation(self):
        filt = NoObjectExtrapolation(pipeline.Guard)
        ids = np.arange(30)
        instances = np.full(30, 3)
        for i in range(3):
            _, _, stats = filt.update(ids, instances, np.full(30, 0.8 + 0.006 * i), np.ones(30, bool), i / 15, True)
        self.assertEqual(stats["guards"][3]["velocity_mps"], 0.0)

    def test_stale_object_guard_recovers_after_two_consistent_observations(self):
        filt = NoObjectExtrapolation(pipeline.Guard)
        ids = np.arange(30)
        instances = np.full(30, 3)
        valid = np.ones(30, bool)
        filt.update(ids, instances, np.full(30, 0.8), valid, 0.0, True)
        filt.update(ids, instances, np.full(30, 1.1), valid, 0.4, True)
        filt.update(ids, instances, np.full(30, np.nan), np.zeros(30, bool), 0.4 + 1 / 30, False)
        _, p, stats = filt.update(ids, instances, np.full(30, 1.101), valid, 0.4 + 1 / 15, True)
        self.assertEqual(stats["object_guards_reinitialized"], [3])
        self.assertTrue((p == 1).all())


if __name__ == "__main__":
    unittest.main()
