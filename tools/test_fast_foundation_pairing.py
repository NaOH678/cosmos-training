import unittest

import numpy as np
from run_fast_foundation_v4 import match_stereo


class StereoPairing(unittest.TestCase):
    def stream(self, counters, times, steps):
        return dict(sequences=np.array(counters), timestamps=np.array(times), training_steps=np.array(steps))

    def test_counter_offset_uses_synchronous_images(self):
        left = self.stream([11, 12, 13], [0.0, 0.067, 0.134], [0, 2, 4])
        right = self.stream([10, 11, 12], [0.0001, 0.0671, 0.1341], [0, 2, 4])
        pairs = match_stereo(left, right, np.arange(8) / 30)
        self.assertEqual([(p["left_video_index"], p["right_video_index"]) for p in pairs], [(0, 0), (1, 1), (2, 2)])
        self.assertTrue(all(abs(p["left_time"] - p["right_time"]) < 0.001 for p in pairs))
        self.assertTrue(all(p["capture_time"] <= p["rgb_time"] for p in pairs))

    def test_mismatched_capture_dropped_and_right_frame_not_reused(self):
        left = self.stream([1, 2, 3], [0.0, 0.0002, 0.1], [0, 1, 3])
        right = self.stream([7, 8], [0.0001, 0.2], [0, 6])
        pairs = match_stereo(left, right, np.arange(10) / 30)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["right_video_index"], 0)


if __name__ == "__main__":
    unittest.main()
