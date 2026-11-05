import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from run_foundationstereo_d435_filters import BoundaryFusion, process, strengthen_static


class ContactBoundaryFilter(unittest.TestCase):
    def test_contact_depth_jump_blocks_both_sides(self):
        labels = np.full((32, 32), 3, np.uint8)
        labels[:, 16:] = 2
        instances = np.ones_like(labels)
        instances[:, 16:] = 2
        z = np.ones(labels.shape, np.float32)
        z[:, 16:] = 0.7
        fusion = BoundaryFusion()
        fusion.prepare_frame(labels, instances)
        result = fusion.gate_buffer(z)
        self.assertFalse(np.isfinite(result[15, 15]))
        self.assertFalse(np.isfinite(result[15, 16]))
        self.assertTrue(np.isfinite(result[15, 10]))
        np.testing.assert_array_equal(
            fusion.blocked(np.array([[15, 15], [16, 15], [10, 15]]), np.array([3, 2, 3])), [True, True, False]
        )

    def test_thin_mask_without_depth_break_is_not_eroded(self):
        labels = np.zeros((32, 32), np.uint8)
        labels[:, 15] = 3
        instances = (labels > 0).astype(np.uint8)
        fusion = BoundaryFusion()
        fusion.prepare_frame(labels, instances)
        result = fusion.gate_buffer(np.ones(labels.shape, np.float32))
        self.assertTrue(np.isfinite(result[:, 15]).all())

    def test_prefix_outputs_do_not_depend_on_future_frames(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            short, long = Path(temp) / "short", Path(temp) / "long"
            process(limit=4, destination=short)
            process(limit=6, destination=long)
            strengthen_static(destination=short)
            strengthen_static(destination=long)
            for path in sorted((short / "frames").glob("*.npz")):
                with np.load(path) as a, np.load(long / "frames" / path.name) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                    self.assertFalse((a["hard_blocked"] & a["valid"]).any())


if __name__ == "__main__":
    unittest.main()
