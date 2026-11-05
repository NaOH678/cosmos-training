import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from filter_scissors_pointworld_causal import noise_mask, run
from sklearn.cluster import DBSCAN


class PointWorldCausalFilter(unittest.TestCase):
    def test_spatial_support_matches_original_dbscan(self):
        rng = np.random.default_rng(0)
        cloud = np.concatenate(
            [rng.normal(0, 0.009, (150, 3)), rng.normal(0.2, 0.012, (100, 3)), rng.uniform(-1, 1, (20, 3))]
        )
        expected = np.zeros(len(cloud), bool)
        for eps in [0.02, 0.05]:
            expected |= DBSCAN(eps=eps, min_samples=5, n_jobs=1).fit_predict(cloud) == -1
        np.testing.assert_array_equal(noise_mask(cloud), expected)

    def test_earlier_outputs_independent_of_future_length(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            short, long = Path(tmp) / "short", Path(tmp) / "long"
            run(limit=8, destination=short)
            run(limit=12, destination=long)
            for f in sorted((short / "frames").glob("*.npz")):
                with np.load(f) as a, np.load(long / "frames" / f.name) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{f.name}:{key}")


if __name__ == "__main__":
    unittest.main()
