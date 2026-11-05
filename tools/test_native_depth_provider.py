import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import lmdb
import numpy as np
import run_dropper_d435_v2 as pipeline
from run_fs_d435_v3_parity import ROOT, Fusion


class NativeDepthProvider(unittest.TestCase):
    def test_sdk_provider_preserves_original_processing(self):
        previous = pipeline.TASK, pipeline.V0, pipeline.OUT
        source = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
        episode = Path(json.loads((source / "source.json").read_text())["episode"])
        env = lmdb.open(str(episode / "auxiliary_camera/depth.lmdb"), readonly=True, lock=False)

        def provider(frame):
            with env.begin() as txn:
                blob = txn.get(f"depth/head/{frame:06d}".encode())
            return cv2.imdecode(np.frombuffer(blob, np.uint8), -1) if blob else None

        try:
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                pipeline.TASK = "scissors"
                pipeline.V0 = source
                pipeline.OUT = Path(temp) / "original"
                pipeline.build(enhancement=Fusion(), limit=8)
                pipeline.OUT = Path(temp) / "provider"
                pipeline.build(enhancement=Fusion(), limit=8, native_provider=provider)
                for f in sorted((Path(temp) / "original/frames").glob("*.npz")):
                    with np.load(f) as a, np.load(Path(temp) / "provider/frames" / f.name) as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{f.name}:{key}")
        finally:
            env.close()
            pipeline.TASK, pipeline.V0, pipeline.OUT = previous


if __name__ == "__main__":
    unittest.main()
