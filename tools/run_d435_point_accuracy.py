"""Render three D435/URDF point-cloud videos concurrently on separate A800 GPUs."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("pointflow_outputs/d435_point_accuracy_20261102")


def main():
    ROOT.mkdir(exist_ok=False)
    children = []
    for task, gpu in [("sandwich", 0), ("dropper", 1), ("micro", 3)]:
        env = os.environ.copy()
        env.update(LD_LIBRARY_PATH="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=str(gpu))
        log = (ROOT / f"{task}.log").open("w")
        cmd = [
            sys.executable,
            "-u",
            "tools/render_d435_point_accuracy.py",
            "--task",
            task,
            "--out",
            str(ROOT),
            "--device",
            "cuda:0",
        ]
        children.append((task, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    while any(proc.poll() is None for _, proc, _ in children):
        for task, proc, _ in children:
            if proc.poll() not in (None, 0):
                for _, other, _ in children:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f"{task} failed; inspect {ROOT / (task + '.log')}")
        print("completed", [name for name, proc, _ in children if proc.poll() == 0], flush=True)
        time.sleep(10)
    result = {}
    for task, proc, log in children:
        log.close()
        if proc.returncode:
            raise RuntimeError(task)
        path = ROOT / task
        summary = json.loads((path / "summary.json").read_text())
        capture = cv2.VideoCapture(str(path / "multiview_h264.mp4"))
        count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape[:2] == (960, 1280)
            count += 1
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        assert count == summary["frames"] and abs(fps - summary["output_fps"]) < 1e-3
        rows = json.loads((path / "per_frame.json").read_text())
        for row in rows:
            assert row["filtered_hand"] <= row["raw_hand"] and row["filtered_objects"] <= row["raw_objects"]
        result[task] = dict(
            decoded_frames=count,
            fps=fps,
            physical_gpu=summary["physical_gpu"],
            projection_check=summary["projection_cpu_cuda_check"],
        )
        for mode in ["raw", "filtered"]:
            values = [row[mode + "_hand_reference_mm"]["p50"] for row in rows if row[mode + "_hand_reference_mm"]]
            result[task][mode + "_median_hand_to_reference_mm"] = float(np.median(values))
        (path / "verification.json").write_text(json.dumps(result[task], indent=2))
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2))
    for file in ["render_d435_point_accuracy.py", "run_d435_point_accuracy.py"]:
        shutil.copyfile(Path("tools") / file, ROOT / file)
    print("ALL COMPLETE", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
