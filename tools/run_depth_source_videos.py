"""Render each completed task in four fixed views, at most four CPU workers."""

import os
import subprocess
import sys
import time
from pathlib import Path

root = Path("pointflow_outputs/depth_source_comparison_20261103")
jobs = {}
todo = [(task, view) for task in ["sandwich", "dropper", "micro"] for view in ["front", "side", "opposite", "overhead"]]
while todo or jobs:
    for key, (process, log) in list(jobs.items()):
        code = process.poll()
        if code is None:
            continue
        log.close()
        if code:
            raise RuntimeError(f"Rendering failed {key}; see log")
        print("COMPLETE", key, flush=True)
        del jobs[key]
    for task, view in list(todo):
        if len(jobs) >= 4:
            break
        if not (root / task / "summary.json").exists():
            continue
        key = task + "_" + view
        log = (root / (key + "_render.log")).open("w")
        env = os.environ.copy()
        env.update(
            LD_LIBRARY_PATH="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/depth_video_" + key
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "tools/render_depth_sources.py", "--task", task, "--view", view],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        jobs[key] = (process, log)
        todo.remove((task, view))
        print("START", key, flush=True)
    time.sleep(5)
print("ALL TWELVE VIDEOS COMPLETE", flush=True)
