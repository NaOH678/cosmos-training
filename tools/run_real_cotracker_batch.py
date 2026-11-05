"""Resume five trials; one GPU point worker and one CPU verification/render worker."""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pointflow_outputs/cotracker_real_20261104"


def run(task, stage):
    marker = {
        "points": "pointflow/COMPLETE_STAGE.json",
        "verify": "verification.json",
        "render": "render_complete.json",
    }[stage]
    if (OUT / task / marker).exists():
        return
    path = OUT / task / (stage + ".log")
    with path.open("a") as log:
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "tools/run_real_cotracker_pipeline.py"), stage, "--task", task],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if r.returncode:
        raise RuntimeError(f"{task} {stage} failed; see {path}")
    print("COMPLETE", task, stage, flush=True)


def cpu(task):
    run(task, "verify")
    run(task, "render")


with ThreadPoolExecutor(max_workers=1) as pool:
    futures = []
    for task in ["bottle", "scissors", "dropper", "micro", "sandwich"]:
        while not (OUT / task / "segmentation_complete.json").exists():
            time.sleep(2)
        run(task, "points")
        futures.append(pool.submit(cpu, task))
    for f in futures:
        f.result()
print("ALL FIVE COMPLETE", flush=True)
