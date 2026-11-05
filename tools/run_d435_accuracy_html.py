"""Build three standalone D435 accuracy HTMLs using existing Viser assets."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("pointflow_outputs/d435_interactive_accuracy_20261102")


def main():
    ROOT.mkdir(exist_ok=False)
    jobs = []
    for i, task in enumerate(["sandwich", "dropper", "micro"]):
        env = os.environ.copy()
        env.update(LD_LIBRARY_PATH="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=str(i))
        log = (ROOT / (task + ".log")).open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "tools/build_d435_accuracy_html.py",
                "--task",
                task,
                "--out",
                str(ROOT),
                "--port",
                str(9130 + i),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((task, proc, log))
    while any(proc.poll() is None for _, proc, _ in jobs):
        for name, proc, _ in jobs:
            if proc.poll() not in (None, 0):
                for _, other, _ in jobs:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f"{name} failed; inspect its log")
        print("complete", [name for name, proc, _ in jobs if proc.poll() == 0], flush=True)
        time.sleep(10)
    summaries = {}
    for name, proc, log in jobs:
        log.close()
        if proc.returncode:
            raise RuntimeError(name)
        summaries[name] = json.loads((ROOT / name / "summary.json").read_text())
    (ROOT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    for file in ["build_d435_accuracy_html.py", "run_d435_accuracy_html.py", "render_d435_point_accuracy.py"]:
        shutil.copyfile(Path("tools") / file, ROOT / file)
    print("ALL HTML COMPLETE", ROOT, flush=True)


if __name__ == "__main__":
    main()
