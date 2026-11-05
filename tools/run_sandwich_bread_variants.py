"""Single-GPU smoke/latency tests, then full eight-worker continuous object tracking."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("pointflow_outputs/sandwich_bread_variants_20261030_v2"))
    p.add_argument("--variants", nargs="+", default=["group", "separate"])
    p.add_argument("--compare-all", action="store_true", help="Also measure all-object baseline on the same GPU")
    args = p.parse_args()
    root = args.root.resolve()
    env = os.environ.copy()
    env.update(
        LD_LIBRARY_PATH="",
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pylibs_sam2",
    )
    base = [sys.executable, "-u", "tools/segment_sandwich_video.py"]
    # Finish all single-card sequential tests before full multi-GPU processing.
    for variant in args.variants:
        out = root / variant
        if args.compare_all:
            with (out / "single_gpu_all_test.log").open("w") as log:
                subprocess.run(
                    base + ["track", "--out", str(out), "--benchmark", "--benchmark-all"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("All-object baseline passed:", variant, flush=True)
        with (out / "single_gpu_test.log").open("w") as log:
            subprocess.run(
                base + ["track", "--out", str(out), "--benchmark"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("Single-GPU test passed:", variant, flush=True)
    for variant in args.variants:
        out = root / variant
        children = []
        for worker in range(8):
            gpu_env = env.copy()
            gpu_env["CUDA_VISIBLE_DEVICES"] = str(worker)
            log = (out / f"gpu{worker}.log").open("w")
            proc = subprocess.Popen(
                base + ["track", "--out", str(out), "--worker", str(worker)],
                env=gpu_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append((proc, log))
        while any(proc.poll() is None for proc, _ in children):
            failed = [proc for proc, _ in children if proc.poll() not in (None, 0)]
            if failed:
                for proc, _ in children:
                    if proc.poll() is None:
                        proc.terminate()
                raise RuntimeError(f"{variant}: worker failed; inspect logs")
            print(variant, "workers completed", sum(proc.poll() is not None for proc, _ in children), "/8", flush=True)
            time.sleep(15)
        for proc, log in children:
            log.close()
            if proc.returncode:
                raise RuntimeError(f"{variant}: worker failed")
        subprocess.run(base + ["merge", "--out", str(out)], env=env, check=True)
        print("COMPLETE", variant, flush=True)
    print("ALL VIDEOS COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
