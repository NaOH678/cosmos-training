"""Download Bench2Dex replay-generalization HDF5s through hf-mirror, verified against <task>.files.json (LFS SHA256).

One file list per task feeds both the VAE cache (A800) and the PointFlow/FK pipeline (4090), so both sides read the
same episodes byte for byte.  Resumable: a file whose size and SHA256 already match is skipped.

layout vae: <root>/<short>/dataset/<task>/replay-generalization/<episode>.hdf5   (tools/cache_bench2dex_multitask_latents.py)
layout pf : <root>/<task>/replay-generalization/<episode>.hdf5

usage: python fetch_bench2dex_replay.py --files-dir DIR --root ROOT --layout vae|pf [--workers 8] TASK ...
"""
import argparse, hashlib, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REV = "b195787c65046083e6a43776b67bdf1389dfa3eb"  # revision locked for task21 (SIM_DATA_PIPELINE.md 2.1)


def short_name(task):
    num, word = task.split("_")[:2]
    return {"80": "task80_desk", "07": "task07_citrus", "06": "task06_fruit", "27": "task27_ball"}.get(num, f"task{num}_{word}")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(rec, out):
    if out.exists() and out.stat().st_size == rec["size"] and sha256(out) == rec["lfs"]["oid"]:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    url = f"https://hf-mirror.com/datasets/Bench2Dex/teleopdata/resolve/{REV}/{rec['path']}"
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size == rec["size"] and sha256(tmp) == rec["lfs"]["oid"]:
                tmp.rename(out)
                return "ok"
            raise ValueError("size/SHA256 mismatch")
        except Exception as e:
            print(f"RETRY {attempt + 1} {rec['path']}: {e!r}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"FAILED {rec['path']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layout", choices=("vae", "pf"), required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("tasks", nargs="+")
    a = p.parse_args()
    jobs = []
    for task in a.tasks:
        recs = json.loads((a.files_dir / f"{task}.files.json").read_text())
        if len(recs) != 100:
            raise ValueError(f"{task}: expected 100 files, got {len(recs)}")
        for r in sorted(recs, key=lambda r: r["path"]):
            rel = r["path"] if a.layout == "vae" else r["path"].split("dataset/", 1)[1]
            jobs.append((task, r, a.root / (short_name(task) if a.layout == "vae" else "") / rel))
    done = {}
    with ThreadPoolExecutor(a.workers) as pool:
        for (task, r, out), res in zip(jobs, pool.map(lambda j: fetch(j[1], j[2]), jobs)):
            done[task] = done.get(task, 0) + 1
            if done[task] == 100:
                print(f"TASK_DONE {task} {time.strftime('%H:%M:%S')}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
