"""Convert a camera-frame PointFlow/FK selection cache into the robot base_link frame.

Builds a NEW bundle (never in place): converted .pt files + manifest in
``<out>/datasets/cosmos-cache/pointflow_fk_handguided1024``; VAE latents/video frames are symlinked;
window_index.json and normalization_train_only.json are copied byte-identically.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data import frame_transform as ft

SELECTION = "pointflow_fk_handguided1024"


def convert_episode(src, dst, geometry_root, name, rows, split):
    torch.set_num_threads(1)
    state = dst / f"{name}.json"
    if state.exists():
        return json.loads(state.read_text())
    root = Path(geometry_root) / name
    base = np.array(json.loads((root / "fk21.json").read_text())["world_from_robot_base"], dtype=np.float64)
    import h5py

    with h5py.File(root / "rgbd.hdf5", "r") as handle:
        extrinsics = handle["cameras/cam_overhead/extrinsic_world_from_cam"][:]
    frames = json.loads((root / "fk21.json").read_text())["frames"]
    if len(extrinsics) != frames:
        raise ValueError(f"{name}: extrinsics {len(extrinsics)} != frames {frames}")
    if np.abs(extrinsics - extrinsics[:1]).max() != 0:
        raise ValueError(f"{name}: camera is not static")
    transform = ft.robot_from_optical(extrinsics[0], base)
    moments = {k: [0, 0.0, 0.0] for k in ("pointflow", "fk")}
    for row in rows:
        sample = torch.load(src / row["path"], map_location="cpu", weights_only=False)
        start = row["start_frame"]
        window_transform = ft.robot_from_optical(extrinsics[start], base)
        if not np.array_equal(window_transform, transform):
            raise ValueError(f"{name}: window {start} transform differs from episode transform")
        new = ft.convert_sample_to_robot(sample, transform)
        path = dst / row["path"]
        tmp = path.with_suffix(".tmp")
        torch.save(new, tmp)
        os.replace(tmp, path)
        if split == "train":
            for key in moments:
                t = new[key]["targets"]
                x = t["displacement"][t["valid"]].astype(np.float64)
                n, s, q = moments[key]
                moments[key] = [n + x.size, s + float(x.sum()), q + float(np.square(x).sum())]
    result = dict(
        moments=moments,
        windows=len(rows),
        robot_from_optical=transform.tolist(),
        world_from_robot_base=base.tolist(),
        world_from_optical=ft.world_from_optical(extrinsics[0]).tolist(),
        extrinsic_frame_index=0,
        extrinsic_static=True,
    )
    state.write_text(json.dumps(result))
    print(name, len(rows), "converted", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New bundle directory; must not hold a manifest yet")
    p.add_argument("--geometry-root", type=Path, help="Defaults to manifest source_geometry_root")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit-episodes", type=int, default=0)
    args = p.parse_args()
    src_cache = args.source_bundle / "datasets/cosmos-cache"
    src = src_cache / SELECTION
    manifest = json.loads((src / "manifest.json").read_text())
    if manifest.get("frame", ft.CAMERA) != ft.CAMERA:
        raise ValueError("Source cache is not in the camera frame")
    geometry_root = args.geometry_root or Path(manifest["source_geometry_root"])
    cache = args.output / "datasets/cosmos-cache"
    dst = cache / SELECTION
    if (dst / "manifest.json").exists():
        raise FileExistsError(f"{dst}/manifest.json exists; use a new output directory")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("vae_window_latents", "video_frames"):
        if not (cache / name).is_symlink():
            (cache / name).symlink_to((src_cache / name).resolve(), target_is_directory=True)
    legacy = args.output / "datasets/bench2dex-task21-cosmos-cache"
    if not legacy.is_symlink():
        legacy.symlink_to("cosmos-cache", target_is_directory=True)
    for rel in ("datasets/cosmos-cache/window_index.json", "normalization_train_only.json"):
        shutil.copyfile(args.source_bundle / rel, args.output / rel)
        if (args.source_bundle / rel).read_bytes() != (args.output / rel).read_bytes():
            raise ValueError(f"{rel} copy differs")
    by_episode = {}
    for row in manifest["windows"]:
        by_episode.setdefault(row["episode"], []).append(row)
    names = sorted(by_episode)[: args.limit_episodes or None]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                convert_episode, src, dst, geometry_root, n, by_episode[n], by_episode[n][0]["split"]
            ): n
            for n in names
        }
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    if args.limit_episodes:
        print("PARTIAL (limit-episodes): manifest not written")
        return
    moments = {k: np.sum([r["moments"][k] for r in results.values()], axis=0) for k in ("pointflow", "fk")}
    scales = {k: math.sqrt(q / n - (s / n) ** 2) for k, (n, s, q) in moments.items()}
    old = manifest["train_only_scales"]
    for k in scales:
        if abs(scales[k] - old[k]) / old[k] > 1e-3:
            raise ValueError(f"train_only_scales[{k}] moved: {old[k]} -> {scales[k]}")
    transforms = {
        n: {k: v for k, v in r.items() if k not in ("moments", "windows")} for n, r in sorted(results.items())
    }
    transforms_path = args.output / "frame_transforms.json"
    transforms_path.write_text(
        json.dumps(
            dict(
                convention="robot_from_optical = inv(world_from_robot_base) @ (E with R <- R_E @ C); "
                "E=cameras/cam_overhead/extrinsic_world_from_cam (Isaac +X fwd,+Y left,+Z up)",
                C=ft.ISAAC_TO_OPTICAL.tolist(),
                episodes=transforms,
            ),
            indent=1,
        )
    )
    new_manifest = dict(
        manifest,
        train_only_scales=scales,
        frame=ft.ROBOT_BASE,
        frame_transforms_sha256=hashlib.sha256(transforms_path.read_bytes()).hexdigest(),
        source_cache=str(src),
        source_train_only_scales=old,
    )
    (dst / "manifest.json").write_text(json.dumps(new_manifest))
    print("COMPLETE", len(manifest["windows"]), "old", old, "new", scales, flush=True)


if __name__ == "__main__":
    main()
