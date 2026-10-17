"""Prepare stride-one training windows from persistent v3 tracks; reuse baseline RGB/VAE/actions."""

import argparse
import hashlib
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from cosmos_framework.data import frame_transform
from cosmos_framework.data.fk_window import fk_window_sample
from cosmos_framework.data.pointflow_anchor_selection import (
    OBJECT_MARGIN_METRES,
    OBJECT_QUOTA_TASKS,
    hand_guided_fps,
    match_quota_objects,
    object_membership,
    object_quota_hand_guided_fps,
    quota_object_in_camera,
)
from cosmos_framework.data.pointflow_dataset import pointflow_sample
from cosmos_framework.data.pointflow_window import PointFlowTiming


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired_split(names, seed):
    groups = {}
    for name in sorted(names):
        groups.setdefault(name.removesuffix("_1"), []).append(name)
    if len(groups) != 50 or any(len(group) != 2 for group in groups.values()):
        raise ValueError("Expected 50 trajectory pairs, each with episode_xxxxxx and episode_xxxxxx_1")
    val_groups = set(random.Random(seed).sample(sorted(groups), 3))
    val = sorted(name for group in val_groups for name in groups[group])
    return dict(seed=seed, group_key="strip _1 suffix", train=sorted(set(names) - set(val)), val=val)


def fit_action_statistics(source_cache, splits):
    actions, joint_names = [], None
    for name in splits["train"]:
        with np.load(source_cache / "episodes" / f"{name}.npz", allow_pickle=False) as a:
            names = a["joint_names"].tolist()
            if joint_names is None:
                joint_names = names
            if names != joint_names:
                raise ValueError(f"Joint order changed in {name}")
            valid = a["action_valid"].astype(bool)
            x = a["action"][valid].astype(np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite valid action in {name}")
            actions.append(x)
    x = np.concatenate(actions)
    if not len(x):
        raise ValueError("No valid training actions")
    return dict(
        source="training episodes only; valid recorded action frames; per-joint q01/q99",
        joint_names=joint_names,
        action=dict(q01=np.quantile(x, 0.01, axis=0).tolist(), q99=np.quantile(x, 0.99, axis=0).tolist()),
        frame_weighting="all_action_valid_frames_once",
        valid_training_frames=len(x),
        state_transform="same_as_action",
        min_scale=0.05,
        forward_clamp=None,
        split=splits,
    )


def load_quota_objects(args, name, geometry_file):
    """Simulator object boxes (world frame) and camera extrinsics of one episode, for --object-quota."""
    import h5py

    box_file = Path(args.box3d_source.format(local_export=args.local_export, episode=name))
    with h5py.File(box_file, "r") as handle:
        matched = match_quota_objects(list(handle["labels/box3d"]), args.object_task)
        boxes = [
            dict(
                name=obj,
                kind=kind,
                center=handle[f"labels/box3d/{obj}/center_world"][:].astype(np.float64),
                quat=handle[f"labels/box3d/{obj}/quat_world"][:].astype(np.float64),
                size=handle[f"labels/box3d/{obj}/size_lwh"][:].astype(np.float64),
            )
            for obj, kind in matched
        ]
        frames = handle["labels/box3d"][matched[0][0]]["center_world"].shape[0]
    with h5py.File(geometry_file, "r") as handle:
        extrinsics = handle["cameras/cam_overhead/extrinsic_world_from_cam"][:]
    if len(extrinsics) != frames:
        raise ValueError(f"{name}: extrinsics {len(extrinsics)} != box3d frames {frames}")
    return boxes, extrinsics


def episode_job(args, name, split, latent):
    torch.set_num_threads(1)

    out = args.training_cache / "pointflow_fk_handguided1024"
    done = out / f"{name}.json"
    if done.exists():
        result = json.loads(done.read_text())
        if any(row["split"] != split for row in result["windows"]):
            raise ValueError(f"Cached split changed for {name}; use a separate output directory")
        return result
    src = args.local_export / name / "pointflow" if args.local_export else args.tracks / name

    def load(key):
        return np.load(src / (key + ".npy"), mmap_mode="r")

    offsets, tracks, xyz, uv, valid, unique, times, intr = [
        load(k)
        for k in (
            "frame_offsets",
            "obs_track",
            "obs_pos",
            "obs_uv",
            "obs_valid",
            "obs_unique",
            "timestamps_sec",
            "intrinsics",
        )
    ]
    cache = args.cloud_cache or args.raw_bundle / "datasets/bench2dex-task21-cosmos-cache"
    if args.local_export:
        caption = args.task_text
        if not caption:
            alignment = json.loads((cache / "alignment" / f"{name}.json").read_text())
            caption = alignment["metadata"].get("instruction", "").strip()
        if not caption:
            raise ValueError(f"Missing source language instruction for {name}; provide --task-text")
        with np.load(cache / "episodes" / f"{name}.npz") as a:
            rec = dict(joint_names=a["joint_names"].tolist(), task_text=caption)
    else:
        source_manifest = json.loads((cache / "manifest.json").read_text())
        rec = next(e for e in source_manifest["episodes"] if e["name"] == name)
    stats = json.loads(args.stats.read_text())
    order = [rec["joint_names"].index(n) for n in stats["joint_names"]]
    with np.load(cache / "episodes" / f"{name}.npz") as a:
        state, actions = a["state"][:, order], a["action"][:, order]
        action_valid = a["action_valid"].copy()
    rgb = np.load(cache / "video_frames" / f"{name}.npy", mmap_mode="r")
    fk_path = (
        args.local_export / name / "fk21.npz"
        if args.local_export
        else args.raw_bundle / "raw_data/bench2dex_task21" / name / "annotations/wuji_fk21.npz"
    )
    preview = (
        args.local_export / name / "head.mp4"
        if args.local_export
        else args.raw_bundle / "raw_data/bench2dex_task21" / name / "videos/head.mp4"
    )
    with np.load(fk_path) as a:
        fk_archive = {k: a[k] for k in a.files}
    rows = []
    moments = {k: [0, 0.0, 0.0] for k in ("pointflow", "fk")}
    object_stats = None
    if args.object_quota or args.save_object_instance:
        geometry_file = args.local_export / name / "rgbd.hdf5"
        if not geometry_file.exists():
            geometry_file = args.local_export / name / "observations.hdf5"
        boxes, box_extrinsics = load_quota_objects(args, name, geometry_file)
        object_stats = dict(objects=[b["name"] for b in boxes], available=[], selected=[])
    if args.frame == "robot":
        if not args.local_export:
            raise ValueError("--frame robot needs --local-export (rgbd.hdf5 + fk21.json)")
        robot_from_cam, extrinsics, _ = frame_transform.read_episode_transform(args.local_export / name)
        if len(extrinsics) != len(fk_archive["source_frame_ids"]) or np.abs(extrinsics - extrinsics[:1]).max() != 0:
            raise ValueError(f"{name}: camera extrinsics are not static / frame count differs")
    for offset, start in enumerate(latent["start_frames"]):
        frame_ids = np.arange(start, start + 33)
        assert frame_ids[-1] < len(state) and action_valid[frame_ids[:-1]].all()
        sl = slice(offsets[start], offsets[start + 1])
        p = np.asarray(xyz[sl])
        u = np.asarray(uv[sl])
        tid = np.asarray(tracks[sl])
        good = valid[sl] & unique[sl] & np.isfinite(p).all(1) & (p[:, 2] > 0) & np.isfinite(u).all(1)
        good &= (u[:, 0] >= 0) & (u[:, 0] < 640) & (u[:, 1] >= 0) & (u[:, 1] < 480)
        p, u, tid = p[good], u[good], tid[good]
        tree = cKDTree(p)
        _, nn = tree.query(p, k=min(16, len(p)), workers=1)
        local = p[nn].astype(np.float64)
        local -= local.mean(1, keepdims=True)
        ev, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local) / local.shape[1])
        good = ev[:, 1] > 1e-12
        p, u, tid, normal = p[good], u[good], tid[good], vec[good, :, 0].astype(np.float32)
        normal[np.sum(normal * p, axis=1) > 0] *= -1
        fk = fk_window_sample(fk_archive, frame_ids, name, 20)
        member = None
        if args.object_quota or args.save_object_instance:
            # Membership from the simulator box3d at the anchor frame (same rule as inference), camera frame.
            world_from_optical = frame_transform.world_from_optical(box_extrinsics[start])
            objects = [
                quota_object_in_camera(b["kind"], b["center"][start], b["quat"][start], b["size"][start], world_from_optical)
                for b in boxes
            ]
            member = object_membership(p, objects)
        if args.object_quota:
            keep, detail = object_quota_hand_guided_fps(p, fk["inputs"]["anchor_xyz"], 1024, 0.05, member, args.object_quota)
            object_stats["available"].append([int((member == i).sum()) for i in range(len(boxes))])
            object_stats["selected"].append([int((member[keep] == i).sum()) for i in range(len(boxes))])
        else:
            keep, detail = hand_guided_fps(p, fk["inputs"]["anchor_xyz"], 1024, 0.05)
        instance = None if member is None else member[keep].astype(np.int8)  # per selected point, -1: none
        p, u, tid, normal = p[keep].copy(), u[keep].copy(), tid[keep].copy(), normal[keep].copy()
        # Composed source RGB has unscaled head in the top 480 rows.
        import cv2

        head = np.asarray(rgb[start, :, :480, :]).transpose(1, 2, 0)
        color = cv2.remap(head, u[:, 0, None], u[:, 1, None], cv2.INTER_LINEAR).reshape(-1, 3)
        shift = (p.min(0) + p.max(0)) / 2
        shift[2] = p[:, 2].min()
        center = p - shift
        grid = np.floor(center / 0.02).astype(np.int64)
        grid -= grid.min(0)
        _, reps, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        feat = np.concatenate([center, color.astype(np.float32) / 255, normal], axis=1)
        disp = np.zeros((32, len(p), 3), np.float32)
        mask = np.zeros((32, len(p)), bool)
        gt_uv = np.broadcast_to(u, (33, len(p), 2)).copy()
        for j, f in enumerate(frame_ids[1:]):
            sl = slice(offsets[f], offsets[f + 1])
            cur = np.asarray(tracks[sl])
            sort = np.argsort(cur)
            cur = cur[sort]
            loc = np.searchsorted(cur, tid)
            hit = loc < len(cur)
            loc = np.minimum(loc, len(cur) - 1)
            hit &= cur[loc] == tid
            obs = offsets[f] + sort[loc]
            hit &= valid[obs] & unique[obs]
            mask[j] = hit
            disp[j, hit] = xyz[obs[hit]] - p[hit]
            gt_uv[j + 1, hit] = uv[obs[hit]]
        k = np.asarray(intr[start]).copy()
        window = dict(
            point_ids=tid,
            anchor_xyz=p,
            anchor_uv=u,
            normal=normal,
            color=color,
            coord_shift=shift,
            coord=center[reps],
            feat=feat[reps],
            grid_coord=grid[reps],
            original_to_voxel=inverse,
            voxel_representatives=reps,
            intrinsics_normalized=k,
            image_size_wh=np.array([640, 480]),
            target_displacement=disp,
            target_valid=mask,
            raw_frame_ids=frame_ids,
            timestamps_sec=times[frame_ids].copy(),
            target_seconds=np.arange(33) / 20,
            point_block_seconds=np.arange(4, 33, 4) / 20,
        )
        point = pointflow_sample(window, name, start, args.seed, PointFlowTiming(fps=20, steps=32, steps_per_token=4))
        if args.save_object_instance:
            # Current-frame task-object id of each selected point (same membership as selection/inference).
            point["inputs"]["point_instance"] = instance
        point["metadata"].update(
            source_path=str(src),
            preview_video=str(preview),
            tracker_size_wh=[640, 480],
            uv_to_video=np.array([[1, 0, 0], [0, 1, 0]], np.float32),
            video_size_wh=np.array([640, 720]),
            geometry_source="v3 episode-global RGB tracks; current anchor/FK selection",
            preview_gt_uv=gt_uv,
            preview_intrinsics_normalized=k,
            uses_gt_query_mask=True,
            selection_uses_future=False,
        )
        fk["metadata"].update(intrinsics_px=fk_archive["intrinsics"].copy(), image_size_wh=np.array([640, 480]))
        sample = dict(
            action=torch.from_numpy(np.concatenate([state[frame_ids[:1]], actions[frame_ids[:-1]]]).astype(np.float32)),
            ai_caption=rec["task_text"],
            raw_frame_ids=frame_ids,
            joint_names=stats["joint_names"],
            pointflow=point,
            fk=fk,
        )
        if args.frame == "robot":
            sample = frame_transform.convert_sample_to_robot(sample, robot_from_cam)
        path = out / f"{name}_{start:06d}.pt"
        torch.save(sample, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        rows.append(dict(episode=name, start_frame=start, frame_ids=frame_ids.tolist(), split=split, path=path.name))
        if split == "train":
            for key in moments:
                t = sample[key]["targets"]
                x = t["displacement"][t["valid"]].astype(np.float64)
                n, s, q = moments[key]
                moments[key] = [n + x.size, s + float(x.sum()), q + float(np.square(x).sum())]
    result = dict(windows=rows, moments=moments)
    if object_stats is not None:
        result["object_quota"] = object_stats
    done.write_text(json.dumps(result))
    print(name, len(rows), "complete", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path)
    p.add_argument("--raw-bundle", type=Path, help="Legacy task21 source bundle; unnecessary with --local-export")
    p.add_argument("--local-export", type=Path, help="Episode directories containing pointflow/, fk21.npz and head.mp4")
    p.add_argument("--cloud-cache", type=Path, help="Existing cloud episodes/ and video_frames/ cache")
    p.add_argument(
        "--task-text", default=None, help="Optional override; otherwise use each episode's recorded instruction"
    )
    p.add_argument("--latent-root", type=Path, help="Defaults to --cloud-cache/vae_window_latents")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--stats", type=Path, help="Existing statistics; local exports fit train-only statistics when omitted"
    )
    p.add_argument("--split", type=Path, help="Existing split/selection manifest; local exports default to paired 94/6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit-episodes", type=int, default=0)
    p.add_argument(
        "--frame",
        choices=["camera", "robot"],
        default="camera",
        help="robot: write anchors/FK/displacements in the URDF base_link frame (manifest frame=robot_base)",
    )
    p.add_argument(
        "--object-quota",
        type=int,
        default=0,
        help="0 (default): original hand_guided_fps. N>0: FPS up to N points inside each task object "
        "(simulator box3d membership) before the 512 global / 2x256 hand / fill budgets",
    )
    p.add_argument("--object-task", choices=sorted(OBJECT_QUOTA_TASKS), help="Task object list for --object-quota / --save-object-instance")
    p.add_argument(
        "--save-object-instance",
        action="store_true",
        help="Also store pointflow.inputs.point_instance (int8 task-object id per selected point, -1 none) from the "
        "same box3d membership; selection unchanged unless --object-quota",
    )
    p.add_argument(
        "--box3d-source",
        default="{local_export}/{episode}/rgbd.hdf5",
        help="Per-episode HDF5 with labels/box3d/<obj>/{center_world,quat_world,size_lwh} (format fields: local_export, episode)",
    )
    args = p.parse_args()
    if args.object_quota < 0 or (args.object_quota > 0 or args.save_object_instance) != (args.object_task is not None):
        p.error("--object-task is required by (and only valid with) --object-quota N>0 / --save-object-instance")
    if args.save_object_instance and not args.local_export:
        p.error("--save-object-instance needs --local-export")
    if args.object_quota and not args.local_export:
        p.error("--object-quota needs --local-export")
    if args.local_export:
        if not args.cloud_cache:
            p.error("--local-export requires --cloud-cache")
        args.local_export = args.local_export.resolve()
    elif not args.tracks:
        p.error("Provide --tracks or --local-export")
    elif args.raw_bundle is None:
        p.error("Legacy --tracks input requires --raw-bundle")
    if args.latent_root is None:
        if args.cloud_cache is None:
            p.error("Provide --latent-root or --cloud-cache")
        args.latent_root = args.cloud_cache / "vae_window_latents"
    lm = json.loads((args.latent_root / "window_manifest.json").read_text())
    names = [r["name"] for r in lm["episodes"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate latent episode names")
    if args.local_export and set(names) != {e.name for e in args.local_export.glob("episode_*") if e.is_dir()}:
        raise ValueError("Local export and latent episode inventories differ")
    cache = args.output / "datasets/cosmos-cache"
    legacy_cache = args.output / "datasets/bench2dex-task21-cosmos-cache"
    if not cache.exists() and legacy_cache.exists():
        cache = legacy_cache
    args.training_cache = cache
    out = cache / "pointflow_fk_handguided1024"
    out.mkdir(parents=True, exist_ok=True)
    source_cache = args.cloud_cache or args.raw_bundle / "datasets/bench2dex-task21-cosmos-cache"
    if not (cache / "video_frames").exists():
        (cache / "video_frames").symlink_to(source_cache / "video_frames", target_is_directory=True)
    existing_manifest = out / "manifest.json"
    stats_path = args.output / "normalization_train_only.json"
    saved_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if args.split:
        splits = json.loads(args.split.read_text())
    elif existing_manifest.exists():
        splits = json.loads(existing_manifest.read_text())["split"]
    elif saved_stats and "split" in saved_stats:
        splits = saved_stats["split"]
    elif args.local_export:
        splits = paired_split(names, args.seed)
    else:
        p.error("Legacy --tracks input requires --split")
    splits = splits.get("split", splits)
    train = set(splits["train"])
    val = set(splits.get("val", splits.get("validation", [])))
    assert not train & val and len(train) == 94 and len(val) == 6
    if set(names) != train | val:
        raise ValueError("Train/validation split does not cover the latent episode inventory")
    if args.stats:
        stats_bytes = args.stats.read_bytes()
    elif saved_stats:
        stats_bytes = stats_path.read_bytes()
    elif args.local_export:
        stats_bytes = (json.dumps(fit_action_statistics(source_cache, splits), indent=2) + "\n").encode()
    else:
        p.error("Legacy --tracks input requires --stats")
    stats = json.loads(stats_bytes)
    if not stats["source"].startswith("training episodes only"):
        raise ValueError("Action statistics must be fitted on training episodes only")
    if "split" in stats and stats["split"] != splits:
        raise ValueError("Action statistics use a different split")
    if saved_stats and stats_path.read_bytes() != stats_bytes:
        raise ValueError("Output already contains different action statistics; use a separate output directory")
    recipe = None
    if args.object_quota:
        recipe = dict(
            name="object_quota_v1",
            quota=args.object_quota,
            task=args.object_task,
            objects=[dict(part=part, kind=kind) for part, kind in OBJECT_QUOTA_TASKS[args.object_task]],
            membership="simulator box3d at the anchor frame, camera optical frame; sphere: |x-c|<=max(size)/2+margin, "
            "box: inside oriented box expanded by margin; smallest normalised score if several",
            margin_metres=OBJECT_MARGIN_METRES,
            order="objects(<=quota each, FPS) -> global FPS to 512 -> left<=256 -> right<=256 (5 cm) -> global fill to 1024",
            box3d_source=args.box3d_source,
        )
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get("selection_recipe") != recipe:
            raise ValueError("Existing cache uses a different selection recipe; use a separate output directory")
        if previous["action_stats_sha256"] != hashlib.sha256(stats_bytes).hexdigest() or previous["split"] != splits:
            raise ValueError("Existing cache uses different statistics/split; use a separate output directory")
    stats_path.write_bytes(stats_bytes)
    args.stats = stats_path
    print(
        f"Preparing {len(names)} episodes: {len(train)} train / {len(val)} validation; action dimensions={len(stats['joint_names'])}",
        flush=True,
    )
    latent_rows = lm["episodes"]
    if args.limit_episodes:
        latent_rows = latent_rows[: args.limit_episodes]
    if any(r["name"] not in train | val for r in latent_rows):
        raise ValueError("Latent episode missing from train/validation split")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(episode_job, args, r["name"], "train" if r["name"] in train else "validation", r)
            for r in latent_rows
        ]
        for f in as_completed(futures):
            results.append(f.result())
    rows = sorted([w for res in results for w in res["windows"]], key=lambda w: (w["episode"], w["start_frame"]))
    ip = cache / "window_index.json"
    ip.write_text(
        json.dumps(
            dict(windows=rows, fps=20, chunk_length=32, enumeration="baseline stride1 starts, persistent v3 IDs")
        )
    )
    moments = {k: np.sum([r["moments"][k] for r in results], axis=0) for k in ("pointflow", "fk")}
    scales = {k: math.sqrt(q / n - (s / n) ** 2) for k, (n, s, q) in moments.items()}
    lr = cache / "vae_window_latents"
    lr.mkdir(exist_ok=True)
    for r in latent_rows:
        r["path"] = str(args.latent_root / r["path"])
        r["frame_ids"] = [list(range(s, s + 33)) for s in r["start_frames"]]
    contract = dict(
        schema="sim_pointfk_explicit_window_latents_v1",
        fps=20,
        chunk_length=32,
        storage="bfloat16_bits_in_uint16",
        window_index_sha256=digest(ip),
    )
    (lr / "window_manifest.json").write_text(json.dumps(dict(contract=contract, episodes=latent_rows)))
    manifest = dict(
        schema="sim_pointfk_handguided1024_v1",
        seed=args.seed,
        points=1024,
        radius_metres=0.05,
        global_points=512,
        per_hand_points=256,
        selection_uses_future=False,
        uses_gt_query_mask=True,
        window_index_sha256=digest(ip),
        train_only_scales=scales,
        windows=rows,
        action_stats_sha256=digest(args.stats),
        split=splits,
    )
    if recipe:
        manifest["selection_recipe"] = recipe
    if args.save_object_instance:
        manifest["point_instance"] = dict(
            task=args.object_task,
            objects=[part for part, _ in OBJECT_QUOTA_TASKS[args.object_task]],
            membership="simulator box3d at the anchor frame, camera optical frame (object_membership)",
            margin_metres=OBJECT_MARGIN_METRES,
            dtype="int8, -1 = no task object",
        )
    if args.local_export:
        # Existing .pt caches may still contain adapter symlinks in their preview
        # metadata. Resolve those at load time without rewriting every tensor file.
        manifest["source_geometry_root"] = str(args.local_export)
    if args.frame == "robot":
        manifest["frame"] = frame_transform.ROBOT_BASE
    (out / "manifest.json").write_text(json.dumps(manifest))
    print("COMPLETE", len(rows), scales, flush=True)


if __name__ == "__main__":
    main()
