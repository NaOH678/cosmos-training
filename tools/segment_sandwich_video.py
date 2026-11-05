"""First-frame DINO prompts, causal SAM2 video tracking, workspace and URDF hand gates.

GPU workers split objects, never time. A separate single-GPU benchmark tracks all
objects jointly. Missing first-frame targets stay missing; no later DINO calls.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from segment_sandwich_dynamic import DynamicSegmenter

RAW = Path("/data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649")
CHECKPOINTS = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/checkpoints")
# This is a search/workspace ROI, not a target's saved bounding box.
ROI = (220, 180, 640, 480)
COLORS = {
    "robot_hand": (255, 145, 35),
    "lettuce": (50, 210, 70),
    "bread": (200, 75, 210),
    "egg": (30, 145, 250),
    "cheese": (20, 220, 240),
    "plate": (210, 180, 70),
    "tray": (180, 180, 180),
    "bread_rack_group": (215, 70, 215),
    "bread_slice_1": (215, 70, 215),
    "bread_slice_2": (70, 225, 230),
}


def metadata(out):
    return json.loads((out / "video.json").read_text())


def task_config(out):
    path = out / "task_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def object_color(item):
    return tuple(item["object"].get("color_bgr", COLORS.get(item["object"]["name"], (180, 180, 180))))


def dynamic_indices(init, recipe):
    names = recipe.get("dynamic_objects")
    if names is None:
        return list(range(len(init["detections"])))
    known = {item["object"]["name"] for item in init["detections"]}
    if set(names) - known:
        raise ValueError(f"Uninitialized dynamic objects: {set(names) - known}")
    return [idx for idx, item in enumerate(init["detections"]) if item["object"]["name"] in names]


def object_hand_gate(item, allowed, side_map=None):
    side = item["object"].get("hand_side")
    if side_map is not None and side:
        return side_map == (2 if side == "right" else 3)
    return allowed


def appearance_gate(image_bgr, obj):
    """Optional task-specific visible-material gates, applied identically per frame."""
    mode = obj.get("appearance")
    if not mode:
        return np.ones(image_bgr.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if mode == "white_material":
        return (s < 90) & (v > 105)
    if mode == "red_material":
        return ((h < 15) | (h > 165)) & (s > 90) & (v > 45)
    if mode == "blue_white":
        return ((s < 90) & (v > 100)) | ((h > 85) & (h < 135) & (s > 60) & (v > 40))
    raise ValueError(f"Unknown appearance gate {mode}")


def geometry(args):
    """Current-qpos URDF silhouette; no previous SAM labels are used as truth."""
    sys.path.append("/data/shichaojian/wuji-mjlab/.venv/lib/python3.10/site-packages")
    import importlib.util

    import mujoco
    from scipy.ndimage import distance_transform_edt

    path = Path("/data/shichaojian/wuji-mjlab/scripts/replay/replay_teleop.py")
    spec = importlib.util.spec_from_file_location("replay_trial", path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    recipe = task_config(args.out)
    urdf = Path(
        recipe.get(
            "urdf", "/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
        )
    )
    model = mujoco.MjModel.from_xml_string(replay.resolve_urdf_resources(urdf))
    data = mujoco.MjData(model)
    trajectory, meta = replay.load_episode(args.episode)
    hand_sides = recipe.get("hand_sides", ["right"])
    arm_sides = recipe.get("arm_sides", hand_sides)
    addr, _, _ = replay.model_joint_map(model, replay.dataset_joint_names(meta))
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_d435_link_optical_frame")
    ci = json.loads((args.episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]["color"]["intrinsics"]
    rng = np.random.default_rng(0)
    points, indices, classes = [], [], []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        hand = any(body.startswith(side + "_hand_") for side in hand_sides)
        arm = any(
            body.endswith("_R" if side == "right" else "_L") or body.startswith(side + "_wrist_") for side in arm_sides
        )
        if not (hand or arm) or "_visual_" not in name or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        f = model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        tri = v[f]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        n = int(np.ceil(area.sum() / (0.001 if hand else 0.002) ** 2))
        chosen = rng.choice(len(tri), n, p=area / area.sum())
        a, b = np.sqrt(rng.random(n)), rng.random(n)
        points.append(
            (1 - a[:, None]) * tri[chosen, 0]
            + (a * (1 - b))[:, None] * tri[chosen, 1]
            + (a * b)[:, None] * tri[chosen, 2]
        )
        indices.append(np.full(n, g))
        classes.append(np.full(n, (3 if body.startswith("left_hand_") else 2) if hand else 4))
    points, indices, classes = np.concatenate(points), np.concatenate(indices), np.concatenate(classes)
    info = metadata(args.out)
    total, height, width = info["frames"], info["height"], info["width"]
    allow = np.lib.format.open_memmap(
        args.out / "hand_allowed.npy", mode="w+", dtype=bool, shape=(total, height, width)
    )
    arms = np.lib.format.open_memmap(args.out / "arm_core.npy", mode="w+", dtype=bool, shape=(total, height, width))
    side_map = np.lib.format.open_memmap(
        args.out / "hand_side.npy", mode="w+", dtype=np.uint8, shape=(total, height, width)
    )
    times = []
    for frame in range(total):
        started = time.perf_counter()
        data.qpos[addr] = trajectory[frame]
        mujoco.mj_forward(model, data)
        world = np.einsum("nij,nj->ni", data.geom_xmat.reshape(-1, 3, 3)[indices], points) + data.geom_xpos[indices]
        camera_points = ((world - data.xpos[camera]) @ data.xmat[camera].reshape(3, 3)) * [-1, -1, 1]
        z = camera_points[:, 2]
        valid = z > 0.05
        u = np.rint(ci["fx"] * camera_points[:, 0] / np.maximum(z, 0.001) + ci["ppx"]).astype(int)
        v = np.rint(ci["fy"] * camera_points[:, 1] / np.maximum(z, 0.001) + ci["ppy"]).astype(int)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ids = np.flatnonzero(valid)
        ids = ids[np.argsort(-z[ids])]
        rendered = np.zeros((height, width), np.uint8)
        rendered[v[ids], u[ids]] = classes[ids]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for cls in (4, 3, 2):
            m = cv2.morphologyEx((rendered == cls).astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            rendered[m & ((rendered == 0) | (cls in (2, 3)))] = cls
        _, near = distance_transform_edt(rendered == 0, return_indices=True)
        hand_pixels = (rendered == 2) | (rendered == 3)
        nearest = rendered[tuple(near)]
        allow[frame] = (distance_transform_edt(~hand_pixels) <= recipe.get("hand_distance_px", 12)) & (
            (nearest == 2) | (nearest == 3)
        )
        side_map[frame] = np.where(allow[frame], nearest, 0)
        arms[frame] = rendered == 4
        times.append((time.perf_counter() - started) * 1000)
        if frame % 200 == 0:
            print("geometry", frame, flush=True)
    allow.flush()
    arms.flush()
    side_map.flush()
    (args.out / "geometry_timing.json").write_text(json.dumps(times))


def prepare(args):
    args.out.mkdir(exist_ok=False, parents=True)
    folder = args.out / "frames"
    folder.mkdir()
    cap = cv2.VideoCapture(str(args.episode / "videos/head.mp4"))
    info = dict(frames=int(cap.get(7)), fps=cap.get(5), width=int(cap.get(3)), height=int(cap.get(4)))
    frame = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(folder / f"{frame:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame += 1
    cap.release()
    assert frame == info["frames"]
    (args.out / "video.json").write_text(json.dumps(info))
    # Render current-frame hand/arm constraints before choosing prompts.
    subprocess.run(
        [sys.executable, __file__, "geometry", "--out", str(args.out), "--episode", str(args.episode)], check=True
    )
    initialize(args)


def initialize(args):
    import torch

    folder = args.out / "frames"
    config = json.loads(Path("examples/deployment/sandwich_segmentation.json").read_text())
    for obj in config["objects"]:
        obj["max_instances"] = {"bread": 2, "tray": 2}.get(obj["name"], 1)
    config.update(
        mode="first_frame_dino_forward_sam2", workspace_xyxy=list(ROI), hand_distance_px=12, max_box_crop_fraction=0.55
    )
    segmenter = DynamicSegmenter(
        config, str(CHECKPOINTS / "grounding-dino-base"), str(CHECKPOINTS / "sam2.1_hiera_large.pt")
    )
    rgb = cv2.cvtColor(cv2.imread(str(folder / "00000.jpg")), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = ROI
    crop = rgb[y0:y1, x0:x1]
    started = time.perf_counter()
    with torch.inference_mode():
        detections, missing = segmenter.detect(crop)
    torch.cuda.synchronize()
    dino_ms = (time.perf_counter() - started) * 1000
    hand = np.load(args.out / "hand_allowed.npy", mmap_mode="r")[0]
    arms = np.load(args.out / "arm_core.npy", mmap_mode="r")[0]
    accepted = []
    rejected = []
    masks = []
    segmenter.sam.set_image(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in sorted(detections, key=lambda item: -item["score"]):
            box = np.asarray(item["box"]) + [x0, y0, x0, y0]
            area = np.prod(box[2:] - box[:2])
            name = item["object"]["name"]
            if area > 0.55 * crop.shape[0] * crop.shape[1]:
                rejected.append(dict(name=name, reason="oversized_box", box=box.tolist()))
                continue
            choices, scores, _ = segmenter.sam.predict(box=box, multimask_output=True)
            mask = choices[int(np.argmax(scores))].astype(bool)
            workspace = np.zeros(mask.shape, bool)
            workspace[y0:y1, x0:x1] = True
            mask &= workspace
            if name == "robot_hand":
                mask &= hand
            else:
                if (mask & arms).sum() > 0.3 * max(int(mask.sum()), 1):
                    rejected.append(dict(name=name, reason="robot_arm_misdetected_as_object", box=box.tolist()))
                    continue
                mask &= ~arms
            if mask.sum() < 20 or mask.sum() > 0.19 * mask.size:
                rejected.append(dict(name=name, reason="empty_or_table_sized_mask", box=box.tolist()))
                continue
            # Distinct class detections of the same object should not become competing tracks.
            overlap = [(mask & prior).sum() / max((mask | prior).sum(), 1) for prior in masks]
            if name != "robot_hand" and overlap and max(overlap) > 0.75:
                rejected.append(dict(name=name, reason="duplicate_instance", box=box.tolist()))
                continue
            item["box"] = box.tolist()
            item["instance_id"] = len(accepted) + 1
            accepted.append(item)
            masks.append(mask)
    if not accepted:
        raise RuntimeError("No usable first-frame prompts")
    np.save(args.out / "initial_masks.npy", np.asarray(masks))
    initialized = {i["object"]["name"] for i in accepted}
    info = dict(
        config=config,
        detections=accepted,
        rejected=rejected,
        missing=[o["name"] for o in config["objects"] if o["name"] not in initialized],
        dino_ms=dino_ms,
        dino_frames=[0],
    )
    (args.out / "initialization.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


class CurrentFrameLoader:
    """Lazy sequential source: fail if the predictor requests a future frame."""

    def __init__(self, files, size):
        self.files, self.size, self.allowed, self.cached = files, size, 0, {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import torch
        from sam2.utils.misc import _load_img_as_tensor

        if index > self.allowed:
            raise RuntimeError(f"Future frame requested: {index}>{self.allowed}")
        if index not in self.cached:
            image, _, _ = _load_img_as_tensor(self.files[index], self.size)
            image = image.float()
            image -= torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            image /= torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            self.cached = {index: image}
        return self.cached[index]


def track(args):
    import sam2.sam2_video_predictor as module
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    torch.manual_seed(0)
    info = metadata(args.out)
    init = json.loads((args.out / "initialization.json").read_text())
    all_masks = np.load(args.out / "initial_masks.npy")
    dynamic = dynamic_indices(init, task_config(args.out))
    selected = list(range(len(all_masks))) if args.benchmark_all else dynamic
    chosen = selected if args.benchmark else selected[args.worker :: 8]
    if not chosen:
        (args.out / f"worker{args.worker}.json").write_text(json.dumps(dict(indices=[], times=[])))
        return
    loader = None

    def load(video_path, image_size, **kwargs):
        nonlocal loader
        loader = CurrentFrameLoader(sorted((args.out / "frames").glob("*.jpg")), image_size)
        return loader, info["height"], info["width"]

    module.load_video_frames = load
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(CHECKPOINTS / "sam2.1_hiera_large.pt"), device="cuda"
    )
    count = min(info["frames"], 120) if args.benchmark else info["frames"]
    benchmark_name = "benchmark_all" if args.benchmark_all else "benchmark"
    dest = args.out / (f"{benchmark_name}_masks.npy" if args.benchmark else f"worker{args.worker}_masks.npy")
    masks = np.lib.format.open_memmap(
        dest, mode="w+", dtype=bool, shape=(count, len(chosen), info["height"], info["width"])
    )
    times, model_times, load_times = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(args.out / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for idx in chosen:
            predictor.add_new_mask(state, frame_idx=0, obj_id=idx + 1, mask=all_masks[idx])
        stream = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=count - 1, reverse=False)
        for frame in range(count):
            loader.allowed = frame
            torch.cuda.synchronize()
            started = time.perf_counter()
            # Read/normalize current RGB before separately measuring SAM2's GPU path.
            loader[frame]
            loaded = time.perf_counter()
            fi, ids, logits = next(stream)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            elapsed = (finished - started) * 1000
            load_times.append((loaded - started) * 1000)
            model_times.append((finished - loaded) * 1000)
            assert fi == frame
            for j, idx in enumerate(chosen):
                masks[frame, j] = (logits[ids.index(idx + 1), 0] > 0).cpu().numpy()
            times.append(elapsed)
            if frame % 100 == 0:
                print("track", args.worker, frame, "ms", elapsed, flush=True)
    masks.flush()
    path = f"{benchmark_name}.json" if args.benchmark else f"worker{args.worker}.json"
    (args.out / path).write_text(
        json.dumps(dict(indices=chosen, times=times, sam_times=model_times, rgb_load_times=load_times))
    )


def merge(args):
    info = metadata(args.out)
    init = json.loads((args.out / "initialization.json").read_text())
    n = len(init["detections"])
    workers = []
    for worker in range(8):
        details = json.loads((args.out / f"worker{worker}.json").read_text())
        if details["indices"]:
            workers.append((details, np.load(args.out / f"worker{worker}_masks.npy", mmap_mode="r")))
    hand = np.load(args.out / "hand_allowed.npy", mmap_mode="r")
    arms = np.load(args.out / "arm_core.npy", mmap_mode="r")
    recipe = task_config(args.out)
    dynamic = dynamic_indices(init, recipe)
    static = sorted(set(range(n)) - set(dynamic))
    roi = recipe.get("workspace_xyxy", ROI)
    side_path = args.out / "hand_side.npy"
    hand_sides = np.load(side_path, mmap_mode="r") if side_path.exists() else None
    initial_support = np.load(args.out / "initial_masks.npy", mmap_mode="r")
    first_image = cv2.imread(str(args.out / "frames" / "00000.jpg")) if static else None
    shape = (info["frames"], info["height"], info["width"])
    labels = np.lib.format.open_memmap(args.out / "labels.npy", mode="w+", dtype=np.uint8, shape=shape)
    instances = np.lib.format.open_memmap(args.out / "instances.npy", mode="w+", dtype=np.uint8, shape=shape)
    writer = cv2.VideoWriter(
        str(args.out / "segmentation.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    assert writer.isOpened()
    post = []
    for frame in range(info["frames"]):
        image = cv2.imread(str(args.out / "frames" / f"{frame:05d}.jpg"))
        started = time.perf_counter()
        masks = np.zeros((n, info["height"], info["width"]), bool)
        for idx in static:
            masks[idx] = initial_support[idx]
        for details, array in workers:
            for j, idx in enumerate(details["indices"]):
                masks[idx] = array[frame, j]
        x0, y0, x1, y1 = roi
        workspace = np.zeros(shape[1:], bool)
        workspace[y0:y1, x0:x1] = True
        masks &= workspace
        for idx, item in enumerate(init["detections"]):
            if item["object"]["label"] == 2:
                masks[idx] &= object_hand_gate(item, hand[frame], hand_sides[frame] if hand_sides is not None else None)
            else:
                masks[idx] &= ~arms[frame]
                masks[idx] &= appearance_gate(image, item["object"])
                if item["object"].get("static_support"):
                    masks[idx] &= initial_support[idx]
                if masks[idx].sum() > 0.19 * masks[idx].size:
                    masks[idx] = False
        if static:
            foreground = np.any(masks[dynamic], axis=0) if dynamic else np.zeros(shape[1:], bool)
            stable = np.max(cv2.absdiff(image, first_image), axis=2) <= recipe.get("static_rgb_change_max", 80)
            for idx in static:
                masks[idx] &= ~foreground & stable
        # Exclusive display: small visible objects override containers, hand overrides all.
        instance = np.zeros(shape[1:], np.uint8)
        label = np.zeros(shape[1:], np.uint8)
        order = sorted(range(n), key=lambda i: (init["detections"][i]["object"]["label"] == 2, -int(masks[i].sum())))
        for idx in order:
            instance[masks[idx]] = idx + 1
            label[masks[idx]] = init["detections"][idx]["object"]["label"]
        instances[frame] = instance
        labels[frame] = label
        post.append((time.perf_counter() - started) * 1000)
        for idx, item in enumerate(init["detections"]):
            color = np.array(object_color(item))
            m = instance == idx + 1
            image[m] = (0.45 * image[m] + 0.55 * color).astype(np.uint8)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
        cv2.putText(
            image,
            f"frame {frame} | DINO: 0 only | SAM: {len(dynamic)}, fixed: {len(static)}",
            (8, 22),
            0,
            0.5,
            (255, 255, 255),
            1,
        )
        for row, item in enumerate(init["detections"]):
            name = item["object"]["name"] + (" [fixed]" if row in static else " [track]")
            cv2.putText(image, name, (8, 44 + 18 * row), 0, 0.45, object_color(item), 1)
        writer.write(image)
        if frame in (0, 60, 238, 476, 715, 953, 1191, info["frames"] // 2, 4 * info["frames"] // 5, info["frames"] - 1):
            cv2.imwrite(str(args.out / f"preview_{frame:04d}.png"), image)
    writer.release()
    labels.flush()
    instances.flush()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.out / "segmentation.mp4"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.out / "segmentation_h264.mp4"),
        ],
        check=True,
    )
    benchmark = json.loads((args.out / "benchmark.json").read_text())["times"][5:]
    geo = json.loads((args.out / "geometry_timing.json").read_text())
    summary = dict(
        frames=info["frames"],
        fps=info["fps"],
        dino_frames=[0],
        initialized_objects=[i["object"]["name"] for i in init["detections"]],
        dynamic_objects=[init["detections"][i]["object"]["name"] for i in dynamic],
        static_objects=[init["detections"][i]["object"]["name"] for i in static],
        missing_initial_objects=init["missing"],
        dino_initialization_ms=init["dino_ms"],
        single_gpu_all_objects_sam_ms=dict(p50=float(np.median(benchmark)), p90=float(np.percentile(benchmark, 90))),
        single_gpu_tracking_objects=len(dynamic),
        urdf_gate_ms=float(np.median(geo)),
        mask_postprocess_ms=float(np.median(post)),
        measurement="Single A800, active tracking objects together, 120 consecutive frames, first 5 excluded; includes lazy JPEG load, excludes model initialization, output and HTTP. URDF timed separately.",
        limitations=[
            "Offline video predictor, streaming deployment not integrated",
            "No redetection: first-frame missing or later arriving objects cannot be recovered",
            "Fixed masks assume a fixed camera and stationary supports; first-frame-occluded surfaces are not reconstructed",
        ],
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "geometry", "initialize", "track", "merge"])
    p.add_argument("--episode", type=Path, default=RAW)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-all", action="store_true", help="Benchmark the previous all-object tracking mode")
    args = p.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
