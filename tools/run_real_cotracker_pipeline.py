"""Real RGB point-flow experiment: causal SAM2, CoTracker and Track4World DA3.

All stages are resumable at stage boundaries. Original recordings are read-only.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, "/data/shichaojian/pylibs_sam2")
import cv2
import numpy as np

OUT = ROOT / "pointflow_outputs/cotracker_real_20261104"
CHECKPOINTS = Path("/data/shichaojian/checkpoints")
TASKS = ["bottle", "scissors", "dropper", "micro", "sandwich"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def objects(task):
    colors = [
        (255, 145, 35),
        (80, 170, 250),
        (200, 60, 210),
        (70, 225, 230),
        (30, 210, 70),
        (220, 180, 70),
        (180, 180, 180),
        (240, 140, 160),
        (90, 80, 240),
    ]
    rows = []

    def add(name, texts, box=None, side=None):
        obj = dict(
            id=len(rows) + 1,
            name=name,
            label=2 if side else 3,
            texts=texts,
            threshold=0.12,
            max_instances=1,
            color_bgr=colors[len(rows) % len(colors)],
        )
        if side:
            obj["hand_side"] = side
        else:
            obj["search_xyxy"] = box
        rows.append(obj)

    for side in ["right"] if task in ["dropper", "sandwich"] else ["left", "right"]:
        add(side + "_hand", ["black robotic hand."], side=side)
    if task == "bottle":
        add("water_bottle", ["transparent plastic water bottle."], [210, 95, 335, 265])
        add("bottle_cap", ["white plastic bottle cap."], [230, 100, 310, 175])
        add("cup", ["dark green cup.", "cup."], [370, 150, 510, 310])
        add("wooden_block", ["wooden block."], [215, 155, 355, 315])
    elif task == "scissors":
        add("scissors", ["black handled scissors.", "scissors."], [250, 170, 435, 270])
        add("red_paper", ["red paper.", "red sheet of paper."], [140, 125, 290, 280])
        add("cup", ["metal cup.", "cup."], [320, 180, 425, 295])
        add("wooden_block_left", ["wooden block."], [150, 180, 290, 310])
        add("wooden_block_right", ["wooden block."], [320, 215, 460, 320])
    elif task == "dropper":
        add("dropper", ["dropper with a blue rubber bulb.", "glass pipette.", "dropper."], [445, 150, 565, 285])
        add("tube_rack_and_tubes", ["red test tube rack with test tubes.", "red test tube rack."], [240, 220, 395, 400])
        add("water_bottle", ["transparent bottle containing water.", "glass bottle."], [365, 130, 460, 265])
        add("beaker", ["glass beaker.", "small glass cup."], [450, 185, 530, 275])
    elif task == "micro":
        add("micropipette", ["white and blue micropipette.", "laboratory pipette."], [65, 190, 165, 340])
        add("test_tube", ["transparent centrifuge tube.", "transparent test tube."], [225, 395, 285, 480])
        add("tube_rack", ["red test tube rack."], [220, 335, 375, 480])
        add("red_liquid_beaker", ["glass beaker containing red liquid.", "glass cup."], [380, 320, 455, 445])
        add("small_cup", ["small glass cup.", "glass beaker containing yellow liquid."], [455, 330, 540, 435])
        add("pipette_stand", ["white pipette stand.", "pipette holder."], [0, 210, 190, 480])
    else:
        add("lettuce", ["green lettuce leaf.", "lettuce."], [235, 240, 435, 400])
        add("bread_slice_1", ["brown bread slice.", "slice of bread."], [478, 270, 535, 395])
        add("bread_slice_2", ["brown bread slice.", "slice of bread."], [528, 270, 580, 395])
        add("cheese", ["yellow slice of cheese.", "cheese slice."], [380, 385, 505, 480])
        add("egg", ["fried egg."], [380, 395, 510, 480])
        add("plate", ["black plate."], [215, 375, 375, 480])
        add("tray", ["white food tray.", "food container."], [355, 375, 535, 480])
    sides = ["right"] if task in ["dropper", "sandwich"] else ["left", "right"]
    if task == "micro":
        for obj in rows:
            if obj["name"] == "pipette_stand":
                obj["appearance"] = "white_material"
            if obj["name"] == "micropipette":
                obj["appearance"] = "blue_white"
    if task == "sandwich":
        for obj in rows:
            if obj["name"] == "egg":
                obj["init_point_fractions"] = [[0.44, 0.92, 1], [0.40, 0.42, 0]]
            if obj["name"] == "cheese":
                obj["init_point_fractions"] = [[0.40, 0.42, 1], [0.44, 0.92, 0]]
    recipe = dict(
        task=task,
        workspace_xyxy=[0, 25, 640, 480]
        if task in ["bottle", "scissors"]
        else (
            [200, 35, 640, 450]
            if task == "dropper"
            else ([0, 180, 640, 480] if task == "micro" else [210, 175, 640, 480])
        ),
        hand_sides=sides,
        arm_sides=sides,
        hand_distance_px=12,
        objects=rows,
        all_objects_continuously_tracked=True,
        detection_frames=[0, 5, 10, 15, 20, 30, 45, 60],
        hand_gate="SAM2 black glove material; 3px morphological opening removes thin cables; URDF silhouette retained as a diagnostic, not a hard gate on newly calibrated recordings",
    )
    recipe["urdf"] = (
        str(ROOT / "assets/dagger_fk/marvin_wuji_d435_dagger.urdf")
        if task in ["bottle", "scissors", "dropper"]
        else "/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
    )
    recipe["hand_gate"] = (
        "Correct per-recording URDF hand silhouette +12px; black glove material; 3px opening for thin cables"
    )
    return recipe


def prepare(task):
    import segment_sandwich_video as base

    out = OUT / task
    source = json.loads((out / "source.json").read_text())
    ep = Path(source["episode"])
    recipe = objects(task)
    save(out / "task_config.json", recipe)
    folder = out / "frames"
    folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ep / "videos/head.mp4"))
    info = dict(frames=int(cap.get(7)), fps=cap.get(5), width=int(cap.get(3)), height=int(cap.get(4)))
    for i in range(info["frames"]):
        ok, bgr = cap.read()
        assert ok
        if not (folder / f"{i:05d}.jpg").exists():
            cv2.imwrite(str(folder / f"{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    save(out / "video.json", info)
    if not (out / "geometry_timing.json").exists():
        base.geometry(SimpleNamespace(out=out, episode=ep))
    print("PREPARED", task, info, flush=True)


def gate(mask, obj, frame, image, allowed, sides, arms, workspace):
    import segment_sandwich_video as base

    mask = mask.astype(bool) & workspace
    if obj["label"] == 2:
        mask &= base.object_hand_gate(dict(object=obj), allowed[frame], sides[frame])
        dark = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2] < 125
        mask &= dark
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        count, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if count > 1:
            sizes = stats[1:, cv2.CC_STAT_AREA]
            accepted = np.flatnonzero(sizes >= max(12, float(sizes.max()) * 0.02)) + 1
            mask = np.isin(cc, accepted)
        return mask
    mask &= ~arms[frame]
    mask &= base.appearance_gate(image, obj)
    if mask.sum() > 0.19 * mask.size:
        mask[:] = False
    return mask


def segment(task, limit=None):
    import sam2.sam2_video_predictor as video_module
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from segment_sandwich_dynamic import DynamicSegmenter
    from segment_sandwich_video import CurrentFrameLoader

    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out = OUT / task
    if (out / "segmentation_complete.json").exists() and not limit:
        return
    recipe = json.loads((out / "task_config.json").read_text())
    info = json.loads((out / "video.json").read_text())
    T = min(info["frames"], limit or info["frames"])
    h, w = info["height"], info["width"]
    allowed, sides, arms = [
        np.load(out / (k + ".npy"), mmap_mode="r") for k in ["hand_allowed", "hand_side", "arm_core"]
    ]
    workspace = np.zeros((h, w), bool)
    x0, y0, x1, y1 = recipe["workspace_xyxy"]
    workspace[y0:y1, x0:x1] = True
    detector = DynamicSegmenter(
        dict(objects=[], box_threshold=0.12, text_threshold=0.12, nms_iou=0.5),
        str(CHECKPOINTS / "grounding-dino-base"),
        str(CHECKPOINTS / "sam2.1_hiera_large.pt"),
    )
    loader = None

    def load(video_path, image_size, **kwargs):
        nonlocal loader
        loader = CurrentFrameLoader(sorted((out / "frames").glob("*.jpg")), image_size)
        return loader, h, w

    video_module.load_video_frames = load
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(CHECKPOINTS / "sam2.1_hiera_large.pt"), device="cuda"
    )
    labels = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.uint8, shape=(T, h, w))
    instances = np.lib.format.open_memmap(out / "instances.npy", mode="w+", dtype=np.uint8, shape=(T, h, w))
    writer = cv2.VideoWriter(str(out / "segmentation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h))
    assert writer.isOpened()
    births = {}
    init = []
    rejected = []
    timings = []
    stream = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(out / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for frame in range(T):
            loader.allowed = frame
            image = cv2.imread(str(out / "frames" / f"{frame:05d}.jpg"))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            new = []
            if frame in recipe["detection_frames"]:
                detector.sam.set_image(rgb)
                for idx, obj in enumerate(recipe["objects"]):
                    if idx in births:
                        continue
                    if obj.get("hand_side"):
                        box = [0, 25, 310, 400] if obj["hand_side"] == "left" else [320, 25, 640, 420]
                    else:
                        box = obj["search_xyxy"]
                    a, b, c, d = box
                    local = dict(obj, max_instances=5)
                    detector.config = dict(
                        objects=[local], box_threshold=obj["threshold"], text_threshold=obj["threshold"], nms_iou=0.5
                    )
                    candidates, _ = detector.detect(rgb[b:d, a:c])
                    best = None
                    for candidate in candidates:
                        bb = np.asarray(candidate["box"]) + [a, b, a, b]
                        if obj.get("init_point_fractions"):
                            template = np.asarray(obj["init_point_fractions"])
                            point_coords = bb[:2] + template[:, :2] * (bb[2:] - bb[:2])
                            mm, ss, _ = detector.sam.predict(
                                point_coords=point_coords,
                                point_labels=template[:, 2].astype(int),
                                multimask_output=True,
                            )
                        else:
                            mm, ss, _ = detector.sam.predict(box=bb, multimask_output=True)
                        raw = mm[int(np.argmax(ss))].astype(bool)
                        m = gate(raw, obj, frame, image, allowed, sides, arms, workspace)
                        # Search crop is initialization-only; never restrict moving masks later.
                        crop = np.zeros((h, w), bool)
                        crop[max(b - 5, 0) : min(d + 5, h), max(a - 5, 0) : min(c + 5, w)] = True
                        m &= crop
                        area = int(m.sum())
                        if area < (35 if obj["label"] == 2 else 16) or (
                            obj["label"] != 2 and area > 0.80 * (c - a) * (d - b)
                        ):
                            continue
                        # Avoid assigning cheese, egg, or separate slices the exact same initial mask.
                        duplicate = False
                        for prev in new:
                            if obj["name"] in ["egg", "cheese", "bread_slice_1", "bread_slice_2"] and prev[1][
                                "name"
                            ] in ["egg", "cheese", "bread_slice_1", "bread_slice_2"]:
                                iou = (m & prev[2]).sum() / max((m | prev[2]).sum(), 1)
                                if iou > 0.7:
                                    duplicate = True
                        if duplicate:
                            continue
                        merit = float(candidate["score"]) + 0.1 * float(ss.max())
                        if best is None or merit > best[0]:
                            best = (merit, m, bb.tolist(), float(candidate["score"]))
                    if best:
                        _, m, bb, score = best
                        new.append((idx, obj, m))
                        births[idx] = frame
                        init.append(
                            dict(
                                instance_id=idx + 1,
                                object=obj,
                                frame=frame,
                                box=bb,
                                score=score,
                                mask_pixels=int(m.sum()),
                            )
                        )
                        np.save(out / f"initial_mask_{idx + 1:02d}.npy", m)
                        predictor.add_new_mask(state, frame_idx=frame, obj_id=idx + 1, mask=m)
                if new:
                    if stream is not None:
                        stream.close()
                    stream = predictor.propagate_in_video(
                        state, start_frame_idx=frame, max_frame_num_to_track=T - frame - 1, reverse=False
                    )
            imap = np.zeros((h, w), np.uint8)
            lmap = np.zeros((h, w), np.uint8)
            if stream is not None:
                fi, ids, logits = next(stream)
                assert fi == frame
                masks = []
                for j, objid in enumerate(ids):
                    obj = recipe["objects"][objid - 1]
                    m = gate((logits[j, 0] > 0).cpu().numpy(), obj, frame, image, allowed, sides, arms, workspace)
                    masks.append((objid, obj, m))
                for objid, obj, m in sorted(masks, key=lambda it: (it[1]["label"] == 2, -int(it[2].sum()))):
                    imap[m] = objid
                    lmap[m] = obj["label"]
            instances[frame] = imap
            labels[frame] = lmap
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - tic) * 1000)
            for idx, obj in enumerate(recipe["objects"]):
                m = imap == idx + 1
                image[m] = (image[m] * 0.45 + np.asarray(obj["color_bgr"]) * 0.55).astype(np.uint8)
            cv2.putText(
                image, f"{task} frame {frame} | all SAM2 tracked: {len(births)}", (7, 22), 0, 0.5, (255, 255, 255), 1
            )
            writer.write(image)
            if frame in [0, 15, 60, T // 4, T // 2, 3 * T // 4, T - 1]:
                cv2.imwrite(str(out / f"preview_{frame:05d}.jpg"), image)
            if frame % 100 == 0:
                print("SAM", task, frame, "/", T, "objects", len(births), "ms", timings[-1], flush=True)
    writer.release()
    labels.flush()
    instances.flush()
    initinfo = dict(
        config=recipe,
        detections=init,
        missing=[obj["name"] for i, obj in enumerate(recipe["objects"]) if i not in births],
        dino_frames=recipe["detection_frames"],
        births=births,
        uses_future_masks=False,
    )
    save(out / "initialization.json", initinfo)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(out / "segmentation.mp4"),
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
            str(out / "segmentation_h264.mp4"),
        ],
        check=True,
    )
    save(
        out / ("segmentation_smoke.json" if limit else "segmentation_complete.json"),
        dict(frames=T, missing=initinfo["missing"], p50_ms=float(np.median(timings[5:])), timings=timings),
    )
    print("SEGMENT COMPLETE", task, initinfo["missing"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "segment", "points", "render", "verify"])
    p.add_argument("--task", choices=TASKS + ["all"], default="all")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    for task in TASKS if a.task == "all" else [a.task]:
        if a.stage == "prepare":
            prepare(task)
        elif a.stage == "segment":
            segment(task, a.limit)
        else:
            from real_cotracker_outputs import run_stage

            run_stage(a.stage, OUT / task, a.limit)


if __name__ == "__main__":
    main()
