"""A-E causal single-image depth evaluation on recorded RGB/D435 observations.

Isolated experiment: no training or serving code is mutated. AnyView geometry is
aligned to current-image Metric depth with calibrated RGB focal length. No chunk
normalization or future frames. D/E use an unfiltered RGB-aligned sensor z-buffer.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
from render_d435_point_accuracy import TASKS, Surface, load_module, measured_cloud, visible_surface
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
MODES = ["A", "B", "C", "D", "E"]


def load_model():
    external = "/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external"
    sys.path.insert(0, external)
    import torch
    from depth_anything_3.cfg import create_object
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.36)
    ckpt = Path("/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1")
    cfg = json.loads((ckpt / "config.json").read_text())["config"]
    cfg["anyview"].pop("gs_head", None)
    cfg["anyview"].pop("gs_adapter", None)
    model = create_object(OmegaConf.create(cfg))
    state = {
        k.removeprefix("model."): v
        for k, v in load_file(str(ckpt / "model.safetensors")).items()
        if not k.startswith(("model.da3.gs_head.", "model.da3.gs_adapter."))
    }
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        src = next((name for name in names if name in state), None)
        if src:
            for name in names:
                state.setdefault(name, state[src])
    model.load_state_dict(state, strict=True)
    del state
    state = torch.load("/data/shichaojian/checkpoints/track4world_da3.pth", map_location="cpu", weights_only=True)
    remapped = {k.replace("backbone.model.", "da3.", 1): v for k, v in state.items() if k.startswith("backbone.model.")}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert all(k.startswith("da3_metric.") for k in missing)
    assert all(k.startswith(("da3.gs_head.", "da3.gs_adapter.")) for k in unexpected)
    del state, remapped
    return model.eval().cuda()


def depth_metrics(points, tree):
    if not len(points):
        return dict(n=0, p50_mm=None, p90_mm=None, mean_mm=None)
    distance = tree.query(points)[0] * 1000
    return dict(
        n=len(points),
        p50_mm=float(np.median(distance)),
        p90_mm=float(np.percentile(distance, 90)),
        mean_mm=float(distance.mean()),
    )


def object_outliers(points):
    """Descriptive SOR rate, not object accuracy or ground truth."""
    if len(points) < 17:
        return None
    d = cKDTree(points).query(points, k=17, workers=1)[0][:, 1:].mean(1)
    return float(np.mean(d > max(float(d.mean() + 2 * d.std()), 0.008)))


def lift(z, intr):
    y, x = np.indices(z.shape)
    return np.stack(((x - intr["ppx"]) / intr["fx"] * z, (y - intr["ppy"]) / intr["fy"] * z, z), axis=-1)


def right_keypoints(reference, sources):
    world = np.array(
        [reference.data.xanchor[index] if kind == "joint" else reference.data.xpos[index] for kind, index in sources]
    )
    return ((world - reference.data.xpos[reference.camera]) @ reference.data.xmat[reference.camera].reshape(3, 3)) * [
        -1,
        -1,
        1,
    ]


def aggregate(rows):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for group in ["all_hand", "selected_hand"]:
            values = [r[mode][group] for r in rows if r[mode][group]["n"]]
            p90 = np.array([r["p90_mm"] for r in values])
            result[mode][group] = (
                dict(
                    valid_frames=len(values),
                    missing_frames=len(rows) - len(values),
                    p90_median_mm=float(np.median(p90)),
                    p90_mean_mm=float(p90.mean()),
                    p90_max_mm=float(p90.max()),
                    frames_p90_gt30=int((p90 > 30).sum()),
                    frame_p50_median_mm=float(np.median([r["p50_mm"] for r in values])),
                    selected_hand_count_mean=float(np.mean([r["n"] for r in values])),
                )
                if len(values)
                else dict(valid_frames=0, missing_frames=len(rows))
            )
        for key in ["selected_object_sor_rate", "selection_ms"]:
            values = [r[mode][key] for r in rows if r[mode][key] is not None]
            result[mode][key] = float(np.mean(values)) if values else None
    for mode in ["D", "E"]:
        scales = np.array([r[mode]["scale"] for r in rows])
        result[mode]["scale"] = dict(
            mean=float(scales.mean()),
            std=float(scales.std()),
            min=float(scales.min()),
            max=float(scales.max()),
            median_abs_step=float(np.median(np.abs(np.diff(scales)))),
            max_abs_step=float(np.max(np.abs(np.diff(scales)))),
            fallback_frames=[r["frame"] for r in rows if r[mode]["scale_fallback"]],
        )
    timing = {}
    for name in ["B", "C"]:
        samples = [r[name]["da3_ms"] for r in rows]
        timing[name] = dict(
            p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)), mean_ms=float(np.mean(samples))
        )
    result["single_frame_da3_timing"] = timing
    return result


def main(args):
    import torch
    import torch.nn.functional as F
    from addict import Dict

    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / "frames").mkdir(exist_ok=args.resume)
    ep, mask = TASKS[args.task]
    episode, mask = Path("/data/shichaojian/raw_data") / ep, Path(mask)
    info = json.loads((mask / "initialization.json").read_text())
    config = (
        json.loads((mask / "task_config.json").read_text()) if (mask / "task_config.json").exists() else info["config"]
    )
    streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    intr = streams["color"]["intrinsics"]
    labels, instances = (np.load(mask / f"{name}.npy", mmap_mode="r") for name in ["labels", "instances"])
    reference = Surface(episode, config)
    fk_module = load_module("sources", "/data/shichaojian/wuji-mjlab/scripts/replay/export_wuji_fk21.py")
    sources = fk_module.fk21_sources(reference.model, "right")
    selector = load_module("selection_preview", str(ROOT / "tools/validate_real_foreground_selection.py"))
    selector.BUDGET = 1024
    if args.fast_fps:
        from depth_comparison_fast_fps import anchor_fps

        selector.fps = anchor_fps
    local = Path("/tmp/d435_point_accuracy_20261102") / args.task
    local.mkdir(parents=True, exist_ok=True)
    original = episode / "auxiliary_camera/depth.lmdb/data.mdb"
    if not (local / "data.mdb").exists() or (local / "data.mdb").stat().st_size != original.stat().st_size:
        shutil.copyfile(original, local / "data.mdb")
    env = lmdb.open(str(local), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        )
    if args.limit:
        frames = frames[: args.limit]
    sync = json.loads((episode / "sync_timestamps.json").read_text())
    cap = cv2.VideoCapture(str(episode / "videos/head.mp4"))
    model = load_model()
    K = torch.tensor(
        [
            [intr["fx"] * 630 / 640, 0, (intr["ppx"] + 0.5) * 630 / 640 - 0.5],
            [0, intr["fy"] * 448 / 480, (intr["ppy"] + 0.5) * 448 / 480 - 0.5],
            [0, 0, 1],
        ],
        device="cuda",
    )
    kwargs = dict(intrinsics=K[None, None], extrinsics=torch.eye(4, device="cuda")[None, None])
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    previous_scale = {"D": 1.0, "E": 1.0}
    rows = []
    if args.resume and (out / "per_frame.jsonl").exists():
        rows = [json.loads(line) for line in (out / "per_frame.jsonl").read_text().splitlines() if line.strip()]
        assert [r["frame"] for r in rows] == frames[: len(rows)]
        if rows:
            previous_scale = {m: rows[-1][m]["scale"] for m in ["D", "E"]}
    completed = len(rows)
    for idx, frame in enumerate(frames):
        if idx < completed:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{frame:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        begin = time.perf_counter()
        points, colors, classes, ids, raw, filtered, uv = measured_cloud(
            depth, streams, instances[frame], labels[frame], image, "cpu"
        )
        sensor_ms = (time.perf_counter() - begin) * 1000
        # Unfiltered calibrated z-buffer; no range/neighbour filter until ratio support.
        zbuf = np.full(image.shape[:2], np.inf, np.float32)
        valid = np.isfinite(points).all(1) & (points[:, 2] > 0) & (depth.ravel() > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        np.minimum.at(zbuf, (uv[valid, 1], uv[valid, 0]), points[valid, 2].astype(np.float32))
        x = (
            torch.from_numpy(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 448)))
            .permute(2, 0, 1)[None]
            .float()
            .cuda()
            / 255
        )
        x = F.interpolate((x - mean) / std, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]

        def metric_forward():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return model.da3_metric(x)

        def predict(conditioned, metric):
            torch.manual_seed(42)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = model.da3(
                    x,
                    **(kwargs if conditioned else {}),
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False,
                    ref_view_strategy="saddle_balanced",
                )
            with torch.inference_mode():
                p = Dict({key: output[key].float() for key in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
                m = Dict({key: metric[key].float().clone() for key in ["depth", "sky"]})
                predK = p.intrinsics
                p.intrinsics = K[None, None]
                model._apply_metric_scaling(p, m)
                p.intrinsics = predK
                model._apply_depth_alignment(p, m)
                model._handle_sky_regions(p, m)
                return F.interpolate(p.depth[0, None], (480, 640), mode="bilinear", align_corners=False)[0, 0]

        if idx == completed:
            for _ in range(5):
                m = metric_forward()
                predict(False, m)
                predict(True, m)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        metric = metric_forward()
        torch.cuda.synchronize()
        metric_ms = (time.perf_counter() - begin) * 1000
        predictions = {}
        timings = {}
        for mode in ["B", "C"] if idx % 2 == 0 else ["C", "B"]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z = predict(mode == "C", metric)
            torch.cuda.synchronize()
            timings[mode] = (time.perf_counter() - begin) * 1000 + metric_ms
            predictions[mode] = z.cpu().numpy()
        del metric, z
        clouds = {"A": (points[filtered], classes[filtered], uv[filtered])}
        roi = instances[frame] > 0
        row = dict(
            frame=frame,
            time=float(sync[frame].get("camera_head", sync[frame]["anchor"])),
            sensor_filter_projection_ms=sensor_ms,
        )
        for mode in ["B", "C", "D", "E"]:
            source = mode if mode in ["B", "C"] else ("B" if mode == "D" else "C")
            z = predictions[source]
            scale = 1.0
            fallback = False
            ratio_count = 0
            if mode in ["D", "E"]:
                support = roi & np.isfinite(z) & (z > 0) & np.isfinite(zbuf) & (zbuf >= 0.15) & (zbuf <= 2.5)
                ratio_count = int(support.sum())
                fallback = ratio_count < 200
                scale = previous_scale[mode] if fallback else float(np.median(zbuf[support] / z[support]))
                previous_scale[mode] = scale
            z = z * scale
            good = roi & np.isfinite(z) & (z > 0)
            y, xpix = np.nonzero(good)
            clouds[mode] = (lift(z, intr)[good], labels[frame][good], np.column_stack((xpix, y)))
            row[mode] = dict(scale=scale, scale_fallback=fallback, ratio_pixels=ratio_count, da3_ms=timings[source])
        row["A"] = {}
        surface = reference.at(frame)
        fk = right_keypoints(reference, sources)
        tree = cKDTree(surface)
        payload = dict(
            urdf=visible_surface(surface, intr).astype(np.float32),
            frame=np.array(frame),
            time=np.array(row["time"]),
            rgb=cv2.resize(image, (320, 240)),
        )
        for mode, (xyz, cls, pixels) in clouds.items():
            begin = time.perf_counter()
            selected = selector.sim_recipe(xyz, fk, np.random.default_rng(frame))
            selection_ms = (time.perf_counter() - begin) * 1000
            hand = cls == 2
            selected_hand = hand[selected]
            row[mode].update(
                all_hand=depth_metrics(xyz[hand], tree),
                selected_hand=depth_metrics(xyz[selected][selected_hand], tree),
                selection_ms=selection_ms,
                selected_object_sor_rate=object_outliers(xyz[selected][~selected_hand]),
                selected_count=len(selected),
                candidate_count=len(xyz),
            )
            p, c = xyz[selected], cls[selected]
            pixel = pixels[selected]
            color = image[pixel[:, 1], pixel[:, 0], ::-1].copy()
            color[c == 2] = [255, 127, 14]
            payload[f"{mode}_points"] = p.astype(np.float32)
            payload[f"{mode}_colors"] = color
            payload[f"{mode}_labels"] = c
        np.savez_compressed(out / "frames" / f"{frame:06d}.npz", **payload)
        rows.append(row)
        with (out / "per_frame.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if idx % 20 == 0:
            print(
                args.task,
                idx,
                "/",
                len(frames),
                "DA3_ms",
                timings,
                "selected_p90",
                {m: row[m]["selected_hand"]["p90_mm"] for m in MODES},
                flush=True,
            )
    summary = aggregate(rows)
    summary["metadata"] = dict(
        task=args.task,
        episode=str(episode),
        mask=str(mask.resolve()),
        frames=len(frames),
        frame_ids=frames,
        urdf=reference.urdf,
        right_hand_selection=True,
        budget=1024,
        pool=8192,
        radius_m=0.05,
        depth_input_frames=1,
        metric="AnyView aligned to current Metric using true RGB focal",
        model_weights="Track4World AnyView + original DA3 Metric",
        warmup=5,
        seed=42,
        timing_scope="AnyView + Metric + metric alignment + depth resize; excludes image decode, upload, masks, point selection",
        gpu=torch.cuda.get_device_name(),
        precision="FP16 autocast; alignment FP32",
        cpu_offload=False,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    cap.release()
    env.close()
    print("COMPLETE", args.task, json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, default=Path("pointflow_outputs/depth_source_comparison_20261103"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fast-fps", action="store_true")
    main(parser.parse_args())
