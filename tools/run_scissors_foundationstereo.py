"""PointWorld FoundationStereo on calibrated, timestamp-paired D435 IR; scissors HTML."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "third_party/pointworld_depth_reference"
FS = REF / "third_party/FoundationStereo"
RUNTIME = ROOT / "third_party/foundation_runtime"
CHECKPOINT = ROOT / "checkpoints/FoundationStereo/23-51-11/model_best_bp2.pth"
V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
OUT = ROOT / "pointflow_outputs/scissors_foundationstereo_20261105"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
from render_d435_point_accuracy import load_module


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT / "native_depth").mkdir(exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    source = json.loads((V0 / "source.json").read_text())
    ep = Path(source["episode"])
    # Keep exports reproducible after the temporary LMDB extraction is removed.
    if (OUT / "calibration.json").exists() and (OUT / "ir_pair_index.json").exists():
        sync = json.loads((ep / "sync_timestamps.json").read_text())
        rgb_ts = np.array([x["camera_head"] for x in sync], np.float64)
        return (
            ep,
            json.loads((OUT / "calibration.json").read_text()),
            json.loads((OUT / "ir_pair_index.json").read_text()),
            rgb_ts,
        )
    meta = json.loads((ep / "auxiliary_camera/metadata.json").read_text())
    streams = meta["capture_metadata"]["cameras"]["head"]["streams"]
    left, right = streams["infrared_left"], streams["infrared_right"]
    R = np.array(left["extrinsics_to_color"]["rotation"]).reshape(3, 3, order="F")
    Rr = np.array(right["extrinsics_to_color"]["rotation"]).reshape(3, 3, order="F")
    assert np.max(abs(R - Rr)) < 1e-8
    translation = R.T @ (
        np.array(right["extrinsics_to_color"]["translation"]) - np.array(left["extrinsics_to_color"]["translation"])
    )
    assert translation[0] > 0 and np.linalg.norm(translation[1:]) < 1e-6
    li, ri = left["intrinsics"], right["intrinsics"]
    assert not np.any(li["coeffs"]) and not np.any(ri["coeffs"])
    for k in ["fx", "fy", "ppx", "ppy"]:
        assert abs(li[k] - ri[k]) < 1e-6
    K = np.array([[li["fx"], 0, li["ppx"]], [0, li["fy"], li["ppy"]], [0, 0, 1]], np.float32)
    env = lmdb.open("/tmp/cotracker_real_depth_20261104/scissors", readonly=True, lock=False)
    with env.begin() as txn:
        index = {
            side: {
                key: pickle.loads(txn.get(f"index/video/head_ir_{side}/{key}".encode()))
                for key in ["sequences", "timestamps", "training_steps"]
            }
            for side in ["left", "right"]
        }
    env.close()
    sync = json.loads((ep / "sync_timestamps.json").read_text())
    rgb_ts = np.array([x["camera_head"] for x in sync], np.float64)
    seq, a, b = np.intersect1d(index["left"]["sequences"], index["right"]["sequences"], return_indices=True)
    pairs = []
    for sequence, i, j in zip(seq, a, b):
        capture = max(float(index["left"]["timestamps"][i]), float(index["right"]["timestamps"][j]))
        saved = max(int(index["left"]["training_steps"][i]), int(index["right"]["training_steps"][j]))
        available = max(saved, int(np.searchsorted(rgb_ts, capture)))
        if available >= len(rgb_ts):
            continue
        pairs.append(
            dict(
                sequence=int(sequence),
                left_video_index=int(i),
                right_video_index=int(j),
                left_time=float(index["left"]["timestamps"][i]),
                right_time=float(index["right"]["timestamps"][j]),
                capture_time=capture,
                saved_rgb_step=saved,
                available_rgb_step=available,
                rgb_time=float(rgb_ts[available]),
            )
        )
    calibration = dict(
        baseline_m=float(translation[0]),
        baseline_vector_left_m=translation.tolist(),
        left_K=K.tolist(),
        streams=streams,
        rectified_parallel=True,
        source="recorded calibration; no nominal50mm substitution",
        pairs=len(pairs),
        unmatched_left=len(index["left"]["sequences"]) - len(pairs),
        unmatched_right=len(index["right"]["sequences"]) - len(pairs),
        lr_time_max_ms=max(abs(p["left_time"] - p["right_time"]) * 1000 for p in pairs),
    )
    save(OUT / "calibration.json", calibration)
    save(OUT / "ir_pair_index.json", pairs)
    for name in ["source.json", "task_config.json", "video.json"]:
        (OUT / name).write_bytes((V0 / name).read_bytes())
    return ep, calibration, pairs, rgb_ts


class PairReader:
    def __init__(self, episode):
        self.caps = [
            cv2.VideoCapture(str(episode / "auxiliary_camera" / f"head_ir_{side}.mp4")) for side in ["left", "right"]
        ]
        self.indices = [-1, -1]
        self.images = [None, None]

    def get(self, pair):
        images = []
        for side, key in enumerate(["left_video_index", "right_video_index"]):
            while self.indices[side] < pair[key]:
                ok, image = self.caps[side].read()
                assert ok, (side, pair[key], self.indices[side])
                self.indices[side] += 1
                self.images[side] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(self.images[side])
        return images

    def close(self):
        for cap in self.caps:
            cap.release()


def load_estimator(calibration):
    import timm
    import torch

    # All parameters are overwritten by the full stereo checkpoint. Avoid redundant
    # ImageNet downloads; use the exact vendored DINOv2 definitions from the pinned repo.
    original_hub = torch.hub.load
    original_timm = timm.create_model
    original_load = torch.load

    def hub(repo, model, *args, **kwargs):
        if repo == "facebookresearch/dinov2":
            kwargs["source"] = "local"
            kwargs["pretrained"] = False
            return original_hub(str(FS / "dinov2"), model, *args, **kwargs)
        return original_hub(repo, model, *args, **kwargs)

    def create(name, *args, **kwargs):
        if name == "edgenext_small":
            kwargs["pretrained"] = False
        return original_timm(name, *args, **kwargs)

    def safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        numeric = [np.core.multiarray.scalar, np.dtype, np.ndarray, np.core.multiarray._reconstruct]
        numeric += [type(np.dtype(dtype)) for dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]]
        import collections
        import typing

        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode

        numeric += [
            ContainerMetadata,
            Metadata,
            DictConfig,
            ListConfig,
            AnyNode,
            typing.Any,
            collections.defaultdict,
            dict,
            list,
            int,
        ]
        with torch.serialization.safe_globals(numeric):
            return original_load(*args, **kwargs)

    torch.hub.load = hub
    timm.create_model = create
    torch.load = safe_load
    try:
        module = load_module("pointworld_stereo_reference", REF / "depth_estimator.py")
        estimator = module.DepthEstimator(
            str(CHECKPOINT),
            baseline=calibration["baseline_m"],
            intrinsic=np.array(calibration["left_K"], np.float32),
            device="cuda",
            cfg_path=str(REF / "assets/foundationstereo/23-51-11/cfg.yaml"),
        )
    finally:
        torch.hub.load = original_hub
        timm.create_model = original_timm
        torch.load = original_load
    estimator.set_inference_params(use_hierarchical=False, iters=32)
    original_forward = estimator.model.forward

    def timed_forward(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = original_forward(*args, **kwargs)
        end.record()
        estimator.last_forward_events = (start, end)
        return value

    estimator.model.forward = timed_forward
    return estimator


def infer(estimator, pairs, episode, calibration):
    import torch

    if (OUT / "depth_complete.json").exists():
        return
    reader = PairReader(episode)
    first = reader.get(pairs[0])
    for _ in range(5):
        estimator.infer_depth(*first)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    bench = []
    for _ in range(30):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        estimator.infer_depth(*first)
        torch.cuda.synchronize()
        bench.append(
            dict(
                api_ms=(time.perf_counter() - tic) * 1000,
                forward_gpu_ms=estimator.last_forward_events[0].elapsed_time(estimator.last_forward_events[1]),
            )
        )
    save(
        OUT / "benchmark_batch1.json",
        dict(
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            input_hw=[480, 640],
            precision="FP16 autocast",
            iterations=32,
            batch_size=1,
            warmup=5,
            repeats=30,
            same_pair=True,
            samples=bench,
            api_p50_ms=float(np.median([b["api_ms"] for b in bench])),
            api_p95_ms=float(np.percentile([b["api_ms"] for b in bench], 95)),
            forward_p50_ms=float(np.median([b["forward_gpu_ms"] for b in bench])),
            forward_p95_ms=float(np.percentile([b["forward_gpu_ms"] for b in bench], 95)),
        ),
    )
    reader.close()
    reader = PairReader(episode)
    mask = np.load(V0 / "instances.npy", mmap_mode="r")
    color_k = calibration["streams"]["color"]["intrinsics"]
    rgb_k = dict(
        fx=color_k["fx"], fy=color_k["fy"] * 448 / 480, ppx=color_k["ppx"], ppy=(color_k["ppy"] + 0.5) * 448 / 480 - 0.5
    )
    integrated = []
    for pair in pairs[:30]:
        frame = pair["available_rgb_step"]
        with np.load(V0 / "pointflow/frames" / f"{frame:06d}.npz") as n:
            uv = n["uv"]
            instance = n["instances"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        left, right = reader.get(pair)
        decoded = time.perf_counter()
        depth = estimator.infer_depth(left, right)
        torch.cuda.synchronize()
        predicted = time.perf_counter()
        z = aligned_samples(depth, calibration["streams"], uv, instance, mask[frame])
        xyz = np.column_stack(
            ((uv[:, 0] - rgb_k["ppx"]) / rgb_k["fx"] * z, (uv[:, 1] - rgb_k["ppy"]) / rgb_k["fy"] * z, z)
        )
        finish = time.perf_counter()
        integrated.append(
            dict(
                sequence=pair["sequence"],
                decode_ms=(decoded - start) * 1000,
                api_ms=(predicted - decoded) * 1000,
                projection_sample_lift_ms=(finish - predicted) * 1000,
                overall_ms=(finish - start) * 1000,
                points=len(xyz),
            )
        )
    save(
        OUT / "benchmark_io_projection_batch1.json",
        dict(
            batch_size=1,
            samples=integrated,
            p50_ms=float(np.median([x["overall_ms"] for x in integrated])),
            p95_ms=float(np.percentile([x["overall_ms"] for x in integrated], 95)),
            includes="IRvideo decode + single stereo API + calibrated RGB projection/point sampling/XYZ lift",
            excludes="model initialization, SAM2, CoTracker, preloading mask/UV metadata, output disk writes, HTML",
        ),
    )
    reader.close()
    reader = PairReader(episode)
    rows = []
    for i, pair in enumerate(pairs):
        path = OUT / "native_depth" / f"{pair['sequence']:06d}.npy"
        header = path.with_suffix(".json")
        tic = time.perf_counter()
        left, right = reader.get(pair)
        read_ms = (time.perf_counter() - tic) * 1000
        if header.exists() and path.exists():
            rows.append(json.loads(header.read_text()))
            continue
        torch.cuda.synchronize()
        tic = time.perf_counter()
        depth = estimator.infer_depth(left, right)
        torch.cuda.synchronize()
        api_ms = (time.perf_counter() - tic) * 1000
        valid = np.isfinite(depth) & (depth > 0.15) & (depth < 2.5)
        depth = np.where(valid, depth, 0).astype(np.float32)
        np.save(path, depth)
        row = dict(
            sequence=pair["sequence"],
            read_ms=read_ms,
            api_ms=api_ms,
            forward_gpu_ms=estimator.last_forward_events[0].elapsed_time(estimator.last_forward_events[1]),
            valid_fraction=float(valid.mean()),
        )
        save(header, row)
        rows.append(row)
        if i % 25 == 0:
            print("STEREO", i, len(pairs), "ms", round(api_ms, 1), flush=True)
    reader.close()
    save(
        OUT / "depth_complete.json",
        dict(
            pairs=len(pairs),
            rows=rows,
            batch_size=1,
            iterations=32,
            peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
        ),
    )


def aligned_samples(depth, streams, uv, point_instances, mask):
    left = streams["infrared_left"]
    k = left["intrinsics"]
    color = streams["color"]["intrinsics"]
    y, x = np.indices(depth.shape)
    points = np.stack(((x - k["ppx"]) / k["fx"] * depth, (y - k["ppy"]) / k["fy"] * depth, depth), -1).reshape(-1, 3)
    ext = left["extrinsics_to_color"]
    R = np.array(ext["rotation"]).reshape(3, 3, order="F")
    points = points @ R.T + np.array(ext["translation"])
    good = (depth.ravel() > 0.15) & (depth.ravel() < 2.5) & np.isfinite(points).all(1) & (points[:, 2] > 0.1)
    projected = np.zeros((len(points), 2), np.float32)
    projected[good] = np.column_stack(
        (
            points[good, 0] / points[good, 2] * color["fx"] + color["ppx"],
            points[good, 1] / points[good, 2] * color["fy"] + color["ppy"],
        )
    )
    q = np.rint(projected).astype(int)
    good &= (q[:, 0] >= 0) & (q[:, 0] < 640) & (q[:, 1] >= 0) & (q[:, 1] < 480)
    buffer = np.full((480, 640), np.inf, np.float32)
    np.minimum.at(buffer, (q[good, 1], q[good, 0]), points[good, 2].astype(np.float32))
    visible = good.copy()
    visible[good] &= points[good, 2] <= buffer[q[good, 1], q[good, 0]] + 0.003
    source_ids = np.zeros(len(points), np.uint8)
    source_ids[visible] = mask[q[visible, 1], q[visible, 0]]
    source_uv = np.column_stack((projected[:, 0], (projected[:, 1] + 0.5) * 448 / 480 - 0.5))
    z = np.full(len(uv), np.nan, np.float32)
    for instance in np.unique(point_instances):
        if not instance:
            continue
        which = np.flatnonzero(visible & (source_ids == instance))
        target = np.flatnonzero(point_instances == instance)
        if not len(which):
            continue
        distance, index = cKDTree(source_uv[which]).query(uv[target], workers=1)
        keep = distance <= 1.5
        z[target[keep]] = points[which[index[keep]], 2]
    return z


def points(calibration, pairs, rgb_ts):
    if (OUT / "points_complete.json").exists():
        return
    k = calibration["streams"]["color"]["intrinsics"]
    K = dict(fx=k["fx"], fy=k["fy"] * 448 / 480, ppx=k["ppx"], ppy=(k["ppy"] + 0.5) * 448 / 480 - 0.5)
    inst = np.load(V0 / "instances.npy", mmap_mode="r")
    by_frame = {}
    for pair in pairs:
        by_frame.setdefault(pair["available_rgb_step"], []).append(pair)
    history = {}
    rows = []
    capture_count = 0
    for frame in range(len(rgb_ts)):
        with np.load(V0 / "pointflow/frames" / f"{frame:06d}.npz") as n:
            data = {key: n[key] for key in ["ids", "uv", "instances", "labels", "colors"]}
        tic = time.perf_counter()
        uv = data["uv"]
        z = np.full(len(uv), np.nan, np.float32)
        provenance = np.zeros(len(uv), np.uint8)
        raw_z = z.copy()
        updates = by_frame.get(frame, [])
        if updates:
            pair = updates[-1]
            assert pair["capture_time"] <= rgb_ts[frame] + 1e-6
            capture_count += 1
            depth = np.load(OUT / "native_depth" / f"{pair['sequence']:06d}.npy")
            raw_z = aligned_samples(depth, calibration["streams"], uv, data["instances"], inst[frame])
            valid = np.isfinite(raw_z) & (raw_z > 0.15) & (raw_z < 2.5)
            z[valid] = raw_z[valid]
            provenance[valid] = 1
            for i in np.flatnonzero(valid):
                history[int(data["ids"][i])] = (float(z[i]), float(rgb_ts[frame]), int(data["instances"][i]))
        else:
            for i, identifier in enumerate(data["ids"]):
                old = history.get(int(identifier))
                if old and rgb_ts[frame] - old[1] <= 0.2 and old[2] == int(data["instances"][i]):
                    z[i] = old[0]
                    provenance[i] = 3
        history = {key: value for key, value in history.items() if rgb_ts[frame] - value[1] <= 0.2}

        def lift(z):
            return np.column_stack(
                ((uv[:, 0] - K["ppx"]) / K["fx"] * z, (uv[:, 1] - K["ppy"]) / K["fy"] * z, z)
            ).astype(np.float32)

        process_ms = (time.perf_counter() - tic) * 1000
        np.savez_compressed(
            OUT / "frames" / f"{frame:06d}.npz",
            **data,
            xyz=lift(z),
            raw_xyz=lift(raw_z),
            valid=provenance > 0,
            provenance=provenance,
            sensor_update=bool(updates),
            timestamp=float(rgb_ts[frame] - rgb_ts[0]),
        )
        rows.append(
            dict(
                frame=frame,
                sensor_update=bool(updates),
                measured=int((provenance == 1).sum()),
                repaired=0,
                interframe=int((provenance == 3).sum()),
                process_ms=process_ms,
                pair_sequence=updates[-1]["sequence"] if updates else None,
            )
        )
        if frame % 200 == 0:
            print("POINTS", frame, len(rgb_ts), flush=True)
    save(OUT / "per_frame.json", rows)
    save(
        OUT / "points_complete.json",
        dict(
            version="foundationstereo",
            frames=len(rgb_ts),
            depth_frames=capture_count,
            real_intrinsics=K,
            da3_used=False,
            uses_future_depth=False,
            depth_source="FoundationStereo ViT-Large on D435 IR",
            xy_from_real_intrinsics=True,
            uv_ids_reused=True,
            no_v3_smoothing=True,
        ),
    )


def html():
    import run_dropper_d435_v2 as export

    export.TASK = "scissors"
    export.VERSION = "foundationstereo"
    export.V0 = V0
    export.OUT = OUT
    legend = '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:335px;font:13px/1.5 system-ui"><b>Scissors · FoundationStereo＋URDF</b><br><b>蓝色：记录关节角驱动的URDF左右手网格表面。橙色：FoundationStereo手点；物体保留RGB。</b><br>左右IR按采集序号/时间戳匹配，32次迭代，真实基线恢复米制深度并标定对齐RGB。不使用DA3，不用未来帧，不叠加V3融合。<br>01：FoundationStereo；02：最近原始深度帧的取样点；03：RGB中间帧沿用历史深度的黄点；04：URDF默认开启；05：旧D435 V2对照。右侧Scene切换。<br>中间帧只沿用同ID最近0.2秒内的深度，不外推速度。无额外旧XYZ展示暂留。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
    export.html(
        comparison=ROOT / "pointflow_outputs/cotracker_real_v2_d435_20261104/scissors",
        show_urdf=True,
        surface_stride=4,
        depth_label="FoundationStereo",
        legend_text=legend,
        display_hold_steps=0,
        highlight_interframe=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "points", "html", "all"])
    args = parser.parse_args()
    cv2.setNumThreads(2)
    episode, calibration, pairs, rgb_ts = prepare()
    if args.stage == "prepare":
        return
    if args.stage in ["infer", "all"]:
        import torch

        torch.manual_seed(0)
        torch.set_num_threads(4)
        estimator = load_estimator(calibration)
        infer(estimator, pairs, episode, calibration)
        del estimator
        torch.cuda.empty_cache()
    if args.stage in ["points", "all"]:
        points(calibration, pairs, rgb_ts)
    if args.stage in ["html", "all"]:
        html()


if __name__ == "__main__":
    main()
