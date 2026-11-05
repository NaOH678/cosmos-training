"""Fast-FoundationStereo on recorded D435 IR, with unchanged causal V4 point rules."""

import argparse
import hashlib
import json
import pickle
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/Fast-FoundationStereo"
WEIGHTS = ROOT / "checkpoints/Fast-FoundationStereo/23-36-37/model_best_bp2_serialize.pth"
BASE = ROOT / "pointflow_outputs/fast_foundationstereo_v4_20261105"
for p in [ROOT / "third_party/foundation_runtime", ROOT / "third_party/fast_foundation_runtime", REPO]:
    sys.path.insert(0, str(p))
import run_dropper_d435_v2 as pipeline
from run_fs_static_v4 import NoObjectExtrapolation, StaticFusion, core
from run_scissors_foundationstereo import PairReader


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def match_stereo(left, right, rgb_times, tolerance=0.001):
    """Camera frame counters may have different offsets; match capture timestamps."""
    pairs = []
    used = set()
    for i, ts in enumerate(left["timestamps"]):
        k = int(np.searchsorted(right["timestamps"], ts))
        options = [j for j in [k - 1, k] if 0 <= j < len(right["timestamps"]) and j not in used]
        if not options:
            continue
        j = min(options, key=lambda x: abs(right["timestamps"][x] - ts))
        if abs(float(right["timestamps"][j] - ts)) > tolerance:
            continue
        capture = max(float(ts), float(right["timestamps"][j]))
        step = max(
            int(left["training_steps"][i]), int(right["training_steps"][j]), int(np.searchsorted(rgb_times, capture))
        )
        if step >= len(rgb_times):
            continue
        used.add(j)
        pairs.append(
            dict(
                sequence=int(left["sequences"][i]),
                right_sequence=int(right["sequences"][j]),
                left_video_index=i,
                right_video_index=j,
                left_time=float(ts),
                right_time=float(right["timestamps"][j]),
                capture_time=capture,
                available_rgb_step=step,
                rgb_time=float(rgb_times[step]),
            )
        )
    return pairs


def prepare(task):
    out = BASE / task
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_depth").mkdir(exist_ok=True)
    source_dir = ROOT / "pointflow_outputs/cotracker_real_20261104" / task
    source = json.loads((source_dir / "source.json").read_text())
    episode = Path(source["episode"])
    streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    left, right = streams["infrared_left"], streams["infrared_right"]
    assert left["intrinsics"] == right["intrinsics"] and not np.any(left["intrinsics"]["coeffs"])
    R = np.array(left["extrinsics_to_color"]["rotation"]).reshape(3, 3, order="F")
    Rr = np.array(right["extrinsics_to_color"]["rotation"]).reshape(3, 3, order="F")
    np.testing.assert_allclose(R, Rr, atol=1e-8)
    b = R.T @ (
        np.array(right["extrinsics_to_color"]["translation"]) - np.array(left["extrinsics_to_color"]["translation"])
    )
    assert b[0] > 0 and np.linalg.norm(b[1:]) < 1e-6
    assert streams["depth"]["intrinsics"] == left["intrinsics"]
    assert streams["depth"]["extrinsics_to_color"] == left["extrinsics_to_color"]
    sync = json.loads((episode / "sync_timestamps.json").read_text())
    rgb = np.array([r["camera_head"] for r in sync])
    env = lmdb.open(str(episode / "auxiliary_camera/depth.lmdb"), readonly=True, lock=False)
    with env.begin() as t:
        index = {
            side: {
                key: pickle.loads(t.get(f"index/video/head_ir_{side}/{key}".encode()))
                for key in ["sequences", "timestamps", "training_steps"]
            }
            for side in ["left", "right"]
        }
    env.close()
    pairs = match_stereo(index["left"], index["right"], rgb)
    assert len({x["sequence"] for x in pairs}) == len(pairs)
    for x in pairs:
        assert x["capture_time"] <= rgb[x["available_rgb_step"]] + 1e-6
    calibration = dict(
        streams=streams,
        baseline_m=float(b[0]),
        fx=float(left["intrinsics"]["fx"]),
        pairs=len(pairs),
        pairing="nearest capture timestamp within 1ms, unique right frame; frame-counter offset allowed",
        max_pair_dt_ms=max(abs(x["left_time"] - x["right_time"]) * 1000 for x in pairs),
    )
    save(out / "calibration.json", calibration)
    save(out / "ir_pair_index.json", pairs)
    for name in ["source.json", "task_config.json", "video.json"]:
        (out / name).write_bytes((source_dir / name).read_bytes())
    print(
        "PREPARE", task, source["frames"], len(pairs), "pairs", "max_dt_ms", calibration["max_pair_dt_ms"], flush=True
    )
    return episode, calibration, pairs


class FastDepth:
    def __init__(self):
        import torch
        from core.utils.utils import InputPadder

        self.torch = torch
        self.Padder = InputPadder
        torch.manual_seed(0)
        torch.set_num_threads(4)
        # Official research checkpoint is a serialized complete model from its linked Google Drive.
        self.model = torch.load(WEIGHTS, map_location="cpu", weights_only=False).cuda().eval()
        self.model.args.valid_iters = 8
        self.model.args.max_disp = 192

    def infer(self, left, right, calibration):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        images = [
            torch.as_tensor(np.repeat(a[..., None], 3, 2)).cuda().float()[None].permute(0, 3, 1, 2).contiguous()
            for a in [left, right]
        ]
        padder = self.Padder(images[0].shape, divis_by=32, force_square=False)
        images = padder.pad(*images)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            begin.record()
            disp = self.model.forward(*images, iters=8, test_mode=True, optimize_build_volume="pytorch1")
            end.record()
            disp = padder.unpad(disp.float()).squeeze()
            xx = torch.arange(disp.shape[1], device=disp.device)[None, :]
            valid = torch.isfinite(disp) & (disp > 0) & ((xx - disp) >= 0)
            depth = torch.where(valid, calibration["fx"] * calibration["baseline_m"] / disp, 0)
            depth = torch.where(torch.isfinite(depth) & (depth >= 0.15) & (depth <= 2.5), depth, 0)
            depth = depth.cpu().numpy().astype(np.float32)
        torch.cuda.synchronize()
        return depth, dict(api_ms=(time.perf_counter() - start) * 1000, forward_gpu_ms=begin.elapsed_time(end))


def infer(task, backend, max_pairs=None):
    episode, cal, pairs = prepare(task)
    out = BASE / task
    if (out / "depth_complete.json").exists():
        return
    reader = PairReader(episode)
    left, right = reader.get(pairs[0])
    if not (out / "benchmark.json").exists():
        for _ in range(10):
            backend.infer(left, right, cal)
        samples = []
        for _ in range(30):
            _, sample = backend.infer(left, right, cal)
            samples.append(sample)
        torch = backend.torch
        gpu_state = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True
        )
        save(
            out / "benchmark.json",
            dict(
                host=socket.gethostname(),
                gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                input_hw=[480, 640],
                iterations=8,
                max_disp=192,
                precision="FP16",
                batch_size=1,
                backend="official PyTorch forward, pytorch1 cost volume, built-in torch.compile",
                warmup=10,
                repeats=30,
                same_pair=True,
                samples=samples,
                api_p50_ms=float(np.median([s["api_ms"] for s in samples])),
                api_p95_ms=float(np.percentile([s["api_ms"] for s in samples], 95)),
                forward_p50_ms=float(np.median([s["forward_gpu_ms"] for s in samples])),
                forward_p95_ms=float(np.percentile([s["forward_gpu_ms"] for s in samples], 95)),
                gpu_state=gpu_state,
                excludes="IR video decode, V4 postprocessing, SAM2, CoTracker, initialization/JIT warmup, output IO",
            ),
        )
    reader.close()
    reader = PairReader(episode)

    def write(path, depth, header):
        np.save(path, depth)
        save(path.with_suffix(".json"), header)

    futures = []
    with ThreadPoolExecutor(max_workers=2) as writer:
        for i, pair in enumerate(pairs[:max_pairs]):
            path = out / "native_depth" / f"{pair['sequence']:06d}.npy"
            if path.exists() and path.with_suffix(".json").exists():
                continue
            tic = time.perf_counter()
            left, right = reader.get(pair)
            read_ms = (time.perf_counter() - tic) * 1000
            depth, header = backend.infer(left, right, cal)
            header.update(sequence=pair["sequence"], read_ms=read_ms, valid_fraction=float((depth > 0).mean()))
            futures.append(writer.submit(write, path, depth, header))
            if len(futures) > 8:
                futures.pop(0).result()
            if i % 50 == 0:
                print("FAST DEPTH", task, i, len(pairs), round(header["api_ms"], 2), "ms", flush=True)
        for future in futures:
            future.result()
    reader.close()
    if max_pairs is not None and max_pairs < len(pairs):
        return
    rows = [json.loads((out / "native_depth" / f"{x['sequence']:06d}.json").read_text()) for x in pairs]
    save(
        out / "depth_complete.json",
        dict(
            pairs=len(pairs),
            rows=rows,
            host=socket.gethostname(),
            api_p50_ms=float(np.median([r["api_ms"] for r in rows])),
            api_p95_ms=float(np.percentile([r["api_ms"] for r in rows], 95)),
        ),
    )


def post(task):
    out = BASE / task
    cal = json.loads((out / "calibration.json").read_text())
    pairs = json.loads((out / "ir_pair_index.json").read_text())
    assert (out / "depth_complete.json").exists()
    by_frame = {x["available_rgb_step"]: x for x in pairs}

    def provider(frame):
        pair = by_frame.get(frame)
        return (
            np.load(out / "native_depth" / f"{pair['sequence']:06d}.npy") / cal["streams"]["depth"]["depth_scale_m"]
            if pair
            else None
        )

    pipeline.TASK = task
    pipeline.VERSION = "fast_foundationstereo_v4"
    pipeline.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104" / task
    pipeline.OUT = out
    pipeline.build(
        enhancement=StaticFusion(),
        native_provider=provider,
        sample_depth=core.sample_continuous,
        stabilizer_factory=NoObjectExtrapolation,
    )
    comparison = (
        ROOT / "pointflow_outputs/scissors_foundationstereo_static_v4_20261105"
        if task == "scissors"
        else ROOT / "pointflow_outputs/cotracker_real_v2_d435_20261104/bottle"
    )
    comparison_name = "05_original_FoundationStereo_V4" if task == "scissors" else "05_previous_D435_V2"
    description = (
        "05：旧FoundationStereo V4（未含保护器修复，仅供参考）"
        if task == "scissors"
        else "05：之前D435 V2（不同后处理，仅供参考）"
    )
    legend = f'<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:350px;font:13px/1.5 system-ui"><b>{task} · Fast-FoundationStereo＋V4</b><br>23-36-37权重，8次迭代，原始640×480 IR输入；按左右采集时间戳匹配。<br>保留V4连续同层取样、静止迟滞、按轨迹ID融合和物体中间帧短时Z沿用。过滤只用当前/历史帧。<br>01：新版；02：原始fast深度点；03：历史补点；04：真实URDF；{description}。<br>蓝色URDF、橙色手点、物体RGB。无额外显示旧XYZ暂留。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
    pipeline.html(
        comparison=comparison if comparison.exists() else None,
        comparison_label=comparison_name,
        show_urdf=True,
        surface_stride=4,
        depth_label="FastFoundationStereo",
        legend_text=legend,
        display_hold_steps=0,
    )
    save(
        out / "method.json",
        dict(
            model="Fast-FoundationStereo",
            checkpoint="23-36-37",
            iterations=8,
            max_disp=192,
            V4_preserved=True,
            object_guard_fix="disable scalar velocity; reacquire stale guard after two consistent observations",
            uses_future_depth=False,
            sdk_depth_used=False,
            upstream_masks_tracks_reused=True,
            host=socket.gethostname(),
            urdf=json.loads((out / "task_config.json").read_text())["urdf"],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "infer", "post", "all"])
    parser.add_argument("--tasks", nargs="+", default=["scissors", "bottle"], choices=["scissors", "bottle"])
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    BASE.mkdir(exist_ok=True)
    if args.stage == "prepare":
        for task in args.tasks:
            prepare(task)
        return
    if args.stage in ["infer", "all"]:
        import torch

        if socket.gethostname() != "dev-fb1f64cc-0":
            raise RuntimeError("GPU inference is authorized only on 10.119.233.102 dev-fb1f64cc-0")
        backend = FastDepth()
        for task in args.tasks:
            infer(task, backend, args.max_pairs)
        del backend
        torch.cuda.empty_cache()
    if args.stage in ["post", "all"]:
        for task in args.tasks:
            post(task)
    save(
        BASE / "provenance.json",
        dict(
            repository="https://github.com/NVlabs/Fast-FoundationStereo",
            commit=(REPO / ".git" / (REPO / ".git/HEAD").read_text().strip().removeprefix("ref: ")).read_text().strip(),
            checkpoint_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            official_weights_file_id="1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E",
            host=socket.gethostname(),
        ),
    )


if __name__ == "__main__":
    main()
