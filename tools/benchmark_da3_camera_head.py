"""Measure camera decoding separately and bypass it in an isolated DA3 A/B run.

Bypass uses cached baseline camera outputs, keeping geometry unchanged. This is
a latency experiment, not a calibrated-camera deployment implementation.
"""

import argparse
import contextlib
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/data/shichaojian/Track4World_portable/Track4World"))
    parser.add_argument("--weights", type=Path, default=Path("/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1"))
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--head-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "track4world/nets/external"))
    sys.path.insert(0, str(args.repo))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from addict import Dict
    from depth_anything_3.api import DepthAnything3

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading local DA3 nested weights", flush=True)
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.weights), local_files_only=True).eval().cuda()
    print(f"Loaded in {time.perf_counter() - started:.2f}s", flush=True)
    camera = model.model.da3
    original = camera._process_camera_estimation
    source = Path(sys.modules[type(camera).__module__].__file__)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        weights=str(args.weights),
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        precision="API default FP16 autocast; camera estimation under its original autocast context",
        scope="DA3 inference_v2, excluding decode/resize/H2D/model loading, no Track4World flow network",
        bypass="reuse baseline camera outputs; no camera prediction, no calibration/geometry change",
        warmup=args.warmup,
        repeats=args.repeats,
        head_repeats=args.head_repeats,
        cases=[],
    )
    video = "/data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649/videos/head.mp4"
    cap = cv2.VideoCapture(video)
    images = []
    for _ in range(max(args.frames)):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError("Failed to read source RGB")
        image = cv2.cvtColor(cv2.resize(image, (640, 448)), cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
    cap.release()
    # Same 1200-token sizing rule as Track4World get_features at 448x640.
    ratio = (1200 * 14**2 / (448 * 640)) ** 0.5
    height, width = int(448 * ratio) // 14 * 14, int(640 * ratio) // 14 * 14
    images = F.interpolate(
        torch.stack(images), (height, width), mode="bilinear", align_corners=False, antialias=True
    ).cuda()

    def stats(values):
        return dict(
            p50_ms=statistics.median(values), mean_ms=statistics.mean(values), p90_ms=float(np.percentile(values, 90))
        )

    def timed(fn, *, reset_seed=False):
        if reset_seed:
            torch.manual_seed(17)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, begin.elapsed_time(end), (time.perf_counter() - wall) * 1000

    with torch.inference_mode():
        for count in args.frames:
            x = images[:count][None]
            print(f"Benchmark shape {tuple(x.shape)}", flush=True)
            captured = {}

            def capture(feats, h, w, output):
                captured["feature"] = feats[-1][1].detach()
                captured["autocast"] = torch.is_autocast_enabled()
                result = original(feats, h, w, output)
                # Nested metric alignment rescales extrinsics later: clone now.
                captured["extrinsics"] = result.extrinsics.detach().clone()
                captured["intrinsics"] = result.intrinsics.detach().clone()
                return result

            camera._process_camera_estimation = capture
            baseline = model.inference_v2(x)
            camera._process_camera_estimation = original
            fake_feats = [(None, captured["feature"])]

            def head():
                ctx = torch.autocast("cuda", dtype=torch.float16) if captured["autocast"] else contextlib.nullcontext()
                with ctx:
                    return original(fake_feats, height, width, Dict(ray=None, ray_conf=None))

            def bypass(feats, h, w, output):
                for key in ("ray", "ray_conf"):
                    if key in output:
                        del output[key]
                output.extrinsics = captured["extrinsics"].clone()
                output.intrinsics = captured["intrinsics"].clone()
                return output

            # Validate that bypass does not change depth or intrinsics for this input.
            torch.manual_seed(17)
            reference = model.inference_v2(x)
            camera._process_camera_estimation = bypass
            torch.manual_seed(17)
            skipped = model.inference_v2(x)
            depth_delta = (reference.depth - skipped.depth).abs().max().item()
            intrinsics_delta = (reference.intrinsics - skipped.intrinsics).abs().max().item()
            if not torch.allclose(reference.depth, skipped.depth, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"Bypass changed depth: {depth_delta}")
            del baseline, reference, skipped
            for _ in range(args.warmup):
                camera._process_camera_estimation = original
                model.inference_v2(x)
                camera._process_camera_estimation = bypass
                model.inference_v2(x)
                head()
            torch.cuda.synchronize()
            head_gpu, head_wall = [], []
            for _ in range(args.head_repeats):
                _, gpu, wall = timed(head)
                head_gpu.append(gpu)
                head_wall.append(wall)
            full_gpu, skip_gpu, full_wall, skip_wall = [], [], [], []
            paired = []
            for i in range(args.repeats):
                values = {}
                order = ("baseline", "bypass") if i % 2 == 0 else ("bypass", "baseline")
                for mode in order:
                    camera._process_camera_estimation = original if mode == "baseline" else bypass
                    output, gpu, wall = timed(lambda: model.inference_v2(x), reset_seed=True)
                    (full_gpu if mode == "baseline" else skip_gpu).append(gpu)
                    (full_wall if mode == "baseline" else skip_wall).append(wall)
                    values[mode] = gpu
                    del output
                paired.append(values["baseline"] - values["bypass"])
            case = dict(
                shape=list(x.shape),
                camera_feature_shape=list(captured["feature"].shape),
                camera_feature_dtype=str(captured["feature"].dtype),
                camera_autocast=captured["autocast"],
                camera_branch_gpu=stats(head_gpu),
                camera_branch_wall=stats(head_wall),
                full_gpu=stats(full_gpu),
                bypass_gpu=stats(skip_gpu),
                full_wall=stats(full_wall),
                bypass_wall=stats(skip_wall),
                paired_saving_gpu=stats(paired),
                raw_gpu=dict(camera=head_gpu, baseline=full_gpu, bypass=skip_gpu, saving=paired),
                depth_max_abs_difference=depth_delta,
                intrinsics_max_abs_difference=intrinsics_delta,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
            )
            camera._process_camera_estimation = original
            report["cases"].append(case)
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in case.items() if k != "raw_gpu"}, indent=2), flush=True)
            captured.clear()
            fake_feats.clear()
    print(f"Results: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
