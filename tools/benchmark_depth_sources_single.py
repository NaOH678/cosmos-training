"""Standalone single-frame B/C timing after bulk experiment processes exit."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from compare_depth_sources import TASKS, load_model


def main():
    root = Path("pointflow_outputs/depth_source_comparison_20261103")
    while not all((root / task / "summary.json").exists() for task in TASKS):
        time.sleep(5)
    # Bulk writers create their summary just before exiting; wait for cleanup.
    time.sleep(3)
    model = load_model()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    report = dict(
        gpu=torch.cuda.get_device_name(),
        single_frame=True,
        cpu_offload=False,
        warmup_each_task=5,
        repeats_each_image=2,
        precision="FP16; FP32 alignment",
        scope="Metric + AnyView + real-focal alignment + resize; excludes RGB decode/normalize/H2D, SAM2, point extraction and selection",
        tasks={},
    )
    for task, (episode, _) in TASKS.items():
        raw = Path("/data/shichaojian/raw_data") / episode
        intr = json.loads((raw / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"][
            "streams"
        ]["color"]["intrinsics"]
        K = torch.tensor(
            [
                [intr["fx"] * 630 / 640, 0, (intr["ppx"] + 0.5) * 630 / 640 - 0.5],
                [0, intr["fy"] * 448 / 480, (intr["ppy"] + 0.5) * 448 / 480 - 0.5],
                [0, 0, 1],
            ],
            device="cuda",
        )
        kwargs = dict(intrinsics=K[None, None], extrinsics=torch.eye(4, device="cuda")[None, None])
        summary = json.loads((root / task / "summary.json").read_text())
        frames = summary["metadata"]["frame_ids"]
        selected = np.unique(np.array(frames)[np.linspace(0, len(frames) - 1, 12).astype(int)])
        cap = cv2.VideoCapture(str(raw / "videos/head.mp4"))
        rows = []
        for index, frame in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            assert ok
            x = (
                torch.from_numpy(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 448)))
                .permute(2, 0, 1)[None]
                .float()
                .cuda()
                / 255
            )
            x = F.interpolate((x - mean) / std, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]

            def run(mode):
                torch.manual_seed(42)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    metric = model.da3_metric(x)
                    output = model.da3(
                        x,
                        **(kwargs if mode == "C" else {}),
                        export_feat_layers=[],
                        infer_gs=False,
                        use_ray_pose=False,
                        ref_view_strategy="saddle_balanced",
                    )
                with torch.inference_mode():
                    p = Dict({k: output[k].float() for k in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
                    m = Dict({k: metric[k].float() for k in ["depth", "sky"]})
                    pred = p.intrinsics
                    p.intrinsics = K[None, None]
                    model._apply_metric_scaling(p, m)
                    p.intrinsics = pred
                    model._apply_depth_alignment(p, m)
                    model._handle_sky_regions(p, m)
                    return F.interpolate(p.depth[0, None], (480, 640), mode="bilinear", align_corners=False)

            if index == 0:
                for _ in range(5):
                    run("B")
                    run("C")
            for repeat in range(2):
                for mode in ["B", "C"] if (index + repeat) % 2 == 0 else ["C", "B"]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    z = run(mode)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    rows.append(dict(frame=int(frame), mode=mode, repeat=repeat, milliseconds=elapsed))
                    del z
        cap.release()
        result = {
            m: dict(
                p50_ms=float(np.median([r["milliseconds"] for r in rows if r["mode"] == m])),
                p95_ms=float(np.percentile([r["milliseconds"] for r in rows if r["mode"] == m], 95)),
                mean_ms=float(np.mean([r["milliseconds"] for r in rows if r["mode"] == m])),
            )
            for m in ["B", "C"]
        }
        report["tasks"][task] = dict(summary=result, rows=rows)
        (root / "standalone_single_frame_timing.json").write_text(json.dumps(report, indent=2))
        print(task, result, flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
