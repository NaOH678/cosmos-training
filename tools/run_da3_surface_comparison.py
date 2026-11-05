"""Recompute paired DA3 geometry on identical hand/object UV support and time it."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base")
EXTERNAL = "/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external"
sys.path.insert(0, EXTERNAL)
sys.path.insert(0, str(BASE / "tools"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--starts", required=True)
args = parser.parse_args()
TARGET = args.out.resolve()
STARTS = [int(value) for value in args.starts.split(",")]
TARGET.mkdir(parents=True, exist_ok=True)
sys.argv = [__file__, str(TARGET), "--low-memory"]
source = (
    (BASE / "tools/track4world_chunk_scale_diagnostic.py")
    .read_text()
    .split("from track4world_scale_fix import ChunkScaleLedger")[0]
)
source = source.replace(
    'torch.cuda.set_per_process_memory_fraction(0.22 if "--low-memory" in sys.argv else 0.80)',
    "torch.cuda.set_per_process_memory_fraction(0.85)",
)
loader_namespace = {"__name__": "__da3_loader__"}
exec(compile(source, "<original_da3_model_loader>", "exec"), loader_namespace)
model = loader_namespace["model"]
EP = loader_namespace["EP"]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import render_fk_vs_pointflow as core  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from addict import Dict  # noqa: E402

ci = core.colour_intrinsics(EP)
K = torch.tensor(
    [
        [ci["fx"] * 630 / 640, 0, (ci["ppx"] + 0.5) * 630 / 640 - 0.5],
        [0, ci["fy"] * 448 / 480, (ci["ppy"] + 0.5) * 448 / 480 - 0.5],
        [0, 0, 1],
    ],
    device="cuda",
)
obs = core.load_pointflow(EP)
mask_root = Path("pointflow_outputs/sandwich_bread_variants_20261030_v2/separate")
object_labels = np.load(mask_root / "labels.npy", mmap_mode="r")
object_instances = np.load(mask_root / "instances.npy", mmap_mode="r")
cap = cv2.VideoCapture(str(core.RAW_ROOT / EP / "videos/head.mp4"))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
yy, xx = torch.meshgrid(torch.arange(448, device="cuda"), torch.arange(630, device="cuda"), indexing="ij")
for start in STARTS:
    end = min(start + 128, T)
    if all((TARGET / f"{mode}_{start:04d}.npz").exists() for mode in ["unconditioned", "conditioned"]):
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    images, hand_uvs, object_uvs = [], [], []
    for frame in range(start, end):
        ok, bgr = cap.read()
        assert ok
        images.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 448)))
        _, uv = core.hand_observations(obs, frame)
        hand_uvs.append(uv[np.linspace(0, len(uv) - 1, min(2500, len(uv))).astype(int)])
        selected = (object_instances[frame] > 0) & (object_labels[frame] != 2)
        samples = []
        for instance in np.unique(object_instances[frame][selected]):
            yy_obj, xx_obj = np.nonzero(selected & (object_instances[frame] == instance))
            take = np.linspace(0, len(xx_obj) - 1, min(900, len(xx_obj))).astype(int)
            samples.append(np.column_stack((xx_obj[take], (yy_obj[take] + 0.5) * 448 / 480 - 0.5)))
        object_uvs.append(np.concatenate(samples) if samples else np.empty((0, 2)))
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
    del images
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[
        None, :, None, None
    ]
    x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
    torch.cuda.synchronize()
    metric_start = time.perf_counter()
    model.da3_metric.cuda()
    metrics = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for offset in range(0, end - start, 4):
            m = model.da3_metric(x[:, offset : offset + 4])
            metrics.append({key: m[key].float().cpu() for key in ["depth", "sky"]})
            del m
    metric = {key: torch.cat([m[key] for m in metrics], dim=1) for key in ["depth", "sky"]}
    del metrics
    model.da3_metric.cpu()
    torch.cuda.synchronize()
    metric_ms = (time.perf_counter() - metric_start) * 1000
    torch.cuda.empty_cache()
    modes = ["unconditioned", "conditioned"]
    if (start // 128) % 2:
        modes.reverse()
    for order, mode in enumerate(modes):
        target = TARGET / f"{mode}_{start:04d}.npz"
        if target.exists():
            continue
        kwargs = {}
        if mode == "conditioned":
            kwargs = dict(
                extrinsics=torch.eye(4, device="cuda")[None, None].expand(1, end - start, 4, 4).clone(),
                intrinsics=K[None, None].expand(1, end - start, 3, 3).clone(),
            )
        print(f"RUN {mode} {start}:{end}", flush=True)
        torch.manual_seed(42)
        torch.cuda.synchronize()
        model_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model.da3(
                x,
                **kwargs,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="saddle_balanced",
            )
            p = Dict({key: output[key].float().clone() for key in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
            del output
        torch.cuda.synchronize()
        anyview_ms = (time.perf_counter() - model_start) * 1000
        post_start = time.perf_counter()
        with torch.inference_mode():
            m = Dict({key: value.to("cuda").clone() for key, value in metric.items()})
            predK = p.intrinsics
            p.intrinsics = K[None, None].expand_as(predK)
            model._apply_metric_scaling(p, m)
            p.intrinsics = predK
            torch.manual_seed(42)
            model._apply_depth_alignment(p, m)
            model._handle_sky_regions(p, m)
            z, k = p.depth[0], predK[0]
            sums = []
            for offset in range(0, end - start, 4):
                kz, zz = k[offset : offset + 4], z[offset : offset + 4]
                raysq = (
                    ((xx - kz[:, 0, 2, None, None]) / kz[:, 0, 0, None, None]) ** 2
                    + ((yy - kz[:, 1, 2, None, None]) / kz[:, 1, 1, None, None]) ** 2
                    + 1
                )
                sums.append((raysq.sqrt() * zz.abs()).sum(dtype=torch.float64))
            norm = float(torch.stack(sums).sum() / z.numel())
            depth = F.interpolate(z[:, None], (448, 640), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            torch.cuda.synchronize()
            post_ms = (time.perf_counter() - post_start) * 1000
            payload = dict(norm=np.array(norm), start=np.array(start), end=np.array(end))
            for i, (hand_uv, object_uv) in enumerate(zip(hand_uvs, object_uvs)):
                for prefix, uv in [("hand", hand_uv), ("object", object_uv)]:
                    sampled = (
                        cv2.remap(
                            depth[i],
                            uv[:, 0].astype("float32")[None],
                            uv[:, 1].astype("float32")[None],
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )[0]
                        if len(uv)
                        else np.empty(0, np.float32)
                    )
                    payload[f"{prefix}_z_{i}"] = sampled
                    payload[f"{prefix}_uv_{i}"] = uv.astype("float32")
            np.savez_compressed(target, **payload)
            timing = dict(
                mode=mode,
                start=start,
                end=end,
                frames=end - start,
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                order=order,
                metric_shared_ms=metric_ms,
                anyview_ms=anyview_ms,
                post_ms=post_ms,
                norm=norm,
                torch_version=torch.__version__,
            )
            target.with_suffix(".json").write_text(json.dumps(timing, indent=2))
            print("DONE", timing, flush=True)
            del p, m, predK, z, k, depth, payload, raysq, zz, kz, sums
        torch.cuda.empty_cache()
    del x, metric
    torch.cuda.empty_cache()
cap.release()
print("COMPLETE", STARTS, flush=True)
