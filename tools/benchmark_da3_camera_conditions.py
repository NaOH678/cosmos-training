"""Warm, same-A800 comparison of camera-conditioned and unconditioned DA3."""

import json
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

root = Path("pointflow_outputs/da3_urdf_two_conditions_20261103")
source = Path("tools/run_da3_surface_comparison.py").read_text().split("obs = core.load_pointflow(EP)")[0]
sys.argv = [__file__, "--out", str(root / "benchmark"), "--starts", "0"]
namespace = {"__name__": "__benchmark_loader__", "__file__": str(Path("tools/run_da3_surface_comparison.py").resolve())}
exec(compile(source, "<paired_model_loader>", "exec"), namespace)
model, K, core, EP, np = (namespace[key] for key in ["model", "K", "core", "EP", "np"])
cap = cv2.VideoCapture(str(core.RAW_ROOT / EP / "videos/head.mp4"))
images = []
for _ in range(128):
    ok, bgr = cap.read()
    assert ok
    images.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 448)))
cap.release()
x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[
    None, :, None, None
]
x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
kwargs = dict(
    intrinsics=K[None, None].expand(1, 128, 3, 3).clone(),
    extrinsics=torch.eye(4, device="cuda")[None, None].expand(1, 128, 4, 4).clone(),
)
rows = []
for index, mode in enumerate(
    ["unconditioned", "conditioned", "unconditioned", "conditioned", "conditioned", "unconditioned"]
):
    torch.manual_seed(42)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model.da3(
            x,
            **(kwargs if mode == "conditioned" else {}),
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del output
    torch.cuda.empty_cache()
    rows.append(dict(mode=mode, warmup=index < 2, seconds_per_128=elapsed, ms_per_frame=elapsed / 128 * 1000))
    print(rows[-1], flush=True)
summary = {
    mode: dict(
        mean_ms_per_frame=float(
            np.mean([row["ms_per_frame"] for row in rows if not row["warmup"] and row["mode"] == mode])
        ),
        samples_ms_per_frame=[row["ms_per_frame"] for row in rows if not row["warmup"] and row["mode"] == mode],
    )
    for mode in ["unconditioned", "conditioned"]
}
(root / "warm_same_gpu_benchmark.json").write_text(
    json.dumps(
        dict(
            summary=summary,
            rows=rows,
            device=torch.cuda.get_device_name(),
            frames=128,
            metric_branch_included=False,
            cpu_offload=True,
        ),
        indent=2,
    )
)
print("COMPLETE", summary, flush=True)
