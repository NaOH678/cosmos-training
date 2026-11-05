"""Compare short-window pair extraction with pretrained-window pair extraction."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/co-tracker"))
from cotracker.predictor import CoTrackerOnlinePredictor

out = ROOT / "pointflow_outputs/cotracker_real_20261104"
task = out / "bottle"
torch.set_num_threads(4)
torch.manual_seed(0)
model = (
    CoTrackerOnlinePredictor(checkpoint=str(ROOT / "third_party/co-tracker/scaled_online.pth"), v2=False, window_len=16)
    .cuda()
    .eval()
)
embedding = model.model.time_emb.clone()
rows = []
for frame in [1, 150, 300, 450, 600, 700]:
    images = [
        cv2.cvtColor(cv2.resize(cv2.imread(str(task / "frames" / f"{i:05d}.jpg")), (640, 448)), cv2.COLOR_BGR2RGB)
        for i in [frame - 1, frame]
    ]
    with np.load(task / "pointflow/frames" / f"{frame - 1:06d}.npz") as n:
        uv = n["uv"][::2]
    video = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)[None].float().cuda()
    q = torch.from_numpy(np.c_[np.zeros(len(uv)), uv]).float()[None].cuda()
    results = {}
    for size in [16, 2]:
        model.model.window_len = size
        model.model.time_emb = embedding[:, :size].clone()
        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(video_chunk=video[:, :1], is_first_step=True, queries=q, add_support_grid=True)
            p, v = model(video_chunk=video, add_support_grid=True)
        torch.cuda.synchronize()
        results[size] = (p[0, 1].float().cpu().numpy(), v[0, 1].cpu().numpy(), (time.perf_counter() - tic) * 1000)
    error = np.linalg.norm(results[2][0] - results[16][0], axis=1)
    both = results[2][1] & results[16][1]
    row = dict(
        frame=frame,
        points=len(uv),
        median_px=float(np.median(error[both])),
        p95_px=float(np.percentile(error[both], 95)),
        max_px=float(error[both].max()),
        visibility_agreement=float(np.mean(results[2][1] == results[16][1])),
        window16_ms=results[16][2],
        window2_ms=results[2][2],
    )
    rows.append(row)
    print(row, flush=True)
(out / "context_audit.json").write_text(json.dumps(rows, indent=2))
