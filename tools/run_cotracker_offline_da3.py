"""v1 offline control: v0 UV tracks + legacy 128-frame DA3, no D435.

Only HTMLs are delivered. All v0 tracking/masks remain read-only. Geometry uses
legacy predicted focal/last-chunk metric restoration and legacy depth filters.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "pointflow_outputs/cotracker_real_20261104"
OUT = ROOT / "pointflow_outputs/cotracker_real_v1_offline_da3_20261104"
TASKS = ["bottle", "scissors", "dropper", "micro", "sandwich"]
LEGACY = Path("/data/shichaojian/Track4World_portable/Track4World")
BASE = Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(LEGACY / "track4world/nets/external"))
from render_d435_point_accuracy import load_module


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    alias = ROOT / "pointflow_outputs/cotracker_real_v0"
    if not alias.exists():
        alias.symlink_to(V0.name, target_is_directory=True)
    version = dict(
        version="v0",
        source=str(V0),
        description="Original five-task causal single-frame DA3 + CoTracker + hand-derived D435 Z shift",
        html="*/cotracker_da3_interactive.html",
    )
    save(V0 / "VERSION.json", version)
    save(
        OUT / "VERSION.json",
        dict(
            version="v1_offline_da3",
            parent_version="v0",
            parent=str(V0),
            uses_future_frames=True,
            d435_used=False,
            delivered_formats=["html"],
            tracking="reuse exact v0 IDs/UV/masks, no new tracker run",
            da3="Track4World AnyView+Metric, up to128-frame joint context, predicted focal, original last-chunk norm restoration",
            filters="Legacy confidence>2/depth_edge .04/semantic erosion1/range .1..6m; real-observation gap readmission<=3frames with20mm line tolerance;30mm out-and-back despike; detached voxel15mm/minfraction.25/touch2/depthgap100mm per instance",
            display="fixed camera, stride2, last-real-position hold3 display frames; no invented training observations",
        ),
    )
    for task in TASKS:
        dest = OUT / task
        dest.mkdir(exist_ok=True)
        (dest / "chunks").mkdir(exist_ok=True)
        (dest / "frames").mkdir(exist_ok=True)
        for name in ["source.json", "task_config.json", "video.json"]:
            target = dest / name
            if not target.exists():
                target.write_bytes((V0 / task / name).read_bytes())


def load_model():
    import torch
    from compare_depth_sources import load_model as loader
    from track4world_memory import offload_retained_features

    model = loader()
    torch.cuda.set_per_process_memory_fraction(0.92)
    # Residency and tokenwise MLP tiling only: full128 temporal attention retained.
    offload_retained_features(model.da3.backbone.pretrained)
    for module in model.modules():
        if module.__class__.__name__ in ("SwiGLUFFN", "SwiGLUFFNFused", "Mlp"):
            original = module.forward

            def tiled(x, forward=original):
                flat = x.reshape(-1, x.shape[-1])
                output = None
                for start in range(0, len(flat), 2048):
                    part = forward(flat[start : start + 2048])
                    if output is None:
                        output = part.new_empty((len(flat), part.shape[-1]))
                    output[start : start + len(part)] = part
                return output.reshape(*x.shape[:-1], output.shape[-1]) if output is not None else forward(x)

            module.forward = tiled
    model.da3_metric.cpu()
    torch.cuda.empty_cache()
    return model


def infer(task, model, max_chunks=None):
    import torch
    import torch.nn.functional as F
    import utils3d
    from addict import Dict

    dest = OUT / task
    old = V0 / task
    source = json.loads((dest / "source.json").read_text())
    T = source["frames"]
    if (dest / "depth_complete.json").exists():
        return True
    cap = cv2.VideoCapture(str(Path(source["episode"]) / "videos/head.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == T
    headers = []
    processed = 0
    for start in range(0, T, 128):
        end = min(start + 128, T)
        target = dest / "chunks" / f"{start:06d}.npz"
        meta = target.with_suffix(".json")
        if target.exists() and meta.exists():
            headers.append(json.loads(meta.read_text()))
            continue
        if max_chunks is not None and processed >= max_chunks:
            cap.release()
            return False
        processed += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        images = []
        for frame in range(start, end):
            ok, image = cap.read()
            assert ok
            images.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 448)))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tic = time.perf_counter()
        # Original forward_sliding1 divides uint8 RGB in fp16 before fp32 ImageNet norm.
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device="cuda", dtype=torch.float16) / 255.0
        x = (x - x.new_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]) / x.new_tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        )[None, :, None, None]
        x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
        del images
        model.da3_metric.cuda()
        metrics = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for offset in range(0, end - start, 4):
                m = model.da3_metric(x[:, offset : offset + 4])
                metrics.append({k: m[k].float().cpu() for k in ["depth", "sky"]})
                del m
        metric = {k: torch.cat([m[k] for m in metrics], dim=1) for k in ["depth", "sky"]}
        del metrics
        model.da3_metric.cpu()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        print("DA3 START", task, start, end, "context", end - start, flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model.da3(
                x, export_feat_layers=[], infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"
            )
        del x
        with torch.inference_mode():
            p = Dict({k: prediction[k].float().clone() for k in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
            del prediction
            m = Dict({k: v.cuda() for k, v in metric.items()})
            del metric
            # Original nested DA3 uses its own predicted intrinsics for Metric focal conversion.
            model._apply_metric_scaling(p, m)
            model._apply_depth_alignment(p, m)
            model._handle_sky_regions(p, m)
            del m
            yy, xx = torch.meshgrid(torch.arange(448, device="cuda"), torch.arange(630, device="cuda"), indexing="ij")
            z = p.depth[0]
            k = p.intrinsics[0]
            xyz = torch.stack(
                (
                    (xx[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None] * z,
                    (yy[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None] * z,
                    z,
                ),
                dim=1,
            )
            norm = float(torch.linalg.vector_norm(xyz, dim=1).mean())
            del xyz
            zn = z / (norm + 1e-6)
            dense = F.interpolate(zn[:, None], (448, 640), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            conf = (
                F.interpolate(p.depth_conf[0, :, None], (448, 640), mode="bilinear", align_corners=False)[:, 0]
                .cpu()
                .numpy()
            )
            fx_patch = float(k[:, 0, 0].mean())
            del z, k, zn, p
        offsets = [0]
        samples = []
        confidence = []
        clean = []
        uv_hash = hashlib.sha256()
        id_hash = hashlib.sha256()
        for local, frame in enumerate(range(start, end)):
            with np.load(old / "pointflow/frames" / f"{frame:06d}.npz") as n:
                uv = n["uv"]
                ids = n["ids"]
            uv_hash.update(uv.tobytes())
            id_hash.update(ids.tobytes())
            zz = cv2.remap(dense[local], uv[:, 0, None], uv[:, 1, None], cv2.INTER_LINEAR).ravel()
            cc = cv2.remap(conf[local], uv[:, 0, None], uv[:, 1, None], cv2.INTER_LINEAR).ravel()
            q = np.rint(uv).astype(int)
            q[:, 0] = q[:, 0].clip(0, 639)
            q[:, 1] = q[:, 1].clip(0, 447)
            with np.errstate(divide="ignore", invalid="ignore"):
                edge = utils3d.numpy.depth_edge(dense[local], rtol=0.04)
            ok = (cc > 2) & ~edge[q[:, 1], q[:, 0]]
            samples.append(zz)
            confidence.append(cc)
            clean.append(ok)
            offsets.append(offsets[-1] + len(uv))
        np.savez_compressed(
            target,
            offsets=np.array(offsets, np.int64),
            z_normalized=np.concatenate(samples).astype(np.float32),
            confidence=np.concatenate(confidence).astype(np.float32),
            clean=np.concatenate(clean),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        header = dict(
            start=start,
            end=end,
            frames=end - start,
            norm=norm,
            fx_patch=fx_patch,
            seconds=elapsed,
            ms_per_frame=elapsed * 1000 / (end - start),
            peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
            uv_sha256=uv_hash.hexdigest(),
            ids_sha256=id_hash.hexdigest(),
            future_context=True,
            d435_used=False,
        )
        save(meta, header)
        headers.append(header)
        save(dest / "progress.json", dict(stage="DA3", completed=end, total=T, last_chunk=header))
        print(
            "DA3 DONE",
            task,
            start,
            end,
            "seconds",
            round(elapsed, 2),
            "peakGiB",
            round(header["peak_cuda_gib"], 2),
            flush=True,
        )
        del dense, conf, samples, confidence, clean, yy, xx
        torch.cuda.empty_cache()
    cap.release()
    headers.sort(key=lambda h: h["start"])
    assert headers[0]["start"] == 0 and headers[-1]["end"] == T
    save(
        dest / "depth_complete.json",
        dict(
            frames=T,
            chunks=headers,
            last_norm=headers[-1]["norm"],
            last_fx_patch=headers[-1]["fx_patch"],
            inference_seconds=sum(h["seconds"] for h in headers),
            d435_used=False,
        ),
    )
    return True


def temporal_filter(ids, frames, points, valid, basic):
    """Legacy real-observation readmission and isolated out-and-back despike."""
    order = np.lexsort((frames, ids))
    tr = ids[order]
    fr = frames[order]
    p = points[order]
    v = valid[order].copy()
    b = basic[order]
    count = len(order)
    at = np.arange(count, dtype=np.int64)
    previous = np.maximum.accumulate(np.where(v, at, -1))
    following = np.minimum.accumulate(np.where(v, at, count)[::-1])[::-1]
    candidate = np.flatnonzero(~v & b & (previous >= 0) & (following < count))
    left = previous[candidate]
    right = following[candidate]
    same = (tr[left] == tr[candidate]) & (tr[right] == tr[candidate])
    gaps = fr[right] - fr[left]
    same &= (gaps >= 2) & (gaps <= 4)
    alpha = (fr[candidate] - fr[left]) / np.maximum(gaps, 1)
    expected = p[left] + alpha[:, None] * (p[right] - p[left])
    admit = same & (np.linalg.norm(p[candidate] - expected, axis=1) < 0.02)
    v[candidate[admit]] = True
    index = np.flatnonzero(v)
    f = fr[index]
    t = tr[index]
    q = p[index]
    link = (t[1:] == t[:-1]) & (f[1:] == f[:-1] + 1)
    step = np.linalg.norm(q[1:] - q[:-1], axis=1)
    bridge = np.linalg.norm(q[2:] - q[:-2], axis=1)
    bad = link[:-1] & link[1:] & (step[:-1] > 0.03) & (step[1:] > 0.03) & (bridge < 0.03)
    v[index[1:-1][bad]] = False
    output = np.empty_like(valid)
    output[order] = v
    return output, dict(readmitted=int(admit.sum()), despiked=int(bad.sum()))


def build_points(task):

    detached = load_module(
        "legacy_detached", "/data/shichaojian/pf_out/9.24/code/pf_out/drop_detached.py"
    ).detached_mask
    dest = OUT / task
    old = V0 / task
    meta = json.loads((dest / "depth_complete.json").read_text())
    T = meta["frames"]
    if (dest / "points_complete.json").exists():
        return
    instances = np.load(old / "instances.npy", mmap_mode="r")
    labels = np.load(old / "labels.npy", mmap_mode="r")
    # Match infer_pair's common last-chunk predicted focal and centered principal point.
    focal = meta["last_fx_patch"] * np.hypot(640, 448) / np.hypot(630, 448)
    last_norm = np.float32(meta["last_norm"])
    all_ids = []
    all_xyz = []
    all_valid = []
    all_basic = []
    lengths = []
    rows = []
    for header in meta["chunks"]:
        start, end = header["start"], header["end"]
        with np.load(dest / "chunks" / f"{start:06d}.npz") as ch:
            offsets = ch["offsets"]
            zs = ch["z_normalized"]
            cs = ch["confidence"]
            clean = ch["clean"]
            uv_hash = hashlib.sha256()
            id_hash = hashlib.sha256()
            for local, frame in enumerate(range(start, end)):
                with np.load(old / "pointflow/frames" / f"{frame:06d}.npz") as n:
                    data = {k: n[k] for k in ["ids", "uv", "labels", "instances", "colors"]}
                uv = data["uv"]
                uv_hash.update(uv.tobytes())
                id_hash.update(data["ids"].tobytes())
                a, b = map(int, offsets[local : local + 2])
                z = zs[a:b] * last_norm
                xyz = np.column_stack(
                    ((uv[:, 0] + 0.5 - 320) / focal * z, (uv[:, 1] + 0.5 - 224) / focal * z, z)
                ).astype(np.float32)
                canvas = cv2.resize(labels[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
                eroded = np.zeros(canvas.shape, np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                for label in [2, 3]:
                    eroded[cv2.erode((canvas == label).astype(np.uint8), kernel) > 0] = label
                q = np.rint(uv).astype(int)
                q[:, 0] = q[:, 0].clip(0, 639)
                q[:, 1] = q[:, 1].clip(0, 447)
                basic = np.isfinite(xyz).all(1) & (z > 0.1) & (z < 6)
                valid = basic & clean[a:b] & (eroded[q[:, 1], q[:, 0]] == data["labels"])
                all_ids.append(data["ids"])
                all_xyz.append(xyz)
                all_valid.append(valid)
                all_basic.append(basic)
                lengths.append(len(z))
                np.savez_compressed(
                    dest / "frames" / f"{frame:06d}.npz",
                    **data,
                    xyz=xyz,
                    valid=valid,
                    depth_conf=cs[a:b],
                    shift=np.float32(0),
                )
                rows.append(
                    dict(
                        frame=frame,
                        points=len(z),
                        initial_valid=int(valid.sum()),
                        chunk=start,
                        attention_context_last_frame=end - 1,
                        uses_future_through_frame=T - 1,
                    )
                )
            assert uv_hash.hexdigest() == header["uv_sha256"] and id_hash.hexdigest() == header["ids_sha256"]
    ids = np.concatenate(all_ids)
    xyz = np.concatenate(all_xyz)
    valid = np.concatenate(all_valid)
    basic = np.concatenate(all_basic)
    off = np.r_[0, np.cumsum(lengths)]
    fr = np.repeat(np.arange(T, dtype=np.int32), lengths)
    valid, temporal_stats = temporal_filter(ids, fr, xyz, valid, basic)
    removed = 0
    for frame in range(T):
        a, b = map(int, off[frame : frame + 2])
        file = dest / "frames" / f"{frame:06d}.npz"
        with np.load(file) as n:
            data = {k: n[k] for k in n.files}
        current = valid[a:b].copy()
        classes = data["instances"]
        for instance in np.unique(classes[current]):
            selected = np.flatnonzero(current & (classes == instance))
            if len(selected) > 20:
                bad = detached(xyz[a:b][selected], 0.015, 0.25, 2, 0.10)
                current[selected[bad]] = False
                removed += int(bad.sum())
        data["valid"] = current
        np.savez_compressed(file, **data)
        rows[frame]["valid_points"] = int(current.sum())
        if frame % 200 == 0:
            print("FILTER", task, frame, T, flush=True)
    save(dest / "per_frame.json", rows)
    save(
        dest / "points_complete.json",
        dict(
            frames=T,
            legacy_fx_pixel=focal,
            legacy_last_norm=float(last_norm),
            d435_used=False,
            zero_shift_all_frames=True,
            ids_uv_equal_v0=True,
            observations=int(off[-1]),
            valid_observations=sum(r["valid_points"] for r in rows),
            detached_removed=removed,
            **temporal_stats,
        ),
    )


def html(task, show_urdf=False, surface_stride=8):
    import viser
    from render_d435_point_accuracy import Surface

    sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
    from export_efep_official_html import package

    dest = OUT / task
    old = V0 / task
    meta = json.loads((dest / "points_complete.json").read_text())
    T = meta["frames"]
    source = json.loads((dest / "source.json").read_text())
    config = json.loads((dest / "task_config.json").read_text())
    reference = Surface(Path(source["episode"]), config)
    target = dest / f"{task}_v1_offline_da3.html"
    if (dest / "html_complete.json").exists():
        return
    server = viser.ViserServer(host="127.0.0.1", port=9195)
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    empty = np.empty((0, 3), np.float32)
    ec = np.empty((0, 3), np.uint8)
    nodes = {
        name: server.scene.add_point_cloud("/" + name, points=empty, colors=ec, point_size=0.0025, visible=visible)
        for name, visible in [
            ("01_v1_multiframe_DA3", True),
            ("02_v0_singleframe_DA3_no_D435", False),
            ("03_removed_candidates", False),
            ("04_URDF_surface", show_urdf),
        ]
    }
    if show_urdf:
        nodes["04_URDF_surface"].point_size = 0.0013
    server._run_garbage_collector(force=True)
    record = server.get_scene_serializer()
    history = deque(maxlen=4)
    frame_ids = list(range(0, T, 2))
    thumbs = []
    centers = []
    stats = []
    for index, frame in enumerate(frame_ids):
        with np.load(dest / "frames" / f"{frame:06d}.npz") as n:
            data = {k: n[k] for k in n.files}
        with np.load(old / "pointflow/frames" / f"{frame:06d}.npz") as n:
            original = n["raw_xyz"]
            oldvalid = n["valid"]
            assert np.array_equal(data["uv"], n["uv"]) and np.array_equal(data["ids"], n["ids"])
        assert data["shift"] == 0
        valid = data["valid"]
        color = data["colors"].copy()
        color[data["labels"] == 2] = [255, 127, 14]
        history.append((data["ids"][valid], data["xyz"][valid], color[valid]))
        seen = set()
        pp = []
        cc = []
        for identifiers, points, colors in reversed(history):
            keep = np.array([int(i) not in seen for i in identifiers])
            seen.update(map(int, identifiers))
            pp.append(points[keep])
            cc.append(colors[keep])
        shown = np.concatenate(pp)
        shown_color = np.concatenate(cc)
        nodes["01_v1_multiframe_DA3"].points = shown
        nodes["01_v1_multiframe_DA3"].colors = shown_color
        nodes["02_v0_singleframe_DA3_no_D435"].points = original[oldvalid]
        nodes["02_v0_singleframe_DA3_no_D435"].colors = color[oldvalid]
        removed = (~valid) & np.isfinite(data["xyz"]).all(1) & (data["xyz"][:, 2] > 0.1) & (data["xyz"][:, 2] < 6)
        nodes["03_removed_candidates"].points = data["xyz"][removed]
        nodes["03_removed_candidates"].colors = np.tile([255, 40, 40], (int(removed.sum()), 1)).astype(np.uint8)
        surface = reference.at(frame)[::surface_stride].astype(np.float32)
        nodes["04_URDF_surface"].points = surface
        nodes["04_URDF_surface"].colors = np.tile([30, 160, 235], (len(surface), 1)).astype(np.uint8)
        dt = (min(frame + 2, T) - frame) / source["fps"]
        record.insert_sleep(dt)
        image = cv2.imread(str(old / "frames" / f"{frame:05d}.jpg"))
        ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 65])
        assert ok
        thumbs.append(base64.b64encode(jpg).decode())
        stats.append(
            dict(
                frame=frame,
                current_valid=int(valid.sum()),
                shown=len(shown),
                historical_hold=len(shown) - int(valid.sum()),
            )
        )
        if len(shown):
            centers.append(np.median(shown, 0))
        if index % 100 == 0:
            print("HTML", task, index, len(frame_ids), flush=True)
    recording = record.serialize()
    server.stop()
    center = np.median(centers, 0)
    camera = dict(position=(center + [0.55, -0.2, -0.15]).tolist(), look_at=center.tolist(), up=[0, -1, 0])
    duration = T / source["fps"]
    metadata = dict(
        version="v1_offline_da3",
        duration=duration,
        frames=T,
        frame_ids=frame_ids,
        frame_times=[f / source["fps"] for f in frame_ids],
        rgb_thumbnails=thumbs,
        stats=stats,
        d435_used=False,
        urdf=config["urdf"],
        uses_future_frames=True,
        source=str(old),
    )
    legend = (
        '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:330px;font:13px/1.5 system-ui"><b>'
        + task
        + ' · v1 离线多帧DA3</b><br>CoTracker/SAM2与v0相同；DA3使用最多128帧联合上下文，含未来帧。<b>没有D435校正。</b><br>01：多帧DA3；02：v0单帧DA3（未加D435，仅对照）；03：被过滤候选；04：对应URDF手表面。右侧Scene切换。<br>旧版尺度/预测焦距恢复，置信度/深度边缘、短缺口与离群片过滤；展示短缺口保持最后真实位置，最多3个显示步。<br>每2帧显示；全部处理帧与候选点保留。鼠标旋转/缩放，空格暂停，拖动时间轴。<div id="source"></div><img id="rgb" style="width:100%"></aside>'
    )
    behavior = """<script>let paused=false;const m=window.__VISER_EXPORT_METADATA__;setInterval(()=>{const icon=document.querySelector('svg.tabler-icon-player-pause-filled');if(!paused&&icon){icon.closest('button').click();paused=true;}const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-m.duration)<.01);const t=slider?Number(slider.getAttribute('aria-valuenow')):0;let i=0;while(i+1<m.frame_times.length&&m.frame_times[i+1]<=t)i++;const r=m.stats[i];document.getElementById('source').textContent=`源帧 ${m.frame_ids[i]} · 有效${r.current_valid}点 · 显示暂留${r.historical_hold}点`;const im=document.getElementById('rgb');if(im.dataset.i!==String(i)){im.src='data:image/jpeg;base64,'+m.rgb_thumbnails[i];im.dataset.i=String(i);}},150);</script>"""
    if show_urdf:
        legend = legend.replace(
            "</b><br>",
            "</b><br><b>蓝色：记录关节角＋URDF网格算出的左右手表面；橙色：深度手点；物体保留RGB颜色。URDF默认开启，右侧Scene眼睛可分别开关。</b><br>",
            1,
        )
        metadata.update(
            urdf_visible_by_default=True, urdf_surface_stride=surface_stride, urdf_surface_point_size_m=0.0013
        )
    temp = target.with_suffix(".html.tmp")
    temp.write_text(
        package(recording, metadata, camera, task + " v1 offline DA3", legend).replace("</body>", behavior + "</body>")
    )
    temp.replace(target)
    save(
        dest / "html_complete.json",
        dict(
            path=str(target),
            frames=T,
            display_frames=len(frame_ids),
            html_bytes=target.stat().st_size,
            d435_used=False,
            urdf=config["urdf"],
            duration_seconds=duration,
        ),
    )
    print("HTML COMPLETE", task, target, flush=True)


def post(task):
    for stage in ["filter", "html"]:
        with (OUT / task / f"{stage}.log").open("a") as log:
            subprocess.run(
                [sys.executable, "-u", __file__, stage, "--task", task],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "infer", "filter", "html", "all"])
    p.add_argument("--task", choices=TASKS + ["all", "remaining"], default="all")
    p.add_argument("--max-chunks", type=int)
    args = p.parse_args()
    prepare()
    tasks = (
        TASKS
        if args.task == "all"
        else [t for t in TASKS if t != "dropper"]
        if args.task == "remaining"
        else [args.task]
    )
    if args.stage == "prepare":
        return
    if args.stage in ["infer", "all"]:
        import torch

        torch.manual_seed(42)
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        model = load_model()
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = []
            for task in tasks:
                complete = infer(task, model, args.max_chunks)
                if complete and args.stage == "all":
                    futures.append(pool.submit(post, task))
            for future in futures:
                future.result()
    elif args.stage == "filter":
        for task in tasks:
            build_points(task)
    elif args.stage == "html":
        for task in tasks:
            html(task)


if __name__ == "__main__":
    main()
