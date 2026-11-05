"""Track4World-style standalone Viser playback with switchable D435 filter layers."""

import argparse
import base64
import gzip
import json
import os
import sys
import time
from pathlib import Path

import cv2
import lmdb
import msgspec
import numpy as np
from render_d435_point_accuracy import CYAN, TASKS, Surface, load_module, measured_cloud, visible_surface
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
from export_efep_official_html import package  # noqa: E402

LAYERS = {
    "01_URDF_surface": ("URDF 手表面参考", True),
    "02_spatial_hand": ("新版三维过滤：手", True),
    "03_spatial_objects": ("新版三维过滤：物体", True),
    "04_raw_hand": ("原始有效测量：手", False),
    "05_raw_objects": ("原始有效测量：物体", False),
    "06_previous_local_filter": ("之前的局部过滤：手＋物体", False),
    "07_removed_candidates": ("新版额外剔除候选（红）", False),
    "08_URDF_prior_hand": ("可选 URDF 先验手：不用于精度评估", False),
}


def spatial_filter(points, ids, previous, detached):
    """Per-instance SOR and small, disconnected, depth-separated components."""
    result = previous.copy()
    counts = dict(statistical=0, detached=0)
    for instance in np.unique(ids[previous]):
        selected = np.flatnonzero(previous & (ids == instance))
        if len(selected) < 32:
            continue
        p = points[selected]
        distances = cKDTree(p).query(p, k=min(17, len(p)), workers=1)[0][:, 1:].mean(1)
        # A global mean/std alone flags ordinary sparse silhouettes as outliers.
        # Keep locally supported metric surfaces even if they are sparser than
        # the flat interior; reserve SOR for genuinely isolated measurements.
        threshold = max(float(distances.mean() + 2.0 * distances.std()), 0.008)
        keep = distances <= threshold
        result[selected[~keep]] = False
        counts["statistical"] += int((~keep).sum())
        remaining = selected[keep]
        if len(remaining) < 32:
            continue
        removed = detached.detached_mask(points[remaining], voxel=0.01, min_frac=0.15, touch=2, depth_gap=0.06)
        result[remaining[removed]] = False
        counts["detached"] += int(removed.sum())
    return result, counts


def thin(points, colors, ids, cap=450):
    selected = []
    for value in np.unique(ids):
        index = np.flatnonzero(ids == value)
        selected.extend(index[np.linspace(0, len(index) - 1, min(len(index), cap), dtype=int)])
    return points[selected].astype(np.float32), colors[selected]


def interface(meta):
    task = meta["task"]
    mapping = "".join(f"<div><code>{name}</code>：{text}</div>" for name, (text, _) in LAYERS.items())
    controls = f"""<aside id="depth-audit" style="position:fixed;left:12px;top:12px;z-index:9999;width:305px;max-height:88vh;overflow:auto;padding:12px;background:#15202be8;color:#eee;font:13px/1.55 system-ui;border:1px solid #687888;border-radius:8px">
<b>{task} · D435＋SAM2-large＋URDF</b><br><span id="source-frame">加载中</span><br>
相机不自动旋转；拖动旋转，滚轮缩放。<br>下方时间轴／空格暂停；右侧 Scene 的眼睛切换图层。
<div id="point-counts" style="color:#bde3d7;margin-top:6px"></div>
<details open><summary>对应 RGB＋mask</summary><img id="source-rgb" style="width:100%;margin-top:5px" /></details>
<details><summary>图层说明／过滤方法</summary>{mapping}
<p>新版＝原局部过滤＋按实例的三维统计离群过滤＋小型远离连通分量过滤。红点只是剔除候选，仍需检查有无误删细结构。</p>
<p>08 层额外使用距离 URDF 表面 30mm 的先验，会倾向参考形状，不能用来证明 D435 变准。</p>
<p>URDF不是实际手套GT；物体无三维GT。未运行点追踪、未补洞或平滑轨迹。</p></details></aside>"""
    behavior = """<script>
window.__D435_UI_READY__=false;
let autoPaused=false;
const metadata=window.__VISER_EXPORT_METADATA__;
function refreshDepthAudit(){
  const pause=document.querySelector('svg.tabler-icon-player-pause-filled');
  if(!autoPaused && pause){pause.closest('button').click();autoPaused=true;window.__D435_UI_READY__=true;}
  const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-metadata.duration)<.01);
  const t=slider?Number(slider.getAttribute('aria-valuenow')):0;
  let i=0;while(i+1<metadata.frame_times.length && metadata.frame_times[i+1]<=t+1e-5)i++;
  document.getElementById('source-frame').textContent=`源帧 ${metadata.frame_ids[i]} · ${t.toFixed(2)}秒 · ${i+1}/${metadata.frame_ids.length}`;
  const r=metadata.stats[i];
  document.getElementById('point-counts').textContent=`原始 ${r.raw} → 旧过滤 ${r.previous} → 新过滤 ${r.spatial} 点；新增剔除 ${r.extra_removed} 点`;
  const image=document.getElementById('source-rgb');if(image.dataset.frame!==String(i)){image.src='data:image/jpeg;base64,'+metadata.rgb_thumbnails[i];image.dataset.frame=String(i);}
  window.__D435_CURRENT_SOURCE_FRAME__=metadata.frame_ids[i];
}
setInterval(refreshDepthAudit,150);
</script>"""
    return controls, behavior


def main(args):
    import viser

    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=False)
    episode_name, mask_path = TASKS[args.task]
    episode = Path("/data/shichaojian/raw_data") / episode_name
    masks = Path(mask_path)
    info = json.loads((masks / "initialization.json").read_text())
    recipe = (
        json.loads((masks / "task_config.json").read_text())
        if (masks / "task_config.json").exists()
        else info["config"]
    )
    streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    causal = None
    if args.causal:
        module = load_module(
            "causal_depth_filter",
            str(Path(__file__).resolve().parents[1] / "cosmos_framework/inference/robot_policy/causal_depth_filter.py"),
        )
        causal = module.CausalDepthFilter(streams["color"]["intrinsics"])
        LAYERS["02_spatial_hand"] = ("之前的单帧三维过滤：手", False)
        LAYERS["03_spatial_objects"] = ("之前的单帧三维过滤：物体", False)
        LAYERS.update(
            {
                "09_causal_hand": ("因果跨帧过滤后的实测手点（橙）", True),
                "10_causal_objects": ("因果跨帧过滤后的实测物体点", True),
                "11_temporal_removed": ("跨帧检查额外剔除（紫红）", False),
                "12_one_frame_repairs": ("仅一帧的历史补点（黄）；非实测", True),
            }
        )
    labels = np.load(masks / "labels.npy", mmap_mode="r")
    instances = np.load(masks / "instances.npy", mmap_mode="r")
    reference = Surface(episode, recipe)
    detached = load_module("depth_detached", "/data/shichaojian/pf_out/9.24/code/pf_out/drop_detached.py")
    local = Path("/tmp/d435_point_accuracy_20261102") / args.task
    assert (local / "data.mdb").exists(), "Run the point-accuracy export first to cache the raw LMDB"
    env = lmdb.open(str(local), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        )
    frames = [f for f in frames if f < len(labels)]
    if args.limit:
        frames = frames[: args.limit]
    sync = json.loads((episode / "sync_timestamps.json").read_text())
    times = np.array([sync[f].get("camera_head", sync[f]["anchor"]) for f in frames])
    times -= times[0]
    assert np.all(np.diff(times) > 0)
    capture = cv2.VideoCapture(str(episode / "videos/head.mp4"))
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.gui.configure_theme(dark_mode=True, control_width="small", show_logo=False, show_share_button=False)
    empty = np.empty((0, 3), np.float32)
    nodes = {
        name: server.scene.add_point_cloud(
            "/" + name,
            points=empty,
            colors=np.empty((0, 3), np.uint8),
            point_size=0.002,
            point_shape="rounded",
            visible=visible,
        )
        for name, (_, visible) in LAYERS.items()
    }
    previous_summary = json.loads(
        (Path("pointflow_outputs/d435_point_accuracy_20261102") / args.task / "summary.json").read_text()
    )
    center = np.array(previous_summary["bounds"]["center"])[[0, 2, 1]] * [1, -1, 1]
    server.scene.add_grid(
        "/00_reference_grid_10cm",
        width=1.2,
        height=1.2,
        plane="xz",
        cell_size=0.1,
        section_size=0.5,
        position=center + [0, 0.3, 0],
    )
    server._run_garbage_collector(force=True)
    serializer = server.get_scene_serializer()
    rows, thumbs, per_instance = [], [], {}
    for index, frame in enumerate(frames):
        frame_start = time.perf_counter()
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{frame:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        assert ok
        points, colors, classes, ids, raw, previous, uv = measured_cloud(
            depth, streams, instances[frame], labels[frame], image, args.device
        )
        cleaned, removed = spatial_filter(points, ids, previous, detached)
        causal_stats = {}
        if causal is not None:
            accepted, repairs, repair_ids, repair_uv, causal_stats = causal.update(
                image, instances[frame], points, uv, ids, cleaned, float(times[index])
            )
        hand = classes == 2
        full_surface = reference.at(frame)
        visible = visible_surface(full_surface, streams["color"]["intrinsics"])
        visible = visible[np.linspace(0, len(visible) - 1, min(4500, len(visible)), dtype=int)]
        distance = np.full(len(points), np.inf)
        distance[previous & hand] = cKDTree(full_surface).query(points[previous & hand])[0]
        geometry = previous & hand & (distance <= 0.03)
        selections = dict(
            **{
                "02_spatial_hand": cleaned & hand,
                "03_spatial_objects": cleaned & ~hand,
                "04_raw_hand": raw & hand,
                "05_raw_objects": raw & ~hand,
                "06_previous_local_filter": previous,
                "07_removed_candidates": previous & ~cleaned,
                "08_URDF_prior_hand": geometry,
            }
        )
        if causal is not None:
            selections.update(
                {
                    "09_causal_hand": accepted & hand,
                    "10_causal_objects": accepted & ~hand,
                    "11_temporal_removed": cleaned & ~accepted,
                }
            )
            p, c = thin(repairs, np.tile([255, 225, 30], (len(repairs), 1)).astype(np.uint8), repair_ids, cap=400)
            nodes["12_one_frame_repairs"].points = p
            nodes["12_one_frame_repairs"].colors = c
        nodes["01_URDF_surface"].points = visible.astype(np.float32)
        nodes["01_URDF_surface"].colors = np.tile(CYAN, (len(visible), 1))
        for name, select in selections.items():
            p, c = thin(points[select], colors[select], ids[select], cap=650 if name.endswith("hand") else 400)
            if name == "07_removed_candidates":
                c = np.tile(np.array([255, 65, 65], np.uint8), (len(p), 1))
            if name == "11_temporal_removed":
                c = np.tile(np.array([255, 40, 200], np.uint8), (len(p), 1))
            nodes[name].points = p
            nodes[name].colors = c
        counts = dict(
            frame=frame,
            time=float(times[index]),
            raw=int(raw.sum()),
            previous=int(previous.sum()),
            spatial=int(cleaned.sum()),
            extra_removed=int((previous & ~cleaned).sum()),
            removed_by_stage=removed,
            geometry_prior_hand=int(geometry.sum()),
            removed_but_near_reference=int((previous & hand & ~cleaned & (distance <= 0.03)).sum()),
        )
        if causal is not None:
            counts.update(causal_stats, causal_measured=int(accepted.sum()))
        counts["processing_ms"] = (time.perf_counter() - frame_start) * 1000
        rows.append(counts)
        for instance in np.unique(ids[previous]):
            name = info["detections"][int(instance) - 1]["object"]["name"]
            item = per_instance.setdefault(name, dict(previous=0, spatial=0))
            item["previous"] += int((previous & (ids == instance)).sum())
            item["spatial"] += int((cleaned & (ids == instance)).sum())
        for cls, color in [(2, (30, 135, 255)), (3, (60, 220, 60))]:
            region = labels[frame] == 2 if cls == 2 else (labels[frame] > 0) & (labels[frame] != 2)
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1)
        success, jpeg = cv2.imencode(".jpg", cv2.resize(image, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 65])
        assert success
        thumbs.append(base64.b64encode(jpeg).decode())
        dt = float(times[index + 1] - times[index]) if index + 1 < len(frames) else 1 / 6
        serializer.insert_sleep(dt)
        if index in {0, len(frames) // 2, len(frames) - 1}:
            np.savez_compressed(
                out / f"cloud_{frame:04d}.npz",
                points=points[raw].astype(np.float32),
                classes=classes[raw],
                previous=previous[raw],
                spatial=cleaned[raw],
                geometry_prior=geometry[raw],
                urdf_surface=full_surface.astype(np.float32),
            )
        if index % 40 == 0:
            print(args.task, index, "/", len(frames), counts, flush=True)
    recording = serializer.serialize()
    server.stop()
    capture.release()
    env.close()
    decoded = msgspec.msgpack.decode(gzip.decompress(recording))
    (out / "scene.viser").write_bytes(recording)
    duration = float(decoded["durationSeconds"])
    metadata = dict(
        task=args.task,
        episode=str(episode),
        mask_source=str(masks.resolve()),
        frame_ids=frames,
        frame_times=times.tolist(),
        duration=duration,
        stats=rows,
        rgb_thumbnails=thumbs,
        layers={k: v[0] for k, v in LAYERS.items()},
    )
    legend, behavior = interface(metadata)
    if causal is not None:
        legend = legend.replace(
            "未运行点追踪、未补洞或平滑轨迹。",
            "使用上一张和当前RGB的双向光流对应；不读取未来帧。黄色是仅沿可靠对应补一帧的历史估计，补点不进入下一帧状态。未做双向时间平滑。",
        )
        behavior = behavior.replace(
            "新增剔除 ${r.extra_removed} 点`",
            "空间剔除 ${r.extra_removed}；跨帧剔除 ${r.temporal_rejected}；历史补点 ${r.repaired}`",
        )
    camera = dict(position=(center + [0.65, -0.2, -0.1]).tolist(), look_at=center.tolist(), up=[0, -1, 0])
    html = package(recording, metadata, camera, args.task + " D435 深度交互检查", legend)
    html = html.replace("</body>", behavior + "</body>")
    path = out / (args.task + "_d435_urdf_interactive.html")
    path.write_text(html)
    summary = dict(
        task=args.task,
        frames=len(frames),
        duration=duration,
        html_bytes=path.stat().st_size,
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        layers={k: v[0] for k, v in LAYERS.items()},
        per_instance=per_instance,
        total_previous=sum(r["previous"] for r in rows),
        total_spatial=sum(r["spatial"] for r in rows),
        total_extra_removed=sum(r["extra_removed"] for r in rows),
        removed_but_near_reference=sum(r["removed_but_near_reference"] for r in rows),
        recording_messages=len(decoded["messages"]),
        filters=dict(
            statistical="per-instance k16 mean distance <= max(mean + 2*std, 8mm)",
            components="per-instance 10mm voxel, small <15% largest, >60mm depth difference and no contact after 20mm dilation",
            urdf_prior="optional separate layer only: hand points <=30mm from URDF; excluded from accuracy assessment",
        ),
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if causal is not None:
        summary["causal"] = dict(
            total_rejected=sum(r["temporal_rejected"] for r in rows),
            total_repairs=sum(r["repaired"] for r in rows),
            total_matched=sum(r["matched"] for r in rows),
            filter_ms={str(p): float(np.percentile([r["milliseconds"] for r in rows], p)) for p in [50, 95]},
            processing_ms={str(p): float(np.percentile([r["processing_ms"] for r in rows], p)) for p in [50, 95]},
            future_frames=False,
            note="DIS optical flow CPU; GPU depth projection. Filter timing is update() only. Processing timing includes RGB/depth read, spatial filtering, URDF lookup and scene node updates; excludes thumbnails, HTML packaging and SAM2 segmentation.",
        )
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE", path, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, default=Path("pointflow_outputs/d435_interactive_accuracy_20261102"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=9120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    main(parser.parse_args())
