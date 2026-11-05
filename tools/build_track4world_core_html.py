"""Six Track4World-cache HTMLs: per-instance core Z translation / ray scale.

Recorded UV/track identities are kept. Each instance estimates its own parameters
from the current available D435 observation; no propagation across instances.
"""

import argparse
import base64
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import lmdb
import msgspec
import numpy as np
from render_d435_point_accuracy import TASKS, Surface, load_module, measured_cloud, visible_surface
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
sys.path.insert(0, "/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools")
from export_efep_official_html import package  # noqa: E402

SOURCES = {
    "sandwich": ("sandwich", "efep_seg_v2", "episode_0013_20260731_133649"),
    "dropper": ("dagger", "efep_seg_v61", "episode_0000_20260830_155455"),
    "micro": ("micropipette", "efep_seg_v2", "episode_0007_20260824_174548"),
}


def main(args):
    taskout = args.out / args.task
    taskout.mkdir(parents=True, exist_ok=True)
    assert not (taskout / "manifest.json").exists(), "Use --export-only to rebuild an existing manifest"
    framesout = taskout / "frames"
    framesout.mkdir(exist_ok=True)
    ep, maskdir = TASKS[args.task]
    episode = Path("/data/shichaojian/raw_data") / ep
    maskdir = Path(maskdir)
    info = json.loads((maskdir / "initialization.json").read_text())
    config = (
        json.loads((maskdir / "task_config.json").read_text())
        if (maskdir / "task_config.json").exists()
        else info["config"]
    )
    names = {i + 1: r["object"]["name"] for i, r in enumerate(info["detections"])}
    metadata = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    intr = metadata["color"]["intrinsics"]
    labels = np.load(maskdir / "labels.npy", mmap_mode="r")
    instances = np.load(maskdir / "instances.npy", mmap_mode="r")
    reference = Surface(episode, config)
    legacy = load_module(
        "legacy_core", "/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/render_da3_sensor_calibration.py"
    )
    source = Path("/data/shichaojian/pf_out/9.24").joinpath(*SOURCES[args.task])
    arrays = {
        key: np.load(source / (key + ".npy"), mmap_mode="r")
        for key in ["obs_pos", "obs_uv", "obs_valid", "obs_unique", "obs_track", "frame_offsets"]
    }
    report = json.loads((source / "report.json").read_text())
    assert report["export"]["infer_width"] == 640 and report["export"]["infer_height"] == 448
    env = lmdb.open(str(Path("/tmp/d435_point_accuracy_20261102") / args.task), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        )
    sync = json.loads((episode / "sync_timestamps.json").read_text())
    times = np.array([sync[f].get("camera_head", sync[f]["anchor"]) for f in frames])
    times -= times[0]
    cap = cv2.VideoCapture(str(episode / "videos/head.mp4"))
    stats = []
    thumbs = []
    center_samples = []
    for index, frame in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{frame:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        sensor, _, _, _, raw, _, _ = measured_cloud(depth, metadata, instances[frame], labels[frame], image, "cpu")
        with np.errstate(divide="ignore", invalid="ignore"):
            edge = legacy.utils3d.numpy.depth_edge(
                depth.astype(np.float32) * metadata["depth"]["depth_scale_m"], rtol=0.04
            ).ravel()
        valid_sensor = raw & np.isfinite(sensor).all(1) & (sensor[:, 2] > 0.1) & (sensor[:, 2] < 6) & ~edge
        sensor = sensor[valid_sensor]
        sensor_uv = np.column_stack(
            (
                sensor[:, 0] / sensor[:, 2] * intr["fx"] + intr["ppx"],
                (sensor[:, 1] / sensor[:, 2] * intr["fy"] + intr["ppy"] + 0.5) * 448 / 480 - 0.5,
            )
        )
        a, b = map(int, arrays["frame_offsets"][frame : frame + 2])
        keep = np.asarray(arrays["obs_valid"][a:b]) & np.asarray(arrays["obs_unique"][a:b])
        uv = np.asarray(arrays["obs_uv"][a:b])[keep].astype(float)
        z = np.asarray(arrays["obs_pos"][a:b])[keep, 2].astype(float)
        track = np.asarray(arrays["obs_track"][a:b])[keep]
        rgb_uv = np.column_stack((uv[:, 0], (uv[:, 1] + 0.5) * 480 / 448 - 0.5))
        q = np.rint(rgb_uv).astype(int)
        inside = (q[:, 0] >= 0) & (q[:, 0] < 640) & (q[:, 1] >= 0) & (q[:, 1] < 480) & np.isfinite(z) & (z > 0)
        assigned = np.zeros(len(q), np.uint8)
        assigned[inside] = instances[frame][q[inside, 1], q[inside, 0]]
        selected = inside & (assigned > 0)
        uv, z, track, q, assigned = (x[selected] for x in [uv, z, track, q, assigned])
        points = np.column_stack(
            ((q[:, 0] - intr["ppx"]) / intr["fx"] * z, (rgb_uv[selected, 1] - intr["ppy"]) / intr["fy"] * z, z)
        )
        # Preserve the original subpixel image ray; q is only for mask/RGB lookup.
        points[:, 0] = (rgb_uv[selected, 0] - intr["ppx"]) / intr["fx"] * z
        colors = image[q[:, 1], q[:, 0], ::-1].copy()
        classes = labels[frame][q[:, 1], q[:, 0]]
        start = time.perf_counter()
        params = []
        mask_canvas = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        for instance, name in names.items():
            mask = (mask_canvas == instance).astype(np.uint8)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            threshold = max(3.0, float(np.percentile(distance[mask > 0], 75))) if mask.any() else 3.0
            core = (mask > 0) & (distance >= threshold)
            which = assigned == instance
            delta, scale, detail = legacy.estimate_anchor(points[which], uv[which], sensor, sensor_uv, core)
            params.append(
                dict(
                    instance=instance,
                    name=name,
                    offset_m=delta,
                    scale=scale,
                    points=int(which.sum()),
                    core_threshold_px=threshold,
                    **{k: v for k, v in detail.items() if k not in ["offset_m", "scale"]},
                )
            )
        calibration_ms = (time.perf_counter() - start) * 1000
        deltas = np.zeros(len(points))
        scales = np.ones(len(points))
        for item in params:
            which = assigned == item["instance"]
            deltas[which] = item["offset_m"]
            scales[which] = item["scale"]
        start = time.perf_counter()
        translated = points.copy()
        translated[:, 2] += deltas
        translation_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        scaled = points * scales[:, None]
        scale_ms = (time.perf_counter() - start) * 1000
        surface = reference.at(frame)
        visible = visible_surface(surface, intr)
        visible = visible[np.linspace(0, len(visible) - 1, min(4500, len(visible))).astype(int)]
        if index % 20 == 0:
            center_samples.append(visible)
        tree = cKDTree(surface)
        hand = classes == 2
        distances = {
            name: float(tree.query(p[hand])[0].mean() * 1000) if hand.any() else None
            for name, p in [("raw", points), ("z_translation", translated), ("ray_scale", scaled)]
        }
        # Stable track-hash display thinning, performed after full-data calibration.
        shown = []
        for instance in names:
            candidates = np.flatnonzero(assigned == instance)
            hashes = (track[candidates].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
            shown.extend(candidates[np.argsort(hashes, kind="stable")[:1000]])
        shown = np.asarray(shown, np.int64)
        np.savez_compressed(
            framesout / f"{frame:06d}.npz",
            points=points[shown].astype(np.float32),
            colors=colors[shown],
            labels=classes[shown],
            instances=assigned[shown],
            track=track[shown],
            translated=translated[shown].astype(np.float32),
            scaled=scaled[shown].astype(np.float32),
            urdf=visible.astype(np.float32),
            sensor=sensor[np.linspace(0, len(sensor) - 1, min(3000, len(sensor))).astype(int)].astype(np.float32),
        )
        stats.append(
            dict(
                frame=frame,
                time=float(times[index]),
                points=len(points),
                shown=len(shown),
                parameters=params,
                calibration_ms=calibration_ms,
                translation_ms=translation_ms,
                scale_ms=scale_ms,
                hand_mean_distance_mm=distances,
            )
        )
        for instance in names:
            contour, _ = cv2.findContours(
                (instances[frame] == instance).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contour, -1, (0, 180, 255), 1)
        ok, jpeg = cv2.imencode(".jpg", cv2.resize(image, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 65])
        assert ok
        thumbs.append(base64.b64encode(jpeg).decode())
        if index % 50 == 0:
            print(
                args.task, index, "/", len(frames), "accepted", [p["name"] for p in params if p["accepted"]], flush=True
            )
    cap.release()
    env.close()
    center = np.median(np.concatenate(center_samples), axis=0)
    manifest = dict(
        task=args.task,
        episode=str(episode),
        source=str(source),
        mask=str(maskdir.resolve()),
        frames=frames,
        times=times.tolist(),
        center=center.tolist(),
        stats=stats,
        thumbs=thumbs,
        instances=names,
        base="Track4World obs_valid & obs_unique; cached DA3 Z reprojected using true RGB K. No single-frame A-E depth; no extra metric recalibration.",
        per_instance=True,
        uv="original observed pixels and native track IDs retained; correction never reprojects/reassigns UV",
        d435="calibrated depth-to-RGB, native depth edge rtol .04, 0.1..6m, RGB z-buffer; no URDF fitting",
    )
    (taskout / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    export(taskout, manifest)


def export(root, manifest, methods=None):
    import viser

    for method, key in methods or [("z_translation", "translated"), ("ray_scale", "scaled")]:
        server = viser.ViserServer(
            host="127.0.0.1", port={"sandwich": 9170, "dropper": 9171, "micro": 9172}[manifest["task"]]
        )
        server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False, control_width="small")
        definitions = {
            "01_URDF_surface": ("URDF手表面参考", True),
            "02_corrected_hand": ("校正后的手（橙）", True),
            "03_corrected_objects": ("各自校正后的物体（RGB）", True),
            "04_original_hand": ("校正前手", False),
            "05_original_objects": ("校正前物体", False),
            "06_sensor_reference": ("D435配对测量参考", False),
        }
        if manifest.get("shared_hand_shift"):
            definitions["03_corrected_objects"] = ("物体共用手部Z平移（RGB）", True)
        empty = np.empty((0, 3), np.float32)
        nodes = {
            name: server.scene.add_point_cloud(
                "/" + name, points=empty, colors=np.empty((0, 3), np.uint8), point_size=0.0025, visible=visible
            )
            for name, (_, visible) in definitions.items()
        }
        server._run_garbage_collector(force=True)
        recorder = server.get_scene_serializer()
        for i, frame in enumerate(manifest["frames"]):
            with np.load(root / "frames" / f"{frame:06d}.npz") as n:
                hand = n["labels"] == 2
                sets = {
                    "01_URDF_surface": (n["urdf"], np.tile([30, 160, 235], (len(n["urdf"]), 1))),
                    "02_corrected_hand": (n[key][hand], np.tile([255, 127, 14], (int(hand.sum()), 1))),
                    "03_corrected_objects": (n[key][~hand], n["colors"][~hand]),
                    "04_original_hand": (n["points"][hand], np.tile([190, 190, 190], (int(hand.sum()), 1))),
                    "05_original_objects": (n["points"][~hand], n["colors"][~hand]),
                    "06_sensor_reference": (n["sensor"], np.tile([55, 220, 120], (len(n["sensor"]), 1))),
                }
                for name, (p, c) in sets.items():
                    nodes[name].points = p
                    nodes[name].colors = c.astype(np.uint8)
            dt = (
                manifest["times"][i + 1] - manifest["times"][i]
                if i + 1 < len(manifest["frames"])
                else float(np.median(np.diff(manifest["times"])))
            )
            recorder.insert_sleep(dt)
        recording = recorder.serialize()
        server.stop()
        decoded = msgspec.msgpack.decode(gzip.decompress(recording))
        duration = float(decoded["durationSeconds"])
        (root / (method + ".viser")).write_bytes(recording)
        meta = dict(
            duration=duration,
            frame_ids=manifest["frames"],
            frame_times=manifest["times"],
            stats=manifest["stats"],
            rgb_thumbnails=manifest["thumbs"],
            method=method,
            layers={name: value[0] for name, value in definitions.items()},
        )
        label = "Z方向平移" if method == "z_translation" else "沿视线缩放"
        legend = f"""<aside style="position:fixed;left:12px;top:12px;z-index:9999;width:330px;max-height:88vh;overflow:auto;background:#15202bf2;color:#eee;padding:12px;border-radius:8px;font:13px/1.5 system-ui"><b>{manifest["task"]} · Track4World＋{label}</b><br>每只手／每个物体各自估计参数；无可靠配对则不校正。<br>固定初始视角；鼠标旋转／缩放，空格暂停，时间轴定位。右侧Scene眼睛切换图层。<div id="source"></div><img id="rgb" style="width:100%"><div id="params"></div><details><summary>图层与方法</summary>蓝色URDF，橙色校正手，RGB校正物体。04/05为原始DA3对照，06为D435参考。<br>核心区域为实例mask内部最高四分位，至少离边界3px；D435主峰±3cm、占比≥55%、30个可靠配对及MAD筛选。<br>独立校正可能改变手物接触关系；缩放会改变物体尺寸。没有未来插值或按URDF拟合。保留原始UV／轨迹身份。只使用实际D435帧。<br>原始深度来自旧Track4World多帧缓存，本次没有把它改成在线单帧模型；也不等同先前仅手部的10.13mm实验。</details></aside>"""
        behavior = """<script>
let paused=false;const m=window.__VISER_EXPORT_METADATA__;
setInterval(()=>{const icon=document.querySelector('svg.tabler-icon-player-pause-filled');if(!paused&&icon){icon.closest('button').click();paused=true;}
const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-m.duration)<.01);const t=slider?Number(slider.getAttribute('aria-valuenow')):0;let i=0;while(i+1<m.frame_times.length&&m.frame_times[i+1]<=t+1e-5)i++;
document.getElementById('source').textContent=`源帧 ${m.frame_ids[i]} · ${t.toFixed(2)}s · ${i+1}/${m.frame_ids.length}`;
const image=document.getElementById('rgb');if(image.dataset.i!==String(i)){image.src='data:image/jpeg;base64,'+m.rgb_thumbnails[i];image.dataset.i=String(i);}
document.getElementById('params').innerHTML=m.stats[i].parameters.map(p=>`${p.name}: ${p.accepted?(m.method==='z_translation'?(p.offset_m*1000).toFixed(1)+' mm':'×'+p.scale.toFixed(3)):'保留原值 · '+p.reason} · ${p.points}点`).join('<br>');
},150);</script>"""
        if manifest.get("shared_hand_shift"):
            legend = legend.replace(
                "每只手／每个物体各自估计参数；无可靠配对则不校正。",
                "只估计手部Z平移量；所有手和物体共用同一个量。D435手深度突变时拒绝该测量，最多按历史趋势预测2帧。",
            )
            legend = legend.replace(
                "独立校正可能改变手物接触关系；缩放会改变物体尺寸。",
                "整个前景同一Z平移，保持已有手物相对位置和尺寸。不是缩放。",
            )
            legend = legend.replace(
                "核心区域为实例mask内部最高四分位", "只用手计算参数；手核心区域为手mask内部最高四分位"
            )
            legend = legend.replace(
                "没有未来插值或按URDF拟合。",
                "仅用4次已接受的历史手深度；门限40mm＋0.20m/s×min(有效测量间隔,0.25s)。预测最长0.5秒且最多2帧，之后丢弃D435校正、显示原始DA3。没有未来插值或按URDF拟合。",
            )
            behavior = behavior.replace(
                "document.getElementById('params').innerHTML=m.stats[i].parameters.map",
                "document.getElementById('params').innerHTML=`共用Z平移 ${(m.stats[i].shared_shift_m*1000).toFixed(1)} mm · ${m.stats[i].shared_shift_status}<br>`+m.stats[i].hand_guard.map(h=>`${h.name}: ${h.status}, 观测Z=${h.observed_z_m===null?'无':h.observed_z_m.toFixed(3)}m`).join('<br>')+'<hr>'+m.stats[i].parameters.map",
            )
        center = np.array(manifest["center"])
        camera = dict(position=(center + [0.65, -0.2, -0.1]).tolist(), look_at=center.tolist(), up=[0, -1, 0])
        title = (
            manifest["task"]
            + " "
            + label
            + (" 手物共用平移＋历史深度突变检查" if manifest.get("shared_hand_shift") else " 逐实例深度校正")
        )
        html = package(recording, meta, camera, title, legend).replace("</body>", behavior + "</body>")
        path = root / f"{manifest['task']}_{method}.html"
        path.write_text(html)
        per_instance = {
            name: dict(
                accepted=sum(
                    p["accepted"] for row in manifest["stats"] for p in row["parameters"] if p["name"] == name
                ),
                frames=len(manifest["frames"]),
            )
            for name in manifest["instances"].values()
        }
        summary = dict(
            task=manifest["task"],
            method=method,
            frames=len(manifest["frames"]),
            duration=duration,
            html_bytes=path.stat().st_size,
            source=manifest["source"],
            per_instance=per_instance,
            calibration_p50_ms=float(np.median([r["calibration_ms"] for r in manifest["stats"]])),
            calibration_p95_ms=float(np.percentile([r["calibration_ms"] for r in manifest["stats"]], 95)),
            apply_p50_ms=float(
                np.median([r["translation_ms" if method == "z_translation" else "scale_ms"] for r in manifest["stats"]])
            ),
            layers=meta["layers"],
        )
        if manifest.get("shared_hand_shift"):
            summary["calibration_p50_ms"] = None
            summary["calibration_p95_ms"] = None
            summary["timing_note"] = (
                "Hand core fits reused from prior cache; complete hand fitting and DA3 runtime not timed here. New scalar-history gate and apply timing only."
            )
            summary["guard_p50_ms"] = float(np.median([r["guard_ms"] for r in manifest["stats"]]))
            summary["guard_p95_ms"] = float(np.percentile([r["guard_ms"] for r in manifest["stats"]], 95))
        path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print("COMPLETE", path, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, default=Path("pointflow_outputs/track4world_instance_core_html_20261104"))
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        root = args.out / args.task
        export(root, json.loads((root / "manifest.json").read_text()))
    else:
        main(args)
