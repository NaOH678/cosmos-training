"""Real recordings v2: cached SAM/CoTracker UVs, only calibrated D435 depth, offline HTML."""

import argparse
import base64
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render_d435_point_accuracy import Surface, load_module, measured_cloud

TASK = "dropper"
VERSION = "v2_d435"
V0 = ROOT / "pointflow_outputs/cotracker_real_20261104" / TASK
OUT = ROOT / "pointflow_outputs/cotracker_real_v2_d435_20261104" / TASK
CORE = ROOT / "cosmos_framework/inference/robot_policy"
Guard = load_module("d435_scalar_guard", CORE / "hand_depth_guard.py").HandDepthGuard
Stabilizer = load_module("d435_tracked_guard", CORE / "tracked_d435_filter.py").TrackedD435Filter
Geometry = load_module("d435_geometry", CORE / "causal_rgb_pointflow.py").CausalGeometryFilter


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def core_depths(sensor, instances, labels):
    result = {}
    for instance in np.unique(instances[labels == 2]):
        mask = ((instances == instance) & (labels == 2)).astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        inside = (mask > 0) & (distance >= max(3, float(np.percentile(distance[mask > 0], 75))))
        values = sensor[inside & np.isfinite(sensor)]
        result[int(instance)] = None
        if len(values) >= 30:
            bins = np.arange(0.1, 2.505, 0.005)
            h, _ = np.histogram(values, bins)
            peak = int(np.argmax(gaussian_filter1d(h.astype(float), 2)))
            z = float((bins[peak] + bins[peak + 1]) / 2)
            if np.mean(abs(values - z) <= 0.03) >= 0.55:
                result[int(instance)] = z
    return result


def sample_same_instance(uv, instance, sensor, mask, radius=2.5):
    z = np.full(len(uv), np.nan, np.float32)
    for value in np.unique(instance):
        if not value:
            continue
        yy, xx = np.nonzero(np.isfinite(sensor) & (mask == value))
        if not len(xx):
            continue
        selected = np.flatnonzero(instance == value)
        distance, index = cKDTree(np.column_stack((xx, yy))).query(uv[selected], workers=1)
        use = distance <= radius
        z[selected[use]] = sensor[yy[index[use]], xx[index[use]]]
    return z


def build(enhancement=None, limit=None, native_provider=None, sample_depth=None, stabilizer_factory=None):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    if (OUT / "points_complete.json").exists():
        return
    source = json.loads((V0 / "source.json").read_text())
    ep = Path(source["episode"])
    T = min(source["frames"], limit or source["frames"])
    for name in ["source.json", "task_config.json", "video.json"]:
        shutil.copyfile(V0 / name, OUT / name)
    streams = json.loads((ep / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"][
        "streams"
    ]
    k = streams["color"]["intrinsics"]
    K = dict(fx=k["fx"], fy=k["fy"] * 448 / 480, ppx=k["ppx"], ppy=(k["ppy"] + 0.5) * 448 / 480 - 0.5)
    sync = json.loads((ep / "sync_timestamps.json").read_text())
    times = np.array([s.get("camera_head", s["anchor"]) for s in sync[:T]])
    times -= times[0]
    labels = np.load(V0 / "labels.npy", mmap_mode="r")
    instances = np.load(V0 / "instances.npy", mmap_mode="r")
    env = None
    if native_provider is None:
        cache = Path("/tmp/cotracker_real_depth_20261104") / TASK
        if not (cache / "data.mdb").exists():
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ep / "auxiliary_camera/depth.lmdb/data.mdb", cache / "data.mdb")
        env = lmdb.open(str(cache), readonly=True, lock=False)
    stabilize = (stabilizer_factory or Stabilizer)(Guard)
    sample_depth = sample_depth or sample_same_instance
    geometry = Geometry()
    rows = []
    native_count = 0
    for frame in range(T):
        image = cv2.imread(str(V0 / "frames" / f"{frame:05d}.jpg"))
        with np.load(V0 / "pointflow/frames" / f"{frame:06d}.npz") as n:
            data = {key: n[key] for key in ["ids", "uv", "labels", "instances", "colors"]}
        uv = data["uv"]
        imap = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        semantics = cv2.resize(labels[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        if enhancement is not None:
            enhancement.prepare_frame(semantics, imap, image)
        native = None
        if native_provider is None:
            with env.begin() as txn:
                blob = txn.get(f"depth/head/{frame:06d}".encode())
        else:
            # Provider returns native optical depth in the recorded depth units.
            native = native_provider(frame)
            blob = True if native is not None else None
        z = np.full(len(uv), np.nan, np.float32)
        raw_z = z.copy()
        cores = {}
        filter_stats = {}
        sensor_ms = 0.0
        filter_ms = 0.0
        tic = time.perf_counter()
        if blob is not None:
            native_count += 1
            if native_provider is None:
                native = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
            points, _, _, _, raw, filtered, pix = measured_cloud(
                native, streams, instances[frame], labels[frame], image, "cpu"
            )
            if enhancement is not None:
                filtered &= ~enhancement.occlusion(points, pix, native, streams)
            buffers = []
            for selected in [raw, filtered]:
                b = np.full(image.shape[:2], np.inf, np.float32)
                np.minimum.at(b, (pix[selected, 1], pix[selected, 0]), points[selected, 2].astype(np.float32))
                buffers.append(cv2.resize(b, (640, 448), interpolation=cv2.INTER_NEAREST))
            raw_z = sample_depth(uv, data["instances"], buffers[0], imap)
            if enhancement is not None:
                buffers[1] = enhancement.gate_buffer(buffers[1])
            z = sample_depth(uv, data["instances"], buffers[1], imap)
            cores = core_depths(buffers[1], imap, semantics)
            sensor_ms = (time.perf_counter() - tic) * 1000
            tic = time.perf_counter()
            q = np.rint(uv).astype(int)
            q[:, 0] = q[:, 0].clip(0, 639)
            q[:, 1] = q[:, 1].clip(0, 447)
            interior = np.zeros(imap.shape, bool)
            for instance in np.unique(imap):
                if instance:
                    interior |= (
                        cv2.erode(
                            (imap == instance).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        )
                        > 0
                    )
            valid = np.isfinite(z) & interior[q[:, 1], q[:, 0]]
            xyz = np.column_stack(((uv[:, 0] - K["ppx"]) / K["fx"] * z, (uv[:, 1] - K["ppy"]) / K["fy"] * z, z)).astype(
                np.float32
            )
            # Spatial and per-ID residual rejection from the earlier pipeline.
            valid, filter_stats = geometry.apply(data["ids"], xyz, valid, data["instances"], float(times[frame]))
        else:
            valid = np.zeros(len(uv), bool)
        blocked = enhancement.blocked(uv, data["labels"]) if enhancement is not None else np.zeros(len(uv), bool)
        valid &= ~blocked
        filtered_z, provenance, stats = stabilize.update(
            data["ids"], data["instances"], z, valid, float(times[frame]), blob is not None, cores
        )
        if blob is not None:
            trusted = provenance == 1
            geometry.last_ids = data["ids"][trusted].copy()
            geometry.last_points = np.column_stack(
                (
                    (uv[trusted, 0] - K["ppx"]) / K["fx"] * z[trusted],
                    (uv[trusted, 1] - K["ppy"]) / K["fy"] * z[trusted],
                    z[trusted],
                )
            ).astype(np.float32)
        if enhancement is not None:
            before_surface = provenance.copy()
            filtered_z, provenance, extra = enhancement.update(
                data["ids"],
                uv,
                data["instances"],
                data["labels"],
                z,
                filtered_z,
                provenance,
                float(times[frame]),
                blob is not None,
                blocked,
            )
            for rejected in np.flatnonzero((before_surface == 1) & (provenance == 0)):
                history = stabilize.samples.get(int(data["ids"][rejected]))
                if history and history[-1][0] == float(times[frame]):
                    history.pop()
                    if not history:
                        del stabilize.samples[int(data["ids"][rejected])]
            if blob is not None:
                trusted = provenance == 1
                geometry.last_ids = data["ids"][trusted].copy()
                geometry.last_points = np.column_stack(
                    (
                        (uv[trusted, 0] - K["ppx"]) / K["fx"] * z[trusted],
                        (uv[trusted, 1] - K["ppy"]) / K["fy"] * z[trusted],
                        z[trusted],
                    )
                ).astype(np.float32)
            stats.update(extra)
            stats.update(
                measured=int((provenance == 1).sum()),
                repaired=int(((provenance == 2) | (provenance == 4)).sum()),
                interframe=int((provenance == 3).sum()),
                invalid=int((provenance == 0).sum()),
            )
        xyz = np.column_stack(
            ((uv[:, 0] - K["ppx"]) / K["fx"] * filtered_z, (uv[:, 1] - K["ppy"]) / K["fy"] * filtered_z, filtered_z)
        ).astype(np.float32)
        raw_xyz = np.column_stack(
            ((uv[:, 0] - K["ppx"]) / K["fx"] * raw_z, (uv[:, 1] - K["ppy"]) / K["fy"] * raw_z, raw_z)
        ).astype(np.float32)
        filter_ms = (time.perf_counter() - tic) * 1000
        np.savez_compressed(
            OUT / "frames" / f"{frame:06d}.npz",
            **data,
            xyz=xyz,
            raw_xyz=raw_xyz,
            valid=provenance > 0,
            provenance=provenance,
            sensor_update=blob is not None,
            timestamp=float(times[frame]),
        )
        rows.append(
            dict(
                frame=frame,
                sensor_update=blob is not None,
                sensor_ms=sensor_ms,
                filter_ms=filter_ms,
                **stats,
                geometry=filter_stats,
            )
        )
        if frame % 100 == 0:
            print(
                "D435",
                frame,
                T,
                "measured",
                stats["measured"],
                "repair",
                stats["repaired"],
                "interframe",
                stats["interframe"],
                flush=True,
            )
    if env is not None:
        env.close()
    save(OUT / "per_frame.json", rows)
    actual = [r for r in rows if r["sensor_update"]]
    save(
        OUT / "points_complete.json",
        dict(
            version=VERSION,
            frames=T,
            depth_frames=native_count,
            da3_used=False,
            uses_future_depth=False,
            uv_ids_equal_v0=True,
            real_intrinsics=K,
            measured_observations=sum(r["measured"] for r in rows),
            repaired_observations=sum(r["repaired"] for r in rows),
            interframe_observations=sum(r["interframe"] for r in rows),
            sensor_ms_p50=float(np.median([r["sensor_ms"] for r in actual])),
            filter_ms_p50=float(np.median([r["filter_ms"] for r in actual])),
            filter_ms_p95=float(np.percentile([r["filter_ms"] for r in actual], 95)),
        ),
    )


def html(
    comparison=None,
    show_urdf=False,
    surface_stride=8,
    depth_label="D435",
    legend_text=None,
    display_hold_steps=3,
    highlight_interframe=False,
    comparison_label="05_V2_comparison",
):
    import viser

    if (OUT / "html_complete.json").exists():
        return

    sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
    from export_efep_official_html import package

    source = json.loads((OUT / "source.json").read_text())
    config = json.loads((OUT / "task_config.json").read_text())
    T = source["frames"]
    rows = json.loads((OUT / "per_frame.json").read_text())
    surface = Surface(Path(source["episode"]), config)
    server = viser.ViserServer(host="127.0.0.1", port=9196)
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    primary = f"01_{VERSION}_stabilized"
    raw_layer = f"02_{depth_label}_raw_actual_frames"
    nodes = {
        name: server.scene.add_point_cloud(
            "/" + name,
            points=np.empty((0, 3), np.float32),
            colors=np.empty((0, 3), np.uint8),
            point_size=0.0025,
            visible=visible,
        )
        for name, visible in [
            (primary, True),
            (raw_layer, False),
            ("03_history_repairs_yellow", False),
            ("04_URDF_surface", show_urdf),
        ]
    }
    if comparison is not None:
        nodes[comparison_label] = server.scene.add_point_cloud(
            "/" + comparison_label,
            points=np.empty((0, 3), np.float32),
            colors=np.empty((0, 3), np.uint8),
            point_size=0.0025,
            visible=False,
        )
    if show_urdf:
        nodes["04_URDF_surface"].point_size = 0.0013
    server._run_garbage_collector(force=True)
    record = server.get_scene_serializer()
    history = deque(maxlen=1 if VERSION == "v3_d435" else display_hold_steps + 1)
    frame_ids = list(range(0, T, 2))
    thumbs = []
    centers = []
    display_stats = []
    raw_points = np.empty((0, 3), np.float32)
    raw_colors = np.empty((0, 3), np.uint8)
    for index, frame in enumerate(frame_ids):
        with np.load(OUT / "frames" / f"{frame:06d}.npz") as n:
            data = {k: n[k] for k in n.files}
        # Last raw actual capture may fall on an odd RGB index after packet jitter.
        for f in range(max(0, frame - 1), frame + 1):
            with np.load(OUT / "frames" / f"{f:06d}.npz") as n:
                if bool(n["sensor_update"]):
                    raw_valid = np.isfinite(n["raw_xyz"]).all(1)
                    raw_points = n["raw_xyz"][raw_valid]
                    raw_colors = n["colors"][raw_valid]
        if comparison is not None:
            with np.load(Path(comparison) / "frames" / f"{frame:06d}.npz") as old:
                compare_valid = old["valid"]
                compare_color = old["colors"].copy()
                compare_color[old["labels"] == 2] = [255, 127, 14]
                nodes[comparison_label].points = old["xyz"][compare_valid]
                nodes[comparison_label].colors = compare_color[compare_valid]
        valid = data["valid"]
        colors = data["colors"].copy()
        colors[data["labels"] == 2] = [255, 127, 14]
        history.append((data["ids"][valid], data["xyz"][valid], colors[valid]))
        seen = set()
        pp = []
        cc = []
        for ids, points, color in reversed(history):
            keep = np.array([int(i) not in seen for i in ids], bool)
            seen.update(map(int, ids))
            pp.append(points[keep])
            cc.append(color[keep])
        shown = np.concatenate(pp)
        shown_colors = np.concatenate(cc)
        nodes[primary].points = shown
        nodes[primary].colors = shown_colors
        nodes[raw_layer].points = raw_points
        nodes[raw_layer].colors = raw_colors
        repaired = valid & (
            (data["provenance"] == 2) | (data["provenance"] == 4) | ((data["provenance"] == 3) & highlight_interframe)
        )
        nodes["03_history_repairs_yellow"].points = data["xyz"][repaired]
        nodes["03_history_repairs_yellow"].colors = np.tile([255, 230, 0], (int(repaired.sum()), 1)).astype(np.uint8)
        ref = surface.at(frame)[::surface_stride].astype(np.float32)
        nodes["04_URDF_surface"].points = ref
        nodes["04_URDF_surface"].colors = np.tile([30, 160, 235], (len(ref), 1)).astype(np.uint8)
        if len(shown):
            centers.append(np.median(shown, 0))
        image = cv2.imread(str(V0 / "frames" / f"{frame:05d}.jpg"))
        ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 65])
        assert ok
        thumbs.append(base64.b64encode(jpg).decode())
        display_stats.append(
            dict(
                frame=frame,
                shown=len(shown),
                hold=len(shown) - int(valid.sum()),
                measured=rows[frame]["measured"],
                repaired=rows[frame]["repaired"],
                interframe=rows[frame]["interframe"],
                sensor_update=rows[frame]["sensor_update"],
            )
        )
        record.insert_sleep((min(frame + 2, T) - frame) / source["fps"])
        if index % 100 == 0:
            print("HTML", index, len(frame_ids), flush=True)
    recording = record.serialize()
    server.stop()
    center = np.median(centers, 0)
    metadata = dict(
        version=VERSION,
        frames=T,
        frame_ids=frame_ids,
        frame_times=[f / source["fps"] for f in frame_ids],
        duration=T / source["fps"],
        rgb_thumbnails=thumbs,
        stats=display_stats,
        da3_used=False,
        urdf=config["urdf"],
    )
    legend = '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:330px;font:13px/1.5 system-ui"><b>TASK_NAME · v2 D435</b><br>同一套SAM2/CoTracker；深度仅D435，完全不用DA3。真实相机标定对齐RGB。<br>01：稳定后点云；02：原始D435有效投影（延续最近真实帧）；03：历史修复黄点；04：URDF手表面。右侧Scene切换。<br>D435按记录帧更新，RGB 30Hz；中间帧使用已有点最近可信深度和短时趋势。历史补点不反馈到测量历史，最多2次深度观测/0.2秒。<br>每2帧显示，最多3个显示步保留最后真实显示位置。鼠标旋转，空格暂停，拖时间轴。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
    behavior = """<script>let paused=false;const m=window.__VISER_EXPORT_METADATA__;setInterval(()=>{const icon=document.querySelector('svg.tabler-icon-player-pause-filled');const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-m.duration)<.01);const t=slider?Number(slider.getAttribute('aria-valuenow')):0;const firstVisible=m.stats.findIndex(r=>r.shown>0);if(!paused&&icon&&t>=m.frame_times[Math.max(0,firstVisible)]){icon.closest('button').click();paused=true;}let i=0;while(i+1<m.frame_times.length&&m.frame_times[i+1]<=t)i++;const r=m.stats[i];document.getElementById('status').textContent=`帧${r.frame} · ${r.sensor_update?'深度更新':'RGB中间帧'} · 测量${r.measured} / 修复${r.repaired} / 中间帧预测${r.interframe} / 显示暂留${r.hold}`;const im=document.getElementById('rgb');if(im.dataset.i!==String(i)){im.src='data:image/jpeg;base64,'+m.rgb_thumbnails[i];im.dataset.i=String(i);}},150);</script>"""
    if VERSION == "v3_d435":
        legend = legend.replace("v2 D435", "v3 D435")
        legend = legend.replace(
            "每2帧显示，最多3个显示步保留最后真实显示位置。",
            "静止物体融合历史可信深度，运动时退出强融合；遮挡边界无效。每2帧显示，无额外旧XYZ暂留。05可切换V2对照。",
        )
    if legend_text is not None:
        legend = legend_text
    legend = legend.replace("TASK_NAME", TASK)
    target = OUT / f"{TASK}_{VERSION}.html"
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
        package(
            recording,
            metadata,
            dict(position=(center + [0.55, -0.2, -0.15]).tolist(), look_at=center.tolist(), up=[0, -1, 0]),
            f"{TASK} {VERSION}",
            legend,
        ).replace("</body>", behavior + "</body>")
    )
    temp.replace(target)
    save(
        OUT / "html_complete.json",
        dict(
            path=str(target),
            frames=T,
            display_frames=len(frame_ids),
            html_bytes=target.stat().st_size,
            da3_used=False,
            urdf=config["urdf"],
        ),
    )
    print("COMPLETE", target, flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["bottle", "scissors", "micro", "sandwich", "dropper", "remaining"], default="dropper"
    )
    args = parser.parse_args()
    for TASK in ["bottle", "scissors", "micro", "sandwich"] if args.task == "remaining" else [args.task]:
        V0 = ROOT / "pointflow_outputs/cotracker_real_20261104" / TASK
        OUT = ROOT / "pointflow_outputs/cotracker_real_v2_d435_20261104" / TASK
        print("TASK START", TASK, flush=True)
        build()
        html()
