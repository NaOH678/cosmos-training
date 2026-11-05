"""Exact old dense-view membership + shared-hand Z correction, no new mask crop.

Same source arrays, obs_valid/obs_unique, time_stride=2 and hold=3 display
observations as build_dense3d_flow.py. Corrected XYZ is original XYZ+[0,0,delta].
The hand-depth gate is reused; it never removes an object point.
"""

import argparse
import base64
import gzip
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import msgspec
import numpy as np
from render_d435_point_accuracy import TASKS, Surface, visible_surface

sys.path.insert(0, "/mnt/afs/WorldAct-native-track4w-dense/new_pointflow")
from export_efep_official_html import package  # noqa: E402


def main(args):
    import viser

    old = Path("pointflow_outputs/track4world_shared_hand_shift_20261104") / args.task
    shared = json.loads((old / "manifest.json").read_text())
    shifts = {r["frame"]: r for r in shared["stats"]}
    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=False)
    source = Path(shared["source"])
    data = {
        k: np.load(source / (k + ".npy"), mmap_mode="r")
        for k in ["frame_offsets", "obs_pos", "obs_rgb", "obs_label", "obs_track", "obs_valid", "obs_unique"]
    }
    tracks = len(np.load(source / "track_label.npy", mmap_mode="r"))
    T = len(data["frame_offsets"]) - 1
    ep, maskdir = TASKS[args.task]
    episode = Path("/data/shichaojian/raw_data") / ep
    maskdir = Path(maskdir)
    config = (
        json.loads((maskdir / "task_config.json").read_text())
        if (maskdir / "task_config.json").exists()
        else json.loads((maskdir / "initialization.json").read_text())["config"]
    )
    reference = Surface(episode, config)
    intr = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"][
        "streams"
    ]["color"]["intrinsics"]
    sync = json.loads((episode / "sync_timestamps.json").read_text())
    all_times = np.array([r.get("camera_head", r["anchor"]) for r in sync])
    all_times -= all_times[0]
    frames = list(range(0, T, 2))
    last_pos = np.zeros((tracks, 3), np.float32)
    last_color = np.zeros((tracks, 3), np.uint8)
    last_label = np.zeros(tracks, np.uint8)
    recent = deque(maxlen=4)
    cap = cv2.VideoCapture(str(episode / "videos/head.mp4"))
    server = viser.ViserServer(host="127.0.0.1", port={"sandwich": 9180, "dropper": 9181, "micro": 9182}[args.task])
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False, control_width="small")
    empty = np.empty((0, 3), np.float32)
    nodes = {
        key: server.scene.add_point_cloud(
            "/" + key, points=empty, colors=np.empty((0, 3), np.uint8), point_size=0.003, visible=visible
        )
        for key, visible in [
            ("01_URDF_surface", True),
            ("02_corrected_full_density", True),
            ("03_original_full_density", False),
        ]
    }
    server._run_garbage_collector(force=True)
    recorder = server.get_scene_serializer()
    rows = []
    thumbs = []
    shift = 0.0
    status = "尚无D435观测"
    depth_frame = None
    pending = sorted(shifts)
    cursor = 0
    checked_raw = 0
    max_shift_error = 0.0
    first_center = None
    for index, frame in enumerate(frames):
        # Consume all actually arrived sensor updates, including updates on odd
        # RGB frames that are not serialized under the old time_stride=2 rule.
        while cursor < len(pending) and pending[cursor] <= frame:
            depth_frame = pending[cursor]
            r = shifts[depth_frame]
            shift = r["shared_shift_m"]
            status = r["shared_shift_status"]
            cursor += 1
        a, b = map(int, data["frame_offsets"][frame : frame + 2])
        keep = np.asarray(data["obs_unique"][a:b]) & np.asarray(data["obs_valid"][a:b])
        ids = np.asarray(data["obs_track"][a:b])[keep]
        pos = np.asarray(data["obs_pos"][a:b])[keep]
        rgb = np.asarray(data["obs_rgb"][a:b])[keep]
        labels = np.asarray(data["obs_label"][a:b])[keep]
        assert np.isfinite(pos).all()
        last_pos[ids] = pos
        last_color[ids] = rgb
        last_label[ids] = labels
        recent.append(ids)
        shown = np.unique(np.concatenate(recent)) if len(recent) > 1 else ids
        raw = last_pos[shown]
        corrected = raw.copy()
        corrected[:, 2] += shift
        np.testing.assert_array_equal(raw[:, :2], corrected[:, :2])
        if len(raw):
            max_shift_error = max(max_shift_error, float(np.max(np.abs(corrected[:, 2] - raw[:, 2] - shift))))
        colors = last_color[shown].copy()
        colors[last_label[shown] == 2] = [255, 127, 14]
        nodes["02_corrected_full_density"].points = corrected
        nodes["02_corrected_full_density"].colors = colors
        nodes["03_original_full_density"].points = raw
        nodes["03_original_full_density"].colors = colors
        surface = visible_surface(reference.at(frame), intr)
        surface = surface[np.linspace(0, len(surface) - 1, min(4500, len(surface))).astype(int)]
        nodes["01_URDF_surface"].points = surface.astype(np.float32)
        nodes["01_URDF_surface"].colors = np.tile(np.array([30, 170, 245], np.uint8), (len(surface), 1))
        if first_center is None:
            first_center = np.median(surface, axis=0)
        # Exact native point membership check (before historical hold).
        assert len(ids) == int(keep.sum())
        checked_raw += len(ids)
        rows.append(
            dict(
                frame=frame,
                time=float(all_times[frame]),
                native_points=len(ids),
                display_points=len(shown),
                historical_hold_points=len(shown) - len(ids),
                hand_points=int((last_label[shown] == 2).sum()),
                object_surface_points=int((last_label[shown] != 2).sum()),
                shift_m=shift,
                status=status,
                last_depth_frame=depth_frame,
            )
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        ok, jpeg = cv2.imencode(".jpg", cv2.resize(image, (256, 192)), [cv2.IMWRITE_JPEG_QUALITY, 55])
        assert ok
        thumbs.append(base64.b64encode(jpeg).decode())
        dt = (
            all_times[frames[index + 1]] - all_times[frame]
            if index + 1 < len(frames)
            else float(np.median(np.diff(all_times)))
        )
        recorder.insert_sleep(float(dt))
        if index in [0, len(frames) // 2, len(frames) - 1]:
            np.savez_compressed(
                out / f"check_frame_{frame:04d}.npz",
                track_ids=shown,
                original=raw,
                corrected=corrected,
                labels=last_label[shown],
            )
        if index % 100 == 0:
            print(args.task, index, "/", len(frames), "native", len(ids), "shown", len(shown), flush=True)
    recording = recorder.serialize()
    server.stop()
    cap.release()
    decoded = msgspec.msgpack.decode(gzip.decompress(recording))
    duration = decoded["durationSeconds"]
    (out / "dense_shared_shift.viser").write_bytes(recording)
    metadata = dict(
        task=args.task,
        duration=duration,
        frame_ids=frames,
        frame_times=[r["time"] for r in rows],
        stats=rows,
        rgb_thumbnails=thumbs,
        time_stride=2,
        hold=3,
        source=str(source),
        native_frames=T,
    )
    legend = f"""<aside style="position:fixed;left:12px;top:12px;z-index:9999;width:310px;max-height:88vh;overflow:auto;background:#15202bf2;color:#eee;padding:12px;border-radius:8px;font:13px/1.5 system-ui"><b>{args.task} · 完整Track4World点云＋共用手部Z平移</b><div id="source"></div><div id="counts"></div><img id="rgb" style="width:100%"><p>蓝色URDF，橙色手，RGB物体／旧操作表面。无新SAM2二次裁点，无每物体1000点上限。沿用旧HTML全密度、每2帧显示及短缺口保留规则。</p><p>相机不自动旋转；鼠标旋转/缩放，空格暂停，拖时间轴。右侧眼睛切换02校正后／03校正前。</p><details><summary>深度与时间规则</summary>所有点共用手部Z平移，保留原XYZ的X/Y及原始UV。D435跳变门控和最多2帧短预测不变。深度仅在实际记录帧更新；中间RGB帧延续最近已到达的平移量，不读取未来值。<br>短缺口保留：最近4个显示帧中出现的轨迹暂留最后真实位置，与旧dense3d页面一致；不是新测量或未来插值。此页面不新增右侧光流副本／拖尾线。底层仍是旧多帧离线缓存。</details></aside>"""
    behavior = """<script>
let paused=false;const m=window.__VISER_EXPORT_METADATA__;
setInterval(()=>{const icon=document.querySelector('svg.tabler-icon-player-pause-filled');if(!paused&&icon){icon.closest('button').click();paused=true;}
const slider=Array.from(document.querySelectorAll('[role="slider"]')).find(e=>Math.abs(Number(e.getAttribute('aria-valuemax'))-m.duration)<.01);const t=slider?Number(slider.getAttribute('aria-valuenow')):0;let i=0;while(i+1<m.frame_times.length&&m.frame_times[i+1]<=t+1e-5)i++;
const r=m.stats[i];document.getElementById('source').textContent=`源帧 ${r.frame} · ${t.toFixed(2)}s · Z平移 ${(r.shift_m*1000).toFixed(1)}mm · ${r.status}`;
document.getElementById('counts').textContent=`当前真实观测 ${r.native_points}，显示 ${r.display_points}（含短缺口保留 ${r.historical_hold_points}）；物体/表面 ${r.object_surface_points}点`;
const image=document.getElementById('rgb');if(image.dataset.i!==String(i)){image.src='data:image/jpeg;base64,'+m.rgb_thumbnails[i];image.dataset.i=String(i);}
},150);</script>"""
    camera = dict(position=(first_center + [0.65, -0.2, -0.1]).tolist(), look_at=first_center.tolist(), up=[0, -1, 0])
    html = package(recording, metadata, camera, args.task + " 完整点云＋共用Z平移", legend).replace(
        "</body>", behavior + "</body>"
    )
    path = out / f"{args.task}_dense_shared_shift.html"
    path.write_text(html)
    summary = dict(
        task=args.task,
        native_rgb_frames=T,
        display_frames=len(frames),
        time_stride=2,
        hold=3,
        duration=duration,
        html_bytes=path.stat().st_size,
        source=str(source),
        new_mask_crop=False,
        instance_cap=None,
        old_native_membership_preserved=True,
        native_observations_checked=checked_raw,
        max_shift_roundoff_m=max_shift_error,
        display_points_median=float(np.median([r["display_points"] for r in rows])),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("pointflow_outputs/track4world_dense_shared_shift_fixed_20261104")
    )
    main(parser.parse_args())
