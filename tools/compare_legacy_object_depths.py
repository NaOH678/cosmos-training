"""Compare original stable Track4World DA3 with matching recorded D435, by object.

Reads only legacy full-density observations, legacy SAM2 masks and recorded sensor
frames. No new CoTracker/DA3 outputs, model runs, historical display holds or
point interpolation are used as measurements.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np
from compare_real_object_depths import pair, stats, zbuffer
from render_d435_point_accuracy import measured_cloud

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pointflow_outputs/track4world_instance_core_html_20261104"
SHARED = ROOT / "pointflow_outputs/track4world_shared_hand_shift_20261104"
OUT = ROOT / "pointflow_outputs/legacy_track4world_object_depth_comparison_20261104"
HAND_OUT = ROOT / "pointflow_outputs/legacy_track4world_hand_depth_comparison_20261104"
TASK_NAMES = {"sandwich": "三明治", "dropper": "滴管", "micro": "移液枪"}
NAMES = {
    "plate": "盘子",
    "lettuce": "菜叶",
    "egg": "煎蛋",
    "tray": "托盘",
    "cheese": "芝士片",
    "bread_slice_1": "面包片1",
    "bread_slice_2": "面包片2",
    "dropper": "滴管",
    "water_bottle": "装水瓶",
    "beaker": "烧杯",
    "tube_rack_group": "试管架＋试管",
    "micropipette": "移液枪",
    "clear_test_tube": "透明试管",
    "red_tube_rack": "红色试管架",
    "red_liquid_beaker": "红液烧杯",
    "yellow_liquid_beaker": "右侧小杯",
    "pipette_stand": "移液枪架",
}


def run(task, selection="objects"):
    import utils3d

    cv2.setNumThreads(1)
    started = time.perf_counter()
    old = json.loads((CACHE / task / "manifest.json").read_text())
    source = Path(old["source"])
    episode = Path(old["episode"])
    maskdir = Path(old["mask"])
    init = json.loads((maskdir / "initialization.json").read_text())
    if selection not in ("objects", "hands"):
        raise ValueError(selection)
    names = {
        i + 1: d["object"]["name"]
        for i, d in enumerate(init["detections"])
        if (d["object"]["label"] == 2) == (selection == "hands")
    }
    arr = {
        k: np.load(source / (k + ".npy"), mmap_mode="r")
        for k in ["frame_offsets", "obs_pos", "obs_uv", "obs_valid", "obs_unique", "obs_label"]
    }
    labels = np.load(maskdir / "labels.npy", mmap_mode="r")
    instances = np.load(maskdir / "instances.npy", mmap_mode="r")
    streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    intr = streams["color"]["intrinsics"]
    env = lmdb.open(str(Path("/tmp/d435_point_accuracy_20261102") / task), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/")
        )
    shifts = {
        r["frame"]: r["shared_shift_m"] for r in json.loads((SHARED / task / "manifest.json").read_text())["stats"]
    }
    coverage = {
        name: dict(sensor_frames=len(frames), frames_with_da3=0, frames_with_pairs=0, da3_points=0, pairs=0)
        for name in names.values()
    }
    values = {name: {"original": [], "shared_hand_shift": [], "core_original": []} for name in names.values()}
    rows = []
    for number, frame in enumerate(frames):
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{frame:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        dummy = np.zeros((*labels.shape[1:], 3), np.uint8)
        sensor, _, _, _, raw_flag, _, sensor_uv = measured_cloud(
            depth, streams, instances[frame], labels[frame], dummy, "cpu"
        )
        # Exact sensor reference gate used by the existing legacy core comparison.
        with np.errstate(divide="ignore", invalid="ignore"):
            edge = utils3d.numpy.depth_edge(
                depth.astype(np.float32) * streams["depth"]["depth_scale_m"], rtol=0.04
            ).ravel()
        sensor_ok = raw_flag & np.isfinite(sensor).all(1) & (sensor[:, 2] > 0.1) & (sensor[:, 2] < 6) & ~edge
        suv, sz, xx, yy = zbuffer(sensor, sensor_uv, sensor_ok, labels.shape[1:])
        a, b = map(int, arr["frame_offsets"][frame : frame + 2])
        semantic_hand = np.isin(arr["obs_label"][a:b], [1, 2])
        use = (
            np.asarray(arr["obs_valid"][a:b])
            & np.asarray(arr["obs_unique"][a:b])
            & (semantic_hand if selection == "hands" else ~semantic_hand)
        )
        uv = np.asarray(arr["obs_uv"][a:b])[use].astype(np.float32)
        z = np.asarray(arr["obs_pos"][a:b])[use, 2].astype(np.float32)
        rgbq = np.rint(np.column_stack((uv[:, 0], (uv[:, 1] + 0.5) * 480 / 448 - 0.5))).astype(int)
        inside = (
            np.isfinite(uv).all(1)
            & np.isfinite(z)
            & (z > 0.1)
            & (z < 6)
            & (rgbq[:, 0] >= 0)
            & (rgbq[:, 0] < 640)
            & (rgbq[:, 1] >= 0)
            & (rgbq[:, 1] < 480)
        )
        assigned = np.zeros(len(uv), np.uint8)
        assigned[inside] = instances[frame][rgbq[inside, 1], rgbq[inside, 0]]
        canvas = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        for instance, name in names.items():
            which = inside & (assigned == instance)
            q_uv = uv[which]
            q_z = z[which]
            cov = coverage[name]
            cov["da3_points"] += len(q_z)
            cov["frames_with_da3"] += int(len(q_z) > 0)
            sensor_select = instances[frame][yy, xx] == instance
            paired = pair(q_uv, q_z, q_z + shifts[frame], suv[sensor_select], sz[sensor_select])
            if paired is None or not len(paired[0]):
                continue
            take, s_z, da3, corrected = paired
            error = da3 - s_z
            new_error = corrected - s_z
            cov["frames_with_pairs"] += 1
            cov["pairs"] += len(error)
            values[name]["original"].append(error)
            values[name]["shared_hand_shift"].append(new_error)
            distance = cv2.distanceTransform((canvas == instance).astype(np.uint8), cv2.DIST_L2, 5)
            q = np.rint(q_uv[take]).astype(int)
            q[:, 0] = q[:, 0].clip(0, 639)
            q[:, 1] = q[:, 1].clip(0, 447)
            core = distance[q[:, 1], q[:, 0]] >= 3
            values[name]["core_original"].append(error[core])
            rows.append(
                dict(
                    frame=frame,
                    instance=instance,
                    object=name,
                    pairs=len(error),
                    d435_z_median_m=float(np.median(s_z)),
                    da3_z_median_m=float(np.median(da3)),
                    original=stats(error),
                    shared_hand_shift=stats(new_error),
                    shared_offset_m=shifts[frame],
                )
            )
        if number % 100 == 0:
            print(task, number, "/", len(frames), flush=True)
    env.close()
    per_object = {}
    for name, data in values.items():
        summaries = {
            key: stats(np.concatenate(chunks) if chunks else np.empty(0, np.float32)) for key, chunks in data.items()
        }
        # Reuse original accepted core-estimation results rather than treating failed offsets=0 as matches.
        offsets = [
            p["offset_m"] for row in old["stats"] for p in row["parameters"] if p["name"] == name and p["accepted"]
        ]
        core_cache = dict(
            accepted_frames=len(offsets),
            total_frames=len(old["stats"]),
            median_sensor_minus_da3_mm=float(np.median(offsets) * 1000) if offsets else None,
            median_abs_sensor_minus_da3_mm=float(np.median(np.abs(offsets)) * 1000) if offsets else None,
        )
        per_object[name] = dict(coverage=coverage[name], depth_difference=summaries, previous_core_cache=core_cache)
    result = dict(
        task=task,
        episode=str(episode),
        source=str(source),
        mask=str(maskdir),
        sensor_lmdb=str(episode / "auxiliary_camera/depth.lmdb"),
        native_sensor_frames=len(frames),
        per_object=per_object,
        elapsed_seconds=time.perf_counter() - started,
    )
    dest = (HAND_OUT if selection == "hands" else OUT) / task
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (dest / "per_frame.json").write_text(json.dumps(rows, ensure_ascii=False))
    print("DONE", task, round(result["elapsed_seconds"], 1), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["sandwich", "dropper", "micro"]))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = [
        "# 旧版稳定 Track4World DA3 与对应 D435：逐物体深度差",
        "",
        "**只使用旧的三条数据，不使用新 CoTracker 产物或单帧 DA3。** 旧 DA3 数值直接取最终 Track4World `obs_pos[:,2]`；不重新预测、不改变尺度、不对统计点做显示暂留。",
        "",
        "|任务|episode|D435实际帧|旧Track4World数据源|",
        "|---|---|---:|---|",
    ]
    for r in results:
        lines.append(f"|{TASK_NAMES[r['task']]}|{Path(r['episode']).name}|{r['native_sensor_frames']}|{r['source']}|")
    lines += [
        "",
        "## 逐物体比较",
        "",
        "单位mm；深度为RGB光学坐标轴向Z。有符号差定义为DA3−D435：正值表示DA3更远，负值更近。平均绝对差和P90按全部匹配观测加权。",
        "",
        "|任务|物体|匹配帧/总D435帧|旧DA3平均绝对差|旧DA3有符号中位差|旧DA3绝对差P90|旧手部平移后平均绝对差|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        for name, v in r["per_object"].items():
            a = v["depth_difference"]["original"]
            b = v["depth_difference"]["shared_hand_shift"]
            c = v["coverage"]
            prefix = f"|{TASK_NAMES[r['task']]}|{NAMES.get(name, name)}|{c['frames_with_pairs']}/{r['native_sensor_frames']}|"
            lines.append(
                prefix
                + (
                    f"{a['mae_mm']:.2f}|{a['bias_median_mm']:+.2f}|{a['p90_abs_mm']:.2f}|{b['mae_mm']:.2f}|"
                    if a["pairs"]
                    else "无可配对点|—|—|—|"
                )
            )
    lines += [
        "",
        "## 统计口径与缓存复用",
        "",
        "- 三明治对应原 `sandwich_episode_0013_v2_dense3d.html` 的最终全密度点缓存；dropper、micro使用此前相应的Track4World最终缓存。物体划分使用此前为这些旧episode生成的全物体SAM2实例mask，不使用新五条数据的mask。",
        "- 排除旧缓存中的手/指尖点，及当前实例mask中的手。只读取旧obs_valid和obs_unique通过的真实观测；不读取HTML显示保持的历史点，不用每实例1000点展示抽样。",
        "- D435沿用之前配对的SDK标定、RGB最近Z可见性、原生深度边缘rtol=0.04和0.1–6m范围过滤。两者在640×448对应像素附近≤1.5px匹配；同一RGB传感器样本每帧最多使用一次。",
        "- “旧手部平移后”仅作辅助对照，直接复用此前 `track4world_shared_hand_shift_20261104/*/manifest.json` 的delta；主比较列为不加D435校正的旧稳定DA3，避免把D435参与过的校正当作独立深度估计。",
        "- 之前逐物体核心区配对已计算，缓存为 `track4world_instance_core_html_20261104/*/manifest.json`。本报告JSON的previous_core_cache直接汇总其accepted结果；失败时offset=0不会当成零差距。该核心区稳健中位差不等于整物体平均绝对差。",
        "- 无可配对点不是差距为0；可能是旧DA3源没有保存该物体、没有同时可见mask、或D435无有效对应。coverage区分有DA3点帧和匹配帧。",
        "- D435不是物体深度真值，透明杯、瓶和细管仍可能读到背景；数值只表示两个来源的差距，不判定谁更准确。",
        "",
        "每物体、每帧的D435/DA3深度中位数、差值分布和覆盖率保存在各任务per_frame.json、summary.json；额外提供mask内部≥3px统计。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
