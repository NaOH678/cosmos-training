"""Same-frame, same-instance D435 vs cached DA3 axial-Z differences.

D435 is an independent noisy sensor, not ground truth. No model inference,
future fill, coordinate fitting or data mutation is performed.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np
from render_d435_point_accuracy import measured_cloud
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pointflow_outputs/cotracker_real_20261104"
OUT = SOURCE / "object_depth_comparison"
TASKS = ["bottle", "scissors", "dropper", "micro", "sandwich"]


def stats(error):
    error = np.asarray(error, dtype=np.float32)
    if not len(error):
        return {"pairs": 0}
    absolute = np.abs(error) * 1000
    return dict(
        pairs=len(error),
        bias_median_mm=float(np.median(error) * 1000),
        mae_mm=float(absolute.mean()),
        median_abs_mm=float(np.median(absolute)),
        p90_abs_mm=float(np.percentile(absolute, 90)),
        p95_abs_mm=float(np.percentile(absolute, 95)),
        fraction_gt50mm=float(np.mean(absolute > 50)),
        fraction_gt100mm=float(np.mean(absolute > 100)),
    )


def zbuffer(points, uv, valid, shape):
    buffer = np.full(shape, np.inf, np.float32)
    np.minimum.at(buffer, (uv[valid, 1], uv[valid, 0]), points[valid, 2].astype(np.float32))
    yy, xx = np.nonzero(np.isfinite(buffer))
    return np.column_stack((xx, (yy + 0.5) * 448 / 480 - 0.5)).astype(np.float32), buffer[yy, xx], xx, yy


def pair(uv, raw, corrected, sensor_uv, sensor_z):
    if not len(uv) or not len(sensor_uv):
        return None
    distance, index = cKDTree(sensor_uv).query(uv, workers=1)
    which = np.flatnonzero(distance <= 1.5)
    which = which[np.argsort(distance[which], kind="stable")]
    _, first = np.unique(index[which], return_index=True)
    which = which[first]
    return which, sensor_z[index[which]], raw[which], corrected[which]


def run(task):
    cv2.setNumThreads(1)
    tic = time.perf_counter()
    source = SOURCE / task
    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)
    episode = Path(json.loads((source / "source.json").read_text())["episode"])
    config = json.loads((source / "task_config.json").read_text())
    names = {i + 1: o["name"] for i, o in enumerate(config["objects"]) if o["label"] != 2}
    streams = json.loads((episode / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"][
        "head"
    ]["streams"]
    labels = np.load(source / "labels.npy", mmap_mode="r")
    instances = np.load(source / "instances.npy", mmap_mode="r")
    env = lmdb.open(str(Path("/tmp/cotracker_real_depth_20261104") / task), readonly=True, lock=False)
    with env.begin() as txn:
        frames = sorted(
            int(k.decode().split("/")[-1])
            for k in txn.cursor().iternext(keys=True, values=False)
            if k.startswith(b"depth/head/") and int(k.decode().split("/")[-1]) < len(labels)
        )
    accum = {}
    rows = []
    coverage = {}
    matched_frames = set()
    for number, frame in enumerate(frames):
        with env.begin() as txn:
            blob = txn.get(f"depth/head/{frame:06d}".encode())
        depth = cv2.imdecode(np.frombuffer(blob, np.uint8), -1)
        image = np.zeros((*labels.shape[1:], 3), np.uint8)
        points, _, _, sensor_instances, raw_flag, filtered, sensor_uv = measured_cloud(
            depth, streams, instances[frame], labels[frame], image, "cpu"
        )
        finite = np.isfinite(points).all(1)
        buffers = {}
        for mode, use in [
            ("unfiltered_sensor", raw_flag & finite & (points[:, 2] > 0.1) & (points[:, 2] < 6)),
            ("filtered_sensor", filtered & finite),
        ]:
            buffers[mode] = zbuffer(points, sensor_uv, use, labels.shape[1:])
        with np.load(source / "pointflow/frames" / f"{frame:06d}.npz") as n:
            observed_uv = n["uv"]
            raw_z = n["raw_xyz"][:, 2]
            corrected_z = n["xyz"][:, 2]
            conf = n["depth_conf"]
            assigned = n["instances"]
            final_valid = n["valid"]
        canvas = cv2.resize(instances[frame], (640, 448), interpolation=cv2.INTER_NEAREST)
        for instance, name in names.items():
            use = (
                (assigned == instance)
                & np.isfinite(raw_z)
                & np.isfinite(corrected_z)
                & (raw_z > 0.1)
                & (raw_z < 2.5)
                & (conf > 2)
            )
            uv = observed_uv[use]
            rz = raw_z[use]
            cz = corrected_z[use]
            fv = final_valid[use]
            coverage.setdefault(
                name,
                dict(
                    sensor_frames=len(frames),
                    frames_with_candidates=0,
                    frames_with_filtered_matches=0,
                    candidates=0,
                    filtered_pairs=0,
                ),
            )
            cov = coverage[name]
            cov["candidates"] += len(uv)
            cov["frames_with_candidates"] += int(len(uv) > 0)
            distance = cv2.distanceTransform((canvas == instance).astype(np.uint8), cv2.DIST_L2, 5)
            for sensor_mode, (suv, sz, xx, yy) in buffers.items():
                take = instances[frame][yy, xx] == instance
                result = pair(uv, rz, cz, suv[take], sz[take])
                if result is None:
                    continue
                which, sensor, raw, corrected = result
                if not len(which):
                    continue
                q = np.rint(uv[which]).astype(int)
                q[:, 0] = q[:, 0].clip(0, 639)
                q[:, 1] = q[:, 1].clip(0, 447)
                core = distance[q[:, 1], q[:, 0]] >= 3
                if sensor_mode == "filtered_sensor":
                    cov["frames_with_filtered_matches"] += 1
                    cov["filtered_pairs"] += len(which)
                    matched_frames.add(frame)
                for region, selection in [
                    ("all", np.ones(len(which), bool)),
                    ("core", core),
                    ("final_valid", fv[which]),
                ]:
                    raw_error = (raw - sensor)[selection]
                    new_error = (corrected - sensor)[selection]
                    for label, error in [("raw_DA3", raw_error), ("corrected_DA3", new_error)]:
                        key = (sensor_mode, region, name, label)
                        accum.setdefault(key, []).append(error.astype(np.float32))
                    if sensor_mode == "filtered_sensor" and region == "all":
                        rows.append(
                            dict(
                                frame=frame,
                                object=name,
                                instance=instance,
                                pairs=len(which),
                                candidates=len(uv),
                                sensor_z_median_m=float(np.median(sensor)),
                                raw_da3_z_median_m=float(np.median(raw)),
                                corrected_da3_z_median_m=float(np.median(corrected)),
                                raw=stats(raw_error),
                                corrected=stats(new_error),
                            )
                        )
        if number % 100 == 0:
            print(task, number, "/", len(frames), flush=True)
    env.close()
    summary = {}
    pooled = {}
    for (sensor_mode, region, name, label), chunks in accum.items():
        values = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
        summary.setdefault(sensor_mode, {}).setdefault(region, {}).setdefault(name, {})[label] = stats(values)
        pooled.setdefault((sensor_mode, region, label), []).append(values)
    total = {}
    for (sensor_mode, region, label), chunks in pooled.items():
        total.setdefault(sensor_mode, {}).setdefault(region, {})[label] = stats(np.concatenate(chunks))
    result = dict(
        task=task,
        episode=str(episode),
        native_sensor_frames=len(frames),
        matched_sensor_frames=len(matched_frames),
        coverage=coverage,
        totals=total,
        per_object=summary,
        elapsed_seconds=time.perf_counter() - tic,
    )
    (dest / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (dest / "per_frame.json").write_text(json.dumps(rows, ensure_ascii=False))
    print("DONE", task, len(frames), round(result["elapsed_seconds"], 1), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, TASKS))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = [
        "# 五类物体深度：D435 与 DA3 的差距",
        "",
        "比较已有五条完整视频中所有实际记录 D435 的帧。手部排除。深度均指对齐到 RGB 光学坐标后的轴向 Z，单位毫米；不是到相机的欧氏距离。",
        "",
        "主表使用与当前流程相同的 D435 范围、RGB可见性及同实例局部邻域过滤。DA3使用已有当前帧缓存，不重新预测。使用DA3置信度>2且Z在0.1–2.5m的物体候选；主表不先按最终三维过滤挑点，避免只比较看起来好的部分。",
        "",
        "以相同帧、相同实例配对，RGB标定后640×448像素坐标距离≤1.5px，一RGB传感器样本最多使用一次。原始DA3与校正后DA3使用完全相同的配对。统计为全部匹配点加权，较大物体/长时间可见物体权重更大。",
        "",
        "**D435不是物体深度真值：透明瓶、玻璃杯、细管可能读到背景，即使D435内部邻域一致也可能错误。差距小不代表准确。**",
        "",
        "|数据|D435帧|匹配点对|原DA3平均绝对差 mm|原DA3 P90 mm|校正后平均绝对差 mm|校正后P90 mm|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        t = r["totals"]["filtered_sensor"]["all"]
        a = t["raw_DA3"]
        b = t["corrected_DA3"]
        lines.append(
            f"|{r['task']}|{r['native_sensor_frames']}|{a['pairs']}|{a['mae_mm']:.2f}|{a['p90_abs_mm']:.2f}|{b['mae_mm']:.2f}|{b['p90_abs_mm']:.2f}|"
        )
    lines += [
        "",
        "## 每个物体（同一配对口径）",
        "",
        "|数据|SAM实例|原DA3平均绝对差 mm|校正后平均绝对差 mm|原DA3有符号中位差 mm|可匹配帧/有候选帧|",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        for name, v in r["per_object"]["filtered_sensor"]["all"].items():
            a = v["raw_DA3"]
            b = v["corrected_DA3"]
            c = r["coverage"][name]
            lines.append(
                f"|{r['task']}|{name}|{a['mae_mm']:.2f}|{b['mae_mm']:.2f}|{a['bias_median_mm']:+.2f}|{c['frames_with_filtered_matches']}/{c['frames_with_candidates']}|"
            )
    lines += [
        "",
        "有符号差定义为 DA3−D435，正值表示DA3估得更远。实例名称来自SAM，实例错分不等于物理类别；手与物体混分仍会影响本统计。",
        "",
        "summary.json 另外提供未做邻域过滤的D435、mask内部距边界≥3px区域、最终保留点子集三种诊断口径；不补无效深度、不使用未来帧。未匹配区域和帧见coverage；它们不被当作零误差。",
        "",
        "深度校正是当前HTML使用的全前景统一手部Z平移，不是按物体拟合D435。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
