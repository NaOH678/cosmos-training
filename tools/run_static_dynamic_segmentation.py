"""Reuse same-episode causal initialization; compare all/dynamic tracking and render three videos."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TASKS = {
    "sandwich": {
        "source": "pointflow_outputs/sandwich_bread_variants_20261030_v2/separate",
        "recipe": "examples/deployment/sandwich_static_dynamic_tracking.json",
        "episode": "/data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649",
    },
    "dropper_mix": {
        "source": "pointflow_outputs/dropper_micro_requested_20261030/dropper_mix",
        "recipe": "examples/deployment/dropper_mix_segmentation.json",
    },
    "micropipette": {
        "source": "pointflow_outputs/dropper_micro_requested_20261030/micropipette",
        "recipe": "examples/deployment/micropipette_segmentation.json",
    },
}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name, spec in TASKS.items():
        source = Path(spec["source"]).resolve()
        out = root / name
        out.mkdir()
        for file in (
            "frames",
            "video.json",
            "initial_masks.npy",
            "hand_allowed.npy",
            "hand_side.npy",
            "arm_core.npy",
            "geometry_timing.json",
            "initialization_preview.jpg",
        ):
            if (source / file).exists():
                (out / file).symlink_to(source / file)
        init = json.loads((source / "initialization.json").read_text())
        config = (
            json.loads((source / "task_config.json").read_text())
            if (source / "task_config.json").exists()
            else init["config"]
        )
        config.update(json.loads(Path(spec["recipe"]).read_text()))
        init["config"] = config
        init["initialization_reused_from_same_episode"] = str(source)
        (out / "task_config.json").write_text(json.dumps(config, indent=2))
        (out / "initialization.json").write_text(json.dumps(init, indent=2))
        if (source / "source.json").exists():
            shutil.copyfile(source / "source.json", out / "source.json")
        else:
            info = json.loads((source / "video.json").read_text())
            (out / "source.json").write_text(json.dumps(dict(episode=spec["episode"], **info), indent=2))
    for file in ("segment_sandwich_video.py", "run_sandwich_bread_variants.py", "run_static_dynamic_segmentation.py"):
        shutil.copyfile(Path("tools") / file, root / file)


def comparisons(root):
    report = {}
    for name in TASKS:
        out = root / name
        modes = {}
        for mode, file in (("all", "benchmark_all.json"), ("dynamic", "benchmark.json")):
            benchmark = json.loads((out / file).read_text())
            modes[mode] = dict(objects=len(benchmark["indices"]))
            for key in ("times", "sam_times", "rgb_load_times"):
                values = benchmark[key][5:]
                modes[mode][key] = dict(p50=float(np.median(values)), p90=float(np.percentile(values, 90)))
        modes["sam_reduction_percent"] = 100 * (
            1 - modes["dynamic"]["sam_times"]["p50"] / modes["all"]["sam_times"]["p50"]
        )
        modes["load_and_sam_reduction_percent"] = 100 * (
            1 - modes["dynamic"]["times"]["p50"] / modes["all"]["times"]["p50"]
        )
        modes["previous_run_ms"] = json.loads((Path(TASKS[name]["source"]) / "summary.json").read_text())[
            "single_gpu_all_objects_sam_ms"
        ]
        # Once full videos exist, compare the same dynamic IDs against the old
        # continuous all-object run at the beginning, middle and end.
        if (out / "summary.json").exists():
            baseline = Path(TASKS[name]["source"])
            old_masks = {}
            for worker in range(8):
                details = json.loads((baseline / f"worker{worker}.json").read_text())
                if details["indices"]:
                    array = np.load(baseline / f"worker{worker}_masks.npy", mmap_mode="r")
                    for pos, idx in enumerate(details["indices"]):
                        old_masks[idx] = (array, pos)
            frames = json.loads((out / "video.json").read_text())["frames"]
            ious = []
            for worker in range(8):
                details = json.loads((out / f"worker{worker}.json").read_text())
                if not details["indices"]:
                    continue
                array = np.load(out / f"worker{worker}_masks.npy", mmap_mode="r")
                for pos, idx in enumerate(details["indices"]):
                    previous, old_pos = old_masks[idx]
                    for frame in (0, 60, frames // 5, frames // 2, 4 * frames // 5, frames - 1):
                        a, b = array[frame, pos], previous[frame, old_pos]
                        union = np.count_nonzero(a | b)
                        ious.append(float(np.count_nonzero(a & b) / union) if union else 1.0)
            modes["sampled_dynamic_masks_vs_previous_full_run_iou"] = dict(
                mean=float(np.mean(ious)), minimum=float(np.min(ious)), samples=len(ious)
            )
        report[name] = modes
    result = dict(
        protocol="A800 GPU0, SAM2.1 Hiera large bf16, same 120 frames and same first-frame masks; first 5 excluded. Each mode runs with all its objects jointly on one GPU. sam_times excludes CPU JPEG load, URDF, postprocessing, initialization, serialization and camera/HTTP. times includes CPU RGB load. Full videos use eight object workers, never temporal chunks.",
        tasks=report,
    )
    (root / "latency_comparison.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# 静止模板与动态跟踪对比",
        "",
        "同一 A800 GPU0、SAM2.1 Hiera large、bf16；同一条数据及首帧初始化，分别单卡联合跟踪全部对象/动态对象。前120帧，排除前5帧。",
        "",
        "| 任务 | SAM2对象数 | SAM2中位耗时 ms/帧 | SAM2 P90 ms/帧 | 中位耗时降低 | 含RGB读取中位耗时 ms/帧 |",
        "|---|---|---|---|---|---|",
    ]
    for name, values in report.items():
        a, d = values["all"], values["dynamic"]
        lines.append(
            f"| {name} | {a['objects']} → {d['objects']} | {a['sam_times']['p50']:.2f} → {d['sam_times']['p50']:.2f} | "
            f"{a['sam_times']['p90']:.2f} → {d['sam_times']['p90']:.2f} | {values['sam_reduction_percent']:.1f}% | "
            f"{a['times']['p50']:.2f} → {d['times']['p50']:.2f} |"
        )
    lines += [
        "",
        "SAM2计时排除CPU RGB读取归一化，包含跟踪执行及同步；不包含初始化、URDF投影、mask后处理、视频输出、相机/HTTP和策略网络。含RGB口径可与先前报告对照，历史记录另存JSON。",
        "",
        "静止模板来自本条数据自己的首帧自动初始化，逐帧扣除动态mask、机械臂和明显RGB变化。模板不重建首帧被遮挡的部分；固定相机和静止物体是前提。训练预处理与推理须共用此规则，当前未集成真机在线接口。",
        "",
    ]
    for name in TASKS:
        lines.append(f"- [{name}完整视频]({name}/segmentation_h264.mp4)")
    (root / "latency_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pointflow_outputs/static_dynamic_segmentation_20261031"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        comparisons(args.root)
        return
    prepare(args.root)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "tools/run_sandwich_bread_variants.py",
            "--root",
            str(args.root),
            "--compare-all",
            "--variants",
            *TASKS,
        ],
        check=True,
        env=os.environ.copy(),
    )
    for name in TASKS:
        subprocess.run(
            [sys.executable, "tools/verify_robot_segmentation_outputs.py", str(args.root / name)], check=True
        )
    comparisons(args.root)


if __name__ == "__main__":
    main()
