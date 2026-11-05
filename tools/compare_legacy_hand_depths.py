"""Hand axial-Z comparison on the same three legacy episodes and sensor frames."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from compare_legacy_object_depths import CACHE, HAND_OUT, TASK_NAMES, run

NAMES = {"robot_hand": "右手", "right_hand": "右手", "left_hand": "左手"}


def worker(task):
    result = run(task, "hands")
    result["per_hand"] = result.pop("per_object")
    (HAND_OUT / task / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    HAND_OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["sandwich", "dropper", "micro"]))
    (HAND_OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = [
        "# 相同旧数据：D435 与 Track4World DA3 手部深度差",
        "",
        "**与上一轮逐物体统计使用相同的三条旧episode、旧最终Track4World点云、D435帧及配对规则；不使用CoTracker或新单帧DA3。**",
        "",
        "单位mm，比较RGB光学坐标中的轴向Z。有符号差=DA3−D435，正值为DA3更远、负值为更近。主表未加D435平移，避免循环校准后再评价；平均绝对差/P90按所有有效匹配点加权。左右手按此前SAM2实例mask区分，并要求源缓存也属于手/指尖语义。",
        "",
        "|任务|episode|手|匹配帧/总深度帧|匹配点对|平均绝对差|有符号中位差|绝对差P90|",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    csvrows = []
    for result in results:
        for name, v in result["per_hand"].items():
            s = v["depth_difference"]["original"]
            coverage = v["coverage"]
            prefix = f"|{TASK_NAMES[result['task']]}|{Path(result['episode']).name}|{NAMES.get(name, name)}|{coverage['frames_with_pairs']}/{result['native_sensor_frames']}|{s['pairs']}|"
            lines.append(
                prefix
                + (
                    f"{s['mae_mm']:.2f}|{s['bias_median_mm']:+.2f}|{s['p90_abs_mm']:.2f}|"
                    if s["pairs"]
                    else "无有效配对|—|—|"
                )
            )
            csvrows.append(
                dict(
                    task=result["task"],
                    episode=Path(result["episode"]).name,
                    hand=name,
                    paired_frames=coverage["frames_with_pairs"],
                    sensor_frames=result["native_sensor_frames"],
                    paired_points=s["pairs"],
                    mae_mm=s.get("mae_mm"),
                    signed_median_mm=s.get("bias_median_mm"),
                    abs_p90_mm=s.get("p90_abs_mm"),
                    core_mae_mm=v["depth_difference"]["core_original"].get("mae_mm"),
                    old_shared_shift_mae_mm=v["depth_difference"]["shared_hand_shift"].get("mae_mm"),
                )
            )
    lines += [
        "",
        "## 先前已经保存的手核心区校准结果",
        "",
        "直接汇总旧 `track4world_instance_core_html_20261104/*/manifest.json` 中通过主峰及MAD检查的核心区结果，不重估。原offset=median(D435−DA3)，下表取负号以统一为DA3−D435。它是逐帧核心区稳健差值的中位数，不是上表整手MAE。",
        "",
        "|任务|手|通过配对帧/总深度帧|核心区有符号中位差mm|",
        "|---|---|---|---:|",
    ]
    for result in results:
        for name, v in result["per_hand"].items():
            s = v["previous_core_cache"]
            value = s["median_sensor_minus_da3_mm"]
            lines.append(
                f"|{TASK_NAMES[result['task']]}|{NAMES.get(name, name)}|{s['accepted_frames']}/{s['total_frames']}|"
                + (f"{-value:+.2f}|" if value is not None else "无有效配对|")
            )
    lines += [
        "",
        "## 口径和来源",
        "",
        "- 统计239/449/423个原始D435实际记录帧；不使用HTML暂留的历史点，不补未来帧。",
        "- D435经SDK depth-to-color标定、RGB最近Z可见性、原生深度边缘rtol=0.04及0.1–6m过滤。与手mask中旧DA3真实观测在640×448画布内≤1.5px配对，每帧每个RGB传感器样本最多使用一次。",
        "- 同时匹配的旧DA3点取obs_valid与obs_unique，且源语义为手/指尖。全手统计和原核心区校准口径不同，不能混用；JSON另存mask内距边缘≥3px的统计。",
        "- 双手mask不意味着旧缓存包含双手三维点。缺失时明确记无有效配对，不能记为0差距。",
        "- D435也存在手部反光、遮挡和飞深度；这些是两个来源的深度差，不是以D435为真值的准确率。",
        "",
        "源数据：",
    ]
    for r in results:
        lines.append(f"- {r['task']}: `{r['source']}`；对应原始数据 `{r['episode']}`。")
    lines += [
        "",
        f"此前核心区缓存：`{CACHE}`。逐帧配对深度中位数与差值分布在各任务 `per_frame.json`，完整统计在 `summary.json`，可导出表为 `per_hand.csv`。",
    ]
    (HAND_OUT / "README.md").write_text("\n".join(lines) + "\n")
    with (HAND_OUT / "per_hand.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    print("COMPLETE", HAND_OUT, flush=True)


if __name__ == "__main__":
    main()
