"""Rebuild the four archived runs' full-range figures and checked statistics."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cosmos-training"))
from mrlv.compare_wam_litegen import LOSS_KEYS, load_recorded_metrics, numerical_config

RUNS = (
    ("wam-steps5k-8xA100", "A | 8 x A100 / Native", "#808080", "-"),
    ("wam-steps5k-8xA100-LiteGen", "B | 8 x A100 / LiteGen", "#00a83b", "-"),
    ("wam-steps5k-8x910B-LiteGen", "C | 8 x 910B / LiteGen", "#0057ff", "--"),
    ("wam-steps5k-32x910B-LiteGen", "D | 32 x 910B / LiteGen", "#ff0000", "--"),
)
STEPS = np.arange(10, 5001, 10)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "svg.hashsalt": "cosmos-four-runs-v1",
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=180, facecolor="white")
    svg = HERE / f"{name}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def configuration_differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            yield from configuration_differences(before.get(key), after.get(key), f"{path}.{key}")
    elif before != after:
        yield {"field": path.lstrip("."), "baseline": before, "candidate": after}


def loss_differences(baseline, candidate):
    assert baseline.shape == candidate.shape == (500,)
    signed = 100 * (candidate.mean() / baseline.mean() - 1)
    windows = 100 * (candidate.reshape(10, 50).mean(axis=1) / baseline.reshape(10, 50).mean(axis=1) - 1)
    return {"full_mean_signed_percent": float(signed), "full_mean_absolute_percent": float(abs(signed)),
            "window_mean_signed_percent": windows.tolist()}


def main():
    for scale, expected in ((1.0, 0.0), (1.1, 10.0), (0.9, -10.0)):
        check = loss_differences(np.ones(500), np.full(500, scale))
        np.testing.assert_allclose(check["full_mean_signed_percent"], expected, atol=1e-12)
        np.testing.assert_allclose(check["full_mean_absolute_percent"], abs(expected), atol=1e-12)
        np.testing.assert_allclose(check["window_mean_signed_percent"], expected, atol=1e-12)
    sources, records, curves = {}, {}, {}
    baseline_config = baseline_stats = None
    for name, label, color, style in RUNS:
        root = ROOT / "reference" / name
        run = load_recorded_metrics(root, 5000)
        loss = pd.DataFrame(run["rows"]).astype(float)
        windows = pd.read_csv(root / "performance_windows.csv", keep_default_na=False)
        perf = run["performance"]
        np.testing.assert_array_equal(loss["iteration"], STEPS)
        assert np.isfinite(loss[list(LOSS_KEYS)]).all().all()
        assert (loss[list(LOSS_KEYS)] > 0).all().all(), "Log-scale loss plots require positive values"
        np.testing.assert_array_equal(windows["last_step"], STEPS)
        np.testing.assert_array_equal(windows["first_step"], STEPS - 9)
        assert (windows["steps"] == 10).all()
        assert np.isfinite(windows[["seconds", "samples", "peak_allocated_gib", "peak_reserved_gib"]]).all().all()
        assert (windows["seconds"] > 0).all()
        assert (windows["samples"] == windows["steps"] * 128).all()
        assert perf["status"] == "measured" and perf["global_batch"] == 128
        assert perf["grad_accum_iter"] == 1 and perf["precision"] == "bfloat16" and perf["seed"] == 42
        assert perf["world_size"] * perf["batch_per_gpu"] == 128
        valid = windows["excluded_reason"] == ""
        steady = windows[valid]
        recomputed = {
            "steady_windows": len(steady),
            "measured_steps": int(steady["steps"].sum()),
            "mean_step_seconds": float(steady["seconds"].sum() / steady["steps"].sum()),
            "median_window_step_seconds": float((steady["seconds"] / steady["steps"]).median()),
            "samples_per_second": float(steady["samples"].sum() / steady["seconds"].sum()),
            "peak_allocated_gib": float(windows["peak_allocated_gib"].max()),
            "peak_reserved_gib": float(windows["peak_reserved_gib"].max()),
        }
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, perf[key], rtol=1e-10, err_msg=f"{name}/{key}")
        assert len(steady) == 492 and recomputed["measured_steps"] == 4920
        config = numerical_config(run["config"])
        stats = json.loads((root / "action_stats.json").read_text())
        if baseline_config is None:
            baseline_config, baseline_stats = config, stats
        assert stats == baseline_stats
        for path in sorted(root.rglob("*")):
            if path.is_file():
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[name] = {
            "label": label,
            "metric_points": len(loss),
            "configuration_differences_from_A": list(configuration_differences(baseline_config, config)),
            "performance": perf,
            "derived_performance": {
                "samples_per_second_per_card": perf["samples_per_second"] / perf["world_size"],
                "steady_rate_5000_step_hours_estimate": perf["mean_step_seconds"] * 5000 / 3600,
                "steady_rate_5000_step_card_hours_estimate": perf["mean_step_seconds"] * 5000 * perf["world_size"] / 3600,
            },
            "excluded_windows": windows.loc[~valid, ["first_step", "last_step", "excluded_reason"]].to_dict("records"),
            "loss": {key: {"recorded_mean": float(loss[key].mean()),
                           "step_10": float(loss[key].iloc[0]), "step_5000": float(loss[key].iloc[-1])}
                     for key in LOSS_KEYS},
        }
        curves[name] = (loss, windows, valid)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.84, hspace=0.29)
    for ax, key, title in zip(axes, LOSS_KEYS, ("Total loss", "Video loss", "Action loss")):
        for name, label, color, style in RUNS:
            values = curves[name][0][key]
            ax.plot(STEPS, values, color=color, linestyle=style, alpha=0.24, linewidth=0.7)
            ax.plot(STEPS, values.rolling(10, min_periods=1).mean(), color=color,
                    linestyle=style, linewidth=1.8, label=label)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        if key != "train/video_loss":
            ax.set_yscale("log")
        ax.set_ylabel("Loss (log scale)" if key != "train/video_loss" else "Loss (linear scale)")
        ax.set_xlim(0, 5000)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Cosmos 4B WAM | Full 5,000-step loss comparison", y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.94, "Faint: all 500 recorded points/run | Bold: trailing 10-point mean (~100 steps)", ha="center")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    save(fig, "loss_full")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.80, hspace=0.28)
    for name, label, color, style in RUNS:
        _, windows, valid = curves[name]
        for ax, values in zip(axes, (windows["seconds"] / windows["steps"], windows["samples"] / windows["seconds"])):
            ax.plot(windows["last_step"], values.where(valid), color=color, linestyle=style, linewidth=1.5, label=label)
    for ax, ylabel, title in zip(axes, ("Seconds / step", "Global samples / second"),
                                 ("10-step window mean latency (lower is better)", "10-step window throughput (higher is better)")):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 5000)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Training step (window end)")
    fig.suptitle("Cosmos 4B WAM | Training performance over 5,000 steps", y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.93, "492 measured windows/run | Excluded warmup, validation and checkpoint windows appear as gaps", ha="center", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)
    save(fig, "performance_curves")

    panels = (
        ("performance", "mean_step_seconds", "Mean step time", "Seconds / step | lower is better"),
        ("performance", "samples_per_second", "Global throughput", "Samples / second | higher is better"),
        ("derived_performance", "samples_per_second_per_card", "Throughput per card", "Samples / second / card | higher is better"),
        ("performance", "peak_allocated_gib", "Peak allocated memory", "GiB / card, max rank | lower is better"),
        ("performance", "peak_reserved_gib", "Peak reserved memory", "GiB / card, max rank | lower is better"),
        ("derived_performance", "steady_rate_5000_step_card_hours_estimate", "5,000-step card-hours (estimate)", "Card-hours at steady rate | lower is better"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.80, hspace=0.48, wspace=0.35)
    for ax, (group, key, title, unit) in zip(axes.flat, panels):
        values = [records[name][group][key] for name, *_ in RUNS]
        bars = ax.bar(list("ABCD"), values, color=[r[2] for r in RUNS], width=0.62)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=10)
        ax.set_ylim(0, max(values) * 1.20)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.suptitle("Cosmos 4B WAM | Efficiency and memory", y=0.98, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.93, "A: 8 x A100 Native    B: 8 x A100 LiteGen    C: 8 x 910B LiteGen    D: 32 x 910B LiteGen", ha="center")
    fig.text(0.5, 0.89, "Global batch = 128 | Timing: 4,920 steady steps | Memory: peak over all recorded windows", ha="center", fontsize=11)
    fig.text(0.5, 0.035, "Card-hours are resource-time estimates, not full job wall time, energy or monetary cost.", ha="center", fontsize=11)
    save(fig, "performance_summary")

    differences = {
        name: {key: loss_differences(curves[RUNS[0][0]][0][key].to_numpy(), curves[name][0][key].to_numpy())
               for key in LOSS_KEYS}
        for name, *_ in RUNS[1:]
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.11, top=0.80, hspace=0.38, wspace=0.23)
    titles = ("Total loss", "Video loss", "Action loss")
    ax = axes[0, 0]
    for index, (name, label, color, _) in enumerate(RUNS[1:]):
        values = [differences[name][key]["full_mean_absolute_percent"] for key in LOSS_KEYS]
        bars = ax.bar(np.arange(3) + (index - 1) * 0.25, values, width=0.25, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=9)
    ax.set_xticks(np.arange(3), titles)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    ax.set_title("Absolute full-run mean difference", loc="left", fontweight="bold")
    ax.set_ylabel("Absolute relative difference (%)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for ax, key, title in zip(list(axes.flat)[1:], LOSS_KEYS, titles):
        ax.axhline(0, color=RUNS[0][2], linewidth=1, label=RUNS[0][1] + " (reference)")
        for name, label, color, style in RUNS[1:]:
            ax.plot(np.arange(500, 5001, 500), differences[name][key]["window_mean_signed_percent"],
                    color=color, linestyle=style, marker="o", markersize=4, linewidth=1.8, label=label)
        ax.set_title(f"{title}: signed 500-step window difference", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Window end step")
        ax.set_ylabel("Signed relative difference (%)")
        ax.set_xlim(0, 5000)
    fig.suptitle("Loss differences relative to 8 x A100 / Native", y=0.98, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.93, "All statistics use unsmoothed recorded losses | Each window contains 50 points over 500 steps", ha="center", fontsize=11)
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)
    fig.text(0.5, 0.025, "Positive: above the native A100 mean. Negative: below it. Windows span the full run; no final-window zoom.", ha="center", fontsize=10)
    save(fig, "loss_difference")

    a, b, c, d = (records[name]["performance"] for name, *_ in RUNS)
    speedup = c["mean_step_seconds"] / d["mean_step_seconds"]
    summary = {
        "scope": "Four complete archived training-loss series and measured performance; no new training or DCP verification",
        "loss_plot": {"range": [10, 5000], "record_interval_steps": 10, "trailing_mean_points": 10,
                      "total_and_action_y_axis": "log", "video_y_axis": "linear"},
        "source_sha256": sources,
        "runs": records,
        "loss_differences_from_A": {"window_end_steps": list(range(500, 5001, 500)), "runs": differences},
        "comparisons": {
            "A100_LiteGen_vs_native_throughput_change_percent": 100 * (b["samples_per_second"] / a["samples_per_second"] - 1),
            "NPU_8_to_32_speedup": speedup,
            "NPU_8_to_32_scaling_efficiency_percent": 100 * speedup / 4,
            "NPU_8_to_32_card_hours_ratio": 4 / speedup,
            "NPU_32_vs_A100_native_throughput_ratio": d["samples_per_second"] / a["samples_per_second"],
            "NPU_32_vs_A100_LiteGen_throughput_ratio": d["samples_per_second"] / b["samples_per_second"],
        },
    }
    for path, digest in sources.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(records), "loss_points_per_run": 500, "steady_windows_per_run": 492,
                      "source_files": len(sources), "comparisons": summary["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
