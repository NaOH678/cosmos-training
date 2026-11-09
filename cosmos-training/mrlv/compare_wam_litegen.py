"""Compare full WAM training runs; incomplete/missing evidence cannot pass."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import yaml

LOSS_KEYS = ("train/loss", "train/video_loss", "train/action_loss")
META_KEYS = (
    "world_size",
    "precision",
    "compile",
    "batch_per_gpu",
    "global_batch",
    "grad_accum_iter",
    "seed",
    "base_checkpoint",
    "dataset_bundle",
    "warmup_steps",
    "window_steps",
)


def load_recorded_metrics(root, steps):
    """Read complete metrics/config; this alone does not prove a DCP exists."""
    root = Path(root)
    performance = json.loads((root / "performance.json").read_text())
    for key, value in performance.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{root}: nonfinite performance value {key}")
    with (root / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    iterations = [int(row["iteration"]) for row in rows]
    expected = list(range(10, steps + 1, 10))
    if iterations != expected:
        raise ValueError(
            f"{root}: expected complete ordered metrics 10..{steps} at interval 10; found {len(rows)} rows"
        )
    for row in rows:
        for key in LOSS_KEYS:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{root}: nonfinite {key} at {row['iteration']}")
    configs = list(root.glob("cosmos3_action/action_sft/*/config.yaml"))
    if len(configs) != 1:
        raise ValueError(f"{root}: expected exactly one effective config")
    config = yaml.safe_load(configs[0].read_text())
    return {"root": root, "performance": performance, "rows": rows, "config": config}


def load_run(root, steps):
    run = load_recorded_metrics(root, steps)
    checkpoint = next(Path(root).glob("cosmos3_action/action_sft/*/config.yaml")).parent / "checkpoints"
    expected_checkpoint = f"iter_{steps:09d}"
    if (checkpoint / "latest_checkpoint.txt").read_text().strip() != expected_checkpoint:
        raise ValueError(f"{root}: final checkpoint missing or wrong step")
    for component in ("model", "optim", "scheduler", "trainer"):
        folder = checkpoint / expected_checkpoint / component
        if not (folder / ".metadata").is_file() or not list(folder.glob("*.distcp")):
            raise ValueError(f"{root}: incomplete final DCP {component}")
    return run


def numerical_config(config):
    """Ignore output paths and labels; retain all numerical policies/callbacks."""
    import copy

    result = {
        key: copy.deepcopy(config[key])
        for key in ("model", "optimizer", "scheduler", "dataloader_train", "dataloader_val")
    }
    # Normalization copies move with the run; the SHA256 and values must match.
    for name in ("dataloader_train", "dataloader_val"):
        for entry in result[name]["dataloader"]["datasets"].values():
            entry["dataset"].pop("action_stats_path", None)
    result["checkpoint"] = {key: value for key, value in config["checkpoint"].items() if key != "type"}
    result["trainer"] = copy.deepcopy({key: value for key, value in config["trainer"].items() if key != "type"})
    result["trainer"]["callbacks"].get("baseline_monitor", {}).pop("log_dir", None)
    result["trainer"]["callbacks"].get("sampled_media", {}).pop("output_uri", None)
    return result


def compare(baseline, candidate, *, mean_rtol=0.05, window_rtol=0.10, performance_ratio=1.10):
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    check("numerical_config", numerical_config(baseline["config"]) == numerical_config(candidate["config"]))
    check(
        "normalization",
        json.loads((baseline["root"] / "action_stats.json").read_text())
        == json.loads((candidate["root"] / "action_stats.json").read_text()),
    )
    bp, cp = baseline["performance"], candidate["performance"]
    for key in META_KEYS:
        check(
            f"metadata/{key}",
            key in bp and key in cp and bp[key] == cp[key],
            baseline=bp.get(key),
            candidate=cp.get(key),
        )
    check(
        "performance_measured", bp.get("status") == cp.get("status") == "measured" and cp.get("measured_steps", 0) > 0
    )
    for key in LOSS_KEYS:
        bv = [float(row[key]) for row in baseline["rows"]]
        cv = [float(row[key]) for row in candidate["rows"]]
        relative = abs(statistics.mean(cv) - statistics.mean(bv)) / max(abs(statistics.mean(bv)), 1e-12)
        # The monitor records a sampled minibatch every ten steps. Aggregate
        # 50 samples (500 training steps) to compare the stochastic curves.
        windows = []
        for offset in range(0, len(bv), 50):
            bmean = statistics.mean(bv[offset : offset + 50])
            cmean = statistics.mean(cv[offset : offset + 50])
            windows.append(abs(cmean - bmean) / max(abs(bmean), 1e-12))
        check(f"{key}/mean", relative <= mean_rtol, relative_error=relative, tolerance=mean_rtol)
        check(f"{key}/500_step_windows", max(windows) <= window_rtol, relative_errors=windows, tolerance=window_rtol)
    for key in ("mean_step_seconds", "peak_allocated_gib", "peak_reserved_gib"):
        bvalue, cvalue = bp.get(key), cp.get(key)
        valid = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (bvalue, cvalue))
        check(
            f"performance/{key}",
            valid and cvalue <= bvalue * performance_ratio,
            baseline=bvalue,
            candidate=cvalue,
            max_ratio=performance_ratio,
        )
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.baseline.resolve() == args.candidate.resolve():
        parser.error("baseline and candidate must be different runs")
    try:
        report = compare(load_run(args.baseline, 5000), load_run(args.candidate, 5000))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"status": "incomplete", "error": str(error)}
    report.update(baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
