"""Compare cross-device WAM loss curves; report hardware performance separately.

Historical baseline checkpoints may have been deleted: baseline acceptance here
concerns its complete recorded metrics. The candidate must retain its final DCP.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from migration.compare_wam_litegen import LOSS_KEYS, META_KEYS, load_recorded_metrics, load_run, numerical_config


def compare(baseline, candidate, *, mean_rtol=0.05, window_rtol=0.10):
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    before, after = numerical_config(baseline["config"]), numerical_config(candidate["config"])
    # Only this numerical-policy difference is allowed; batch, accumulation,
    # clipping, LR, EMA and every other training policy must still match.
    for config in (before, after):
        config["model"]["config"]["compile"]["enabled"] = False
    check("numerical_config_except_compile_enabled", before == after)
    check(
        "normalization",
        json.loads((baseline["root"] / "action_stats.json").read_text())
        == json.loads((candidate["root"] / "action_stats.json").read_text()),
    )
    bp, cp = baseline["performance"], candidate["performance"]
    for key in META_KEYS:
        if key != "compile":
            check(
                f"metadata/{key}",
                key in bp and key in cp and bp[key] == cp[key],
                baseline=bp.get(key),
                candidate=cp.get(key),
            )
    check("equal_metric_count", len(baseline["rows"]) == len(candidate["rows"]) > 0)
    for key in LOSS_KEYS:
        bv = [float(row[key]) for row in baseline["rows"]]
        cv = [float(row[key]) for row in candidate["rows"]]
        relative = abs(statistics.mean(cv) - statistics.mean(bv)) / max(abs(statistics.mean(bv)), 1e-12)
        windows = []
        for start in range(0, min(len(bv), len(cv)), 50):
            bmean = statistics.mean(bv[start : start + 50])
            cmean = statistics.mean(cv[start : start + 50])
            windows.append(abs(cmean - bmean) / max(abs(bmean), 1e-12))
        check(
            f"{key}/mean",
            math.isfinite(relative) and relative <= mean_rtol,
            relative_error=relative,
            tolerance=mean_rtol,
        )
        check(
            f"{key}/500_step_windows",
            bool(windows) and all(math.isfinite(x) and x <= window_rtol for x in windows),
            relative_errors=windows,
            tolerance=window_rtol,
        )
    # Throughput/memory on a different chip is evidence, not an A100 speed gate.
    measured = bp.get("status") == cp.get("status") == "measured" and cp.get("measured_steps", 0) > 0
    for data in (bp, cp):
        measured = measured and all(
            isinstance(data.get(k), (int, float)) and math.isfinite(data[k]) and data[k] > 0
            for k in ("mean_step_seconds", "peak_allocated_gib", "peak_reserved_gib")
        )
    check("performance_measured", measured)
    return {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "scope": "loss-curve engineering gate; fixed-input operator and task evaluations remain separate",
        "checks": checks,
        "expected_differences": {
            "baseline_device": bp.get("device", "cuda"),
            "candidate_device": cp.get("device"),
            "baseline_compile": bp.get("compile"),
            "candidate_compile": cp.get("compile"),
            "optimizer": "CUDA FusedAdam; NPU LiteGen MasterShadowAdamW (FP32 masters/moments)",
        },
        "performance": {"baseline": bp, "candidate": cp},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.baseline.resolve() == args.candidate.resolve():
        parser.error("baseline and candidate must differ")
    try:
        report = compare(load_recorded_metrics(args.baseline, 5000), load_run(args.candidate, 5000))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"status": "incomplete", "error": str(error)}
    report.update(
        baseline=str(args.baseline.resolve()),
        candidate=str(args.candidate.resolve()),
        baseline_checkpoint_required=False,
        candidate_final_checkpoint_required=True,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
