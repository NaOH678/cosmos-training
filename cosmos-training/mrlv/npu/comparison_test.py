"""Ensure the cross-device gate keeps numerical policies and loss tolerances."""

import copy

import pytest

from mrlv.compare_wam_cross_device import compare
from mrlv.compare_wam_litegen import load_recorded_metrics
from mrlv.litegen_bridge_test import BASELINE


def test_compile_difference_is_reported_without_an_a100_speed_gate():
    baseline = load_recorded_metrics(BASELINE, 5000)
    candidate = copy.deepcopy(baseline)
    candidate["config"]["model"]["config"]["compile"]["enabled"] = False
    candidate["performance"].update(device="npu", compile=False, mean_step_seconds=100)
    report = compare(baseline, candidate)
    assert report["status"] == "passed"
    assert report["expected_differences"]["candidate_device"] == "npu"


@pytest.mark.parametrize("change", ["batch", "clip", "loss", "nonfinite"])
def test_numerical_regressions_fail(change):
    baseline = load_recorded_metrics(BASELINE, 5000)
    candidate = copy.deepcopy(baseline)
    if change == "batch":
        candidate["performance"]["grad_accum_iter"] = 2
    elif change == "clip":
        candidate["config"]["trainer"]["callbacks"]["grad_clip"]["clip_norm"] = 999
    else:
        for row in candidate["rows"]:
            row["train/action_loss"] = float(row["train/action_loss"]) * (1.3 if change == "loss" else float("nan"))
    assert compare(baseline, candidate)["status"] == "failed"
