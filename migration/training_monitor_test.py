"""Resume keeps one loss record per step and all valid performance windows."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from migration.training_monitor import TrainingMonitor, restore_history


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setenv("LITEGEN_DEVICE", "cpu")
    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))

    def start(iteration=0, warmup=0, max_iter=40):
        callback = TrainingMonitor(tmp_path / "tensorboard", warmup_steps=warmup)
        callback.config = NS(
            trainer=NS(logging_iter=10, max_iter=max_iter, grad_accum_iter=1, seed=42),
            dataloader_train=NS(dataloader=NS(batch_size=1)),
            model=NS(config=NS(precision="bfloat16", compile=NS(enabled=False))),
        )
        callback.on_train_start(None, iteration=iteration)
        return callback

    return start


def run_steps(callback, first, last):
    loss = torch.tensor(1.0)
    batch = torch.ones(1, 2)
    output = {"flow_matching_loss_vision": loss, "flow_matching_loss_action": loss}
    for iteration in range(first, last + 1):
        callback.on_training_step_batch_end(None, batch, output, loss, iteration=iteration - 1)
        callback.on_training_step_end(None, batch, output, loss, iteration=iteration)
    callback.on_train_end(None, iteration=last)


@pytest.mark.parametrize("warmup,measured", [(0, 40), (10, 20)])
def test_resume_replaces_abandoned_steps_and_merges_performance(tmp_path, monitor, warmup, measured):
    run_steps(monitor(warmup=warmup), 1, 30)
    names = ("metrics.csv", "performance_windows.csv", "performance.json")
    originals = {name: (tmp_path / name).read_bytes() for name in names}
    assert not (tmp_path / "resume-backups").exists()
    resumed = monitor(iteration=20, warmup=warmup)
    assert [int(r["iteration"]) for r in read_rows(tmp_path / "metrics.csv")] == [10, 20]
    run_steps(resumed, 21, 40)
    assert [int(r["iteration"]) for r in read_rows(tmp_path / "metrics.csv")] == [10, 20, 30, 40]
    windows = read_rows(tmp_path / "performance_windows.csv")
    assert [int(r["last_step"]) for r in windows] == [10, 20, 30, 40]
    summary = json.loads((tmp_path / "performance.json").read_text())
    valid = [r for r in windows if not r["excluded_reason"]]
    seconds = sum(float(r["seconds"]) for r in valid)
    assert summary["measured_steps"] == measured
    assert summary["mean_step_seconds"] == pytest.approx(seconds / measured)
    assert summary["samples_per_second"] == pytest.approx(measured / seconds)
    assert summary["segment_start_iteration"] == 20
    backup, = (tmp_path / "resume-backups").iterdir()
    assert {name: (backup / name).read_bytes() for name in names} == originals


def test_resume_between_logging_steps_does_not_invent_timing(tmp_path, monitor):
    run_steps(monitor(), 1, 30)
    run_steps(monitor(iteration=25), 26, 40)
    windows = restore_history(tmp_path, 40, 10)
    assert [w["first_step"] for w in windows] == [1, 11, 26, 31]
    assert json.loads((tmp_path / "performance.json").read_text())["measured_steps"] == 35


def test_final_partial_logging_window_is_retained(tmp_path, monitor):
    run_steps(monitor(max_iter=25), 1, 25)
    assert [r["last_step"] for r in restore_history(tmp_path, 25, 10)] == [10, 20, 25]


@pytest.fixture
def history(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "iteration,train/loss,train/video_loss,train/action_loss\n10,1,1,1\n20,2,2,2\n30,3,3,3\n"
    )
    (tmp_path / "performance_windows.csv").write_text(
        "first_step,last_step,steps,seconds,samples,peak_allocated_gib,peak_reserved_gib,excluded_reason\n"
        "1,10,10,10,10,1,2,\n11,20,10,20,10,3,4,\n21,30,10,30,10,5,6,\n"
    )
    return tmp_path


@pytest.mark.parametrize("damage", [
    "missing_metrics", "missing_windows", "missing_step", "duplicate_step", "nonfinite_loss",
    "incomplete_row", "wrong_header", "nonfinite_seconds", "overlapping_window", "wrong_step_count",
])
def test_invalid_history_is_rejected_without_rewriting(history, damage):
    metrics = history / "metrics.csv"
    windows = history / "performance_windows.csv"
    if damage.startswith("missing_") and damage != "missing_step":
        (metrics if damage == "missing_metrics" else windows).unlink()
    elif damage in ("missing_step", "duplicate_step", "nonfinite_loss", "incomplete_row", "wrong_header"):
        before, after = {
            "missing_step": ("20,2,2,2\n", ""),
            "duplicate_step": ("20,2,2,2\n", "20,2,2,2\n20,2,2,2\n"),
            "nonfinite_loss": ("20,2,2,2", "20,nan,2,2"),
            "incomplete_row": ("20,2,2,2", "20,2,2"),
            "wrong_header": ("train/action_loss", "unknown"),
        }[damage]
        metrics.write_text(metrics.read_text().replace(before, after))
    else:
        before, after = {
            "nonfinite_seconds": ("11,20,10,20", "11,20,10,nan"),
            "overlapping_window": ("11,20,10", "10,20,11"),
            "wrong_step_count": ("11,20,10", "11,20,9"),
        }[damage]
        windows.write_text(windows.read_text().replace(before, after))
    original = {p.name: p.read_bytes() for p in history.iterdir()}
    with pytest.raises(ValueError):
        restore_history(history, 20, 10)
    assert {p.name: p.read_bytes() for p in history.iterdir()} == original


def test_failed_replace_keeps_original_backups_and_can_retry(history, monkeypatch):
    original = {p.name: p.read_bytes() for p in history.iterdir()}
    replace = Path.replace

    def fail_windows(path, target):
        if target.name == "performance_windows.csv":
            raise OSError("volume unavailable")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_windows)
        with pytest.raises(OSError, match="volume unavailable"):
            restore_history(history, 20, 10)
    backup, = (history / "resume-backups").iterdir()
    assert {p.name: p.read_bytes() for p in backup.iterdir()} == original
    assert [w["last_step"] for w in restore_history(history, 20, 10)] == [10, 20]


def test_restart_from_zero_preserves_old_history_in_backup(history):
    assert restore_history(history, 0, 10) == []
    assert read_rows(history / "metrics.csv") == []
    backup, = (history / "resume-backups").iterdir()
    assert len(read_rows(backup / "metrics.csv")) == 3


def test_fresh_run_creates_no_backup_and_does_not_change_rng(tmp_path):
    state = torch.get_rng_state().clone()
    assert restore_history(tmp_path, 0, 10) == []
    assert list(tmp_path.iterdir()) == []
    assert torch.equal(torch.get_rng_state(), state)


def test_only_rank_zero_restores_history(tmp_path, monitor, monkeypatch):
    from migration import training_monitor

    monkeypatch.setattr(training_monitor.distributed, "is_rank0", lambda: False)
    monkeypatch.setattr(training_monitor, "restore_history", lambda *args: pytest.fail("nonzero rank must not write logs"))
    assert monitor(iteration=20).writer is None
    assert not (tmp_path / "metrics.csv").exists()
