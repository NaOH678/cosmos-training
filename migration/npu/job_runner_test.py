"""Acceptance supervisor gates, independent of Torch and accelerator hardware."""

import json
import sys

import pytest

from migration.npu.job_runner import Job, verify_junit, verify_wam


@pytest.mark.parametrize("incomplete", ["skipped", "failure", "error", "missing"])
def test_rejects_incomplete_hardware_acceptance(tmp_path, incomplete):
    cases = "".join(f'<testcase name="kernel_{i}"/>' for i in range(4))
    if incomplete != "missing":
        cases += f'<testcase name="last_kernel"><{incomplete}/></testcase>'
    xml = tmp_path / "hardware.xml"
    xml.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    with pytest.raises(ValueError, match="executed, passing NPU kernel tests"):
        verify_junit(xml)


def test_hardware_acceptance_requires_executed_tests(tmp_path):
    xml = tmp_path / "hardware.xml"
    xml.write_text("<testsuite>" + '<testcase name="kernel"/>' * 5 + "</testsuite>")
    assert verify_junit(xml)["executed_tests"] == 5


def test_worker_failure_is_persisted_and_stops_pipeline(tmp_path):
    job = Job(tmp_path, checks_only=True)
    job.activate = lambda: job.command(
        "activation", [sys.executable, "-c", "print('native import failed'); raise SystemExit(7)"], timeout=5
    )
    job.preflight = lambda: pytest.fail("must not start preflight after activation failure")
    assert job.run() == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_stage"] == "activation"
    assert "exited 7" in status["error"]
    assert "native import failed" in (tmp_path / "activation.log").read_text()


def test_timeout_stops_the_worker(tmp_path):
    job = Job(tmp_path)
    with pytest.raises(TimeoutError):
        job.command("stalled", [sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.1)
    assert job.active_process is None


def test_nonfinite_wam_loss_cannot_pass(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "iteration,train/loss,train/video_loss,train/action_loss\n1,1,1,1\n2,nan,1,1\n"
    )
    with pytest.raises(ValueError, match="NaN/Inf"):
        verify_wam(tmp_path)


@pytest.mark.parametrize("world_size", [8, 16, 32])
def test_wam_result_requires_requested_ranks_and_all_components(tmp_path, world_size):
    (tmp_path / "metrics.csv").write_text(
        "iteration,train/loss,train/video_loss,train/action_loss\n1,1,1,1\n2,1,1,1\n"
    )
    (tmp_path / "performance.json").write_text(json.dumps({
        "world_size": world_size, "device": "npu", "window_steps": 1,
    }))
    job = tmp_path / "cosmos3_action/action_sft/test"
    job.mkdir(parents=True)
    (job / "config.yaml").touch()
    (job / "litegen_runtime.json").write_text(json.dumps({
        "runtime": {"device": "npu"}, "optimization": {"parallel": {"strategy": "fsdp2"}},
    }))
    checkpoint = job / "checkpoints/iter_000000002"
    for name in ("model", "optim", "scheduler", "trainer"):
        component = checkpoint / name
        component.mkdir(parents=True)
        (component / ".metadata").write_bytes(b"metadata checked separately")
        for rank in range(world_size):
            (component / f"__{rank}_0.distcp").write_bytes(b"data" if name != "scheduler" or rank == 0 else b"")
    (checkpoint.parent / "latest_checkpoint.txt").write_text(checkpoint.name)
    assert verify_wam(tmp_path, world_size=world_size)["world_size"] == world_size
    with pytest.raises(ValueError, match="NPU training ranks"):
        verify_wam(tmp_path, world_size=world_size * 2)
    (checkpoint / "optim" / f"__{world_size-1}_0.distcp").unlink()
    with pytest.raises(ValueError, match="Incomplete WAM DCP"):
        verify_wam(tmp_path, world_size=world_size)


@pytest.mark.parametrize("temporary", [False, True])
def test_preserves_evidence_without_copying_checkpoint_payload(tmp_path, temporary):
    report = tmp_path / "report"
    report.mkdir()
    scratch = tmp_path / "owned-output"
    scratch.mkdir()
    (scratch / "metrics.csv").write_text("small diagnostic")
    (scratch / ".metadata").write_bytes(b"diagnostic metadata")
    (scratch / "rank0.distcp").write_bytes(b"weight payload")
    job = Job(report)
    job.wam_root = scratch
    job.temporary_wam = temporary
    job.preserve_wam()
    assert (report / "wam_artifacts/metrics.csv").read_text() == "small diagnostic"
    assert (report / "wam_artifacts/.metadata").is_file()
    assert not list(report.rglob("*.distcp"))
    assert scratch.exists() is not temporary


def test_artifact_failure_does_not_report_job_passed(tmp_path):
    job = Job(tmp_path, checks_only=True)
    job.activate = job.preflight = job.attention = job.distributed = lambda: None

    def failed_copy():
        raise OSError("diagnostic volume is full")

    job.preserve_wam = failed_copy
    assert job.run() == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert "volume is full" in status["artifact_error"]


def test_original_acceptance_pins_eight_ranks_even_with_multinode_environment(tmp_path, monkeypatch):
    from migration.npu import job_runner

    job = Job(tmp_path)
    job.python = sys.executable
    job.env.update(NNODES="4", NPROC_PER_NODE="16", NODE_RANK="3", GLOBAL_BATCH_SIZE="128",
                   NPU_WAM_OUTPUT_ROOT=str(tmp_path / "smoke"))
    monkeypatch.setattr(job_runner.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 80 * 2**30})())
    monkeypatch.setattr(job_runner, "verify_wam", lambda _: {})
    commands = []
    job.command = lambda stage, args, **kwargs: commands.append((stage, kwargs.get("env")))
    job.wam()
    env = commands[0][1]
    assert (env["NNODES"], env["NPROC_PER_NODE"], env["NODE_RANK"]) == ("1", "8", "0")
    assert env["GLOBAL_BATCH_SIZE"] == "8"
