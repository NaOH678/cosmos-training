"""Acceptance supervisor gates, independent of Torch and accelerator hardware."""

import json
import sys

import pytest

from mrlv.npu.job_runner import Job, verify_junit, verify_wam


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
