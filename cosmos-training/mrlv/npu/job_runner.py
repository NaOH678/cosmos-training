"""Unattended ARM/NPU acceptance job; no accelerator imports in the supervisor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
LOSS_KEYS = ("train/loss", "train/video_loss", "train/action_loss")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    bad = [
        case.get("name") for case in cases if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))
    ]
    if len(cases) < 5 or bad:
        raise ValueError(
            f"Expected >=5 executed, passing NPU kernel tests; found {len(cases)} cases; incomplete: {bad}"
        )
    return {"executed_tests": len(cases), "skips": 0, "failures": 0}


def verify_wam(root, *, world_size=8, steps=2):
    root = Path(root)
    with (root / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not all(math.isfinite(float(row[key])) for row in rows for key in LOSS_KEYS):
        raise ValueError("WAM loss contains NaN/Inf")
    performance = json.loads((root / "performance.json").read_text())
    if performance["world_size"] != world_size or performance["device"] != "npu":
        raise ValueError(f"Expected {world_size} NPU training ranks")
    interval = performance["window_steps"]
    if not isinstance(interval, int) or interval < 1 or steps % interval:
        raise ValueError("Expected steps divisible by the recorded logging interval")
    if [int(row["iteration"]) for row in rows] != list(range(interval, steps + 1, interval)):
        raise ValueError(f"WAM did not record the complete metrics through step {steps}")
    configs = list(root.glob("cosmos3_action/action_sft/*/config.yaml"))
    if len(configs) != 1:
        raise ValueError("Expected one effective WAM configuration")
    runtime = json.loads((configs[0].parent / "litegen_runtime.json").read_text())
    if runtime["runtime"]["device"] != "npu" or runtime["optimization"]["parallel"]["strategy"] != "fsdp2":
        raise ValueError("Expected LiteGen NPU FSDP2 training")
    checkpoint = configs[0].parent / "checkpoints"
    checkpoint_name = f"iter_{steps:09d}"
    if (checkpoint / "latest_checkpoint.txt").read_text().strip() != checkpoint_name:
        raise ValueError("WAM final checkpoint is missing or has the wrong iteration")
    components = {}
    for name in ("model", "optim", "scheduler", "trainer"):
        folder = checkpoint / checkpoint_name / name
        shards = list(folder.glob("*.distcp"))
        nonempty = sum(p.stat().st_size > 0 for p in shards)
        # DCP deduplicates replicated scheduler state onto rank 0. Empty,
        # unreferenced scheduler files are valid; the schema stage checks every
        # metadata reference and reads scheduler/trainer state back from disk.
        if (not (folder / ".metadata").is_file() or len(shards) != world_size
                or nonempty == 0 or (name != "scheduler" and nonempty != world_size)):
            raise ValueError(f"Incomplete WAM DCP component: {name}")
        components[name] = {"shards": len(shards), "nonempty_shards": nonempty,
                            "bytes": sum(p.stat().st_size for p in shards)}
    return {"iterations": steps, "world_size": world_size, "device": "npu",
            "metrics": rows, "checkpoint_components": components}


class Job:
    def __init__(self, report, *, checks_only=False, image=None):
        self.report = report
        self.checks_only = checks_only
        self.env = dict(os.environ)
        self.active_process = None
        self.wam_root = None
        self.temporary_wam = False
        self.state = {
            "status": "running",
            "started_at": now(),
            "supervisor_pid": os.getpid(),
            "mode": "checks_only" if checks_only else "checks_and_wam_smoke",
            "requested_image": image,
            "host_architecture": platform.machine(),
            "stages": {},
            "report_directory": str(report),
            "scope": "NPU kernels, 2-rank FSDP2/DCP, 8-rank two-step WAM; no 5000-step alignment",
        }
        self.save()

    def save(self, **values):
        self.state.update(values, updated_at=now())
        write_json(self.report / "status.json", self.state)

    def stop_child(self):
        process = self.active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass

    def command(self, stage, command, *, timeout, env=None, capture=False):
        log_path = self.report / f"{stage}.log"
        started = time.monotonic()
        print(f"[{now()}] {stage}: {shlex.join(map(str, command))}", flush=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                list(map(str, command)),
                cwd=REPO,
                env=env or self.env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active_process = process
            position = 0
            try:
                while True:
                    try:
                        remaining = timeout - (time.monotonic() - started)
                        returncode = process.wait(timeout=min(5, max(0.01, remaining)))
                    except subprocess.TimeoutExpired:
                        returncode = None
                    if not capture:
                        with log_path.open("rb") as reader:
                            reader.seek(position)
                            data = reader.read()
                            position = reader.tell()
                        if data:
                            sys.stdout.write(data.decode(errors="replace"))
                            sys.stdout.flush()
                    self.save(current_stage=stage, stage_elapsed_seconds=round(time.monotonic() - started, 1))
                    if returncode is not None:
                        break
                    if time.monotonic() - started > timeout:
                        self.stop_child()
                        raise TimeoutError(f"{stage} exceeded {timeout} seconds; its worker group was stopped")
            finally:
                self.stop_child()
                self.active_process = None
        if returncode:
            raise RuntimeError(f"{stage} exited {returncode}; see {log_path}")
        return log_path.read_text(errors="replace") if capture else None

    def step(self, name, operation):
        self.state["stages"][name] = {"status": "running", "started_at": now()}
        self.save(current_stage=name)
        try:
            details = operation() or {}
        except BaseException as error:
            self.state["stages"][name].update(status="failed", finished_at=now(), error=str(error))
            self.save(status="failed", failed_stage=name, error=str(error))
            raise
        self.state["stages"][name].update(status="passed", finished_at=now(), details=details)
        self.save()

    def activate(self):
        if platform.machine() != "aarch64":
            raise RuntimeError("This job uses the prepared ARM environment; the allocated host must be aarch64")
        # Capture only environment settings relevant to Conda/CANN. Never dump
        # the complete platform environment (which may include credentials).
        code = (
            "import json,os; keys={'PATH','LD_LIBRARY_PATH','PYTHONPATH','CONDA_PREFIX',"
            "'CONDA_DEFAULT_ENV','TRAIN_PYTHON','LITEGEN_ROOT','COSMOS_NPU_ENV','COMPILE',"
            "'LITEGEN_DEVICE','COSMOS_DEVICE','IMAGEIO_FFMPEG_EXE','TOOLCHAIN_HOME'}; "
            "print('__COSMOS_ENV__'+json.dumps({k:v for k,v in os.environ.items() "
            "if k in keys or k.startswith(('ASCEND_','CANN_','ATB_'))}))"
        )
        output = self.command(
            "activation",
            [
                "bash",
                "-c",
                'set -e; source "$1"; "$TRAIN_PYTHON" -c "$2"',
                "cosmos-npu",
                REPO / "mrlv/npu/activate_conda.sh",
                code,
            ],
            timeout=120,
            capture=True,
        )
        lines = [line[len("__COSMOS_ENV__") :] for line in output.splitlines() if line.startswith("__COSMOS_ENV__")]
        if len(lines) != 1:
            raise ValueError("Could not obtain the activated ARM environment")
        self.env.update(json.loads(lines[0]))
        self.env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        self.python = self.env["TRAIN_PYTHON"]
        return {"python": self.python, "device": self.env["LITEGEN_DEVICE"]}

    def preflight(self):
        self.command(
            "preflight",
            [
                self.python,
                "mrlv/npu/preflight.py",
                "--expected-devices",
                "8",
                "--report",
                self.report / "preflight.json",
            ],
            timeout=300,
        )
        result = json.loads((self.report / "preflight.json").read_text())
        if result.get("status") != "passed" or "910b" not in result.get("device_name", "").lower():
            raise ValueError(f"Expected passing readiness and an actual 910B; detected {result.get('device_name')!r}")
        if not str(result["torch"]).split("+")[0] == "2.10.0" or result["torch_npu"] != "2.10.0.post6":
            raise ValueError("The activated Torch/TorchNPU versions differ from the prepared NPU lock")
        if not result["python"].startswith("3.12."):
            raise ValueError("Expected Python 3.12")
        return result

    def attention(self):
        xml = self.report / "attention.xml"
        self.command(
            "attention",
            [
                self.python,
                "-m",
                "pytest",
                "--confcutdir=mrlv/npu",
                "-o",
                "addopts=",
                "mrlv/npu/hardware_test.py",
                "-q",
                "-rs",
                f"--junitxml={xml}",
            ],
            timeout=900,
        )
        return verify_junit(xml)

    def distributed(self):
        self.command(
            "fsdp_dcp",
            [
                self.python,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "--master_addr=127.0.0.1",
                "--master_port=29674",
                "mrlv/npu/distributed_smoke.py",
                "--device",
                "npu",
                "--output-dir",
                self.report / "fsdp-dcp",
            ],
            timeout=1200,
        )
        result = json.loads((self.report / "fsdp-dcp/result.json").read_text())
        if result.get("status") != "passed" or result.get("world_size") != 2 or not result.get("step_after_restore"):
            raise ValueError("Distributed optimizer/RNG/DCP round-trip did not pass")
        return result

    def wam(self):
        persistent = self.env.get("NPU_WAM_OUTPUT_ROOT")
        if persistent:
            self.wam_root = Path(persistent).resolve()
            self.wam_root.mkdir(parents=True, exist_ok=False)
        else:
            scratch = Path(self.env.get("NPU_SCRATCH_ROOT", "/tmp"))
            scratch.mkdir(parents=True, exist_ok=True)
            self.wam_root = Path(tempfile.mkdtemp(prefix="cosmos-wam-910b-", dir=scratch))
            self.temporary_wam = True
        free = shutil.disk_usage(self.wam_root).free / 2**30
        self.save(wam_output_directory=str(self.wam_root), checkpoint_is_temporary=self.temporary_wam)
        if free < 40:
            raise ValueError(f"WAM output filesystem has {free:.2f} GiB free; requires >=40 GiB for its complete DCP")
        env = dict(self.env)
        env.update(
            OUTPUT_ROOT=str(self.wam_root),
            NPROC_PER_NODE="8",
            NNODES="1",
            NODE_RANK="0",
            GLOBAL_BATCH_SIZE="8",
            SENSECORE_ACCELERATE_DEVICE_COUNT="8",
            SENSECORE_PYTORCH_NNODES="1",
            SENSECORE_PYTORCH_NODE_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29684",
            MAX_ITER="2",
            SAVE_ITER="2",
            BATCH_SIZE="1",
            GRAD_ACCUM_ITER="1",
            COMPILE="false",
            RUN_VALIDATION="false",
            LOGGING_ITER="1",
            ALLOW_RESUME="0",
            BASELINE_MONITOR="1",
            JOB_NAME="sim_task27_wam_910b_smoke",
        )
        self.command(
            "wam_smoke",
            ["bash", "mrlv/launch_wam_litegen.sh", "--smoke"],
            timeout=int(env.get("NPU_WAM_TIMEOUT_SECONDS", "3600")),
            env=env,
        )
        result = verify_wam(self.wam_root)
        self.command(
            "checkpoint_schema",
            [self.python, "-m", "mrlv.npu.checkpoint_verify", self.wam_root,
             "--report", self.report / "checkpoint_schema.json"],
            timeout=120,
        )
        write_json(self.report / "wam_result.json", result)
        return result

    def preserve_wam(self):
        if self.wam_root is None or not self.wam_root.exists():
            return
        destination = self.report / "wam_artifacts"
        names = {
            "metrics.csv",
            "performance.json",
            "performance_windows.csv",
            "action_stats.json",
            "config.yaml",
            "litegen_runtime.json",
            "litegen_sharding.json",
            "latest_checkpoint.txt",
        }
        for source in self.wam_root.rglob("*"):
            if source.is_symlink() or not source.is_file():
                continue
            if source.name not in names and source.name != ".metadata" and source.suffix != ".log":
                continue
            target = destination / source.relative_to(self.wam_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if self.temporary_wam:
            shutil.rmtree(self.wam_root)
            self.save(temporary_checkpoint_removed=True)

    def run(self):
        failure = None
        try:
            self.step("activation", self.activate)
            self.step("preflight", self.preflight)
            self.step("attention", self.attention)
            self.step("fsdp_dcp", self.distributed)
            if not self.checks_only:
                self.step("wam_smoke", self.wam)
        except Exception as error:
            print(f"[{now()}] JOB FAILED: {error}", file=sys.stderr, flush=True)
            failure = error
        finally:
            self.stop_child()
            try:
                self.preserve_wam()
            except Exception as error:
                print(f"[{now()}] Artifact preservation failed: {error}", file=sys.stderr, flush=True)
                self.save(artifact_error=str(error))
                failure = failure or error
        if failure is not None:
            self.save(status="failed", finished_at=now(), error=str(failure))
            return 1
        self.save(status="passed", finished_at=now(), wam_executed=not self.checks_only)
        print(f"[{now()}] JOB PASSED; reports: {self.report}", flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true", help="Run kernels and tiny FSDP/DCP; omit real 4B WAM")
    parser.add_argument(
        "--report-root", type=Path, help="New persistent report directory; defaults under workspace/outputs"
    )
    args = parser.parse_args()
    if args.report_root:
        report = args.report_root.resolve()
        report.mkdir(parents=True, exist_ok=False)
    else:
        base = WORKSPACE / "outputs"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
        report = Path(tempfile.mkdtemp(prefix=f"wam-910b-job-{stamp}-", dir=base))
    print(f"NPU job reports: {report}", flush=True)
    job = Job(report, checks_only=args.checks_only, image=os.environ.get("NPU_JOB_IMAGE"))

    def interrupted(signum, _frame):
        job.stop_child()
        raise RuntimeError(f"Task interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return job.run()


if __name__ == "__main__":
    raise SystemExit(main())
