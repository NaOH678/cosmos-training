"""Shared environments must be complete before any training process starts."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from check_environment import check_ready


@pytest.mark.parametrize("state", ["ready", "missing", "different_lock", "moved"])
def test_shared_environment_readiness(tmp_path, state):
    lock = tmp_path / "requirements.lock"
    lock.write_text("example==1.0\n")
    prefix = tmp_path / "venv"
    prefix.mkdir()
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if state != "missing":
        marker = f"{digest}:{prefix}"
        if state == "different_lock":
            marker = f"wrong:{prefix}"
        elif state == "moved":
            marker = f"{digest}:/old/environment"
        (prefix / ".cosmos-npu-lock").write_text(marker)
    if state == "ready":
        check_ready(prefix, lock)
    else:
        with pytest.raises(ValueError, match="incomplete, moved"):
            check_ready(prefix, lock)


@pytest.mark.parametrize("failure,dry_run", [("", False), ("", True), ("environment", False), ("pip", False), ("preflight", False)])
def test_submission_checks_before_training_without_installing(tmp_path, failure, dry_run):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(Path(__file__).with_name("submit_sensecore.sh"), scripts)
    interpreter = tmp_path / "python"
    interpreter.write_text('''#!/bin/bash
set -eu
if [[ "$1" == scripts/check_environment.py ]]; then stage=environment
elif [[ "$*" == "-m pip check" ]]; then stage=pip
elif [[ "$*" == *mrlv.npu.preflight* ]]; then stage=preflight
else exit 91
fi
[[ "$LD_LIBRARY_PATH" == /vendor/cann/libraries ]]
if [[ "$stage" == preflight ]]; then
    [[ "$PYTHONPATH" == /vendor/cann/python ]]
else
    [[ -z "${PYTHONPATH+x}" ]]
fi
echo "$stage" >> "$STAGES"
if [[ "$stage" == "$FAIL_STAGE" ]]; then exit 17; fi
''')
    interpreter.chmod(0o755)
    (scripts / "activate.sh").write_text(f'export TRAIN_PYTHON="{interpreter}"\n'
                                        'export PYTHONPATH=/vendor/cann/python LD_LIBRARY_PATH=/vendor/cann/libraries\n')
    (tmp_path / "train.sh").write_text('''set -eu
[[ "$NNODES:$NPROC_PER_NODE:$NODE_RANK" == 4:8:2 ]]
[[ "$PYTHONPATH" == /vendor/cann/python && "$LD_LIBRARY_PATH" == /vendor/cann/libraries ]]
echo train >> "$STAGES"
''')
    stages = tmp_path / "stages"
    env = {"PATH": os.environ["PATH"], "WORLD_SIZE": "4", "RANK": "2", "MASTER_ADDR": "master.test",
           "MASTER_PORT": "23456", "OUTPUT_ROOT": str(tmp_path / "output"), "STAGES": str(stages), "FAIL_STAGE": failure}
    command = ["bash", str(scripts / "submit_sensecore.sh"), *(["--dry-run"] if dry_run else [])]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    expected = ["environment", "pip", *([] if dry_run else ["preflight"]), "train"]
    if failure:
        expected = expected[:expected.index(failure) + 1]
    assert result.returncode == (17 if failure else 0), result.stderr
    assert stages.read_text().splitlines() == expected


@pytest.mark.parametrize("fail_check", [False, True])
def test_bootstrap_never_installs_into_published_environment(tmp_path, fail_check):
    prefix = tmp_path / "published"
    (prefix / "bin").mkdir(parents=True)
    (prefix / ".cosmos-shared").write_text("published")
    interpreter = prefix / "bin/python"
    interpreter.write_text('''#!/bin/bash
set -eu
if [[ "$1" == */check_environment.py && "$*" == *--require-ready* ]]; then
    echo validate >> "$STAGES"
    exit "$CHECK_EXIT"
elif [[ "$*" == "-m pip check" ]]; then
    echo pip-check >> "$STAGES"
else exit 91
fi
''')
    interpreter.chmod(0o755)
    stages = tmp_path / "stages"
    env = {"PATH": os.environ["PATH"], "COSMOS_NPU_ENV": str(prefix),
           "NPU_BOOTSTRAP_PYTHON": "/must/not/create/another/environment",
           "NPU_BOOTSTRAP_REPORT_ROOT": str(tmp_path / "reports"),
           "STAGES": str(stages), "CHECK_EXIT": "17" if fail_check else "0"}
    result = subprocess.run(["bash", str(Path(__file__).with_name("bootstrap.sh"))],
                            env=env, capture_output=True, text=True)
    assert result.returncode == (17 if fail_check else 0), result.stderr
    assert stages.read_text().splitlines() == (["validate"] if fail_check else ["validate", "pip-check"])
    assert not Path(f"{prefix}.bootstrap.lock").exists()
