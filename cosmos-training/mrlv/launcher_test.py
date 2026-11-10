"""Exercise the real shell entry points without importing accelerator runtimes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

LAUNCHER = Path(__file__).with_name("launch_wam_litegen.sh")


@pytest.fixture
def launch(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'cosmos_framework.scripts.train']:\n"
        "    print(json.dumps({'overrides': sys.argv[sys.argv.index('--')+1:],\n"
        "                      'topology': [os.environ[k] for k in ('NNODES', 'NPROC_PER_NODE', 'NODE_RANK')]}))\n"
        "elif len(sys.argv) > 1 and sys.argv[-1].endswith('manifest.json'):\n"
        "    print('1\\n1\\ncamera')\n"
        "elif len(sys.argv) > 1 and sys.argv[1].endswith('prepare_normalized_run.py'):\n"
        "    print('/unused/action_stats.json\\ndigest')\n"
        "elif sys.argv[1:3] == ['-m', 'torch.distributed.run']:\n"
        "    print(json.dumps({'torchrun': sys.argv[3:]}))\n"
        "else:\n"
        "    sys.stdin.read()\n"
    )
    interpreter.chmod(0o755)
    for name in (
        "model/config.json", "model/tokenizer_config.json", "model/tokenizer.json",
        "checkpoint/model/.metadata", "vae.pth", "data/normalization_train_only.json",
        "data/datasets/cosmos-cache/pointflow_fk_handguided1024/manifest.json",
        "data/datasets/cosmos-cache/window_index.json",
        "data/datasets/cosmos-cache/vae_window_latents/window_manifest.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "data/datasets/cosmos-cache/video_frames").mkdir()
    env = {
        "PATH": os.environ["PATH"], "TRAIN_PYTHON": str(interpreter), "LITEGEN_DEVICE": "npu",
        "NNODES": "1", "NPROC_PER_NODE": "8", "NODE_RANK": "0", "MASTER_ADDR": "10.0.0.1",
        "OUTPUT_ROOT": str(tmp_path / "output"), "WAM_TMPDIR": str(tmp_path / "scratch"),
        "EDGE_DROID_MODEL_PATH": str(tmp_path / "model"),
        "BASE_CHECKPOINT_PATH": str(tmp_path / "checkpoint"), "WAN_VAE_PATH": str(tmp_path / "vae.pth"),
        "SIM_POINTFK_BUNDLE": str(tmp_path / "data"),
    }

    def run(overrides=None, *, smoke=False, dry_run=True):
        current = {**env, **(overrides or {})}
        current = {k: str(v) for k, v in current.items() if v is not None}
        return subprocess.run(
            ["bash", str(LAUNCHER), *(["--dry-run"] if dry_run else []), *(["--smoke"] if smoke else [])],
            env=current, capture_output=True, text=True, timeout=20,
        )

    return run


def test_launch_forwards_node_topology_to_torchrun(launch):
    result = launch({"NNODES": 4, "NODE_RANK": 3}, dry_run=False)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])["torchrun"]
    assert {"--nnodes=4", "--nproc_per_node=8", "--node_rank=3",
            "--master_addr=10.0.0.1", "--master_port=29644"}.issubset(args)
    assert "model.config.parallelism.data_parallel_shard_degree=32" in args


@pytest.mark.parametrize("nodes,cards", [(1, 8), (2, 8), (4, 8), (1, 16), (1, 32)])
def test_default_global_batch_and_full_world_sharding(launch, nodes, cards):
    result = launch({"NNODES": nodes, "NPROC_PER_NODE": cards, "NODE_RANK": nodes - 1})
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    overrides = record["overrides"]
    assert record["topology"] == list(map(str, (nodes, cards, nodes - 1)))
    assert f"model.config.parallelism.data_parallel_shard_degree={nodes * cards}" in overrides
    assert "model.config.parallelism.data_parallel_replicate_degree=1" in overrides
    assert f"dataloader_train.dataloader.batch_size={128 // nodes // cards}" in overrides
    assert "trainer.max_iter=5000" in overrides
    assert "scheduler.cycle_lengths=[40000]" in overrides
    assert "global_batch=128" in result.stdout


def test_canonical_topology_overrides_platform_aliases(launch):
    result = launch({"NNODES": 4, "NODE_RANK": 3, "SENSECORE_PYTORCH_NNODES": 1,
                     "SENSECORE_PYTORCH_NODE_RANK": 0, "SENSECORE_ACCELERATE_DEVICE_COUNT": 2})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["topology"] == ["4", "8", "3"]


def test_legacy_platform_aliases_still_work(launch):
    result = launch({"NNODES": None, "NODE_RANK": None, "NPROC_PER_NODE": None,
                     "SENSECORE_PYTORCH_NNODES": 2, "SENSECORE_PYTORCH_NODE_RANK": 1,
                     "SENSECORE_ACCELERATE_DEVICE_COUNT": 8})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["topology"] == ["2", "8", "1"]


@pytest.mark.parametrize("smoke,settings,batch,global_batch", [
    (True, {}, 1, 8),
    (True, {"NNODES": 4}, 1, 32),
    (False, {"NNODES": 4, "GRAD_ACCUM_ITER": 2}, 2, 128),
    (False, {"NNODES": 4, "BATCH_SIZE": 16}, 16, 512),
    (False, {"NNODES": 4, "GLOBAL_BATCH_SIZE": 256}, 8, 256),
])
def test_batch_and_smoke_policies(launch, smoke, settings, batch, global_batch):
    result = launch(settings, smoke=smoke)
    assert result.returncode == 0, result.stderr
    assert f"batch/rank={batch} global_batch={global_batch}" in result.stdout
    if smoke:
        assert "trainer.max_iter=2" in result.stdout
        assert "trainer.run_validation=false" in result.stdout


@pytest.mark.parametrize("settings,message", [
    ({"NNODES": 0}, "NNODES must be a positive integer"),
    ({"NNODES": "1+3"}, "NNODES must be a positive integer"),
    ({"NPROC_PER_NODE": 0}, "NPROC_PER_NODE must be a positive integer"),
    ({"NODE_RANK": 1}, "NODE_RANK must be in"),
    ({"NNODES": 4, "NODE_RANK": None}, "requires NODE_RANK"),
    ({"NNODES": 4, "NODE_RANK": 4}, "NODE_RANK must be in"),
    ({"NNODES": 4, "MASTER_ADDR": None}, "requires MASTER_ADDR"),
    ({"NNODES": 4, "MASTER_ADDR": "127.0.0.1"}, "must be reachable"),
    ({"NNODES": 4, "OUTPUT_ROOT": None}, "same absolute OUTPUT_ROOT"),
    ({"NNODES": 4, "OUTPUT_ROOT": "relative"}, "same absolute OUTPUT_ROOT"),
    ({"MASTER_PORT": 70000}, "Invalid MASTER_PORT"),
    ({"GLOBAL_BATCH_SIZE": 127}, "must be divisible"),
    ({"GLOBAL_BATCH_SIZE": 128, "BATCH_SIZE": 1}, "conflicts"),
    ({"GLOBAL_BATCH_SIZE": "bad"}, "must be a positive integer"),
    ({"GRAD_ACCUM_ITER": 0}, "must be a positive integer"),
])
def test_invalid_launch_fails_before_training(launch, settings, message):
    result = launch(settings)
    assert result.returncode != 0
    assert message in result.stderr
    assert '"overrides"' not in result.stdout
