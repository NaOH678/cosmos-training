"""Regression checks for deduplicated DCP state and corrupt references."""

import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

from migration.npu.checkpoint_verify import read_small_state, verify_component, verify_checkpoint


@pytest.fixture
def replicated_checkpoint(tmp_path):
    directory = tmp_path / "scheduler"
    dcp.save({"last_epoch": 2, "rates": torch.tensor([0.01])}, checkpoint_id=directory)
    for rank in range(1, 8):
        (directory / f"__{rank}_0.distcp").touch()
    return directory


def test_replicated_scheduler_reads_without_nonempty_duplicate_files(replicated_checkpoint):
    result = verify_component(replicated_checkpoint, replicated=True)
    assert result["nonempty_shards"] == 1
    assert len(result["empty_unreferenced_shards"]) == 7
    assert read_small_state(replicated_checkpoint)["last_epoch"] == 2


def test_sharded_component_cannot_have_empty_rank_files(replicated_checkpoint):
    with pytest.raises(ValueError, match="Empty required DCP shard"):
        verify_component(replicated_checkpoint)


def test_missing_rank_file_cannot_pass(replicated_checkpoint):
    (replicated_checkpoint / "__7_0.distcp").unlink()
    with pytest.raises(ValueError, match="Missing or unexpected DCP rank files"):
        verify_component(replicated_checkpoint, replicated=True)


def test_truncated_referenced_file_cannot_pass(replicated_checkpoint):
    with (replicated_checkpoint / "__0_0.distcp").open("r+b") as stream:
        stream.truncate(1)
    with pytest.raises(ValueError, match="Truncated DCP reference"):
        verify_component(replicated_checkpoint, replicated=True)


def test_missing_metadata_reference_cannot_pass(replicated_checkpoint):
    path = replicated_checkpoint / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    metadata.storage_data.pop(next(iter(metadata.storage_data)))
    path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError, match="Incomplete DCP metadata chunk references"):
        verify_component(replicated_checkpoint, replicated=True)


def test_32_rank_rng_and_requested_step_are_checked(monkeypatch, tmp_path):
    from migration.npu import checkpoint_verify as module

    monkeypatch.setattr(module, "verify_component", lambda *args, **kwargs: {})
    trainer = {"iteration": 5000, **{f"rng_state_{rank}": {
        key: b"rng" for key in ("torch", "torch_npu", "numpy_packed_bytes", "random_packed_bytes")
    } for rank in range(32)}}
    monkeypatch.setattr(module, "read_small_state",
                        lambda path: {"last_epoch": 5000} if path.name == "scheduler" else trainer)
    result = verify_checkpoint(tmp_path, world_size=32, steps=5000)
    assert result["rng_ranks_read"] == list(range(32))
    with pytest.raises(ValueError, match="restore step 2"):
        verify_checkpoint(tmp_path, world_size=32)
    del trainer["rng_state_31"]["torch_npu"]
    with pytest.raises(ValueError, match="rank 31"):
        verify_checkpoint(tmp_path, world_size=32, steps=5000)
