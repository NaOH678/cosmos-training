"""Validate saved WAM DCP references and read back the small training states."""

import argparse
import json
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def verify_component(folder, *, world_size=8, replicated=False):
    folder = Path(folder).resolve()
    shards = {p.name: p for p in folder.glob("*.distcp")}
    expected = {f"__{rank}_0.distcp" for rank in range(world_size)}
    if set(shards) != expected:
        raise ValueError(f"Missing or unexpected DCP rank files: {folder}")
    nonempty = {name for name, path in shards.items() if path.stat().st_size > 0}
    if not nonempty or (not replicated and nonempty != expected):
        raise ValueError(f"Empty required DCP shard: {folder}")
    metadata = FileSystemReader(folder).read_metadata()
    if not metadata.state_dict_metadata or not metadata.storage_data:
        raise ValueError(f"Empty DCP metadata: {folder}")
    references = set()
    indices = set()
    for index, info in metadata.storage_data.items():
        path = (folder / info.relative_path).resolve()
        if path.parent != folder or path.name not in shards:
            raise ValueError(f"Missing or invalid DCP reference: {path}")
        if info.offset < 0 or info.length <= 0 or path.stat().st_size < info.offset + info.length:
            raise ValueError(f"Truncated DCP reference: {path} ({index.fqn})")
        references.add(path.name)
        indices.add((index.fqn, tuple(index.offset) if index.offset is not None else None))
    expected_indices = set()
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, TensorStorageMetadata):
            expected_indices.update((key, tuple(chunk.offsets)) for chunk in value.chunks)
        else:
            expected_indices.add((key, None))
    if indices != expected_indices:
        raise ValueError(f"Incomplete DCP metadata chunk references: {folder}")
    if references != nonempty:
        raise ValueError(f"DCP shard/reference mismatch: {folder}")
    return {
        "metadata_keys": len(metadata.state_dict_metadata),
        "referenced_chunks": len(indices),
        "rank_files": len(shards),
        "nonempty_shards": len(nonempty),
        "empty_unreferenced_shards": sorted(set(shards) - nonempty),
        "referenced_shards": sorted(references),
        "reference_bounds_checked": True,
    }


def read_small_state(folder):
    state = {}
    # Same reconstruction path used by Torch's dcp_to_torch_save. Read only
    # scheduler/trainer here; never materialize the full model on this process.
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(folder),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state


def verify_checkpoint(checkpoint, *, world_size=8):
    checkpoint = Path(checkpoint)
    components = {
        name: verify_component(checkpoint / name, world_size=world_size, replicated=name == "scheduler")
        for name in ("model", "optim", "scheduler", "trainer")
    }
    scheduler = read_small_state(checkpoint / "scheduler")
    trainer = read_small_state(checkpoint / "trainer")
    if scheduler["last_epoch"] != 2 or trainer["iteration"] != 2:
        raise ValueError("DCP scheduler/trainer did not restore step 2")
    rng_ranks = []
    for rank in range(world_size):
        state = trainer[f"rng_state_{rank}"]
        if not {"torch", "torch_npu", "numpy_packed_bytes", "random_packed_bytes"}.issubset(state):
            raise ValueError(f"Incomplete DCP RNG state for rank {rank}")
        rng_ranks.append(rank)
    return {
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "components": components,
        "scheduler_last_epoch": scheduler["last_epoch"],
        "trainer_iteration": trainer["iteration"],
        "rng_ranks_read": rng_ranks,
        "small_states_read_back": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = list(args.root.glob("cosmos3_action/action_sft/*/checkpoints/iter_000000002"))
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one WAM checkpoint at step 2")
    result = verify_checkpoint(checkpoints[0])
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
