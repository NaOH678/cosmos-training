"""Keep the newest complete checkpoint after a successful local DCP save."""

import re
import shutil
from pathlib import Path

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


def prune_older_checkpoints(directory, iteration):
    """Remove older iteration directories only after the latest pointer is committed."""
    root = Path(directory).resolve()
    current_name = f"iter_{iteration:09d}"
    latest = root / "latest_checkpoint.txt"
    if not latest.is_file() or latest.read_text().strip() != current_name:
        log.warning(f"Checkpoint retention skipped: latest pointer does not identify {current_name}")
        return []
    current = root / current_name
    components = ("model", "optim", "scheduler", "trainer")
    if current.is_symlink() or not all((current / name / ".metadata").is_file() for name in components):
        log.warning(f"Checkpoint retention skipped: {current_name} is not a complete training checkpoint")
        return []
    removed = []
    for candidate in sorted(root.iterdir()):
        match = re.fullmatch(r"iter_([0-9]{9,})", candidate.name)
        if not match or int(match[1]) >= iteration or candidate.is_symlink() or not candidate.is_dir():
            continue
        if candidate.resolve().parent != root:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate.name)
        log.info(f"Checkpoint retention: removed {candidate.name}; kept {current_name}")
    return removed


class KeepLatestCheckpoint(Callback):
    """Registered separately from optional TensorBoard/performance monitoring."""

    def on_train_start(self, model, iteration=0):
        if self.trainer.checkpointer.save_to_object_store:
            raise ValueError("KeepLatestCheckpoint requires a local or mounted filesystem output directory")

    def on_save_checkpoint_success(self, iteration=0, elapsed_time=0):
        if distributed.is_rank0():
            prune_older_checkpoints(self.trainer.checkpointer.save_dirname, iteration)
