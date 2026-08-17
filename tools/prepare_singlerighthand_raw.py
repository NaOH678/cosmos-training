#!/usr/bin/env python3
"""Extract Cosmos action state/targets from single-right-hand raw LMDB episodes."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-text", default="make a sandwich")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_state_action(
    raw_action: np.ndarray,
    observed_eef: np.ndarray,
    observed_hand_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the active right side and build matching 27-D state/action arrays."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    observed_eef = np.asarray(observed_eef, dtype=np.float32)
    observed_hand_deg = np.asarray(observed_hand_deg, dtype=np.float32)
    if raw_action.ndim != 2 or raw_action.shape[1] != 54:
        raise ValueError(f"Expected action shape [N,54], got {raw_action.shape}")
    if observed_eef.shape != (raw_action.shape[0], 14):
        raise ValueError(f"Expected observed EEF shape {(raw_action.shape[0], 14)}, got {observed_eef.shape}")
    if observed_hand_deg.shape != (raw_action.shape[0], 40):
        raise ValueError(f"Expected observed hand shape {(raw_action.shape[0], 40)}, got {observed_hand_deg.shape}")

    state = np.concatenate(
        [observed_eef[:, 7:14], np.deg2rad(observed_hand_deg[:, 20:40])],
        axis=1,
    )
    action = raw_action[:, 27:54].copy()
    action[:, 7:] = np.deg2rad(action[:, 7:])
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contains non-finite values")
    quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
    if not np.allclose(quat_norm, 1.0, atol=2e-3):
        raise ValueError(f"Observed right EEF quaternion norm outside tolerance: {quat_norm.min()}..{quat_norm.max()}")
    return state.astype(np.float32, copy=False), action.astype(np.float32, copy=False)


def _read_lmdb_arrays(lmdb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python-lmdb is required only for this preprocessing command. On this cluster use /usr/bin/python3."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="singlerighthand_lmdb_") as tmp:
        local_lmdb = Path(tmp) / "lmdb"
        shutil.copytree(lmdb_dir, local_lmdb)
        env = lmdb.open(str(local_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin(write=False) as txn:
                values = []
                for key in (b"action", b"/observations/eef", b"/observations/hand_joint_deg"):
                    payload = txn.get(key)
                    if payload is None:
                        raise KeyError(f"Missing LMDB key {key.decode()} in {lmdb_dir}")
                    values.append(np.asarray(pickle.loads(payload), dtype=np.float32))
        finally:
            env.close()
    return build_state_action(*values)


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    episode_output = output_root / "episodes"
    episode_output.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {raw_root}")

    manifest_rows = []
    for index, episode_dir in enumerate(episode_dirs):
        output_path = episode_output / f"{episode_dir.name}.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path) as cached:
                num_frames = int(cached["state"].shape[0])
        else:
            state, action = _read_lmdb_arrays(episode_dir / "lmdb")
            num_frames = len(state)
            temp_path = output_path.with_suffix(".npz.tmp")
            with temp_path.open("wb") as file:
                np.savez(file, state=state, action=action)
            temp_path.replace(output_path)

        with (episode_dir / "meta_info.pkl").open("rb") as file:
            metadata = pickle.load(file)
        expected_steps = int(metadata.get("num_steps", num_frames))
        if num_frames != expected_steps:
            raise ValueError(f"{episode_dir.name}: arrays={num_frames}, metadata num_steps={expected_steps}")
        for camera in ("head", "right_wrist"):
            if not (episode_dir / "videos" / f"{camera}.mp4").is_file():
                raise FileNotFoundError(episode_dir / "videos" / f"{camera}.mp4")
        manifest_rows.append(
            {
                "name": episode_dir.name,
                "num_frames": num_frames,
                "source_fps": float(metadata.get("frame_rate", 30.0)),
            }
        )
        print(f"[{index + 1:03d}/{len(episode_dirs):03d}] {episode_dir.name}: {num_frames} frames")

    manifest = {
        "schema_version": 2,
        "raw_root": str(raw_root),
        "task_text": args.task_text,
        "state_layout": "right_eef_xyz_quat_xyzw + right_hand_joint_rad_20",
        "action_layout": "right_eef_xyz_quat_xyzw + right_hand_target_rad_20",
        "episodes": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
