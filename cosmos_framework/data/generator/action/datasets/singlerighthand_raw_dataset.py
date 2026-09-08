# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-policy adapter for the single-right-hand raw episode format.

The raw LMDB files live on GPFS, where LMDB mmap is not supported. A small
one-time preprocessing step extracts only the aligned state/action arrays into
NPZ sidecars; videos continue to be decoded from the original episode MP4s.
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Joint, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id


@dataclass(frozen=True)
class _Episode:
    name: str
    num_frames: int
    source_fps: float


class _TorchCodecFrameReader:
    """Batch-decode CPU frames while the dataset cache handles window overlap."""

    def __init__(self, path: Path) -> None:
        from torchcodec.decoders import VideoDecoder

        self._decoder: Any = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=1)

    def get_frames(self, indices: np.ndarray) -> torch.Tensor:
        return self._decoder.get_frames_at(indices.tolist()).data

    def close(self) -> None:
        self._decoder = None


class SingleRightHandRawDataset(Dataset):
    """Read 30-fps single-right-hand episodes as official 15-fps policy windows."""

    EMBODIMENT_TYPE = "singlerighthand"

    def __init__(
        self,
        *,
        root: str,
        cache_root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        sample_stride: int = 1,
        mode: str = "wam",
        use_state: bool = True,
        viewpoint: str = "concat_view",
        task_text: str | None = None,
        use_image_augmentation: bool = False,
        use_precomputed_video: bool | str = False,
        video_cache_resolution: str | int | None = None,
        frame_cache_size: int | None = None,
    ) -> None:
        super().__init__()
        self._root = Path(root)
        self._cache_root = Path(cache_root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._use_state = bool(use_state)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id(self.EMBODIMENT_TYPE)
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None
        if isinstance(use_precomputed_video, str):
            use_precomputed_video = use_precomputed_video.lower() in {"1", "true", "yes", "on"}
        self._use_precomputed_video = bool(use_precomputed_video)
        if self._use_precomputed_video and self._use_image_augmentation:
            raise ValueError("Precomputed video frames cannot be combined with online image augmentation")

        manifest_path = self._cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run tools/prepare_singlerighthand_raw.py before training."
            )
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("schema_version", 0)) != 2:
            raise ValueError(
                f"Unsupported cache schema in {manifest_path}; rerun tools/prepare_singlerighthand_raw.py --overwrite"
            )
        self._arm_action_space = str(manifest.get("arm_action_space", "eef")).lower().strip()
        if self._arm_action_space not in {"eef", "joint"}:
            raise ValueError(
                f"Unsupported arm_action_space={self._arm_action_space!r} in {manifest_path}; expected 'eef' or 'joint'"
            )
        self._task_text = task_text or str(manifest.get("task_text", "make a sandwich"))

        all_episodes = [
            _Episode(
                name=str(row["name"]),
                num_frames=int(row["num_frames"]),
                source_fps=float(row["source_fps"]),
            )
            for row in manifest["episodes"]
        ]
        if not all_episodes:
            raise ValueError(f"No episodes listed in {manifest_path}")

        self._video_cache_rows: dict[str, dict[str, Any]] = {}
        if self._use_precomputed_video:
            video_manifest_path = self._cache_root / "video_manifest.json"
            if not video_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing {video_manifest_path}. Run tools/prepare_singlerighthand_video_cache.py first."
                )
            video_manifest = json.loads(video_manifest_path.read_text())
            if int(video_manifest.get("schema_version", 0)) != 1:
                raise ValueError(f"Unsupported video cache schema in {video_manifest_path}")
            if video_cache_resolution is not None and str(video_manifest.get("resolution")) != str(
                video_cache_resolution
            ):
                raise ValueError(
                    f"Video cache resolution={video_manifest.get('resolution')} does not match "
                    f"training resolution={video_cache_resolution}"
                )
            self._video_cache_rows = {str(row["name"]): row for row in video_manifest["episodes"]}
            missing = [episode.name for episode in all_episodes if episode.name not in self._video_cache_rows]
            if missing:
                raise ValueError(f"Video cache is missing {len(missing)} episodes; first missing episode: {missing[0]}")

        split_key = split.lower().strip()
        if split_key not in {"train", "val", "full"}:
            raise ValueError(f"Unsupported split={split!r}; expected train, val, or full")
        num_val = int(round(len(all_episodes) * split_val_ratio))
        order = torch.randperm(len(all_episodes), generator=torch.Generator().manual_seed(split_seed)).tolist()
        selected = order[:num_val] if split_key == "val" else order[num_val:] if split_key == "train" else order
        self._episodes = [all_episodes[idx] for idx in selected]

        self._source_strides: list[int] = []
        self._valid_windows: list[int] = []
        self._cumulative_ends: list[int] = []
        total = 0
        for episode in self._episodes:
            stride_float = episode.source_fps / self._fps
            source_stride = int(round(stride_float))
            if source_stride < 1 or not np.isclose(stride_float, source_stride, atol=1e-6):
                raise ValueError(
                    f"Episode {episode.name}: source_fps={episode.source_fps} is not an integer multiple of fps={self._fps}"
                )
            last_start = episode.num_frames - 1 - self._chunk_length * source_stride
            valid_windows = last_start // self._sample_stride + 1 if last_start >= 0 else 0
            self._source_strides.append(source_stride)
            self._valid_windows.append(valid_windows)
            total += valid_windows
            self._cumulative_ends.append(total)

        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._array_cache_size = 2
        self._reader_cache: OrderedDict[tuple[int, str], _TorchCodecFrameReader] = OrderedDict()
        self._reader_cache_size = 2
        self._video_array_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._video_array_cache_size = 2
        self._frame_cache_size = frame_cache_size or (self._chunk_length + 1) * max(self._source_strides)
        self._composed_frame_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self._composed_frame_cache_size = self._frame_cache_size

    @property
    def action_dim(self) -> int:
        return 27

    @property
    def action_spec(self) -> ActionSpec:
        if self._arm_action_space == "joint":
            return build_action_spec(
                Joint(n=7, label="arm", prefix="right"),
                Joint(n=20, label="hand", prefix="right"),
            )
        return build_action_spec(
            Pos(prefix="right"), Rot("quat_xyzw", prefix="right"), Joint(n=20, label="hand", prefix="right")
        )

    @property
    def arm_action_space(self) -> str:
        return self._arm_action_space

    @property
    def action_names(self) -> list[str]:
        return self.action_spec.names

    def __len__(self) -> int:
        return self._cumulative_ends[-1] if self._cumulative_ends else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        for length in self._valid_windows:
            if length > 0:
                blocks.append((start, length))
            start += length
        return blocks

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        episode_idx = bisect_right(self._cumulative_ends, idx)
        block_start = 0 if episode_idx == 0 else self._cumulative_ends[episode_idx - 1]
        return episode_idx, idx - block_start

    def _window_indices(self, episode_idx: int, window_offset: int) -> tuple[np.ndarray, np.ndarray]:
        source_stride = self._source_strides[episode_idx]
        raw_start = window_offset * self._sample_stride
        observation = raw_start + np.arange(self._chunk_length + 1, dtype=np.int64) * source_stride
        return observation, observation[:-1]

    def _load_arrays(self, episode_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._array_cache.get(episode_idx)
        if cached is not None:
            self._array_cache.move_to_end(episode_idx)
            return cached
        episode = self._episodes[episode_idx]
        with np.load(self._cache_root / "episodes" / f"{episode.name}.npz") as data:
            state = np.asarray(data["state"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
        if state.shape != (episode.num_frames, self.action_dim) or action.shape != state.shape:
            raise ValueError(
                f"Episode {episode.name}: expected state/action {(episode.num_frames, self.action_dim)}, "
                f"got {state.shape}/{action.shape}"
            )
        cached = (state, action)
        self._array_cache[episode_idx] = cached
        while len(self._array_cache) > self._array_cache_size:
            self._array_cache.popitem(last=False)
        return cached

    def _get_reader(self, episode_idx: int, camera: str) -> _TorchCodecFrameReader:
        key = (episode_idx, camera)
        reader = self._reader_cache.get(key)
        if reader is not None:
            self._reader_cache.move_to_end(key)
            return reader
        path = self._root / self._episodes[episode_idx].name / "videos" / f"{camera}.mp4"
        reader = _TorchCodecFrameReader(path)
        self._reader_cache[key] = reader
        while len(self._reader_cache) > self._reader_cache_size:
            _, old_reader = self._reader_cache.popitem(last=False)
            old_reader.close()
        return reader

    @staticmethod
    def _close_memmap(array: np.ndarray) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _get_precomputed_video(self, episode_idx: int) -> np.ndarray:
        cached = self._video_array_cache.get(episode_idx)
        if cached is not None:
            self._video_array_cache.move_to_end(episode_idx)
            return cached

        episode = self._episodes[episode_idx]
        row = self._video_cache_rows[episode.name]
        path = self._cache_root / str(row["path"])
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in row["shape"])
        if cached.dtype != np.uint8 or cached.shape != expected_shape:
            self._close_memmap(cached)
            raise ValueError(
                f"Episode {episode.name}: expected uint8 video cache {expected_shape}, got {cached.dtype} {cached.shape}"
            )
        if expected_shape[0] != episode.num_frames or expected_shape[1] != 3:
            self._close_memmap(cached)
            raise ValueError(
                f"Episode {episode.name}: video cache must have shape [num_frames,3,H,W], got {expected_shape}"
            )

        self._video_array_cache[episode_idx] = cached
        while len(self._video_array_cache) > self._video_array_cache_size:
            _, old_array = self._video_array_cache.popitem(last=False)
            self._close_memmap(old_array)
        return cached

    @staticmethod
    def _compose_views(head: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """Stack aspect-preserved wrist and head views at a shared width."""
        target_width = min(head.shape[-1], wrist.shape[-1])

        def resize_to_width(video: torch.Tensor) -> torch.Tensor:
            height, width = video.shape[-2:]
            target_height = max(1, round(height * target_width / width))
            if (height, width) == (target_height, target_width):
                return video
            return transforms_F.resize(
                video,
                [target_height, target_width],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        wrist = resize_to_width(wrist)
        head = resize_to_width(head)
        return torch.cat([wrist, head], dim=-2)  # [T,C,H_wrist+H_head,W]

    def _load_video(self, episode_idx: int, observation_indices: np.ndarray) -> torch.Tensor:
        missing_indices = [
            int(frame_idx)
            for frame_idx in observation_indices
            if (episode_idx, int(frame_idx)) not in self._composed_frame_cache
        ]
        if missing_indices:
            missing = np.asarray(missing_indices, dtype=np.int64)
            if self._use_precomputed_video:
                array = self._get_precomputed_video(episode_idx)
                composed = torch.from_numpy(np.array(array[missing], copy=True))
            else:
                head = self._get_reader(episode_idx, "head").get_frames(missing)
                wrist = self._get_reader(episode_idx, "right_wrist").get_frames(missing)
                composed = self._compose_views(head, wrist)
            for frame_idx, frame in zip(missing_indices, composed, strict=True):
                self._composed_frame_cache[(episode_idx, frame_idx)] = frame
            while len(self._composed_frame_cache) > self._composed_frame_cache_size:
                self._composed_frame_cache.popitem(last=False)

        frames = []
        for frame_idx_value in observation_indices:
            key = (episode_idx, int(frame_idx_value))
            frame = self._composed_frame_cache[key]
            self._composed_frame_cache.move_to_end(key)
            frames.append(frame)
        video = torch.stack(frames)
        if self._use_image_augmentation:
            if self._image_augmentor is None:
                height, width = video.shape[-2:]
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(height * 0.95), int(width * 0.95))),
                        T.Resize((height, width), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            video = self._image_augmentor(video)
        return video.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W], uint8

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, window_offset = self._resolve_index(idx)
        observation_indices, action_indices = self._window_indices(episode_idx, window_offset)
        state, action = self._load_arrays(episode_idx)
        raw_action = torch.from_numpy(action[action_indices].copy()).float()
        if self._use_state:
            initial_state = torch.from_numpy(state[observation_indices[0]].copy()).float()
            raw_action = torch.cat([initial_state.unsqueeze(0), raw_action], dim=0)

        mode = random.choice(("forward_dynamics", "inverse_dynamics", "wam")) if self._mode == "joint" else self._mode
        return {
            "ai_caption": self._task_text,
            "video": self._load_video(episode_idx, observation_indices),
            "action": raw_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "additional_view_description": (
                "The upper view is from the right wrist-mounted camera. "
                "The lower view is from the head-mounted third-person camera."
            ),
        }

    def __del__(self) -> None:
        for reader in getattr(self, "_reader_cache", {}).values():
            reader.close()
        for array in getattr(self, "_video_array_cache", {}).values():
            self._close_memmap(array)


__all__ = ["SingleRightHandRawDataset"]
