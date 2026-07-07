# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``DataPacker`` for action-policy SFT (video + action) on a LeRobot dataset.

Wraps :class:`ActionTransformPipeline` for per-sample processing and emits a
batch dict in the exact structure consumed by ``OmniMoTModel`` (mirrors
``cosmos_framework.inference.action.build_action_batch``):

* ``video``/``image`` key  -> list[ list[ Tensor[C,T,H,W] ] ]  (per-sample, per vision item)
* ``action``               -> list[ list[ Tensor[T, max_action_dim] ] ]
* ``image_size``           -> Tensor[B, 4]
* every other field        -> list of length B
"""

from __future__ import annotations

import math
from typing import Any

import torch

from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.data_packer import DataPacker


class ActionDataPacker(DataPacker):
    def __init__(
        self,
        tokenizer_config: dict | None = None,
        max_action_dim: int = 64,
        resolution: str = "480",
        input_video_key: str = "video",
        caption_key: str = "ai_caption",
        cfg_dropout_rate: float = 0.1,
        spatial_compression: int = 16,
        temporal_compression: int = 4,
        patch_spatial: int = 2,
        append_viewpoint_info: bool = True,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
    ) -> None:
        self.resolution = str(resolution)
        self.input_video_key = input_video_key
        self.caption_key = caption_key
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression
        self.patch_spatial = patch_spatial
        self.pipeline = ActionTransformPipeline(
            pad_keys=["video"],
            keep_aspect_ratio=True,
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            caption_key=caption_key,
            max_action_dim=max_action_dim,
            action_channel_masking=True,
            append_viewpoint_info=append_viewpoint_info,
            append_duration_fps_timestamps=append_duration_fps_timestamps,
            append_resolution_info=append_resolution_info,
            append_idle_frames=False,
        )

    # ----------------------------------------------------------------- process
    def sft_process_sample(self, item: dict) -> dict:
        # ActionTransformPipeline mutates the dict in place: resizes/pads "video",
        # adds "image_size", tokenizes the caption -> "text_token_ids", pads
        # "action" to max_action_dim, records "raw_action_dim", and builds
        # "sequence_plan" from "mode".
        return self.pipeline(item, resolution=self.resolution)

    # ----------------------------------------------------------------- tokens
    def compute_num_tokens(self, sample: dict) -> int:
        tokens = 1
        text = sample.get("text_token_ids")
        if text is not None:
            tokens += int(len(text))
        v = sample.get(self.input_video_key)
        if isinstance(v, torch.Tensor):
            _, T, H, W = v.shape
            latent_h = math.ceil(H / (self.spatial_compression * self.patch_spatial))
            latent_w = math.ceil(W / (self.spatial_compression * self.patch_spatial))
            latent_t = 1 + (T - 1) // self.temporal_compression
            tokens += latent_h * latent_w * latent_t + 2
        a = sample.get("action")
        if isinstance(a, torch.Tensor):
            tokens += int(a.shape[0])  # one token per action frame
        return tokens

    # ----------------------------------------------------------------- collate
    def sft_collate_fn(self, samples: list[dict], max_len: int, ignore_label_id: int = -100) -> dict:
        def col(key: str) -> list[Any]:
            return [s[key] for s in samples]

        batch: dict[str, Any] = {
            self.input_video_key: [[s[self.input_video_key]] for s in samples],
            "action": [[s["action"]] for s in samples],
            "raw_action_dim": col("raw_action_dim"),
            "mode": col("mode"),
            self.caption_key: col(self.caption_key),
            "conditioning_fps": col("conditioning_fps"),
            "image_size": torch.stack(col("image_size")),  # [B, 4]
            "domain_id": col("domain_id"),
            "sequence_plan": col("sequence_plan"),
        }
        if "text_token_ids" in samples[0]:
            # Same [[per-sample tensor]] nesting as video/action: OmniMoTModel's
            # _load_and_tokenize_text_data does `[t.tolist() for x in batch for t in x]`,
            # so each sample must be a list wrapping its 1-D token-id tensor.
            batch["text_token_ids"] = [[s["text_token_ids"]] for s in samples]
        return batch
