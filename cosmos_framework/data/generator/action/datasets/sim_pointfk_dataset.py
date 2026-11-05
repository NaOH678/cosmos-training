"""Explicit-window dual-hand simulation samples with causal selected anchors."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data import frame_transform
from cosmos_framework.data.pointflow_window import read_frame


_MODES = ("wam", "forward_dynamics", "inverse_dynamics")


def parse_mode_probs(spec):
    """Parse ``SIM_MODE_PROBS`` ("wam:0.75,forward_dynamics:0.25") into normalised (modes, probs).

    Empty / unset / pure ``wam`` returns None, i.e. the legacy wam-only path with no RNG use.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    weights = {}
    for item in spec.split(","):
        name, _, value = item.strip().partition(":")
        if name not in _MODES or not value:
            raise ValueError(f"Bad SIM_MODE_PROBS entry {item!r}; expected <{'|'.join(_MODES)}>:<prob>")
        if name in weights:
            raise ValueError(f"Duplicate mode in SIM_MODE_PROBS: {name}")
        weights[name] = float(value)
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"SIM_MODE_PROBS weights must be non-negative with positive sum: {spec!r}")
    modes = [m for m in _MODES if weights.get(m, 0) > 0]
    if modes == ["wam"]:
        return None
    total = sum(weights[m] for m in modes)
    return modes, [weights[m] / total for m in modes]


def draw_mode(modes_probs, seed, index, rank, worker, visit):
    """Stateless per-sample mode draw keyed on (seed, window index, rank, worker, visit count)."""
    modes, probs = modes_probs
    u = np.random.default_rng([int(seed), int(index), int(rank), int(worker), int(visit)]).random()
    cum = np.cumsum(probs)
    return modes[min(int(np.searchsorted(cum, u, side="right")), len(modes) - 1)]


class SimPointFKSFTDataset(Dataset):
    def __init__(
        self,
        bundle,
        selection_root,
        split,
        tokenizer_config,
        cfg_dropout_rate,
        max_action_dim,
        *,
        cached_video_only=False,
        include_geometry=True,
        action_stats_path=None,
        action_stats_sha256=None,
        mode_probs=None,
        mode_seed=42,
    ):
        self.mode_probs = mode_probs
        self.mode_seed = mode_seed
        self._visits = {}
        self.mode_counts = {}
        self.cached_video_only = cached_video_only
        self.include_geometry = include_geometry
        self.root = Path(bundle)
        self.cache = self.root / "datasets/cosmos-cache"
        if not self.cache.exists():
            self.cache = self.root / "datasets/bench2dex-task21-cosmos-cache"
        self.selected = Path(selection_root)
        selection = json.loads((self.selected / "manifest.json").read_text())
        self.source_geometry_root = selection.get("source_geometry_root")
        if selection["schema"] != "sim_pointfk_handguided1024_v1":
            raise ValueError("Unsupported point selection cache")
        # Old caches carry no ``frame`` field: they are camera-frame.
        self.frame = selection.get("frame", frame_transform.CAMERA)
        expected_frame = os.environ.get(frame_transform.FRAME_ENV)
        if expected_frame and expected_frame != self.frame:
            raise ValueError(f"{frame_transform.FRAME_ENV}={expected_frame} but cache frame is {self.frame}")
        self.rows = [r for r in selection["windows"] if r["split"] == ("validation" if split == "val" else split)]
        if not self.rows:
            raise ValueError(f"Empty split {split}")
        self.latent_root = self.cache / "vae_window_latents"
        manifest = json.loads((self.latent_root / "window_manifest.json").read_text())
        contract = manifest["contract"]
        if (contract["schema"], contract["fps"], contract["chunk_length"], contract["storage"]) != (
            "sim_pointfk_explicit_window_latents_v1",
            20,
            32,
            "bfloat16_bits_in_uint16",
        ):
            raise ValueError("Wrong latent contract")
        index_hash = hashlib.sha256((self.cache / "window_index.json").read_bytes()).hexdigest()
        if contract["window_index_sha256"] != index_hash or selection["window_index_sha256"] != index_hash:
            raise ValueError("Selection, latent and dataset window indices differ")
        self.latents = {r["name"]: r for r in manifest["episodes"]}
        self.offsets = {
            name: {start: i for i, start in enumerate(r["start_frames"])} for name, r in self.latents.items()
        }
        stats_path = Path(action_stats_path) if action_stats_path else self.root / "normalization_train_only.json"
        payload = stats_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if action_stats_sha256 and digest != action_stats_sha256:
            raise ValueError("Simulation action statistics SHA256 mismatch")
        if action_stats_sha256 and not action_stats_path:
            raise ValueError("action_stats_sha256 requires action_stats_path")
        if selection.get("action_stats_sha256") and selection["action_stats_sha256"] != digest:
            raise ValueError("Selection and action statistics differ")
        norm = json.loads(payload)
        if not norm["source"].startswith("training episodes only"):
            raise ValueError("Action statistics include validation")
        self.joint_names = norm["joint_names"]
        low, high = [np.asarray(norm["action"][k], np.float32) for k in ("q01", "q99")]
        self.normalizer = ActionAffineNormalization(
            torch.from_numpy((low + high) / 2), torch.from_numpy(np.maximum((high - low) / 2, 0.05))
        )
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            video_temporal_downsample=4,
            format_prompt_as_json=True,
        )

    def __len__(self):
        return len(self.rows)

    def get_shuffle_blocks(self):
        result = []
        for i, row in enumerate(self.rows):
            if i == 0 or row["episode"] != self.rows[i - 1]["episode"]:
                result.append([i, 1])
            else:
                result[-1][1] += 1
        return [tuple(r) for r in result]

    def _sample_mode(self, index):
        if self.mode_probs is None:
            return "wam"
        wi = get_worker_info()
        visit = self._visits.get(index, 0)
        self._visits[index] = visit + 1
        mode = draw_mode(
            self.mode_probs, self.mode_seed, index, int(os.environ.get("RANK", 0)), wi.id if wi else 0, visit
        )
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        if sum(self.mode_counts.values()) % 32 == 0:  # per-worker progress, for smoke / sanity checks
            print(f"[modemix] rank={os.environ.get('RANK', 0)} worker={wi.id if wi else 0} counts={self.mode_counts}", flush=True)
        return mode

    def __getitem__(self, index):
        row = self.rows[index]
        saved = torch.load(self.selected / row["path"], map_location="cpu", weights_only=False)
        ids = np.asarray(row["frame_ids"])
        name, start = row["episode"], row["start_frame"]
        if self.include_geometry and self.source_geometry_root:
            episode_root = Path(self.source_geometry_root) / name
            saved["pointflow"]["metadata"].update(
                source_path=str(episode_root / "pointflow"), preview_video=str(episode_root / "head.mp4")
            )
        np.testing.assert_array_equal(saved["raw_frame_ids"], ids)
        if self.include_geometry and saved["pointflow"]["metadata"].get("frame", frame_transform.CAMERA) != self.frame:
            raise ValueError("Sample frame differs from the selection manifest frame")
        if saved["joint_names"] != self.joint_names:
            raise ValueError("Action joint order changed")
        offset = self.offsets[name][start]
        latent_record = self.latents[name]
        np.testing.assert_array_equal(latent_record["frame_ids"][offset], ids)
        rgb_ids = ids[:1] if self.cached_video_only else ids
        frames = np.stack([read_frame(self.cache / "video_frames" / f"{name}.npy", int(i)) for i in rgb_ids])
        bits = read_frame(self.latent_root / latent_record["path"], offset)
        if bits.dtype != np.uint16 or bits.shape != (48, 9, 40, 40):
            raise ValueError("Wrong latent dtype/shape")
        sample = dict(
            video=torch.from_numpy(frames).permute(1, 0, 2, 3).contiguous(),
            action=saved["action"],
            ai_caption=saved["ai_caption"],
            conditioning_fps=torch.tensor(20),
            mode=self._sample_mode(index),
            domain_id=torch.tensor(28),
            viewpoint="concat_view",
            additional_view_description="Head on top; left wrist bottom-left; right wrist bottom-right.",
            episode_name=name,
            raw_frame_ids=ids,
            vae_latent_cache=torch.from_numpy(bits).view(torch.bfloat16).unsqueeze(0),
        )
        if self.include_geometry:
            sample.update(pointflow=saved["pointflow"], fk=saved["fk"])
        if self.cached_video_only:
            # RGB contains only the anchor; targets and full temporal extent live in the cache.
            sample["cached_video_num_frames"] = len(ids)
        result = self.transform(sample, "480", action_normalizer=self.normalizer)
        if result["image_size"].tolist() != latent_record["image_size"]:
            raise ValueError("Spatial transform differs from latent cache")
        return result


def get_sim_pointfk_sft_dataset(
    *,
    bundle,
    selection_root,
    split="train",
    tokenizer_config=None,
    cfg_dropout_rate=0.1,
    max_action_dim=64,
    iterable_shuffle=True,
    use_image_augmentation=False,
    cached_video_only=True,
    include_geometry=True,
    action_stats_path=None,
    action_stats_sha256=None,
    mode_probs=None,
):
    if use_image_augmentation:
        raise ValueError("Image augmentation is incompatible with fixed VAE latents")
    # Opt-in mode mixing (SIM_MODE_PROBS env or mode_probs arg): training stream only; eval stays wam.
    train_stream = split == "train" and iterable_shuffle
    probs = parse_mode_probs(mode_probs if mode_probs is not None else os.environ.get("SIM_MODE_PROBS", ""))
    ds = SimPointFKSFTDataset(
        bundle,
        selection_root,
        split,
        tokenizer_config,
        cfg_dropout_rate,
        max_action_dim,
        # Eval callbacks reconstruct even train cases with iterable_shuffle=False.
        cached_video_only=cached_video_only and split == "train" and iterable_shuffle,
        include_geometry=include_geometry,
        action_stats_path=action_stats_path,
        action_stats_sha256=action_stats_sha256,
        mode_probs=probs if train_stream else None,
    )
    return ActionIterableShuffleDataset(ds, seed=42, max_block_size=4) if iterable_shuffle else ds
