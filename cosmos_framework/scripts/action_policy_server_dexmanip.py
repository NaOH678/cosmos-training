# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DexManip-compatible ZMQ policy server for the dual-arm Franka + XHand policy.

This server keeps the local ``dexhand_deployment/deployment.py`` client unchanged:
it accepts the existing pickle+ZMQ observation dictionary, runs Cosmos policy
inference on a GPU node, denormalizes the predicted 38-D action chunk, and
returns one absolute action step per request.

The checkpoint path may point directly at a DCP iteration directory such as
``.../checkpoints/iter_000012500``; ``CheckpointOverrides`` infers the
``franka_xhand_dual_policy`` experiment name from that path and resolves the
underlying ``/model`` directory automatically.
"""

from __future__ import annotations

from cosmos_framework.inference.common.init import init_script

init_script()

import collections
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
import zmq

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.franka_xhand_dual_packer import ActionDataPacker
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_robolab import _load_training_config
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log

_DEFAULT_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "dexmanip"
_DEFAULT_CAMERA_SERIALS = ("241122305539", "236422070385", "115222071207")
_DEFAULT_ACTION_STATS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "vfm"
    / "action"
    / "datasets"
    / "franka_xhand_dual_normalization.json"
)
_DEFAULT_LOCAL_TEXT_TOKENIZER = "/mnt/inspurfs/eailab_os/scj_ckpt/cosmos3-nano/text_tokenizer"
_DEFAULT_WAN_VAE_PATH = "/mnt/inspurfs/eailab_os/scj_ckpt/wan2.2_VAE.pth/Wan2.2_VAE.pth"
_DOMAIN_NAME = "franka_xhand_dual"
_VIEWPOINT = "concat_view"

# The deployment cluster is offline; default to local-only asset resolution.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def _parse_sides(sides_from_obs: Any, obs: dict[str, Any], forced_arms: list[str]) -> list[str]:
    if forced_arms:
        return list(forced_arms)
    if isinstance(sides_from_obs, list) and sides_from_obs:
        return [str(side) for side in sides_from_obs]
    has_left = "franka_state_left" in obs
    has_right = "franka_state_right" in obs
    if has_left and has_right:
        return ["left", "right"]
    if has_right:
        return ["right"]
    return ["left"]


def _resolve_side_state(obs: dict[str, Any], side: str, prefix: str) -> dict[str, Any]:
    key = f"{prefix}_{side}"
    state = obs.get(key)
    if state is None and side == "left":
        state = obs.get(prefix)
    if not isinstance(state, dict):
        raise KeyError(f"Observation missing {key!r}")
    return state


def _as_float64_array(value: Any, *, expected_width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] != expected_width:
        raise ValueError(f"Expected width {expected_width}, got {array.shape[0]}")
    return array


@dataclass(frozen=True)
class DexManipPolicyConfig:
    checkpoint_path: str
    prompt: str
    port: int
    host: str
    camera_serials: tuple[str, ...]
    bgr2rgb: bool
    action_chunk_size: int
    conditioning_fps: float
    resolution: str
    action_dim: int
    max_action_dim: int
    guidance: float
    num_steps: int
    shift: float
    bootstrap_timeout_s: float
    prefetch_trigger: int


class DexManipServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str
    """DCP checkpoint iteration directory or consolidated model directory."""
    experiment: str | None = None
    """Optional explicit experiment name. Usually inferred from the checkpoint path."""
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Extra Hydra overrides forwarded to OmniSetup."""
    credential_path: str | None = None
    """Optional object-store credential path for remote checkpoints."""
    gpu_id: int = 0
    """CUDA device id for single-process serving. Default uses local device 0."""

    host: str = "0.0.0.0"
    """ZMQ bind host."""
    port: int = 25555
    """ZMQ REP port."""
    output_dir: Path | None = None
    """Output directory for temporary inference artifacts."""
    sampler: str = "unipc"
    """Diffusion sampler used by OmniInference."""

    prompt: str = "capture the tube"
    """Task prompt. Must match the training caption semantics."""
    camera_serials: list[str] = pydantic.Field(default_factory=lambda: list(_DEFAULT_CAMERA_SERIALS))
    """Camera serial order matching training cam_1/cam_2/cam_3."""
    bgr2rgb: bool = True
    """Convert incoming RealSense BGR frames to RGB before building the tiled observation."""

    resolution: str = "480"
    """Action transform resolution bucket."""
    conditioning_fps: float = 15.0
    """Conditioning FPS used during training."""
    action_chunk_size: int = 16
    """Number of action steps predicted per chunk."""
    action_dim: int = 38
    """Raw action dimension for the dual-arm embodiment."""
    guidance: float = 3.0
    """Classifier-free guidance scale."""
    num_steps: int = 4
    """Number of denoising steps."""
    shift: float = 5.0
    """UniPC sampler shift."""

    action_stats_path: Path = _DEFAULT_ACTION_STATS
    """Path to the dual-arm action normalization JSON used for denormalization."""
    bootstrap_timeout_s: float = 20.0
    """Maximum time to wait for the first generated chunk before falling back to hold-current."""
    prefetch_trigger: int = 15
    """Start generating the next chunk when the active buffer has at most this many actions left."""
    arms: list[str] = pydantic.Field(default_factory=list)
    """Optional forced arm order, e.g. ['left', 'right']."""


class DexManipPolicyService:
    def __init__(self, args: DexManipServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DexManip policy serving.")

        maybe_init_distributed(gpu_id=int(args.gpu_id))
        setup_args = self._build_setup_args(args)
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args = pipe.setup_args

        training_config = _load_training_config(self.setup_args, args.checkpoint_path)
        self.packer, max_action_dim = self._build_packer(training_config, args)

        self.cfg = DexManipPolicyConfig(
            checkpoint_path=self.setup_args.checkpoint_path,
            prompt=args.prompt,
            port=int(args.port),
            host=args.host,
            camera_serials=tuple(args.camera_serials),
            bgr2rgb=bool(args.bgr2rgb),
            action_chunk_size=int(args.action_chunk_size),
            conditioning_fps=float(args.conditioning_fps),
            resolution=str(args.resolution),
            action_dim=int(args.action_dim),
            max_action_dim=int(max_action_dim),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            shift=float(args.shift),
            bootstrap_timeout_s=float(args.bootstrap_timeout_s),
            prefetch_trigger=int(args.prefetch_trigger),
        )
        self.forced_arms = list(args.arms)

        stats = load_action_stats(str(args.action_stats_path))
        self.action_q01 = torch.from_numpy(stats["q01"]).float()
        self.action_range = torch.clamp(torch.from_numpy(stats["q99"]).float() - self.action_q01, min=1e-6)
        if self.action_q01.shape[0] != self.cfg.action_dim:
            raise ValueError(
                f"Action stats dim {self.action_q01.shape[0]} does not match action_dim={self.cfg.action_dim}"
            )

        self._rng = np.random.default_rng(0)
        self._infer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._worker_event = threading.Event()
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._generation_loop, name="DexManipChunkWorker", daemon=True)

        self._latest_obs: dict[str, Any] | None = None
        self._latest_obs_seq = 0
        self._requested_obs_seq = 0
        self._active_chunk: collections.deque[np.ndarray] = collections.deque()
        self._pending_chunk: collections.deque[np.ndarray] | None = None
        self._active_chunk_obs_seq = 0
        self._pending_chunk_obs_seq = 0
        self._generation_inflight = False
        self._underruns = 0
        self._last_action_vec: np.ndarray | None = None
        self._has_bootstrapped = False

        log.info(
            "[dexmanip-policy-server] ready "
            f"host={self.cfg.host} "
            f"port={self.cfg.port} "
            f"checkpoint={self.cfg.checkpoint_path!r} "
            f"prompt={self.cfg.prompt!r} "
            f"cams={list(self.cfg.camera_serials)} "
            f"chunk={self.cfg.action_chunk_size} "
            f"fps={self.cfg.conditioning_fps:.1f} "
            f"guidance={self.cfg.guidance:.2f} "
            f"steps={self.cfg.num_steps} "
            f"shift={self.cfg.shift:.2f} "
            f"resolution={self.cfg.resolution}"
        )

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._worker_stop.set()
        self._worker_event.set()
        self._worker.join(timeout=3.0)

    def _build_setup_args(self, args: DexManipServerArgs) -> OmniSetupArgs:
        experiment_overrides = [
            f"model.config.vlm_config.tokenizer.pretrained_model_name={_DEFAULT_LOCAL_TEXT_TOKENIZER}",
            "model.config.vlm_config.tokenizer.config_variant=hf",
            f"model.config.tokenizer.vae_path={_DEFAULT_WAN_VAE_PATH}",
            *list(args.experiment_overrides),
        ]
        setup_overrides: dict[str, Any] = {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir or _DEFAULT_OUTPUT_DIR,
            "sampler": args.sampler,
            "guardrails": False,
            "experiment_overrides": experiment_overrides,
        }
        if args.experiment is not None:
            setup_overrides["experiment"] = args.experiment
        if args.credential_path is not None:
            setup_overrides["credential_path"] = args.credential_path
        overrides = OmniSetupOverrides.model_validate(setup_overrides)
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _build_packer(self, training_config: Any | None, args: DexManipServerArgs) -> tuple[ActionDataPacker, int]:
        tokenizer_config = None
        max_action_dim = getattr(getattr(self.model, "config", None), "max_action_dim", 64)
        resolution = args.resolution
        conditioning_fps = args.conditioning_fps
        action_chunk_size = args.action_chunk_size

        if training_config is not None:
            try:
                dataloader_cfg = training_config.dataloader_train
                data_source_cfg = dataloader_cfg.data_source
                data_packer_cfg = dataloader_cfg.data_packer
                tokenizer_config = getattr(data_packer_cfg, "tokenizer_config", None)
                max_action_dim = int(getattr(data_packer_cfg, "max_action_dim", max_action_dim))
                resolution = str(getattr(data_packer_cfg, "resolution", resolution))
                conditioning_fps = float(getattr(data_source_cfg, "fps", conditioning_fps))
                action_chunk_size = int(getattr(data_source_cfg, "chunk_length", action_chunk_size))
            except Exception as exc:
                log.warning(
                    "[dexmanip-policy-server] could not read training config; "
                    f"using CLI defaults: {exc}"
                )

        args.resolution = resolution
        args.conditioning_fps = conditioning_fps
        args.action_chunk_size = action_chunk_size

        packer = ActionDataPacker(
            tokenizer_config=tokenizer_config,
            max_action_dim=max_action_dim,
            resolution=resolution,
            cfg_dropout_rate=0.0,
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
        )
        return packer, max_action_dim

    def _compose_tiled_image(self, obs: dict[str, Any]) -> np.ndarray:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError("Observation missing 'images' dict")

        frames: list[np.ndarray] = []
        for serial in self.cfg.camera_serials:
            frame = images.get(serial)
            if not isinstance(frame, dict) or "color" not in frame:
                raise KeyError(f"Observation missing color frame for camera {serial!r}")
            color = np.asarray(frame["color"])
            if color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError(f"Camera {serial!r} color frame must have shape [H,W,3], got {color.shape}")
            if color.dtype != np.uint8:
                color = np.clip(color, 0, 255).astype(np.uint8)
            if self.cfg.bgr2rgb:
                color = np.ascontiguousarray(color[..., ::-1])
            else:
                color = np.ascontiguousarray(color)
            frames.append(color)

        top = frames[0]
        half_h = top.shape[0] // 2
        half_w = top.shape[1] // 2
        bottom_left = _resize_rgb_uint8(frames[1], (half_h, half_w))
        bottom_right = _resize_rgb_uint8(frames[2], (half_h, half_w))
        bottom = np.concatenate([bottom_left, bottom_right], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def _build_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        image = self._compose_tiled_image(obs)
        height, width = image.shape[:2]

        video = torch.zeros((3, self.cfg.action_chunk_size + 1, height, width), dtype=torch.uint8)
        video[:, 0] = torch.from_numpy(image).permute(2, 0, 1)
        action = torch.zeros((self.cfg.action_chunk_size, self.cfg.action_dim), dtype=torch.float32)

        sample = {
            "ai_caption": self.cfg.prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id(_DOMAIN_NAME), dtype=torch.long),
            "viewpoint": _VIEWPOINT,
        }
        sample = self.packer.sft_process_sample(sample)
        max_len = self.packer.compute_num_tokens(sample)
        return self.packer.sft_collate_fn([sample], max_len=max_len)

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**31))

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        normalized = action[..., : self.cfg.action_dim]
        q01 = self.action_q01.to(normalized.device)
        qrange = self.action_range.to(normalized.device)
        return (normalized + 1.0) / 2.0 * qrange + q01

    def _infer_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        batch = self._build_batch(obs)
        seed = self._next_seed()
        t0 = time.monotonic()
        with self._infer_lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )
        pred_action = samples["action"][0].float().squeeze(0)
        pred_action = self._denormalize_action(pred_action)
        dt_ms = (time.monotonic() - t0) * 1000.0
        pred_np = pred_action.detach().cpu().numpy().astype(np.float64)
        log.info(
            "[dexmanip-policy-server] generated chunk "
            f"seed={seed} "
            f"steps={self.cfg.num_steps} "
            f"latency={dt_ms:.1f}ms "
            f"shape={tuple(pred_np.shape)}"
        )
        return pred_np

    def _request_async_generation_locked(self) -> None:
        if self._latest_obs is None or self._generation_inflight:
            return
        self._requested_obs_seq = self._latest_obs_seq
        self._generation_inflight = True
        self._worker_event.set()

    def _generation_loop(self) -> None:
        while not self._worker_stop.is_set():
            self._worker_event.wait(timeout=0.1)
            self._worker_event.clear()
            if self._worker_stop.is_set():
                break

            with self._state_lock:
                if not self._generation_inflight or self._latest_obs is None:
                    continue
                obs = self._latest_obs
                obs_seq = self._requested_obs_seq

            try:
                chunk = self._infer_chunk(obs)
                pending = collections.deque(np.asarray(step, dtype=np.float64) for step in chunk)
                with self._state_lock:
                    self._pending_chunk = pending
                    self._pending_chunk_obs_seq = obs_seq
                    self._generation_inflight = False
            except Exception:
                with self._state_lock:
                    self._generation_inflight = False
                log.exception("[dexmanip-policy-server] async chunk generation failed")

    def _activate_pending_locked(self) -> None:
        if not self._active_chunk and self._pending_chunk is not None:
            self._active_chunk = self._pending_chunk
            self._active_chunk_obs_seq = self._pending_chunk_obs_seq
            self._pending_chunk = None
            self._pending_chunk_obs_seq = 0

    def _consume_action_locked(self, *, allow_prefetch: bool) -> tuple[np.ndarray | None, int, int]:
        self._activate_pending_locked()

        if not self._active_chunk:
            return None, 0, self._latest_obs_seq

        action_vec = self._active_chunk.popleft()
        active_remaining = len(self._active_chunk)
        latest_obs_seq = self._latest_obs_seq

        if (
            allow_prefetch
            and active_remaining <= self.cfg.prefetch_trigger
            and self._pending_chunk is None
            and not self._generation_inflight
            and latest_obs_seq > self._active_chunk_obs_seq
        ):
            self._request_async_generation_locked()

        self._last_action_vec = np.asarray(action_vec, dtype=np.float64)
        return self._last_action_vec, active_remaining, latest_obs_seq

    def _pop_action_vec(self, obs: dict[str, Any]) -> tuple[np.ndarray | None, int, int]:
        with self._state_lock:
            self._latest_obs = obs
            self._latest_obs_seq += 1
            return self._consume_action_locked(allow_prefetch=True)

    def _bootstrap_chunk(self, obs: dict[str, Any]) -> None:
        chunk = self._infer_chunk(obs)
        with self._state_lock:
            self._active_chunk = collections.deque(np.asarray(step, dtype=np.float64) for step in chunk)
            self._active_chunk_obs_seq = self._latest_obs_seq
            self._has_bootstrapped = True

    def _hold_action(self, obs: dict[str, Any], sides: list[str]) -> dict[str, Any]:
        action: dict[str, Any] = {"timestamp": time.time(), "sides": list(sides), "hold": True}
        for side in sides:
            franka_state = _resolve_side_state(obs, side, "franka_state")
            xhand_state = _resolve_side_state(obs, side, "xhand_state")
            ee_pos = _as_float64_array(franka_state["ee_pos"], expected_width=3)
            ee_quat = _as_float64_array(franka_state["ee_quat"], expected_width=4)
            xhand_rad = _as_float64_array(xhand_state["joint_pos"], expected_width=12)
            action[f"franka_action_{side}"] = {"ee_pos": ee_pos, "ee_quat": ee_quat}
            action[f"xhand_action_{side}"] = np.rad2deg(xhand_rad).tolist()
        if len(sides) == 1:
            side0 = sides[0]
            action["franka_action"] = action[f"franka_action_{side0}"]
            action["xhand_action"] = action[f"xhand_action_{side0}"]
        return action

    def _vector_to_action(self, action_vec: np.ndarray, sides: list[str]) -> dict[str, Any]:
        vec = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        if vec.shape[0] < 38:
            raise ValueError(f"Predicted action width must be 38, got {vec.shape[0]}")

        segments = {
            "left": vec[0:19],
            "right": vec[19:38],
        }
        action: dict[str, Any] = {"timestamp": time.time(), "sides": list(sides)}
        for side in sides:
            seg = segments[side]
            action[f"franka_action_{side}"] = {
                "ee_pos": seg[0:3].copy(),
                "ee_quat": seg[3:7].copy(),
            }
            action[f"xhand_action_{side}"] = seg[7:19].tolist()
        if len(sides) == 1:
            side0 = sides[0]
            action["franka_action"] = action[f"franka_action_{side0}"]
            action["xhand_action"] = action[f"xhand_action_{side0}"]
        return action

    def serve_forever(self) -> None:
        address = f"tcp://{self.cfg.host}:{self.cfg.port}"
        ctx = zmq.Context()
        socket_rep = ctx.socket(zmq.REP)
        socket_rep.bind(address)

        self.start()

        total = 0
        t_start = time.time()
        local_ip = get_local_ip()
        log.info(
            "[dexmanip-policy-server] listening on "
            f"{address} "
            f"(client: tcp://{local_ip}:{self.cfg.port})"
        )
        log.info(
            "[dexmanip-policy-server] "
            f"prompt={self.cfg.prompt!r} "
            f"camera_serials={list(self.cfg.camera_serials)} "
            f"forced_arms={self.forced_arms or '<auto>'}"
        )

        try:
            while True:
                raw = socket_rep.recv()
                recv_time = time.time()
                try:
                    obs = pickle.loads(raw)
                    if not isinstance(obs, dict):
                        raise TypeError("Observation payload must deserialize to a dict")
                except Exception as exc:
                    log.warning(f"[dexmanip-policy-server] failed to deserialize request: {exc}")
                    socket_rep.send(pickle.dumps({"error": str(exc)}))
                    continue

                sides = _parse_sides(obs.get("arms"), obs, self.forced_arms)
                obs_ts = float(obs.get("timestamp", 0.0) or 0.0)

                try:
                    action_vec, ready_remaining, latest_obs_seq = self._pop_action_vec(obs)
                    if action_vec is None:
                        if not self._has_bootstrapped:
                            t_wait0 = time.monotonic()
                            self._bootstrap_chunk(obs)
                            wait_ms = (time.monotonic() - t_wait0) * 1000.0
                            with self._state_lock:
                                action_vec, ready_remaining, latest_obs_seq = self._consume_action_locked(
                                    allow_prefetch=False
                                )
                            if wait_ms > self.cfg.bootstrap_timeout_s * 1000.0:
                                log.warning(
                                    "[dexmanip-policy-server] bootstrap exceeded warning threshold: "
                                    f"{wait_ms:.1f}ms"
                                )
                            log.info(
                                "[dexmanip-policy-server] bootstrap "
                                f"latency={wait_ms:.1f}ms "
                                f"latest_obs_seq={latest_obs_seq}"
                            )
                        else:
                            with self._state_lock:
                                if self._pending_chunk is None and not self._generation_inflight:
                                    self._request_async_generation_locked()
                    if action_vec is None:
                        self._underruns += 1
                        response = self._hold_action(obs, sides)
                    else:
                        response = self._vector_to_action(action_vec, sides)
                except Exception:
                    self._underruns += 1
                    log.exception("[dexmanip-policy-server] request handling failed; falling back to hold-current")
                    response = self._hold_action(obs, sides)

                socket_rep.send(pickle.dumps(response))

                total += 1
                elapsed = recv_time - t_start
                avg_hz = total / elapsed if elapsed > 0 else 0.0
                latency_ms = (recv_time - obs_ts) * 1000.0 if obs_ts else 0.0
                active_ready = len(self._active_chunk)
                pending_ready = 0 if self._pending_chunk is None else len(self._pending_chunk)
                if total == 1 or total % 25 == 0:
                    log.info(
                        "[dexmanip-policy-server] "
                        f"req={total} "
                        f"lat={latency_ms:.1f}ms "
                        f"avg={avg_hz:.1f}Hz "
                        f"active={active_ready} "
                        f"pending={pending_ready} "
                        f"underruns={self._underruns} "
                        f"sides={sides}"
                    )

        except KeyboardInterrupt:
            log.info("[dexmanip-policy-server] interrupted")
        finally:
            self.stop()
            socket_rep.close()
            ctx.term()


def main() -> None:
    args = tyro_cli(DexManipServerArgs, description=__doc__)
    service = DexManipPolicyService(args)
    service.serve_forever()


if __name__ == "__main__":
    main()
