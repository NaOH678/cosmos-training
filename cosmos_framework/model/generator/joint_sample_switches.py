# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in, inference-only switches for joint video/action/PointFlow/FK sampling.

Both switches default to OFF and then change nothing:

``JOINT_ATTN_BLOCK``  (e.g. ``action<-pointflow,action<-fk``)
    During joint sampling, NOISY (generated) tokens of the query modality may not
    attend to NOISY tokens of the key modality.  Conditioning tokens (first-frame
    vision latent, action row 0 / state, PointFlow / FK anchor tokens) and text
    tokens are always visible.  Only active inside ``active_joint_attn_block``,
    which ``generate_samples_from_batch`` enters, i.e. never in training.

``JOINT_GEOM_LEAD``  (integer k >= 0)
    Custom Euler loop in which PointFlow / FK follow a compressed ``N-k``-step
    schedule (start at pure noise, reach sigma 0 after ``N-k`` of the ``N`` steps),
    while vision / action follow the normal N-step grid (``0 < k < N``).

``JOINT_SAMPLER_STEPS_OVERRIDE``  (integer >= 1)
    Replace the number of sampler steps of the joint rollout.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Callable, Sequence

import torch

MODALITIES = ("vision", "action", "pointflow", "fk")
_MODALITY_ID = {name: i for i, name in enumerate(MODALITIES)}
NO_MODALITY = -1  # text, sound, padding: never blocked in either role

# ---------------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------------


def parse_joint_attn_block(spec: str | None) -> tuple[tuple[str, str], ...]:
    """``"action<-pointflow, action<-fk"`` -> ``(("action","pointflow"),("action","fk"))``.

    Empty / unset -> ``()``.  Unknown modality names, malformed items and
    ``q<-q`` (a modality blocking itself) raise ``ValueError``.
    """
    rules: list[tuple[str, str]] = []
    for item in filter(None, (part.strip() for part in (spec or "").split(","))):
        if item.count("<-") != 1:
            raise ValueError(f"JOINT_ATTN_BLOCK: malformed item {item!r}, expected 'query<-key'")
        query, key = (part.strip() for part in item.split("<-"))
        for name in (query, key):
            if name not in _MODALITY_ID:
                raise ValueError(f"JOINT_ATTN_BLOCK: unknown modality {name!r}, expected one of {list(MODALITIES)}")
        if query == key:
            raise ValueError(f"JOINT_ATTN_BLOCK: {item!r} blocks a modality from itself, which is not supported")
        if (query, key) not in rules:
            rules.append((query, key))
    return tuple(rules)


def joint_attn_block_from_env() -> tuple[tuple[str, str], ...]:
    return parse_joint_attn_block(os.environ.get("JOINT_ATTN_BLOCK", ""))


def parse_joint_geom_lead(spec: str | None) -> int:
    text = (spec or "").strip()
    if not text:
        return 0
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"JOINT_GEOM_LEAD: expected a non-negative integer, got {spec!r}") from exc
    if value < 0:
        raise ValueError(f"JOINT_GEOM_LEAD: expected a non-negative integer, got {value}")
    return value


def joint_geom_lead_from_env() -> int:
    return parse_joint_geom_lead(os.environ.get("JOINT_GEOM_LEAD", ""))


def parse_joint_steps_override(spec: str | None) -> int | None:
    text = (spec or "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"JOINT_SAMPLER_STEPS_OVERRIDE: expected an integer >= 1, got {spec!r}") from exc
    if value < 1:
        raise ValueError(f"JOINT_SAMPLER_STEPS_OVERRIDE: expected an integer >= 1, got {value}")
    return value


def joint_steps_override_from_env() -> int | None:
    return parse_joint_steps_override(os.environ.get("JOINT_SAMPLER_STEPS_OVERRIDE", ""))


# ---------------------------------------------------------------------------
# attention block: active-rule scope (module global, set only by generate_samples_from_batch)
# ---------------------------------------------------------------------------

_ACTIVE_ATTN_BLOCK: tuple[tuple[str, str], ...] = ()


def get_active_joint_attn_block() -> tuple[tuple[str, str], ...]:
    return _ACTIVE_ATTN_BLOCK


@contextmanager
def active_joint_attn_block(rules: Sequence[tuple[str, str]]):
    """Make ``rules`` visible to the network for the duration of a joint sampling call."""
    global _ACTIVE_ATTN_BLOCK
    previous = _ACTIVE_ATTN_BLOCK
    _ACTIVE_ATTN_BLOCK = tuple(rules)
    try:
        yield
    finally:
        _ACTIVE_ATTN_BLOCK = previous


# ---------------------------------------------------------------------------
# mask construction
# ---------------------------------------------------------------------------


def noisy_token_flags(data) -> tuple[torch.Tensor, torch.Tensor]:
    """(sequence_indexes, is_noisy) of a vision/action ``ModalityData``.

    Same frame-major arithmetic as ``_apply_timestep_embeds_to_noisy_tokens``:
    token ``i`` of the packed payload list lives at ``sequence_indexes[i]`` and is
    noisy iff its frame is in that payload's ``noisy_frame_indexes``.
    """
    seq = data.sequence_indexes
    flags = []
    for noisy_frames, shape in zip(data.noisy_frame_indexes, data.token_shapes, strict=True):
        frames = torch.zeros(int(shape[0]), dtype=torch.bool, device=seq.device)
        frames[noisy_frames.to(seq.device).long()] = True
        flags.append(frames.repeat_interleave(math.prod(shape[1:])))
    is_noisy = torch.cat(flags) if flags else torch.zeros(0, dtype=torch.bool, device=seq.device)
    if is_noisy.numel() != seq.numel():
        raise ValueError(f"noisy flags cover {is_noisy.numel()} tokens but the modality has {seq.numel()}")
    return seq, is_noisy


def build_modality_layout(packed_seq, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``(modality_id [N] long, is_noisy [N] bool)``.

    Ids follow ``MODALITIES``; text / sound are ``NO_MODALITY`` and count as
    not-noisy.  PointFlow / FK: ``noisy_indexes`` are the generated tokens, the
    remaining ``sequence_indexes`` are the anchor (conditioning) tokens.
    """
    n = int(packed_seq.sequence_length)
    device = device if device is not None else packed_seq.text_indexes.device
    ids = torch.full((n,), NO_MODALITY, dtype=torch.long, device=device)
    noisy = torch.zeros(n, dtype=torch.bool, device=device)
    for name in ("vision", "action"):
        data = getattr(packed_seq, name)
        if data is None or not isinstance(data.sequence_indexes, torch.Tensor):
            continue
        if name == "action" and data.tokens is None:
            continue
        seq, flags = noisy_token_flags(data)
        ids[seq.to(device)] = _MODALITY_ID[name]
        noisy[seq.to(device)] = flags.to(device)
    for name, attr in (("pointflow", "point"), ("fk", "fk")):
        payload = getattr(packed_seq, attr)
        if payload is None:
            continue
        ids[payload.sequence_indexes.to(device)] = _MODALITY_ID[name]
        noisy[payload.noisy_indexes.to(device).flatten()] = True
    return ids, noisy


def build_blocked_matrix(
    modality_ids: torch.Tensor,
    is_noisy: torch.Tensor,
    rules: Sequence[tuple[str, str]],
    query_start: int = 0,
) -> torch.Tensor | None:
    """Bool ``[N - query_start, N]``, True where query row may NOT attend to the key.

    Blocked iff (modality(q), modality(k)) is a rule AND both q and k are noisy.
    Returns ``None`` for an empty rule set (the caller keeps the unmasked path).
    """
    if not rules:
        return None
    q_ids, q_noisy = modality_ids[query_start:], is_noisy[query_start:]
    blocked = torch.zeros(q_ids.numel(), modality_ids.numel(), dtype=torch.bool, device=modality_ids.device)
    for query, key in rules:
        rows = (q_ids == _MODALITY_ID[query]) & q_noisy
        cols = (modality_ids == _MODALITY_ID[key]) & is_noisy
        blocked |= rows[:, None] & cols[None, :]
    return blocked


def build_joint_attn_allowed(packed_seq, rules: Sequence[tuple[str, str]], device=None) -> torch.Tensor | None:
    """Allowed-mask for the generation rows of a single-sample two-way pack, or ``None``.

    Shape ``[N_gen, N]`` (rows = positions ``>= text length``), True = may attend.
    """
    if not rules:
        return None
    if len(packed_seq.sample_lens) != 1 or packed_seq.attn_modes != ["causal", "full"]:
        raise NotImplementedError("JOINT_ATTN_BLOCK supports a single two-way (causal/full) sample only")
    ids, noisy = build_modality_layout(packed_seq, device)
    text_len = int(packed_seq.split_lens[0])
    present = {name for name in MODALITIES if bool((ids == _MODALITY_ID[name]).any())}
    for query, key in rules:
        if query not in present or key not in present:
            # A rule over an absent modality is a no-op; surface it instead of passing silently.
            raise ValueError(
                f"JOINT_ATTN_BLOCK rule {query}<-{key} refers to a modality absent from this sequence "
                f"(present: {sorted(present)})"
            )
    return ~build_blocked_matrix(ids, noisy, rules, query_start=text_len)


# ---------------------------------------------------------------------------
# masked attention (replaces two_way_attention when a mask is active)
# ---------------------------------------------------------------------------


def masked_two_way_attention(q_pack, k_pack, v_pack, allowed: torch.Tensor, text_len: int, normalized_k=None):
    """Same maths as ``two_way_attention`` for one sample, plus an explicit gen-row mask.

    Text rows: causal over text keys (identical to the causal split).  Generation
    rows: all keys (diffusion-normalised text K when supplied) minus ``~allowed``.
    Inference only: dense SDPA, no varlen.
    """
    from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq

    if torch.is_grad_enabled():
        raise RuntimeError("JOINT_ATTN_BLOCK masked attention is inference-only (run under torch.no_grad())")
    if q_pack["is_sharded"]:
        raise NotImplementedError("JOINT_ATTN_BLOCK requires CP=1")
    q, k, v = (get_all_seq(p) for p in (q_pack, k_pack, v_pack))  # [N,H(kv),D]
    kn = k if normalized_k is None else get_all_seq(normalized_k)
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k, v, kn = (x.repeat_interleave(groups, dim=1) for x in (k, v, kn))
    n = q.shape[0]
    if allowed.shape != (n - text_len, n):
        raise ValueError(f"mask shape {tuple(allowed.shape)} does not match {(n - text_len, n)}")

    def sdpa(qx, kx, vx, **kw):
        out = torch.nn.functional.scaled_dot_product_attention(
            qx.transpose(0, 1)[None], kx.transpose(0, 1)[None], vx.transpose(0, 1)[None], **kw
        )
        return out[0].transpose(0, 1)  # [S,H,D]

    outs = []
    if text_len > 0:
        outs.append(sdpa(q[:text_len], k[:text_len], v[:text_len], is_causal=True))
    if n > text_len:
        outs.append(sdpa(q[text_len:], kn, v, attn_mask=allowed[None, None]))
    return from_all_seq(torch.cat(outs).flatten(-2), q_pack)


# ---------------------------------------------------------------------------
# geometry lead sampler
# ---------------------------------------------------------------------------


def geom_lead_schedule(num_steps: int, lead: int) -> list[tuple[int, int, int]]:
    """Per outer step ``i``: ``(vision_idx, geom_idx, geom_next_idx)``.

    Vision / action index the sampler's ``num_steps``-step grid (``i -> i+1``).
    Geometry indexes its OWN compressed grid of ``num_steps - lead`` steps (built by
    the sampler's schedule for that step count): it starts at its grid[0] (pure
    noise, correctly labelled), reaches sigma 0 after ``num_steps - lead`` steps and
    is then left untouched (``g == g_next == num_steps - lead``, dt = 0).
    Requires ``0 <= lead < num_steps`` (``lead == 0`` is the unchanged schedule).
    """
    if num_steps < 1 or not 0 <= lead < num_steps:
        raise ValueError(f"JOINT_GEOM_LEAD: need 0 <= lead < num_steps, got lead={lead}, num_steps={num_steps}")
    geom_steps = num_steps - lead
    return [(i, min(i, geom_steps), min(i + 1, geom_steps)) for i in range(num_steps)]


def unipc_sigma_grid(sampler, num_steps: int, shift: float, device="cpu"):
    """(sigmas [num_steps+1], timesteps [num_steps]) exactly as ``UniPCSampler`` builds them."""
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=sampler.cfg.num_train_timesteps,
        shift=sampler.cfg.shift,
        use_dynamic_shifting=sampler.cfg.use_dynamic_shifting,
    )
    scheduler.set_timesteps(num_steps, device=device, shift=shift)
    return scheduler.sigmas.clone(), scheduler.timesteps.clone()


@torch.no_grad()
def euler_geom_lead_sample(
    velocity_fn: Callable,
    latents: list[torch.Tensor],
    geom_lens: Sequence[int],
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    lead: int,
    geom_sigmas: torch.Tensor | None = None,
    geom_timesteps: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Euler loop with a per-segment sigma schedule.

    ``latents[b]`` is the flat per-sample state ``[head | geometry]`` where the last
    ``geom_lens[b]`` elements are PointFlow + FK.  Vision / action follow
    ``sigmas`` (num_steps + 1 points); geometry follows ``geom_sigmas`` (the
    sampler's grid for ``num_steps - lead`` steps), then stays fixed at sigma 0.
    ``velocity_fn(state, t, geom_t)`` gets ``(1, 1)`` timesteps in the sampler's own
    units (``sigma * num_train_timesteps``, int64-truncated like ``UniPCSampler``).
    ``lead == 0`` reuses ``sigmas`` for geometry (plain Euler on the sampler grid).
    """
    num_steps = len(timesteps)
    if len(sigmas) != num_steps + 1:
        raise ValueError("sigmas must have num_steps + 1 entries")
    schedule = geom_lead_schedule(num_steps, lead)
    if lead == 0:
        geom_sigmas, geom_timesteps = sigmas, timesteps
    elif geom_sigmas is None or geom_timesteps is None:
        raise ValueError("lead > 0 requires geom_sigmas and geom_timesteps")
    if len(geom_sigmas) != num_steps - lead + 1 or len(geom_timesteps) != num_steps - lead:
        raise ValueError(f"geometry grid must have {num_steps - lead} steps")
    t_grid = torch.cat([timesteps.reshape(-1), timesteps.new_zeros(1)])  # sigma 0 -> timestep 0
    gt_grid = torch.cat([geom_timesteps.reshape(-1), geom_timesteps.new_zeros(1)])
    state = [x.clone() for x in latents]
    for i, g, g_next in schedule:
        v = velocity_fn(state, t_grid[i].reshape(1, 1), gt_grid[g].reshape(1, 1))
        dt_head = float(sigmas[i + 1] - sigmas[i])
        dt_geom = float(geom_sigmas[g_next] - geom_sigmas[g])
        new_state = []
        for x, vx, glen in zip(state, v, geom_lens, strict=True):
            split = x.numel() - glen
            vx = vx.to(x.dtype)
            new_state.append(torch.cat([x[:split] + dt_head * vx[:split], x[split:] + dt_geom * vx[split:]]))
        state = new_state
    return state
