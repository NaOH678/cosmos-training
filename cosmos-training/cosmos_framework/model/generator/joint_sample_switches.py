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

``JOINT_VISION_LEAD``  (integer k >= 0)
    Same loop for the vision stream: vision follows the compressed ``N-k``-step
    schedule and is committed (sigma 0) k steps before action / PointFlow / FK finish.
    Mutually exclusive with ``JOINT_GEOM_LEAD``.

``JOINT_SAMPLER_STEPS_OVERRIDE``  (integer >= 1)
    Replace the number of sampler steps of the joint rollout.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

MODALITIES = ("vision", "action", "pointflow", "fk")
_MODALITY_ID = {name: i for i, name in enumerate(MODALITIES)}
NO_MODALITY = -1  # text, sound, padding: never blocked in either role

# ---------------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------------


NOISY_SUFFIX = ":noisy"  # fixed-mask key ``vision:noisy``: only the noisy tokens of that modality


def split_rule_key(key: str) -> tuple[str, bool]:
    """``"vision:noisy"`` -> ``("vision", True)``; ``"vision"`` -> ``("vision", False)``."""
    return (key[: -len(NOISY_SUFFIX)], True) if key.endswith(NOISY_SUFFIX) else (key, False)


def parse_joint_attn_block(spec: str | None, allow_noisy_keys: bool = False) -> tuple[tuple[str, str], ...]:
    """``"action<-pointflow, action<-fk"`` -> ``(("action","pointflow"),("action","fk"))``.

    Empty / unset -> ``()``.  Unknown modality names, malformed items and
    ``q<-q`` (a modality blocking itself) raise ``ValueError``.  With
    ``allow_noisy_keys`` the key may carry ``:noisy`` (``action<-vision:noisy``).
    """
    rules: list[tuple[str, str]] = []
    for item in filter(None, (part.strip() for part in (spec or "").split(","))):
        if item.count("<-") != 1:
            raise ValueError(f"JOINT_ATTN_BLOCK: malformed item {item!r}, expected 'query<-key'")
        query, key = (part.strip() for part in item.split("<-"))
        key_name, noisy_key = split_rule_key(key)
        if noisy_key and not allow_noisy_keys:
            raise ValueError(f"JOINT_ATTN_BLOCK: {item!r}: ':noisy' keys are only valid in joint_attn_mask")
        for name in (query, key_name):
            if name not in _MODALITY_ID:
                raise ValueError(f"JOINT_ATTN_BLOCK: unknown modality {name!r}, expected one of {list(MODALITIES)}")
        if query == key_name:
            raise ValueError(f"JOINT_ATTN_BLOCK: {item!r} blocks a modality from itself, which is not supported")
        if (query, key) not in rules:
            rules.append((query, key))
    return tuple(rules)


def joint_attn_block_from_env() -> tuple[tuple[str, str], ...]:
    return parse_joint_attn_block(os.environ.get("JOINT_ATTN_BLOCK", ""))


def _parse_lead(spec: str | None, name: str) -> int:
    text = (spec or "").strip()
    if not text:
        return 0
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{name}: expected a non-negative integer, got {spec!r}") from exc
    if value < 0:
        raise ValueError(f"{name}: expected a non-negative integer, got {value}")
    return value


def parse_joint_geom_lead(spec: str | None) -> int:
    return _parse_lead(spec, "JOINT_GEOM_LEAD")


def joint_geom_lead_from_env() -> int:
    return parse_joint_geom_lead(os.environ.get("JOINT_GEOM_LEAD", ""))


def parse_joint_vision_lead(spec: str | None) -> int:
    return _parse_lead(spec, "JOINT_VISION_LEAD")


def joint_vision_lead_from_env() -> int:
    return parse_joint_vision_lead(os.environ.get("JOINT_VISION_LEAD", ""))


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
# fixed modality mask (training + inference): ``model.config.joint_attn_mask``
# ---------------------------------------------------------------------------

# Rule sets of the modality-routing ablation.  Every preset keeps vision<->action
# (the native WAM path) and only removes edges that involve PointFlow / FK.
JOINT_ATTN_MASK_PRESETS: dict[str, str] = {
    # M1: geometry only reads; neither video nor action reads PointFlow / FK.
    "geom_readonly": "action<-pointflow,action<-fk,vision<-pointflow,vision<-fk",
    # M2: action does not read geometry directly (video still does).
    "action_no_geom": "action<-pointflow,action<-fk",
    # M3: FK exchanges only with action, PointFlow only with video, no PointFlow<->FK.
    "physical_split": "fk<-vision,vision<-fk,pointflow<-action,action<-pointflow,pointflow<-fk,fk<-pointflow",
    # M4: action reads only the observed (conditioning) vision frames, not the noisy / predicted ones.
    "action_observed_vision": "action<-vision:noisy",
}


def resolve_joint_attn_mask(spec: str | None) -> tuple[tuple[str, str], ...]:
    """Preset name (``JOINT_ATTN_MASK_PRESETS``) or explicit ``q<-k`` rules separated
    by ``,`` or ``;`` -> rules.  Empty / unset -> ``()`` (no mask).  A key may be
    ``modality:noisy`` to cut only that modality's noisy tokens (conditioning stays visible)."""
    spec = (spec or "").strip()
    return parse_joint_attn_block(JOINT_ATTN_MASK_PRESETS.get(spec, spec.replace(";", ",")), allow_noisy_keys=True)


def build_modality_ids(packed_seq, device=None) -> torch.Tensor:
    """Per-token modality id ``[N]`` following ``MODALITIES``; text / sound are ``NO_MODALITY``.

    Unlike :func:`build_modality_layout` every token of a modality counts, its
    conditioning tokens (first vision frame, action row 0, PointFlow / FK anchors)
    included: the fixed mask cuts a modality pair completely.
    """
    n = int(packed_seq.sequence_length)
    ids = torch.full((n,), NO_MODALITY, dtype=torch.long, device=device)
    for name in ("vision", "action"):
        data = getattr(packed_seq, name)
        if data is None or not isinstance(data.sequence_indexes, torch.Tensor):
            continue
        if name == "action" and data.tokens is None:
            continue
        ids[data.sequence_indexes.to(ids.device)] = _MODALITY_ID[name]
    for name, attr in (("pointflow", "point"), ("fk", "fk")):
        payload = getattr(packed_seq, attr)
        if payload is not None:
            ids[payload.sequence_indexes.to(ids.device)] = _MODALITY_ID[name]
    return ids


@dataclass(frozen=True)
class ModalityGather:
    """Index plan that turns masked full attention into plain varlen attention.

    Generation queries are regrouped into one segment per (sample, query modality);
    the segment's keys are the tokens of that sample whose modality the query
    modality may read (text always).  The ordinary varlen kernel on the gathered
    tensors computes exactly the masked attention, gradients included.
    """

    q_index: torch.Tensor  # [N_gen] long: rows of the full-only (generation) sequence, segment order
    kv_index: torch.Tensor  # [N_kv] long: rows of the all-token sequence, segment order
    cu_seqlens_q: torch.Tensor  # [S+1] int32
    cu_seqlens_kv: torch.Tensor  # [S+1] int32
    max_seqlen_q: int
    max_seqlen_kv: int


def build_modality_gather(packed_seq, rules: Sequence[tuple[str, str]], device=None) -> ModalityGather | None:
    """:class:`ModalityGather` for a two-way pack (any number of samples), or ``None`` without rules.

    A rule ``q<-k`` removes every key token of modality ``k`` from every query token
    of modality ``q`` in the same sample; ``q<-k:noisy`` removes only the noisy ones
    (``build_modality_layout`` flags; conditioning tokens stay visible).  Text keys and tokens outside the four
    modalities are never removed.
    """
    if not rules:
        return None
    sample_lens, split_lens = list(packed_seq.sample_lens), list(packed_seq.split_lens)
    if list(packed_seq.attn_modes) != ["causal", "full"] * len(sample_lens):
        raise NotImplementedError("joint_attn_mask requires two-way (causal text + full generation) samples")
    if sum(split_lens) != int(packed_seq.sequence_length):
        raise NotImplementedError("joint_attn_mask does not support padded packs")
    # index 0 = NO_MODALITY; blocked_noisy only cuts keys flagged noisy
    blocked = torch.zeros(len(MODALITIES) + 1, len(MODALITIES) + 1, dtype=torch.bool)
    blocked_noisy = torch.zeros_like(blocked)
    for query, key in rules:
        key, noisy_only = split_rule_key(key)
        (blocked_noisy if noisy_only else blocked)[_MODALITY_ID[query] + 1, _MODALITY_ID[key] + 1] = True
    ids = build_modality_ids(packed_seq, device="cpu") + 1
    noisy = build_modality_layout(packed_seq, device="cpu")[1] if blocked_noisy.any() else None
    present = set(ids.unique().tolist())
    for query, key in rules:
        key = split_rule_key(key)[0]
        if _MODALITY_ID[query] + 1 not in present or _MODALITY_ID[key] + 1 not in present:
            raise ValueError(f"joint_attn_mask rule {query}<-{key} refers to a modality absent from the batch")
    q_parts, kv_parts, q_lens, kv_lens = [], [], [], []
    start = gen_base = 0
    for i, sample_len in enumerate(sample_lens):
        text_len, gen_len = split_lens[2 * i], split_lens[2 * i + 1]
        if text_len + gen_len != sample_len:
            raise ValueError(f"sample {i}: split lens {text_len}+{gen_len} != sample len {sample_len}")
        sample = ids[start : start + sample_len]
        sample_noisy = None if noisy is None else noisy[start : start + sample_len]
        gen = sample[text_len:]
        for group in gen.unique().tolist():
            q = (gen == group).nonzero().flatten()
            drop = blocked[group][sample]
            if sample_noisy is not None:
                drop = drop | (blocked_noisy[group][sample] & sample_noisy)
            kv = (~drop).nonzero().flatten()
            q_parts.append(gen_base + q)
            kv_parts.append(start + kv)
            q_lens.append(q.numel())
            kv_lens.append(kv.numel())
        start += sample_len
        gen_base += gen_len

    def cumulative(lens):
        return torch.tensor([0] + lens, dtype=torch.int64).cumsum(0).to(torch.int32)

    return ModalityGather(
        q_index=torch.cat(q_parts).to(device),
        kv_index=torch.cat(kv_parts).to(device),
        cu_seqlens_q=cumulative(q_lens).to(device),
        cu_seqlens_kv=cumulative(kv_lens).to(device),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
    )


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
    lead_head: bool = False,
) -> list[torch.Tensor]:
    """Euler loop with a per-segment sigma schedule.

    ``latents[b]`` is the flat per-sample state ``[head | geometry]`` where the last
    ``geom_lens[b]`` elements are PointFlow + FK.  Vision / action follow
    ``sigmas`` (num_steps + 1 points); geometry follows ``geom_sigmas`` (the
    sampler's grid for ``num_steps - lead`` steps), then stays fixed at sigma 0.
    ``velocity_fn(state, t, geom_t)`` gets ``(1, 1)`` timesteps in the sampler's own
    units (``sigma * num_train_timesteps``, int64-truncated like ``UniPCSampler``).
    ``lead == 0`` reuses ``sigmas`` for geometry (plain Euler on the sampler grid).
    ``lead_head=True`` (JOINT_VISION_LEAD): the compressed segment is instead the FIRST
    ``geom_lens[b]`` elements (vision) and ``velocity_fn(state, t, None, lead_t)`` gets
    its timestep as the 4th argument; the remaining elements follow ``sigmas``.
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
        t, gt = t_grid[i].reshape(1, 1), gt_grid[g].reshape(1, 1)
        v = velocity_fn(state, t, None, gt) if lead_head else velocity_fn(state, t, gt)
        dt_head = float(sigmas[i + 1] - sigmas[i])
        dt_geom = float(geom_sigmas[g_next] - geom_sigmas[g])
        new_state = []
        for x, vx, glen in zip(state, v, geom_lens, strict=True):
            vx = vx.to(x.dtype)
            if lead_head:
                new_state.append(torch.cat([x[:glen] + dt_geom * vx[:glen], x[glen:] + dt_head * vx[glen:]]))
                continue
            split = x.numel() - glen
            new_state.append(torch.cat([x[:split] + dt_head * vx[:split], x[split:] + dt_geom * vx[split:]]))
        state = new_state
    return state
