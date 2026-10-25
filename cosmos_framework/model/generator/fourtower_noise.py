"""E1 of the four-tower plan (10-25): opt-in per-sample noise mixture over video / PointFlow / FK / action.

FOURTOWER_NOISE_MIX=1 turns it on; it needs independent_{action,pointflow,fk}_schedule=True (so every tower has its
own draw) and SIM_MODE_PROBS on the dataset for the extreme samples (e.g. wam:0.8,forward_dynamics:0.1,inverse_dynamics:0.1).
Per training sample:
  wam, prob 1-FOURTOWER_INDEP_FRAC: shared sigma -- action / PF / FK take the video sigma (the legacy synced path);
  wam, prob FOURTOWER_INDEP_FRAC (default 0.375, i.e. 30% of all samples at 80% wam): every tower keeps its own draw;
  forward_dynamics (action rows clean by the dataset plan): FK clean too (sigma 0, no FK loss); video + PF noisy at the video sigma;
  inverse_dynamics (all video frames clean by the plan): PF clean too (sigma 0, no PF loss); action + FK noisy at one shared sigma.
Clean towers drop out of their loss, and the video / action losses are averaged over the samples that still have noisy
tokens (an all-wam batch is unchanged). Validation and inference are untouched (synchronous denoising).
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch


def enabled() -> bool:
    return os.environ.get("FOURTOWER_NOISE_MIX", "0") == "1"


def indep_frac() -> float:
    frac = float(os.environ.get("FOURTOWER_INDEP_FRAC", "0.375"))
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"FOURTOWER_INDEP_FRAC must be in [0, 1], got {frac}")
    return frac


def mix_sigmas(vision_clean, action_clean, action_index, sigma_vision, sigma_action, sigma_pf, sigma_fk, *, frac, generator=None):
    """Return (sigma_action [n,1], sigma_pf [B,1], sigma_fk [B,1], pf_drop [B], fk_drop [B], metrics).

    vision_clean [B] / action_clean [n] bool: the sample's video / action is all conditioning (IDM / FDM by the plan).
    action_index [n]: batch slot of each dense action entry. All sigmas are the independent draws; the video sigma is
    the shared one ([B,1], one value per sample).
    """
    B = sigma_vision.shape[0]
    if sigma_vision.shape[1] != 1:
        raise ValueError("FOURTOWER_NOISE_MIX needs one video sigma per sample (no diffusion forcing)")
    dev = sigma_vision.device
    fdm = torch.zeros(B, dtype=torch.bool, device=dev)
    fdm[action_index[action_clean]] = True
    idm = vision_clean.to(dev) & ~fdm
    wam = ~(fdm | idm)
    indep = wam & (torch.rand(B, generator=generator, device=dev) < frac)
    shared = ~indep  # shared wam rows, plus the noisy side of FDM / IDM
    sv = sigma_vision[:, :1]
    sigma_pf = torch.where(shared[:, None], sv, sigma_pf)
    sigma_fk = torch.where(shared[:, None], sv, sigma_fk)
    sigma_pf = torch.where(idm[:, None], torch.zeros_like(sigma_pf), sigma_pf)
    sigma_fk = torch.where(fdm[:, None], torch.zeros_like(sigma_fk), sigma_fk)
    sigma_action = torch.where(shared[action_index][:, None], sv[action_index], sigma_action)
    metrics = {
        "fourtower_mix_shared": (wam & ~indep).sum().float(),
        "fourtower_mix_indep": indep.sum().float(),
        "fourtower_mix_fdm": fdm.sum().float(),
        "fourtower_mix_idm": idm.sum().float(),
    }
    return sigma_action, sigma_pf, sigma_fk, idm, fdm, metrics


def drop_samples(clean, drop):
    """Copy of a PointFlow / FK label batch whose supervision skips the samples in ``drop`` [B]."""
    if drop is None or not bool(drop.any()):
        return clean
    keep = ~drop.to(clean.valid.device)[clean.inputs["point_batch"]]  # [sum(N)]
    return replace(clean, valid=clean.valid & keep[None])


def active_rescale(loss, condition_mask):
    """Batch-mean loss -> mean over the samples that still have noisy tokens (fully clean samples add exact zeros)."""
    n_active = sum(int(not bool(torch.all(m == 1))) for m in condition_mask)
    return loss * (len(condition_mask) / max(n_active, 1))
