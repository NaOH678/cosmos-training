"""E1 of the four-tower plan (10-25): opt-in per-sample noise mixture over video / PointFlow / FK / action.

FOURTOWER_NOISE_MIX=1 turns it on; it needs independent_{action,pointflow,fk}_schedule=True (so every tower has its
own draw) and SIM_MODE_PROBS on the dataset for the extreme samples (e.g. wam:0.8,forward_dynamics:0.1,inverse_dynamics:0.1).
Per training sample (the PDF's 50 / 30 / 20 mixture):
  wam, prob 1-FOURTOWER_INDEP_FRAC: shared sigma -- action / PF / FK take the video sigma (the legacy synced path);
  wam, prob FOURTOWER_INDEP_FRAC (default 0.375, i.e. 30% of all samples at 80% wam): every tower draws sigma ~ U(0, 1);
  forward_dynamics (action rows clean by the dataset plan): FK clean too (sigma 0, no FK loss); video + PF at full noise (sigma 1);
  inverse_dynamics (all video frames clean by the plan): PF clean too (sigma 0, no PF loss); action + FK at full noise (sigma 1).
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
    """Return (sigma_vision [B,1], sigma_action [n,1], sigma_pf [B,1], sigma_fk [B,1], pf_drop [B], fk_drop [B], metrics).

    vision_clean [B] / action_clean [n] bool: the sample's video / action is all conditioning (IDM / FDM by the plan).
    action_index [n]: batch slot of each dense action entry. The video sigma ([B,1]) is the shared draw; the other
    inputs are only used for their shape and device.
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
    shared = wam & ~indep
    sv = sigma_vision[:, :1]

    def pick(n, rows):  # per-row sigma [n,1]: shared -> video draw, independent -> U(0,1), FDM / IDM -> fixed below
        u = torch.rand(n, 1, generator=generator, device=dev)
        return torch.where(shared[rows][:, None], sv[rows], u)

    every = torch.arange(B, device=dev)
    one, zero = torch.ones(B, 1, device=dev), torch.zeros(B, 1, device=dev)
    sigma_vision = torch.where(fdm[:, None], one, pick(B, every))  # IDM video is conditioning (sigma ignored)
    sigma_pf = torch.where(fdm[:, None], one, torch.where(idm[:, None], zero, pick(B, every)))
    sigma_fk = torch.where(idm[:, None], one, torch.where(fdm[:, None], zero, pick(B, every)))
    sigma_action = torch.where(idm[action_index][:, None], 1.0, pick(len(action_index), action_index))  # FDM: conditioning
    metrics = {
        "fourtower_mix_shared": (wam & ~indep).sum().float(),
        "fourtower_mix_indep": indep.sum().float(),
        "fourtower_mix_fdm": fdm.sum().float(),
        "fourtower_mix_idm": idm.sum().float(),
    }
    if os.environ.get("FOURTOWER_NOISE_DEBUG") == "1":  # smoke tests: per-step mixture on stdout
        print("[noisemix] " + " ".join(f"{k}={int(v)}" for k, v in metrics.items())
              + f" sv={sigma_vision[:, 0].tolist()} sa={sigma_action[:, 0].tolist()} spf={sigma_pf[:, 0].tolist()} sfk={sigma_fk[:, 0].tolist()}", flush=True)
    return sigma_vision, sigma_action, sigma_pf, sigma_fk, idm, fdm, metrics


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
