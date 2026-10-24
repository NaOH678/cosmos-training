"""Opt-in four-tower cross-modal consistency losses (E2 of the System-1 four-tower plan, 10-24).

The physical chain action -> FK keypoints -> PointFlow (3D point tracks) -> video, written into the objective as three
losses on the one-step clean estimates ``x0_hat = x_t - sigma * v_hat`` of the current denoising step:

``L_JM``  action -> FK.  The predicted action is pushed through the differentiable URDF FK (``pointfk_consistency``)
          and its camera-frame keypoint displacement is compared step by step with the FK tower's x0 (same rate).
``L_MF``  FK -> PointFlow.  Per 4-step window (one latent), each keypoint's accumulated displacement is compared with
          the PointFlow displacement over the same window, read at the keypoint's location (Gaussian-weighted mean of
          the PF points around it at the window start).  Our PointFlow is 3D, so this is done in 3D, not after
          projection to the image as for 2D optical flow.
``L_FV``  PointFlow -> video.  The predicted video latents are decoded (frozen Wan VAE, with gradient) and pooled to a
          low resolution; every PF point is projected into consecutive frames and the colour it carries in frame k
          must match the colour at its predicted position in frame k+1 (photometric consistency along the tracks).
          PointFlow is sparse, so this is a per-track warp rather than a dense image warp; occluded points are removed
          with the tracks' visibility (``valid``) instead of a forward-backward check.

Both sides of every pair receive gradient.  A pair is active for a sample only when both modalities' sigma is below
``FOURTOWER_CONS_SIGMA_MAX`` (default 0.5).  ``L_FV`` is computed on a random ``FOURTOWER_CONS_FV_FRAC`` (default
0.25) of the active samples because of the decoder cost.  Every loss is off unless its weight is positive.

Environment
-----------
FOURTOWER_CONS_W_JM / _W_MF / _W_FV  float, default 0.  Weights added to ``total_loss``.
FOURTOWER_CONS_SIGMA_MAX             float, default 0.5.
FOURTOWER_CONS_FV_FRAC               float, default 0.25.
FOURTOWER_CONS_FV_POOL               int, default 2 (decoded frames are average-pooled by this factor).
FOURTOWER_CONS_FV_MAX                int, default 1 (at most this many decoded samples per rank and step).
FOURTOWER_CONS_MF_RADIUS             metres, default 0.015 (Gaussian sigma of the PF read-out around a keypoint).
FOURTOWER_CONS_EPS                   Charbonnier eps in each loss's units, default 0.01.
L_JM also needs POINTFK_CONSISTENCY_STATS (and optionally _URDF / _FK_JSON), see ``pointfk_consistency``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

KEYS = ("JM", "MF", "FV")


def weights() -> dict[str, float]:
    out = {}
    for k in KEYS:
        raw = os.environ.get(f"FOURTOWER_CONS_W_{k}", "0")
        w = float(raw)
        if not w >= 0:
            raise ValueError(f"FOURTOWER_CONS_W_{k} must be >= 0, got {raw!r}")
        out[k] = w
    return out


def _env(name, default):
    return float(os.environ.get(name, default))


def _charbonnier(diff, eps):
    """diff [..., C] -> [...]; sqrt(mean_c d^2 + eps^2) - eps."""
    return (diff.square().mean(-1) + eps**2).sqrt() - eps


def _masked_mean(err, mask):
    mask = mask.to(err.dtype)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def _sigma_of(sigma_list):
    return torch.stack([s.float().max() for s in sigma_list])


def x0_vision(out_net, gen_data_noised, i):
    """fp32 clean video latent estimate [C,T,H,W] and its sigma (max over latent frames)."""
    eps = gen_data_noised.epsilon_vision[i].float()
    sigma = gen_data_noised.sigmas_vision[i].float()
    vt = gen_data_noised.vt_target_vision[i].float()
    v = out_net["preds_vision"][i].float()
    return eps - (1.0 - sigma) * vt - sigma * v, sigma.max()


def _per_sample_bhk(x, batch, k):
    """[H, B*K, ...] (samples contiguous) -> [B, H, K, ...]."""
    return x.reshape(x.shape[0], batch, k, *x.shape[2:]).transpose(0, 1)


def loss_jm(out_net, data_batch_packed, gen_data_noised, *, fk_scale, active, eps):
    """action -> FK -> keypoints vs FK tower, per step.  Returns (loss, metrics)."""
    from cosmos_framework.model.generator.pointfk_consistency import _resources, action_x0_hat, keypoints_displacement

    fk_data, fk_state = data_batch_packed.fk_data, gen_data_noised.fk
    batch = len(fk_data.labeled)
    fk, offset, scale = _resources(fk_data.displacement.device)
    k = fk.num_keypoints
    x0_action = torch.stack([action_x0_hat(out_net, gen_data_noised, i)[0] for i in range(batch)])  # [B,H+1,D]
    anchor = fk_data.inputs["anchor_xyz"].reshape(batch, k, 3).float()
    d_act = keypoints_displacement(x0_action[..., : offset.numel()] * scale + offset, anchor, fk, fk_scale)  # [B,H,K,3]
    sigma = fk_state.sigma.float()[fk_data.inputs["point_batch"]][None, :, None]
    x0_fk = _per_sample_bhk(fk_state.xt.float() - sigma * out_net["preds_fk"].float(), batch, k)  # model space
    valid = _per_sample_bhk(fk_data.valid, batch, k) & active[:, None, None]
    err = _charbonnier(d_act - x0_fk, eps)
    loss = _masked_mean(err, valid)
    with torch.no_grad():
        mm = _masked_mean((d_act - x0_fk).norm(dim=-1) * fk_scale * 1000.0, valid)
    return loss, {"fourtower_jm_loss": loss.detach(), "fourtower_jm_mm": mm, "fourtower_jm_active": active.float().sum()}


def _windows(steps, per):
    return [(w * per, (w + 1) * per) for w in range(steps // per)]  # (start, end) in steps; 0 = anchor


def _at(d, t):
    """Displacement at step t (0 = anchor -> zeros) from [H, N, 3] indexed by step-1."""
    return torch.zeros_like(d[0]) if t == 0 else d[t - 1]


def loss_mf(out_net, data_batch_packed, gen_data_noised, *, fk_scale, pf_scale, active, eps, radius):
    """FK keypoint window displacement vs PointFlow window displacement read at the keypoint (3D)."""
    fk_data, fk_state = data_batch_packed.fk_data, gen_data_noised.fk
    pf_data, pf_state = data_batch_packed.pointflow_data, gen_data_noised.pointflow
    batch = len(fk_data.labeled)
    fk_sig = fk_state.sigma.float()[fk_data.inputs["point_batch"]][None, :, None]
    x0_fk = (fk_state.xt.float() - fk_sig * out_net["preds_fk"].float()) * fk_scale  # metres [H,sumK,3]
    pf_sig = pf_state.sigma.float()[pf_data.inputs["point_batch"]][None, :, None]
    x0_pf = (pf_state.xt.float() - pf_sig * out_net["preds_pointflow"].float()) * pf_scale  # metres [H,sumN,3]
    per = int(pf_data.timing.steps_per_token)
    steps = int(pf_data.timing.steps)
    if int(fk_data.timing.steps) != steps:
        raise ValueError("L_MF needs FK and PointFlow on the same step grid")
    errs, masks = [], []
    fk_b, pf_b = fk_data.inputs["point_batch"], pf_data.inputs["point_batch"]
    for b in range(batch):
        if not bool(active[b]):
            continue
        fi, pi = (fk_b == b).nonzero().squeeze(1), (pf_b == b).nonzero().squeeze(1)
        if not len(pi):
            continue
        fk_anchor, pf_anchor = fk_data.inputs["anchor_xyz"][fi].float(), pf_data.inputs["anchor_xyz"][pi].float()
        dk, dp = x0_fk[:, fi], x0_pf[:, pi]
        vk, vp = fk_data.valid[:, fi], pf_data.valid[:, pi]
        for s, e in _windows(steps, per):
            with torch.no_grad():  # read-out weights from the window-start positions (no gradient through the weights)
                pk = fk_anchor + _at(dk, s)
                pp = pf_anchor + _at(dp, s)
                d2 = torch.cdist(pk, pp).square()  # [K, N]
                ok_p = (vp[s - 1] if s > 0 else torch.ones_like(vp[0])) & vp[e - 1]
                w = torch.exp(-d2 / (2 * radius**2)) * (d2 < (3 * radius) ** 2) * ok_p[None]
                support = w.sum(1)
                ok_k = (support > 0.5) & vk[e - 1] & ((vk[s - 1] if s > 0 else torch.ones_like(vk[0])))
            flow_k = _at(dk, e) - _at(dk, s)  # [K,3] metres
            flow_p = (w @ (_at(dp, e) - _at(dp, s))) / support.clamp_min(1e-6)[:, None]
            errs.append(_charbonnier((flow_k - flow_p) / fk_scale, eps))
            masks.append(ok_k)
    if not errs:
        zero = x0_fk.sum() * 0.0 + x0_pf.sum() * 0.0
        return zero, {"fourtower_mf_loss": zero.detach()}
    err, mask = torch.cat(errs), torch.cat(masks)
    loss = _masked_mean(err, mask)
    return loss, {"fourtower_mf_loss": loss.detach(), "fourtower_mf_keypoints": mask.float().sum()}


def decode_with_grad(model, latent):
    """Frozen Wan2.2 VAE decode that keeps the autograd graph: [C,T,H,W] -> [3,T',H',W'] in [-1,1]."""
    iface = model.tokenizer_vision_gen.model  # the @no_grad wrapper; its .model is the WanVAE_ module
    z = (latent if latent.ndim == 5 else latent[None]).to(iface.dtype)  # vision items are stored [1,C,T,H,W]
    return iface.model.decode(z, iface.scale, True)[0].float()


def project_to_video(xyz, inputs, b):
    """Camera-frame points [..., 3] -> video pixel coordinates [..., 2] for sample b."""
    k = torch.diag(torch.cat([inputs["image_size_wh"][b].float(), xyz.new_ones(1)])) @ inputs["intrinsics_normalized"][b].float()
    q = xyz @ k.T
    uv = q[..., :2] / q[..., 2:].clamp_min(1e-4)
    a = inputs["uv_to_video"][b].float()  # [2,3] tracker pixel -> video pixel
    return uv @ a[:, :2].T + a[:, 2]


def loss_fv(model, out_net, data_batch_packed, gen_data_noised, *, pf_scale, active, eps, frac, pool, generator=None):
    """Photometric consistency of decoded consecutive frames along the predicted PointFlow tracks."""
    pf_data, pf_state = data_batch_packed.pointflow_data, gen_data_noised.pointflow
    inputs = pf_data.inputs
    pf_sig = pf_state.sigma.float()[inputs["point_batch"]][None, :, None]
    x0_pf = (pf_state.xt.float() - pf_sig * out_net["preds_pointflow"].float()) * pf_scale  # metres [H,sumN,3]
    pick = active & (torch.rand(active.shape, device=active.device, generator=generator) < frac)
    cap = int(_env("FOURTOWER_CONS_FV_MAX", "1"))  # decoder activations are large: at most this many decodes per rank and step
    pick &= torch.cumsum(pick.long(), 0) <= cap
    errs, masks = [], []
    for b in pick.nonzero().squeeze(1).tolist():
        pi = (inputs["point_batch"] == b).nonzero().squeeze(1)
        if not len(pi):
            continue
        lat, _ = x0_vision(out_net, gen_data_noised, b)
        frames = decode_with_grad(model, lat)  # [3,T,H,W]
        t_frames, h, w = frames.shape[1:]
        if t_frames != pf_data.timing.steps + 1:
            raise ValueError(f"L_FV: {t_frames} decoded frames vs {pf_data.timing.steps} PointFlow steps")
        # The model crops the right/bottom padding of the 640x640 canvas off the latents (_remove_padding_from_latent),
        # so pixel coordinates are unchanged and points beyond the decoded width fall outside (masked below).
        vw, vh = inputs["video_size_wh"][b].tolist()
        if h > vh or w > vw:
            raise ValueError(f"L_FV: decoded {w}x{h} exceeds video_size_wh {vw}x{vh}")
        low = F.avg_pool2d(frames.transpose(0, 1), pool) if pool > 1 else frames.transpose(0, 1)  # [T,3,h',w']
        anchor = inputs["anchor_xyz"][pi].float()
        pos = torch.cat([anchor[None], anchor[None] + x0_pf[:, pi]])  # [T,N,3], step 0 = anchor
        uv = project_to_video(pos, inputs, b)  # [T,N,2] full-res video pixels
        grid = torch.stack([(uv[..., 0] + 0.5) / w * 2 - 1, (uv[..., 1] + 0.5) / h * 2 - 1], -1)  # align_corners=False
        color = F.grid_sample(low, grid[:, :, None], mode="bilinear", align_corners=False)[..., 0]  # [T,3,N]
        valid = torch.cat([torch.ones_like(pf_data.valid[:1, pi]), pf_data.valid[:, pi]])  # [T,N]
        inside = (grid.abs() <= 1).all(-1)
        ok = valid[:-1] & valid[1:] & inside[:-1] & inside[1:]
        errs.append(_charbonnier((color[1:] - color[:-1]).transpose(1, 2), eps))  # [T-1,N]
        masks.append(ok)
    if not errs:
        zero = x0_pf.sum() * 0.0 + sum(p.sum() for p in out_net["preds_vision"]) * 0.0
        return zero, {"fourtower_fv_loss": zero.detach(), "fourtower_fv_samples": pick.float().sum()}
    err, mask = torch.cat([e.flatten() for e in errs]), torch.cat([m.flatten() for m in masks])
    loss = _masked_mean(err, mask)
    return loss, {"fourtower_fv_loss": loss.detach(), "fourtower_fv_samples": pick.float().sum(), "fourtower_fv_tracks": mask.float().sum()}


def compute_fourtower_losses(model, out_net, data_batch_packed, gen_data_noised, *, fk_scale, pf_scale, w):
    """Weighted sum of the enabled losses (scalar) and detached metrics.  ``w`` = ``weights()``."""
    sigma_max = _env("FOURTOWER_CONS_SIGMA_MAX", "0.5")
    eps = _env("FOURTOWER_CONS_EPS", "0.01")
    fk_data, pf_data = data_batch_packed.fk_data, data_batch_packed.pointflow_data
    batch = len(fk_data.labeled)
    if len(out_net["preds_action"]) != batch or len(out_net["preds_vision"]) != batch:
        raise ValueError("four-tower consistency needs one action and one video item per FK sample")
    if pf_data is None or len(pf_data.labeled) != batch:
        raise ValueError("four-tower consistency needs PointFlow on every sample")
    s_act = torch.stack([gen_data_noised.sigmas_action[i].float().max() for i in range(batch)])
    s_fk, s_pf = gen_data_noised.fk.sigma.float(), gen_data_noised.pointflow.sigma.float()
    s_vis = _sigma_of(gen_data_noised.sigmas_vision)
    low = lambda s: (s < sigma_max) & (s > 0)  # noqa: E731  (sigma 0 = all-conditioning sample, no x0 to rebuild)
    total, metrics = 0.0, {"fourtower_sigma_lt_max_frac": low(s_pf).float().mean()}
    if w["JM"] > 0:
        l, m = loss_jm(out_net, data_batch_packed, gen_data_noised, fk_scale=fk_scale, active=low(s_act) & low(s_fk), eps=eps)
        total = total + w["JM"] * l
        metrics.update(m)
    if w["MF"] > 0:
        l, m = loss_mf(out_net, data_batch_packed, gen_data_noised, fk_scale=fk_scale, pf_scale=pf_scale,
                       active=low(s_fk) & low(s_pf), eps=eps, radius=_env("FOURTOWER_CONS_MF_RADIUS", "0.015"))
        total = total + w["MF"] * l
        metrics.update(m)
    if w["FV"] > 0:
        l, m = loss_fv(model, out_net, data_batch_packed, gen_data_noised, pf_scale=pf_scale, active=low(s_pf) & low(s_vis),
                       eps=eps, frac=_env("FOURTOWER_CONS_FV_FRAC", "0.25"), pool=int(_env("FOURTOWER_CONS_FV_POOL", "2")))
        total = total + w["FV"] * l
        metrics.update(m)
    # Keep every head in the graph on every rank even when no sample is active (FSDP reduce must not hang).
    keep = sum(p.sum() for p in out_net["preds_action"]) + out_net["preds_fk"].sum() + out_net["preds_pointflow"].sum()
    total = total + 0.0 * (keep + sum(p.sum() for p in out_net["preds_vision"]))
    metrics["fourtower_loss"] = total.detach()
    if os.environ.get("FOURTOWER_CONS_DEBUG") == "1":  # smoke tests: per-step values and peak memory on stdout
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print("[fourtower] " + " ".join(f"{k}={float(v):.4g}" for k, v in metrics.items()) + f" peak_mem_gib={mem:.1f}", flush=True)
    return total, metrics
