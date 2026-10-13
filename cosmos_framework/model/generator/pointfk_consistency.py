"""Opt-in cross-modal consistency loss: action <-> FK keypoints (joint video/action/PointFlow/FK model).

Disabled unless ``POINTFK_CONSISTENCY_WEIGHT`` is a positive float; while disabled
``OmniMoTModel._compute_losses`` never imports this module's heavy parts, adds no loss
key and does no extra compute.

Environment
-----------
POINTFK_CONSISTENCY_WEIGHT  float, default 0 (off).  Multiplies the consistency loss added to ``total_loss``.
POINTFK_CONSISTENCY_MODE    ``task`` | ``gt`` | ``pred_to_action`` | ``task+pred`` | ``both`` (default ``task``).
POINTFK_CONSISTENCY_STATS   normalization_train_only.json used by the dataset (``action.q01/q99``, ``joint_names``).
POINTFK_CONSISTENCY_URDF    robot URDF (default: the task21/27 Multi_UR5_wuji_with_flange.urdf reference copy).
POINTFK_CONSISTENCY_FK_JSON an ``fk21.json`` (keypoint ``mapping`` + ``world_from_robot_base``; constant across episodes).
POINTFK_CONSISTENCY_NORM    ``charbonnier`` (default, eps=POINTFK_CONSISTENCY_EPS=0.01 in model units) | ``mse`` | ``l1``.

Definitions (all in FK model space = metres / fk_displacement_scale)
-------------------------------------------------------------------
x0_action_hat = x_t - sigma * v_pred (fp32; x_t is rebuilt exactly in fp32 as eps - (1-sigma) * v_target, so
bf16 rounding of the stored noisy tokens does not enter).  Row 0 is the conditioning state (sigma 0 -> exact).
Denormalise with the dataset's affine stats, K_k = FK(a_k), D_act[k] = (K_k - K_0) / scale  for k = 1..H, rotated
into the camera frame.  The camera is per-episode; it is recovered per sample by a Kabsch fit of FK(a_0) onto the
batch's own ``fk_data.inputs['anchor_xyz']`` (camera-frame keypoints of window frame 0), so no data change is needed.

``task``            L = d(D_act(pred x0), D_act(GT x0))   grad -> action head only, weight (1 - sigma_action)^2.
                    GT x0 = eps - v_target (the normalised training target), pushed through the same FK chain, so
                    there is no commanded-vs-realised mismatch.  Default.
``gt``              L = d(D_act, FK_gt)                  grad -> action head, weight (1 - sigma_action)^2.
                    BIASED: FK_gt is built from realised qpos[s+k] while action rows are commanded[s+k-1]; their
                    displacements differ by ~16.5 mm mean (a ~17.7 mm ADE floor even for perfect actions), so this
                    pulls actions toward realised rather than commanded motion.
``pred_to_action``  L = d(x0_fk, stopgrad(D_act))        grad -> FK head only, weight (1 - sigma_fk)^2.
``task+pred``       task + pred_to_action.
``both``            gt + pred_to_action.
"""

from __future__ import annotations

import json
import os

import torch

DEFAULT_URDF = "/data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2/reference/Multi_UR5_wuji_with_flange.urdf"
DEFAULT_FK_JSON = "/data/shichaojian/sim_data/bench2dex_task27_ball/episode_000000/fk21.json"
MODES = ("task", "gt", "pred_to_action", "task+pred", "both")
NORMS = ("charbonnier", "mse", "l1")
_CACHE: dict = {}


def consistency_weight() -> float:
    raw = os.environ.get("POINTFK_CONSISTENCY_WEIGHT", "0")
    try:
        weight = float(raw)
    except ValueError as err:
        raise ValueError(f"POINTFK_CONSISTENCY_WEIGHT must be a float, got {raw!r}") from err
    if weight < 0 or weight != weight:
        raise ValueError(f"POINTFK_CONSISTENCY_WEIGHT must be >= 0, got {raw!r}")
    return weight


def consistency_mode() -> str:
    mode = os.environ.get("POINTFK_CONSISTENCY_MODE", "task")
    if mode not in MODES:
        raise ValueError(f"POINTFK_CONSISTENCY_MODE must be one of {MODES}, got {mode!r}")
    return mode


def load_action_stats(path: str):
    """(joint_names, offset[J], scale[J]) exactly as ``sim_pointfk_dataset.SimPointFKSFTDataset`` builds them."""
    norm = json.loads(open(path).read())
    low, high = (torch.tensor(norm["action"][k], dtype=torch.float32) for k in ("q01", "q99"))
    return list(norm["joint_names"]), (low + high) / 2, torch.clamp((high - low) / 2, min=0.05)


def _resources(device):
    key = str(device)
    if key not in _CACHE:
        from cosmos_framework.model.generator.robot_fk_torch import TorchRobotFK

        stats = os.environ.get("POINTFK_CONSISTENCY_STATS")
        if not stats:
            raise ValueError("POINTFK_CONSISTENCY_STATS (dataset normalization_train_only.json) must be set")
        names, offset, scale = load_action_stats(stats)
        fk = TorchRobotFK.from_json(
            os.environ.get("POINTFK_CONSISTENCY_URDF", DEFAULT_URDF),
            os.environ.get("POINTFK_CONSISTENCY_FK_JSON", DEFAULT_FK_JSON),
            names,
        )
        _CACHE[key] = (fk.to(device), offset.to(device), scale.to(device))
    return _CACHE[key]


def _distance(a, b, norm, eps):
    squared = (a - b).square().mean(-1)
    if norm == "mse":
        return squared
    if norm == "l1":
        return squared.clamp_min(0).sqrt()
    return (squared + eps**2).sqrt() - eps


def _per_sample(error, valid, weight):
    """Masked mean over [H, keypoints] per sample, scaled by the per-sample sigma weight; returns [B]."""
    valid = valid.to(error.dtype)
    return (error * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1) * weight


def action_x0_hat(out_net, gen_data_noised, index):
    """fp32 ``x_t - sigma * v`` for dense action sample ``index`` -> ([T, D] estimate, [T, 1] sigma)."""
    eps = gen_data_noised.epsilon_action[index].float()
    sigma = gen_data_noised.sigmas_action[index].float()
    target_velocity = gen_data_noised.vt_target_action[index].float()
    velocity = out_net["preds_action"][index].float()
    return eps - (1.0 - sigma) * target_velocity - sigma * velocity, sigma


def keypoints_displacement(raw_joints, anchor_xyz, fk, scale):
    """Camera-frame keypoint displacement ``(K_k - K_0)`` in metres/scale.  ``raw_joints [B,T,J]`` (rad) -> [B,T-1,42,3]."""
    from cosmos_framework.model.generator.robot_fk_torch import estimate_rigid_transform

    world = fk(raw_joints)  # [B,T,K,3] robot-world frame, fp32, differentiable
    rotation, _ = estimate_rigid_transform(world[:, 0], anchor_xyz)  # cam <- world, no grad (translation cancels below)
    return torch.einsum("bij,btkj->btki", rotation, world[:, 1:] - world[:, :1]) / scale


def compute_consistency_losses(out_net, data_batch_packed, gen_data_noised, *, fk_scale, mode=None, norm=None, eps=None):
    """Returns (loss, metrics): unweighted consistency loss (scalar) and detached scalar diagnostics."""
    mode = mode or consistency_mode()
    norm = norm or os.environ.get("POINTFK_CONSISTENCY_NORM", "charbonnier")
    if norm not in NORMS:
        raise ValueError(f"POINTFK_CONSISTENCY_NORM must be one of {NORMS}, got {norm!r}")
    eps = float(os.environ.get("POINTFK_CONSISTENCY_EPS", "0.01")) if eps is None else eps
    fk_data, fk_state = data_batch_packed.fk_data, gen_data_noised.fk
    preds_action = out_net["preds_action"]
    batch = len(fk_data.labeled)
    if len(preds_action) != batch or not bool(fk_data.labeled.all()):
        raise ValueError(
            "consistency loss needs every batch sample to carry both action and FK "
            f"(n_action={len(preds_action)}, fk_labeled={int(fk_data.labeled.sum())}/{batch})"
        )
    horizon = fk_data.timing.steps
    device = fk_data.displacement.device
    fk, offset, scale = _resources(device)
    joints = offset.numel()

    hats, sigmas_a = zip(*(action_x0_hat(out_net, gen_data_noised, i) for i in range(batch)), strict=True)
    if any(h.shape[0] != horizon + 1 for h in hats):
        raise ValueError(f"action rows must be 1 + FK steps ({horizon + 1}), got {[h.shape[0] for h in hats]}")
    x0_action = torch.stack(hats)  # [B,H+1,D] normalised
    sigma_action = torch.stack([s.max() for s in sigmas_a])  # [B]; conditioning row is 0, noisy rows share one sigma
    # An all-conditioning action sample has eps = v = 0 stored, so x0 can't be rebuilt: exclude it.
    action_ok = (sigma_action > 0).float()

    points = fk_data.inputs["anchor_xyz"]
    if points.shape[0] != batch * fk.num_keypoints:
        raise ValueError("every FK sample must carry exactly the robot's keypoints")
    anchor = points.reshape(batch, fk.num_keypoints, 3).float()
    # FK tensors are laid out [H, sum(N), 3] with samples contiguous -> [B, H, K, 3].
    to_bhk = lambda x: x.reshape(horizon, batch, fk.num_keypoints, *x.shape[2:]).transpose(0, 1)  # noqa: E731
    target_gt = to_bhk(fk_data.displacement.float()) / fk_scale
    valid = to_bhk(fk_data.valid)

    metrics, total = {}, x0_action.new_zeros(())
    if mode in ("task", "task+pred"):
        x0_gt = torch.stack([gen_data_noised.epsilon_action[i].float() - gen_data_noised.vt_target_action[i].float() for i in range(batch)])
        with torch.no_grad():
            d_gt_act = keypoints_displacement(x0_gt[..., :joints] * scale + offset, anchor, fk, fk_scale)
        d_act = keypoints_displacement(x0_action[..., :joints] * scale + offset, anchor, fk, fk_scale)
        weight = (1.0 - sigma_action).square() * action_ok
        per_sample = _per_sample(_distance(d_act, d_gt_act, norm, eps), valid, weight)
        loss_task = per_sample.sum() / action_ok.sum().clamp_min(1)
        total = total + loss_task
        metrics["consistency_loss_task"] = loss_task.detach()
        with torch.no_grad():
            low = (sigma_action < 0.5) & (action_ok > 0)
            if bool(low.any()):
                ade = (d_act - d_gt_act).norm(dim=-1) * fk_scale * 1000.0
                m = valid[low].float()
                metrics["consistency_loss_task_ade_mm_sigma_lt_0.5"] = (ade[low] * m).sum() / m.sum().clamp_min(1)
    if mode in ("gt", "both"):
        raw = x0_action[..., :joints] * scale + offset  # denormalise (dataset affine); padded dims dropped
        d_act = keypoints_displacement(raw, anchor, fk, fk_scale)
        weight = (1.0 - sigma_action).square() * action_ok
        per_sample = _per_sample(_distance(d_act, target_gt, norm, eps), valid, weight)
        loss_gt = per_sample.sum() / action_ok.sum().clamp_min(1)
        total = total + loss_gt
        metrics["consistency_loss_gt"] = loss_gt.detach()
        with torch.no_grad():
            ade = (d_act - target_gt).norm(dim=-1) * fk_scale * 1000.0
            low = (sigma_action < 0.5) & (action_ok > 0)
            if bool(low.any()):
                m = valid[low].float()
                metrics["consistency_loss_gt_ade_mm_sigma_lt_0.5"] = (ade[low] * m).sum() / m.sum().clamp_min(1)
    if mode in ("pred_to_action", "task+pred", "both"):
        with torch.no_grad():  # stop-gradient target: nothing flows into the action head from this term
            raw = x0_action.detach()[..., :joints] * scale + offset
            d_act_sg = keypoints_displacement(raw, anchor, fk, fk_scale)
        sigma_fk = fk_state.sigma.float()  # [B]
        per_point_sigma = sigma_fk[fk_data.inputs["point_batch"]][None, :, None]
        x0_fk = to_bhk(fk_state.xt.float() - per_point_sigma * out_net["preds_fk"].float())  # model space
        weight = (1.0 - sigma_fk).square() * action_ok
        per_sample = _per_sample(_distance(x0_fk, d_act_sg, norm, eps), valid, weight)
        loss_p2a = per_sample.sum() / action_ok.sum().clamp_min(1)
        total = total + loss_p2a
        metrics["consistency_loss_pred_to_action"] = loss_p2a.detach()
    metrics["consistency_loss"] = total.detach()
    metrics["consistency_loss_active_samples"] = action_ok.sum().detach()
    # Keep both heads in the graph even if no sample is eligible (FSDP/DDP reduce must not hang); value is exactly 0.
    total = total + 0.0 * (sum(p.sum() for p in preds_action) + out_net["preds_fk"].sum())
    return total, metrics
