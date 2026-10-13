"""CPU tests for the action<->FK consistency loss (real task27 data for FK validation; skipped if absent)."""

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.data.fk_batch import build_fk_batch
from cosmos_framework.model.generator import pointfk_consistency as pc
from cosmos_framework.model.generator.fk_training import fk_add_noise
from cosmos_framework.model.generator.robot_fk_torch import TorchRobotFK, estimate_rigid_transform

DATA = Path(os.environ.get("TASK27_RAW", "/data/shichaojian/sim_data/bench2dex_task27_ball"))
CACHE = Path(os.environ.get("TASK27_CACHE", str(DATA) + "_training/datasets/cosmos-cache/pointflow_fk_handguided1024"))
STATS = os.environ.get("POINTFK_CONSISTENCY_STATS", str(DATA) + "_training/normalization_train_only.json")
SCALE = 0.06544707570831351
needs_data = pytest.mark.skipif(not (DATA.exists() and CACHE.exists() and Path(STATS).exists()), reason="task27 data absent")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("POINTFK_CONSISTENCY_STATS", STATS)
    pc._CACHE.clear()


def _isaac(E):
    o = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    out = E.copy()
    out[..., :3, :3] = E[..., :3, :3] @ o
    return out


@needs_data
def test_fk_matches_stored_targets():
    import h5py

    urdf = pc.DEFAULT_URDF
    errors = []
    eps = sorted(glob.glob(str(DATA / "episode_*")))
    for ep in (eps[0], eps[len(eps) // 2], eps[-1]):
        with h5py.File(ep + "/rgbd.hdf5", "r") as f:
            names = [n.decode() for n in f["robot/joint_names"][:]]
            q = torch.tensor(f["robot/qpos"][:])
            ext = _isaac(f["cameras/cam_overhead/extrinsic_world_from_cam"][:])
        fk = TorchRobotFK.from_json(urdf, ep + "/fk21.json", names)
        stored = np.load(ep + "/fk21.npz")["positions_camera"].reshape(-1, 42, 3)
        world = fk(q).double().numpy()
        inv = np.linalg.inv(ext.astype(np.float64))
        cam = np.einsum("tij,tkj->tki", inv[:, :3, :3], world) + inv[:, None, :3, 3]
        errors.append(np.linalg.norm(cam - stored, axis=-1).mean() * 1000)
        for s in (1, len(stored) // 2):  # displacement windows
            disp = cam[s + 1 : s + 33] - cam[s]
            assert np.linalg.norm(disp - (stored[s + 1 : s + 33] - stored[s]), axis=-1).mean() * 1000 < 1.0
    assert max(errors) < 1.0, errors  # mean error in mm


def test_rigid_fit_recovers_transform():
    torch.manual_seed(0)
    src = torch.randn(3, 42, 3)
    q, _ = torch.linalg.qr(torch.randn(3, 3, 3))
    q = q * torch.sign(torch.linalg.det(q))[:, None, None]
    t = torch.randn(3, 3)
    r, tt = estimate_rigid_transform(src, torch.einsum("bij,bkj->bki", q, src) + t[:, None])
    assert torch.allclose(r, q, atol=1e-5) and torch.allclose(tt, t, atol=1e-5)


def _batch(sigma_action=0.3, sigma_fk=0.4, seed=0, consistent=False):
    """Two real windows -> (out_net, data_batch_packed, gen_data_noised); perfect-velocity predictions (leaf, grad on)."""
    names = pc.load_action_stats(STATS)[0]
    _, offset, scale = pc.load_action_stats(STATS)
    files = sorted(CACHE.glob("episode_*_0000*.pt"))
    picks = [files[3], files[len(files) // 2]]
    samples, actions = [], []
    for f in picks:
        d = torch.load(f, weights_only=False)
        assert d["joint_names"] == names
        samples.append(d["fk"])
        actions.append((d["action"] - offset) / scale)
    fk_data = build_fk_batch(samples, batch_size=2)
    g = torch.Generator().manual_seed(seed)
    x0 = torch.zeros(2, 33, 64)
    x0[..., :52] = torch.stack(actions)
    if consistent:  # make FK labels exactly what FK(action) predicts -> loss must vanish for perfect predictions
        fk, off, sc = pc._resources("cpu")
        d_act = pc.keypoints_displacement(x0[..., :52] * sc + off, fk_data.inputs["anchor_xyz"].reshape(2, 42, 3), fk, 1.0)
        fk_data.displacement = d_act.transpose(0, 1).reshape(32, 84, 3).contiguous()
    eps = torch.randn(2, 33, 64, generator=g)
    sig = torch.full((2, 33, 1), sigma_action)
    sig[:, 0] = 0
    vt = eps - x0
    xt = sig * eps + (1 - sig) * x0
    preds_action = [vt[i].clone().requires_grad_(True) for i in range(2)]
    fk_state = fk_add_noise(fk_data, torch.full((2,), sigma_fk), scale=SCALE, generator=g)
    preds_fk = fk_state.velocity_target.clone().requires_grad_(True)
    noised = SimpleNamespace(
        epsilon_action=list(eps), vt_target_action=list(vt), sigmas_action=list(sig), xt_tokens_action=list(xt), fk=fk_state
    )
    packed = SimpleNamespace(fk_data=fk_data, action=SimpleNamespace(), pointflow_data=None)
    return {"preds_action": preds_action, "preds_fk": preds_fk}, packed, noised


@needs_data
@pytest.mark.parametrize("mode", ["task", "gt", "pred_to_action", "task+pred", "both"])
def test_gradients_and_modes(mode):
    out, packed, noised = _batch()
    # Perturb the action prediction (at GT the task loss is exactly 0, as is its Charbonnier gradient).
    out["preds_action"] = [(p.detach() + 0.1 * torch.randn_like(p)).requires_grad_(True) for p in out["preds_action"]]
    loss, metrics = pc.compute_consistency_losses(out, packed, noised, fk_scale=SCALE, mode=mode)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    ga = [p.grad for p in out["preds_action"]]
    gf = out["preds_fk"].grad
    if mode in ("task", "gt", "both", "task+pred"):
        assert all(g is not None and g.abs().sum() > 0 for g in ga)
        assert all(g[0].abs().sum() == 0 for g in ga)  # conditioning row (sigma 0) never gets gradient
        assert all(g[:, 52:].abs().sum() == 0 for g in ga)  # padded dims untouched
    else:
        assert all(g is None or g.abs().sum() == 0 for g in ga)  # stop-gradient on the action side
    if mode in ("pred_to_action", "both", "task+pred"):
        assert gf is not None and gf.abs().sum() > 0
    else:
        assert gf is None or gf.abs().sum() == 0
    assert "consistency_loss" in metrics


@needs_data
def test_consistent_labels_give_zero_loss_and_sigma_weight():
    out, packed, noised = _batch(consistent=True)
    loss, _ = pc.compute_consistency_losses(out, packed, noised, fk_scale=1.0, mode="gt", norm="mse")
    assert loss.item() < 1e-8
    # (1-sigma)^2 weighting: a fixed wrong prediction costs less at higher sigma
    vals = []
    for s in (0.1, 0.6):
        out, packed, noised = _batch(sigma_action=s)
        out["preds_action"] = [p.detach() + 0.05 for p in out["preds_action"]]
        loss, _ = pc.compute_consistency_losses(out, packed, noised, fk_scale=SCALE, mode="gt", norm="mse")
        vals.append(loss.item())
    assert vals[0] != vals[1]


@needs_data
def test_task_mode_zero_at_gt_and_nonzero_grad_otherwise():
    out, packed, noised = _batch()
    loss, _ = pc.compute_consistency_losses(out, packed, noised, fk_scale=SCALE, mode="task", norm="mse")
    assert loss.item() < 1e-10  # perfect velocity -> predicted action == GT action (no commanded/realised floor)
    out["preds_action"] = [p.detach() + 0.1 * torch.randn_like(p) for p in out["preds_action"]]
    for p in out["preds_action"]:
        p.requires_grad_(True)
    loss, _ = pc.compute_consistency_losses(out, packed, noised, fk_scale=SCALE, mode="task", norm="mse")
    loss.backward()
    assert loss.item() > 1e-6 and all(p.grad.abs().sum() > 0 for p in out["preds_action"])
    assert pc.consistency_mode() == "task"


@needs_data
def test_perfect_velocity_recovers_exact_x0():
    out, packed, noised = _batch(sigma_action=0.7)
    x0_hat, _ = pc.action_x0_hat(out, noised, 0)
    ref = noised.epsilon_action[0] - noised.vt_target_action[0]
    assert torch.allclose(x0_hat, ref, atol=1e-6)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("POINTFK_CONSISTENCY_WEIGHT", raising=False)
    assert pc.consistency_weight() == 0.0
    monkeypatch.setenv("POINTFK_CONSISTENCY_WEIGHT", "-1")
    with pytest.raises(ValueError):
        pc.consistency_weight()


def _stub_model():
    rf = SimpleNamespace(
        fk_displacement_scale=SCALE, fk_loss_weight=1.0, fk_loss_objective="mse", fk_charbonnier_eps=0.01,
        normalize_loss_by_active=False, pointflow_displacement_scale=1.0,
    )
    config = SimpleNamespace(rectified_flow_training_config=rf, vision_gen=False, action_gen=False, sound_gen=False)
    return SimpleNamespace(config=config, tensor_kwargs_fp32=dict(device="cpu", dtype=torch.float32))


@needs_data
def test_compute_losses_integration_and_noop(monkeypatch):
    """Real OmniMoTModel._compute_losses: unset/0 -> no consistency keys and identical total; >0 -> adds weight*loss."""
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    monkeypatch.setenv("POINTFK_CONSISTENCY_MODE", "gt")  # default "task" is exactly 0 at perfect predictions
    results = {}
    for weight in (None, "0", "0.5"):
        if weight is None:
            monkeypatch.delenv("POINTFK_CONSISTENCY_WEIGHT", raising=False)
        else:
            monkeypatch.setenv("POINTFK_CONSISTENCY_WEIGHT", weight)
        out, packed, noised = _batch()
        total, losses = OmniMoTModel._compute_losses(_stub_model(), out, packed, noised, torch.zeros(2, 1), False)
        results[weight] = (float(total.detach()), sorted(losses))
    assert results[None] == results["0"]
    assert not any(k.startswith("consistency") for k in results[None][1])
    assert any(k.startswith("consistency_loss") for k in results["0.5"][1])
    assert results["0.5"][0] > results[None][0]
