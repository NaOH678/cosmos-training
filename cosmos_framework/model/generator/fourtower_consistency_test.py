"""CPU checks for fourtower_consistency on synthetic tensors: shapes, gating, zero loss for consistent inputs, gradients."""
import os
from types import SimpleNamespace as NS

import torch

from cosmos_framework.model.generator import fourtower_consistency as fc

torch.manual_seed(0)
B, H, K, N, PER = 2, 32, 42, 64, 4
TIMING = NS(steps=H, steps_per_token=PER)


def make(sigma=0.3, consistent=True):
    fk_anchor = torch.rand(B * K, 3) * 0.2 + torch.tensor([0, 0, 0.6])
    # PF points = jittered copies of FK keypoints (within 5 mm) plus far points
    near = fk_anchor[torch.randint(0, K, (B * N // 2,))].view(B, N // 2, 3) + 0.005 * torch.randn(B, N // 2, 3)
    far = torch.rand(B, N // 2, 3) + torch.tensor([1.0, 1.0, 1.5])
    pf_anchor = torch.cat([near, far], 1).reshape(B * N, 3)
    # per sample, keypoints get one rigid translation trajectory; PF near points copy it when consistent
    traj = torch.cumsum(0.002 * torch.randn(B, H, 1, 3), 1)
    fk_disp = traj.expand(B, H, K, 3).transpose(0, 1).reshape(H, B * K, 3)
    pf_traj = traj if consistent else traj + torch.cumsum(0.004 * torch.randn(B, H, 1, 3), 1)
    pf_disp = pf_traj.expand(B, H, N, 3).transpose(0, 1).reshape(H, B * N, 3)
    fk_scale, pf_scale = 0.0766, 0.0366
    s = torch.full((B,), sigma)
    fk_b, pf_b = torch.arange(B).repeat_interleave(K), torch.arange(B).repeat_interleave(N)
    preds_fk = torch.zeros(H, B * K, 3, requires_grad=True)
    preds_pf = torch.zeros(H, B * N, 3, requires_grad=True)
    fk_state = NS(xt=fk_disp / fk_scale, sigma=s)  # pred 0 -> x0 = xt
    pf_state = NS(xt=pf_disp / pf_scale, sigma=s)
    fk_data = NS(labeled=torch.ones(B, dtype=torch.bool), displacement=fk_disp, valid=torch.ones(H, B * K, dtype=torch.bool),
                 inputs={"anchor_xyz": fk_anchor, "point_batch": fk_b}, timing=TIMING)
    k_px = torch.tensor([[0.8, 0, 0.5], [0, 0.8, 0.5], [0, 0, 1.0]])
    pf_data = NS(labeled=torch.ones(B, dtype=torch.bool), displacement=pf_disp, valid=torch.ones(H, B * N, dtype=torch.bool), timing=TIMING,
                 inputs={"anchor_xyz": pf_anchor, "point_batch": pf_b, "image_size_wh": torch.tensor([[64, 64]] * B),
                         "intrinsics_normalized": k_px.expand(B, 3, 3), "uv_to_video": torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).expand(B, 2, 3),
                         "video_size_wh": torch.tensor([[64, 64]] * B)})
    out = {"preds_fk": preds_fk, "preds_pointflow": preds_pf}
    noised = NS(fk=fk_state, pointflow=pf_state)
    return out, NS(fk_data=fk_data, pointflow_data=pf_data), noised, fk_scale, pf_scale


def test_mf():
    out, data, noised, fs, ps = make(consistent=True)
    act = torch.ones(B, dtype=torch.bool)
    l0, m0 = fc.loss_mf(out, data, noised, fk_scale=fs, pf_scale=ps, active=act, eps=0.01, radius=0.015)
    assert float(l0) < 1e-6 and float(m0["fourtower_mf_keypoints"]) > 0, (float(l0), m0)
    out, data, noised, fs, ps = make(consistent=False)
    l1, _ = fc.loss_mf(out, data, noised, fk_scale=fs, pf_scale=ps, active=act, eps=0.01, radius=0.015)
    l1.backward()
    assert float(l1) > 1e-3
    assert out["preds_fk"].grad.abs().sum() > 0 and out["preds_pointflow"].grad.abs().sum() > 0  # both sides
    l2, _ = fc.loss_mf(*make(consistent=False)[:3], fk_scale=fs, pf_scale=ps, active=torch.zeros(B, dtype=torch.bool), eps=0.01, radius=0.015)
    assert float(l2) == 0.0
    print("mf ok", float(l0), float(l1))


def test_fv():
    out, data, noised, fs, ps = make()
    T = H + 1
    # a static textured video: colour depends on pixel position only; moving points then see colour changes
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    img = torch.stack([xx, yy, xx * yy])[:, None].expand(3, T, 64, 64).clone().requires_grad_(True)
    fc.x0_vision = lambda o, g, i: (o["lat"], torch.tensor(0.3))
    fc.decode_with_grad = lambda model, lat: lat
    out["lat"] = img
    out["preds_vision"] = [img] * B
    act = torch.ones(B, dtype=torch.bool)
    l, m = fc.loss_fv(None, out, data, noised, pf_scale=ps, active=act, eps=0.01, frac=1.0, pool=1)
    l.backward()
    assert float(m["fourtower_fv_tracks"]) > 0 and float(l) > 0
    assert img.grad.abs().sum() > 0 and out["preds_pointflow"].grad.abs().sum() > 0
    print("fv ok", float(l), int(m["fourtower_fv_tracks"]))


if __name__ == "__main__":
    test_mf()
    test_fv()
    print("ALL OK")
