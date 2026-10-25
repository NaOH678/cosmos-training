"""CPU checks for fourtower_noise: mixture proportions, clean towers, loss masking."""
from types import SimpleNamespace as NS

import torch

from cosmos_framework.model.generator import fourtower_noise as fn


def test_mix():
    g = torch.Generator().manual_seed(0)
    B = 4000
    kind = torch.randint(0, 10, (B,), generator=g)  # 0 -> FDM, 1 -> IDM, else wam (10% / 10% / 80%)
    vision_clean = kind == 1
    action_index = torch.arange(B)
    action_clean = kind == 0
    sv, sa, spf, sfk = (torch.rand(B, 1, generator=g) for _ in range(4))
    a, pf, fk, pf_drop, fk_drop, m = fn.mix_sigmas(vision_clean, action_clean, action_index, sv, sa, spf, sfk,
                                                   frac=0.375, generator=g)
    frac = {k: float(v) / B for k, v in m.items()}
    assert abs(frac["fourtower_mix_shared"] - 0.5) < 0.03 and abs(frac["fourtower_mix_indep"] - 0.3) < 0.03, frac
    assert abs(frac["fourtower_mix_fdm"] - 0.1) < 0.02 and abs(frac["fourtower_mix_idm"] - 0.1) < 0.02, frac
    fdm, idm = kind == 0, kind == 1
    assert torch.equal(fk_drop, fdm) and torch.equal(pf_drop, idm)
    assert (fk[fdm] == 0).all() and (pf[fdm] == sv[fdm]).all()  # FDM: MANO clean, flow at the video sigma
    assert (pf[idm] == 0).all() and (fk[idm] == sv[idm]).all() and (a[idm] == sv[idm]).all()  # IDM: action + MANO shared
    shared = (~fdm & ~idm) & (a[:, 0] == sv[:, 0])
    assert (pf[shared] == sv[shared]).all() and (fk[shared] == sv[shared]).all()
    indep = (~fdm & ~idm) & ~shared
    assert indep.sum() > 0 and (pf[indep] == spf[indep]).all() and (fk[indep] == sfk[indep]).all()
    # dense action subset: action entries only for some batch slots
    idx = torch.tensor([0, 2])
    a2, *_ = fn.mix_sigmas(torch.zeros(3, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), idx,
                           torch.tensor([[0.1], [0.2], [0.3]]), torch.tensor([[0.9], [0.8]]), torch.rand(3, 1), torch.rand(3, 1), frac=0.0)
    assert torch.allclose(a2, torch.tensor([[0.1], [0.3]]))
    print("mix ok", frac)


def test_drop_and_rescale():
    from dataclasses import dataclass

    @dataclass
    class Lab:
        valid: torch.Tensor
        inputs: dict

    lab = Lab(valid=torch.ones(3, 5, dtype=torch.bool), inputs={"point_batch": torch.tensor([0, 0, 1, 1, 1])})
    out = fn.drop_samples(lab, torch.tensor([False, True]))
    assert out.valid[:, :2].all() and not out.valid[:, 2:].any() and lab.valid.all()
    assert fn.drop_samples(lab, torch.tensor([False, False])) is lab
    cm = [torch.zeros(4, 1), torch.ones(4, 1), torch.zeros(4, 1)]
    assert float(fn.active_rescale(torch.tensor(2.0), cm)) == 3.0
    print("drop/rescale ok")


if __name__ == "__main__":
    test_mix()
    test_drop_and_rescale()
    print("ALL OK")
