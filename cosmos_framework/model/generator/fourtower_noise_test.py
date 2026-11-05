"""CPU checks for fourtower_noise: mixture proportions, uniform / full-noise / clean towers, loss masking."""
import torch

from cosmos_framework.model.generator import fourtower_noise as fn


def test_mix():
    g = torch.Generator().manual_seed(0)
    B = 4000
    kind = torch.randint(0, 10, (B,), generator=g)  # 0 -> FDM, 1 -> IDM, else wam (10% / 10% / 80%)
    fdm, idm = kind == 0, kind == 1
    action_index = torch.arange(B)
    sv = torch.rand(B, 1, generator=g) * 0.5 + 0.25  # the shared draw lives in (0.25, 0.75) so U(0,1) rows are told apart
    junk = torch.full((B, 1), -1.0)  # independent draws are ignored: those rows are re-drawn U(0,1)
    v, a, pf, fk, pf_drop, fk_drop, m = fn.mix_sigmas(idm, fdm, action_index, sv, junk, junk, junk, frac=0.375, generator=g)
    frac = {k: float(x) / B for k, x in m.items()}
    assert abs(frac["fourtower_mix_shared"] - 0.5) < 0.03 and abs(frac["fourtower_mix_indep"] - 0.3) < 0.03, frac
    assert abs(frac["fourtower_mix_fdm"] - 0.1) < 0.02 and abs(frac["fourtower_mix_idm"] - 0.1) < 0.02, frac
    assert torch.equal(fk_drop, fdm) and torch.equal(pf_drop, idm)
    assert (v[fdm] == 1).all() and (pf[fdm] == 1).all() and (fk[fdm] == 0).all()  # FDM: video + flow full noise, MANO clean
    assert (a[idm] == 1).all() and (fk[idm] == 1).all() and (pf[idm] == 0).all()  # IDM: action + MANO full noise, flow clean
    wam = ~fdm & ~idm
    shared = wam & (a[:, 0] == sv[:, 0])
    assert (v[shared] == sv[shared]).all() and (pf[shared] == sv[shared]).all() and (fk[shared] == sv[shared]).all()
    indep = wam & ~shared
    for x in (v, a, pf, fk):  # independent rows: four separate U(0,1) draws covering the whole range
        xi = x[indep, 0]
        assert (xi >= 0).all() and (xi <= 1).all() and xi.min() < 0.1 and xi.max() > 0.9
    assert (v[indep] != pf[indep]).float().mean() > 0.99 and (a[indep] != fk[indep]).float().mean() > 0.99
    # dense action subset: action entries only for some batch slots
    idx = torch.tensor([0, 2])
    _, a2, *_ = fn.mix_sigmas(torch.zeros(3, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), idx,
                              torch.tensor([[0.1], [0.2], [0.3]]), torch.zeros(2, 1), torch.zeros(3, 1), torch.zeros(3, 1), frac=0.0)
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
