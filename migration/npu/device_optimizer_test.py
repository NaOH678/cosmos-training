"""Device routing, RNG replay and FP32 master-optimizer portability."""

import copy

import pytest
import torch
from litegen import device
from litegen.optim.master_shadow import MasterShadowAdamW

from cosmos_framework.utils.generator.optimizer import _optimizer_cls
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict


def test_explicit_device_precedes_environment(monkeypatch):
    monkeypatch.setenv("LITEGEN_DEVICE", "npu")
    assert device.get_device("cpu") == torch.device("cpu")
    assert device.device_type("cuda:2") == "cuda"
    assert device.statistics_dtype() == torch.float32


def test_cpu_rng_replay_and_wrong_backend_rejection(monkeypatch):
    monkeypatch.setenv("LITEGEN_DEVICE", "cpu")
    state = get_rand_state_dict()
    expected = torch.randn(10)
    set_rand_state_dict(state)
    torch.testing.assert_close(torch.randn(10), expected, rtol=0, atol=0)
    state["torch_cuda"] = state.pop("torch_cpu")
    with pytest.raises(ValueError, match="cross-device"):
        set_rand_state_dict(state)


def test_npu_optimizer_selection_uses_litegen_master_weights(monkeypatch):
    monkeypatch.setenv("LITEGEN_DEVICE", "npu")
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    opt = _optimizer_cls([p], "FusedAdam", lr=0.01, fused=True, weight_decay=0.05)
    assert isinstance(opt, MasterShadowAdamW)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.state[p]["master"].dtype == torch.float32
    assert opt.state[p]["exp_avg"].dtype == torch.float32
    assert opt.state[p]["exp_avg_sq"].dtype == torch.float32


def test_master_optimizer_multistep_matches_fp32_oracle_and_resume():
    torch.manual_seed(9)
    model = torch.nn.Parameter(torch.randn(8).bfloat16())
    oracle = torch.nn.Parameter(model.detach().float().clone())
    config = dict(lr=0.02, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05, foreach=False, fused=False)
    opt = MasterShadowAdamW([model], **config)
    reference = torch.optim.AdamW([oracle], **config)
    for step in range(12):
        gradient = torch.randn_like(model)
        model.grad = gradient
        oracle.grad = gradient.float()
        opt.step()
        reference.step()
        torch.testing.assert_close(opt.state[model]["master"], oracle, rtol=0, atol=0)
        torch.testing.assert_close(model, oracle.bfloat16(), rtol=0, atol=0)
        if step == 5:
            saved = copy.deepcopy(opt.state_dict())
            restored = MasterShadowAdamW([model], **config)
            restored.load_state_dict(saved)
            opt = restored


def test_fp32_sampler_coefficients_match_fp64_near_zero(monkeypatch):
    from cosmos_framework.utils.functional import runge_kutta

    t = torch.tensor([-1.0, -0.1, -0.009, -1e-5, 1e-5, 0.009, 0.1, 1.0])
    expected = [function(t) for function in (runge_kutta.phi1, runge_kutta.phi2)]
    monkeypatch.setattr(runge_kutta.accelerator, "statistics_dtype", lambda *_: torch.float32)
    for function, oracle in zip((runge_kutta.phi1, runge_kutta.phi2), expected):
        torch.testing.assert_close(function(t), oracle, atol=3e-6, rtol=3e-6)
    assert runge_kutta.phi1(torch.tensor(0.0)).item() == 1
    assert runge_kutta.phi2(torch.tensor(0.0)).item() == 0.5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lr", [0.0, 0.02])
@pytest.mark.parametrize("restore_unstarted", [False, True])
def test_pretrained_load_before_first_update_matches_fp32_oracle(dtype, lr, restore_unstarted):
    model = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=dtype))
    config = dict(lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05, foreach=False, fused=False)
    optimizer = MasterShadowAdamW([model], **config)
    if restore_unstarted:
        optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    # Cosmos loads pretrained model weights after constructing its optimizer.
    with torch.no_grad():
        model.copy_(torch.tensor([2.0, -3.0], dtype=dtype))
    oracle = torch.nn.Parameter(model.detach().float().clone())
    reference = torch.optim.AdamW([oracle], **config)
    model.grad = torch.tensor([0.125, -0.25], dtype=dtype)
    oracle.grad = model.grad.float()
    optimizer.step()
    reference.step()
    torch.testing.assert_close(optimizer.state[model]["master"], oracle, rtol=0, atol=0)
    torch.testing.assert_close(model, oracle.to(dtype), rtol=0, atol=0)


def test_prepare_keeps_unused_optimizer_state_lazy():
    from types import SimpleNamespace
    from litegen.optim.prepare import capture_optimizer, prepare_fsdp2_optimizer

    model = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.bfloat16))
    template = torch.optim.AdamW([model], lr=0.0, foreach=False, fused=False)
    template.state[model]  # Accessing an unused state must not fix its master weight.
    optimization = SimpleNamespace(
        optimizer=SimpleNamespace(backend="single_tensor"), parallel=SimpleNamespace(strategy="fsdp2")
    )
    with pytest.warns(UserWarning, match="returned a new optimizer"):
        optimizer = prepare_fsdp2_optimizer(capture_optimizer(template), optimization, True, {}, {id(model)})
    with torch.no_grad():
        model.fill_(2.0)
    model.grad = torch.zeros_like(model)
    optimizer.step()
    assert model.item() == 2.0
