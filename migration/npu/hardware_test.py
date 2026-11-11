"""Real Ascend kernels: skipped on hosts without NPU; skips are not acceptance."""

import pytest
import torch
from litegen import device

from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.npu import npu_attention, reference_attention

pytestmark = pytest.mark.skipif(
    device.device_type() != "npu" or not device.is_available(), reason="requires real Ascend hardware"
)


@pytest.mark.parametrize("causal", [None, CausalType.TopLeft, CausalType.BottomRight])
def test_real_npu_fusion_forward_and_backward(causal, monkeypatch):
    import torch_npu

    calls = []
    original = torch_npu.npu_fusion_attention

    def tracked(*args, **kwargs):
        calls.append(kwargs["input_layout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", tracked)
    torch.manual_seed(42)
    cpu = [
        torch.randn(1, length, heads, 64).bfloat16().float().requires_grad_()
        for length, heads in [(32, 4), (48, 2), (48, 2)]
    ]
    actual = [tensor.detach().to(device.get_device(), dtype=torch.bfloat16).requires_grad_() for tensor in cpu]
    expected = reference_attention(*cpu, is_causal=causal is not None, causal_type=causal)
    result = npu_attention(*actual, is_causal=causal is not None, causal_type=causal)
    expected.float().square().mean().backward()
    result.float().square().mean().backward()
    assert calls, "test did not execute the fused NPU operator"
    for a, b in [(result, expected), *((x.grad, y.grad) for x, y in zip(actual, cpu))]:
        a = a.detach().float().cpu()
        b = b.detach().float().cpu()
        assert torch.isfinite(a).all()
        relative = (a - b).norm() / b.norm().clamp_min(1e-8)
        assert relative < 0.02, relative.item()


def test_real_npu_packed_fusion_no_cross_sample_leakage():
    torch.manual_seed(42)
    cpu = [torch.randn(1, length, heads, 64).bfloat16().float() for length, heads in [(13, 4), (17, 2), (17, 2)]]
    qoff = torch.tensor([0, 5, 13], dtype=torch.int64)
    koff = torch.tensor([0, 7, 17], dtype=torch.int64)
    expected = reference_attention(*cpu, cumulative_seqlen_Q=qoff, cumulative_seqlen_KV=koff)
    actual = [x.to(device.get_device(), dtype=torch.bfloat16).requires_grad_() for x in cpu]
    result = npu_attention(
        *actual, cumulative_seqlen_Q=qoff.to(device.get_device()), cumulative_seqlen_KV=koff.to(device.get_device())
    )
    torch.testing.assert_close(result.float().cpu(), expected, rtol=0.02, atol=0.02)
    result[:, :5].float().sum().backward()
    assert actual[2].grad[:, 7:].abs().sum() == 0


def test_real_npu_differentiable_lse_merge():
    from cosmos_framework.model.attention.npu import merge_reference

    torch.manual_seed(42)
    cpu = [torch.randn(1, length, 2, 64).bfloat16().float().requires_grad_() for length in (7, 13, 13)]
    actual = [x.detach().to(device.get_device(), dtype=torch.bfloat16).requires_grad_() for x in cpu]
    parts = [npu_attention(actual[0], actual[1][:, :cut], actual[2][:, :cut], return_lse=True) for cut in (5, 13)]
    result, lse = merge_reference([p[0] for p in parts], [p[1] for p in parts])
    expected_parts = [reference_attention(cpu[0], cpu[1][:, :cut], cpu[2][:, :cut], return_lse=True) for cut in (5, 13)]
    expected, expected_lse = merge_reference([p[0] for p in expected_parts], [p[1] for p in expected_parts])
    torch.testing.assert_close(result.float().cpu(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.002, atol=0.002)
    result.float().square().mean().backward()
    expected.square().mean().backward()
    for x, y in zip(actual, cpu):
        assert (x.grad.float().cpu() - y.grad).norm() / y.grad.norm().clamp_min(1e-8) < 0.02
