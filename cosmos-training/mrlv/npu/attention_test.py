"""Device-independent math and TorchNPU fusion contract tests."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.npu import merge_reference, npu_attention, reference_attention


@pytest.mark.parametrize("causal", [None, CausalType.TopLeft, CausalType.BottomRight])
@pytest.mark.parametrize("q_len,k_len", [(3, 5), (5, 3), (4, 4)])
@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2)])
def test_reference_forward_backward_against_independent_sdpa(causal, q_len, k_len, heads, kv_heads):
    torch.manual_seed(42)
    q = torch.randn(2, q_len, heads, 8, requires_grad=True)
    k = torch.randn(2, k_len, kv_heads, 8, requires_grad=True)
    v = torch.randn(2, k_len, kv_heads, 6, requires_grad=True)
    got = reference_attention(q, k, v, is_causal=causal is not None, causal_type=causal, chunk_size=2)
    mask = None
    if causal is not None:
        shift = k_len - q_len if causal == CausalType.BottomRight else 0
        mask = torch.arange(q_len)[:, None] + shift >= torch.arange(k_len)[None, :]
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(heads // kv_heads, 2).transpose(1, 2),
        v.repeat_interleave(heads // kv_heads, 2).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    a = torch.autograd.grad(got.square().sum(), (q, k, v), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-5)


def test_packed_attention_has_no_cross_sample_leakage():
    q = torch.randn(1, 5, 2, 4, requires_grad=True)
    k = torch.randn(1, 7, 2, 4, requires_grad=True)
    v = torch.randn(1, 7, 2, 4, requires_grad=True)
    args = dict(cumulative_seqlen_Q=torch.tensor([0, 2, 5]), cumulative_seqlen_KV=torch.tensor([0, 4, 7]))
    got = reference_attention(q, k, v, **args)
    expected = torch.cat(
        [reference_attention(q[:, :2], k[:, :4], v[:, :4]), reference_attention(q[:, 2:], k[:, 4:], v[:, 4:])], dim=1
    )
    torch.testing.assert_close(got, expected)
    grad = torch.autograd.grad(got[:, :2].sum(), v)[0]
    assert grad[:, 4:].abs().sum() == 0


def test_empty_key_segment_has_finite_zero_output_and_gradient():
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    k = torch.randn(1, 0, 2, 4, requires_grad=True)
    v = torch.randn(1, 0, 2, 4, requires_grad=True)
    out, lse = reference_attention(q, k, v, return_lse=True)
    assert out.eq(0).all() and lse.isneginf().all()
    out.sum().backward()
    assert q.grad.eq(0).all()


def test_merge_lse_matches_concatenated_keys_and_gradients():
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    k = torch.randn(1, 7, 2, 4, requires_grad=True)
    v = torch.randn(1, 7, 2, 4, requires_grad=True)
    a = reference_attention(q, k[:, :2], v[:, :2], return_lse=True)
    b = reference_attention(q, k[:, 2:], v[:, 2:], return_lse=True)
    got, lse = merge_reference([a[0], b[0]], [a[1], b[1]])
    expected, expected_lse = reference_attention(q, k, v, return_lse=True)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(lse, expected_lse)
    ga = torch.autograd.grad(got.sum(), (q, k, v), retain_graph=True)
    gb = torch.autograd.grad(expected.sum(), (q, k, v))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_fusion_tnd_contract_preserves_segment_boundaries(monkeypatch):
    captured = {}

    def fusion(q, k, v, **kwargs):
        captured.update(kwargs)
        assert q.shape == (5, 4, 8) and k.shape == (7, 4, 8)
        return (torch.zeros_like(q),)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fusion))
    q = torch.randn(1, 5, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 7, 2, 8, dtype=torch.bfloat16)
    v = k.clone()
    out = npu_attention(
        q, k, v, cumulative_seqlen_Q=torch.tensor([0, 2, 5]), cumulative_seqlen_KV=torch.tensor([0, 4, 7])
    )
    assert out.shape == q.shape
    assert captured["input_layout"] == "TND"
    assert captured["actual_seq_qlen"] == [2, 5] and captured["actual_seq_kvlen"] == [4, 7]
    assert captured["sparse_mode"] == 0


def test_fusion_custom_mask_uses_true_for_blocked_positions(monkeypatch):
    captured = {}

    def fusion(q, k, v, **kwargs):
        captured.update(kwargs)
        return (q.clone(),)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fusion))
    q = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
    npu_attention(q, q, q, is_causal=True, causal_type=CausalType.TopLeft)
    assert captured["input_layout"] == "BNSD"
    assert torch.equal(captured["atten_mask"], torch.ones(3, 3, dtype=torch.bool).triu(1))


def test_invalid_offsets_rejected():
    q = torch.randn(1, 3, 2, 4)
    with pytest.raises(ValueError, match="exceed the tensor length"):
        reference_attention(
            q, q, q, cumulative_seqlen_Q=torch.tensor([0, 4]), cumulative_seqlen_KV=torch.tensor([0, 3])
        )


@pytest.mark.parametrize("offsets", [torch.tensor([0.0, 2.0, 4.0]), torch.tensor([[0, 2, 4]])])
def test_rejects_ambiguous_packed_offset_types(offsets):
    tensors = [torch.randn(1, 4, 2, 8) for _ in range(3)]
    with pytest.raises(ValueError, match="1-D integer"):
        reference_attention(*tensors, cumulative_seqlen_Q=offsets, cumulative_seqlen_KV=offsets)


def test_dontcare_mask_rejects_unequal_sequence_lengths():
    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    with pytest.raises(ValueError, match="equal Q/K"):
        reference_attention(q, k, k, is_causal=True, causal_type=CausalType.DontCare)


def test_merge_all_masked_rows_has_finite_backward():
    q = torch.randn(1, 5, 2, 8, requires_grad=True)
    k = torch.randn(1, 2, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    parts = [
        reference_attention(q, k, v, is_causal=True, causal_type=CausalType.BottomRight, return_lse=True)
        for _ in range(2)
    ]
    out, lse = merge_reference([p[0] for p in parts], [p[1] for p in parts])
    out.square().sum().backward()
    assert out[:, :3].count_nonzero() == 0
    assert lse[:, :3].isneginf().all()
    assert all(x.grad.isfinite().all() for x in (q, k, v))


@pytest.mark.parametrize("causal", [False, True])
def test_packed_trailing_padding_is_excluded_from_outputs_and_gradients(causal):
    q = torch.randn(1, 8, 4, 8, requires_grad=True)
    k = torch.randn(1, 9, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = reference_attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=torch.tensor([0, 3, 6]),
        cumulative_seqlen_KV=torch.tensor([0, 4, 7]),
        is_causal=causal,
        causal_type=CausalType.TopLeft,
    )
    expected = torch.cat(
        [
            reference_attention(q[:, :3], k[:, :4], v[:, :4], is_causal=causal, causal_type=CausalType.TopLeft),
            reference_attention(q[:, 3:6], k[:, 4:7], v[:, 4:7], is_causal=causal, causal_type=CausalType.TopLeft),
        ],
        dim=1,
    )
    torch.testing.assert_close(out[:, :6], expected)
    assert out[:, 6:].count_nonzero() == 0
    out.sum().backward()
    assert q.grad[:, 6:].count_nonzero() == k.grad[:, 7:].count_nonzero() == v.grad[:, 7:].count_nonzero() == 0


def test_fusion_strips_packed_trailing_padding(monkeypatch):
    def fusion(q, k, v, **kwargs):
        assert q.shape[0] == 6 and k.shape[0] == 7 and v.shape[0] == 7
        return (torch.ones_like(q),)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fusion))
    q = torch.randn(1, 8, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 9, 2, 8, dtype=torch.bfloat16)
    out = npu_attention(
        q, k, k, cumulative_seqlen_Q=torch.tensor([0, 3, 6]), cumulative_seqlen_KV=torch.tensor([0, 4, 7])
    )
    assert out.shape == q.shape and out[:, 6:].count_nonzero() == 0
