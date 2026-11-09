"""Ascend attention with explicit packed-sequence and causal-mask semantics.

The BF16/FP16 forward/backward path uses TorchNPU's fusion operator. FP32,
empty segments and differentiable LSE use a chunked PyTorch reference path.
CUDA attention selection and kernels are unaffected.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.model.attention.masks import CausalType


def _segments(query, key, value, cumulative_seqlen_Q=None, cumulative_seqlen_KV=None, seqlens_Q=None, seqlens_KV=None):
    if (
        query.device != key.device
        or query.device != value.device
        or query.dtype != key.dtype
        or query.dtype != value.dtype
    ):
        raise ValueError("Q/K/V must share their device and dtype")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects BSHD tensors")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("incompatible Q/K/V batch or sequence shapes")
    if key.shape[-2] < 1 or query.shape[-2] < 1 or query.shape[-1] != key.shape[-1] or query.shape[-2] % key.shape[-2]:
        raise ValueError("incompatible attention head dimensions")
    if (cumulative_seqlen_Q is None) != (cumulative_seqlen_KV is None):
        raise ValueError("both cumulative offsets must be provided")
    if (seqlens_Q is None) != (seqlens_KV is None):
        raise ValueError("both sequence lengths must be provided")
    for offsets in (cumulative_seqlen_Q, cumulative_seqlen_KV, seqlens_Q, seqlens_KV):
        if offsets is not None and (offsets.ndim != 1 or offsets.dtype not in (torch.int32, torch.int64)):
            raise ValueError("sequence lengths and offsets must be 1-D integer tensors")
    if cumulative_seqlen_Q is not None and seqlens_Q is not None:
        raise ValueError("provide cumulative offsets or sequence lengths, not both")
    if cumulative_seqlen_Q is None and seqlens_Q is not None:
        cumulative_seqlen_Q = torch.nn.functional.pad(seqlens_Q.cumsum(0), (1, 0))
        cumulative_seqlen_KV = torch.nn.functional.pad(seqlens_KV.cumsum(0), (1, 0))
    if cumulative_seqlen_Q is None:
        return [(query[b], key[b], value[b]) for b in range(query.shape[0])], False
    if query.shape[0] != 1:
        raise ValueError("packed attention requires batch dimension 1")
    qoff, koff = cumulative_seqlen_Q.tolist(), cumulative_seqlen_KV.tolist()
    if len(qoff) != len(koff) or len(qoff) < 2 or qoff[0] != 0 or koff[0] != 0:
        raise ValueError("invalid packed offsets")
    if qoff[-1] > query.shape[1] or koff[-1] > key.shape[1]:
        raise ValueError("packed offsets exceed the tensor length")
    if any(b < a for a, b in zip(qoff, qoff[1:])) or any(b < a for a, b in zip(koff, koff[1:])):
        raise ValueError("packed offsets must be nondecreasing")
    return [
        (query[0, qa:qb], key[0, ka:kb], value[0, ka:kb]) for qa, qb, ka, kb in zip(qoff, qoff[1:], koff, koff[1:])
    ], True


def _expand_kv(query, key, value):
    factor = query.shape[-2] // key.shape[-2]
    if factor > 1:
        key = key.repeat_interleave(factor, dim=-2)
        value = value.repeat_interleave(factor, dim=-2)
    return key, value


def _assemble(values, query, packed, *, fill=0):
    if not packed:
        return torch.stack(values)
    output = torch.cat(values)
    # Cosmos may append padding after all real packed tokens. Padding is not
    # an extra sample and must never become a key for a real query.
    missing = query.shape[1] - output.shape[0]
    if missing:
        output = torch.cat([output, output.new_full((missing, *output.shape[1:]), fill)])
    return output.unsqueeze(0)


def _check_causal(segments, is_causal, causal_type):
    if is_causal and not isinstance(causal_type, CausalType):
        raise ValueError("causal_type must be a CausalType for causal attention")
    if is_causal and causal_type == CausalType.DontCare and any(q.shape[0] != k.shape[0] for q, k, _ in segments):
        raise ValueError("DontCare causal masking requires equal Q/K lengths in every segment")


def _reference_segment(q, k, v, scale, is_causal, causal_type, chunk_size):
    k, v = _expand_kv(q, k, v)
    if q.shape[0] == 0 or k.shape[0] == 0:
        anchor = (q.sum() + k.sum() + v.sum()) * 0
        output = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1])) + anchor
        return output, q.new_full(q.shape[:2], float("-inf"), dtype=torch.float32)
    outputs, lses = [], []
    for start in range(0, q.shape[0], chunk_size):
        stop = min(start + chunk_size, q.shape[0])
        positions = torch.arange(start, stop, device=q.device)
        shift = k.shape[0] - q.shape[0] if causal_type == CausalType.BottomRight else 0
        allowed = positions[:, None] + shift >= torch.arange(k.shape[0], device=q.device)[None, :]
        if not is_causal:
            allowed = torch.ones_like(allowed)

        def compute(qc, key, val, mask):
            scores = torch.einsum("qhd,khd->hqk", qc.float(), key.float()) * scale
            scores = scores.masked_fill(~mask[None], float("-inf"))
            valid = mask.any(-1)[None, :, None]
            safe = torch.where(valid, scores, torch.zeros_like(scores))
            weights = safe.softmax(-1) * valid
            out = torch.einsum("hqk,khd->qhd", weights, val.float()).to(qc.dtype)
            lse = torch.where(valid.squeeze(-1), safe.logsumexp(-1), float("-inf")).transpose(0, 1)
            return out, lse

        # Checkpoint chunks so the reference LSE path does not retain every
        # full attention matrix until backward.
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
            out, lse = checkpoint(compute, q[start:stop], k, v, allowed, use_reentrant=False)
        else:
            out, lse = compute(q[start:stop], k, v, allowed)
        outputs.append(out)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses)


def reference_attention(
    query,
    key,
    value,
    *,
    scale=None,
    is_causal=False,
    causal_type=None,
    return_lse=False,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
    seqlens_Q=None,
    seqlens_KV=None,
    chunk_size=128,
    **kwargs,
):
    if is_causal and causal_type is None:
        raise ValueError("causal_type is required for causal attention")
    segments, packed = _segments(query, key, value, cumulative_seqlen_Q, cumulative_seqlen_KV, seqlens_Q, seqlens_KV)
    _check_causal(segments, is_causal, causal_type)
    results = [
        _reference_segment(
            q, k, v, scale if scale is not None else query.shape[-1] ** -0.5, is_causal, causal_type, chunk_size
        )
        for q, k, v in segments
    ]
    if not results:
        raise ValueError("attention requires at least one segment")
    out = _assemble([r[0] for r in results], query, packed)
    lse = _assemble([r[1] for r in results], query, packed, fill=float("-inf"))
    return (out, lse) if return_lse else out


def npu_attention(
    query,
    key,
    value,
    *,
    scale=None,
    is_causal=False,
    causal_type=None,
    return_lse=False,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
    seqlens_Q=None,
    seqlens_KV=None,
    deterministic=False,
    backend_kwargs=None,
    **kwargs,
):
    if backend_kwargs:
        raise ValueError("NPU attention has no backend_kwargs overrides")
    if is_causal and causal_type is None:
        raise ValueError("causal_type is required for causal attention")
    if deterministic:
        # Do not claim a fused kernel's deterministic backward without a
        # backend guarantee; the mathematical reference remains available.
        return reference_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=is_causal,
            causal_type=causal_type,
            return_lse=return_lse,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            seqlens_Q=seqlens_Q,
            seqlens_KV=seqlens_KV,
        )
    segments, packed = _segments(query, key, value, cumulative_seqlen_Q, cumulative_seqlen_KV, seqlens_Q, seqlens_KV)
    _check_causal(segments, is_causal, causal_type)
    if (
        return_lse
        or query.dtype not in {torch.bfloat16, torch.float16}
        or any(q.shape[0] == 0 or k.shape[0] == 0 for q, k, v in segments)
    ):
        return reference_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=is_causal,
            causal_type=causal_type,
            return_lse=return_lse,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            seqlens_Q=seqlens_Q,
            seqlens_KV=seqlens_KV,
        )
    import torch_npu

    scale = scale if scale is not None else query.shape[-1] ** -0.5
    # Full noncausal attention over packed segments is a single TND call.
    if packed and not is_causal:
        qends = cumulative_seqlen_Q[1:].tolist() if cumulative_seqlen_Q is not None else seqlens_Q.cumsum(0).tolist()
        kends = cumulative_seqlen_KV[1:].tolist() if cumulative_seqlen_KV is not None else seqlens_KV.cumsum(0).tolist()
        q = query[0, : qends[-1]]
        k, v = _expand_kv(q, key[0, : kends[-1]], value[0, : kends[-1]])
        out = torch_npu.npu_fusion_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            head_num=query.shape[-2],
            input_layout="TND",
            scale=scale,
            keep_prob=1.0,
            actual_seq_qlen=qends,
            actual_seq_kvlen=kends,
            sparse_mode=0,
        )[0]
        return _assemble([out], query, packed)
    outputs = []
    for q, k, v in segments:
        k, v = _expand_kv(q, k, v)
        mask = None
        if is_causal:
            shift = k.shape[0] - q.shape[0] if causal_type == CausalType.BottomRight else 0
            mask = (
                torch.arange(k.shape[0], device=q.device)[None, :]
                > torch.arange(q.shape[0], device=q.device)[:, None] + shift
            )
            if mask.all(-1).any():
                outputs.append(_reference_segment(q, k, v, scale, is_causal, causal_type, 128)[0])
                continue
        output = torch_npu.npu_fusion_attention(
            q.transpose(0, 1).unsqueeze(0).contiguous(),
            k.transpose(0, 1).unsqueeze(0).contiguous(),
            v.transpose(0, 1).unsqueeze(0).contiguous(),
            head_num=q.shape[-2],
            input_layout="BNSD",
            atten_mask=mask,
            scale=scale,
            keep_prob=1.0,
            sparse_mode=0,
        )[0]
        outputs.append(output[0].transpose(0, 1))
    return _assemble(outputs, query, packed)


def merge_reference(outputs, lse_tensors):
    if len(outputs) != len(lse_tensors) or len(outputs) < 2:
        raise ValueError("merge requires matching output/LSE lists with at least two entries")
    stacked = torch.stack([x.float() for x in lse_tensors])
    valid = torch.isfinite(stacked).any(0)
    # logsumexp(-inf, -inf) has an undefined backward, even when a later
    # where masks its result. Keep that operation finite for empty rows.
    safe_total = torch.where(valid.unsqueeze(0), stacked, torch.zeros_like(stacked)).logsumexp(0)
    total = torch.where(valid, safe_total, float("-inf"))
    safe_total = torch.where(valid, safe_total, torch.zeros_like(safe_total))
    weights = (stacked - safe_total).exp()
    out = sum(x.float() * w.unsqueeze(-1) for x, w in zip(outputs, weights))
    return out.to(outputs[0].dtype), total
