"""Optional per-point transformer refinement between broadcast and the velocity head.

The codec's default decode is broadcast + static local features + a per-point MLP:
points never talk to each other after the cluster stage.  PointDecoder inserts a
small stack of pre-LN transformer blocks that let the points of each sample
attend to one another inside every motion block, so members of one cluster can
re-differentiate with task context instead of only through their static offsets.
The output projection is zero-initialised so the branch starts as a no-op.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PointDecoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def _attention(self, x, point_offsets):
        """Full self-attention within each sample's point segment (ragged batch)."""
        n, d = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.heads, d // self.heads).permute(1, 2, 0, 3)
        outputs = []
        start = 0
        for end in point_offsets:
            if end > start:
                q, k, v = qkv[:, :, start:end].unbind(0)
                outputs.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(end - start, d))
            start = end
        return torch.cat(outputs) if outputs else x.new_zeros((0, d))

    def forward(self, x, point_offsets):
        x = x + self.proj(self._attention(self.norm1(x), point_offsets))
        return x + self.ffn(self.norm2(x))


class PointDecoder(nn.Module):
    """Fused per-point features -> velocity, via per-sample transformer blocks."""

    def __init__(self, fused_dim, anchor_dim, out_dim, *, dim=256, heads=4, blocks=2):
        super().__init__()
        if blocks < 1:
            raise ValueError("PointDecoder requires at least one block")
        self.in_proj = nn.Linear(fused_dim, dim)
        self.anchor_encoder = nn.Sequential(nn.Linear(anchor_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([PointDecoderBlock(dim, heads) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, fused, anchor, point_offsets):
        x = self.in_proj(fused) + self.anchor_encoder(anchor)
        for block in self.blocks:
            x = block(x, point_offsets)
        return self.out(x)
