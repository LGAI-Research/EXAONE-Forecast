# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the Chronos-2 model code of the Amazon Chronos
# project (https://github.com/amazon-science/chronos-forecasting), specifically
# `src/chronos/chronos2/layers.py` and `src/chronos/chronos2/model.py`.
#
# Modifications by LG AI Research:
#   - split the upstream single-module layer file into named layer classes
#   - added query-aware scalable softmax (SSMax) attention scaling
#   - restricted cross-series attention to the final encoder block
#   - made every linear projection replaceable by a demand adapter
#   - rewrote the docstrings
#
# See Notice.md at the repository root for the full third-party notice.
"""Encoder layers of EXAONE Demand.

The stack is a bidirectional transformer over patches of a single series, with
one departure: the final block also attends *across* series in the batch, so a
group of related series can inform one another. Everything here is frozen at
inference; what adapts to demand lives in :mod:`.adapter`.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

__all__ = [
    "RMSNorm",
    "RoPE",
    "rotate_half",
    "apply_rotary_pos_emb",
    "Patch",
    "InstanceNorm",
    "MHA",
    "SwiGLU",
    "Embedding",
    "Block",
    "Encoder",
]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    Scales by the RMS of the features without subtracting their mean, which is
    what makes it cheaper than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x).type_as(x) * self.weight


class RoPE(nn.Module):
    """Rotary position embedding: the frequency table, not the rotation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.type_as(self.inv_freq)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]``."""
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``pos.shape[-1]`` features of ``t`` by ``pos``."""
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]

    # Fewer tokens than precomputed positions: keep the rightmost ones.
    pos = pos[..., -seq_len:, :]
    pos = pos.type_as(t)

    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim=-1)


class Patch(nn.Module):
    """Cut a series into fixed-size patches, left-padding when it does not divide.

    Padding goes on the left so that the most recent observations always land on
    a patch boundary.
    """

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.shape[-1]

        if seq_length % self.patch_size != 0:
            pad_size = (*x.shape[:-1], self.patch_size - (seq_length % self.patch_size))
            pad = torch.full(size=pad_size, fill_value=0).type_as(x)
            x = torch.concat((pad, x), dim=-1)

        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


class InstanceNorm(nn.Module):
    """Per-series standardisation followed by ``arcsinh``.

    Demand series differ in units by orders of magnitude, and the ``arcsinh``
    keeps a spike from dominating the scale without clipping it away: it is
    approximately the identity near zero and logarithmic in the tails.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # nanmean skips missing entries; an all-missing series falls back to
        # loc 0 / scale 1 rather than producing NaN.
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = torch.arcsinh((x - loc) / scale)
        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        return (torch.sinh(x) * scale + loc).to(orig_dtype)


class QASSMaxMLP(nn.Module):
    """Query-aware scalable softmax.

    ``q_scaled = q * base(log n) * (1 + tanh(query(q)))``. The first factor lets
    attention temperature depend on how many tokens there are, which matters at
    a context of 8,192; the second lets a single query sharpen or flatten its
    own distribution.
    """

    def __init__(self, num_heads: int, head_dim: int, num_hidden: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.base_mlp = nn.Sequential(nn.Linear(1, num_hidden), nn.GELU(), nn.Linear(num_hidden, num_heads))
        self.query_mlp = nn.Sequential(nn.Linear(head_dim, num_hidden), nn.GELU(), nn.Linear(num_hidden, 1))

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        logn = torch.tensor(math.log(max(n, 1))).type_as(q).reshape(1, 1)
        base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, 1)
        modulation = 1 + torch.tanh(self.query_mlp(q))
        return q * base_scales * modulation


class MHA(nn.Module):
    """Multi-head attention with RoPE and optional SSMax scaling.

    Attention is bidirectional: the encoder sees the whole context window at
    once, and the forecast is produced by dedicated future tokens rather than by
    rolling the sequence forward.
    """

    def __init__(
        self,
        query_dim: int,
        inner_dim: int | None = None,
        context_dim: int | None = None,
        num_heads: int = 12,
        is_causal: bool = False,
        dropout: float = 0.0,
        ssmax: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(query_dim)
        self.norm2 = RMSNorm(context_dim) if context_dim is not None else self.norm1

        inner_dim = inner_dim if inner_dim is not None else query_dim
        context_dim = context_dim if context_dim is not None else query_dim

        if inner_dim % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")

        self.rotary_emb = RoPE(dim=inner_dim // num_heads)

        self.num_heads = num_heads
        self.is_causal = is_causal
        self.dropout = dropout

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

        self.ssmax = QASSMaxMLP(num_heads, inner_dim // num_heads) if ssmax else None
        self.resid_drop = nn.Dropout(dropout, inplace=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotary_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert not (self.is_causal and mask is not None), "both attn_mask and is_causal are set"

        q = self.to_q(self.norm1(q))
        k = self.to_k(self.norm2(k))
        v = self.to_v(self.norm2(v))

        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v))

        # RoPE goes on queries and keys only; values carry content, not position.
        if rotary_pos is not None:
            rotary_pos = self.rotary_emb(rotary_pos)
            q = apply_rotary_pos_emb(rotary_pos, q)
            k = apply_rotary_pos_emb(rotary_pos, k)

        if self.ssmax is not None:
            q = self.ssmax(q, n=k.shape[2])

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=self.is_causal,
            dropout_p=(self.dropout if self.training else 0),
        )

        x = rearrange(x, "b h s d -> b s (h d)")
        return self.resid_drop(self.to_out(x))


class SwiGLU(nn.Module):
    """Gated feed-forward block. ``w3(x * SiLU(gate))`` with ``[x, gate] = w12(norm(x))``."""

    def __init__(self, in_feat: int, hidden_feat: int) -> None:
        super().__init__()
        self.norm = RMSNorm(in_feat)
        self.w12 = nn.Linear(in_feat, 2 * hidden_feat)
        self.w3 = nn.Linear(hidden_feat, in_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w12(self.norm(x))
        x, gate = x.chunk(2, dim=-1)
        return self.w3(x * F.silu(gate))


class Embedding(nn.Module):
    """Residual MLP used for both the input and the output patch projection."""

    def __init__(self, in_feat: int, hidden_feat: int, out_feat: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.norm = RMSNorm(out_feat)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_feat, hidden_feat)
        self.linear2 = nn.Linear(hidden_feat, out_feat)
        self.linear3 = nn.Linear(in_feat, out_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.linear2(self.act(self.linear1(x))))
        return self.norm(self.linear3(x) + h)


class Block(nn.Module):
    """One encoder block: temporal attention, optional cross-series attention, SwiGLU.

    ``group_attn`` is set only on the final block. Cross-series attention is what
    lets a group of related series inform one another, and one block of it at
    the top was enough; putting it in every block cost memory without helping.
    """

    def __init__(
        self,
        dim_model: int = 768,
        num_heads: int = 12,
        multiplier: float = 2.667,
        is_causal: bool = False,
        dropout: float = 0.0,
        ssmax: bool = False,
        group_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        hidden_dim = int(multiplier * dim_model)
        self.group_attn = group_attn

        layers: list[nn.Module] = [
            MHA(
                query_dim=dim_model,
                inner_dim=dim_model,
                num_heads=num_heads,
                is_causal=is_causal,
                dropout=dropout,
                ssmax=ssmax,
            )
        ]

        if group_attn:
            layers.append(
                MHA(
                    query_dim=dim_model,
                    inner_dim=dim_model,
                    num_heads=num_heads,
                    is_causal=is_causal,
                    dropout=dropout,
                )
            )

        layers.append(SwiGLU(in_feat=dim_model, hidden_feat=hidden_dim))
        self.block = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        group_time_mask: torch.Tensor | None = None,
        rotary_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.block[0](x, x, x, mask=attn_mask, rotary_pos=rotary_pos) + x

        if self.group_attn:
            # Transposing to (seq, batch, dim) makes attention run across the
            # batch, which is where the other series of the group sit.
            x = rearrange(x, "b s d -> s b d")
            x = self.block[1](x, x, x, mask=group_time_mask) + x
            x = rearrange(x, "s b d -> b s d")

        return self.block[-1](x) + x


class Encoder(nn.Module):
    """The block stack, plus the two masks the blocks need."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.norm = RMSNorm(config.model.dim_model)

        num_layers = config.model.num_layers
        self.block = nn.ModuleList(
            [
                Block(**config.model, is_causal=False, group_attn=(i == num_layers - 1))
                for i in range(num_layers)
            ]
        )

    @staticmethod
    def _expand_time_attn_mask(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``(batch, seq)`` of 1/0 to ``(batch, 1, 1, seq)`` of 0/-inf."""
        assert attn_mask.ndim == 2, "attn_mask must have shape (batch, seq_len)"
        attn_mask = attn_mask[:, None, None, :]
        return (1 - attn_mask) * torch.finfo(dtype).min

    @staticmethod
    def _construct_group_time_mask(
        group_ids: torch.Tensor, attn_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Let a series attend only to series carrying the same group id.

        Series batched together are independent unless the caller says
        otherwise, so this is what keeps one forecast out of another's way.
        """
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attn_mask)
        group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
        return (1 - group_time_mask) * torch.finfo(dtype).min

    def forward(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length = x.shape[:-1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=x.device)

        if attn_mask is None:
            attn_mask = torch.ones(batch_size, seq_length).type_as(x)

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=x.device)

        ext_attn_mask = Encoder._expand_time_attn_mask(attn_mask, x.dtype)
        group_time_mask = Encoder._construct_group_time_mask(group_ids, attn_mask, x.dtype)

        for block in self.block:
            x = block(
                x,
                attn_mask=ext_attn_mask,
                group_time_mask=group_time_mask,
                rotary_pos=position_ids,
            )

        return self.norm(x)
