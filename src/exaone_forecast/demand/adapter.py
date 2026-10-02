"""The demand-aware adapter: what makes this model a demand model.

The backbone is frozen. Everything the model learned about demand lives in five
low-rank branches attached to each of its linear projections, mixed per series:

    y = W0 x + w * s0 * B0 A0 x + sum_e g_e * s_e * B_e A_e x

Branch 0 is shared and always contributes a fixed weight ``w``; it carries the
correction every demand series needs. Branches 1..4 are routed, one per demand
class, and their weights ``g_e`` come from a router that reads eight scale-free
statistics of the raw input window.

The router reads the *raw* series rather than the hidden state on purpose. A
router looking at the hidden state relearns what the backbone already knows, its
gate collapses to uniform, and a uniform mixture of E branches of rank r is
algebraically a single branch of rank E*r — a mixture in name only.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "demand_stats",
    "demand_membership",
    "MoLoRALinear",
    "attach_adapter",
    "DEMAND_CLASSES",
    "NUM_STATS",
]

# Syntetos-Boylan quadrant boundaries, the standard cut points in demand forecasting.
ADI_CUT = 1.32
CV2_CUT = 0.49

#: Order of the routed branches.
DEMAND_CLASSES = ("smooth", "intermittent", "erratic", "lumpy")

#: Number of statistics the router reads.
NUM_STATS = 8

# Softness of the continuous quadrant membership, measured on the log axis.
SOFT_TAU_ADI = 0.35
SOFT_TAU_CV2 = 0.60


# --------------------------------------------------------------------------- #
# Per-batch side channel
#
# The gate needs statistics of the raw series, which an ``nn.Linear`` cannot
# see. The model computes them once per forward pass and leaves them here for
# the adapters to read. The batch dimension is preserved throughout the encoder,
# so a per-series vector broadcasts over tokens directly.
# --------------------------------------------------------------------------- #
_CTX: dict[str, Any] = {"stats": None}


def set_batch_statistics(stats: torch.Tensor | None) -> None:
    """Register the per-series statistics for the batch about to be encoded."""
    _CTX["stats"] = stats


def clear_batch_statistics() -> None:
    _CTX["stats"] = None


def _series_moments(context: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every quantity the router input and the class membership are built from.

    Nothing here depends on the scale of the series. A router that keys on
    "large revenue numbers" routes by data source rather than by behaviour, and
    then fails on the first dataset it has not seen.
    """
    x = torch.as_tensor(context).detach().float()   # NumPy arrays are accepted too
    valid = torch.isfinite(x)
    x = torch.where(valid, x, torch.zeros_like(x))
    n = valid.sum(dim=-1).clamp(min=1).float()

    mean = (x * valid).sum(dim=-1) / n
    var = ((x - mean[:, None]) ** 2 * valid).sum(dim=-1) / n
    std = var.clamp(min=0).sqrt()

    # What counts as "no demand" has to be relative to the size of the series;
    # units differ by orders of magnitude across the corpus.
    scale = x.abs().sum(dim=-1) / n
    eps = (scale * 1e-3).clamp(min=1e-8)
    nz = (x.abs() > eps[:, None]) & valid
    n_nz = nz.sum(dim=-1).clamp(min=1).float()

    zero_frac = 1.0 - n_nz / n

    # Average inter-demand interval: the standard measure of intermittency.
    adi = n / n_nz

    # Squared coefficient of variation over the non-zero values: the standard
    # measure of how erratic the order sizes are.
    nz_mean = (x.abs() * nz).sum(dim=-1) / n_nz
    nz_var = (((x.abs() - nz_mean[:, None]) ** 2) * nz).sum(dim=-1) / n_nz
    cv2 = nz_var / (nz_mean**2).clamp(min=1e-8)

    # Correlation with time: separates a product in launch or decline from one
    # holding a level.
    length = x.shape[-1]
    t = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)[None, :]
    t = t * valid
    t = t - (t.sum(dim=-1) / n)[:, None]
    t = t * valid
    xc = (x - mean[:, None]) * valid
    denom = (t.pow(2).sum(dim=-1).sqrt() * xc.pow(2).sum(dim=-1).sqrt()).clamp(min=1e-8)
    trend = (t * xc).sum(dim=-1) / denom

    # First-order autocorrelation: persistence against noise.
    a, b = xc[:, :-1], xc[:, 1:]
    m = (valid[:, :-1] & valid[:, 1:]).float()
    ac1 = (a * b * m).sum(dim=-1) / (
        (a * m).pow(2).sum(dim=-1).sqrt() * (b * m).pow(2).sum(dim=-1).sqrt()
    ).clamp(min=1e-8)

    # Share of upward steps: describes the shape of the path, not its level.
    d = (b - a) * m
    up = ((d > 0).float() * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)

    cv_all = std / mean.abs().clamp(min=1e-8)

    return {
        "n": n, "zero_frac": zero_frac, "adi": adi, "cv2": cv2,
        "trend": trend, "ac1": ac1, "up": up, "cv_all": cv_all,
    }


def demand_stats(context) -> torch.Tensor:
    """Summarise what kind of demand each series in ``context`` is.

    Args:
        context: ``(batch, length)`` raw series, a tensor or NumPy array.
            Missing entries are NaN.

    Returns:
        ``(batch, NUM_STATS)`` router input, every entry held near zero. The
        order is zero share, ADI, CV^2, trend, AC(1), upward steps, CV, length.
    """
    m = _series_moments(context)

    stats = torch.stack(
        [
            m["zero_frac"],
            torch.log1p(m["adi"]).clamp(max=6.0) / 3.0,
            torch.log1p(m["cv2"]).clamp(max=6.0) / 3.0,
            m["trend"],
            m["ac1"],
            m["up"] * 2.0 - 1.0,
            torch.log1p(m["cv_all"]).clamp(max=6.0) / 3.0,
            torch.log(m["n"]) / 8.0,
        ],
        dim=-1,
    )

    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def demand_membership(context) -> torch.Tensor:
    """Continuous membership over the four demand classes, ``(batch, 4)``.

    Not part of the forward pass: it supervised the router during training, and
    is exposed because it says what the router is being asked to reproduce.
    Measured on the log axis because ADI and CV^2 are both ratios — on a linear
    axis the gap between ADI 1.2 and 1.4 would weigh the same as between 8 and 8.2.
    """
    m = _series_moments(context)
    adi, cv2 = m["adi"], m["cv2"]

    s_adi = torch.sigmoid((torch.log(adi.clamp(min=1e-6)) - math.log(ADI_CUT)) / SOFT_TAU_ADI)
    s_cv = torch.sigmoid((torch.log(cv2.clamp(min=1e-6)) - math.log(CV2_CUT)) / SOFT_TAU_CV2)

    soft = torch.stack(
        [(1 - s_adi) * (1 - s_cv), s_adi * (1 - s_cv), (1 - s_adi) * s_cv, s_adi * s_cv],
        dim=-1,
    )
    soft = torch.nan_to_num(soft, nan=0.25).clamp(min=1e-6)
    return soft / soft.sum(dim=-1, keepdim=True)


class MoLoRALinear(nn.Module):
    """A frozen linear layer with a routed mixture of low-rank branches on top."""

    def __init__(self, base: nn.Linear, spec: Any) -> None:
        super().__init__()

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.e = int(spec.num_experts)
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()

        din, dout = base.in_features, base.out_features

        # Branches carry different ranks. They live in one tensor sized by the
        # largest so that the mixture is two GEMMs rather than a loop; the
        # smaller branches have their unused rank columns masked out.
        ranks = list(spec.ranks)
        if len(ranks) != self.e:
            raise ValueError(f"{len(ranks)} ranks given for {self.e} branches")
        self.ranks = ranks
        rmax = max(ranks)
        self.rmax = rmax

        self.lora_a = nn.Parameter(torch.zeros(self.e, rmax, din))
        self.lora_b = nn.Parameter(torch.zeros(self.e, dout, rmax))

        mask = torch.zeros(self.e, rmax)
        for i, r in enumerate(ranks):
            mask[i, :r] = 1.0
        self.register_buffer("rank_mask", mask, persistent=False)

        # alpha/rank is kept per branch, so branches of different rank arrive at
        # comparable magnitudes.
        scale = torch.tensor([float(spec.alpha) / r for r in ranks])
        self.register_buffer("expert_scale", scale, persistent=False)

        self.gate = nn.Sequential(
            nn.Linear(NUM_STATS, 32), nn.GELU(), nn.Linear(32, self.e, bias=False)
        )

        self.shared_w = float(spec.shared_weight)
        self.temp = float(spec.temperature)

    def _stat_gate(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, 1, E)`` mixing weights from the raw-series statistics.

        Falls back to a uniform mixture when the leading dimension is not the
        series axis. That happens in the final block, whose cross-series
        attention transposes tokens and series; one attention out of twenty-four
        blocks runs unrouted rather than being mapped across by guesswork.
        """
        stats = _CTX["stats"]

        if stats is None or stats.shape[0] != x.shape[0]:
            return x.new_full((x.shape[0], 1, self.e), 1.0 / self.e)

        logits = self.gate(stats.type_as(x)) / self.temp
        return F.softmax(logits.float(), dim=-1).type_as(x)[:, None, :]

    def _branch(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Mixture of the low-rank branches, weighted by ``g``.

        Written as two GEMMs rather than one ``einsum``. Contracting the branch
        axis directly materialises a ``[b, n, E, out]`` intermediate, which for
        the feed-forward projections is hundreds of megabytes per layer.
        """
        b, n, _ = x.shape
        a = (self.lora_a * self.rank_mask[:, :, None]).reshape(self.e * self.rmax, -1)
        xa = F.linear(self.dropout(x), a)                       # [b, n, E*r]

        w = g * self.expert_scale                               # [b, n|1, E]
        xa = (xa.view(b, n, self.e, self.rmax) * w[..., None]).reshape(b, n, -1)

        # Column order of lora_b must match xa: branch major, rank minor.
        bw = (self.lora_b * self.rank_mask[:, None, :]).permute(1, 0, 2).reshape(
            self.lora_b.shape[1], self.e * self.rmax
        )
        return F.linear(xa, bw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)

        if x.dim() != 3:
            return y

        g = self._stat_gate(x)

        # Branch 0 is shared and takes a fixed share of the mixture; the routed
        # branches split what is left. As the router sharpens, a single series
        # would otherwise fall back to the rank of one branch — the shared
        # branch is what returns that capacity.
        q = g[..., 1:]
        q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        g = torch.cat([x.new_full((*q.shape[:-1], 1), self.shared_w), q * (1.0 - self.shared_w)], dim=-1)

        return y + self._branch(x, g)


def attach_adapter(model: nn.Module, config: Any) -> int:
    """Wrap every targeted ``nn.Linear`` in ``model`` with :class:`MoLoRALinear`.

    Returns the number of layers wrapped.
    """
    spec = config.adapter
    targets = set(spec.targets)

    wrapped = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, MoLoRALinear(child, spec))
                wrapped += 1

    return wrapped
