# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The quantile utilities in this file (`_prob_mass_per_quantile`,
# `interpolate_quantiles`, `weighted_quantile`) are derived from the Amazon
# Chronos project (https://github.com/amazon-science/chronos-forecasting),
# specifically `src/chronos/utils.py`.
#
# Modifications by LG AI Research:
#   - rewrote the long-horizon unrolling schedule around them
#   - added the user-facing predict / point / interval surface
#   - rewrote the docstrings
#
# See Notice.md at the repository root for the full third-party notice.
"""User-facing forecasting surface for EXAONE Demand."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from einops import rearrange, repeat

from .model import EXAONEDemand

__all__ = ["EXAONEDemandForecaster"]


# --------------------------------------------------------------------------- #
# Quantile utilities
# --------------------------------------------------------------------------- #
def _prob_mass_per_quantile(quantile_levels: torch.Tensor) -> torch.Tensor:
    """Probability mass each quantile level stands for, by the trapezoidal rule."""
    assert quantile_levels.ndim == 1
    assert quantile_levels.min() > 0 and quantile_levels.max() < 1

    device = quantile_levels.device
    boundaries = torch.cat(
        [torch.tensor([0], device=device), quantile_levels, torch.tensor([1], device=device)]
    )
    prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
    return prob_mass / prob_mass.sum()


def interpolate_quantiles(
    query_quantile_levels: torch.Tensor | list[float],
    original_quantile_levels: torch.Tensor | list[float],
    original_values: torch.Tensor,
) -> torch.Tensor:
    """Read values at ``query_quantile_levels`` off a piecewise-linear quantile function.

    Unlike ``torch.quantile`` the original levels need not be equally spaced, and
    they may vary per row — which is what lets :func:`weighted_quantile` pass an
    empirical CDF in.
    """
    assert torch.is_floating_point(original_values), "`original_values` must be floating point"
    orig_dtype = original_values.dtype

    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(original_quantile_levels, list):
        original_quantile_levels = torch.tensor(original_quantile_levels, dtype=torch.float32)

    assert query_quantile_levels.ndim == 1, "`query_quantile_levels` must be 1-dimensional"
    if original_quantile_levels.ndim > 1:
        assert original_quantile_levels.shape == original_values.shape
    else:
        assert len(original_quantile_levels) == original_values.shape[-1]

    original_quantile_levels = torch.clamp(original_quantile_levels, min=0.0, max=1.0)

    device = original_values.device
    query_quantile_levels = query_quantile_levels.to(device)
    original_quantile_levels = original_quantile_levels.to(device)
    original_values = original_values.to(torch.float32)

    orig_values_shape = original_values.shape
    num_original = original_quantile_levels.shape[-1]
    original_values = original_values.reshape(-1, num_original)
    batch_size = original_values.shape[0]

    if original_quantile_levels.ndim == 1:
        original_quantile_levels = original_quantile_levels.expand(batch_size, -1)
    else:
        original_quantile_levels = original_quantile_levels.reshape(-1, num_original)

    sorted_levels, sorted_indices = torch.sort(original_quantile_levels, dim=-1)
    sorted_values = torch.gather(original_values, dim=-1, index=sorted_indices)

    # Pad with levels 0 and 1 carrying the extreme values, so a query outside the
    # supplied range extrapolates flat rather than off the end.
    zeros = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=device)

    levels_parts, values_parts = [], []
    if original_quantile_levels.min() > 0.0:
        levels_parts.append(zeros)
        values_parts.append(sorted_values[:, :1])
    levels_parts.append(sorted_levels)
    values_parts.append(sorted_values)
    if original_quantile_levels.max() < 1.0:
        levels_parts.append(ones)
        values_parts.append(sorted_values[:, -1:])

    sorted_levels = torch.cat(levels_parts, dim=-1).contiguous()
    sorted_values = torch.cat(values_parts, dim=-1)

    query_expanded = repeat(query_quantile_levels, "q -> b q", b=batch_size).contiguous()

    upper_idx = torch.searchsorted(sorted_levels, query_expanded, right=True)
    upper_idx = torch.clamp(upper_idx, max=sorted_levels.shape[-1] - 1)
    lower_idx = upper_idx - 1

    lower_levels = torch.gather(sorted_levels, dim=1, index=lower_idx)
    upper_levels = torch.gather(sorted_levels, dim=1, index=upper_idx)
    lower_values = torch.gather(sorted_values, dim=1, index=lower_idx)
    upper_values = torch.gather(sorted_values, dim=1, index=upper_idx)

    weight = torch.nan_to_num((query_expanded - lower_levels) / (upper_levels - lower_levels), nan=0.0)
    out = lower_values + weight * (upper_values - lower_values)

    return out.reshape((*orig_values_shape[:-1], len(query_quantile_levels))).to(orig_dtype)


def weighted_quantile(
    query_quantile_levels: torch.Tensor | list[float],
    sample_weights: torch.Tensor | list[float],
    samples: torch.Tensor,
) -> torch.Tensor:
    """Quantiles of a weighted sample set, read off its empirical CDF."""
    assert torch.is_floating_point(samples), "`samples` must be floating point"
    orig_dtype = samples.dtype

    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(sample_weights, list):
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    assert query_quantile_levels.ndim == 1 and sample_weights.ndim == 1
    assert len(sample_weights) == samples.shape[-1]
    assert sample_weights.min() > 0.0, "`sample_weights` must be > 0"

    device = samples.device
    query_quantile_levels = query_quantile_levels.to(device)
    sample_weights = sample_weights.to(device)
    samples = samples.to(torch.float32)

    orig_shape = samples.shape
    num_samples = len(sample_weights)
    samples = samples.reshape(-1, num_samples)
    batch_size = samples.shape[0]

    sample_weights = sample_weights / sample_weights.sum(dim=-1, keepdim=True)
    sample_weights = sample_weights.expand(batch_size, -1).contiguous()

    sorted_samples, sort_indices = torch.sort(samples, dim=-1)
    sorted_weights = torch.gather(sample_weights, dim=-1, index=sort_indices)

    cumul_weights = torch.clamp(torch.cumsum(sorted_weights, dim=-1), min=0.0, max=1.0)

    out = interpolate_quantiles(
        query_quantile_levels=query_quantile_levels,
        original_quantile_levels=cumul_weights,
        original_values=sorted_samples,
    )
    return out.reshape((*orig_shape[:-1], len(query_quantile_levels))).to(orig_dtype)


def _is_ragged(series) -> bool:
    """True for a list of 1-d series that do not all share one length."""
    return (
        isinstance(series, (list, tuple))
        and len(series) > 1
        and all(np.ndim(s) == 1 for s in series)
        and len({len(s) for s in series}) > 1
    )


# --------------------------------------------------------------------------- #
# Forecaster
# --------------------------------------------------------------------------- #
class EXAONEDemandForecaster:
    """Zero-shot probabilistic forecasting with EXAONE Demand.

        fc = from_pretrained(device="cuda:0")
        q  = fc.predict(series, horizon=28)      # (21, 28)

    Series batched into one call are independent of one another unless
    ``group_ids`` says otherwise.
    """

    def __init__(self, model: EXAONEDemand, device: str | torch.device | None = None) -> None:
        self.model = model.eval()
        if device is not None:
            self.model = self.model.to(device)

        self.quantiles: Sequence[float] = list(model.config.data.quantiles)
        self.context_length: int = model.config.data.context_length
        self.max_horizon: int = model.config.max_horizon

    # -- input handling -----------------------------------------------------
    def _as_batch(self, series) -> tuple[torch.Tensor, bool]:
        """``(batch, context_length)`` left-padded with NaN, plus whether input was a single series."""
        if isinstance(series, torch.Tensor):
            arr = series.detach().cpu().to(torch.float32)
        else:
            arr = torch.as_tensor(np.asarray(series, dtype=np.float32))

        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"expected a 1-d or 2-d series, got shape {tuple(arr.shape)}")

        # Keep the most recent observations; pad on the left so every series in
        # the batch ends at the same point in time.
        if arr.shape[-1] > self.context_length:
            arr = arr[..., -self.context_length :]

        return arr, single

    # -- prediction ---------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        series,
        horizon: int,
        group_ids: Sequence[int] | None = None,
        unrolled_quantiles: Sequence[float] | None = None,
    ):
        """Quantile forecasts.

        Args:
            series: one series ``(length,)``, several ``(n_series, length)``, or a
                list of 1-d series of different lengths.
                Missing observations are NaN.
            horizon: number of steps to forecast. Horizons beyond
                :attr:`max_horizon` are produced by unrolling.
            group_ids: optional group label per series. Series sharing a label
                may attend to one another in the final encoder block.
            unrolled_quantiles: quantile levels used as continuation paths when
                unrolling. Must be a subset of :attr:`quantiles`; fewer paths
                make long horizons cheaper and coarser. Defaults to all of them.

        Returns:
            ``(n_quantiles, horizon)`` for a single series, else
            ``(n_series, n_quantiles, horizon)``. NumPy in, NumPy out.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        if _is_ragged(series):
            # Series of different lengths are forecast one length at a time, so each
            # comes out as it would on its own. Left-padding the shorter ones with
            # NaN instead would move them within the context window and change
            # their forecasts.
            if group_ids is not None:
                raise ValueError("group_ids requires series of equal length")
            lengths = [len(s) for s in series]
            out = [None] * len(series)
            for n in sorted(set(lengths)):
                idx = [i for i, m in enumerate(lengths) if m == n]
                batch = np.stack([np.asarray(series[i], dtype=np.float32) for i in idx])
                pred = self.predict(batch, horizon, unrolled_quantiles=unrolled_quantiles)
                for j, i in enumerate(idx):
                    out[i] = pred[j]
            return np.stack(out)

        was_numpy = not isinstance(series, torch.Tensor)
        context, single = self._as_batch(series)
        device = self.model.device
        context = context.to(device)

        if group_ids is None:
            gids = torch.arange(context.shape[0], dtype=torch.long, device=device)
        else:
            gids = torch.as_tensor(list(group_ids), dtype=torch.long, device=device)
            if gids.shape[0] != context.shape[0]:
                raise ValueError("group_ids must have one entry per series")

        levels = list(self.quantiles) if unrolled_quantiles is None else list(unrolled_quantiles)
        if not set(levels).issubset(set(self.quantiles)):
            raise ValueError(
                f"unrolled_quantiles must be a subset of the model's quantiles {self.quantiles}"
            )

        pred = self._predict_batch(context, gids, horizon, torch.tensor(levels))
        pred = pred[0] if single else pred

        return pred.cpu().numpy() if was_numpy else pred.cpu()

    def _num_output_patches(self, remaining: int) -> int:
        patch = self.model.config.data.output_patch_size
        return min(math.ceil(remaining / patch), self.model.config.data.max_output_patches)

    def _predict_batch(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        horizon: int,
        unrolled_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        """Forecast ``horizon`` steps, unrolling when one pass is not enough.

        Beyond :attr:`max_horizon` the model is re-fed its own forecast. Feeding
        back the median alone would make the second segment pretend the first
        was certain, so each quantile level is continued as its own path and the
        paths are recombined by their probability mass.
        """
        device = context.device
        model_quantiles = torch.tensor(self.quantiles)
        remaining = horizon

        pred = self.model(
            context=context,
            group_ids=group_ids,
            num_output_patches=self._num_output_patches(remaining),
        )
        pred_list = [pred]
        remaining -= pred.shape[-1]

        if remaining <= 0:
            return torch.cat(pred_list, dim=-1)[..., :horizon]

        n_paths = len(unrolled_quantiles)
        path_context = repeat(context, "b t -> b q t", q=n_paths)
        # Each path is its own group, so paths never attend across one another.
        path_gids = repeat(group_ids, "b -> b q", q=n_paths)
        path_gids = path_gids * n_paths + torch.arange(n_paths, device=device).unsqueeze(0)

        sample_weights = torch.outer(
            _prob_mass_per_quantile(unrolled_quantiles),
            _prob_mass_per_quantile(model_quantiles),
        )

        while remaining > 0:
            continuation = interpolate_quantiles(
                query_quantile_levels=unrolled_quantiles,
                original_quantile_levels=model_quantiles,
                original_values=rearrange(pred, "b q h -> b h q"),
            )
            continuation = rearrange(continuation, "b h q -> b q h")
            path_context = torch.cat([path_context, continuation], dim=-1)[..., -self.context_length :]

            pred = self.model(
                context=rearrange(path_context, "b n t -> (b n) t"),
                group_ids=rearrange(path_gids, "b n -> (b n)"),
                num_output_patches=self._num_output_patches(remaining),
            )

            # (batch * paths, q, h) -> one quantile forecast per series again.
            pred = weighted_quantile(
                query_quantile_levels=model_quantiles,
                sample_weights=rearrange(sample_weights, "n q -> (n q)"),
                samples=rearrange(pred, "(b n) q h -> b h (n q)", n=n_paths),
            )
            pred = rearrange(pred, "b h q -> b q h")

            pred_list.append(pred)
            remaining -= pred.shape[-1]

        return torch.cat(pred_list, dim=-1)[..., :horizon]

    # -- convenience --------------------------------------------------------
    def point(self, series, horizon: int, **kwargs):
        """Median forecast."""
        q = self.predict(series, horizon, **kwargs)
        idx = self.quantiles.index(0.5)
        return q[..., idx, :]

    def interval(self, series, horizon: int, lower: float = 0.1, upper: float = 0.9, **kwargs):
        """``(low, high)`` bounds at the requested levels."""
        for level in (lower, upper):
            if level not in self.quantiles:
                raise ValueError(f"{level} is not one of the model's quantiles {self.quantiles}")
        q = self.predict(series, horizon, **kwargs)
        return q[..., self.quantiles.index(lower), :], q[..., self.quantiles.index(upper), :]
