# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the Chronos-2 model code of the Amazon Chronos
# project (https://github.com/amazon-science/chronos-forecasting), specifically
# `src/chronos/chronos2/model.py`, `pipeline.py` and `dataset.py`.
#
# Modifications by LG AI Research:
#   - attached the demand adapter of `adapter.py` to every linear projection
#   - restricted cross-series attention to the final encoder block
#   - removed the training path (loss, optimiser, data pipeline)
#   - rewrote the docstrings
#
# See Notice.md at the repository root for the full third-party notice.
"""EXAONE Demand — the model itself.

A context window goes in, quantile forecasts come out. Along the way the window
also reaches the adapter's router, which decides how much of each demand-class
branch to apply; see :mod:`.adapter`.
"""
from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

from . import adapter as _adapter
from .config import EXAONEDemandConfig
from .layers import Embedding, Encoder, InstanceNorm, Patch

__all__ = ["EXAONEDemand"]


class EXAONEDemand(nn.Module):
    """Quantile forecaster for demand series."""

    config_class = EXAONEDemandConfig

    def __init__(self, config: EXAONEDemandConfig) -> None:
        super().__init__()
        self.config = config

        self.time_encoding_scale = config.data.context_length

        if config.data.input_patch_size != config.data.output_patch_size:
            raise ValueError(
                "input_patch_size and output_patch_size must be equal, but found "
                f"{config.data.input_patch_size} and {config.data.output_patch_size}"
            )

        # [REG] token, appended between the context and the future tokens.
        self.vocab_embed = nn.Embedding(config.data.num_reg_tokens, config.model.dim_model)

        self.patch = Patch(
            patch_size=config.data.input_patch_size,
            patch_stride=config.data.input_patch_stride,
        )

        # Each patch enters as [time encoding | values | observed mask].
        self.input_embed = Embedding(
            in_feat=config.data.input_patch_size * 3,
            hidden_feat=config.model.dim_embed,
            out_feat=config.model.dim_model,
            dropout=config.model.dropout,
        )

        self.norm = InstanceNorm()
        self.encoder = Encoder(config)

        quantiles = torch.tensor(config.data.quantiles)
        self.num_quantiles = len(quantiles)
        self.register_buffer("quantiles", quantiles)

        self.output_embed = Embedding(
            in_feat=config.model.dim_model,
            hidden_feat=config.model.dim_embed,
            out_feat=config.data.output_patch_size * self.num_quantiles,
            dropout=config.model.dropout,
        )

        if config.adapter.enabled:
            _adapter.attach_adapter(self, config)

    # -- input preparation --------------------------------------------------
    def _get_time_encoding(
        self, batch_size: int, length: int, num_patches: int, mode: str = "context"
    ) -> torch.Tensor:
        """Per-timestep position index, negative for the past and positive for the future."""
        if mode == "context":
            time_enc = torch.arange(start=-length, end=0, device=self.device, dtype=torch.float32)
        elif mode == "future":
            time_enc = torch.arange(start=0, end=length, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid mode: {mode!r}. Expected 'context' or 'future'.")

        patch_size = self.config.data.output_patch_size

        return (
            repeat(time_enc, "(n p) -> b n p", b=batch_size, n=num_patches, p=patch_size)
            .div(int(self.time_encoding_scale))
            .to(self.dtype)
        )

    @staticmethod
    def _generate_mask(x: torch.Tensor, x_mask: torch.Tensor | None) -> torch.Tensor:
        """Observed mask, inferred from NaNs when not supplied."""
        if x_mask is not None:
            return x_mask.type_as(x)
        return torch.isnan(x).logical_not().type_as(x)

    def _prepare_patched_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalise, patch, and mask the context window."""
        batch_size, context_length = context.shape

        context_mask = self._generate_mask(context, context_mask)

        # Keep the most recent observations when the window is longer than the model reads.
        if context_length > self.config.data.context_length:
            context = context[..., -self.config.data.context_length :]
            context_mask = context_mask[..., -self.config.data.context_length :]

        context, loc_scale = self.norm(context)

        patched_context = self.patch(context)
        patched_context_mask = torch.nan_to_num(self.patch(context_mask), nan=0)
        patched_context = torch.where(patched_context_mask > 0, patched_context, 0)

        # A patch is attended to if it holds at least one observation, which
        # keeps a partly observed boundary patch usable.
        attn_mask = patched_context_mask.sum(dim=-1) > 0
        num_context_patches = attn_mask.shape[-1]

        context_time_enc = self._get_time_encoding(
            batch_size,
            num_context_patches * self.config.data.output_patch_size,
            num_context_patches,
            mode="context",
        )

        patched_context = torch.cat([context_time_enc, patched_context, patched_context_mask], dim=-1)
        return patched_context, attn_mask, loc_scale

    def _prepare_patched_future(
        self,
        batch_size: int,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the future tokens the forecast is read off.

        ``future`` carries known-future covariates when the caller has them. The
        released checkpoint was trained without any, so leaving it ``None`` —
        which fills the value and mask channels with zeros — is the supported path.
        """
        output_patch_size = self.config.data.output_patch_size
        final_future_length = num_output_patches * output_patch_size

        if future is not None:
            future_mask = self._generate_mask(future, future_mask)
            future = torch.where(future_mask > 0, future, 0)

            if torch.isnan(future).any():
                raise ValueError(
                    "future covariates contain NaN at positions the mask marks as observed"
                )

            future, _ = self.norm(future, loc_scale)

            if final_future_length > future.shape[-1]:
                pad_size = (*future.shape[:-1], final_future_length - future.shape[-1])
                future = torch.cat([future, torch.zeros(pad_size).type_as(future)], dim=-1)
                future_mask = torch.cat([future_mask, torch.zeros(pad_size).type_as(future_mask)], dim=-1)
            elif future.shape[-1] > final_future_length:
                future = future[..., :final_future_length]
                future_mask = future_mask[..., :final_future_length]

            patched_future = rearrange(future, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size)
            patched_future_mask = rearrange(
                future_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
        else:
            shape = (batch_size, num_output_patches, output_patch_size)
            patched_future = torch.zeros(shape, device=self.device, dtype=self.dtype)
            patched_future_mask = torch.zeros(shape, device=self.device, dtype=self.dtype)

        future_time_enc = self._get_time_encoding(
            batch_size, final_future_length, num_output_patches, mode="future"
        )

        patched_future = torch.cat([future_time_enc, patched_future, patched_future_mask], dim=-1)
        attn_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)

        return patched_future, attn_mask

    # -- forward ------------------------------------------------------------
    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int]:
        """Run the encoder over ``[context patches, REG, future tokens]``."""
        batch_size = context.shape[0]

        patched_context, context_attn_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )

        context_embeds = self.input_embed(patched_context)
        num_context_patches = context_attn_mask.shape[-1]

        num_reg_tokens = self.config.data.num_reg_tokens
        reg_input_ids = torch.arange(num_reg_tokens, device=self.device).expand(batch_size, -1)
        reg_embeds = self.vocab_embed(reg_input_ids)

        context_embeds = torch.cat([context_embeds, reg_embeds], dim=-2)
        context_attn_mask = torch.cat(
            [context_attn_mask, torch.ones_like(reg_input_ids).type_as(context_attn_mask)], dim=-1
        )

        patched_future, future_attn_mask = self._prepare_patched_future(
            batch_size=batch_size,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            future=future_covariates,
            future_mask=future_covariates_mask,
        )

        future_embeds = self.input_embed(patched_future)

        input_embeds = torch.cat([context_embeds, future_embeds], dim=-2)
        attn_mask = torch.cat([context_attn_mask, future_attn_mask], dim=-1)

        # Without explicit group ids every series is its own group, so nothing
        # in the batch attends to anything else.
        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        outputs = self.encoder(
            input_embeds, group_ids=group_ids, attn_mask=attn_mask, position_ids=None
        )

        return outputs, loc_scale, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Quantile forecasts of shape ``(batch, n_quantiles, horizon)``.

        ``horizon`` is ``num_output_patches * output_patch_size``.
        """
        batch_size = context.shape[0]

        # The router cannot see the raw series from inside an nn.Linear, so the
        # statistics are computed once here and left where the adapters read them.
        if self.config.adapter.enabled:
            _adapter.set_batch_statistics(_adapter.demand_stats(context))
        try:
            h, loc_scale, num_context_patches = self.encode(
                context=context,
                context_mask=context_mask,
                group_ids=group_ids,
                future_covariates=future_covariates,
                future_covariates_mask=future_covariates_mask,
                num_output_patches=num_output_patches,
            )
        finally:
            _adapter.clear_batch_statistics()

        expected = num_context_patches + self.config.data.num_reg_tokens + num_output_patches
        assert h.shape == (batch_size, expected, self.config.model.dim_model)

        # The context tokens and the [REG] token have done their work inside the
        # encoder; only the future tokens are read out.
        #
        # The slice is made contiguous on purpose. Feeding a strided view to the
        # projection below lets the CPU GEMM pick a different accumulation order
        # from one process to the next, which moved the last forecast digit
        # around (~1e-6 relative) for no reason. Copying first makes the output
        # reproducible.
        forecast_embeds = h[:, -num_output_patches:].contiguous()

        quantile_preds = self.output_embed(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.config.data.output_patch_size,
        )

        horizon = num_output_patches * self.config.data.output_patch_size
        quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)")
        quantile_preds = self.norm.inverse(quantile_preds, loc_scale)

        return rearrange(quantile_preds, "b (q h) -> b q h", q=self.num_quantiles, h=horizon)

    # -- conveniences -------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(param.device for param in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(param.dtype for param in self.parameters() if param.is_floating_point())
