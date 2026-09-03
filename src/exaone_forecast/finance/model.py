# The EXAONE Finance model: an attention-free encoder with a quantile head.
# Temporal mixing: 1D causal convolution
# Variate mixing:  group-aware pooling + MLP

import copy
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .ops import InstanceNorm, Patch

from .config import EXAONEFinanceConfig, ForecastingConfig
from .layers import (
    RMSNorm,
    FeedForward,
    MLP,
    ResidualBlock,
    TemporalCNNBlock,
    VariateMLPBlock,
)


@dataclass
class EncoderBlockOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None


class EncoderBlock(nn.Module):
    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        self.temporal = TemporalCNNBlock(config)
        self.variate = VariateMLPBlock(config)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        group_ids: torch.Tensor,
    ) -> EncoderBlockOutput:
        hidden_states = self.temporal(hidden_states)
        hidden_states = self.variate(hidden_states, group_ids=group_ids)
        hidden_states = self.ffn(hidden_states)
        return EncoderBlockOutput(hidden_states=hidden_states)


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None


class Encoder(nn.Module):
    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        self.block = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = RMSNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
    ) -> EncoderOutput:
        hidden_states = self.dropout(inputs_embeds)

        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                group_ids=group_ids,
            )
            hidden_states = layer_outputs.hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return EncoderOutput(last_hidden_state=hidden_states)


@dataclass
class EXAONEFinanceOutput(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None


class EXAONEFinance(PreTrainedModel):
    config_class = EXAONEFinanceConfig  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True

    def __init__(self, config: EXAONEFinanceConfig):
        assert hasattr(config, "forecasting_config"), "Not a valid EXAONE Finance config"

        super().__init__(config)
        self.config: EXAONEFinanceConfig
        self.model_dim = config.d_model

        config.forecasting_config["time_encoding_scale"] = config.forecasting_config.get(
            "time_encoding_scale", config.forecasting_config["context_length"]
        )
        self.forecasting_config = ForecastingConfig(**config.forecasting_config)

        assert self.forecasting_config.input_patch_size == self.forecasting_config.output_patch_size

        if self.forecasting_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.forecasting_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.input_patch_embedding = ResidualBlock(
            in_dim=self.forecasting_config.input_patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        self.patch = Patch(
            patch_size=self.forecasting_config.input_patch_size,
            patch_stride=self.forecasting_config.input_patch_stride,
        )

        self.instance_norm = InstanceNorm(use_arcsinh=self.forecasting_config.use_arcsinh)

        self.encoder = Encoder(config)

        self.num_quantiles = len(self.forecasting_config.quantiles)
        quantiles = torch.tensor(self.forecasting_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.forecasting_config.output_patch_size,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, RMSNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, MLP):
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, TemporalCNNBlock):
            for conv in module.conv_layers:
                if isinstance(conv, nn.Conv1d):
                    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                    if conv.bias is not None:
                        conv.bias.data.zero_()
            nn.init.kaiming_normal_(module.proj.weight, nonlinearity="linear")
            if module.proj.bias is not None:
                module.proj.bias.data.zero_()
        elif isinstance(module, VariateMLPBlock):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
        elif isinstance(module, EXAONEFinance):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()
            module.residual_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()
            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def _validate_input(self, context, context_mask, group_ids, future_covariates,
                        future_covariates_mask, num_output_patches, future_target, future_target_mask):
        output_patch_size = self.forecasting_config.output_patch_size
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch_size, context_length), found: {tuple(context.shape)}")
        if context_mask is not None and context_mask.shape != context.shape:
            raise ValueError(f"mask must have shape {tuple(context.shape)}, found: {tuple(context_mask.shape)}")
        if future_covariates is not None:
            if future_covariates.shape[0] != context.shape[0] or future_covariates.ndim != 2:
                raise ValueError("future_covariates shape mismatch")
            if future_covariates.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError("num_output_patches too small for future_covariates")
            if future_target is not None and future_target.shape != future_covariates.shape:
                raise ValueError("future_target shape must match future_covariates")
        if future_covariates_mask is not None:
            if future_covariates is None:
                raise ValueError("future_covariates must be provided if future_covariates_mask is provided")
            if future_covariates_mask.shape != future_covariates.shape:
                raise ValueError("future_covariates_mask shape must match future_covariates")
        if group_ids is not None and group_ids.shape != (context.shape[0],):
            raise ValueError(f"group_ids must have shape (batch_size,), found: {tuple(group_ids.shape)}")
        if future_target is not None:
            if future_target.shape[0] != context.shape[0] or future_target.ndim != 2:
                raise ValueError("future_target shape mismatch")
            if future_target.shape[-1] > output_patch_size * num_output_patches:
                raise ValueError("num_output_patches too small for future_target")
        if future_target_mask is not None:
            if future_target is None:
                raise ValueError("future_target must be provided if future_target_mask is provided")
            if future_target_mask.shape != future_target.shape:
                raise ValueError("future_target_mask shape must match future_target")

    def _prepare_patched_context(self, context, context_mask=None):
        context_mask = (
            context_mask.to(context.dtype) if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        batch_size, context_length = context.shape
        if context_length > self.forecasting_config.context_length:
            context = context[..., -self.forecasting_config.context_length:]
            context_mask = context_mask[..., -self.forecasting_config.context_length:]

        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]

        final_context_length = num_context_patches * self.forecasting_config.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=self.device, dtype=torch.float32)
        context_time_enc = (
            repeat(context_time_enc, "(n p) -> b n p", b=batch_size, n=num_context_patches, p=self.forecasting_config.input_patch_size)
            .div(cast(int, self.forecasting_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(self, future_covariates, future_covariates_mask, loc_scale, num_output_patches, batch_size):
        output_patch_size = self.forecasting_config.output_patch_size
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = cast(torch.Tensor, future_covariates)
            future_covariates = future_covariates.to(self.dtype)
            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)
            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)
            if torch.isnan(future_covariates).any():
                raise ValueError("future_covariates contains NaN values at indices not masked")
            if num_output_patches * output_patch_size > future_covariates.shape[-1]:
                padding_shape = (*future_covariates.shape[:-1], num_output_patches * output_patch_size - future_covariates.shape[-1])
                future_covariates = torch.cat([future_covariates, torch.zeros(padding_shape).to(future_covariates)], dim=-1)
                future_covariates_mask = torch.cat([future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)], dim=-1)
            patched_future_covariates = rearrange(future_covariates, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size)
            patched_future_covariates_mask = rearrange(future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size)
        else:
            patched_future_covariates = torch.zeros(batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype)
            patched_future_covariates_mask = torch.zeros(batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype)

        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(start=0, end=final_future_length, device=self.device, dtype=torch.float32)
        future_time_enc = (
            repeat(future_time_enc, "(n p) -> b n p", b=batch_size, n=num_output_patches, p=output_patch_size)
            .div(cast(int, self.forecasting_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_future = torch.cat([future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1)
        return patched_future, patched_future_covariates_mask

    def _compute_loss(self, quantile_preds, future_target, future_target_mask, patched_future_covariates_mask, loc_scale, num_output_patches):
        batch_size = future_target.shape[0]
        output_patch_size = self.forecasting_config.output_patch_size
        assert quantile_preds.shape[0] == batch_size and quantile_preds.shape[-1] >= future_target.shape[-1]

        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1).to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device) if future_target_mask is not None else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (*future_target.shape[:-1], quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat([future_target, torch.zeros(padding_shape).to(future_target)], dim=-1)
            future_target_mask = torch.cat([future_target_mask, torch.zeros(padding_shape).to(future_target_mask)], dim=-1)

        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs((future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles))
        inv_future_covariate_mask = 1 - rearrange(patched_future_covariates_mask, "b n p -> b 1 (n p)", b=batch_size, n=num_output_patches, p=output_patch_size)
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        loss = quantile_loss * loss_mask
        loss = loss.mean(dim=-1).sum(dim=-1).mean()
        return loss

    def encode(self, context, context_mask=None, group_ids=None, future_covariates=None,
               future_covariates_mask=None, num_output_patches=1, future_target=None,
               future_target_mask=None, output_attentions=False):
        self._validate_input(context=context, context_mask=context_mask, future_covariates=future_covariates,
                             future_covariates_mask=future_covariates_mask, group_ids=group_ids,
                             num_output_patches=num_output_patches, future_target=future_target,
                             future_target_mask=future_target_mask)

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context=context, context_mask=context_mask)
        num_context_patches = attention_mask.shape[-1]

        input_embeds = self.input_patch_embedding(patched_context)
        if self.forecasting_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates, future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale, num_output_patches=num_output_patches, batch_size=batch_size,
        )

        future_embeds = self.input_patch_embedding(patched_future)
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds,
            group_ids=group_ids,
        )
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(self, context, context_mask=None, group_ids=None, future_covariates=None,
                future_covariates_mask=None, num_output_patches=1, future_target=None,
                future_target_mask=None, output_attentions=False) -> EXAONEFinanceOutput:
        batch_size = context.shape[0]
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context, context_mask=context_mask, group_ids=group_ids,
            future_covariates=future_covariates, future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches, future_target=future_target,
            future_target_mask=future_target_mask, output_attentions=output_attentions,
        )
        hidden_states = encoder_outputs[0]
        reg_offset = 1 if self.forecasting_config.use_reg_token else 0
        assert hidden_states.shape == (batch_size, num_context_patches + reg_offset + num_output_patches, self.model_dim)

        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(quantile_preds, "b n (q p) -> b q (n p)",
                                   n=num_output_patches, q=self.num_quantiles, p=self.forecasting_config.output_patch_size)

        loss = (
            self._compute_loss(quantile_preds=quantile_preds, future_target=future_target,
                               future_target_mask=future_target_mask, patched_future_covariates_mask=patched_future_covariates_mask,
                               loc_scale=loc_scale, num_output_patches=num_output_patches)
            if future_target is not None else None
        )

        quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)", b=batch_size, q=self.num_quantiles,
                                   h=num_output_patches * self.forecasting_config.output_patch_size)
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(quantile_preds, "b (q h) -> b q h", q=self.num_quantiles,
                                   h=num_output_patches * self.forecasting_config.output_patch_size)

        return EXAONEFinanceOutput(loss=loss, quantile_preds=quantile_preds)
