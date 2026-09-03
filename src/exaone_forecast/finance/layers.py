# Layers of the EXAONE Finance attention-free encoder.
# Temporal mixing: 1D causal convolution
# Variate mixing:  group-aware pooling + MLP
# Feed-forward:    gate-free MLP, T5 layer naming

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .config import EXAONEFinanceConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(RMSNorm)


class MLP(nn.Module):
    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        assert not config.is_gated_act
        self.mlp: nn.Module = MLP(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class TemporalCNNBlock(nn.Module):
    """Temporal mixing via 1D causal convolution.

    Applies stacked 1D convolutions along the time axis with causal padding
    (left-padding to preserve sequence length). Final 1x1 conv projects
    back to d_model.

    Shape: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    """

    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        d_model = config.d_model
        channels = config.cnn_channels
        kernel_size = config.cnn_kernel_size
        num_conv = config.cnn_num_conv_layers

        self.layer_norm = RMSNorm(d_model, eps=config.layer_norm_epsilon)

        convs = []
        for i in range(num_conv):
            in_c = d_model if i == 0 else channels
            # Causal padding: pad left only
            convs.append(nn.Conv1d(in_c, channels, kernel_size, padding=0))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(config.dropout_rate))
        self.conv_layers = nn.ModuleList(convs)

        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.proj = nn.Conv1d(channels, d_model, 1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        # Conv1d expects [B, C, T]
        x = normed.transpose(1, 2)  # [B, D, T]

        idx = 0
        for i in range(self.num_conv):
            # Causal left-padding
            pad_len = self.kernel_size - 1
            x = nn.functional.pad(x, (pad_len, 0))
            x = self.conv_layers[idx](x)     # Conv1d
            x = self.conv_layers[idx + 1](x)  # GELU
            x = self.conv_layers[idx + 2](x)  # Dropout
            idx += 3

        x = self.proj(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        return hidden_states + self.dropout(x)


class VariateMLPBlock(nn.Module):
    """Variate mixing via group-aware mean-pooling + MLP."""

    def __init__(self, config: EXAONEFinanceConfig):
        super().__init__()
        d_model = config.d_model
        mlp_dim = config.mlp_variate_dim

        self.layer_norm = RMSNorm(d_model, eps=config.layer_norm_epsilon)
        self.fc1 = nn.Linear(d_model * 2, mlp_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, group_ids: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        group_mask = (group_ids[:, None] == group_ids[None, :]).float()
        group_counts = group_mask.sum(dim=1, keepdim=True).clamp(min=1)
        group_mean = torch.einsum("ij, jtd -> itd", group_mask, normed) / group_counts.unsqueeze(-1)
        combined = torch.cat([normed, group_mean], dim=-1)
        out = self.fc2(self.act(self.fc1(combined)))
        return hidden_states + self.dropout(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, act_fn_name, dropout_p=0.0, use_layer_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = RMSNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res
        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
