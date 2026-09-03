# Configuration for the EXAONE Finance attention-free encoder.

from dataclasses import dataclass
from typing import List, Literal

from transformers.configuration_utils import PretrainedConfig


class EXAONEFinanceConfig(PretrainedConfig):
    """
    Configuration for :class:`EXAONEFinance`.

    The encoder is attention-free: temporal mixing is a 1D causal convolution
    and variate mixing is a group-aware pooling MLP, so cost grows linearly in
    both sequence length and variate count.

    Extra args over base:
        cnn_channels:         Number of conv channels.
        cnn_kernel_size:      Conv kernel size.
        cnn_num_conv_layers:  Number of stacked conv layers per block.
        mlp_variate_dim:      Hidden dim for variate MLP mixing.
        num_groups:           Max number of groups for variate mixing.
    """

    model_type = "t5"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 0.05,
        feed_forward_proj: str = "relu",
        vocab_size: int = 2,
        pad_token_id: int = 0,
        rope_theta: float = 10000.0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        # encoder sizing
        cnn_channels: int = 512,
        cnn_kernel_size: int = 7,
        cnn_num_conv_layers: int = 2,
        mlp_variate_dim: int = 512,
        num_groups: int = 16,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_kv = int(d_kv)
        self.d_ff = int(d_ff)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.initializer_factor = float(initializer_factor)
        self.feed_forward_proj = str(feed_forward_proj)
        self.rope_theta = float(rope_theta)
        self.cnn_channels = int(cnn_channels)
        self.cnn_kernel_size = int(cnn_kernel_size)
        self.cnn_num_conv_layers = int(cnn_num_conv_layers)
        self.mlp_variate_dim = int(mlp_variate_dim)
        self.num_groups = int(num_groups)

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        assert not self.is_gated_act, "gated activation is not supported"

        attn_implementation = attn_implementation or "sdpa"
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=int(pad_token_id), is_encoder_decoder=False,
            attn_implementation=attn_implementation, **kwargs
        )


def _to_bool(val: bool | str) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


@dataclass
class ForecastingConfig:
    context_length: int
    output_patch_size: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False
    use_arcsinh: bool = False
    max_output_patches: int = 1
    time_encoding_scale: int | None = None

    def __post_init__(self) -> None:
        self.context_length = int(self.context_length)
        self.output_patch_size = int(self.output_patch_size)
        self.input_patch_size = int(self.input_patch_size)
        self.input_patch_stride = int(self.input_patch_stride)
        self.quantiles = [float(q) for q in self.quantiles]
        self.use_reg_token = _to_bool(self.use_reg_token)
        self.use_arcsinh = _to_bool(self.use_arcsinh)
        self.max_output_patches = int(self.max_output_patches)
        if self.time_encoding_scale is not None:
            self.time_encoding_scale = int(self.time_encoding_scale)

    @classmethod
    def editable_fields(cls) -> list[str]:
        return ["context_length", "max_output_patches"]
