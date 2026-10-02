"""Configuration for EXAONE Demand.

The released checkpoint stores a flat ``config.json``, which this class reads.
The modules below were written against a nested configuration object, so the
flat fields are also exposed as three namespaces — ``config.data``,
``config.model`` and ``config.adapter`` — and the model code reads those.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["EXAONEDemandConfig"]

# Quantile levels the model emits, in the order they occupy the quantile axis.
QUANTILES = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]


class _Namespace:
    """Attribute view over a subset of the flat config.

    ``Block(**config.model)`` in the encoder expects a mapping, so this also
    behaves as one.
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Namespace({self.__dict__})"


class EXAONEDemandConfig:
    """Hyperparameters of the released EXAONE Demand model.

    A plain class rather than a ``transformers.PretrainedConfig``: EXAONE Demand
    runs on torch alone, so installing it never pulls in transformers or the
    packages that come with it. Keys in ``config.json`` that this model does not
    read (``architectures``, ``torch_dtype``, ...) are kept as attributes.
    """

    model_type = "exaone-demand"

    def __init__(
        self,
        # --- backbone ---
        dim_model: int = 1024,
        dim_embed: int = 2048,
        num_heads: int = 16,
        num_layers: int = 24,
        multiplier: float = 2.667,
        dropout: float = 0.05,
        ssmax: bool = True,
        # --- series handling ---
        context_length: int = 8192,
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        output_patch_size: int = 16,
        max_output_patches: int = 4,
        num_reg_tokens: int = 1,
        quantiles: list[float] | None = None,
        # --- demand adapter ---
        adapter_enabled: bool = True,
        adapter_ranks: list[int] | None = None,
        adapter_alpha: int = 8,
        adapter_dropout: float = 0.05,
        adapter_shared_weight: float = 0.5,
        adapter_temperature: float = 1.0,
        adapter_targets: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.dim_model = dim_model
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.multiplier = multiplier
        self.dropout = dropout
        self.ssmax = ssmax

        self.context_length = context_length
        self.input_patch_size = input_patch_size
        self.input_patch_stride = input_patch_stride
        self.output_patch_size = output_patch_size
        self.max_output_patches = max_output_patches
        self.num_reg_tokens = num_reg_tokens
        self.quantiles = list(quantiles) if quantiles is not None else list(QUANTILES)

        # Branch 0 is the shared branch and carries the larger rank; branches
        # 1..4 are the routed ones, one per demand class.
        self.adapter_enabled = adapter_enabled
        self.adapter_ranks = list(adapter_ranks) if adapter_ranks is not None else [16, 4, 4, 4, 4]
        self.adapter_alpha = adapter_alpha
        self.adapter_dropout = adapter_dropout
        self.adapter_shared_weight = adapter_shared_weight
        self.adapter_temperature = adapter_temperature
        self.adapter_targets = list(adapter_targets) if adapter_targets is not None else [
            "to_q", "to_k", "to_v", "to_out", "w12", "w3",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    # -- serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EXAONEDemandConfig":
        return cls(**config)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str | os.PathLike) -> "EXAONEDemandConfig":
        """Read ``config.json`` from a checkpoint directory."""
        with open(Path(ckpt_dir) / "config.json", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    # -- nested views -------------------------------------------------------
    @property
    def model(self) -> _Namespace:
        return _Namespace(
            dim_model=self.dim_model,
            dim_embed=self.dim_embed,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            multiplier=self.multiplier,
            dropout=self.dropout,
            ssmax=self.ssmax,
        )

    @property
    def data(self) -> _Namespace:
        return _Namespace(
            context_length=self.context_length,
            input_patch_size=self.input_patch_size,
            input_patch_stride=self.input_patch_stride,
            output_patch_size=self.output_patch_size,
            max_output_patches=self.max_output_patches,
            num_reg_tokens=self.num_reg_tokens,
            quantiles=self.quantiles,
        )

    @property
    def adapter(self) -> _Namespace:
        return _Namespace(
            enabled=self.adapter_enabled,
            num_experts=len(self.adapter_ranks),
            ranks=self.adapter_ranks,
            alpha=self.adapter_alpha,
            dropout=self.adapter_dropout,
            shared_weight=self.adapter_shared_weight,
            temperature=self.adapter_temperature,
            targets=self.adapter_targets,
        )

    @property
    def max_horizon(self) -> int:
        """Longest horizon one forward pass produces."""
        return self.max_output_patches * self.output_patch_size
