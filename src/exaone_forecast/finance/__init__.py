"""EXAONE Finance — the financial domain model of the EXAONE Forecast family.

    from exaone_forecast.finance import from_pretrained

    fc = from_pretrained(device="cuda:0")
    q  = fc.predict(my_series, horizon=20)

The encoder is attention-free: temporal mixing is a 1D causal convolution and
variate mixing is a group-aware pooling MLP, so cost grows linearly in both
sequence length and variate count. This design is particular to this model and
is not shared family-wide.

The weights are published separately at ``LG-AI-Research/EXAONE-Finance`` on the
Hugging Face Hub and are covered by the EXAONE AI Model License Agreement
1.2 - NC (non-commercial). Installing this package does not grant commercial
rights to the weights it downloads.
"""
from __future__ import annotations

import os

from .._hub import download_checkpoint
from .._loading import load_checkpoint
from .config import EXAONEFinanceConfig, ForecastingConfig
from .forecaster import EXAONEFinanceForecaster
from .model import EXAONEFinance

__all__ = [
    "from_pretrained",
    "load_pretrained",
    "EXAONEFinance",
    "EXAONEFinanceConfig",
    "EXAONEFinanceForecaster",
    "ForecastingConfig",
    "REPO_ID",
    "CHECKPOINTS",
    "VERSION",
]

#: Released version of this model.
VERSION = "1.0"

#: Hugging Face repository holding the released weights.
REPO_ID = "LG-AI-Research/EXAONE-Finance"

#: Published checkpoints, keyed by name.
CHECKPOINTS = {
    "default": "exaone-finance-1.0.safetensors",
}


def from_pretrained(
    checkpoint: str = "default",
    device: str | None = None,
    repo_id: str = REPO_ID,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> EXAONEFinanceForecaster:
    """Download the released checkpoint and return a ready-to-use forecaster.

    Args:
        checkpoint: which published checkpoint to load (see :data:`CHECKPOINTS`).
        device: torch device string, e.g. ``"cuda:0"``. Defaults to CUDA when
            available, otherwise CPU.
        repo_id: Hugging Face repository holding the weights.
        revision: optional git revision (branch, tag, or commit) of that repository.
        cache_dir: optional Hugging Face cache directory.

    Returns:
        :class:`~exaone_forecast.finance.forecaster.EXAONEFinanceForecaster` —
        call ``.predict`` / ``.point`` / ``.interval`` on it.
    """
    if checkpoint not in CHECKPOINTS:
        raise ValueError(
            f"unknown checkpoint {checkpoint!r}; available: {sorted(CHECKPOINTS)}"
        )

    ckpt_dir = download_checkpoint(
        repo_id=repo_id,
        weight_filename=CHECKPOINTS[checkpoint],
        staging_name=f"exaone-finance-{checkpoint}",
        revision=revision,
        cache_dir=cache_dir,
    )
    return EXAONEFinanceForecaster(ckpt_dir=ckpt_dir, device=device)


def load_pretrained(ckpt_dir: str | os.PathLike, device=None) -> EXAONEFinance:
    """Load an EXAONE Finance checkpoint directory into the model class.

    Use this when you want the raw model. For forecasting from your own series,
    use :func:`from_pretrained` or :class:`EXAONEFinanceForecaster` instead.
    """
    return load_checkpoint(ckpt_dir, EXAONEFinance, EXAONEFinanceConfig, device=device)
