"""EXAONE Demand — the demand-forecasting model of the EXAONE Forecast family.

    from exaone_forecast.demand import from_pretrained

    fc = from_pretrained(device="cuda:0")
    q  = fc.predict(my_series, horizon=28)

Demand series do not all behave alike: some are smooth and seasonal, some are
intermittent spare-part orders, some spike on promotion. The model keeps a
general-domain backbone frozen and adapts it with five low-rank branches — one
shared, four tied to those demand classes — mixed per series by a router that
reads eight scale-free statistics of the input window. This design is particular
to this model and is not shared family-wide.

The weights are published separately at ``LG-AI-Research/EXAONE-Demand-1.0`` on
the Hugging Face Hub and are covered by the EXAONE AI Model License Agreement
1.2 - NC (non-commercial). Installing this package does not grant commercial
rights to the weights it downloads. They were trained on synthetic demand data
only; see the model card for what that does and does not cover.
"""
from __future__ import annotations

import os

from .._hub import download_checkpoint
from .._loading import load_checkpoint
from .adapter import DEMAND_CLASSES, demand_membership, demand_stats
from .config import EXAONEDemandConfig
from .forecaster import EXAONEDemandForecaster
from .model import EXAONEDemand

__all__ = [
    "from_pretrained",
    "load_pretrained",
    "EXAONEDemand",
    "EXAONEDemandConfig",
    "EXAONEDemandForecaster",
    "demand_stats",
    "demand_membership",
    "DEMAND_CLASSES",
]

#: Hub repository holding the released weights.
HF_REPO_ID = "LG-AI-Research/EXAONE-Demand-1.0"

#: Name of the weight file inside that repository.
HF_WEIGHT_FILE = "exaone-demand-1.0.safetensors"


def load_pretrained(ckpt_dir: str | os.PathLike, device=None) -> EXAONEDemand:
    """Load a local checkpoint directory into :class:`EXAONEDemand`.

    Reads ``config.json`` and ``model.safetensors`` and loads the weights
    verbatim. See :mod:`exaone_forecast._loading` for why the usual
    ``from_pretrained`` conversion path is bypassed.
    """
    return load_checkpoint(ckpt_dir, EXAONEDemand, EXAONEDemandConfig, device=device)


def from_pretrained(
    ckpt_dir: str | os.PathLike | None = None,
    device=None,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> EXAONEDemandForecaster:
    """Return a ready-to-use forecaster.

    Args:
        ckpt_dir: a local checkpoint directory. When omitted, the released
            weights are fetched from the Hugging Face Hub.
        device: torch device to run on, e.g. ``"cuda:0"``.
        revision: Hub revision to pin. Ignored when ``ckpt_dir`` is given.
        cache_dir: where to cache the download. Ignored when ``ckpt_dir`` is given.
    """
    if ckpt_dir is None:
        ckpt_dir = download_checkpoint(
            repo_id=HF_REPO_ID,
            weight_filename=HF_WEIGHT_FILE,
            staging_name="exaone-demand-1.0",
            revision=revision,
            cache_dir=cache_dir,
        )

    model = load_pretrained(ckpt_dir, device=device)
    return EXAONEDemandForecaster(model, device=device)
