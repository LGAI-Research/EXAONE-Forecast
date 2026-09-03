"""EXAONE Forecast — LG AI Research's family of time series foundation models.

Each model in the family targets its own domain and carries its own architecture,
training recipe, and released weights. Nothing about a model's design is implied
by its membership here; what the family guarantees is a common calling
convention, described by :class:`exaone_forecast.Forecaster`.

Import the model you want:

    from exaone_forecast.finance import from_pretrained    # EXAONE Finance

This package is inference-only. Training pipelines are not shipped.
"""
from ._api import Forecaster

__version__ = "1.0.0"

__all__ = ["Forecaster", "__version__"]
