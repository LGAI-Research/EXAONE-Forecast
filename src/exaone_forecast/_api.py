"""The forecasting surface every model in the family exposes.

Members of EXAONE Forecast do not share an architecture — each is designed for
its own domain and documented on its own model page. What they do share is this
calling convention, so code written against one model runs against another.

A model package provides ``from_pretrained(...)``, which returns an object
satisfying :class:`Forecaster`.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

__all__ = ["Forecaster"]


@runtime_checkable
class Forecaster(Protocol):
    """Zero-shot probabilistic forecasting over user-supplied series."""

    #: Quantile levels the model emits, in the order they appear on the quantile axis.
    quantiles: Sequence[float]
    #: Number of trailing observations the model reads from each series.
    context_length: int
    #: Largest horizon the model can forecast in one call.
    max_horizon: int

    def predict(self, series, horizon: int):
        """Quantile forecasts. ``(n_quantiles, horizon)``, or ``(n_series, ...)`` for many."""
        ...

    def point(self, series, horizon: int):
        """Median forecast. ``(horizon,)``, or ``(n_series, horizon)`` for many."""
        ...

    def interval(self, series, horizon: int, lower: float, upper: float):
        """``(low, high)`` quantile bounds, each shaped like :meth:`point`."""
        ...
