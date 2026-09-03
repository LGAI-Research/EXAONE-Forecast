"""The EXAONE Finance forecaster — a high-level API for custom datasets.

A thin wrapper around the released model. You give it your own time series (any
length, missing values allowed) and a forecast horizon, and it returns
probabilistic (quantile) forecasts. No dataset-specific code required.

    from exaone_forecast.finance import from_pretrained

    fc = from_pretrained(device="cuda:0")
    q  = fc.predict(my_series, horizon=20)     # -> (n_quantiles, horizon)
    yhat = fc.point(my_series, horizon=20)     # -> (horizon,) median forecast

The model is univariate and zero-shot: it forecasts each series from its own
recent history. It was pretrained on a synthetic financial corpus and expects
inputs on their natural scale (it normalizes internally).
"""
import math
from pathlib import Path

import numpy as np
import torch

from .._loading import load_checkpoint
from .config import EXAONEFinanceConfig
from .model import EXAONEFinance


class EXAONEFinanceForecaster:
    """Load an EXAONE Finance checkpoint and forecast custom series."""

    def __init__(self, ckpt_dir=None, device=None):
        """Load a checkpoint (~202M params) and forecast custom series.

        ckpt_dir: an explicit checkpoint directory; defaults to ``checkpoint/``
                  next to this file.
        """
        if ckpt_dir is None:
            ckpt_dir = str(Path(__file__).resolve().parent / "checkpoint")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.model = load_checkpoint(ckpt_dir, EXAONEFinance, EXAONEFinanceConfig, device=device)
        self.context_length = int(self.model.forecasting_config.context_length)   # 512
        self._patch = int(self.model.forecasting_config.output_patch_size)         # 16
        self._max_op = int(self.model.forecasting_config.max_output_patches)       # 128
        # the 21 quantile levels the model outputs, e.g. 0.01 .. 0.5 .. 0.99
        self.quantiles = [round(float(q), 4) for q in self.model.quantiles.detach().cpu().tolist()]
        self.max_horizon = self._max_op * self._patch                              # 2048

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_series_list(series):
        """Normalize the input into (list_of_1d_float_arrays, was_single)."""
        arr = None
        if isinstance(series, np.ndarray):
            arr = series
        elif isinstance(series, torch.Tensor):
            arr = series.detach().cpu().numpy()
        if arr is not None:
            if arr.ndim == 1:
                return [arr.astype(np.float32)], True
            if arr.ndim == 2:
                return [row.astype(np.float32) for row in arr], False
            raise ValueError("array input must be 1D (one series) or 2D (n_series, T)")
        # list/tuple: either a flat list of numbers, or a list of series
        seq = list(series)
        if len(seq) == 0:
            raise ValueError("empty input")
        if np.ndim(seq[0]) == 0:                       # flat list of numbers -> one series
            return [np.asarray(seq, dtype=np.float32)], True
        return [np.asarray(s, dtype=np.float32).ravel() for s in seq], False

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, series, horizon, batch_size=1024):
        """Quantile forecasts for one or many univariate series.

        Args:
            series: a single series (1D array / list of numbers), or many series
                (list of 1D arrays of possibly different lengths, or a 2D
                ``(n_series, T)`` array). Missing values may be encoded as NaN.
            horizon: number of future steps to forecast (1 .. ``max_horizon``).
            batch_size: series processed per forward pass.

        Returns:
            np.ndarray of quantile forecasts.
              * single series in  -> shape ``(n_quantiles, horizon)``
              * many series in    -> shape ``(n_series, n_quantiles, horizon)``
            Quantile levels are in ``self.quantiles`` (index of 0.5 is the median).
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")

        series_list, single = self._as_series_list(series)
        L = self.context_length

        # (B, L): keep the last L observations; left-pad short series with NaN
        ctx = np.full((len(series_list), L), np.nan, dtype=np.float32)
        for i, s in enumerate(series_list):
            s = np.asarray(s, dtype=np.float32).ravel()
            if s.size == 0:
                raise ValueError(f"series {i} is empty")
            s = s[-L:]
            ctx[i, L - s.size:] = s

        n_op = max(1, math.ceil(horizon / self._patch))
        chunks = []
        for start in range(0, len(series_list), batch_size):
            cb = torch.from_numpy(ctx[start:start + batch_size]).to(self.device)
            gids = torch.arange(cb.shape[0], device=self.device, dtype=torch.long)
            out = self.model(
                context=cb,
                group_ids=gids,
                future_covariates=None,
                num_output_patches=n_op,
            )
            chunks.append(out.quantile_preds[..., :horizon].float().cpu().numpy())
        q = np.concatenate(chunks, axis=0)             # (B, n_quantiles, horizon)
        return q[0] if single else q

    # ------------------------------------------------------------ convenience
    def point(self, series, horizon, **kw):
        """Median (q=0.5) point forecast. Shape (horizon,) or (n_series, horizon)."""
        q = self.predict(series, horizon, **kw)
        mid = self.quantiles.index(0.5)
        return q[mid] if q.ndim == 2 else q[:, mid, :]

    def interval(self, series, horizon, lower=0.1, upper=0.9, **kw):
        """(low, high) quantile bounds for a prediction interval.

        Each of ``low``/``high`` has the same shape as :meth:`point`.
        """
        q = self.predict(series, horizon, **kw)
        li, ui = self.quantiles.index(round(lower, 4)), self.quantiles.index(round(upper, 4))
        if q.ndim == 2:
            return q[li], q[ui]
        return q[:, li, :], q[:, ui, :]
