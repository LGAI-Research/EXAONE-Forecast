"""EXAONE Finance — minimal inference example on a custom dataset.

    pip install "exaone-forecast[finance] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
    python domains/finance/quickstart.py

The released weights are downloaded from the Hugging Face Hub on first use
(LG-AI-Research/EXAONE-Finance-1.0) and cached locally.
"""
import numpy as np

from exaone_forecast.finance import from_pretrained


def load_my_dataset():
    """Return your time series as a list of 1D numpy arrays (any length each).

    Replace this with your own data. A few common patterns:

        # a CSV where each column is one series
        import pandas as pd
        df = pd.read_csv("my_prices.csv")
        return [df[c].to_numpy(dtype=float) for c in df.columns]

        # a 2D numpy array of shape (n_series, T)
        return np.load("my_series.npy")

    Notes:
      * Series may have different lengths; each is forecast independently.
      * Feed values on their natural scale (prices, rates, ...); the model
        normalizes internally and uses only the last `context_length` points.
      * Missing values are allowed — encode them as np.nan.
    """
    rng = np.random.default_rng(0)
    series = []
    for k in range(6):
        n = int(rng.integers(120, 800))
        t = np.arange(n)
        level = 100 + 20 * np.sin(2 * np.pi * t / (40 + 10 * k))
        trend = 0.05 * k * t
        noise = np.cumsum(rng.normal(0, 1.5, n)) * 0.3
        series.append((level + trend + noise).astype(np.float32))
    series[0][50:60] = np.nan          # a missing gap, to show NaN handling
    return series


def main():
    horizon = 20

    fc = from_pretrained()             # add device="cuda:0" to pin a GPU
    n_params = sum(p.numel() for p in fc.model.parameters()) / 1e6
    print(f"loaded EXAONE Finance  params={n_params:.0f}M  device={fc.device}")
    print(f"context_length={fc.context_length}  max_horizon={fc.max_horizon}")
    print(f"quantile levels ({len(fc.quantiles)}): {fc.quantiles}")

    data = load_my_dataset()
    print(f"\nforecasting {len(data)} series, horizon={horizon} ...")

    quantiles = fc.predict(data, horizon=horizon)      # (n_series, n_quantiles, horizon)
    median = fc.point(data, horizon=horizon)           # (n_series, horizon)
    lo, hi = fc.interval(data, horizon=horizon, lower=0.1, upper=0.9)

    print(f"  quantile forecast shape = {quantiles.shape}")
    print(f"\nseries #0: history length = {len(data[0])}")
    print(f"  last 5 observed : {np.asarray(data[0])[-5:]}")
    print(f"  median forecast : {np.round(median[0][:5], 3)} ...")
    print(f"  10% lower bound : {np.round(lo[0][:5], 3)} ...")
    print(f"  90% upper bound : {np.round(hi[0][:5], 3)} ...")


if __name__ == "__main__":
    main()
