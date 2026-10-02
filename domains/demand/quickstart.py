"""EXAONE Demand — minimal inference example on a custom dataset.

    pip install "exaone-forecast[demand] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
    python domains/demand/quickstart.py

The released weights are downloaded from the Hugging Face Hub on first use
(LG-AI-Research/EXAONE-Demand-1.0) and cached locally.
"""
import numpy as np

from exaone_forecast.demand import DEMAND_CLASSES, demand_membership, from_pretrained

HORIZON = 28


def load_my_dataset():
    """Return your demand series as a list of 1D numpy arrays (any length each).

    Replace this with your own data. A few common patterns:

        # a CSV where each column is one SKU
        import pandas as pd
        df = pd.read_csv("my_sales.csv")
        return [df[c].to_numpy(dtype=float) for c in df.columns]

        # a 2D numpy array of shape (n_series, T)
        return np.load("my_series.npy")

    Notes:
      * Series may have different lengths; each is forecast independently.
      * Feed counts on their natural scale (units sold, orders, ...); the model
        normalises internally and reads only the last `context_length` points.
      * Missing observations are allowed — encode them as np.nan. A genuine zero
        ("nothing was demanded") is a 0, not a NaN, and the difference matters
        here: the zero share is one of the statistics that picks the adapter.
    """
    rng = np.random.default_rng(0)
    t = np.arange(400)

    smooth = 50 + 12 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, t.size)
    intermittent = (rng.random(t.size) < 0.18) * rng.integers(1, 9, t.size)
    erratic = np.abs(30 + rng.normal(0, 18, t.size))
    lumpy = (rng.random(t.size) < 0.12) * rng.integers(5, 60, t.size)

    return [np.clip(s, 0, None).astype(float) for s in (smooth, intermittent, erratic, lumpy)]


def main():
    series = load_my_dataset()

    forecaster = from_pretrained(device="cpu")  # "cuda:0" if you have a GPU
    print(f"quantile levels : {list(forecaster.quantiles)}")
    print(f"context length  : {forecaster.context_length}")
    print(f"single-pass horizon: {forecaster.max_horizon} (longer horizons unroll)\n")

    for i, s in enumerate(series):
        # What the router makes of this series, before any forecasting.
        membership = demand_membership(np.asarray(s)[None, :])[0]
        profile = "  ".join(f"{c}={v:.2f}" for c, v in zip(DEMAND_CLASSES, membership))

        quantiles = forecaster.predict(s, horizon=HORIZON)   # (21, HORIZON)
        median = forecaster.point(s, horizon=HORIZON)        # (HORIZON,)
        low, high = forecaster.interval(s, horizon=HORIZON, lower=0.1, upper=0.9)

        print(f"series {i}  len={len(s):4d}  {profile}")
        print(f"  median[:5]   {np.round(median[:5], 2)}")
        print(f"  80% interval {np.round(low[:5], 2)} .. {np.round(high[:5], 2)}")
        print(f"  full quantile grid: {quantiles.shape}\n")

    # Several series in one call. They stay independent of one another unless
    # you pass group_ids saying which ones belong together.
    batch = np.stack([s[-200:] for s in series])
    print("batched:", forecaster.predict(batch, horizon=HORIZON).shape)

    # Sibling series that should inform one another — e.g. SKUs in one category.
    grouped = forecaster.predict(batch, horizon=HORIZON, group_ids=[0, 0, 1, 1])
    print("grouped:", grouped.shape)


if __name__ == "__main__":
    main()
