<br>

<div align="center">
  <img src="../../assets/exaone_logo.png" alt="EXAONE Demand" width="140">
  <h1>EXAONE Demand</h1>
  <p><em>A member of the <a href="../../README.md">EXAONE Forecast</a> family</em></p>
</div>

<br>

<div align="center">
  <a href="https://huggingface.co/LG-AI-Research/EXAONE-Demand-1.0" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🤗-Weights-FC926C?style=for-the-badge" alt="Weights">
  </a>
  <a href="https://arxiv.org/abs/2609.30880" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/📄-Technical%20Report-4B5563?style=for-the-badge" alt="Technical Report">
  </a>
  <img src="https://img.shields.io/badge/version-1.0-A451E4?style=for-the-badge" alt="Version 1.0">
</div>

<br><br>

**EXAONE Demand** is a **time series foundation model for demand forecasting**. General time series
corpora contain little demand data, and what makes demand hard is exactly what they
under-represent: **short histories, frequent zeros, censoring by stock-outs, and exogenous events**
that move a series without appearing in it.

EXAONE Demand keeps a **general-domain backbone frozen** and adapts it with a **demand-aware
adapter**: five low-rank branches — one shared, and one for each of the four demand classes
(**smooth, intermittent, erratic, lumpy**) — mixed by a **router that reads eight scale-free
statistics** of the input series.

The released weights were trained on **synthetic data only**: the backbone on KernelSynth and the
adapter on a synthetic demand corpus. No real-world series entered any stage of training.

Forecasting is **zero-shot and probabilistic** — each series is forecast from its own recent
history, with no per-dataset training.

<br>

## Architecture

<div style="background-color: rgba(128, 128, 128, 0.1); border-radius: 12px; padding: 12px 24px;">

- Backbone: frozen transformer encoder — instance normalization → patching → 24 blocks → quantile head
- Adapter: every attention projection (`W_q`, `W_k`, `W_v`, `W_o`) and both feed-forward projections
  carry five low-rank branches — 1 shared (fixed weight 0.5) + 4 routed, one per demand class
- Router: eight scale-free statistics of the raw context — zero share, ADI, CV², trend, AC(1), share
  of upward steps, CV, length — mapped to softmax weights over the four routed branches
- Router supervision: trained against the Syntetos–Boylan demand-class membership of each series
- Long horizons: one forward pass covers 64 steps; beyond that each quantile level is unrolled as
  its own path and the paths are recombined by their probability mass

</div>

<div align="center">
  <img src="https://huggingface.co/LG-AI-Research/EXAONE-Demand-1.0/resolve/main/figures/fig_router.png" alt="From raw context to eight statistics to the router's mixture over four experts plus a shared branch" width="100%">
  <br>
  <em>Figure 1. The router reads eight scale-free statistics of the raw context and weighs the four
  demand-class experts; a shared branch is added on top with a fixed weight.</em>
</div>

<br>

## Model Configuration

<div style="background-color: rgba(128, 128, 128, 0.1); border-radius: 12px; padding: 12px 24px;">

- Model Type: adapted time series foundation model (frozen backbone + demand-aware adapter)
- Total parameters: 348,041,240 (≈348M)
  — backbone 311,928,216 (frozen) · adapter and router 36,113,024

- Encoder blocks: 24
- Model width: 1024
- Embedding width: 2048
- Attention heads: 16 (RoPE)
- Patch size (input = output): 16

- Adapted projections: 6 per block × 24 blocks = 144
- Adapter branches: 5 — shared rank 16, routed rank 4
- Router: 8 statistics → hidden width 32 → 5 branch weights

- Context length: 8192
- Horizon per forward pass: 64 (longer horizons are unrolled)
- Quantile levels: 21 (0.01, 0.05, 0.10, …, 0.90, 0.95, 0.99)
- Dtype: float32

</div>

<br>

## Evaluation Results

Zero-shot results on **22 held-out demand datasets**. Both versions of EXAONE Demand from the
technical report are compared against **36 TSFMs** and the frozen backbone, all run under the same
protocol. Error metrics are geometric means over the 22 datasets (lower is better). *Avg. Rank* is
the mean position by per-dataset MASE, and *Win rate* is the share of the 836 per-dataset MASE
comparisons against the other 38 rows that a model wins.

| # | Model | Avg. Rank | Win rate | MASE | ND | WQL | MAPE | MAE | MSIS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | EXAONE Demand | 4.09 | 91.9% | 1.0667 | 0.1409 | 0.1138 | 0.2565 | 60.24 | 10.70 |
| **2** | **EXAONE Demand (Synthetic) — released** | **5.55** | **88.0%** | **1.0742** | **0.1420** | **0.1147** | **0.2583** | **60.73** | **10.63** |
| 3 | TiRex-1.1 | 7.64 | 82.5% | 1.0818 | 0.1451 | 0.1161 | 0.2911 | 62.06 | 10.75 |
| 4 | Chronos-2 | 8.59 | 80.0% | 1.0885 | 0.1497 | 0.1406 | 0.3010 | 64.02 | 14.48 |
| 5 | EXAONE Backbone (zero-shot) | 7.95 | 81.7% | 1.1050 | 0.1445 | 0.1169 | 0.2609 | 61.82 | 11.41 |
| 6 | TimesFM-2.5 | 10.05 | 76.2% | 1.1073 | 0.1486 | 0.1215 | 0.3032 | 63.58 | 12.31 |

The two versions differ only in the data the adapter is trained on:

- **EXAONE Demand (Synthetic)** — synthetic demand only. **This is the released model.**
- **EXAONE Demand** — real-world and synthetic demand together. Not released.

A model trained on open demand data inherits the licenses of that data, while one trained on series
we generated ourselves inherits none of them. The released synthetic-only model is ahead of every
one of the 36 baselines on **all six error metrics, the average rank, and the win rate**, and its
MSIS is the best of all 39 rows.

<div align="center">
  <img src="https://huggingface.co/LG-AI-Research/EXAONE-Demand-1.0/resolve/main/figures/fig_mase.png" alt="MASE over 22 held-out demand datasets for EXAONE Demand and the strongest baselines" width="70%">
  <br>
  <em>Figure 2. MASE over the 22 held-out demand datasets (lower is better). The released model is
  EXAONE Demand (Synthetic).</em>
</div>

Full protocol and results are in the
[Technical Report](https://arxiv.org/abs/2609.30880).

<br>

## Quickstart

```bash
pip install "exaone-forecast[demand] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
```

```python
import numpy as np
from exaone_forecast.demand import from_pretrained

fc = from_pretrained(device="cuda:0")          # downloads the weights on first use

rng = np.random.default_rng(0)
series = [(rng.random(300) < 0.2) * rng.integers(1, 9, 300) for _ in range(4)]

q = fc.predict(series, horizon=28)             # (4, 21, 28)  all quantile levels
yhat = fc.point(series, horizon=28)            # (4, 28)      median forecast
lo, hi = fc.interval(series, horizon=28, lower=0.1, upper=0.9)
```

A genuine zero ("nothing was demanded") is a `0`, not a `NaN`: the zero share is one of the
statistics the router reads. To see what the router sees:

```python
from exaone_forecast.demand import demand_stats, demand_membership, DEMAND_CLASSES

demand_stats(x)        # (batch, 8)  the router's input
demand_membership(x)   # (batch, 4)  membership over DEMAND_CLASSES
```

A runnable end-to-end example is in [`quickstart.py`](./quickstart.py).

<br>

## Available checkpoints

| File | Version | Parameters | Dtype | Notes |
|---|---|---|---|---|
| `exaone-demand-1.0.safetensors` | 1.0 | 348M | float32 | Synthetic-only training; 21-quantile head; loaded by `from_pretrained()` |

The file lives in the [Hugging Face repository](https://huggingface.co/LG-AI-Research/EXAONE-Demand-1.0)
alongside its `config.json`, which describes the architecture. `from_pretrained` fetches both and
loads the weights verbatim.

<br>

## Intended use

EXAONE Demand is intended for **zero-shot probabilistic forecasting** of demand series — retail
sales, spare-part orders, bookings, electricity load, ridership and the like — for **research and
educational** purposes. Each series is forecast from its own recent history; no per-dataset training
is required. Use of the released weights is limited to non-commercial research and education by the
EXAONE model license.

**Not intended for:** any **commercial** use of the released weights, or any use excluded by the
[license](#license).

<br>

## Limitation

**Domain coverage.** The adapter is specialised to demand-shaped series. Outside that domain,
behaviour falls back toward the general-domain backbone.

**Synthetic training data.** The released weights never saw a real-world series. Behaviour the
generators do not reproduce — an industry's idiosyncratic seasonality, an unrecorded external
event — was not learned. The technical report's real-world version is ahead by 0.0075 MASE.

**Short windows and unrecorded events.** The router judges a series from its input window alone, so
a very short window makes the demand-class assignment less reliable. Stock-out censoring and
promotions are not modelled unless they show up in the series itself.

<br>

## License

The **model weights** are released under the **EXAONE AI Model License Agreement 1.2 - NC**, which
limits use to non-commercial research and education. The full terms ship with the weights on the
[Hugging Face repository](https://huggingface.co/LG-AI-Research/EXAONE-Demand-1.0).

The **code** in this repository is licensed separately under the
[BSD-3-Clause-LG AI Research License](../../LICENSE), which permits commercial use.

Third-party open source components and their licenses are listed in [Notice.md](./Notice.md).

<br>

## Citation

```bibtex
@article{lgai2026exaonedemand,
  title   = {EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting},
  author  = {Lee, Seunghan and Han, Sangjun and Seo, Jun and Kang, Junhyeok and
             Lee, Jaehoon and Lim, Tae Yoon and Kang, Dongwan and Choi, Hwanil and
             Kim, Minjae and Yoo, Sungdong and Lee, Soonyoung and Ahn, Wonbin},
  journal = {arXiv preprint arXiv:2609.30880},
  year    = {2026}
}
```

<br>

## Contact

LG AI Research Technical Support: contact_us@lgresearch.ai
