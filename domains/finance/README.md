<br>

<div align="center">
  <img src="../../assets/exaone_logo.png" alt="EXAONE Finance" width="140">
  <h1>EXAONE Finance</h1>
  <p><em>A member of the <a href="../../README.md">EXAONE Forecast</a> family</em></p>
</div>

<br>

<div align="center">
  <a href="https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🤗-Weights-FC926C?style=for-the-badge" alt="Weights">
  </a>
  <a href="https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0/resolve/main/EXAONE_Finance_v1.0_Technical_Report.pdf" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/📄-Technical%20Report-4B5563?style=for-the-badge" alt="Technical Report">
  </a>
  <img src="https://img.shields.io/badge/version-1.0-A451E4?style=for-the-badge" alt="Version 1.0">
</div>

<br><br>

**EXAONE Finance** is a **time series foundation model for financial forecasting**. It uses an
**attention-free** architecture: self-attention is replaced by two linear-time operators — a
**causal 1D convolution** for temporal mixing and a **group-aware pooling MLP** for variate mixing —
so cost grows **linearly** in both sequence length and variate count. A **masked-context
augmentation** exposes the model to contiguous missing spans during training, making it robust to
the missingness pervasive in financial markets.

It is pretrained on a **synthetic financial corpus** whose generator is designed to reproduce the
properties of financial series: heavy tails, volatility clustering, jumps, regime shifts, and
cross-asset dependence.

Forecasting is **zero-shot and probabilistic** — each series is forecast from its own recent
history, with no per-dataset training.

<br>

## Architecture

<div style="background-color: rgba(128, 128, 128, 0.1); border-radius: 12px; padding: 12px 24px;">

- Encoder: attention-free — temporal mixing → variate mixing → feed-forward, each residual
- Temporal mixing: stacked causal 1D convolutions, left-padded (no future leakage)
- Variate mixing: group-aware mean pooling followed by an MLP
- Input pipeline: instance normalization (optional arcsinh) → patching → residual-MLP embedding
- Head: patch decoder emitting multi-quantile trajectories
- Training augmentation: masked-context — a contiguous input span is marked known-missing

</div>

<br>

## Model Configuration

<div style="background-color: rgba(128, 128, 128, 0.1); border-radius: 12px; padding: 12px 24px;">

- Model Type: attention-free time series foundation model
- Total parameters: 202,319,520 (≈202M)

- Encoder blocks: 12
- Hidden dimension: 1024
- Feed-forward dimension: 4096
- Temporal mixer: 512 channels · kernel 7 · 2 conv layers per block
- Variate mixer: hidden dimension 512
- Patch size (input = output): 16

- Context length: 512
- Maximum horizon: 2048
- Quantile levels: 21 (0.01, 0.05, 0.10, …, 0.90, 0.95, 0.99)
- Attention layers: none
- Dtype: float32

</div>

<br>

## Evaluation Results

Zero-shot results on **FinVerse**, a financial forecasting benchmark spanning FX, commodities,
crypto-assets, fixed income, equities, ETFs, and macroeconomic indicators. **44 models** were scored
on identical windows. Models are ranked per (tier, spec, metric) cell; *Avg. Rank* is the geometric
mean of per-cell ranks, and **Rank Sum** is the sum of the three per-tier placements — lower is
better.

| # | Model | Tier 1 (Point) | Avg. Rank | Tier 2 (IC) | Avg. Rank | Tier 3 (Portfolio) | Avg. Rank | Rank Sum |
|---|---|---|---|---|---|---|---|---|
| **1** | **EXAONE Finance** | **1** | **6.53** | **1** | **7.45** | **1** | **7.43** | **3** |
| 2 | Chronos-2 (Synthetic) | 9 | 11.68 | 3 | 8.74 | 2 | 9.90 | 14 |
| 3 | Reverso (Small) | 6 | 9.60 | 2 | 8.64 | 6 | 13.04 | 14 |
| 4 | TiRex-1.1 | 2 | 6.98 | 4 | 9.71 | 14 | 15.76 | 20 |
| 5 | Chronos-2 | 8 | 10.14 | 13 | 16.12 | 9 | 15.17 | 30 |

The three tiers score complementary objectives:

- **Tier 1 — Point accuracy.** MASE against an in-context naïve forecast, plus directional hit rate.
- **Tier 2 — Cross-sectional skill.** Information coefficient: the Spearman rank correlation between
  predicted and realized changes across assets at each origin.
- **Tier 3 — Portfolio backtest.** Annualized return, Sharpe ratio, volatility, and maximum drawdown
  of a top-*K*% equal-weight long-only portfolio.

EXAONE Finance ranks **first in all three tiers** for a perfect rank sum of **3**, ahead of the
strongest baseline at 14.

<div align="center">
  <img src="https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0/resolve/main/figures/fig_tier_transposed.png" alt="Average rank per tier, EXAONE Finance against the strongest baselines" width="100%">
  <br>
  <em>Figure 1. Average rank within each tier (lower is better). EXAONE Finance leads all three,
  and no baseline is consistently strong across them.</em>
</div>

At **202M parameters** it sits on the Pareto frontier, outperforming models more than an order of
magnitude larger — the scaling law familiar from general-domain forecasting does not yet hold here.

<div align="center">
  <img src="https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0/resolve/main/figures/fig_size_vs_perf.png" alt="Aggregate rank versus parameter count" width="88%">
  <br>
  <em>Figure 2. Aggregate rank versus model size. A larger model does not reliably improve the rank.</em>
</div>

It is also the **only model that wins its head-to-head comparison against all 43 baselines**, with
per-opponent win rates from 0.51 to 0.90.

<div align="center">
  <img src="https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0/resolve/main/figures/fig_winrate.png" alt="Pairwise win rate between every pair of models" width="88%">
  <br>
  <em>Figure 3. Pairwise win rate, pooled over all (tier, spec, metric) cells. EXAONE Finance is the
  only model whose entire row exceeds 0.5.</em>
</div>

Full protocol and results are in the
[Technical Report](https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0/resolve/main/EXAONE_Finance_v1.0_Technical_Report.pdf).

> The technical report and the figures above refer to this model by its full name,
> **EXAONE Forecast for Finance**. **EXAONE Finance** is the short form used throughout this
> repository and on the Hub.

<br>

## Quickstart

```bash
pip install "exaone-forecast[finance] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
```

```python
import numpy as np
from exaone_forecast.finance import from_pretrained

fc = from_pretrained(device="cuda:0")          # downloads the weights on first use

series = [np.random.randn(500).cumsum() + 100 for _ in range(4)]

q = fc.predict(series, horizon=20)             # (4, 21, 20)  all quantile levels
yhat = fc.point(series, horizon=20)            # (4, 20)      median forecast
lo, hi = fc.interval(series, horizon=20, lower=0.1, upper=0.9)
```

A runnable end-to-end example is in [`quickstart.py`](./quickstart.py).

<br>

## Available checkpoints

| File | Version | Parameters | Dtype | Notes |
|---|---|---|---|---|
| `exaone-finance-1.0.safetensors` | 1.0 | 202M | float32 | 21-quantile head; loaded by `from_pretrained("default")` |

The file lives in the [Hugging Face repository](https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0)
alongside its `config.json`, which describes the architecture. `from_pretrained` fetches both and
loads the weights verbatim.

<br>

## Intended use

EXAONE Finance is intended for **zero-shot probabilistic forecasting** of financial and other
real-valued time series, for **research and educational** purposes. Each series is forecast from its
own recent history; no per-dataset training is required. Use of the released weights is limited to
non-commercial research and education by the EXAONE model license.

**Not intended for:** any **commercial** use of the released weights, or any use excluded by the
[license](#license). Forecasts are statistical outputs, **not financial advice** — do not use them
as the sole basis for an investment decision.

<br>

## Limitation

**Domain coverage.** Results are reported on a financial benchmark. Broad general-domain
benchmarking is future work, and behaviour outside the tested asset classes is not characterised.

**Zero-shot only.** Only zero-shot performance is reported. Parameter-efficient fine-tuning and
ensembling are promising but untested directions.

**Variate grouping.** In this release every series is treated as an individual channel. The
group-aware mixer natively supports richer cross-series grouping — from the multiple fields of a
single security to related instruments within a portfolio — but that capability is not exercised by
the published checkpoint. Systematically exploring these grouping schemes, and scaling to broader
multivariate financial panels, is a promising direction for further gains.

<br>

## License

The **model weights** are released under the **EXAONE AI Model License Agreement 1.2 - NC**, which
limits use to non-commercial research and education. The full terms ship with the weights on the
[Hugging Face repository](https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0).

The **code** in this repository is licensed separately under the
[BSD-3-Clause-LG AI Research License](../../LICENSE), which permits commercial use.

<br>

## Citation

```bibtex
@techreport{lgai2026exaoneforecastfinance,
  title       = {EXAONE Forecast for Finance: An Attention-free Time Series Foundation Model for Financial Time Series},
  author      = {Lee, Seunghan and Lee, Jaehoon and Seo, Jun and Lim, Tae Yoon and
                 Kang, Dongwan and Choi, Hwanil and Kim, Minjae and Yoo, Sungdong and
                 Kang, Junhyeok and Han, Sangjun and Lee, Soonyoung and Ahn, Wonbin},
  year        = {2026},
  institution = {LG AI Research},
  note        = {Technical Report}
}
```

<br>

## Contact

LG AI Research Technical Support: contact_us@lgresearch.ai
