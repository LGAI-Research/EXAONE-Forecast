<br>

<div align="center">
  <img src="assets/exaone_logo.png" alt="EXAONE Forecast" width="160">
  <h1>EXAONE Forecast</h1>
</div>

<br>

<div align="center">
  <a href="https://huggingface.co/LG-AI-Research/EXAONE-Finance" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🤗-HuggingFace-FC926C?style=for-the-badge" alt="HuggingFace">
  </a>
  <a href="https://github.com/LGAI-Research/EXAONE-Forecast" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🖥️-GitHub-2B3137?style=for-the-badge" alt="GitHub">
  </a>
</div>

<br><br>

**EXAONE Forecast** is LG AI Research's **family of time series foundation models**. Each member
targets its own domain and carries its own architecture, training recipe, and released weights. Two
members may share a design or may not, so those details are **documented per model** rather than
assumed family-wide.

This repository is the **`exaone-forecast` inference runtime** — a self-contained package that loads
a released checkpoint and serves probabilistic forecasts through a small, uniform API. Every model
in the family exposes the same `from_pretrained()` / `predict()` surface, so the code you write does
not change when you switch models.

The code here is permissively licensed; the released **weights are non-commercial** — see
[License](#license).

**EXAONE Finance** is the first released member. See
[Available models](#available-models) and its
[model page](domains/finance/README.md).

<br>

## Available models

| Model | Domain | Version | Weights | Documentation |
|---|---|---|---|---|
| **EXAONE Finance** | Financial time series | 1.0 | [`LG-AI-Research/EXAONE-Finance`](https://huggingface.co/LG-AI-Research/EXAONE-Finance) | [`domains/finance`](domains/finance/README.md) |

Each model page carries its own architecture, evaluation results, intended use, and limitations.
Weights live in that model's own Hugging Face repository and are fetched on first use.

<br>

## Requirements

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0 &nbsp;(a **CUDA GPU is recommended**; CPU inference works and is fine for small
  workloads)
- NumPy ≥ 1.20 · transformers ≥ 4.30 · einops ≥ 0.6 · safetensors ≥ 0.3 · huggingface_hub ≥ 0.20
  &nbsp;(floors are the versions this release was validated against)

Install the package, naming the model you want in the extras selector — the dependencies above
come with it:

```bash
pip install "exaone-forecast[finance] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
```

The selector installs one model's requirements, not the whole family's. Every model shares the
backbone and the checkpoint loader listed above; anything a model needs beyond that is scoped to
its own extra, so adding a model to the family never changes what an existing installation pulls
in. Use `[all]` for every model, or no selector at all for the shared runtime only.

Model weights are **not** part of the package. They live in that model's own Hugging Face
repository and are downloaded on first use, so installing costs a few hundred KB regardless of how
many models the family contains.

From a checkout, `pip install ".[finance]"` (add `-e` for an editable install) does the same.

`huggingface_hub` is included, so `from_pretrained` can fetch the released weights out of the box.
Downloads honor the standard Hub environment (`HF_HOME` for the cache, `HF_TOKEN` for a gated repo).

Verify the install:

```python
import exaone_forecast.finance as m
print(m.VERSION, m.REPO_ID)
```

> Dependency ranges are declared in [`pyproject.toml`](pyproject.toml)
> (distribution name `exaone-forecast`, import name `exaone_forecast`).

<br>

## Quickstart

Each model is one import away, and `from_pretrained` handles the rest in a single call: it fetches
that model's released checkpoint from the Hub, builds the architecture from its config, and loads
the weights. The repository id and checkpoint name are baked into the package, so there is nothing
to configure by hand.

Forecasting is **zero-shot and probabilistic**: you pass your own series and a horizon, and get back
quantile trajectories. No per-dataset training, no gradient updates.

> **Inputs are plain arrays.** A single 1-D series, a list of 1-D arrays of differing lengths, or a
> 2-D `(n_series, T)` array. Feed values on their **natural scale** — the model normalizes
> internally and reads only the last `context_length` points. **Missing observations are supported**:
> encode them as `np.nan`.

<details open>
<summary><b>Forecasting with EXAONE Finance</b></summary>

```python
import numpy as np
from exaone_forecast.finance import from_pretrained

fc = from_pretrained(device="cuda:0")          # downloads the weights on first use

series = [np.random.randn(500).cumsum() + 100 for _ in range(4)]

q = fc.predict(series, horizon=20)             # (4, 21, 20)  all quantile levels
yhat = fc.point(series, horizon=20)            # (4, 20)      median forecast
lo, hi = fc.interval(series, horizon=20, lower=0.1, upper=0.9)
```

`fc.quantiles` lists the quantile levels in the order they appear on axis 1;
`fc.context_length` and `fc.max_horizon` report the model's input and output limits.

</details>

<details>
<summary><b>Loading a local checkpoint</b></summary>

```python
from exaone_forecast.finance import EXAONEFinanceForecaster

fc = EXAONEFinanceForecaster(ckpt_dir="/path/to/checkpoint", device="cuda:0")
```

The directory must contain `config.json` and `model.safetensors`. Load the checkpoint through this
API rather than `PreTrainedModel.from_pretrained`: the config declares `model_type="t5"` and the
feed-forward blocks use T5 layer names, so `from_pretrained` on transformers ≥ 5 applies a T5 weight
conversion that corrupts them.

</details>

A runnable end-to-end example is in
[`domains/finance/quickstart.py`](domains/finance/quickstart.py).

<br>

## Repository layout

<div style="background-color: rgba(128, 128, 128, 0.1); border-radius: 12px; padding: 12px 24px;">

- `src/exaone_forecast/_api.py` — the forecasting surface every model exposes
- `src/exaone_forecast/_hub.py` · `_loading.py` — shared checkpoint download and loading
- `src/exaone_forecast/finance/` — **EXAONE Finance**: entry point, model definition, forecaster
- `domains/<model>/` — per-model documentation, technical report, and runnable example

</div>

A model owns its architecture outright: everything that defines EXAONE Finance lives under
`finance/`, and nothing above that directory assumes a convolutional encoder, a quantile head, or
any other design choice. A new model joins the family as its own subpackage plus its own `domains/`
page — it shares the download and loading helpers and the calling convention, and nothing else — so
adding one leaves the existing models untouched.

<br>

## License

Two licenses apply, to two different things:

- **The code in this repository** — the `exaone-forecast` inference runtime — is released under the
  [BSD-3-Clause-LG AI Research License](LICENSE), which permits commercial use.
- **The released model weights** are licensed separately under the **EXAONE AI Model License
  Agreement 1.2 - NC**, which limits use to non-commercial research and education. The full terms
  ship with the weights on each model's Hugging Face repository.

Installing this package therefore does not grant commercial rights to the weights it downloads.

Third-party open source components and their licenses are listed in [Notice.md](Notice.md).

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
