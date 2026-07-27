# Adding a model

A model is **one file** in `src/scale_forecasting/models/` that ends with a `register(...)`
call, plus **one import line** in `models/__init__.py`. Nothing else changes — not the
worker, not the engines, not the registry. That is the whole point of the factory: the
system discovers models by name at import time.

## The 4-step checklist

1. **Copy the template.** `docs/model_template.py` → `src/scale_forecasting/models/my_model.py`.
2. **Name it.** Rename the class and set `name = "my_model"` (unique, lowercase, snake_case).
   This string is how users select the model in a run config and in the playground.
3. **Fill in `fit` and `predict`.** See the contract below.
4. **Register it.** Add `my_model,` to the import block in
   `src/scale_forecasting/models/__init__.py`. (The file already ends with `register(...)`;
   this import is what triggers it.)

Then try it immediately, offline:

```bash
python -m scale_forecasting.playground --list           # my_model now appears
python -m scale_forecasting.playground --model my_model --backtest
```

or open `notebooks/model_playground.ipynb`, set `MODEL = "my_model"`, and run the cells.

## The contract every model owes

A model is a `BaseModel` subclass (see `models/base_model.py`). The seams:

| Piece | What it is |
|-------|------------|
| `name` | unique selector string |
| `runtime` | `"python"` (runs in a Spark/Ray cell) or `"bigquery"` (SQL, Arc B) |
| `family` | `"statistical"` \| `"ml"` \| `"deep_learning"` \| `"native"` (metadata only) |
| `supports_exog` | `True` if `fit`/`predict` use the `X` frame |
| `supports_native_intervals` | `True` if you produce your own prediction bounds |
| `fit(y, X)` | fit on one series; `y` is indexed by `ds`, already transformed per config |
| `predict(horizon, X, quantiles)` | return the canonical frame in **original units** |

**`fit(y, X)`** — `y` is one series' target (a `pd.Series` indexed by datetime `ds`), already
transformed per the run config (e.g. `log1p`). `X` is the aligned feature/exog frame, or
`None`. Stash what `predict` needs on `self`. Per-run context is on `self.ctx`
(`freq`, `horizon`, `seed`, `transform`); HPO hyperparameters are on `self.params`. Raise
`ModelError` on a genuinely unfittable series — the worker captures it into an *error cell*
and the batch survives; it never propagates.

**`predict(horizon, X, quantiles)`** — return the canonical prediction frame (CONTRACTS §2.1):
columns `ds, yhat, yhat_lower, yhat_upper, quantiles`, in **original units**, with ordered
bounds (`lower ≤ yhat ≤ upper`). The base class does the assembly for you:

- build a point forecast array,
- turn it into a `{quantile: array}` map — either from your own native bounds, or via
  `self.residual_intervals(mean, quantiles)` if you recorded residuals in `fit` with
  `self._set_residuals(...)`,
- invert the transform with `invert_transform(values, self.ctx.transform)` so values are in
  original units,
- call `self._assemble_frame(ds, qmap)` (use `self._future_index(last_date, horizon)` for the
  dates).

### Rules that keep the product coherent

- **One model, one file.** No model imports another model. Shared machinery goes in a helper
  module (see `_lag_forecaster.py`, used by both tree models).
- **Never read global config.** Everything a model needs arrives via `self.ctx` and
  `self.params`. This is what lets the identical code run local, on Spark, and on Ray (G1).
- **Never hard-code a seasonal period.** Use `seasonality.seasonal_period(self.ctx.freq)` —
  the one shared freq→period source. (pandas 3 spellings: `D`, `W`, `MS`, `ME`, `h`.)
- **Original units out.** The frame `predict` returns must be in original units; invert any
  transform yourself.
- **Optional heavy deps import lazily inside `fit`.** If your model needs an extra (torch,
  xgboost, …), import it *inside* `fit` and raise `ModelError("… install the 'models'
  extra")` on `ImportError`. That way the model still *registers* without the dep, and the
  contract test skips it cleanly instead of breaking the suite. See `xgboost_model.py`.

## What you get for free

- **`python -m scale_forecasting.playground`** and the notebook list and run your model with
  no changes — they read the factory registry.
- **The contract test** (`tests/unit/test_models_contract.py`) is parametrized over every
  registered model, so your model is automatically checked for: canonical frame shape,
  ordered bounds, original-units output, determinism under a fixed seed, and `supports_exog`
  honored. Run `pytest tests/unit/test_models_contract.py -k my_model`.
- **HPO** (optional): implement `search_space(cls, trial)` to expose an Optuna search space;
  it's used only when `hpo.enabled` in the config.
