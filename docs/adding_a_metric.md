# Adding a metric

A metric is **one file** in `src/scale_forecasting/metrics/` that ends with a `register(...)`
call, plus **one import line** and **one entry in `METRIC_NAMES`** in `metrics/__init__.py`.
Nothing else changes — not the worker, not the engines, not the DDL, not the write path, not
the views. The metric panel is generated from that one list, so a metric you add is a real
column with a real value on the next run.

This mirrors [adding a model](./adding_a_model.md) deliberately: same factory pattern, same
one-thing-one-file rule, same `register(...)` at the bottom of the file.

## The 4-step checklist

1. **Copy the template.** `docs/metric_template.py` →
   `src/scale_forecasting/metrics/my_metric.py`.
2. **Name it.** Rename the class and set `name = "my_metric"` (unique, lowercase, snake_case).
   This string becomes a **BigQuery column name**, so it has to be a bare identifier —
   `register()` rejects anything that isn't.
3. **Declare and compute.** Set `direction`, set any `needs_*` flags, and fill in
   `compute(ctx)`. See the contract below.
4. **Register it.** Add `my_metric,` to the import block in
   `src/scale_forecasting/metrics/__init__.py`, and add `"my_metric"` to `METRIC_NAMES` at the
   position you want the column to sit in. (The import block is alphabetical and does *not* set
   the order; `METRIC_NAMES` does — see [Why the order is written twice](#why-the-order-is-written-twice).)

Then try it immediately, offline:

```python
from scale_forecasting.metrics import compute_metrics, list_metrics

list_metrics()                                     # "my_metric" now appears
compute_metrics([10.0, 12.0, 11.0], [9.0, 13.0, 11.5])["my_metric"]
```

Your metric is now selectable as the run's decision metric — `"backtest": {"decision_metric":
"my_metric"}` — with no change to `config.py`.

## What you get for free

Four things that look like they'd need editing don't, because all four are generated from
`METRIC_NAMES`:

| Derived thing | Where | What happens |
|---|---|---|
| The `forecast_metadata` column | `registry/ddl.py` | A `my_metric FLOAT64` line is rendered into the `CREATE TABLE`. |
| The migration for existing deployments | `registry/ddl.render_migrations` | An `ADD COLUMN IF NOT EXISTS my_metric FLOAT64` is emitted, and `ensure_tables` runs it **at the start of every run** — so an existing deployment gains the column with no manual step. |
| The Storage Write API field spec | `registry/write_api.py` | The proto descriptor picks the column up, so rows carry the value. |
| The leaderboard aggregate projection | `registry/reads.py`, `review.py` | `mean_my_metric` / `p10_` / `p50_` / `p90_` appear in the per-run aggregate and in `review_run`. |

Also free: **every engine computes it.** There is exactly one function that turns arrays into
metric numbers — `metrics.compute_metrics` — and the Spark worker, the Ray worker, the ensemble
scorer, and the BigQuery-native path all call it. (The native path pulls each fold's eval frame
back to the driver and scores it in Python precisely so that a `wape` from `arima_plus` and a
`wape` from `xgboost` are the same quantity.) There is no SQL implementation of any metric
anywhere, so a Python metric works for all four families the day you write it.

**Two things are *not* generated**, by design:

- `v_model_leaderboard` carries a fixed headline pair, `mean_wape` and `mean_mae`. It is a
  glanceable summary, not the panel. Your metric is on `forecast_metadata` and in the
  aggregate projection above; widening the view is a deliberate edit, not an automatic one.
- The metric tables in [`configuration_reference.md`](./configuration_reference.md#decision_metric)
  and [`output_schemas.md`](./output_schemas.md) are prose. Add a row.

## The contract every metric owes

A metric is a `BaseMetric` subclass (see `metrics/base_metric.py`). The seams:

| Piece | What it is |
|-------|------------|
| `name` | unique, lowercase, snake_case, **valid SQL identifier** — it becomes a column |
| `direction` | `"lower"` \| `"higher"` \| `"zero"` |
| `needs_intervals` | `True` if you read `ctx.lower` / `ctx.upper` |
| `needs_train_history` | `True` if you read `ctx.y_train` |
| `needs_seasonal_period` | `True` if you read `ctx.seasonal_period` |
| `compute(ctx)` | the value for one scored window, or NaN — **never raises** |

**`direction` is not cosmetic.** `loss_of` reads it to turn any metric into a comparable loss,
and three separate places consume that loss: `inverse_error` ensemble weighting, HPO's
objective, and `prune_threshold`. Three answers, not two:

- `"lower"` — an error. Smaller is better, zero is perfect. Most of the panel.
- `"higher"` — a fraction of successes. Larger is better, one is perfect. Only `coverage`.
- `"zero"` — a signed quantity where **either** sign is a fault. Only `bias`.

Declaring `"lower"` on a higher-is-better metric doesn't fail; it silently inverts every one of
those three consumers, which is exactly the bug the `direction` field exists to prevent.

**`compute(ctx)` sees `ctx` and nothing else.** `MetricContext` carries `y_true` and `yhat`
(equal-length float arrays, never empty), the precomputed `err = yhat − y_true` and `abs_err`,
and the optional `y_train`, `lower`, `upper`, `seasonal_period`. A metric never reads global
config — that is what lets the identical code score a Spark cell, a Ray cell, an ensemble's
out-of-fold predictions and a BigQuery-native fold.

**Return NaN, don't raise.** A metric that can't be computed for one cell must not sink the
batch, so guard every divisor and every optional input and return `float("nan")`. The only two
exceptions in the whole layer live in `MetricContext.from_arrays` — mismatched lengths and an
empty window — because those are caller bugs, not undefined metrics.

**Build on another metric with `ctx.value(name)`.** The panel is not independent numbers:
`mase` divides `mae`, `rmsse` divides `rmse`, `mase_seasonal` divides `mae` again. Ask the
context for the number by name rather than recomputing it — the context memoises it across the
whole panel, so the arithmetic happens once and the math for `mae` stays in `mae.py` where a
reader will look for it.

### A metric is computed by the framework, never supplied by a model

This is the rule the leaderboard rests on. A model returns a *forecast*; the framework scores
it. No model may hand back a value for a registered metric name, because the moment one does,
a column that still looks uniform stops meaning one thing — `wape` would be "whatever each
library calls wape" rather than a single function of `(y_true, yhat)`.

Numbers only a fitting library can produce — AIC, a log-likelihood, a training loss, an
early-stopping iteration, a selected `(p,d,q)` — are **fit diagnostics**, not metrics. They
exist for some models and not others and are not comparable across them, so they do not belong
in the panel. They get their own JSON channel beside `best_params`.

### Why the order is written twice

The import block in `metrics/__init__.py` is alphabetical; `METRIC_NAMES` is in panel order.
That looks redundant and isn't. Registry insertion order *is* import order, and import order is
owned by an import sorter — so deriving the column order from registration would let a
formatter silently reorder the columns of a BigQuery table. `METRIC_NAMES` is therefore
hand-written, and an import-time check raises `ConfigError` if it and the registry ever
disagree. Forget the `METRIC_NAMES` line and the package won't import, which is the failure you
want.

## Rules that keep the product coherent

- **One metric, one file.** Shared helpers go in a private module (see `_intervals.py`, which
  holds the interval quantiles the four interval metrics agree on).
- **Never read global config.** Everything arrives on `ctx`.
- **Never raise from `compute`.** NaN is the answer for "undefined here".
- **The `needs_*` flags are declarative, not enforcement.** They exist so config validation can
  say at plan time that a chosen `decision_metric` will be NaN for every cell of this run.
  `compute` must still be NaN-safe on its own; don't rely on the flags to gate anything.
- **Don't reuse a name that a model's `params()` can emit.** The contract test checks this.

## What the tests give you

`tests/unit/test_metrics_contract.py` is parametrized over every registered metric, so yours is
automatically checked for: a unique valid-identifier name, a valid `direction`, a float-or-NaN
return on degenerate input (zeros, constants, a single point, missing optional inputs), and
that `compute` never raises. Run `pytest tests/unit/test_metrics_contract.py -k my_metric`.
