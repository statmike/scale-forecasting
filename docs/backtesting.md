# Backtesting

Backtesting is how this product decides which model to believe. It fits on history, predicts a
window it was not allowed to see, and writes every one of those predictions down next to the actual
value — so the leaderboard you read afterwards is made of out-of-sample numbers rather than of how
well each model remembers its own training data.

This page is about the *method*: what a fold is here, what is reserved from fitting and why, what
happens to a series too short for the grid you asked for, and what the four schemes actually
measure. For the field-by-field reference — types, defaults, validation rules — see
[configuration_reference.md](./configuration_reference.md#backtest--backtestconfig). For the columns
the results land in, see [output_schemas.md](./output_schemas.md).

Backtesting is **off by default**, because the cheapest possible first run is one full-history fit
per series and no scoring at all. Turn it on with `{"backtest": {"enabled": true}}`. Almost
everything else in the product that has an opinion — hyperparameter search, learned ensembles,
`inverse_error` weighting, the comparable leaderboard — depends on it, and says so when it is
missing.

---

## 1. What a backtest produces, and what it costs

One cell is one `(series, model)` pair. When backtesting is on, that cell does more than fit once:

1. It lays out the folds for this series (`make_folds`).
2. For each fold, it trains on that fold's history and predicts that fold's validation window.
3. It scores the prediction against the actuals and keeps **both** the per-row predictions and the
   per-fold metric panel.
4. Then — separately, and always — it fits once more on the series' whole history and produces the
   forecast you actually ship.

That last step is the one to hold on to. **The shipped forecast is not a fold.** Backtesting scores
a model; it does not produce the forecast. Everything on this page about a series being scored on
fewer folds, or on no folds at all, is about the score. The forecast is written either way.

Two things land in the registry:

| Where | Grain | What it is |
|-------|-------|------------|
| `backtest_oof` | one row per `(series, model, fold, date)` | The out-of-fold predictions themselves — the raw material. This is what the learned ensembler trains on and what `v_model_leaderboard_comparable` is computed from. |
| `forecast_metadata` | one row per cell (`fold_id IS NULL`) | The rolled-up metric panel for that cell, plus how the scoring went: `backtest_status`, `n_folds_achieved`, `backtest_note`, `backtest_refit`, `staleness_gap`. |

An out-of-fold row carries more than a prediction. Every row has `y_true`, the model's own point
forecast before any bias correction (`yhat_raw`), the corrected one (`yhat_adjusted`), whichever of
the two this run selected (`yhat`), a prediction interval (`yhat_lower` / `yhat_upper`), the
`cutoff_date` the model was trained to, and `horizon_step` — the 1-based position within that fold's
horizon, so *"how fast does this model decay as it forecasts further out?"* is a `GROUP BY` rather
than another run.

The intervals are scored for real. `coverage`, `pinball`, `interval_score` and `interval_width` are
computed from the bounds the model actually returned — natively if it has them, from its own
in-sample residual quantiles if it does not — so a model that is right on average but wildly
uncertain cannot hide behind its point forecast.

### What it costs

The honest multiplier is not `n_folds + 1`. A fold trains on *less* history than the final
full-history fit, so counting fits overstates the bill. `fit_rows` reports the training-row count of
every fit a cell performs, and `--feasibility` turns that into a single number:

```bash
python -m scale_forecasting.main --config configs/explode_demo.json --feasibility
```

```
feasibility: 1000 series x 4 models = 4000 cells
  fits: 15640 (19,477,760 training rows) = 3.862 whole-history fits per cell
  3 of 3 folds: 970 series (97.0%) — all folds
  0 of 3 folds: 30 series (3.0%) — UNSCORED
  30 series get no folds at all: they are still fit and forecast, but they contribute nothing
  to any leaderboard (see v_backtest_coverage)
  min_train=180 is within budget: up to 316 still keeps 90% of series at full folds
```

A three-fold backtest is not 3× a plain run — it is **3.862** whole-history fits per cell here,
because each fold's training window is shorter than the final full-history fit and the multiplier is
whatever those shrinking windows add up to. The number comes from the same `make_folds` the workers
use, so it cannot drift from what actually runs. On a 100,000-series panel the difference between
the honest multiplier and `n_folds + 1` is real money.

`--feasibility` implies `--dry-run`, reads the source panel's series lengths, and writes nothing. It
is the one part of planning that cannot be done offline, because how many folds a series achieves is
a fact about your data rather than about your config. Run it before you spend anything.

---

## 2. Laying out the folds

Folds are anchored from the **end** of each series, never from the beginning.

The newest fold validates on the final `horizon` observations. Each earlier fold steps its
validation window back by `step`. With `n_folds: 3, horizon: 28, step: 28` on a 1,460-point daily
series:

```
             fold 0            fold 1            fold 2 (holdout)
             ├──────┤          ├──────┤          ├──────┤
 ...─────────────────────────────────────────────────────────────►
 0                1376    1404    1404    1432  1432    1460
       train ──────────►
       train ────────────────────────────►
       train ──────────────────────────────────────────►
```

Training for fold *k* stops at `val_start − gap`, and the invariant that holds for every fold under
every scheme and every policy is:

```
train_end + gap == val_start
```

At the default `gap` of 0 the two are adjacent: training stops exactly where validation begins, and
no fold has ever seen a value it is scored on.

Where training *starts* is the only thing `scheme` changes about the geometry. `expanding`,
`expanding_frozen` and `expanding_stale` all start at position 0 — every fold sees all the history
it can. `sliding` is the one scheme with a fixed-width window: it starts at `train_end − window`,
where `window` defaults to `min_train`.

A series needs

```
min_train + gap + horizon + (n_folds − 1) · step
```

observations to support the whole grid. Under the defaults that is `180 + 0 + 28 + 2·28 = 264`.

### The embargo is a training-side change

`gap` opens a band of observations between the end of training and the start of validation that are
neither trained on nor scored. It models a forecast issued with a reporting lag: if the last
fortnight of actuals is never in the warehouse when the model runs, then scoring a model that
trained right up to the cutoff flatters it.

The embargo moves the **training end**, not the validation window. That is the part worth
internalising, because the alternative is subtly broken: if `gap` pushed validation later instead,
then fold 1 of a `gap: 14` run would cover a different fortnight than fold 1 of the same config at
`gap: 0`, and the two runs would stop being comparable. As built, they score identical dates and the
difference between them is exactly the cost of the lag.

It does cost history, which is why `gap` appears in the feasibility formula. A larger embargo means
a short series achieves fewer folds.

### Two horizons, and the one the fold really asks for

`data.horizon` is how far the shipped forecast reaches. `backtest.horizon` is how far each fold
predicts before it is scored. They are separate fields on purpose, and setting them apart is legal —
but a mismatch earns a warning, because a model that wins at seven steps is not automatically the
one to trust at twenty-eight.

Under an embargo the model is asked for `gap + horizon` steps and the first `gap` of them are thrown
away, because a forecast is a recursion for most of these models and you cannot reach step 15
without producing steps 1 through 14. Both engines do this the same way, so `horizon_step = 1` means
"the first *scored* point" on the Python path and in BigQuery alike.

---

## 3. The holdout fold fits nothing

This is the invariant the rest of the scoring rests on.

**Fold `n_folds − 1` — the newest one, the one with the smallest step-back — trains nothing.** Not
stacker weights. Not `inverse_error` weights. Not hyperparameters. Everything in this product that
*learns* from the backtest learns from the inner folds; the newest fold is what those learned things
are then judged on.

Without that reservation, a learned ensemble's leaderboard number is an in-sample fit statistic
sitting in the same column as the base models' out-of-sample numbers, and a per-series
hyperparameter search is scored on the exact folds it optimised against. Both would look like wins.

Three mechanisms make it work, and they are worth naming because each one is load-bearing:

- **`holdout_fold_id(cfg)` is a pure function of the config** — `n_folds − 1` — rather than a
  per-series lookup. That is what lets it be used as a join key across a whole panel.
- **Short series keep their original fold numbering.** `make_folds` drops the *oldest* folds a short
  series cannot afford and leaves the survivors' `fold_id`s alone, so every series that achieved any
  fold at all achieved the holdout. If survivors were renumbered 0..k, a short series' fold 0 would
  silently line up against a long series' fold 0 covering completely different dates.
- **The BigQuery-native path numbers folds identically.** `engines.bigquery_sql.fold_plan` derives
  the same ordinals from the same config, so the two runtimes agree on which fold is the holdout
  without comparing dates to find out.

### When the split cannot exist

Two ordinary situations leave nothing to reserve: the run asked for `n_folds: 1`, or a short series
in a ragged panel achieved only the newest fold.

Falling back is the right behaviour — a stacker with no rows to fit on is worse than an optimistic
one — but it has to be **said**, because an in-sample number is indistinguishable from an honest one
once it is sitting in a metric column. So the run records it:

| Where | Column | Values |
|-------|--------|--------|
| `forecast_metadata` (per cell) | `hpo_scoring` | `holdout` / `in_sample` / `NULL` |
| the ensemble's rows | `ensemble_scoring` | `holdout` / `in_sample`, or `NULL` for `mean`/`median`, which fit nothing |
| `run_registry.job_telemetry` | `$.scoring.hpo` | `off` / `holdout` / `in_sample` for the whole run |

`n_folds: 1` is legal and it works. It just means every learned weight and every tuned parameter is
scored on the fold it was chosen by. **`n_folds: 2` is the smallest setting that buys an honest
number.**

### The comparable leaderboard is the holdout fold

`v_model_leaderboard` averages each model's per-cell metrics. That is the right first look, and it
is not a fair fight on a ragged panel: one model's `mean_wape` can be an average over ten folds of
two thousand series while its neighbour's is an average over one fold of two hundred.

`v_model_leaderboard_comparable` fixes the question rather than the answer. Two differences, and
both are the point:

- **It is restricted to the holdout fold** — derived from the rows as `MAX(fold_id) OVER (PARTITION
  BY run_id)`, because a view has no config to read. Every series that achieved any fold achieved
  this one, so the rows being compared cover the same window.
- **The error is pooled, not averaged** — `SUM(|y_true − yhat|) / SUM(|y_true|)` over the whole
  panel at once, so a fleet number is one WAPE of everything rather than the mean of per-series
  WAPEs, where a single near-zero series can dominate the ranking.

Carry `n_series` into any comparison you draw from it. Equal `n_series` across the rows is the
evidence that the models answered the same question; unequal `n_series` is a finding, not a
footnote.

---

## 4. Short series degrade; they never vanish

A panel of real series is ragged. Some of them will not hold the grid you asked for, and there is no
universally right answer to that, because every answer gives something up.

**The one thing that never happens is losing the forecast.** A series too short to score is still
fit and still forecast. This was not always true — a scoring shortfall used to raise, the blanket
cell handler turned it into an error cell, and short history became the single largest error class
in the registry. The fit itself was never in question. `make_folds` now clamps under every policy,
including `error`.

`backtest.short_series` is where you choose what to give up:

| Policy | What it does | What you give up |
|--------|--------------|------------------|
| `adapt` (default) | Scores the series on the folds it *can* support, dropping the oldest. | **Folds.** A short series is ranked on thinner evidence than a long one. |
| `overlap` | Narrows `step` until all `n_folds` fit — the widest step that works, so the least overlap that works. | **Fold independence.** Validation windows share observations, so the per-fold scores are correlated and their mean is more confident than the evidence warrants. |
| `shrink_train` | Lowers `min_train`, as little as the shortfall demands, never below `min_train_floor`. | **Training history.** Early folds fit on less data than you said a model needs. |
| `skip` | Leaves the series out of the backtest entirely. | **The series.** Every series on the leaderboard was then measured on identical geometry. |
| `error` | Refuses the whole run before it starts. | **The run.** Nothing is produced until you change the geometry or the policy. |

All of the reasoning lives in one function, `backtest.resolve_geometry`, and `make_folds` reads its
answer rather than re-deriving it. A policy decides how much evidence a short series contributes; it
never decides where a fold sits relative to its neighbours.

Four behaviours are worth knowing:

- **`min_folds` judges the result, not the starting point.** A rescue policy gets to try first; if
  what it achieved is still below `min_folds`, the series is left unscored rather than ranked on
  evidence too thin to rank it. One fold of five is not a fifth of an answer. The default of `1` is
  exactly the historical behaviour: score anything that supports at least one fold.
- **A rescue is a rescue, not a rewrite.** `overlap` and `shrink_train` change nothing at all for a
  series that was long enough already, so a mixed panel still lays its long series out the way
  `adapt` would.
- **`error` refuses at plan time, not per cell.** It is checked by
  `launch_plan.preflight_short_series` at submit — one aggregation against the warehouse, before
  anything is provisioned — and re-checked by each Python engine's driver, which covers the launch
  paths that do not go through the submit verb (a staged config, an emitted command, a Composer
  task). The refusal names how many series fall short and by how much, because *"some series are
  too short"* is not actionable and *"412 of 5,000 series need 89 more observations"* is.
- **`short_series` is a Python-path policy.** The BigQuery-native models count their folds back from
  one global `MAX(ds)` in SQL, so there is no per-series grid for a policy to adapt; a short series
  there simply contributes fewer scored rows. `error` is the exception, because it is checked
  against the panel before either engine starts.

### Telling the three apart afterwards

A reduced backtest and a full one both produce numbers. An unscored series produces the same NULLs
as a run with backtesting switched off. They support very different conclusions, so the cell records
which one happened:

| `backtest_status` | Meaning |
|-------------------|---------|
| `full` | The geometry the config asked for, exactly. |
| `reduced` | Scored, but **not on the requested geometry**. |
| `unscored` | No folds; still fit, still forecast, contributes nothing to any leaderboard. |
| `failed` | The backtest itself raised; `backtest_note` carries the exception. |
| `NULL` | Backtesting was never asked for. |

`reduced` is wider than "fewer folds". Under `overlap` or `shrink_train` a series can reach the full
fold *count* by spending something else, which produces the otherwise-impossible pair of `reduced`
with `n_folds_achieved` equal to the requested `n_folds` — and that pair is precisely the signal
that the count was bought with overlap or with a shorter training window. `backtest_note` names the
trade in words.

`v_backtest_coverage` is the view that answers *how much of the panel did each model actually get
scored on?* One row per `(run_id, model_type, ensemble_id, backtest_status, n_folds_achieved,
backtest_refit)` with the series count and its share of that model's panel. **Read it beside the
leaderboard: a model whose panel is mostly `reduced` won on an easier question.**

---

## 5. A fold is a date, not an ordinal

`fold_id` is an ordinal within **one series' own plan**, counted back from **that series' last
observation**. On a ragged panel, two series' fold 3 are two different date windows. And the
BigQuery-native path counts back from a single global `MAX(ds)`, so it numbers differently again.

Both of those are correct behaviours, and together they are a trap: rows pair up on a matching
ordinal that stands for different training windows, and the rows that *should* have paired fall out
of the join. Silently. Nothing errors.

So every out-of-fold row also carries **`cutoff_date`** — the last date the model was allowed to
see, the origin it forecast from. That is what identifies a fold when rows are compared across
series or across engines, and it is what the ensemble joins on (`ensembler._fold_key` keys on
`(ts_id, cutoff_date, forecast_date)`, falling back to `fold_id` only for older out-of-fold frames
written before the cutoff was projected on both engines). Two models blend when they saw the same
history and forecast the same date — which is what the cutoff says and the ordinal does not.

The same fact drives the scale-free metrics. MASE and RMSSE divide by the mean step of the
**training** data, so which history goes into the denominator is not a detail; it *is* the number.
`backtest_cell` has always handed each fold its own slice. The two paths that score from a separate
history read — the BigQuery-native engine and the ensemble scorer — used to pass the whole series,
including the very window being scored, which made a native model's MASE and a Python model's MASE
answers to different questions. `backtest.training_window` is the one rule all of them now apply:
every observation at or before the cutoff, narrowed to the last `window` observations under
`sliding`.

---

## 6. Frozen backtesting is two tiers

`scheme` decides what a fold's score is a score *of*. The four answers are different numbers, not
cheaper approximations of one number.

| `scheme` | At each origin | The question it answers |
|----------|----------------|-------------------------|
| `expanding` (default) | A fresh model is fit on all history up to the cutoff. | How good is this model when freshly trained? |
| `sliding` | A fresh model is fit on a fixed-width `window`. | The same, with bounded memory. |
| `expanding_frozen` | **One** fit on the oldest fold's window, then handed the observations that arrived since, parameters held fixed. | What does refitting less often cost me? |
| `expanding_stale` | **One** fit, and the model is never told what happened next. | How fast does this decay if nobody touches it? |

The first two refit. The last two do not, and **they are not two intensities of the same
measurement** — they are two different measurements that happen to share a fit.

### Tier one: re-condition is a genuine forecast from a later origin

Under `expanding_frozen`, the model is fit once on the oldest surviving fold's window. Before each
later fold, it is handed the real observations that arrived between the two origins and asked to
absorb them **without re-estimating its parameters** — a state-space filter updating its state, a
lag-feature model seeing new lags.

So a fold-3 prediction under this scheme is a real forecast issued from fold 3's origin, by a model
that knows everything up to that origin. What it does not have is *refreshed parameters*. That is a
production reality, not a handicap: most teams fit monthly and forecast daily.

Ten of the sixteen Python models have that seam (`supports_recondition`). The other six refit for
that fold and record `backtest_refit = 'unsupported'` on `forecast_metadata` rather than pretending
otherwise. A model that declares the seam can still refuse a particular series at runtime — a filter
can fail to converge on the extension — and that drops the whole cell to `unsupported` and refits
the remaining folds. That slightly over-reports, deliberately: *"some of this cell was frozen"* is
not a claim a leaderboard column can carry, and the honest summary of a cell that fell back partway
is that it is not cleanly frozen.

### Tier two: extrapolate-only measures staleness, not skill

Under `expanding_stale`, the model is fit once and **never told what happened next**. It is walked
forward by telling it only how far the clock moved (plus any exogenous values covering the skipped
span, which are inputs rather than outcomes) and asked to predict.

A fold-3 prediction under this scheme is not a forecast anyone would ship. It is a measurement of
**decay**: how far a two-months-stale model has drifted from reality. A model can post an excellent
`expanding_stale` score by being flat and boring, and a genuinely skilful model that tracks
short-term structure will look worse the further out you push it. **Ranking models by an
`expanding_stale` score ranks them by inertia.**

What it is good for is exactly one thing, and it is a thing nothing else does: every one of the
sixteen models supports it identically. There is no `unsupported` fallback and no seam to differ
over, so it is the one scheme where a cross-model comparison is asking all sixteen the same
question, with nothing varying between them but the model.

### The two never share a leaderboard slice

`scheme` is a run-level config field and it is inside the `run_id` digest. One run is therefore
exactly one scheme, and both leaderboard views group by `run_id`. **A `recondition` number and an
`extrapolate` number cannot appear in the same slice of the same leaderboard, because they cannot
appear in the same run.** Comparing them means comparing two `run_id`s, deliberately, which is a
thing you have to write down rather than something you can do by accident.

Inside a run, the remaining way two rows can be answers to different questions is a model that fell
back. `refit_modes` — a sorted `STRING_AGG(DISTINCT backtest_refit)` on **both** leaderboards — is
how you see it:

| `refit_modes` reads | What it means |
|---------------------|---------------|
| `per_fold` | Every cell refit. The refit schemes, and the correct amount of information for this column to add there. |
| `recondition` | The whole cohort was genuinely frozen. |
| `recondition,unsupported` | **Some of it refit instead.** This ranking is mixing two questions. |
| `extrapolate` | The whole cohort was walked forward blind. |
| `mixed` (on an ensemble row) | The members disagreed. A blend is only as frozen as its least-frozen member. |

### The control arm, and `staleness_gap`

Both frozen schemes also score a **control arm**: the same single fit walked forward blind, on the
same dates, against the same actuals, with its own intervals. It costs a forecast, not a fit. It
lands per row in `backtest_oof.yhat_stale` and is summarised per cell as
`forecast_metadata.staleness_gap`:

```
staleness_gap = loss(blind arm) − loss(primary arm)
```

under the run's `decision_metric`, with both sides restated through `metrics.loss_of` so the sign
means the same thing for `coverage` and `bias` as it does for `wape`. **Positive is the ordinary
reading: never refreshing the model costs you that much accuracy.**

Set **`backtest.control_arm: true`** to get that arm on `expanding` or `sliding` too. This is the
knob that lets the run most people actually make answer *"how much of my accuracy is the refitting
rather than the model?"* — a question that previously required switching to a frozen scheme, which
changes what the primary arm measures, so you got the counterfactual and lost the number you came
for. With the flag, the primary arm is untouched, `backtest_refit` still reads `per_fold`, not one
shipped number moves, and `yhat_stale` is simply filled in beside it. The cost is **one extra fit
per cell** — on the oldest fold's window — plus a forecast per fold.

On `expanding_stale` the flag is **refused, not ignored**: that scheme's primary arm already *is*
the blind model, so the control arm would be the same model twice and the gap would be zero by
construction. Silently dropping it would be worse, because the run would then look like it had
answered the question the flag asks.

The control arm is never scored into the primary metric panel. Everything downstream — arm
selection, calibration, the leaderboard — reads that panel, and a second set of numbers in it would
be picked up as if it were a second model.

### Why freezing is anchored on the oldest fold

Both frozen schemes, and the control arm on the refit schemes, anchor the single fit on
**`folds[0]`** — the oldest surviving fold. Its training window is a prefix of every later fold's,
so a model fit there has seen nothing any fold is scored on. Anchoring on the newest fold would be
cheaper to write and would leak the future into every earlier score.

Because all four schemes use the same anchor, "the blind arm" means one thing across the whole
product.

---

## 7. Reading the results

Start here, in this order.

**Did each model get scored on the same thing?**

```sql
SELECT model_type, backtest_status, n_folds_achieved, backtest_refit, n_series, series_share
FROM `PROJECT.scale_forecasting.v_backtest_coverage`
WHERE run_id = 'YOUR_RUN_ID'
ORDER BY model_type, n_folds_achieved DESC;
```

If one model's panel is mostly `full` and another's is mostly `reduced`, the leaderboard below is
comparing two different questions. If `backtest_refit` shows `unsupported` rows on a frozen run,
those cells refit.

**Which model won, holding the question fixed?**

```sql
SELECT model_type, ensemble_id, n_series, n_points, pooled_wape, refit_modes
FROM `PROJECT.scale_forecasting.v_model_leaderboard_comparable`
WHERE run_id = 'YOUR_RUN_ID'
ORDER BY pooled_wape;
```

Check that `n_series` is equal across the rows before you believe the ordering.

**What is refitting buying me?**

```sql
SELECT model_type, mean_staleness_gap, refit_modes
FROM `PROJECT.scale_forecasting.v_model_leaderboard`
WHERE run_id = 'YOUR_RUN_ID' AND mean_staleness_gap IS NOT NULL
ORDER BY mean_staleness_gap DESC;
```

A large positive gap says this model goes stale fast and is worth refitting often. A gap near zero
says your refit cadence is buying you very little, which is a cost decision you can now make with a
number instead of a habit.

**How fast does accuracy decay across the horizon?**

```sql
SELECT model_type, horizon_step,
       SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape
FROM `PROJECT.scale_forecasting.backtest_oof`
WHERE run_id = 'YOUR_RUN_ID'
GROUP BY model_type, horizon_step
ORDER BY model_type, horizon_step;
```

`horizon_step` exists so this is one query rather than a re-run at a different horizon.

`review.py` and the run-review notebook surface the same facts without SQL — see
[running_and_reviewing.md](./running_and_reviewing.md).

### Choosing a scheme

| If you want to know… | Use | And also set |
|----------------------|-----|--------------|
| Which model is best, freshly trained (the default question) | `expanding` | `control_arm: true` if you also want the staleness counterfactual for free-ish |
| The same, but recent history is more representative than old history | `sliding` | `window` |
| What your real refit cadence costs, for models that can absorb data | `expanding_frozen` | — |
| How fast models decay untouched, compared like with like across all sixteen | `expanding_stale` | — |
| A forecast issued with a known reporting lag | any | `gap` |

### A worked config

```json
{
  "run_name": "backtest_example",
  "data": { "source_table": "source_series_iceberg", "horizon": 28 },
  "models": ["theta", "holtwinters", "xgboost"],
  "backtest": {
    "enabled": true,
    "scheme": "expanding",
    "n_folds": 3,
    "horizon": 28,
    "step": 28,
    "min_train": 180,
    "gap": 0,
    "short_series": "adapt",
    "min_folds": 2,
    "control_arm": true,
    "decision_metric": "wape"
  }
}
```

Three folds, so one is reserved and two are available to fit on. `min_folds: 2` says a series scored
on a single fold is not worth ranking. `control_arm: true` buys the never-refreshed counterfactual
for one extra fit per cell. Run `--feasibility` against your own panel before committing to
`min_train: 180`.

---

## Related pages

- [configuration_reference.md](./configuration_reference.md#backtest--backtestconfig) — every
  `backtest` field, its type, default and validation rule.
- [output_schemas.md](./output_schemas.md) — `backtest_oof`, `forecast_metadata`, and the views.
- [running_and_reviewing.md](./running_and_reviewing.md) — `--feasibility`, `review.py`, the
  monitoring surface.
- [API: backtest](./api/backtest.md) — the module's own docstrings, generated from source.
