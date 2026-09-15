# Configuration reference

One run is **one JSON config**. It is validated into a frozen `RunConfig`
(`src/scale_forecasting/config.py`), logged verbatim to the run registry, and *is* the experiment
record — behavior changes come from this file, not code edits. This page documents every
field.

Two invariants apply everywhere:

- **Immutable + strict.** Every section is frozen and `extra="forbid"` — an unknown key is a
  validation error, not a silent no-op. A typo fails fast with a clear message.
- **Config-derived identity.** The `run_id` is a digest of the config (`make_run_id`), so re-running
  the same config is idempotent at the logical level. Give each execution a fresh `run_name` (the
  notebooks timestamp it) if you want a distinct run.

Load + validate a file with `load_config(path)`; every failure mode (missing file, bad JSON, invalid
schema) surfaces as a single `ConfigError`.

## Top level — `RunConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `run_name` | `str` | *required* | Human name for the run. |
| `data` | `DataConfig` | *required* | Where the series come from and their shape. |
| `python_runtime` | `"spark"` \| `"ray"` | `"spark"` | Run-level **default** runtime for the Python model families; each family can override it (see below). |
| `models` | `list[str]` | *required* (≥1) | Model names to run (see `playground --list`). |
| `model_params` | `dict[str, dict[str, …]]` | `{}` | Per-model hyperparameters, keyed by model name — see below. |
| `features` | `FeaturesConfig` | `{}` | Optional feature engineering. |
| `backtest` | `BacktestConfig` | `{}` | Time-series cross-validation. |
| `output` | `OutputConfig` | `{}` | What the shipped `yhat` means — see below. |
| `hpo` | `HpoConfig` | `{}` | Hyperparameter optimization. |
| `ensemble` | `EnsembleConfig` | `{}` | Consensus across base models. |
| `compute` | `ComputeConfig` | `{}` | Runtime scale + cost guardrails. |

**Cross-field rules** (enforced after parsing):

- Duplicate entries in `models` → error.
- `hpo.enabled` requires `backtest.enabled` → error otherwise (HPO tunes on folds).
- `ensemble.enabled` without `backtest.enabled` → the **learned** strategies (`nnls`/`ridge`/`xgb`)
  are dropped with a warning (they need OOF); calculated strategies remain. Not an error.

**`python_runtime` — the run-level default runtime for the Python model families** (the native family
always runs in parallel in BigQuery, regardless of this choice):

- `spark` (default) — Dataproc Serverless. The **100k CPU workhorse**; it fans out one task per
  `(series, model)` cell (series cross-joined with the family's models), so a family's job finishes in
  ~its slowest cell.
- `ray` — Ray on Vertex AI. Its reason to exist is **fractional-GPU packing** for NeuralProphet (many
  series share one T4); the Ray `compute` knobs apply.

A run resolves its models into **one job per family** (`statistical` / `ml` / `deep_learning`, plus
`native` in BigQuery), all running in parallel under one `run_id`. Each Python family runs on
`python_runtime` unless it is overridden **per family** via `compute.families` (below) — so one run
can put its statistical family on Spark and its deep-learning family on Ray. See the DAG model in
[architecture.md](./architecture.md).

## `data` — `DataConfig`

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `source_table` | `str` | *required* | — | Source table for series (e.g. `source_series_iceberg` / `source_series_native`). |
| `ts_id_col` | `str` | `"ts_id"` | — | Series-id column. |
| `date_col` | `str` | `"ds"` | — | Date column. |
| `target_col` | `str` | `"y"` | — | Target column. |
| `freq` | `str` | `"D"` | one of `D W MS ME h` | Series frequency (pandas ≥2.2 spellings). Sets the seasonal period used by the models — `D`=daily (period 7), `W`=weekly (52), `MS`/`ME`=month start/end (12), `h`=hourly (24). An unsupported freq is rejected at validation. |
| `horizon` | `int` | `28` | `> 0` | Forecast horizon (steps). |
| `series_limit` | `int` \| `null` | `null` | `> 0` when set | `null` = all series; an int subsets the shipped data (demo small → scale large). |

## `features` — `FeaturesConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `holidays` | `list[str]` | `[]` | Holiday country codes to add (e.g. `["US"]`). |
| `transform` | `"none"` \| `"log1p"` \| `"boxcox"` | `"none"` | Target transform, inverted on output. |
| `exog` | `list[str]` | `[]` | Exogenous driver columns — a **started-but-unexampled** seam: consumed by `sarimax`/`ucm`/`prophet`/`lightgbm`/`xgboost`, but the shipped source is univariate (bring your own table with these columns to use it). |
| `lags` | `list[int]` | `[]` | Lag features. |
| `fourier` | `bool` | `false` | Fourier seasonality terms. |
| `level_shift` | `bool` | `false` | Detect one abrupt regime change and add it as a `level_shift` step dummy. |

**What each option produces** ([`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py)):

- **`transform`** — reshapes the target before fitting and inverts it on output:
  - `none` — identity (no constraint on `y`).
  - `log1p` — `log(1+y)` forward, `expm1` back. Tames multiplicative growth / right-skew. **Requires
    `y >= -1`.** Stateless.
  - `boxcox` — a **power transform** whose λ is fit **per series** by maximum likelihood; the *same* λ
    is reused across backtest folds and the final fit. Strongest variance-stabilizer of the three.
    **Requires strictly positive `y` (`y > 0`).**
- **`holidays`** — adds a single `is_holiday` flag (1.0 on holiday dates) from the `holidays` package
  for each ISO country code (e.g. `["US", "GB"]`). The same calendar feeds the BigQuery-native models,
  so holiday handling matches across runtimes. An unknown code fails fast.
- **`lags`** — for each integer `L`, adds a `lag_{L}` column (the target shifted back `L` steps).
  Gives the ML models (`lightgbm`/`xgboost`) autoregressive signal. Values must be positive.
- **`fourier`** — adds sine/cosine **yearly** seasonality terms (order 3 → 6 columns). Smooth periodic
  signal for the regression-based models.
- **`exog`** — passes named driver columns straight from the source table through to the models that
  accept exogenous regressors. See the univariate-shipped-data caveat above.
- **`level_shift`** — detects a single abrupt **regime change** in the series and adds one
  `level_shift` column: `0` before the changepoint, `1` from it onward, and `1` across the whole
  forecast horizon. It is a **step, not a spike** — that persistence is exactly what distinguishes a
  level shift from an outlier, and it lets a regression model absorb the jump as one coefficient
  instead of fitting the average of two regimes and staying biased for the entire horizon. Detection
  is a single-changepoint scan standardized by a robust (MAD-based) noise estimate, accepted only
  above 3σ; below that the column is all zeros, because a spurious regressor on one series in ten
  thousand is a bad forecast nobody reviews. Worth turning on when your history contains
  re-baselinings, store openings/closings, or a unit-of-measure change — the shipped example data
  contains them by construction (`data_gen.generator` plants one per series with archetype-specific
  probability).

**How these features are valued over the forecast horizon.** A model is fit on history and then
asked to predict dates it has never seen, so the same feature columns have to exist for those
dates too (`features.build_future_features`). Most of them are a deterministic function of the
date and are therefore **recomputed exactly** at the future dates — `is_holiday` reflects the
holidays that actually fall in the horizon, and the Fourier terms continue the real seasonal
phase. `level_shift` is carried forward as `1`. Configured `lag_L` columns are genuine
observations for the first `L` steps and then hold the last observed level.

The one exception is **`exog`**, which is genuinely unknown until the future arrives: those
columns fall back to the most recent `horizon` observed rows, so an exog-driven forecast is
*indicative* rather than authoritative. To get a real forward-looking exog path, extend your
source table past the target cutoff with the driver values (a price plan, a promo calendar, a
published index) — the read picks them up with no config change.

## `backtest` — `BacktestConfig`

Off by default (cheapest first run). Turn it on to get an OOF metric panel — and it's a prerequisite
for HPO and learned ensembles.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn backtesting on. |
| `scheme` | `"expanding"` \| `"sliding"` \| `"expanding_frozen"` \| `"expanding_stale"` | `"expanding"` | — | How the model is carried between fold origins. See [Backtest schemes](#backtest-schemes) below. |
| `n_folds` | `int` | `3` | `≥ 1` | Number of folds. |
| `horizon` | `int` | `28` | `> 0` | Per-fold forecast horizon. Linked to `data.horizon` — see [Two horizons](#two-horizons) below. |
| `step` | `int` | `28` | `> 0` | Step between folds. |
| `min_train` | `int` | `180` | `> 0` | Minimum training length. |
| `decision_metric` | see below | `"wape"` | — | Metric folds are judged on. |
| `short_series` | `"adapt"` \| `"overlap"` \| `"shrink_train"` \| `"skip"` \| `"error"` | `"adapt"` | — | What a series too short for the fold grid gets. See [Short series](#short-series) below. |
| `min_folds` | `int` | `1` | `≥ 1`, `≤ n_folds` | Fewer achievable folds than this and the series is left unscored rather than weakly scored. |
| `min_train_floor` | `int \| null` | `null` | `> 0` | The hard training-length minimum `shrink_train` may not cross. **Required with that policy, and rejected without it.** |
| `gap` | `int` | `0` | `≥ 0` | The embargo: observations discarded between train and validation, for a known reporting lag. See [The embargo](#the-embargo-gap) below. |
| `window` | `int \| null` | `null` | `> 0` | A fixed `sliding` training width, decoupled from `min_train`. Defaults to `min_train`. |
| `control_arm` | `bool` | `false` | rejected on `expanding_stale` | Also score a blind, never-refreshed arm on a refit scheme. See [The control arm](#the-control-arm) below. |

This section is the field reference. For the reasoning behind it — how folds are anchored, why the
newest fold fits nothing, what a short series loses, and what each scheme actually measures — see
**[backtesting.md](./backtesting.md)**.

**`scheme` — how the training window moves** ([`backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/backtest.py)).
Folds are anchored from the **end** of each series: the latest fold validates on the final `horizon`
points, and each earlier fold steps its validation window back by `step`. The two schemes differ only
in where training *starts*:

- **`expanding`** (default) — training uses **all history** from the series start up to each fold's
  validation point. The train window grows fold to fold. Best default: every fold sees maximum
  history.
- **`sliding`** — training uses a **fixed-width** window of the last `window` observations
  immediately before each fold, and `window` defaults to `min_train`. Older history is dropped. Use
  when the series' behavior drifts and recent history is more representative than old history.

`n_folds`, `horizon`, `step`, `min_train` and `gap` lay the folds out together, and a series needs at
least `min_train + gap + horizon + (n_folds−1)·step` observations to be scored on *all* of them.

<a id="two-horizons"></a>
### Two horizons

A config carries two of them and they are easy to confuse. `data.horizon` is how far the **shipped
forecast** reaches. `backtest.horizon` is how far **each fold** predicts before it is scored. They
are separate fields on purpose — you can score a model over seven steps while shipping twenty-eight
— and setting them apart is legal on every engine.

**A mismatch earns a warning**, in either direction, because the leaderboard then ranks models over
a horizon the run does not deliver, and a model that wins at seven steps is not automatically the
one to trust at twenty-eight. The warning names both fields; nothing is rewritten, because a config
that repaired itself would move its own `run_id` and land in the registry describing a run you did
not ask for.

**How many steps a fold really asks for is `backtest.gap + backtest.horizon`,** not
`backtest.horizon` alone. A fold's model stops training at its cutoff, and with an embargo the
scored window starts `gap` observations later — so the model has to forecast *across* the embargo
before it reaches anything that gets scored, and the first `gap` values are thrown away. This is
the number that sizes everything: the Python engines are handed it as their horizon, and each
BigQuery-native fold model is created with `OPTIONS(horizon = gap + backtest.horizon)` while the
final model stays at `data.horizon`. BigQuery bakes that number into the trained model and rejects
an `ML.FORECAST` asking for more, so each model is created for exactly what its own forecast asks.
(`timesfm` is the exception that needs no accounting: `AI.FORECAST` has no training step, so there
is no ceiling to fit under.)

**A shorter series is scored on fewer folds; it never loses its forecast.** Backtesting scores a
model — it does not produce the forecast — so a scoring shortfall costs only the score. The fold
grid shrinks to whatever the series supports, dropping the **oldest** folds first (so every series
is scored on the most recent window it can reach) and keeping the survivors' original `fold_id`s, so
a fold is never renumbered by the series that happens to be short. A series too short for even one
fold is fit and forecast unscored.

`fold_id` is an ordinal within one series' own plan, and it is counted back from that series' last
observation — so on a ragged panel two series' fold 3 are two different date windows, and the
BigQuery-native path (which counts back from one global `MAX(ds)`) numbers differently again. Each
out-of-fold row therefore also carries `cutoff_date`, the last date the model was allowed to see.
That is what identifies a fold when rows have to be compared across series or across engines, and
it is what the ensemble joins on — see
[output_schemas.md](./output_schemas.md#which-fold-is-this-really).

**The newest fold is reserved from every fit.** Fold `n_folds − 1` — the one with the smallest
step-back — trains no stacker, sets no `inverse_error` weight, and is never in a hyperparameter
search's objective. It is still *scored*, like every other fold; it is simply not *fit on*, so the
things this run learns are judged on a window they never saw. That is why `n_folds: 1` is worth a
second thought before you pick it: it is legal, it works, and it leaves nothing to reserve, so
every learned weight and every tuned parameter is scored on the fold it was chosen by. The run says
so rather than hiding it — `ensemble_scoring` / `hpo_scoring` read `in_sample`, and the run header
carries the same claim under `job_telemetry.$.scoring`. `n_folds: 2` is the smallest setting that
buys an honest number.

Three columns on `forecast_metadata` record how the scoring went, separately from how the cell went:

| Column | Meaning |
|--------|---------|
| `backtest_status` | `full` \| `reduced` \| `unscored` \| `failed`, or `NULL` when backtesting was never asked for. |
| `n_folds_achieved` | Folds actually scored. **This is the column that makes a leaderboard readable across a ragged panel** — two series with the same WAPE are not comparable if one was scored on five folds and the other on one. |
| `backtest_note` | Why it was not `full`: the shortfall arithmetic, or the exception. |

All three `NULL` is the one case where a `NULL` metric panel is not a shortfall. Without them, a
reduced backtest and a full one look identical through the metric columns, and an unscored series
looks exactly like a run with backtesting switched off.

(Two earlier versions of this paragraph were wrong in opposite directions: one said a short series
was "skipped for backtesting", which was never true; the correction said it "fails its cell", which
was true at the time and is the behaviour this change removed.)

Features are built once and a **fresh** model is fit per fold, so no state leaks across folds and
`train_end + gap == val_start` always (no leakage; at the default `gap` of 0 the two are adjacent).

<a id="the-embargo-gap"></a>
**`gap` — the embargo.** Raising `gap` opens a band of observations between the end of training and
the start of validation that are neither trained on nor scored. That is how you measure a forecast
issued with a reporting lag: if the last fortnight of actuals is never in the warehouse when the
model runs, then scoring a model that trained right up to the cutoff flatters it. Set `gap: 14` and
each fold's model stops learning fourteen days earlier.

Two things about how it is applied are worth knowing, because they are what make the numbers usable:

- **The embargo moves the training end, not the validation window.** A `gap: 14` run scores exactly
  the same dates as the same config at `gap: 0`, so the two runs are directly comparable and the
  difference between them is the cost of the lag. The alternative — pushing validation later —
  would have changed both the model and the test set at once.
- **It costs history.** Each fold gives up `gap` observations it would otherwise have trained on,
  which is why `gap` is in the feasibility formula above: a short series achieves fewer folds at a
  larger embargo.

The BigQuery-native path mirrors this exactly. It forecasts `gap + horizon` points from its cutoff
and then discards the first `gap` of them before joining to actuals, so `horizon_step` means the
same thing on both engines: position 1 is the first *scored* point, not the first point the model
emitted.

<a id="short-series"></a>
### Short series

A panel of real series is ragged, so some of them will not hold the fold grid you asked for. There
is no universally right answer to that, because every answer gives something up. `short_series`
is where you say which thing you would rather lose, and the five policies are listed here in the
order of how much they cost you:

| `short_series` | What it does for a series too short for the grid | What you give up |
|----------------|--------------------------------------------------|------------------|
| `adapt` (default) | Scores it on the folds it *can* support, dropping the oldest ones. | **Folds.** A short series is ranked on thinner evidence than a long one. |
| `overlap` | Narrows `step` until all `n_folds` fit — the widest step that works, so the least overlap that works. | **Fold independence.** The validation windows share observations, so the fold scores are no longer independent evidence. |
| `shrink_train` | Lowers `min_train` — as little as the shortfall demands — until all `n_folds` fit, never below `min_train_floor`. | **Training history.** Early folds are fit on less data than you said a model needs. |
| `skip` | Leaves the series out of the backtest entirely. | **The series.** It is still fit and forecast; it just never appears on a leaderboard. |
| `error` | Refuses the whole run before it starts. | **The run.** Nothing is produced until you change the geometry or the policy. |

Three things are worth knowing about how these behave together.

- **`min_folds` is the give-up floor, and it judges the result, not the starting point.** A rescue
  policy gets to try first; if what it achieved is still below `min_folds`, the series is left
  unscored rather than ranked on evidence too thin to rank it. The default of `1` is exactly
  today's behaviour — score anything that supports at least one fold.
- **A rescue is a rescue, not a rewrite.** `overlap` and `shrink_train` change nothing at all for a
  series that was long enough already, so a panel of mixed lengths still lays its long series out
  the way `adapt` would.
- **Nothing here can cost you a forecast.** Backtesting scores a model; it does not produce the
  forecast. Every policy except `error` still fits and forecasts the series, and `error` refuses
  at plan time — before any cell exists — rather than failing cells one at a time. A series that
  ends up unscored gets `backtest_status = 'unscored'` and a `backtest_note` naming the policy that
  left it that way; its forecast is written either way.

When a policy adjusts the geometry, the cell's `backtest_status` is `reduced` and its
`backtest_note` names the policy and the trade. `reduced` means "not the geometry the config asked
for", so `overlap` and `shrink_train` produce the otherwise-impossible pair of `reduced` with
`n_folds_achieved` equal to the requested `n_folds` — the signal that the full fold count was
bought with something other than folds.

**`short_series` is a Python-path policy.** The BigQuery-native models count their folds back from
a single global `MAX(ds)` in SQL, so there is no per-series grid for a policy to adapt. `error` is
the exception, because it is checked against the panel at submit time rather than per cell — one
aggregation before anything is provisioned — so it stops a native run too. Each Python engine
re-checks it on the panel it read, which is what covers the launch paths that do not go through
the submit verb: a staged config run from an emitted command, or a Composer task.

### Backtest schemes

`scheme` decides what a fold's score is a score *of*. The four answers are different numbers, not
cheaper approximations of one number, so pick the one that matches the question you are asking.

| `scheme` | What happens at each origin | The question it answers |
|----------|-----------------------------|-------------------------|
| `expanding` (default) | A fresh model is fit on all history up to the cutoff. | How good is this model when freshly trained? |
| `sliding` | A fresh model is fit on a fixed-width `window` (default `min_train`). | Same, but with a bounded memory. |
| `expanding_frozen` | One fit on the oldest fold's window, then handed the observations that arrived since, parameters held fixed. | What does refitting less often cost me? |
| `expanding_stale` | One fit, and the model is never told what happened next. | How fast does this decay if nobody touches it? |

Fold *geometry* is identical across all four except `sliding`, which is the only one with a
fixed-width training window. What changes is how the model is carried between origins.

Ten of the sixteen Python models can absorb a new observation without re-estimating, so they can
answer `expanding_frozen`; the rest refit for that fold and record `backtest_refit = 'unsupported'`
on `forecast_metadata` rather than pretending otherwise. Every model answers `expanding_stale`,
which is what makes it the scheme where a cross-model leaderboard compares like with like. The
BigQuery-native models always report `per_fold` — `CREATE MODEL` is the only way to fit them.

An ensemble row inherits `backtest_refit` from the members its blend was built from: their mode
when they agree, and `mixed` when they do not. A blend is only as frozen as its least-frozen
member, and `mixed` is how you see that a consensus sitting on an `expanding_frozen` leaderboard
is not comparable to the single-model rows around it.

<a id="the-control-arm"></a>
### The control arm — what is the refitting buying you?

Both frozen schemes additionally score a **control arm**: the same fit walked forward blind, on the
same dates. It costs a forecast, not a fit. It lands in `backtest_oof.yhat_stale` per row, and is
summarised per cell as `forecast_metadata.staleness_gap` — the blind arm's loss minus the primary
arm's, under this run's `decision_metric`, positive when never refreshing the model hurts.

Set **`backtest.control_arm: true`** to get that same second arm on `expanding` or `sliding`. The
question — *how much of my accuracy is the refitting rather than the model?* — is one the default
scheme could not answer before: asking it meant switching to a frozen scheme, which changes what
the primary arm measures, so you got the counterfactual and lost the number you came for. With the
flag, the primary arm is untouched (`backtest_refit` still reads `per_fold`, and not one shipped
number moves) and `yhat_stale` is simply filled in beside it.

It costs **one extra fit per cell**, on the oldest fold's window, plus a forecast per fold — not a
fit per fold — which is what makes it affordable on the path most runs take. A model with no blind
seam keeps its scheme and leaves the column NULL. On `expanding_stale` the flag is **refused**, not
ignored: that scheme's primary arm already *is* the blind model, so the control arm would be the
same model twice and the gap would be zero by construction.

### The fields that arrived ahead of their code

`short_series`, `min_folds`, `min_train_floor`, `gap` and `window` — along with the frozen schemes
and `model_params` — were all added to the schema in a single commit, before anything read them.
That was deliberate. A new config field moves every `run_id` that has ever been recorded, so
landing them together cost one identity break instead of seven.

**All of them are honoured now**, and they are documented above alongside every other field; there
is no longer an inert corner of this schema. What survives from that decision is the property that
made it worth making: setting any one of them changes your `run_id`.
`tests/unit/test_declared_ahead_fields.py` keeps watching for that, because a field that quietly
left the digest would have to be paid for a second time to put it back.

**`decision_metric` — what folds are judged on** (definitions in
[`metrics.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics.py); `err = yhat − y_true`). This single choice drives
fold selection, HPO's objective, `inverse_error` weighting, and `prune_threshold`.

| Metric | Definition | Notes |
|--------|-----------|-------|
| `mae` | mean(\|err\|) | Mean absolute error. Scale-dependent. |
| `rmse` | √mean(err²) | Penalizes large misses more than `mae`. |
| `mse` | mean(err²) | Squared error; `rmse` without the root. |
| `mape` | mean(\|err\| / \|y_true\|) | % error. **NaN if any `y_true == 0`.** |
| `smape` | mean(2\|err\| / (\|y_true\|+\|yhat\|)) | Symmetric %; bounded, handles zeros gracefully. |
| `wape` | Σ\|err\| / Σ\|y_true\| | Weighted absolute % error — the scale-safe default. NaN only if the series sums to 0. |
| `mase` | mae / mae(naïve-1-step) | Scaled vs. a naïve forecast; **needs training history**. <1 beats naïve. The canonical Hyndman–Koehler definition, whose naïve is always the one-step one — `mase_seasonal` below is an *additional* metric, not a correction to this one. |
| `rmsse` | rmse / rmse(naïve-1-step) | Squared analog of `mase`; **needs training history**. |
| `bias` | mean(err) | Mean error — sign shows over/under-forecast. HPO minimizes \|bias\|. |
| `coverage` | fraction of `y_true` inside [lower, upper] | **Needs prediction intervals**; want it near the nominal level. |
| `pinball` | avg quantile loss at the 0.1 / 0.9 bounds | **Needs prediction intervals**; scores interval sharpness+calibration. |
| `mase_seasonal` | mae / mae(seasonal naïve) | `mase` against `y_{t−m}` instead of `y_{t−1}`, where *m* is the `freq`'s cycle (7 daily, 12 monthly, 24 hourly). **Needs training history longer than one cycle.** |
| `maape` | mean(arctan(\|err\| / \|y_true\|)) | `mape` that survives zeros — a zero actual contributes π/2 instead of NaN-ing the window. Range [0, π/2]. |
| `interval_score` | mean Winkler score at α = 0.2 | **Needs prediction intervals**; width plus a penalty for each miss. Lower is better. |
| `interval_width` | mean(upper − lower) | **Needs prediction intervals**; sharpness only, in the units of the series. |

Pick `wape` (default) or `smape` for a robust scale-independent choice; `maape` instead of `mape`
on intermittent-demand series that hit zero; `mase`/`rmsse` to beat a naïve baseline, or
`mase_seasonal` when the series is strongly seasonal and the one-step naïve is too easy to beat.
The four interval metrics — `coverage`, `pinball`, `interval_score`, `interval_width` — only mean
something when you care about the prediction bands, and ensemble OOF has no intervals, so all four
read NaN for ensembles.

**What "needs training history" means for `mase`, `rmsse` and `mase_seasonal`.** All three divide
the error by the average step of a naïve forecast over the training data, so which history goes in
decides the number. Every engine uses the same rule: **the fold's own training window** — the
observations at or before that fold's `cutoff_date`, and under `backtest.scheme: sliding` only the
last `window` (default `min_train`) of them. The window the fold is *scored* on is never in its own denominator, so a
`mase` from a Python model and a `mase` from a BigQuery-native model for the same series are
answers to the same question and can sit in the same leaderboard column. An ensemble is scaled the
same way, at the cutoff its blended rows carry. One consequence worth knowing: because each fold
has a different training window, the per-fold `mase` values a run rolls up are each scaled slightly
differently — that is what makes them honest, and it is why `mase` across runs with different fold
geometry is not a like-for-like comparison.

**On the two interval scores:** `coverage` and `interval_width` are each half of the story and
each trivially gamed — an infinitely wide band covers everything, a zero-width one is maximally
sharp. `interval_score` is the one number that combines them, which is why it is the sensible
`decision_metric` if intervals are what you are ranking on. The other two are worth reading
alongside it because they say *how* a model got its score. Whichever you pick, read it next to
`forecast_metadata.interval_source`: a model that computed its own interval and one whose band was
manufactured from its residuals are not scored on the same thing (see
[output_schemas.md](./output_schemas.md)).

**Which direction is better is not the same answer for every metric.** Thirteen of the fifteen are
errors, so smaller is better. `coverage` is a hit rate, so larger is better. `bias` is signed, so
what you want is *near zero* — a large negative bias is exactly as wrong as a large positive one.

Three places in the system have to rank things by the chosen metric — the HPO objective, the
`inverse_error` ensemble weights, and `prune_threshold` — and each of them needs the same
translation into a single "smaller is better" number. `metrics.METRIC_DIRECTION` states the
direction for each metric and `metrics.loss_of` applies it: an error passes straight through,
`coverage` becomes `1 − coverage`, `bias` becomes `|bias|`, and a metric that could not be computed
becomes infinity so it ranks last rather than first. That one map is why picking a different
`decision_metric` does not require thinking about which of the three consumers handles it correctly.

Two caveats the direction map deliberately does not try to fix. Coverage is really best *at* its
nominal level, not at 1.0 — a band wide enough to cover everything scores perfectly here — and
`interval_width` on its own rewards a band of zero width. Neither is a good `decision_metric` alone;
`interval_score` is the one that trades them off.

## `output` — `OutputConfig`

What the number in `yhat` actually *is*. One field.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `point_forecast` | `"raw"` \| `"median"` \| `"mean"` \| `"auto"` | `"auto"` with a backtest, `"median"` without | Which arm ships in `yhat`. |

Every model emits one number per future date, so something decides what that number is:

- **`raw`** — the model's own output, untouched.
- **`median`** — that output plus the median residual. Minimises absolute error, which is what most
  of the metric panel measures.
- **`mean`** — plus the mean residual. Minimises squared error, and drives `bias` to zero by
  construction. **Requires `backtest.enabled`**: the mean shift is estimated from out-of-fold
  residuals and there is no in-sample equivalent for a model that builds its band from quantiles.
  Asking for it without a backtest is an error, not a silent downgrade.
- **`auto`** — decide per series *and* model, from that cell's own held-out folds. The three above
  are one setting for the whole fleet; this one is the admission that a fleet is not uniform. Also
  **requires `backtest.enabled`**, for the same reason and more sharply: there is nothing to decide
  from otherwise. See *Letting each series choose* below.

`yhat_raw` and `yhat_adjusted` are both written to `forecast_predictions` and `backtest_oof`
whatever you set here, so the choice is never destructive — you can re-score a finished run on the
other arm without re-fitting anything.

**A backtest earns `auto`.** Leave `point_forecast` unset and a run with backtesting on resolves to
`auto` — the arm is measured per series and model rather than assigned to the whole fleet. Without a
backtest there is nothing to measure from, so it resolves to `median`. The resolution happens at
parse time and lands in the serialized config, so the `run_id` records what was asked for rather
than an instruction to decide later; `auto` stays as written, because its answer is per cell and no
single arm can stand for it.

**Why not the fleetwide rule?** Because it is a theorem about the *expected* case and the fleet is
not the expected case. The pairing is real — the median minimises absolute error, the mean minimises
squared error — and it is still what an `auto` cell falls back to when it has too few folds to
choose, and still what warns you if you pair an explicit `median` with a squared-error metric. But
measured at 3 folds across ten models, per-series selection landed fleet RMSE at 28.77 against
30.54 for the fleetwide rule, and MAE at 18.75 against 19.32. The case that settled it: under a
squared-error metric the fleetwide rule picks `mean`, and `mean` scored **worse than applying no
correction at all** (30.54 against 30.05). A default that is beaten by doing nothing is not a
default.

**Why this field exists at all.** For a long time the project decided this by accident: ten of the
sixteen models built their band from residual quantiles, the frame assembler took the 0.5 quantile
as `yhat`, and the shipped forecast was silently the model's prediction plus its median in-sample
residual — un-named, un-configurable, and applied to some models and not others, so the leaderboard
was comparing corrected models against uncorrected ones. Measuring it on ten models found the
correction was worth keeping (it moved fleet WAPE by 5.7%), so it stayed the default. This field is
what turns it from an accident into a default: the alternative is now sayable, and
`sf.calibration_report(run_id)` reports what the choice was worth on your data.

### Letting each series choose — `point_forecast: "auto"`

The 5.7% figure above is a fleet average, and a fleet average is the thing that hides the series it
does not apply to. The residual correction helps a model that is genuinely biased on a given series
and hurts one that is not, because on an unbiased series the shift is fitted noise. `auto` asks the
question per cell instead of once for everybody: for each series-and-model pair, does the corrected
arm beat the raw arm on that cell's own held-out folds?

**The rule.** The corrected arm keeps its place only if it wins on a **strict majority of the cell's
held-out folds**, with a minimum of **three** folds. Fewer than three, or no usable backtest at all,
and the fleetwide arm applies unchanged. A tie goes to `raw`.

Three things about that rule are worth stating because none of them is the obvious choice:

- **There is no margin threshold**, although the natural design has one ("switch only if the
  correction wins by more than 5%"). At three folds it cannot work. In a simulation with no real
  effect at all, the pooled margin's 90th percentile is 0.113 — an 11% apparent improvement out of
  pure noise — and under a real effect the margin stays negative until the bias is roughly half a
  standard deviation, so the two distributions overlap almost completely. Every threshold tried
  (2%, 5%, 10%) came out *worse* than no threshold at every fold count. Counting how many folds
  agree is the robust form of the same question, and it measured better.
- **Three folds, not five.** Two is not merely weak, it is wrong in a known direction: with two
  folds the comparison has a single fold to grade on and grades the correction on the residuals it
  was fitted from, which flatters it systematically. Three is where the evidence becomes honest, and
  it is also where selection posts its largest gain — a minimum of five would have discarded that.
- **A tie goes to `raw`.** Not a coin flip: in the folds where the two arms genuinely cannot be told
  apart, the corrected arm is still carrying the estimation variance of a shift it did not need, so
  equal *measured* loss is not equal *expected* loss. Sending ties the other way was measured and
  cost about 3% of fleet error.

**What it costs and what it buys.** One extra pass over the out-of-fold residuals per cell, no
re-fitting. On a ten-model fleet it beat both fleetwide arms at every fold count tried, on both the
absolute-error and squared-error pairings, capturing roughly 45% of the gap to an oracle that knows
the answer in advance.

**What lands in the record.** `auto` is the one arm that stays unresolved in the serialized config,
because the resolution is per cell — the `run_id` records that selection was asked for, and
`forecast_metadata` records what each cell did with it:
`point_forecast_source` names the arm that shipped, and `point_forecast_decision` says how it got
there (`auto-raw`, `auto-corrected`, `auto-few-folds`, `auto-no-backtest`, and `configured` on a run
that named an arm). `sf.calibration_report(run_id)` rolls that up per model as `raw_arm_rate` — the
share of a model's series that ended up on the raw arm, which is 0 or 1 under a fleetwide setting
and anything in between under `auto`.

## `model_params` — hyperparameters you set yourself

HPO searches for hyperparameters. `model_params` is the other half of that surface: the place to
*state* them, when you already know what you want and would rather not pay for a search.

```json
"model_params": {
  "neuralprophet": {"n_lags": 28, "n_forecasts": 28, "learning_rate": 0.01},
  "xgboost": {"max_depth": 6, "n_estimators": 400}
}
```

Keyed by model name, then by that model's own parameter names. Values may be a scalar (`bool`, `int`,
`float`, `str`, `null`) or a flat list of scalars — anything that survives a JSON round-trip, since
`run_id` is a digest of the serialized config. `NaN` and `±inf` are rejected at parse time for
exactly that reason: `json.dumps` writes them as bare `NaN` / `Infinity`, which is not JSON, and a
digest nobody else's parser can reproduce is not an identity.

**Where an authored value takes effect, and what beats it.** Your block is the layer underneath
everything: it is applied at the cell, and it is also applied underneath *every HPO trial*, so a
study tunes the same model the run will actually fit. Where the two name the same key, **HPO wins** —
pinning `epochs` on a model whose search space also searches `epochs` means the trial's value is used
and your pin is ignored, because a study that scored one value and shipped another would publish a
metric that does not belong to the fitted model. Keys the search space does not name are untouched,
which is the common case: an authored `n_lags` survives a tuned `learning_rate`. With HPO off, your
block is simply the params, filling in over each model's own defaults.

**Unknown model names are refused before anything is provisioned, not at parse time.** A block keyed
by a name no model is registered under is a typo, and a typo here is silent — the params just never
reach a model. It is not checked in `config.py` because that would mean importing the model registry
on the job-submission path, where the model stack is deliberately absent (eager model imports there
have broken a live run before). It is checked instead at plan time, on the paths that are about to
spend: a direct `run`, a staged launch, and the Airflow DAG's first task. A block for a model your
`models` list does not select is a warning, not an error — it has no effect.

Unknown *parameter* names still pass: each model reads the specific keys it knows and ignores the
rest. A model may also refuse a combination it cannot honour — NeuralProphet, for example, rejects
`n_lags > 0` with an `n_forecasts` below the run's longest horizon, because in autoregressive mode it
emits exactly `n_forecasts` direct steps and does not recurse to fill a longer request. That refusal
happens at plan time too, so a config that would return a horizon of `NaN` costs nothing instead of a
fleet-hour.

## `hpo` — `HpoConfig`

Optional Optuna tuning on the aligned backtest. **Requires `backtest.enabled`.**

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn HPO on. |
| `engine` | `"optuna"` | `"optuna"` | — | HPO engine. |
| `n_trials` | `int` | `20` | `> 0` | Trials per study. |
| `granularity` | `"fleetwide"` \| `"per_series"` | `"fleetwide"` | — | The cost knob — see below. |
| `sample_size` | `int` | `20` | `> 0` | Fleetwide only: how many series to tune on before applying the winner fleet-wide. |

**Fleetwide vs. per-series — the cost trade-off.** `fleetwide` (default) tunes each model **once**,
on a driver-side sample of `sample_size` series, then applies the winning hyperparameters to the
whole fleet — one study per model, cost independent of series count. `per_series` tunes **inside
every cell** — the best accuracy a model can reach on each series, but `n_trials` studies *per
series*, so cost scales with the fleet. Start fleetwide; reach for per-series only when a model's
optimal hyperparameters genuinely vary series to series and the accuracy is worth the spend.

`sample_size: 20` means twenty series drawn from the whole fleet, not twenty per anything — the
winner from those twenty is what every series in the run is fitted with.

Tuned hyperparameters flow to the workers through the engine, **not** through the config — so HPO
never shifts the config-derived `run_id`, keeping runs reproducible and idempotent.

**A caveat worth knowing before you compare a tuned model to an untuned one.** HPO runs *by*
backtesting: each trial runs the aligned backtest on the sampled series and is scored on
`decision_metric`. The winning params are then handed to every cell, which backtests **again** — and
*that* second backtest is what fills the leaderboard, `forecast_metadata`, and the `inverse_error`
ensemble weights. Both passes use the same folds. So a tuned model's published metric is measured on
the data its hyperparameters were selected on, which makes it optimistic; an untuned model's is not.
Ranking tuned against untuned models on one leaderboard therefore tilts toward the tuned ones by an
amount nothing currently reports. The fix — holding the most recent fold out of the search so the
reported score is scored on a fold that fitted nothing — is planned and not yet shipped. Until then,
treat a tuned model's leaderboard number as an upper bound, and prefer a like-for-like comparison
(both tuned, or both not).

## `ensemble` — `EnsembleConfig`

Consensus across the base models (scored onto the same leaderboard). See
[running_and_reviewing.md](./running_and_reviewing.md) for the re-ensemble workflow.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn ensembling on. |
| `strategies` | `list` | `["median"]` | each a known strategy | Consensus strategies (run several at once). |
| `prune_threshold` | `float` | `0.0` | `≥ 0.0` | Drop base models weaker than this before blending. |

`strategies` ∈ **calculated** `mean, median, inverse_error` (no backtest needed) and **learned**
`nnls, ridge, xgb` (need `backtest.enabled`). A bare `"strategy": "nnls"` is accepted as shorthand
for `"strategies": ["nnls"]`. Run several at once — each produces its own `ensemble_<strategy>` rows
and earns a line on the leaderboard next to the base models.

**What each strategy does** (implementation in
[`ensembler.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ensembler.py)):

| Strategy | Kind | How the blend is formed | Needs |
|----------|------|-------------------------|-------|
| `mean` | calculated | Unweighted arithmetic mean of the base forecasts, row by row. Missing models are skipped and weights renormalize over those present. | — |
| `median` | calculated | Unweighted row-wise **median** — robust to a single wild base forecast. | — |
| `inverse_error` | calculated | Weighted mean where each model's weight ∝ **1 / its error**, normalized to sum to 1. For the future forecast the error is the mean `decision_metric` from `forecast_metadata`; for the on-fold (OOF) score it's each series' own WAPE. A zero-error model captures the weight; if no error is usable it degrades to `mean`. | (weights sharpen with `backtest`) |
| `nnls` | learned | **Non-negative least squares** meta-learner: solves for weights ≥ 0 that best reconstruct the truth from the base models' out-of-fold predictions. No intercept; weights can't go negative. | `backtest.enabled` |
| `ridge` | learned | **L2-regularized linear regression** (closed-form, α=1.0) over the same OOF matrix. Weights *may* be negative (a model can be a corrective term). | `backtest.enabled` |
| `xgb` | learned | **Gradient-boosted** meta-learner (`XGBRegressor`, 200 trees, depth 3) fit on the OOF matrix; captures non-linear interactions between base models. Reported "weights" are its normalized feature importances. | `backtest.enabled`, `xgboost` installed |

The three **learned** strategies train **only** on `backtest_oof` rows (never in-sample), so leakage
is structurally impossible — which is exactly why they require `backtest.enabled` (without it they're
dropped at config-load with a warning, not an error). Each fitted meta-learner is pickled to a GCS
artifact for lineage.

**`prune_threshold`** applies only to the **calculated** strategies: when `> 0`, any base model whose
mean `backtest.decision_metric` is *worse than* the threshold is dropped from the blend fleet-wide
before combining. `0.0` (default) prunes nothing.

"Worse than" is measured as a loss (`metrics.loss_of`, above), so the comparison reads the same way
whichever metric you chose: a threshold of `0.3` on `wape` drops models above 0.3, and the same
`0.3` on `coverage` drops models that cover less than 70% of actuals. A model with no scored rows at
all is kept — absence of evidence is not evidence of a bad model, and pruning on it would silently
empty the blend on a run where the metric frame came back short.

## `compute` — `ComputeConfig`

Runtime scale, dependency delivery, and cost guardrails. Defaults are tuned for a first run; the Ray
knobs only matter for a family that runs on Ray.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `families` | `dict[family → FamilyCompute]` | `{}` | keys ∈ `statistical`/`ml`/`deep_learning` | Per-family runtime/hardware overrides (see below). |
| `max_parallelism` | `int` | `1000` | `> 0` | Max parallel tasks. |
| `bucket_target_cells` | `int` | `8` | `> 0` | Target cells per Spark bucket (shuffle-partition sizing). |
| `max_executors` | `int \| null` | `null` | `> 0` | Operator ceiling on the Spark fleet — most executors a batch may scale to, most workers a cluster may hold. `null` sizes to the fan-out alone, which at 100k series asks for hundreds and is rejected outright by a regional CPU quota. Budget for concurrency: a run's families submit together. |
| `machine_family` | `"auto"` \| `"n1"` \| `"n2"` \| `"n2d"` \| `"e2"` \| `"c2"` | `"auto"` | — | GCE machine family for a **Dataproc cluster's** master + CPU workers (`"auto"` = `n1`). No-op on Serverless and on GPU workers — see below. |
| `spark_deps` | `"packed_venv"` \| `"container"` | `"packed_venv"` | — | How a **Dataproc cluster** family gets its dependencies. `"container"` raises: it is a Serverless mechanism. See `cluster_deps._resolve_cluster_deps`. |
| `persist_models` | `bool` | `false` | — | Persist each fitted model as a GCS artifact (lineage). |
| `use_gpu` | `bool` | `false` | — | **Legacy.** Fleet-wide "put the deep-learning family on GPU", on *every* runtime — Ray, Serverless and a Dataproc cluster alike. Prefer `families.deep_learning.hardware`; see below. |
| `gpu_type` | `str` | `"T4"` | — | **Legacy.** Accelerator type for the `use_gpu` pair. Prefer `families.<f>.gpu_type`. |
| `gpu_fraction` | `"auto"` \| `float` | `"auto"` | float ∈ `(0, 1]` | `"auto"` = profile-driven fractional GPU, else a fixed fraction. |
| `budget_usd` | `float` | `50.0` | `≥ 0.0` | Cost guardrail (USD). |
| `ray_cluster_name` | `str` \| `null` | `null` | — | Reuse a standing Ray cluster by name; `null` = ephemeral. |
| `ray_regions` | `list[str]` \| `null` | `null` | — | Priority-ordered candidate regions for the ephemeral cluster. |
| `ray_head_machine_type` | `str` | `"n1-standard-16"` | — | Head-node type (don't drop below, or job submit hangs). |
| `ray_cpu_machine_type` | `str` | `"n1-standard-8"` | — | CPU worker-pool machine type. |
| `ray_gpu_machine_type` | `str` | `"n1-standard-8"` | — | GPU worker-pool type (must be N1 for T4). |
| `accelerator_count` | `int` | `1` | T4 ∈ `{1,2,4}` | GPUs per GPU worker node. |
| `ray_target_cells_per_slot` | `int` | `8` | `> 0` | Cells one worker slot chews before a node is added. |
| `ray_max_nodes` | `int` | `16` | `> 0` | Shared per-pool ceiling; the fallback when a pool's own max is unset. |
| `ray_autoscale` | `bool` | `true` | — | Autoscale each worker pool between its min/max (default). `false` restores fixed-size sizing (the derived `node_count`, no autoscaling spec). |
| `ray_cpu_min_nodes` | `int` | `1` | `> 0` | CPU pool floor. Low = shrink when the queue drains. |
| `ray_cpu_max_nodes` | `int` \| `null` | `null` | `> 0` | CPU pool ceiling; `null` falls back to `ray_max_nodes`. Raise to grow under load. |
| `ray_gpu_min_nodes` | `int` | `1` | `> 0` | GPU pool floor. Low = shrink idle (expensive) T4s. |
| `ray_gpu_max_nodes` | `int` \| `null` | `null` | `> 0` | GPU pool ceiling; `null` falls back to `ray_max_nodes`. Cap independently for cost. |
| `gpu_calibration_samples` | `int` | `3` | `> 0` | Series to profile for auto `gpu_fraction`. |
| `gpu_safety_margin` | `float` | `1.3` | `> 1.0` | Headroom multiplier on measured peak GPU memory. |
| `ray_read_mode` | `"driver_collect"` \| `"ray_data"` | `"driver_collect"` | — | Ray source reader: the proven Storage Read client, or `ray.data.read_bigquery` (same Storage Read API, opt-in). |
| `read_max_streams` | `int` | `0` | `≥ 0` | Max Storage Read streams for the source read, shared by the Spark connector (`maxParallelism`) and Ray's `driver_collect` reader (`max_stream_count`). `0` lets the server size it from the table; a positive value caps read parallelism (e.g. to fit a slot/quota budget). Inert for the `ray_data` path and BigQuery-native models. See [reading_source_data.md](./reading_source_data.md). |

**`machine_family` — one knob, three deliberate boundaries.** It selects the GCE family for a
**Dataproc cluster's** master and CPU workers (`worker_machine_type` / `master_machine_type`), and
`"auto"` resolves to `n1` — today's shipped shape, so an existing config renders an identical
cluster. What it does *not* do is as important:

- **It does not pick a size.** Cores are fixed (master 4, workers 8) because the profiler derives
  the executor shape *from* the machine; letting you set both gives two knobs that can disagree.
- **It does not reach GPU workers.** The accelerator dictates the machine — a T4 is an add-on card
  that only attaches to `n1`, an L4 is bundled inside `g2` — so a family override there would ask
  GCE for a shape it does not sell. A run spanning both hardware kinds gets its CPU workers on your
  family and its GPU workers on the accelerator's.
- **It does nothing on Serverless or Ray.** Serverless has no machine concept at all (its shape is
  executor cores/memory properties); the Ray pools have their own explicit
  `ray_*_machine_type` knobs.

The offered families are exactly those `resources` can price (`_MEMORY_PER_CORE_GIB`), so the
sizing plan stays honest for whichever you pick — choosing `n2` moves the worker from 30 GiB to
32 GiB and the executor split follows. An unlisted family is rejected at config load rather than at
cluster create.

**Autoscaling (default):** each Ray worker pool scales between its own `[min, max]`; a `null` pool
max resolves to `ray_max_nodes`. Config validation requires `min ≤ resolved max` per pool. The
*initial* pool size stays a deterministic function of the config (fan-out ÷
`ray_target_cells_per_slot`, clamped into the bounds), so the run remains reproducible — the whole
spec is hashed into `run_id` and stamped to `run_registry.job_telemetry`. Set `ray_autoscale=false`
to opt out and get a fixed-size pool instead (no autoscaling spec at all) — worth reaching for when
you want a cluster whose cost is flat and predictable rather than demand-shaped. See the Ray runtime
in [architecture.md](./architecture.md).

### `compute.capacity` — how hard to look for room

"Resources are not available" is a **state**, not an exception. When a create fails for want of
machines the launcher walks its candidate places (Ray: `ray_regions`; a Dataproc cluster: the
zone/region fallback map; Serverless: the region), and if none has room it backs off and walks them
again until a budget runs out. While it waits the family's `run_jobs` row reads `AWAITING_CAPACITY`
and carries a ledger of every attempt; on exhaustion the row is `FAILED` with
`failure_reason = CAPACITY_EXHAUSTED`. See
[troubleshooting.md](./troubleshooting.md#capacity--the-cloud-has-no-room) for reading it.

`{"enabled": false}` (top level) collapses every service to a single pass with no back-off — the
pre-retry behaviour exactly. Otherwise each service takes a **partial** override: name only the
number you have an opinion about and inherit the rest.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `true` | — | `false` = one pass over the candidates, no back-off, all services. Beats an authored `max_passes`. |
| `preflight` | `bool` | `true` | — | Read each candidate region's quota **before** the first create, and skip or clamp accordingly. See below. |
| `ray` \| `dataproc_cluster` \| `dataproc_serverless` | `object` | `{}` | — | Per-service partial override; unset fields inherit the shipped default below. |
| `retry` | `object` | `{}` | — | How wide a **repair** runs — not a capacity policy, but the same digest-excluded surface. See below. |

`preflight` is the cheap half of the same problem: retrying is for a region that is *temporarily*
full, and a preflight is for one that was never going to work. It reads the region's allowance,
drops a region that cannot host even the minimum (recorded as a hard ceiling, no create attempted),
and lowers this run's pool ceilings — or, on the Dataproc-cluster path, its physical worker count —
to what a smaller region will grant rather than failing there. It applies to Ray on Vertex and to
ephemeral Dataproc clusters, each read against *its own* service's meters. It only ever lowers, and
it never touches `run_id`. `--quota` prints the same report without
launching — see [Quota and scale](./quota_and_scale.md#4-which-quotas-and-where). Set it `false`
only if the runner service account lacks `serviceusage.services.get`; the check degrades to silence
on any read failure, so a missing permission costs you the diagnostic, not the run.

Per-service fields, all optional, `0` disables that bound:

| Field | Type | `ray` | `dataproc_cluster` | `dataproc_serverless` | Purpose |
|-------|------|-------|--------------------|-----------------------|---------|
| `max_attempts` | `int ≥ 0` | `6` | `8` | `10` | Total attempts across all candidates. |
| `max_wall_seconds` | `float ≥ 0` | `3600` | `2700` | `1800` | Clock budget for the whole walk. |
| `max_passes` | `int ≥ 0` | unbounded | unbounded | unbounded | Full sweeps of the candidate list. |
| `backoff_seconds` | `float ≥ 0` | `120` | `60` | `30` | First wait after a fruitless pass. |
| `backoff_multiplier` | `float ≥ 1.0` | `2.0` | `2.0` | `2.0` | Growth per pass. |
| `backoff_max_seconds` | `float ≥ 0` | `600` | `300` | `120` | Cap on the wait. |

The three differ because the services do. A Vertex Ray GPU provision costs ~12 minutes per attempt,
so it gets the fewest tries and the longest patience; a Serverless batch is rejected in seconds, so
retrying is nearly free and the clock is the bound that matters.

**BigQuery has no entry, deliberately.** Slot contention is resolved BigQuery-side and surfaces as
latency, not as a create that failed somewhere else it could be tried. There is no candidate list to
walk.

**None of this moves the `run_id`.** Patience is an operational knob, not part of what was asked
for — same rule as `compute.profile.source`. If it hashed into the digest, waiting longer for a GPU
would fork your run identity and break dedupe-on-read.

```json
"compute": {
  "capacity": {
    "ray": {"max_wall_seconds": 7200},
    "dataproc_serverless": {"max_attempts": 20}
  }
}
```

#### `compute.capacity.retry` — how wide a repair runs

A repair (`--retry`, `Forecaster.retry`, or the emitted DAG's retry node) re-submits a handful of
cells out of a run that was sized for all of them. The fleet arithmetic does not know that: it sizes
from the config's fan-out, so a forty-cell repair of a hundred-thousand-cell family asks for the
hundred-thousand-cell fleet and pays for it.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `max_executors` | `int` \| `null` | `null` | `> 0` | Ceiling on the repair's Spark batch (`spark.dynamicAllocation.maxExecutors`). `null` = size from the fan-out, as a first attempt does. |

It is here rather than beside `compute.max_executors` because a repair that resized itself must stay
the **same run**: everything under `compute.capacity` is excluded from the `run_id` digest, and a
forked id would mean the repair wrote its rows under a run nobody is looking at.

It is in the config rather than only on `--max-executors` because the unattended repair paths launch
from a config and nothing else — a ceiling reachable only through a flag is one an orchestrated
repair can never set. An explicit `--max-executors` still wins; the operator passing it is looking at
the run, and the config was written before the run failed.

Two limits worth knowing. It caps a Spark **batch** and is ignored by the Ray and in-process paths,
so a Ray family's repair still provisions from the fan-out. And it is the only resource a repair can
vary: a repair reuses the config its run staged, so anything the *worker* reads out of that config
(`bucket_target_cells`, `max_parallelism`, the model list, the data block) is the same bytes for the
repair as for the attempt, by construction.

```json
"compute": {
  "capacity": {
    "retry": {"max_executors": 8}
  }
}
```

### `compute.profile` — measured compute profiling

Sizing is otherwise a pure cell **count**, divided by a flat cells-per-slot constant. The count is

```
cells = n_series × n_models
```

and **folds are not in it** — a cell is one model fitted to one series, and a backtested cell is
still one cell; it just does more fits inside itself. (This document used to write
`n_series × n_models × n_folds`, which is why a two-fold dry run once reported three times the cells
it would ever write a row for. `estimate_fanout` has not multiplied by folds for some time.) What
folds do change is the *work* per cell:

```
fits = cells × (n_folds + 1)
```

— the `+ 1` being the final fit on all of the history, which is the forecast you actually ship.
That is the same arithmetic [quota and scale](./quota_and_scale.md#1-the-arithmetic) plans a run
with. `estimate_workload` goes one step further when it can see series lengths: the fold fits train
on shorter windows than the final one, so it also reports `full_fit_equivalents`, which is what a
backtested cell costs relative to the same cell with backtesting off. On four years of daily history
with a 28-day horizon that comes out at **2.94** at two folds and **3.862** at three, against the
3.0 and 4.0 that `n_folds + 1` would imply. Run `--feasibility` to see it for your own panel.

None of that arithmetic can know that a deep-learning fit and a naive mean differ by orders of
magnitude, so the fleet is provisioned for the count rather than for the work. `compute.profile` is
the machinery for replacing that guess with a measurement.

Think of it as the general form of `gpu_calibration_samples` / `gpu_safety_margin` above, which
already do exactly this for one axis (GPU bytes), one model, one runtime.

#### What this actually does today — read this before setting anything

Two things are true at once, and conflating them will mislead you.

**Fleet shaping is on, everywhere, and it does not need a measurement.** Every Spark job now carries
a derived properties overlay: executor cores, heap and memoryOverhead, the dynamic-allocation band,
`spark.task.cpus` bounded by the accelerator, matching thread pins, and — on a Dataproc cluster — a
worker count derived from the run's fan-out. That arithmetic runs from the config alone. Setting
`mode = "off"` is what turns it off, and that is the escape hatch to reach for if a fleet shape ever
misbehaves in production.

**Measuring *inside the run that needs it* does not work on Spark, and structurally cannot.**
`spark.executor.cores` and `spark.task.cpus` are fixed at submit (Serverless) or at create (cluster),
before any of our code reaches the cluster; and the submit host is deliberately lean, carrying no
model stack to fit with. A same-run pre-pass has nowhere to run that is both early enough to matter
and equipped to measure anything. Only the Ray engine can do it — it requests per-task `num_cpus` /
`num_gpus` in-run against an autoscaling pool — and with the defaults below (`mode = "auto"`,
`min_cells = 1000`) it does not fire for a small run. So on Spark, `"auto"` and `"always"` size
identically today; only `"off"` changes anything.

**So measurement is decoupled from consumption: one run produces the evidence, a later run is sized
from it.** That is what `measure` is. Every cell of an ordinary run records what it cost — three
cheap probes around a fit that was happening anyway — onto `forecast_metadata`. A completed `run_id`
is therefore a measured cost model you can point a bigger run at. It is on by default, because the
run you wish you had measured is always the one you already finished.

**Three knobs, three questions.** `mode` is the master switch — whether the fleet arithmetic runs at
all, and the threshold the in-run Ray pre-pass compares against. `measure` decides what evidence this
run **produces**. `source` decides what evidence it **consumes**. They are orthogonal because the
questions are: a run can harvest without consuming (the first one ever), consume without harvesting
(a one-off resize), or do both (the ordinary case, and the default). `mode = "off"` vetoes all of it,
so there is still exactly one switch that makes the whole feature inert.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|------------|---------|
| `mode` | `"off"` \| `"auto"` \| `"always"` | `"auto"` | — | `off` disables **both** the derived fleet overlay and any measurement — the escape hatch, and the only value that changes a Spark run today. `auto` measures when the cell count reaches `min_cells`; `always` measures unconditionally. Both currently affect the Ray path only. |
| `samples` | `int` | `8` | `> 0` | Series fitted in the pre-pass, spread across length/complexity strata. |
| `min_cells` | `int` | `1000` | `> 0` | The threshold `auto` compares the cell count against. |
| `memory_margin` | `float` | `1.3` | `> 1.0` | Headroom on the measured **max**, which sizes the slot. |
| `time_margin` | `float` | `1.2` | `> 1.0` | Headroom on the measured **median**, which sizes the fleet. |
| `measure` | `"off"` \| `"harvest"` \| `"controlled"` | `"harvest"` | — | What evidence this run **produces**. See the table below. |
| `source` | `"auto"` \| `"baseline"` \| `"none"` \| `"<run_id>"` | `"auto"` | keyword or a well-formed run_id | What evidence this run **consumes**. See the precedence chain below. |

| `measure` | What the run does | Use it when |
|-----------|-------------------|-------------|
| `"harvest"` | Records per-cell CPU time, absolute process memory, peak device bytes, the thread cap and `n_obs` onto `forecast_metadata`. Changes nothing about how the run executes. | Always — this is the default. |
| `"controlled"` | Harvest, **plus** the Spark fleet leaves the native thread pools uncapped so a fit's measured `effective_cores` reflects the library instead of the cap. | A deliberate calibration run, and never a production one: the executors are knowingly oversubscribed, so the run is slower and its shape is not a real run's shape. The translation carries a note saying so. |
| `"off"` | Writes NULL in all five columns. | You have a reason to skip three probes per fit. |

| `source` | Where the numbers come from | Provenance basis |
|----------|-----------------------------|------------------|
| `"auto"` | The newest completed run whose harvest matches this run's data signature; failing that, the shipped baseline; failing that, static config. Resolved **at plan time** and written into the staged config as a concrete `run_id`, so the run records what it actually sized from rather than a search that might resolve differently tomorrow. | `measured` / `reference` |
| `"<run_id>"` | That run's harvest, whatever its signature. Naming a run is a decision, so it is honoured — a drifted signature comes back as a warning, not a substitution. | `measured` / `reference` |
| `"baseline"` | The shipped, versioned reference measurements — see below. Real numbers, taken on reference hardware and reference data, never on yours. | `reference` |
| `"none"` | Nothing is consulted; size from declared config. | — |

The precedence, outside-in: **explicit `compute` settings > `compute.profile.source` > shipped
baseline > static config.** Anything you set by hand always wins; the evidence only fills what you
left to be derived, and the static floor is the behaviour this product shipped with.

Every resolved profile carries a `provenance` block naming its basis, the `run_id` and timestamp
behind it, the data signature it was measured on, and any drift warnings. `measured` means the
evidence matches this run's signature on every axis both sides can see; **`reference` means measured,
but not on your data**. That third value exists because a pinned profile from months ago on a
different table is worse than no profile at all — precisely because it looks authoritative. A
mismatch is never silent, and never fatal: sizing off drifted evidence still beats sizing off none.

If BigQuery is unreachable when the source is resolved, the run logs a warning and falls through the
rest of the chain — which today means the shipped baseline. Evidence is an optimisation; a registry
hiccup must not stop a run from submitting.

**What ships in the baseline.** It was cut from a real 100,000-series run on Ray (`ray_100k`,
recorded in [validation.md](validation.md)): daily series of 1,460 observations, four models across
the `statistical` and `ml` families. It therefore sizes those two families and **not**
`deep_learning`, and it carries no GPU bound — that run had neither. An unmeasured family resolves
to nothing rather than to a guess, so a run with a deep-learning family gets the static arithmetic
for it and measured numbers for the rest. The number it is really there for is
`slot_cores: 1`: every one of those fits measured single-threaded, and that is a property of the
libraries rather than of your panel, so it transfers in a way a memory bound does not.

**What the resolved profile actually changes.** Only the memory axis. The executor cores, the thread
pins, the warm `initialExecutors`, the device-aware `spark.task.cpus` and the worker/executor counts
all follow from the fan-out and the machine type alone and are emitted with or without evidence.
Memory is the one thing that cannot be derived — a Serverless executor's shape is fixed at *submit*
and a cluster's at *create*, both before any of our code runs — so with no profile the memory
properties are simply not emitted and the platform's defaults stand, exactly as before.

**Reading back what a run decided.** The whole decision is stamped into the run header's
`job_telemetry`, one entry per family job under `sizing.<family>`, and surfaced by `v_run_summary`
as its `sizing` column: the fleet the arithmetic asked for, what that became in platform settings,
and the profile it was sized off (with the provenance naming whose run supplied the evidence). So
"why was this run this shape" is a query against the registry rather than a hunt through driver
logs — see [output_schemas.md](./output_schemas.md).

**Why the two margins differ, and why they apply to different tails.** Over-estimating time buys
extra slots, which costs money; under-estimating memory OOM-kills the task, which costs the run.
Asymmetric risk, asymmetric margin — so memory carries the wider one. They also attach to different
statistics on purpose: a slot must hold the *worst* series that lands in it (max), while a fleet is
sized for *typical* work (median). Sizing the fleet off the worst case over-provisions every run;
sizing memory off the median OOM-kills it. Using one tail for both is the mistake the split exists
to prevent.

`compute.profile` is part of the `run_id` digest, like everything else under `compute`. It changes
the resource shape rather than the forecasts, so that is a deliberate choice: the config is the
experiment record, and a run whose fleet was sized differently is not the same run for performance
purposes. One practical consequence: adding these fields moved every pre-existing `run_id`, so a
config saved before the profiler existed no longer re-derives the id it originally produced — and
adding `measure` and then `source` moved them again. Re-running an older config produces a new
`run_id` and therefore a new run rather than a resumed one.

**None of the derived fleet arithmetic has run on live infrastructure yet.** It is offline-proven
self-consistent — the legal-value snapping, the AM reserve, the worker derivation all have unit
tests — but Dataproc has never been asked to accept it. See
[Validation ledger](./validation.md), where the Spark rows are currently `STALE` for exactly this
reason.

### `compute.families` — per-family runtime & hardware

By default every Python family runs on the run-level `python_runtime` on CPU. `compute.families` maps
a family name to a `FamilyCompute` that overrides that placement for **that family only** — the lever
behind "one run, a job per family, each on its own runtime". Every `FamilyCompute` field is optional;
an unset field inherits the run-level default. The `native` family is never listed here (it always
runs in BigQuery).

**This block is canonical; the flat `compute.use_gpu` / `compute.gpu_type` pair is legacy.** Both
still work and neither is deprecated, but they are the coarse version of what `families` says
precisely, and one sentence about `use_gpu` has been wrong in this document for a long time: it is
**not** a Ray switch. `use_gpu` is read by `Config.resolve_family_compute`, the single resolver every
runtime goes through, so it puts the deep-learning family on an accelerator on **Ray, Dataproc
Serverless and a Dataproc cluster alike** — an L4 on Serverless, a T4 on the other two. It never
touches any other family: `hardware` is `"cpu"` for `statistical` and `ml` no matter what the flag
says.

Resolution is a single layering, and the per-family value always wins:

| Asked for | Resolves to |
|---|---|
| `families.deep_learning.hardware` set | that value — `use_gpu` is ignored for this family |
| unset, `use_gpu: true` | `"gpu"` for `deep_learning`, `"cpu"` for everything else |
| unset, `use_gpu: false` (default) | `"cpu"` |
| `gpu_type` unset on a GPU family | `compute.gpu_type` — except on Serverless, which is forced to `L4` and raises if a `T4` was named |

So `use_gpu: true` with `families.deep_learning.hardware: "cpu"` is not a contradiction the product
has to guess at: the family runs on CPU and **no accelerator is provisioned for it at all**. That is
smoke 20's whole job, and it is the reason the two settings can coexist safely. Prefer the
per-family form in new configs — it says which family, which runtime and which device in one place,
and it is the form the plan-time report and the device audit speak.

Before reaching for `hardware: "gpu"` at all, read
[what that setting guarantees](./quota_and_scale.md#what-hardware-gpu-guarantees-and-what-it-does-not).
It guarantees placement, not utilisation, and for `neuralprophet` at its shipped defaults the
measured answer is CPU.

| Field | Type | Options | Purpose |
|-------|------|---------|---------|
| `runtime` | `str` | `"spark"` \| `"ray"` | Runtime for this family (overrides `python_runtime`). |
| `spark_mode` | `str` | `"serverless"` \| `"cluster"` | Spark launch mode (Spark only). `"cluster"` runs on a Dataproc cluster — needed for a T4 GPU on Spark. |
| `spark_cluster_name` | `str` | — | Reuse an existing Dataproc cluster by name (requires `spark_mode="cluster"`). |
| `hardware` | `str` | `"cpu"` \| `"gpu"` | Hardware profile for this family (GPU only for `deep_learning`). |
| `gpu_type` | `str` | `"T4"` \| `"L4"` | GPU type when `hardware="gpu"`. |

**Cross-field rules** (enforced at config-load):

- `spark_mode` / `spark_cluster_name` are valid only when `runtime="spark"`; `spark_cluster_name`
  requires `spark_mode="cluster"`.
- A GPU (`hardware="gpu"` or `gpu_type` set) is allowed **only** for the `deep_learning` family.
- A T4 on Spark requires `spark_mode="cluster"` (Serverless can't attach a T4) — or route the family
  to `runtime="ray"`.
- `hardware="cpu"` with a `gpu_type` set → error (drop `gpu_type` or set `hardware="gpu"`).

**A GPU plan is checked again at plan time, before anything is provisioned.** Two separate things
happen there and the difference between them matters. A plan that would *buy* a device nothing can
be scheduled onto is **refused** outright — that has one correct answer and the remedy is always a
config edit, so there is no override for it. A plan where a device *will* be used but will barely be
touched only **warns**, because that is a cost judgement rather than a mistake.

The warning is the one you are most likely to see, and it is worth reading rather than dismissing.
NeuralProphet is the only model with a tensor library under it, and at its shipped defaults
(`n_lags` unset) the network is a few hundred trend and Fourier parameters. Measured across 31,356
fits on live T4s: peak device memory 50–78 KB against a 17 GB card, and `cpu_seconds / fit_seconds`
between 0.93 and 0.996. The card is attached, the tensors are on it, and it is doing essentially
nothing — a CPU run being billed as a GPU run. There are two remedies and they are not equally
proven: setting `hardware: "cpu"` costs nothing in accuracy and is what every green run in the
ledger did, while turning on autoregression with `model_params.neuralprophet.n_lags` is what would
actually make the device earn its cost but has no accuracy A/B and no live smoke behind it yet.

The same report fires in reverse: `use_gpu: true` with no deep-learning model selected means no job
resolves to GPU hardware at all, so the flag does nothing except make the config read as a GPU run.
You will see these lines on a dry run, on `--quota`, from the SDK's `.dag`, and in the submit log.

**At runtime, `hardware` decides the device rather than suggesting it.** Every model used to hand
its trainer `accelerator="auto"`, which can never fail — and that is what made "I paid for a card
and got none" invisible, and what made `hardware: "cpu"` unenforceable on a Dataproc cluster whose
executors expose a card to every family sharing them. A cell now selects its device outright, from
two facts that have to agree: the job must have been *provisioned* onto GPU hardware, and this
family must resolve to `hardware="gpu"` in your config.

The first fact is not in the config, because a config cannot know it — the same file runs on a
workstation with no card and on a Vertex Ray cluster with eight. The submitter states it on the job
it creates, as a `--provisioned-hardware gpu` driver argument plus the matching Spark
`executorEnv` / Ray `runtime_env` entry so the executors and task workers hear it too. You will see
it in the emitted `gcloud` command for a GPU batch; nothing else in the command changes. There is
nothing to set: run locally, from the SDK, from a notebook, or on any CPU job and the argument is
simply absent, which selects the old `auto` behaviour byte-for-byte.

Two consequences worth knowing. A family set to `hardware="gpu"` on a job that really was
provisioned for GPUs, but whose worker cannot see a CUDA device, now **fails that cell immediately**
with a message naming the family and the field to change, instead of quietly fitting on the CPU for
the rest of the fleet-hour. And the driver-side fits — the sizing pre-pass (`compute.profile`) and
every HPO trial — stay on `auto` deliberately, because they run on a Ray head node or a Spark
driver, which has no accelerator.

**How many clusters a run creates.** One per **hardware kind** among its ephemeral cluster families,
not one per run — a Dataproc cluster has exactly one worker machine type, so it is a CPU cluster or a
GPU cluster and cannot be both. A run whose cluster families are all CPU gets one cluster named
`sf-cluster-<run_id>`; a run mixing CPU and GPU families gets `sf-cluster-<run_id>-cpu` and
`sf-cluster-<run_id>-gpu`, each sized only for the families that land on it. That is why a GPU
`deep_learning` family no longer makes the rest of the run pay for accelerators it never uses.

Ray is deliberately different: a Vertex Ray cluster carries separate CPU and GPU worker *pools*, so
a mixed run shares **one** Ray cluster. The asymmetry is the hardware, not an inconsistency.

**Cluster lifetime.** A cluster the run creates (no `spark_cluster_name`) is deleted when the run
ends — every one of them, including on a failure part-way through creating the second. Each also
carries server-side bounds so it cannot outlive the orchestrator that made it: it
self-deletes after **30 min idle** or **24 h** total, whichever comes first. Those are backstops for
the case where the run process is killed before its teardown runs — when teardown works they never
fire. Override with `SF_CLUSTER_IDLE_TTL` / `SF_CLUSTER_MAX_AGE` (seconds; `0` disables a bound).
They are environment, not config, so changing them does **not** change your `run_id`.

A cluster named by `spark_cluster_name` gets neither — the run does not create it and does not own
when it ends, so reclaiming it is yours to do.

**How long the launcher waits, and what happens when it stops.** Separately from those server-side
bounds, the process that submitted the job blocks until the job is terminal. That wait is patience,
not a cost control — the launcher going away stops no meter — so it is set high, at **24 hours**,
and is moved with `SF_CLUSTER_JOB_WAIT_S` (seconds). Environment, not config: it does not change
your `run_id`.

If the wait does run out, the launcher **leaves the cluster running** and tells you where it is.
That is deliberate. A wait expiring says only that the launcher stopped looking, and the job it
stopped looking at is very likely still fitting; deleting its cluster at that moment would be the
launcher destroying its own healthy run. Follow the job with `--probe`, and the idle and max-age
bounds above reclaim the cluster once it finishes or goes quiet.

A job that genuinely wedges is caught by a different mechanism, one that watches output instead of
the clock: a job that has written no forecast rows after `SF_STALL_GRACE_S` (45 minutes by default,
`0` to switch it off) is cancelled. A single written row is proof of life and stands the watch down
for the rest of the run. Serverless batches have been under that watchdog since it was written; the
cluster path joined them.

Serverless has the same patience setting for the same reason, as `SF_BATCH_JOB_WAIT_S` (also 24
hours, also overridable per submit with `--wait-timeout`). It was never as dangerous there — a wait
expiring destroys nothing, because the batch runs on under its own ttl — but a healthy long batch
would still lose its Dataproc telemetry stamp and report a failure to whatever launched it.

```json
"compute": {
  "families": {
    "deep_learning": { "runtime": "ray", "hardware": "gpu", "gpu_type": "T4" }
  }
}
```

That routes the deep-learning family (e.g. `neuralprophet`) to Ray-on-GPU while the statistical and ml
families stay on the default Spark runtime and the native family runs in BigQuery — four families,
three runtimes, one `run_id`. See [`configs/per_family_runtimes_demo.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/per_family_runtimes_demo.json).

**If you are copying one of these, copy the CPU one.**
[`configs/per_family_runtimes_cpu_demo.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/per_family_runtimes_cpu_demo.json)
is the same four-family, three-runtime split with `"hardware": "cpu"` on the deep-learning family —
the two files differ in that one word, so they read as a diff. The CPU one is the better starting
point: on both accelerator A/B runs it finished *faster* than its GPU twin at identical accuracy and
roughly half the cost. The GPU config is kept because the numbers quoted in
[the validation ledger](./validation.md) were measured on it, not because the accelerator earned its
place; see [what `hardware: "gpu"` actually buys you](./quota_and_scale.md#what-hardware-gpu-guarantees-and-what-it-does-not).

## A minimal config

```json
{
  "run_name": "my first run",
  "python_runtime": "spark",
  "data": { "source_table": "source_series_iceberg", "horizon": 28, "series_limit": 100 },
  "models": ["theta", "holtwinters", "arima_plus"],
  "features": { "holidays": ["US"] }
}
```

`theta`/`holtwinters` run on Spark; `arima_plus` runs in BigQuery — both under one `run_id`, in
parallel. See [`configs/`](https://github.com/statmike/scale-forecasting/tree/main/configs) for worked examples (demo and 100k),
[running_and_reviewing.md](./running_and_reviewing.md) to submit and review one, and
[output_schemas.md](./output_schemas.md) for the tables the run writes to (the whole config lands
verbatim in `run_registry.raw_config`).
