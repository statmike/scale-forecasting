# Auto-Forecaster Benchmarks: 4-Tier Progression, 20 Synthetic DGPs, AlphaEvolve & 9 Public Datasets

> **Part of the Glass-Box Auto-Forecaster Guide:** [**Main Guide & Architecture (`auto_forecaster.md`)**](./auto_forecaster.md) · **29 Benchmarks & AlphaEvolve (`auto_forecaster_benchmarks.md`)** · [**7-Runtime Cloud Scale & AutoML (`auto_forecaster_cloud_scale.md`)**](./auto_forecaster_cloud_scale.md) · [**AlphaEvolve Runtime Harness (`alphaevolve_runner.md`)**](./alphaevolve_runner.md)

This page contains the complete offline and open-data empirical evaluation of [`Forecaster.build_config()`](api/sdk.md) and [`Forecaster.auto()`](api/sdk.md) across all **4 exposure tiers (`first_light`, `wide_field`, `deep_sky`, `deep_field`) $\times$ `{autofocus=False, autofocus=True}`**:

1. **[Executive Summary: The 4-Tier × `autofocus` Progression Curve](#1-executive-summary-the-4-tier--autofocus-progression-curve)** (`Tables 1A & 1B`)
2. **[20 Synthetic Ground-Truth Recipes (`dgp_01`–`dgp_20`) Across All 4 Exposure Tiers](#2-20-synthetic-ground-truth-recipes-dgp_01dgp_20-across-all-4-exposure-tiers)** (`Tables 2A, 2B & 2C`)
3. **[How AlphaEvolve on Gemini Enterprise Evolved the Policy](#3-how-alphaevolve-on-gemini-enterprise-evolved-the-policy)** (`Table 3`)
4. **[9 Real-World Public Open-Data Benchmarks Across All 4 Exposure Tiers (M5, Favorita, M4, Monash, ETT)](#4-9-real-world-public-open-data-benchmarks-across-all-4-exposure-tiers-m5-favorita-m4-monash-ett)** (`Tables 4A, 4B & 4C`)
5. **[Built-In AlphaEvolve Policy (`Forecaster.auto`) vs. Opt-In Runtime Evolution (`Forecaster.evolve`)](#5-built-in-alphaevolve-policy-forecasterauto-vs-opt-in-runtime-evolution-forecasterevolve)** (`Table 5`)

---

## 1. Executive Summary: The 4-Tier × `autofocus` Progression Curve

Running every benchmark across all four exposure tiers—both as a single pass (`autofocus=False`) and as an iterative multi-branch tournament (`autofocus=True`)—reveals a clear, predictable progression in both model coverage and accuracy:

### Table 1A — What Each Exposure Tier Unlocks (CV Folds, Quantiles & Model Portfolio)

*How to read this table:* **`CV Folds`** = number of expanding out-of-fold validation windows (`n_folds`); **`Quantile Grid`** = prediction interval quantiles emitted; **`Max autofocus Depth`** = maximum tournament passes when `autofocus=True`.

| Exposure Tier | CV Folds | Quantile Grid | Max `autofocus` Depth | Candidate Families, `@global` Models & Foundation Backbones Added |
| :--- | :---: | :---: | :---: | :--- |
| **`first_light`** | `1` fold | `[0.1, 0.9]` | `2` passes | Fast local baselines (`naive_seasonal`, `theta`, `holtwinters`) + `lightgbm`; Pass 2/3 selectively adds `@transform_log1p`, `@direct`, `croston`, and `mint_nonneg` |
| **`wide_field`** *(default)* | `2` folds | `[0.1, 0.9]` | `2` passes | Adds `xgboost`, `autoarima`, `mstlm`, zero-shot `chronos_bolt` (`chronos_bolt@global` on `medium+` panels), and `timesfm` (on `bigquery`) |
| **`deep_sky`** | `3` folds | `[0.1, 0.9]` | `3` passes | Unlocks `@global` neural models (`tide@global`, `tsmixer@global`, `tft@global`), `chronos_bolt@global`, `timesfm_torch@global`, `catboost`, and Optuna HPO (`6` trials) |
| **`deep_field`** | `3` folds | `7` quantiles (`0.05..0.95`) | `4` passes | Unlocks `chronos2@global` (`amazon/chronos-2` with `cross_learning=True`), `patchtst@global`, `stl_bagging`, `tbats`, HPO (`12` trials), and optional `include_automl=True` (`vertex_tide`, `vertex_l2l`, `automl`) |

### Table 1B — Measured Accuracy, Wall-Clock Time & Cell Borrowing by Tier (9 Public Benchmarks Mean)

*How to read the scores in this table:* **`OOF FVA %`** = Out-of-Fold Forecast Value Added (`%` error reduction vs. `naive_seasonal` across CV folds; **$\uparrow$ higher is better**) · **`autofocus OOF Lift`** = additional FVA `%` gained by multi-pass `autofocus=True` over Pass 1 (**$\uparrow$ higher is better**) · **`Held-Out Test Lift`** = `%` error reduction vs. `naive_seasonal` on the unseen post-cutoff test window (`oracle_future_df`; **$\uparrow$ higher is better**) · **`Wall-Clock (`s`)`** = mean elapsed runtime in seconds (**$\downarrow$ lower is faster**) · **`Mean Models (`Exec` / `Borrowed`)`** = newly fitted models (`Exec`) vs. finished models reused from earlier passes at zero compute cost (`Borrowed`; **$\uparrow$ higher `Borrowed` saves more compute**).

| Exposure Tier | Pass 1 OOF FVA (`autofocus=False`) $\uparrow$ | Multi-Pass OOF FVA (`autofocus=True`) $\uparrow$ | `autofocus` OOF Lift $\uparrow$ | Pass 1 Held-Out Test Lift $\uparrow$ | Best Held-Out Test Lift $\uparrow$ | Pass 1 Wall-Clock $\downarrow$ | `autofocus` Wall-Clock $\downarrow$ | Mean Models (`Exec` / `Borrowed`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`first_light`** | `+22.24%` | **`+27.67%`** | **`+5.43%`** | `+13.79%` | `+16.69%` | `24.1s` | `91.4s` | `10.4` / `7.0` |
| **`wide_field`** *(default)* | `+20.93%` | **`+21.94%`** *(2-fold CV)* | `+1.01%` | `+16.15%` | `+16.34%` | `85.9s` | `138.4s` | `11.1` / `11.6` |
| **`deep_sky`** | `+29.36%` | **`+30.43%`** | `+1.07%` | **`+18.51%`** | **`+18.51%`** | `181.5s` | `267.1s` | `13.4` / `7.0` |
| **`deep_field`** | **`+31.36%`** | **`+32.66%`** | `+1.30%` | `+12.79%` | `+13.06%` | `344.9s` | `396.4s` *(`227.2s` net)* | `9.7` / **`16.0`** |

> [!TIP]
> **When to Use Each Combination:**
> - **`tier="first_light", autofocus=True` (`~90s` local):** Best for rapid dataset triage and feature-engineering discovery. Multi-branch `autofocus` delivers its largest single-tier accuracy jump here (**`+5.43%`** additional OOF FVA on public benchmarks and up to **`+43.93%`** error reduction on synthetic DGPs) because Pass 1 starts with a lean 3-model core and Pass 2/3 selectively injects `@transform_log1p`, `@direct`, `exog_lags`, or `mint_nonneg` only when diagnostics warrant it.
> - **`tier="wide_field", autofocus=True` (Default, `~138s` local):** Best balanced production default. Moving from `1` fold to `2` expanding CV folds eliminates single-fold selection overfitting (lifting mean Pass 1 Held-Out Test Lift from `+13.79%` to **`+16.15%`**) while borrowing an average of `11.6` model evaluations across passes.
> - **`tier="deep_sky"` & `tier="deep_field"` (`3`-fold CV + `@global` Neural & Foundation Models):** Best when cross-series patterns, zero-shot pretrained foundation representations (`chronos_bolt@global`, `chronos2@global`, `timesfm_torch@global`), or full 7-quantile probabilistic fans are required. On the 9 public benchmarks, unlocking `@global` foundation and neural models at `deep_sky` and `deep_field` lifts mean OOF FVA to **`+30.43%`–`+32.66%`** and achieves the highest mean Held-Out Test Lift (**`+18.51%`** at `deep_sky`).

---

## 2. 20 Synthetic Ground-Truth Recipes (`dgp_01`–`dgp_20`) Across All 4 Exposure Tiers

A **Data-Generating Process (DGP)** is a synthetic dataset where we wrote the underlying mathematical equations ourselves—so we know the exact ground-truth structure before the forecaster ever sees the data. For example, in `dgp_04_true_vs_decoy_covs`, sales are driven by 2 true covariates (`promo_depth` and `price_index`) mixed with 3 columns of pure random noise (`noise_a`, `noise_b`, `noise_c`), testing whether `autofocus` prunes the 3 decoy columns in Pass 2. In `dgp_13_all_decoy_univariate_wins`, all 4 exogenous columns are random noise—verifying that `autofocus` drops 100% of the decoys and crowns a fast univariate champion.

All 20 test datasets are built into [`scale_forecasting.benchmarks`](api/benchmarks.md) (`generate_benchmark_dataset("<name>")`). To keep every view easy to scan, the results are organized into three tables below:

- **Table 2A — Scenario & Diagnostic Response:** What real-world challenge each DGP simulates and how `autofocus` responds.
- **Table 2B — 4-Tier × `autofocus` Error Progression:** Side-by-side numeric error scores (`Pass 1` vs. `Autofocus`) across all 4 tiers.
- **Table 2C — Runtime, Cell Borrowing & Single-Model vs. Foundation-Model Comparison:** Wall-clock seconds, borrowed model counts, `#1 Single Model`, and `#1 @global / Foundation Model`.

### Table 2A — What Each Synthetic Recipe (`dgp_01`–`dgp_20`) Simulates & How `autofocus` Solves It

*How to read the scores in this table:* **`Metric`** = primary decision metric auto-selected for the regime (`wape` for smooth/promotional demand, `mase` for zero-heavy intermittent demand, `pinball_loss` for asymmetric $q=0.90$ service-level inventory) · **`Peak FVA %`** = Forecast Value Added (`%` out-of-fold error reduction vs. `naive_seasonal`; **$\uparrow$ higher is better**, e.g. `+77.57%` means the champion eliminated more than three-quarters of baseline error).

| Benchmark | Metric | Real-World Scenario Simulated | Diagnostic Trigger & `autofocus` Action | Overall Winning Champion | Peak FVA % $\uparrow$ |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **`dgp_01_smooth_seasonal`** | `wape` | Clean retail series with steady trend + weekly & yearly seasonality | Stops after Pass 1 *(already optimal)* | `ensemble_horizon_specialist` | **`+40.33%`** |
| **`dgp_02_multiplicative_log`** | `wape` | Fast-growing series where seasonal swings widen as volume grows | Pass 2 adds `@transform_log1p` & `@direct` specialists | `ensemble_horizon_specialist` | **`+44.89%`** |
| **`dgp_03_intermittent_lumpy`** | `mase` | Spare-parts / lumpy demand where most days have zero sales | Pass 2 adds `croston` & `naive_mean`; `chronos2@global_univariate` leads `deep_field` | `ensemble_horizon_specialist` | **`+40.07%`** |
| **`dgp_04_true_vs_decoy_covs`** | `wape` | 2 real promo/price drivers mixed with 3 fake random-noise columns | Drops the 3 noise columns in Pass 2 and blends in Pass 3 | `sarimax` | **`+77.57%`** |
| **`dgp_05_horizon_crossover`** | `wape` | Short-term momentum dominates days 1–5; weekly cycle dominates days 6–14 | Pass 2 adds `@direct` horizon models + `horizon_specialist` blend | `ensemble_horizon_specialist` | **`+59.87%`** |
| **`dgp_06_structural_break_spikes`** | `wape` | Sudden $+45\%$ permanent jump in demand plus 3 extreme one-day spikes | Pass 2 turns on `level_shift` + `stl_resid` spike scrubbing | `ensemble_horizon_specialist` | **`+58.53%`** |
| **`dgp_07_censored_stockouts`** | `wape` | Inventory ran out on peak days (`stockout_flag=1`), clipping true demand | Pass 2 fits `xgboost` & `regression_lags` over reconstructed demand | `ensemble_median` | **`+71.86%`** |
| **`dgp_08_hierarchical_nonneg`** | `wape` | 2-level `region -> store` hierarchy where store forecasts must sum to region | Reconciles via `mint_nonneg` (`0.0` coherence gap); `chronos2@global_univariate_mint_nonneg` #1 single | `ensemble_horizon_specialist` | **`+43.39%`** |
| **`dgp_09_cross_series_global`** | `wape` | 12 related items sharing a common nonlinear promotion curve | Pass 2 expands cross-series & `@direct` horizon-specialist models | `ensemble_nnls` | **`+73.37%`** |
| **`dgp_10_newsvendor_service`** | `pinball_loss` | High stockout penalty ($9\times$ holding cost) requiring a 90th-percentile forecast | Tunes quantile learners around `0.90`; `chronos2@global_univariate` #1 single model | `ensemble_horizon_specialist` | **`+42.45%`** |
| **`dgp_11_cold_start_mixed`** | `wape` | Mature products ($160$ days history) mixed with brand-new launches ($18$ days) | Pass 2 turns on `regime_routing=True` & cross-series pooling | `ensemble_horizon_specialist` | **`+40.62%`** |
| **`dgp_12_already_optimal_white_noise`** | `wape` | Flat average plus pure random noise (nothing for complex models to learn) | Stops immediately after Pass 1 *(0 wasted passes)* | `ensemble_horizon_specialist` | **`+42.95%`** |
| **`dgp_13_all_decoy_univariate_wins`** | `wape` | 4 pure random-noise feature columns (`decoy_1`..`decoy_4`) with zero true signal | Drops 100% of decoy columns; univariate champion beats multivariate ML | `ensemble_horizon_specialist` | **`+41.78%`** |
| **`dgp_14_large_hierarchy_intermittent_leaves`** | `mase` | 3-level `region -> warehouse -> sku` hierarchy with intermittent leaves & smooth root | Applies `wls_struct` + `mint_nonneg` reconciliation across all 3 levels | `ensemble_horizon_specialist` | **`+38.26%`** |
| **`dgp_15_scale_crossover_panel`** | `wape` | Scalable panel (`n_series=6` micro vs. `n_series=120` medium) with shared seasonal curve | Adapts portfolio from local stats (`micro`) to global cross-series pooling (`medium`) | `ensemble_horizon_specialist` | **`+44.08%`** |
| **`dgp_16_monthly_trading_days`** | `wape` | Monthly (`freq="MS"`) revenue panel driven by calendar business-day counts | Enables `trading_days=True` calendar features & `@direct` lags | `ensemble_nnls` | **`+82.71%`** |
| **`dgp_17_missing_gaps_and_leading_zeros`** | `wape` | Series with pre-launch leading zeros (`15%`) and missing date gaps (`5%`) | Enables `trim_leading_zeros=True` and `grid_policy="fill_zero"` | `ensemble_horizon_specialist` | **`+39.33%`** |
| **`dgp_18_promo_lag_pullforward`** | `wape` | Promotions that lift current-day sales (`t`) and pull forward demand from `t+1, t+2` | Synthesizes `exog_lags` (`[1, 2, 7]`) on `promo_flag` in Branch `2B` | `ensemble_nnls` | **`+94.08%`** |
| **`dgp_19_hourly_diurnal_weekly_load`** | `wape` | Hourly (`freq="h"`) transformer load with $24\text{h}$ diurnal + $168\text{h}$ weekly cycle + temperature | Multi-seasonal `fourier` ($24\text{h}$, $12\text{h}$, $168\text{h}$) + `chronos2@global` overall winner (`+73.72%` FVA) | `chronos2@global` | **`+73.72%`** |
| **`dgp_20_monthly_regime_drift_damped`** | `wape` | Monthly (`freq="MS"`, $T=60$) macro/tourism series with 12-month cycle, break & damped trend | Enables monthly lags (`[1, 2, 3, 6, 12]`), `level_shift`, and `@direct` specialists | `ensemble_horizon_specialist` | **`+69.46%`** |

### Table 2B — 4-Tier × `autofocus` Error Progression & Held-Out Test Lift Across All 20 DGPs

*How to read the scores in this table:* **All `Pass 1`, `autofocus`, and `Full Bank` columns** report raw out-of-fold error on the row's `Metric` (`wape` = Weighted Absolute Percentage Error, `mase` = Mean Absolute Scaled Error vs. in-sample seasonal naive, `pinball_loss` = $q=0.90$ quantile pinball loss; **$\downarrow$ lower is better**, `0.0000` is a perfect forecast; bold marks where `autofocus=True` reduced error below Pass 1 or achieved the lowest overall error) · **`Peak FVA %`** = `%` out-of-fold error reduction vs. `naive_seasonal` (**$\uparrow$ higher is better**) · **`Peak Test Lift %`** = `%` error reduction vs. `naive_seasonal` on the unseen post-cutoff oracle test window (`oracle_future_df`; **$\uparrow$ higher is better**).

| Benchmark | Metric | `first_light` Pass 1 $\downarrow$ | `first_light` `autofocus` $\downarrow$ | `wide_field` Pass 1 $\downarrow$ | `wide_field` `autofocus` $\downarrow$ | `deep_sky` Pass 1 $\downarrow$ | `deep_sky` `autofocus` $\downarrow$ | `deep_field` Pass 1 $\downarrow$ | `deep_field` `autofocus` $\downarrow$ | `deep_field` Full Bank $\downarrow$ | Peak FVA % $\uparrow$ | Peak Test Lift % $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dgp_01_smooth_seasonal`** | `wape` | `0.0282` | `0.0282` | `0.0260` | `0.0260` | `0.0260` | `0.0260` | `0.0261` | `0.0261` | **`0.0219`** | **`+40.33%`** | **`+32.07%`** |
| **`dgp_02_multiplicative_log`** | `wape` | `0.1531` | **`0.0931`** | `0.1204` | `0.1204` | `0.1179` | `0.1179` | `0.1171` | `0.1171` | **`0.0996`** | **`+44.89%`** | **`+42.68%`** |
| **`dgp_03_intermittent_lumpy`** | `mase` | `0.8005` | **`0.7440`** | `0.8060` | `0.8060` | `0.7294` | `0.7294` | `0.7344` | `0.7344` | **`0.6902`** | **`+40.07%`** | **`+45.71%`** |
| **`dgp_04_true_vs_decoy_covs`** | `wape` | `0.0498` | **`0.0473`** | `0.0359` | **`0.0321`** | `0.0329` | **`0.0314`** | `0.0427` | **`0.0380`** | **`0.0217`** | **`+77.57%`** | **`+30.09%`** |
| **`dgp_05_horizon_crossover`** | `wape` | `0.1333` | `0.1333` | `0.1214` | **`0.1202`** | `0.1169` | `0.1169` | `0.1155` | **`0.1150`** | **`0.0954`** | **`+59.87%`** | **`+33.77%`** |
| **`dgp_06_structural_break_spikes`** | `wape` | `0.0212` | **`0.0209`** | `0.0501` | **`0.0498`** | `0.0392` | `0.0392` | `0.0400` | `0.0400` | **`0.0434`** | **`+58.53%`** | `-45.34%` |
| **`dgp_07_censored_stockouts`** | `wape` | `0.0541` | **`0.0411`** | `0.0438` | `0.0438` | `0.0415` | `0.0415` | `0.0439` | **`0.0411`** | **`0.0306`** | **`+71.86%`** | **`+51.55%`** |
| **`dgp_08_hierarchical_nonneg`** | `wape` | `0.0484` | **`0.0425`** | `0.0423` | `0.0423` | `0.0448` | `0.0448` | `0.0453` | `0.0453` | **`0.0360`** | **`+43.39%`** | **`+26.87%`** |
| **`dgp_09_cross_series_global`** | `wape` | `0.0692` | **`0.0388`** | `0.0509` | `0.0509` | `0.0551` | `0.0551` | `0.0571` | `0.0571` | **`0.0296`** | **`+73.37%`** | **`+20.78%`** |
| **`dgp_10_newsvendor_service`** | `pinball_loss` | `0.0865` | `0.0865` | `0.0878` | `0.0878` | `0.0895` | `0.0895` | `0.0893` | `0.0893` | **`0.0679`** | **`+42.45%`** | **`+49.81%`** |
| **`dgp_11_cold_start_mixed`** | `wape` | `0.0236` | **`0.0209`** | `0.0221` | `0.0221` | `0.0219` | `0.0219` | `0.0218` | `0.0218` | **`0.0176`** | **`+40.62%`** | **`+20.86%`** |
| **`dgp_12_already_optimal_white_noise`** | `wape` | `0.0153` | `0.0153` | `0.0156` | `0.0156` | `0.0155` | `0.0155` | `0.0155` | `0.0155` | **`0.0127`** | **`+42.95%`** | **`+38.99%`** |
| **`dgp_13_all_decoy_univariate_wins`** | `wape` | `0.0173` | **`0.0161`** | `0.0157` | `0.0157` | `0.0160` | `0.0160` | `0.0162` | `0.0162` | **`0.0121`** | **`+41.78%`** | **`+30.57%`** |
| **`dgp_14_large_hierarchy_intermittent_leaves`** | `mase` | `0.8463` | **`0.7864`** | `0.7055` | `0.7055` | `0.7425` | `0.7425` | `0.7514` | `0.7514` | **`0.6234`** | **`+38.26%`** | **`+54.53%`** |
| **`dgp_15_scale_crossover_panel`** | `wape` | `0.0224` | `0.0224` | `0.0191` | `0.0191` | `0.0198` | `0.0198` | `0.0199` | `0.0199` | **`0.0144`** | **`+44.08%`** | **`+35.86%`** |
| **`dgp_16_monthly_trading_days`** | `wape` | `0.0115` | **`0.0075`** | `0.0113` | `0.0113` | `0.0110` | `0.0110` | `0.0111` | `0.0111` | **`0.0099`** | **`+82.71%`** | **`+49.10%`** |
| **`dgp_17_missing_gaps_and_leading_zeros`** | `wape` | `0.0183` | `0.0183` | `0.0176` | `0.0176` | `0.0166` | `0.0166` | `0.0167` | `0.0167` | **`0.0146`** | **`+39.33%`** | **`+21.83%`** |
| **`dgp_18_promo_lag_pullforward`** | `wape` | `0.0621` | `0.0621` | `0.0277` | `0.0277` | `0.0232` | **`0.0230`** | `0.0255` | `0.0255` | **`0.0134`** | **`+94.08%`** | **`+20.58%`** |
| **`dgp_19_hourly_diurnal_weekly_load`** | `wape` | `0.0210` | `0.0210` | `0.0150` | `0.0150` | `0.0164` | `0.0164` | `0.0147` | **`0.0147`** | **`0.0147`** | **`+73.72%`** | **`+66.16%`** |
| **`dgp_20_monthly_regime_drift_damped`** | `wape` | `0.0424` | **`0.0355`** | `0.0392` | **`0.0356`** | `0.0537` | **`0.0476`** | `0.0530` | **`0.0465`** | **`0.0465`** | **`+69.46%`** | `-15.73%` |

### Table 2C — Runtime (`s`), Borrowed Models (`b=N`) & `#1 Single` vs. `#1 @global / Foundation` Model Across All 20 DGPs

*How to read the scores in this table:* **`Wall (`s`)`** = total session elapsed seconds (**$\downarrow$ lower is faster**) · **`Borrowed (`b=N`)`** = number of finished `(model, spec_hash)` evaluations reused from earlier passes at zero compute cost (**$\uparrow$ higher saves more compute**) · **`#1 Single` / `#1 @global / Foundation` / `Overall Champion` (`Score`)** = raw out-of-fold error on the row's primary metric (`wape`, `mase`, or `pinball_loss`; **$\downarrow$ lower is better**).

| Benchmark | `first_light` Wall $\downarrow$ (`Borrowed` $\uparrow$) | `wide_field` Wall $\downarrow$ (`Borrowed` $\uparrow$) | `deep_sky` Wall $\downarrow$ (`Borrowed` $\uparrow$) | `deep_field` Wall $\downarrow$ (`Borrowed` $\uparrow$) | #1 Single Non-Ensemble Model (`Error` $\downarrow$) | #1 `@global` / Foundation Model (`Error` $\downarrow$) | Overall Champion (`Error` $\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **`dgp_01_smooth_seasonal`** | `12s` (`b=0`) | `48s` (`b=0`) | `152s` (`b=0`) | `256s` (`b=0`) | `holtwinters` (`0.0260`) | `chronos2@global_univariate` (`0.0264`) | `ensemble_horizon_specialist` (`0.0219`) |
| **`dgp_02_multiplicative_log`** | `45s` (`b=7`) | `58s` (`b=0`) | `116s` (`b=0`) | `98s` (`b=0`) | `auto_theta` (`0.1175`) | `chronos2@global_univariate` (`0.1272`) | `ensemble_horizon_specialist` (`0.0996`) |
| **`dgp_03_intermittent_lumpy`** | `10s` (`b=4`) | `26s` (`b=0`) | `77s` (`b=0`) | `57s` (`b=0`) | `chronos2@global_univariate` (`0.7662`) | `chronos2@global_univariate` (`0.7662`) | `ensemble_horizon_specialist` (`0.6902`) |
| **`dgp_04_true_vs_decoy_covs`** | `17s` (`b=2`) | `58s` (`b=8`) | `347s` (`b=11`) | `252s` (`b=9`) | `sarimax` (`0.0217`) | `chronos2@local_cov_all` (`0.0315`) | `sarimax` (`0.0217`) |
| **`dgp_05_horizon_crossover`** | `19s` (`b=0`) | `182s` (`b=26`) | `490s` (`b=17`) | `350s` (`b=15`) | `kalman` (`0.1169`) | `tft@global` (`0.1475`) | `ensemble_horizon_specialist` (`0.0954`) |
| **`dgp_06_structural_break_spikes`** | `37s` (`b=0`) | `126s` (`b=19`) | `933s` (`b=30`) | `174s` (`b=11`) | `ucm` (`0.0464`) | `chronos2@global_univariate` (`0.0483`) | `ensemble_horizon_specialist` (`0.0434`) |
| **`dgp_07_censored_stockouts`** | `61s` (`b=4`) | `188s` (`b=0`) | `351s` (`b=0`) | `180s` (`b=15`) | `sarimax` (`0.0353`) | `chronos2@global_cov_all` (`0.1566`) | `ensemble_median` (`0.0306`) |
| **`dgp_08_hierarchical_nonneg`** | `84s` (`b=3`) | `295s` (`b=0`) | `1106s` (`b=0`) | `157s` (`b=0`) | `chronos2@global_univariate_mint_nonneg` (`0.0437`) | `chronos2@global_univariate_mint_nonneg` (`0.0437`) | `ensemble_horizon_specialist` (`0.0360`) |
| **`dgp_09_cross_series_global`** | `65s` (`b=4`) | `100s` (`b=0`) | `366s` (`b=0`) | `169s` (`b=15`) | `sarimax` (`0.0298`) | `chronos2@local_cov_all` (`0.0405`) | `ensemble_nnls` (`0.0296`) |
| **`dgp_10_newsvendor_service`** | `62s` (`b=0`) | `164s` (`b=0`) | `590s` (`b=0`) | `251s` (`b=0`) | `chronos2@global_univariate` (`0.0875`) | `chronos2@global_univariate` (`0.0875`) | `ensemble_horizon_specialist` (`0.0679`) |
| **`dgp_11_cold_start_mixed`** | `46s` (`b=3`) | `73s` (`b=0`) | `475s` (`b=0`) | `101s` (`b=0`) | `auto_theta` (`0.0222`) | `chronos2@local_univariate` (`0.0231`) | `ensemble_horizon_specialist` (`0.0176`) |
| **`dgp_12_already_optimal_white_noise`** | `18s` (`b=0`) | `77s` (`b=0`) | `277s` (`b=0`) | `104s` (`b=0`) | `auto_ces` (`0.0156`) | `chronos2@global_univariate` (`0.0157`) | `ensemble_horizon_specialist` (`0.0127`) |
| **`dgp_13_all_decoy_univariate_wins`** | `42s` (`b=2`) | `79s` (`b=11`) | `382s` (`b=0`) | `123s` (`b=0`) | `autoets` (`0.0159`) | `chronos2@global_cov_all` (`0.0167`) | `ensemble_horizon_specialist` (`0.0121`) |
| **`dgp_14_large_hierarchy_intermittent_leaves`** | `147s` (`b=4`) | `253s` (`b=0`) | `640s` (`b=0`) | `176s` (`b=0`) | `naive_mean` (`0.7284`) | `chronos2@global_univariate` (`0.7491`) | `ensemble_horizon_specialist` (`0.6234`) |
| **`dgp_15_scale_crossover_panel`** | `43s` (`b=0`) | `184s` (`b=0`) | `537s` (`b=0`) | `143s` (`b=0`) | `holtwinters` (`0.0191`) | `chronos2@global_univariate` (`0.0196`) | `ensemble_horizon_specialist` (`0.0144`) |
| **`dgp_16_monthly_trading_days`** | `69s` (`b=3`) | `101s` (`b=0`) | `400s` (`b=0`) | `64s` (`b=0`) | `kalman` (`0.0108`) | `chronos2@global_univariate` (`0.0169`) | `ensemble_nnls` (`0.0099`) |
| **`dgp_17_missing_gaps_and_leading_zeros`** | `27s` (`b=0`) | `131s` (`b=0`) | `346s` (`b=0`) | `121s` (`b=0`) | `autoets` (`0.0174`) | `chronos2@global_univariate` (`0.0187`) | `ensemble_horizon_specialist` (`0.0146`) |
| **`dgp_18_promo_lag_pullforward`** | `58s` (`b=2`) | `78s` (`b=8`) | `266s` (`b=11`) | `266s` (`b=9`) | `sarimax` (`0.0136`) | `chronos2@global_cov_all` (`0.0532`) | `ensemble_nnls` (`0.0134`) |
| **`dgp_19_hourly_diurnal_weekly_load`** | `46s` (`b=2`) | `71s` (`b=0`) | `350s` (`b=13`) | `174s` (`b=0`) | `chronos2@global` (`0.0147`) | `chronos2@global` (`0.0147`) | `chronos2@global` (`0.0147`) |
| **`dgp_20_monthly_regime_drift_damped`** | `29s` (`b=10`) | `66s` (`b=18`) | `88s` (`b=10`) | `108s` (`b=12`) | `holtwinters` (`0.0627`) | `chronos2@global` (`0.0844`) | `ensemble_horizon_specialist` (`0.0465`) |

> [!IMPORTANT]
> **Where Foundation & `@global` Models Win on the 20 Synthetic DGPs:**
> - **Outright Overall Champion on Sub-Daily Multi-Seasonality (`dgp_19_hourly_diurnal_weekly_load`):** Zero-shot **`chronos2@global`** wins outright across all 74 model variants and ensembles (`WAPE = 0.0147`, **`+73.72%` FVA**), beating every individual statistical, tree, and deep-learning model.
> - **#1 Single Non-Ensemble Model on Lumpy, Hierarchical & Newsvendor Regimes:** **`chronos2@global_univariate`** is the #1 single model on `dgp_03_intermittent_lumpy` (`MASE = 0.7662`), `dgp_08_hierarchical_nonneg` (`chronos2@global_univariate_mint_nonneg` `WAPE = 0.0437`), and `dgp_10_newsvendor_service` (`0.0875`), and finishes within `0.0001`–`0.0008` of the #1 single model on `dgp_01`, `dgp_06`, `dgp_11`, `dgp_12`, `dgp_13`, `dgp_15`, and `dgp_17`.
> - **Where Covariate-Aware Ensembles & Trees Pull Ahead:** On regimes driven by exogenous promotions, stockouts, or calendar trading days (`dgp_04`, `dgp_07`, `dgp_09`, `dgp_16`, `dgp_18`), `autofocus` pairs `@direct` gradient-boosted trees (`lightgbm@direct`, `xgboost@cov_all`) and `regression_lags` with `chronos2@global` inside `ensemble_horizon_specialist` and `ensemble_nnls`, cutting Pass 1 error by **`+24.03%` to `+43.93%`**.

---

## 3. How AlphaEvolve on Gemini Enterprise Evolved the Policy

Instead of hand-picking the thresholds and starting model lists in [`scale_forecasting.auto_policy`](api/auto.md), we used **AlphaEvolve on Gemini Enterprise** ([`Google-Cloud-AI/alphaevolve-on-googlecloud`](https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud) via `discoveryengine.googleapis.com/v1alpha`) with **`gemini-3.8-flash`** and **`gemini-4-argon`** to evolve the policy against the benchmark suite.

### Table 3 — AlphaEvolve Policy Evolution Across 5 Rounds

*How to read the scores in this table:* **`Overall Score`** = composite AlphaEvolve tournament fitness across the benchmark suite (**$\uparrow$ higher is better**) · **`Average FVA vs. Baseline`** = mean `%` out-of-fold error reduction vs. `naive_seasonal` (**$\uparrow$ higher is better**) · **`Average Multi-Pass autofocus Gain`** = mean `%` error reduction from Pass 1 to the final `autofocus` pass, or mean Held-Out Test Lift in Rounds 4–5 (**$\uparrow$ higher is better**) · **`Wasted-Pass Penalty`** = penalty for triggering unnecessary follow-up passes on pure white-noise series (**`0.00` is optimal**; negative values penalize wasted compute).

| Evolution Stage | Overall Score $\uparrow$ | Average FVA vs. Baseline $\uparrow$ | Average Multi-Pass `autofocus` Gain $\uparrow$ | Wasted-Pass Penalty (`0.00` Best) | What AlphaEvolve Discovered |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Initial Hand-Written Seed** | `26.08` | `+18.25%` | `+2.26%` | `-8.00` | Putting too many models into Pass 1 (`first_light`) with 1-fold per-series selection overfit noisy series and wasted a second pass on pure white noise (`dgp_12`). |
| **Generation 1 — Smart Early Stopping** | `34.08` | `+18.25%` | `+2.26%` | `0.00` | Added a residual white-noise stop gate so `autofocus` stops right after Pass 1 when no learnable pattern remains. |
| **Generation 2 — Lean Pass 1 + Pass 2 Upgrades** | `52.86` | `+19.65%` | `+11.61%` | `0.00` | Keeping Pass 1 lean (`naive_seasonal`, `theta`, `lightgbm`) and saving `@transform_log1p` and `@direct` for Pass 2 boosted Pass 2 gains more than $5\times$. |
| **Generation 3 — Single-Fold Overfit Guard** | `56.03` | `+20.63%` | `+12.46%` | `0.00` | Using `"best_single"` selection when `n_folds=1` (and per-series `"champion"` only when `n_folds >= 2`) eliminated 1-fold overfitting. |
| **Round 2 — Multi-Branch DAG + Domain Specialists** | `74.82` | `+41.96%` | `+24.16%` | `0.00` | Splitting Pass 2 into parallel branches (`2A`, `2B`, `2C`) + Level 3 synthesis (`3`) and adding intermittent, cold-start, hierarchy (`mint_nonneg`), and stockout specialists solved all 12 test regimes. |
| **Round 3A & 3B — 18-DGP Scale-Crossover & Cloud Holdouts (`300` Generations)** | **`71.07`** *(18-DGP scale-aware score)* | **`+45.40%`** | **`+25.60%` regret reduction** | **`0.00`** | Evolved `n_series` scale-regime cutoffs (`micro=9`, `small=99`, `medium=999`, `large=9,999`), multi-anchor baseline rules (`100%` rule accuracy across `18/18` DGPs), and hierarchical reconciliation ladders; verified on `60/60` multi-branch tournaments and unseen cloud holdouts up to `25,000` series (`+50.07%` MASE lift / `+45.04%` WAPE reduction on `holdout_25000`). |
| **Round 4 — Real-World Open-Data Generalization & Cross-Fitted Ensembling** | **`9 / 9` Non-Negative Test Lift** | **`+24.30%` OOF FVA** | **`+14.63%` Mean Test Lift** | **`0.00`** | Added leave-one-fold-out cross-fitting for `ensemble_champion` and smoothed step-window shrinkage for `ensemble_horizon_specialist`, defaulted accuracy-mode point forecasts to unbiased `"raw"` model outputs, added dynamic quota preflight across all 6 cloud runtimes (including Dataproc Serverless Spark), and verified across 9 public real-world open-data benchmarks. |
| **Round 5 — 20-DGP Sub-Daily Multi-Seasonality, Frequency-Matched Lags & Finite-Sample Intervals (`200` Generations)** | **`73.71`** *(20-DGP score · `100%` rule accuracy)* | **`+25.18%` OOF FVA** | **`+14.92%` Mean Test Lift** | **`0.00`** | Fixed physical annual (`365.25d`) and sub-daily (`24h`/`12h`/`168h`) Fourier harmonics in `features.py`, added frequency-matched lag engineering (`_frequency_matched_lags`) for hourly and short monthly/weekly/quarterly series, and upgraded ensemble prediction intervals to finite-sample conformal order statistics (`_empirical_quantile`). |

---

## 4. 9 Real-World Public Open-Data Benchmarks Across All 4 Exposure Tiers (M5, Favorita, M4, Monash, ETT)

Beyond the 20 synthetic ground-truth DGPs above, [`scale_forecasting.benchmarks`](api/benchmarks.md) includes on-demand fetchers ([`list_public_benchmarks()`](api/benchmarks.md), [`public_benchmark_catalog()`](api/benchmarks.md), and [`fetch_public_benchmark()`](api/benchmarks.md)) for **9 canonical academic and competition time-series datasets** spanning retail hierarchies, grocery promotions, macroeconomics, healthcare patient counts, intermittent spare parts, and hourly electricity transformer loads.

> [!NOTE]
> **Zero Data Rehosting & Full Attribution:** `scale-forecasting` **never rehosts** external benchmark data in the Git repository or Python package wheel. Calling `fetch_public_benchmark("<name>")` downloads the dataset on demand from its official public archive (Nixtla open mirrors, Zenodo Monash Time Series Forecasting Archive, or ETT GitHub) into a local temporary cache (`/tmp/scale_forecasting_public_benchmarks`) and returns a ready-to-run `BenchmarkDataset` carrying `panel_df`, `future_df`, `oracle_future_df` (held-out test actuals), official URLs, license metadata, and academic citations.

```python
from scale_forecasting import Forecaster, fetch_public_benchmark, public_benchmark_catalog

# 1. Inspect the 9 public open-data benchmarks, licenses, and citations
print(public_benchmark_catalog()[["name", "domain", "freq", "horizon", "license"]])

# 2. Fetch M5 Walmart or Corporación Favorita on demand and run multi-branch autofocus
ds = fetch_public_benchmark("m5_walmart", n_series=16)
f = Forecaster.auto(ds.panel_df, future_df=ds.future_df, tier="wide_field", autofocus=True)
print(f.assess())
```

### Table 4A — Dataset Catalog, Peak Accuracy & #1 Single Model Across the 9 Open-Data Benchmarks

*How to read the scores in this table:* **`Peak OOF FVA vs. Seasonal Naive`** = `%` out-of-fold cross-validation error reduction vs. `naive_seasonal` (**$\uparrow$ higher is better**) · **`Peak Held-Out Test Lift vs. SNaive`** = `%` error reduction vs. `naive_seasonal` on the unseen post-cutoff test horizon (`oracle_future_df`; **$\uparrow$ higher is better**) · **`#1 Single Non-Ensemble Model (`Score`)`** = raw out-of-fold error (`WAPE`, or `MASE` on `monash_car_parts`; **$\downarrow$ lower is better**).

| Public Benchmark | Domain & Structural Features | Freq / Horizon | Best Tier & Winning Model (`Pass`) | Peak OOF FVA vs. Seasonal Naive $\uparrow$ | Peak Held-Out Test Lift vs. `SNaive` $\uparrow$ | #1 Single Non-Ensemble Model (`Error` $\downarrow$) | License & Source |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **`m5_walmart`** | Walmart retail unit sales (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` hierarchy + SNAP/event covariates) | `D` ($H=28$) | `first_light` `ensemble_champion` (`P2B`) / `deep_field` `ensemble_inverse_error` (`P1`) | **`+26.39%`** | **`+18.00%`** | `lightgbm` (`0.2969`) · `chronos2@global` (`0.2981`) | CC BY-NC-SA 4.0 ([M5 / Makridakis et al., 2022](https://doi.org/10.1016/j.ijforecast.2021.11.013)) |
| **`favorita_grocery`** | Corporación Favorita grocery unit sales with future `onpromotion` binary flag & store hierarchy | `D` ($H=16$) | `deep_sky` `ensemble_ridge` (`P1`) / `deep_field` `ensemble_horizon_specialist` (`P2`) | **`+25.23%`** | **`+14.42%`** | **`chronos2@global_mint_nonneg` (`0.0718`)** | CC BY-NC-SA 4.0 ([Favorita / Nixtla Open Mirror](https://zenodo.org/records/4656021)) |
| **`m4_daily`** | M4 Competition daily financial, macroeconomic, and demographic series | `D` ($H=14$) | `wide_field` `ensemble_nnls` (`P3`) / `deep_field` `ensemble_ridge` (`P1`) | **`+36.98%`** | **`+29.93%`** | **`chronos_bolt@global` (`0.0436`)** | CC BY 4.0 ([M4 / Makridakis et al., 2020](https://doi.org/10.1016/j.ijforecast.2019.04.014)) |
| **`m4_weekly`** | M4 Competition weekly business and industry series ($m=52$) | `W` ($H=13$) | `deep_field` `ensemble_horizon_specialist` (`P2`) / `first_light` `ensemble_inverse_error` (`P1`) | **`+39.21%`** | **`+65.10%`** *(`+49.49%` on `deep_field`)* | **`chronos2@global` (`0.0695`)** | CC BY 4.0 ([M4 / Makridakis et al., 2020](https://doi.org/10.1016/j.ijforecast.2019.04.014)) |
| **`m4_monthly`** | M4 Competition monthly macroeconomic and industry series ($m=12$) | `MS` ($H=18$) | `wide_field` `ensemble_inverse_error` (`P1`) / `deep_field` `ensemble_horizon_specialist` (`P2`) | **`+46.10%`** | **`+49.15%`** | **`chronos2@global` (`0.0353`)** · **`chronos_bolt@global` (`0.0323` `wf`)** | CC BY 4.0 ([M4 / Makridakis et al., 2020](https://doi.org/10.1016/j.ijforecast.2019.04.014)) |
| **`monash_tourism_monthly`** | Australian & global monthly tourism volumes with strong trend and seasonal swings | `MS` ($H=24$) | `deep_sky` `ensemble_horizon_specialist` (`P2`) / `first_light` `ensemble_inverse_error` (`P1`) | **`+20.75%`** | **`+3.07%`** *(`+0.76%` on `deep_sky`/`deep_field` P1)* | `theta` (`0.1532`) · `chronos_bolt@global` (`0.1576`) | CC BY 4.0 ([Athanasopoulos et al., 2011 / Monash](https://zenodo.org/records/4656096)) |
| **`monash_hospital`** | Monthly patient admission count series across medical categories | `MS` ($H=12$) | `wide_field` `ensemble_nnls` (`P1`) / `deep_field` `ensemble_inverse_error` (`P2`) | **`+28.29%`** | **`+12.16%`** | **`chronos2@global` (`0.2667`)** | CC BY 4.0 ([Godahewa et al., 2021 / Monash](https://zenodo.org/records/4656014)) |
| **`monash_car_parts`** | Lumpy & intermittent monthly automotive spare-parts demand ($>80\%$ zeros) | `MS` ($H=12$) | `deep_sky` **`chronos_bolt@global` (`P1`)** / `deep_field` `ensemble_inverse_error` (`P1`) | **`+58.63%`** | **`+58.33%`** | **`chronos2@global` (`0.1869`)** · **`chronos_bolt@global` (`0.1869`)** | CC BY 4.0 ([Hyndman & Snyder / Monash](https://zenodo.org/records/4656021)) |
| **`etth1`** | Electricity Transformer Temperature (`OT`) hourly oil temperature with 6 load series (`HUFL`..`LULL`) | `h` ($H=24$) | `deep_field` **`chronos2@global` (`P1`)** / `ensemble_horizon_specialist` (`P2`) | **`+20.83%`** | **`+3.53%`** | **`chronos2@global` (`0.2002`)** | CC BY 4.0 ([Zhou et al., AAAI 2021](https://github.com/zhouhaoyi/ETDataset)) |

### Table 4B — Side-by-Side 4-Tier Progression Summary (1 Row per Benchmark)

*How to read the scores in this table:* Each tier cell shows **`OOF FVA %` $\uparrow$ / `Best Test Lift %` $\uparrow$ / `Wall-Clock s` $\downarrow$**, where **`OOF FVA %`** (**$\uparrow$ higher is better**) is out-of-fold error reduction vs. `naive_seasonal`, **`Best Test`** (**$\uparrow$ higher is better**) is held-out future test error reduction vs. `naive_seasonal`, and **`Time`** (**$\downarrow$ lower is faster**) is total session seconds. The final column reports the **`#1 @global / Foundation Model (`OOF Error` $\downarrow$)`**, where **lower error is better**.

| Public Benchmark | `first_light` (`OOF FVA` $\uparrow$ / `Test` $\uparrow$ / `Time` $\downarrow$) | `wide_field` (`OOF FVA` $\uparrow$ / `Test` $\uparrow$ / `Time` $\downarrow$) | `deep_sky` (`OOF FVA` $\uparrow$ / `Test` $\uparrow$ / `Time` $\downarrow$) | `deep_field` (`OOF FVA` $\uparrow$ / `Test` $\uparrow$ / `Time` $\downarrow$) | #1 `@global` / Foundation Model (`OOF Error` $\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`m5_walmart`** | **`+26.39%`** / `+18.00%` / `81s` | **`+18.44%`** / `+17.44%` / `92s` | **`+17.85%`** / `+17.68%` / `110s` | **`+18.88%`** / `+15.95%` / `192s` | `chronos2@global` (`0.2981`) |
| **`favorita_grocery`** | **`+28.04%`** / `+1.03%` / `173s` | **`+23.70%`** / `+6.81%` / `123s` | **`+19.43%`** / `+14.42%` / `278s` | **`+25.23%`** / `+8.97%` / `568s` | `chronos2@global_mint_nonneg` (`0.0718`) |
| **`m4_daily`** | **`+40.67%`** / `+24.41%` / `68s` | **`+24.94%`** / `+29.93%` / `137s` | **`+31.98%`** / `-6.73%` / `428s` | **`+36.98%`** / `-6.20%` / `413s` | `chronos_bolt@global` (`0.0436`) |
| **`m4_weekly`** | **`+34.38%`** / `+65.10%` / `46s` | **`+30.11%`** / `+40.94%` / `98s` | **`+36.60%`** / `+47.32%` / `247s` | **`+39.21%`** / `+49.49%` / `612s` | `chronos2@global` (`0.0695`) |
| **`m4_monthly`** | **`+52.88%`** / `+42.11%` / `139s` | **`+45.10%`** / `+49.15%` / `188s` | **`+43.66%`** / `+45.72%` / `282s` | **`+46.10%`** / `+44.11%` / `263s` | `chronos2@global` (`0.0353`) |
| **`monash_tourism_monthly`** | **`+13.26%`** / `+3.07%` / `178s` | **`+9.36%`** / `-8.17%` / `341s` | **`+20.75%`** / `+0.76%` / `651s` | **`+20.72%`** / `+0.76%` / `542s` | `chronos_bolt@global` (`0.1576`) |
| **`monash_hospital`** | **`+27.43%`** / `+5.81%` / `27s` | **`+28.29%`** / `+12.16%` / `53s` | **`+25.62%`** / `+1.16%` / `162s` | **`+27.34%`** / `+0.95%` / `504s` | `chronos2@global` (`0.2667`) |
| **`monash_car_parts`** | **`+0.00%`** / `+0.00%` / `45s` | **`+0.00%`** / `+0.00%` / `35s` | **`+57.16%`** / `+58.33%` / `74s` | **`+58.63%`** / `+0.00%` / `224s` | `chronos2@global` (`0.1869`) |
| **`etth1`** | **`+25.99%`** / `-9.28%` / `66s` | **`+17.52%`** / `-1.24%` / `178s` | **`+20.81%`** / `-12.05%` / `172s` | **`+20.83%`** / `+3.53%` / `250s` | `chronos2@global` (`0.2002`) |

### Table 4C — Complete 36-Cell Ledger (`9 Public Benchmarks × 4 Tiers`) with Dedicated Numeric Columns

*How to read the scores in this table:* **`P1 OOF` / `AF OOF` / `#1 Single (`OOF`)` / `#1 @global / Foundation (`OOF`)`** = raw out-of-fold error (`WAPE`, or `MASE` on `monash_car_parts`; **$\downarrow$ lower is better**) · **`P1 FVA %` / `AF FVA %`** = `%` out-of-fold error reduction vs. `naive_seasonal` (**$\uparrow$ higher is better**) · **`P1 Test Lift` / `AF Test Lift`** = `%` held-out test window error reduction vs. `naive_seasonal` (**$\uparrow$ higher is better**; `0.00%` ties `naive_seasonal`; negative means `naive_seasonal` had lower error on that single test window) · **`P1 Wall` / `AF Wall`** = elapsed seconds (**$\downarrow$ lower is faster**) · **`Models (`Exec` / `Borrowed`)`** = newly fitted models (`Exec`) vs. cached models reused across passes (`Borrowed`; **$\uparrow$ higher `Borrowed` saves more compute**).

| Public Benchmark | Exposure Tier | Pass 1 Model (`autofocus=False`) | P1 OOF $\downarrow$ | P1 FVA % $\uparrow$ | P1 Test Lift $\uparrow$ | P1 Wall $\downarrow$ | Final Model (`autofocus=True`) | Pass | AF OOF $\downarrow$ | AF FVA % $\uparrow$ | AF Test Lift $\uparrow$ | AF Wall $\downarrow$ | #1 Single Model (`OOF` $\downarrow$) | #1 `@global` / Foundation (`OOF` $\downarrow$) | Models (`Exec` / `Borrowed`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **`m5_walmart`** | `first_light` | `ensemble_inverse_error` | `0.2699` | `+24.21%` | `+17.90%` | `20.9s` | `ensemble_champion` | `P2B` | `0.2621` | **`+26.39%`** | `+18.00%` | `81.2s` | `lightgbm` (`0.2794`) | *(not in `first_light`)* | `12` / `4` |
| **`m5_walmart`** | `wide_field` | `ensemble_inverse_error` | `0.2968` | `+18.44%` | `+17.44%` | `92.1s` | `ensemble_inverse_error` | `P1` | `0.2968` | **`+18.44%`** | `+17.44%` | `92.5s` | `xgboost` (`0.3077`) | `chronos_bolt@global` (`0.3082`) | `11` / `0` |
| **`m5_walmart`** | `deep_sky` | `ensemble_inverse_error` | `0.2934` | `+17.85%` | `+17.68%` | `110.2s` | `ensemble_inverse_error` | `P1` | `0.2934` | **`+17.85%`** | `+17.68%` | `110.3s` | `lightgbm` (`0.2969`) | `chronos_bolt@global` (`0.3058`) | `13` / `0` |
| **`m5_walmart`** | `deep_field` | `ensemble_inverse_error` | `0.2898` | `+18.88%` | `+15.95%` | `192.2s` | `ensemble_inverse_error` | `P1` | `0.2898` | **`+18.88%`** | `+15.95%` | `192.3s` | `lightgbm` (`0.2969`) | `chronos2@global` (`0.2981`) | `16` / `0` |
| **`favorita_grocery`** | `first_light` | `ensemble_inverse_error` | `0.0590` | `+20.22%` | `+1.03%` | `25.0s` | `ensemble_champion` | `P3` | `0.0532` | **`+28.04%`** | `-0.29%` | `172.5s` | `theta_mint_nonneg` (`0.0624`) | *(not in `first_light`)* | `16` / `12` |
| **`favorita_grocery`** | `wide_field` | `ensemble_nnls` | `0.0659` | `+18.65%` | `+5.19%` | `85.4s` | `ensemble_horizon_specialist` | `P2` | `0.0620` | **`+23.70%`** | `+6.81%` | `123.4s` | `theta_mint_nonneg` (`0.0686`) | `chronos_bolt@global_mint_nonneg` (`0.0793`) | `13` / `11` |
| **`favorita_grocery`** | `deep_sky` | `ensemble_ridge` | `0.0706` | `+19.43%` | `+14.42%` | `197.1s` | `ensemble_ridge` | `P1` | `0.0706` | **`+19.43%`** | `+14.42%` | `278.5s` | `theta_mint_nonneg` (`0.0720`) | `chronos_bolt@global_mint_nonneg` (`0.0814`) | `17` / `13` |
| **`favorita_grocery`** | `deep_field` | `ensemble_inverse_error` | `0.0694` | `+21.00%` | `+7.35%` | `368.1s` | `ensemble_horizon_specialist` | `P2` | `0.0657` | **`+25.23%`** | `+8.97%` | `568.0s` | `chronos2@global_mint_nonneg` (`0.0718`) | `chronos2@global_mint_nonneg` (`0.0718`) | `6` / `33` |
| **`m4_daily`** | `first_light` | `theta` | `0.0430` | `+28.21%` | `+24.41%` | `12.1s` | `ensemble_champion` | `P2B` | `0.0356` | **`+40.67%`** | `-13.55%` | `68.1s` | `theta` (`0.0430`) | *(not in `first_light`)* | `13` / `9` |
| **`m4_daily`** | `wide_field` | `ensemble_nnls` | `0.0431` | `+24.62%` | `+29.90%` | `72.4s` | `ensemble_nnls` | `P3` | `0.0430` | **`+24.94%`** | `+29.93%` | `137.0s` | `chronos_bolt@global` (`0.0454`) | `chronos_bolt@global` (`0.0454`) | `13` / `29` |
| **`m4_daily`** | `deep_sky` | `ensemble_ridge` | `0.0396` | `+31.98%` | `-6.73%` | `129.3s` | `ensemble_ridge` | `P1` | `0.0396` | **`+31.98%`** | `-6.73%` | `428.2s` | `chronos_bolt@global` (`0.0436`) | `chronos_bolt@global` (`0.0436`) | `17` / `10` |
| **`m4_daily`** | `deep_field` | `ensemble_ridge` | `0.0369` | `+36.98%` | `-6.20%` | `402.6s` | `ensemble_ridge` | `P1` | `0.0369` | **`+36.98%`** | `-6.20%` | `413.2s` | `chronos_bolt@global` (`0.0436`) | `chronos_bolt@global` (`0.0436`) | `2` / `25` |
| **`m4_weekly`** | `first_light` | `ensemble_inverse_error` | `0.0694` | `+34.38%` | `+65.10%` | `25.4s` | `ensemble_inverse_error` | `P1` | `0.0694` | **`+34.38%`** | `+65.10%` | `45.5s` | `theta` (`0.0858`) | *(not in `first_light`)* | `8` / `0` |
| **`m4_weekly`** | `wide_field` | `chronos_bolt` | `0.0714` | `+30.11%` | `+40.94%` | `98.1s` | `chronos_bolt` | `P1` | `0.0714` | **`+30.11%`** | `+40.94%` | `98.2s` | `chronos_bolt` (`0.0714`) | `chronos_bolt` (`0.0714`) | `8` / `0` |
| **`m4_weekly`** | `deep_sky` | `ensemble_inverse_error` | `0.0641` | `+36.60%` | `+47.32%` | `246.6s` | `ensemble_inverse_error` | `P1` | `0.0641` | **`+36.60%`** | `+47.32%` | `246.7s` | `chronos_bolt@global` (`0.0723`) | `chronos_bolt@global` (`0.0723`) | `10` / `0` |
| **`m4_weekly`** | `deep_field` | `ensemble_inverse_error` | `0.0623` | `+38.39%` | `+48.70%` | `547.3s` | `ensemble_horizon_specialist` | `P2` | `0.0615` | **`+39.21%`** | `+49.49%` | `611.9s` | `chronos2@global` (`0.0695`) | `chronos2@global` (`0.0695`) | `15` / `12` |
| **`m4_monthly`** | `first_light` | `ensemble_inverse_error` | `0.0422` | `+34.59%` | `+16.09%` | `39.9s` | `ensemble_champion` | `P3` | `0.0304` | **`+52.88%`** | `+42.11%` | `138.6s` | `random_forest` (`0.0392`) | *(not in `first_light`)* | `7` / `8` |
| **`m4_monthly`** | `wide_field` | `ensemble_nnls` | `0.0304` | `+45.10%` | `+49.15%` | `104.0s` | `ensemble_nnls` | `P1` | `0.0304` | **`+45.10%`** | `+49.15%` | `187.6s` | `chronos_bolt@global` (`0.0323`) | `chronos_bolt@global` (`0.0323`) | `11` / `18` |
| **`m4_monthly`** | `deep_sky` | `ensemble_nnls` | `0.0344` | `+41.41%` | `+45.72%` | `192.8s` | `ensemble_horizon_specialist` | `P2` | `0.0332` | **`+43.66%`** | `+43.04%` | `281.7s` | `chronos_bolt@global` (`0.0375`) | `chronos_bolt@global` (`0.0375`) | `13` / `10` |
| **`m4_monthly`** | `deep_field` | `ensemble_inverse_error` | `0.0327` | `+44.34%` | `+44.11%` | `252.9s` | `ensemble_horizon_specialist` | `P2` | `0.0317` | **`+46.10%`** | `+43.82%` | `262.5s` | `chronos2@global` (`0.0353`) | `chronos2@global` (`0.0353`) | `2` / `25` |
| **`monash_tourism_monthly`** | `first_light` | `ensemble_inverse_error` | `0.2252` | `+5.84%` | `+3.07%` | `29.9s` | `ensemble_champion` | `P3` | `0.2078` | **`+13.26%`** | `-174.32%` | `178.1s` | `naive_seasonal` (`0.2395`) | *(not in `first_light`)* | `15` / `11` |
| **`monash_tourism_monthly`** | `wide_field` | `ensemble_inverse_error` | `0.1600` | `+7.85%` | `-8.17%` | `122.0s` | `ensemble_horizon_specialist` | `P3` | `0.1573` | **`+9.36%`** | `-18.45%` | `340.6s` | `naive_seasonal` (`0.1734`) | `chronos_bolt@global` (`0.1795`) | `11` / `18` |
| **`monash_tourism_monthly`** | `deep_sky` | `ensemble_inverse_error` | `0.1384` | `+18.80%` | `+0.76%` | `469.9s` | `ensemble_horizon_specialist` | `P2` | `0.1350` | **`+20.75%`** | `-10.25%` | `650.7s` | `theta` (`0.1532`) | `chronos_bolt@global` (`0.1576`) | `13` / `10` |
| **`monash_tourism_monthly`** | `deep_field` | `ensemble_inverse_error` | `0.1387` | `+18.63%` | `+0.76%` | `534.7s` | `ensemble_horizon_specialist` | `P2` | `0.1351` | **`+20.72%`** | `-9.46%` | `542.1s` | `theta` (`0.1532`) | `chronos_bolt@global` (`0.1576`) | `2` / `25` |
| **`monash_hospital`** | `first_light` | `ensemble_inverse_error` | `0.1965` | `+27.43%` | `+5.81%` | `26.8s` | `ensemble_inverse_error` | `P1` | `0.1965` | **`+27.43%`** | `+5.81%` | `27.0s` | `lightgbm` (`0.2010`) | *(not in `first_light`)* | `3` / `0` |
| **`monash_hospital`** | `wide_field` | `ensemble_nnls` | `0.2038` | `+27.55%` | `+12.16%` | `42.2s` | `ensemble_inverse_error` | `P2` | `0.2016` | **`+28.29%`** | `+2.64%` | `52.9s` | `theta` (`0.2091`) | `chronos_bolt` (`0.2343`) | `10` / `8` |
| **`monash_hospital`** | `deep_sky` | `ensemble_inverse_error` | `0.2643` | `+25.53%` | `+1.16%` | `107.0s` | `ensemble_inverse_error` | `P2` | `0.2639` | **`+25.62%`** | `+0.85%` | `161.5s` | `theta` (`0.2903`) | `chronos_bolt@global` (`0.2904`) | `11` / `10` |
| **`monash_hospital`** | `deep_field` | `ensemble_inverse_error` | `0.2587` | `+27.25%` | `+0.95%` | `385.5s` | `ensemble_inverse_error` | `P2` | `0.2583` | **`+27.34%`** | `+0.74%` | `504.1s` | `chronos2@global` (`0.2667`) | `chronos2@global` (`0.2667`) | `13` / `12` |
| **`monash_car_parts`** | `first_light` | `naive_seasonal` | `0.4364` | `+0.00%` | `+0.00%` | `22.3s` | `naive_seasonal` | `P3` | `0.4364` | **`+0.00%`** | `+0.00%` | `45.0s` | `naive_seasonal` (`0.4364`) | *(not in `first_light`)* | `10` / `13` |
| **`monash_car_parts`** | `wide_field` | `naive_seasonal` | `0.4364` | `+0.00%` | `+0.00%` | `28.7s` | `naive_seasonal` | `P2` | `0.4364` | **`+0.00%`** | `+0.00%` | `35.3s` | `naive_seasonal` (`0.4364`) | `chronos_bolt` (`0.4364`) | `13` / `12` |
| **`monash_car_parts`** | `deep_sky` | `chronos_bolt@global` | `0.1869` | `+57.16%` | `+58.33%` | `74.2s` | `chronos_bolt@global` | `P1` | `0.1869` | **`+57.16%`** | `+58.33%` | `74.2s` | `chronos_bolt@global` (`0.1869`) | `chronos_bolt@global` (`0.1869`) | `14` / `0` |
| **`monash_car_parts`** | `deep_field` | `ensemble_inverse_error` | `0.1806` | `+58.63%` | `+0.00%` | `223.6s` | `ensemble_inverse_error` | `P1` | `0.1806` | **`+58.63%`** | `+0.00%` | `223.6s` | `chronos2@global` (`0.1869`) | `chronos2@global` (`0.1869`) | `16` / `0` |
| **`etth1`** | `first_light` | `theta` | `0.1693` | `+25.30%` | `-9.28%` | `14.7s` | `ensemble_champion` | `P3` | `0.1604` | **`+25.99%`** | `-31.11%` | `66.3s` | `theta` (`0.1688`) | *(not in `first_light`)* | `10` / `6` |
| **`etth1`** | `wide_field` | `ensemble_inverse_error` | `0.1809` | `+16.09%` | `-1.24%` | `128.6s` | `ensemble_inverse_error` | `P2` | `0.1774` | **`+17.52%`** | `-2.48%` | `178.4s` | `chronos_bolt` (`0.1868`) | `chronos_bolt` (`0.1868`) | `10` / `8` |
| **`etth1`** | `deep_sky` | `chronos_bolt@global` | `0.2085` | `+15.52%` | `-12.05%` | `106.3s` | `ensemble_horizon_specialist` | `P2` | `0.1899` | **`+20.81%`** | `-24.44%` | `172.3s` | `chronos_bolt@global` (`0.2085`) | `chronos_bolt@global` (`0.2085`) | `13` / `10` |
| **`etth1`** | `deep_field` | `chronos2@global` | `0.2002` | `+18.11%` | `+3.53%` | `197.9s` | `ensemble_horizon_specialist` | `P2` | `0.1901` | **`+20.83%`** | `-22.16%` | `250.0s` | `chronos2@global` (`0.2002`) | `chronos2@global` (`0.2002`) | `15` / `12` |

> [!IMPORTANT]
> **Three Practitioner Takeaways from the 36-Cell Public Benchmark Matrix:**
> 1. **Zero-Shot Foundation Models Are the #1 Single Model on 7 of 9 Real-World Datasets:** Once `wide_field`, `deep_sky`, and `deep_field` introduce `chronos_bolt@global` and `chronos2@global`, a pretrained foundation model ranks as the **#1 single non-ensemble model** on **`favorita_grocery`** (`chronos2@global_mint_nonneg` `0.0718`), **`m4_daily`** (`chronos_bolt@global` `0.0436`), **`m4_weekly`** (`chronos2@global` `0.0695`), **`m4_monthly`** (`chronos_bolt@global` `0.0323` / `chronos2@global` `0.0353`), **`monash_hospital`** (`chronos2@global` `0.2667`), **`monash_car_parts`** (`chronos_bolt@global` / `chronos2@global` `0.1869`), and **`etth1`** (`chronos2@global` `0.2002`).
> 2. **Why Tier Progression (`first_light` $\rightarrow$ `deep_sky` / `deep_field`) Solves Sparse & Non-Stationary Series:** On `monash_car_parts` ($>80\%$ zeros), local statistical and tree models at `first_light` and `wide_field` score `0.00%` FVA (`0.1250` MAE, tying `naive_seasonal`). Stepping up to `deep_sky` pools the panel into **`chronos_bolt@global`**, jumping accuracy to **`+57.16%` OOF FVA and `+58.33%` Held-Out Test Lift** (`0.0521` MAE) in `74.2s`. Similarly, on `etth1` hourly transformer loads, stepping up to `deep_field` (`chronos2@global`) turns negative lower-tier test lift into **`+3.53%` Held-Out Test Lift**.
> 3. **Why Multi-Fold CV (`wide_field` / `deep_sky`) Beats 1-Fold CV (`first_light`) on Noisy Macro Series:** On `m4_daily`, `m4_monthly`, and `monash_tourism_monthly`, `first_light` uses a single backtest fold (`n_folds=1`) for speed (`12s`–`40s`), where aggressive multi-pass stacking can overfit the single validation window. Moving to `wide_field` (`2` folds) or `deep_sky` (`3` folds) stabilizes out-of-fold selection—jumping `m4_monthly` Held-Out Test Lift from `+16.09%` (`first_light` P1) to **`+49.15%`** (`wide_field`) and `favorita_grocery` Held-Out Test Lift from `+1.03%` (`first_light` P1) to **`+14.42%`** (`deep_sky`).

---

## 5. Built-In AlphaEvolve Policy (`Forecaster.auto`) vs. Opt-In Runtime Evolution (`Forecaster.evolve`)

> [!IMPORTANT]
> **Pre-Calibrated Policy (`Forecaster.auto()`) vs. Opt-In Runtime Code Evolution (`Forecaster.evolve()`):**
>
> - **1. Built-In AlphaEvolve Policy (`Forecaster.auto()` — Zero License Required):** All 5 rounds of evolutionary discovery in [Section 3](#3-how-alphaevolve-on-gemini-enterprise-evolved-the-policy) above are already compiled into pure Python inside [`scale_forecasting.auto_policy`](api/auto.md). Running `Forecaster.build_config()` or `Forecaster.auto()` requires **zero LLM API calls, zero external dependencies, and zero Gemini Enterprise licensing**.
> - **2. Opt-In Runtime Evolution (`Forecaster.evolve()` — [AlphaEvolve Runtime Harness (`alphaevolve_runner.md`)](./alphaevolve_runner.md)):** When your organization wants to evolve custom `# EVOLVE-BLOCK` Python code (`"features"`, `"ensemble"`, `"model"`, or `"policy"`) on a proprietary dataset on top of `Forecaster.auto()`, use [`Forecaster.evolve()`](api/sdk.md) ([`scale_forecasting.evolve`](api/evolve.md)). Its primary default **`backend="alphaevolve"`** connects to Gemini Enterprise Discovery Engine (`discoveryengine.googleapis.com/v1alpha`, requiring **Gemini Enterprise licensing + AlphaEvolve entitlement** and `enable_alphaevolve = true` in Terraform), while its non-default backup **`backend="vertex_gemini"`** runs on standard Vertex AI Gemini (`aiplatform.googleapis.com`, default `"gemini-3.8-flash"`, customizable via `model_name="..."`). Both backends enforce the exact same AST security sandbox, `+1e6` future-leakage probe, and **3-Gate Anti-Overfitting Ratchet** (including an unseen Held-Out Lockdown Fold).

### Table 5 — Built-In Policy vs. Opt-In Runtime Evolution (`scale_forecasting.evolve`)

*How to read this table:* Compare the zero-license built-in `Forecaster.auto()` policy against the opt-in `Forecaster.evolve()` runtime harness backends.

| Dimension | 1. Built-In AlphaEvolve Policy (`Forecaster.auto()`) | 2A. Opt-In Runtime Evolution — Primary (`Forecaster.evolve(..., backend="alphaevolve")`) | 2B. Opt-In Runtime Evolution — Backup (`Forecaster.evolve(..., backend="vertex_gemini")`) |
| :--- | :--- | :--- | :--- |
| **Licensing & Cloud API** | **Zero license required** *(100% pure Python in `auto_policy.py`)* | **Requires Gemini Enterprise license + AlphaEvolve entitlement** (`discoveryengine.googleapis.com/v1alpha`, `enable_alphaevolve = true`) | **Standard Google Cloud project** (`aiplatform.googleapis.com`, default `"gemini-3.8-flash"`, `enable_alphaevolve = false`) |
| **How It Works** | Deterministic 4-tier compiler + multi-branch `autofocus` DAG (`Pass 1 -> 2A/2B/2C -> 3`) | Managed multi-island Discovery Engine service (`:acquirePrograms` / `:submitProgramsEvaluations`) mutating `# EVOLVE-BLOCK` targets | In-process MAP-Elites island population mutating `# EVOLVE-BLOCK` targets via Vertex AI Gemini |
| **Anti-Overfitting & Causality Guards** | 4-Anchor Baseline Ladder, multi-fold CV, parsimony margin (`0.5%`), Champion Ratchet | **AST Security Sandbox + `+1e6` Future-Shock Probe + 3-Gate Ratchet** (Inner CV Lift $\rightarrow$ Diebold–Mariano $p_{\text{DM}}$ $\rightarrow$ Unseen Held-Out Lockdown Fold) | **Identical AST Sandbox + `+1e6` Future-Shock Probe + 3-Gate Ratchet** |

---

Continue to **[Auto-Forecaster Cloud Scale (`auto_forecaster_cloud_scale.md`)](./auto_forecaster_cloud_scale.md)**, explore the **[Opt-In AlphaEvolve Runtime Harness (`alphaevolve_runner.md`)](./alphaevolve_runner.md)**, or return to the **[Main Auto-Forecaster Guide (`auto_forecaster.md`)](./auto_forecaster.md)**.
