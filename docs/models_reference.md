# Models & ensembles reference

`scale-forecasting` ships **37 built-in forecasting models** (31 Python models, 4 Vertex AI AutoML models, and 2 BigQuery-native SQL models) alongside **9 ensemble blending and stacking strategies** and **8 hierarchical forecast reconciliation methods**. Every model implements the [`BaseModel`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py) contract in its own module under [`src/scale_forecasting/models/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/models) and registers itself at import time.

When a run is submitted, the orchestrator groups the requested models by **`family`** (`statistical`, `ml`, `deep_learning`, `automl`, and `native`) and routes each family to its designated compute runtime in parallel.

```mermaid
flowchart TD
    Base["BaseModel Contract (models/base_model.py)<br/>fit(y, X) · predict(horizon, X, quantiles)<br/>fit_panel(panel, ...) · predict_panel(horizon, ...)<br/>feature_attributions · search_space(trial) · recondition(y, X)"]

    subgraph Stat["Family: statistical (18 Python Models — Spark / Ray / Vertex / GCE / GKE)"]
        direction TB
        S1["Baselines, Spectral & Intermittent<br/>naive_mean · naive_seasonal · naive_drift<br/>naive_moving_average · croston · fft"]
        S2["Exponential Smoothing & Decomposition<br/>theta · auto_theta · holtwinters · autoets<br/>auto_ces · tbats · stl_bagging"]
        S3["ARIMA, State-Space & Additive<br/>auto_arima · sarimax · ucm · kalman · prophet"]
    end

    subgraph ML["Family: ml (5 Python Models via _lag_forecaster.py — Spark / Ray / Vertex / GCE / GKE)"]
        M1["Recursive & Direct Target Lags + Calendar & Exog + @monotone Sign Constraints + Two-Tier Explainability<br/>regression_lags (ridge_lags alias) · random_forest · lightgbm · xgboost · catboost"]
    end

    subgraph DL["Family: deep_learning (8 Python Models — Spark / Ray / Vertex / GCE / GKE CPU or GPU)"]
        direction TB
        D1["PyTorch Local, Global & Hybrid Panel Forecasters<br/>neuralprophet · tide · tft · tsmixer · patchtst"]
        D2["Open-Weight Zero-Shot Foundation Forecasters<br/>chronos_bolt · chronos2 · timesfm_torch"]
    end

    subgraph AML["Family: automl (4 Vertex AI AutoML / Tabular Workflow Models — vertex_automl)"]
        A1["Managed Global KFP v2 Pipelines + Reusable Stage-1 HPO + Explainability<br/>vertex_l2l · vertex_tide · vertex_tft · vertex_seq2seq"]
    end

    subgraph Nat["Family: native (2 BigQuery SQL Models — BigQuery ML)"]
        N1["Serverless SQL Execution<br/>arima_plus / arima_plus_xreg (ARIMA_PLUS / ARIMA_PLUS_XREG + ML.EXPLAIN_FORECAST + ML.ARIMA_EVALUATE)<br/>timesfm / bq_timesfm (AI.FORECAST 2.0 / 2.5 / 3.0 + Multivariate Covariates)"]
    end

    Rec["Hierarchical Reconciliation (reconciliation.py)<br/>bottom_up · top_down · middle_out · ols · wls_struct · wls_var · mint_shrink · mint_nonneg"]
    Ens["Ensemble Engine (Driver Pandas + Storage Write API)<br/>Calculated & Selection: mean · median · trimmed_mean · inverse_error · champion · horizon_specialist<br/>Learned Stacking: nnls · ridge · xgb"]

    Base --> Stat & ML & DL & AML & Nat
    Stat & ML & DL & AML & Nat --> Rec --> Ens
```

---

## Complete model catalog

Every model declares its upstream open-source package (`package` and `package_url` on [`BaseModel`](./api/models_base_model.md)), execution family, univariate & three-tier covariate support (`supports_future_covariates`, `supports_past_covariates`, `supports_static_covariates`), two-tier feature attribution support (`supports_explainability`), supported training modes (`local`, `global`, `hybrid`), hierarchical reconciliation support, and fast state updates (`expanding_frozen` backtesting via `supports_recondition`).

| Model | Family | Runtime | Upstream Package | Univariate | Covariates (`Future` / `Past` / `Static`) | Explainability (`attributions_df`) | Training Modes | Reconciliation | Native Intervals | Frozen Origin (`recondition`) | Methodology & Summary |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **`naive_mean`** | `statistical` | Python | [`numpy`](https://numpy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | No | Constant forecast equal to the training sample mean with empirical residual intervals. |
| **`naive_seasonal`** | `statistical` | Python | [`numpy`](https://numpy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | Yes | Repeats the final observed seasonal cycle (`m` derived from `data.freq`). |
| **`naive_drift`** | `statistical` | Python | [`numpy`](https://numpy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | Yes | Linear trend extrapolation between the first and last training observations. |
| **`naive_moving_average`** | `statistical` | Python | [`numpy`](https://numpy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | Yes | Flat forecast equal to the mean of the trailing `window` observations. |
| **`croston`** | `statistical` | Python | [`numpy`](https://numpy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | Yes | Intermittent-demand decomposition (`classic`, `sba`, or `tsb` variant) for zero-heavy series. |
| **`fft`** | `statistical` | Python | [`scipy`](https://scipy.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | No | Discrete Fourier Transform spectral extrapolation with polynomial trend detrending (`scipy.fft.rfft`). |
| **`theta`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | No | Assimakopoulos-Nikolopoulos Theta decomposition (`statsmodels.tsa.forecasting.theta.ThetaModel`). |
| **`auto_theta`** | `statistical` | Python | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | No | Automated Theta selection across Standard, Optimized (`OTM`), Dynamic (`DSTM`, `DOTM`) variants (`AutoTheta`). |
| **`holtwinters`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | No | Yes | Additive Holt-Winters seasonal exponential smoothing (`ExponentialSmoothing`). |
| **`autoets`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | Yes | State-space Error-Trend-Seasonal model (`ETSModel`) with analytical prediction intervals. |
| **`auto_ces`** | `statistical` | Python | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | No | Automated Complex Exponential Smoothing (`AutoCES`) across `"N"`, `"S"`, `"P"`, and `"F"` seasonality. |
| **`tbats`** | `statistical` | Python | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | No | Trigonometric seasonality, Box-Cox transform, ARMA errors, Trend, and Seasonal components (`AutoTBATS`). |
| **`stl_bagging`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | No / No / No | Structural | `local` | All 8 FPP3 | Yes | No | Bergmeir-Hyndman-Benítez STL decomposition with moving-block-bootstrapped ETS ensembles. |
| **`auto_arima`** | `statistical` | Python | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | Yes | Yes / Yes / No | Structural | `local` | All 8 FPP3 | Yes | No | Hyndman-Khandakar stepwise AICc seasonal ARIMA order search (`AutoARIMA`) with exogenous regressors. |
| **`sarimax`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | Yes / Yes / No | Structural | `local` | All 8 FPP3 | Yes | Yes | Seasonal ARIMA with exogenous regressors (`SARIMAX`) and analytical state-space intervals. |
| **`ucm`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | Yes / Yes / No | Structural | `local` | All 8 FPP3 | Yes | Yes | Structural Unobserved Components state-space model (`UnobservedComponents`) with local linear trend. |
| **`kalman`** | `statistical` | Python | [`statsmodels`](https://www.statsmodels.org/) | Yes | Yes / Yes / No | Structural | `local` | All 8 FPP3 | Yes | Yes | Linear Gaussian state-space Kalman filter (`UnobservedComponents`) with trigonometric seasonal harmonics. |
| **`prophet`** | `statistical` | Python | [`prophet`](https://facebook.github.io/prophet/) | Yes | Yes / Yes / No | Structural | `local` | All 8 FPP3 | Yes | No | Piecewise linear/logistic trend, multi-period Fourier seasonality, holidays, and exogenous regressors. |
| **`regression_lags`** | `ml` | Python | [`scikit-learn`](https://scikit-learn.org/) | Yes | Yes / Yes / No | **Tier 1 + Tier 2** *(Linear)* | `local`, `@monotone` | All 8 FPP3 | No | Yes | L2-regularized `Ridge` regression (`ridge_lags` alias) over recursive or direct target lags (`1, 2, 3, 7, 14, 28`), calendar features, `exog`, bounded `lsq_linear` sign constraints, and exact linear attributions. |
| **`random_forest`** | `ml` | Python | [`scikit-learn`](https://scikit-learn.org/) | Yes | Yes / Yes / No | **Tier 1 + Tier 2** *(Deviation)* | `local`, `@monotone` | All 8 FPP3 | No | Yes | Bagged decision tree ensemble (`RandomForestRegressor`) over recursive or direct target lags, calendar features, `exog`, and native `monotonic_cst` sign constraints. |
| **`lightgbm`** | `ml` | Python | [`lightgbm`](https://lightgbm.readthedocs.io/) | Yes | Yes / Yes / No | **Tier 1 + Tier 2** *(TreeSHAP)* | `local`, `@monotone` | All 8 FPP3 | No | Yes | Gradient-boosted decision trees (`LGBMRegressor`) over recursive or direct target lags, calendar features, `exog`, native `monotone_constraints`, and TreeSHAP attributions. |
| **`xgboost`** | `ml` | Python | [`xgboost`](https://xgboost.readthedocs.io/) | Yes | Yes / Yes / No | **Tier 1 + Tier 2** *(TreeSHAP)* | `local`, `@monotone` | All 8 FPP3 | No | Yes | Histogram-based gradient-boosted trees (`XGBRegressor`) over recursive or direct target lags, calendar features, `exog`, native `monotone_constraints`, and TreeSHAP attributions. |
| **`catboost`** | `ml` | Python | [`catboost`](https://catboost.ai/) | Yes | Yes / Yes / No | **Tier 1 + Tier 2** *(ShapValues)* | `local`, `@monotone` | All 8 FPP3 | No | Yes | Symmetric (oblivious) gradient-boosted trees (`CatBoostRegressor`) over recursive or direct target lags, calendar, `exog`, native `monotone_constraints`, and `ShapValues`. |
| **`neuralprophet`** | `deep_learning` | Python | [`neuralprophet`](https://neuralprophet.com/) | Yes | No / No / No | Structural | `local`, `global`, `hybrid` | All 8 FPP3 + Global Panel | Yes | No | PyTorch AR-Net with piecewise trend, Fourier seasonality, and quantile heads (`0.1, 0.5, 0.9`). |
| **`tide`** | `deep_learning` | Python | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | Yes | Yes / Yes / Yes | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | No | Time-series Dense Encoder (`TiDE`, Das et al. 2023) with future, past, and static covariates and `MQLoss` quantiles. |
| **`tft`** | `deep_learning` | Python | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | Yes | Yes / Yes / Yes | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | No | Temporal Fusion Transformer (`TFT`, Lim et al. 2021) with variable selection networks and interpretable multi-head attention. |
| **`tsmixer`** | `deep_learning` | Python | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | Yes | Yes / Yes / Yes | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | No | All-MLP time- and feature-mixing network (`TSMixerx`, Chen et al. 2023) with future, past, and static covariates. |
| **`patchtst`** | `deep_learning` | Python | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | Yes | No / No / No | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | No | Channel-independent subseries-patch Transformer (`PatchTST`, Nie et al. 2023) with `MQLoss` quantile heads. |
| **`chronos_bolt`** | `deep_learning` | Python | [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) | Yes | No / No / No | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | Yes | Amazon Chronos-Bolt zero-shot patch-based T5 foundation model (`amazon/chronos-bolt-base` default, `small` / `mini` / `tiny` selectable) with native decile heads. |
| **`chronos2`** | `deep_learning` | Python | [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) | Yes | Yes / Yes / Yes | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | Yes | Amazon Chronos-2 zero-shot foundation model (`amazon/chronos-2`) with group attention across future, past, and static covariates and panel `cross_learning`. |
| **`timesfm_torch`** | `deep_learning` | Python | [`timesfm`](https://github.com/google-research/timesfm) | Yes | Yes / Yes / Yes | Structural | `local`, `global` | All 8 FPP3 + Global Panel | Yes | Yes | Google Research open-weight PyTorch `TimesFM 3.0` (`google/timesfm-3.0-pytorch` default with native Variate Attention over covariates) and `TimesFM 2.5` (`version="2.5"` / `@v2.5`). |
| **`vertex_l2l`** | `automl` | Vertex AutoML | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | Yes | Yes / Yes / Yes | **Tier 1 + Tier 2** *(Vertex Baseline)* | `global` | All 8 FPP3 + Global Panel | Yes | No | Managed Vertex AI AutoML / Learn-to-Learn neural architecture search & ensemble with reusable Stage-1 HPO and explainability. |
| **`vertex_tide`** | `automl` | Vertex AutoML | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | Yes | Yes / Yes / Yes | **Tier 1 + Tier 2** *(Vertex Baseline)* | `global` | All 8 FPP3 + Global Panel | Yes | No | Managed Vertex AI Time-series Dense Encoder (`TiDE`) Tabular Workflow with reusable Stage-1 HPO and explainability. |
| **`vertex_tft`** | `automl` | Vertex AutoML | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | Yes | Yes / Yes / Yes | **Tier 1 + Tier 2** *(Vertex Baseline)* | `global` | All 8 FPP3 + Global Panel | Yes | No | Managed Vertex AI Temporal Fusion Transformer (`TFT`) Tabular Workflow with attention & baseline attributions. |
| **`vertex_seq2seq`** | `automl` | Vertex AutoML | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | Yes | Yes / Yes / Yes | **Tier 1 + Tier 2** *(Vertex Baseline)* | `global` | All 8 FPP3 + Global Panel | Yes | No | Managed Vertex AI Sequence-to-Sequence (`Seq2Seq+`) encoder-decoder Tabular Workflow with two-tier explainability. |
| **`arima_plus`** | `native` | BigQuery | [`bigquery-ml`](https://cloud.google.com/bigquery/docs/bqml-introduction) | Yes | Yes / No / No | **Tier 1 + Tier 2** *(ML.EXPLAIN_FORECAST)* | `local` | N/A (SQL) | Yes | No | BigQuery ML `ARIMA_PLUS` / `ARIMA_PLUS_XREG` (`arima_plus_xreg` alias) with custom country holiday CTEs, future covariates, `ML.EXPLAIN_FORECAST` attributions, `ML.ARIMA_EVALUATE` / `ML.ARIMA_COEFFICIENTS` diagnostics, and `ML.DETECT_ANOMALIES` / `ML.DETECT_CHANGE_POINTS`. |
| **`timesfm`** | `native` | BigQuery | [`bigquery-ml`](https://cloud.google.com/bigquery/docs/bqml-introduction) | Yes | Yes / Yes / No | Structural | `local` (zero-shot) | N/A (SQL) | Yes | No | Zero-shot foundation-model forecasting via BigQuery `AI.FORECAST` (`bq_timesfm` alias; `TimesFM 2.0`, `TimesFM 2.5` univariate default, or `TimesFM 3.0` with `future_covariates` and `past_covariates` via `target_cols => ['y']`) plus `AI.DETECT_ANOMALIES`. |

### Covariate tiers & mixed-model fallback policy

- **Univariate-only runs** require no `features` covariate fields (`future_covariates: []`, `past_covariates: []`, `static_covariates: []`, `exog: []`). All 37 models support univariate forecasting.
- **Three covariate tiers** when configured under `features`:
  - **`future_covariates`** (`exog` alias): known across both history and the forecast horizon (e.g. promotions, scheduled prices, calendar events). Supported by 21 models (`auto_arima`, `sarimax`, `ucm`, `kalman`, `prophet`, all 5 `ml` models, `tide`, `tft`, `tsmixer`, `chronos2`, `timesfm_torch`, all 4 `automl` models, `arima_plus` via `ARIMA_PLUS_XREG`, and `timesfm` via `TimesFM 3.0`).
  - **`past_covariates`**: observed historically up to `cutoff_date` and masked in validation/future windows during rolling-origin backtesting and HPO so future actuals never leak. Supported by 20 dynamic-covariate models (natively as historical encoders in `tide`, `tft`, `tsmixer`, `chronos2`, `timesfm_torch`, `automl` models, and `timesfm` via `TimesFM 3.0`, or via `exog_lags` / cutoff carry-forward in `ml` and `statistical` models).
  - **`static_covariates`**: time-invariant per-series metadata (`region`, `category`). Consumed by the cross-series `neuralforecast` and open-weight foundation architectures (`tide`, `tft`, `tsmixer`, `chronos2`, `timesfm_torch`) and all 4 `automl` models (`vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq`).
- **Mixed univariate + covariate runs (`features.on_unsupported_covariates`):**
  - **`"fallback"` (default):** A multi-model run mixing covariate-capable and univariate models (e.g. `["theta", "xgboost", "tide"]` with all three covariate tiers) logs a clear preflight warning per model and strips unsupported tiers (and their `exog_lags`) for that model across fit, backtesting, and HPO—so univariate models run on their full, un-truncated target history.
  - **`"error"` (override):** Fails fast at `dag.check_model_params` with a `ConfigError` naming every model and the covariate tier(s) it does not support.

---

## Model families & hyperparameter reference

Per-model hyperparameters can be authored statically under `model_params.<model_name>` in the run configuration or tuned automatically with Optuna when `hpo.enabled: true`.

### Tagged model variants (`<base_model>@<tag>`)

Any of the 37 registered models can be listed multiple times in `models` using the **`<base_model>@<tag>`** syntax (for example, `xgboost@univariate`, `xgboost@direct`, `lightgbm@monotone`, `regression_lags@monotone` / `ridge_lags@monotone`, `tide@global_cov_all`, `tide@local_univariate`, `neuralprophet@hybrid_univariate`, `chronos_bolt@tiny`, `timesfm_torch@v2.5`, `croston@classic`, `croston@tsb`). Each tagged entry executes as an independent model in `forecast_predictions`, `backtest_oof`, `forecast_metadata`, and ensemble/reconciliation pipelines while resolving to `<base_model>` for class lookup:

- **Per-variant hyperparameter overrides:** `model_params` accepts keys for either the base name (`model_params.xgboost`) or the full tagged name (`model_params["xgboost@direct"]`), merging variant-specific overrides on top of base defaults. When `hpo.enabled: true`, Optuna tunes tunable hyperparameters while **preserving explicit variant pins** (such as `strategy: "direct"`, `variant: "tsb"`, or `training_mode: "global"`).
- **Per-variant covariate, feature-ablation & economic sign-constraint tags:** Suffixes declared in [`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py) and [`monotonicity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/monotonicity.py) automatically filter the exogenous feature matrix `X` or enforce directional monotonicity for that variant so side-by-side ablations run in a single job:
  - `@monotone` (composable with `+`, e.g., `lightgbm@monotone`, `xgboost@direct+monotone`, `catboost@monotone`, `random_forest@monotone`, `regression_lags@monotone` / `ridge_lags@monotone`; discoverable programmatically via `list_models(supports_monotonic_constraints=True)` or `list_monotonic_models()`) — enforces directional economic sign constraints (`features.monotonic_constraints`, or auto-inferred `price* -> -1`, `promo* / discount* -> +1` when `monotonic_constraints` is empty) on that variant while leaving untagged sibling models unconstrained for A/B comparison.
  - `@univariate` / `@local_univariate` / `@global_univariate` / `@hybrid_univariate` — strips all exogenous covariates (`X = None`), fitting the model purely on the target history.
  - `@cov_<col>` — restricts `X` to a single covariate column `<col>` (plus any matching `<col>_lag_*` columns).
  - `@cov_drop_<col>` — drops `<col>` (and `<col>_lag_*`) while keeping all other covariates (leave-one-covariate-out ablation).
  - `@no_holidays` — drops country-holiday indicator columns while keeping all other features.
  - `@no_lags` — drops engineered exogenous lag columns (`*_lag_*`) while keeping contemporaneous covariates.

### 1. Statistical family (`statistical`)

Models in the `statistical` family fit per-series time-series equations on CPU workers (Spark, Ray, Vertex CustomJob, GCE, or GKE).

#### Baselines, intermittent demand & spectral models

| Model | Upstream Package | Authored `model_params` & Defaults | Optuna `search_space` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `naive_mean` | [`numpy`](https://numpy.org/) | — | — | Predicts the arithmetic mean of training observations. |
| `naive_seasonal` | [`numpy`](https://numpy.org/) | — | — | Tiles the last $m$ observations forward ($m = 7$ daily, $52$ weekly, $12$ monthly, $24$ hourly). |
| `naive_drift` | [`numpy`](https://numpy.org/) | — | — | Extrapolates the average step change $(y_T - y_1) / (T - 1)$. |
| `naive_moving_average` | [`numpy`](https://numpy.org/) | `window: 7` (`int >= 1`) | `window` $\in [2, 56]$ | Predicts the trailing `window`-step mean; clamps to series length on short series. |
| `croston` | [`numpy`](https://numpy.org/) | `alpha: 0.1`, `variant: "sba"` (`"classic"` \| `"sba"` \| `"tsb"`), `beta: 0.1` | `alpha` $\in [0.02, 0.5]$, `variant` $\in \{\text{classic}, \text{sba}, \text{tsb}\}$, `beta` $\in [0.02, 0.5]$ | Decomposes intermittent demand into non-zero demand size and inter-arrival interval (Syntetos-Boylan or Teunter-Syntetos-Babai). |
| `fft` | [`scipy`](https://scipy.org/) | `K: 10` (top frequencies), `trend_poly_degree: 1` (`0..3`) | `K` $\in [2, 30]$, `trend_poly_degree` $\in [0, 2]$ | Detrends with a polynomial of degree `trend_poly_degree`, filters to the `K` highest-amplitude `rfft` harmonics, and projects forward analytically. |

#### Exponential smoothing, Theta & decomposition models

| Model | Upstream Package | Authored `model_params` & Defaults | Optuna `search_space` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `theta` | [`statsmodels`](https://www.statsmodels.org/) | `theta: 2.0` (`float > 1.0`) | `theta` $\in [1.1, 4.0]$ | Deseasonalizes when the series spans at least $2m$ observations and passes an ACF seasonal test. |
| `auto_theta` | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | `decomposition_type: "additive"` (`"additive"` \| `"multiplicative"`), `model: None` (`"STM"` \| `"OTM"` \| `"DSTM"` \| `"DOTM"`) | `decomposition_type` $\in \{\text{additive}, \text{multiplicative}\}$ | Evaluates standard, optimized, and dynamic Theta specifications via `statsforecast.models.AutoTheta`. |
| `holtwinters` | [`statsmodels`](https://www.statsmodels.org/) | `damped_trend: False` | `damped_trend` $\in \{\text{True}, \text{False}\}$ | Fits additive trend and seasonality; falls back to trend-only if the series has fewer than $2m$ observations. |
| `autoets` | [`statsmodels`](https://www.statsmodels.org/) | `damped_trend: True` | `damped_trend` $\in \{\text{True}, \text{False}\}$ | State-space `ETSModel` with L-BFGS maximum likelihood estimation and analytical state-space intervals. |
| `auto_ces` | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | `model: "Z"` (`"Z"` \| `"N"` \| `"S"` \| `"P"` \| `"F"`) | `model` $\in \{\text{Z}, \text{N}, \text{S}, \text{P}\}$ | Complex Exponential Smoothing (`statsforecast.models.AutoCES`); `"Z"` selects information-criterion optimal seasonality. |
| `tbats` | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | `use_boxcox: None`, `use_trend: None`, `use_damped_trend: None`, `use_arma_errors: True` | `use_trend` $\in \{\text{True}, \text{False}, \text{None}\}$, `use_damped_trend` $\in \{\text{True}, \text{False}, \text{None}\}$, `use_arma_errors` $\in \{\text{True}, \text{False}\}$ | Trigonometric seasonal representation (`AutoTBATS`) capable of modeling complex and non-integer seasonal cycles. |
| `stl_bagging` | [`statsmodels`](https://www.statsmodels.org/) | `n_bags: 10`, `block_size: None` (defaults to $2m$) | `n_bags` $\in [5, 20]$ | Decomposes series via `STL`, applies moving-block bootstrap to the remainder, fits an `ETSModel` per bag, and takes empirical quantiles across the bag ensemble. |

#### ARIMA, state-space & structural additive models

| Model | Upstream Package | Authored `model_params` & Defaults | Optuna `search_space` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `auto_arima` | [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/) | `max_p: 3`, `max_q: 3`, `max_P: 1`, `max_Q: 1`, `max_d: 2`, `max_D: 1`, `stepwise: True`, `seasonal: True` | `max_p` $\in [1, 5]$, `max_q` $\in [1, 5]$, `seasonal` $\in \{\text{True}, \text{False}\}$ | Hyndman-Khandakar automatic seasonal ARIMA order selection (`AutoARIMA`) with native exogenous regressor support. |
| `sarimax` | [`statsmodels`](https://www.statsmodels.org/) | `order: (1, 1, 1)` | `p` $\in [0, 2]$, `d` $\in [0, 1]$, `q` $\in [0, 2]$ | Fixed-order `SARIMAX` with seasonal order $(1, 0, 1, m)$ when length $\ge 3m$, plus exogenous regressors. |
| `ucm` | [`statsmodels`](https://www.statsmodels.org/) | `level: "local linear trend"`, `freq_seasonal_harmonics: 2` | `level` $\in \{\text{local level}, \text{local linear trend}, \text{smooth trend}\}$, `freq_seasonal_harmonics` $\in [1, 4]$ | Structural state-space decomposition (`UnobservedComponents`) with frequency-domain seasonal harmonics and `exog`. |
| `kalman` | [`statsmodels`](https://www.statsmodels.org/) | `level: "local linear trend"`, `harmonics: 3`, `autoregressive: 0` (`0..3`) | `level` $\in \{\text{local level}, \text{local linear trend}, \text{random walk with drift}\}$, `harmonics` $\in [1, 5]$, `autoregressive` $\in [0, 2]$ | State-space Kalman filter (`UnobservedComponents`) combining stochastic state transitions, seasonal harmonics, optional AR($p$) innovations, and `exog`. |
| `prophet` | [`prophet`](https://facebook.github.io/prophet/) | `changepoint_prior_scale: 0.05`, `seasonality_prior_scale: 10.0`, `seasonality_mode: "additive"` | `changepoint_prior_scale` $\in [10^{-3}, 0.5]$, `seasonality_prior_scale` $\in [0.01, 10.0]$, `seasonality_mode` $\in \{\text{additive}, \text{multiplicative}\}$ | Piecewise trend + Fourier seasonality + exogenous covariates via `cmdstanpy` L-BFGS optimization. |

---

### 2. Tabular machine learning family (`ml`)

All five models in the `ml` family share the autoregressive lag engine in [`_lag_forecaster.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/_lag_forecaster.py):

- **Feature matrix construction:** Automatically constructs target lags (default `lags: [1, 2, 3, 7, 14, 28]`, customizable per model and automatically dropping any lag $\ge \lfloor T/2 \rfloor$ on short series), four deterministic calendar features (`dow`, `dom`, `month`, `doy`), and any configured dynamic `features` columns (`holidays`, `holiday_window`, `fourier`, `level_shift`, `exog`, `future_covariates`, `past_covariates`, `exog_lags`).
- **Recursive vs. direct multi-step forecasting (`strategy: "recursive" | "direct"`):**
  - **`"recursive"` (default):** Fits a single 1-step-ahead estimator (`build_lag_matrix`) and rolls predictions forward across $h \in \{1, \dots, H\}$ (`recursive_predict`), appending each predicted $\hat{y}_{T+h}$ to the target history buffer so subsequent steps read recursive lags. Historical-only `past_covariates` are carried forward from the last observed cutoff value without lookahead into the validation window.
  - **`"direct"`:** Fits an independent horizon-specific estimator $f_h$ for each step $h \in \{1, \dots, H\}$ directly against $y_{t+h-1}$ using fixed forecast-origin target lags $y_{t-\ell}$ alongside step-$h$ calendar and future covariates (`fit_direct_models` / `direct_predict` / `direct_explain`), eliminating recursive error accumulation at longer horizons. Compare both side-by-side in a single run using tagged model variants (for example, `["xgboost", "xgboost@direct"]` with `model_params: {"xgboost@direct": {"strategy": "direct"}}`).
- **Economic sign & monotonicity constraints (`features.monotonic_constraints` & `@monotone`):** All five `ml` models (`lightgbm`, `xgboost`, `catboost`, `random_forest`, and `regression_lags` / `ridge_lags` alias) declare `supports_monotonic_constraints = True` (queryable via `list_models(supports_monotonic_constraints=True)`) and enforce directional sign constraints $c_j \in \{-1, 0, +1\}$ aligned to `self._features` via [`monotonicity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/monotonicity.py) — passing `monotone_constraints` to `LGBMRegressor`, `XGBRegressor`, and `CatBoostRegressor`, `monotonic_cst` to `RandomForestRegressor`, and solving box-constrained Ridge (`scipy.optimize.lsq_linear`) in `RegressionLagsModel`.
- **Two-tier feature attributions (`supports_explainability = True`):** Computes exact per-horizon-step local attributions (`forecast_predictions.explanations`) and series-level driver importance (`forecast_metadata.fit_diagnostics["feature_attributions"]`) for both `"recursive"` and `"direct"` strategies using native C++ TreeSHAP (`xgboost`, `lightgbm`, `catboost`), closed-form linear weights (`regression_lags`), or importance-weighted feature deviations (`random_forest`).
- **Fast state updates (`expanding_frozen`):** Appends newly observed actuals to the lag history buffer without re-fitting tree or regression weights (`supports_recondition = True`).

| Model | Upstream Package | Authored `model_params` & Defaults | Optuna `search_space` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `regression_lags` | [`scikit-learn`](https://scikit-learn.org/) | `alpha: 1.0`, `strategy: "recursive"` (`"recursive"` \| `"direct"`), `lags: [1, 2, 3, 7, 14, 28]` | `alpha` $\in [10^{-2}, 10^{2}]$ (log scale) | L2-regularized `Ridge` linear regressor (`ridge_lags` alias) over the lag + calendar + `exog` design matrix; enforces exact sign bounds via `scipy.optimize.lsq_linear` when `@monotone` or `monotonic_constraints` is active. |
| `random_forest` | [`scikit-learn`](https://scikit-learn.org/) | `n_estimators: 100`, `max_depth: 10`, `min_samples_leaf: 2`, `max_features: 1.0`, `strategy: "recursive"` (`"recursive"` \| `"direct"`), `lags: [1, 2, 3, 7, 14, 28]` | `n_estimators` $\in [50, 250]$, `max_depth` $\in [4, 16]$, `min_samples_leaf` $\in [1, 10]$, `max_features` $\in \{1.0, \text{sqrt}\}$ | Bootstrap-aggregated `RandomForestRegressor` (`n_jobs=1` per worker to prevent executor oversubscription). |
| `lightgbm` | [`lightgbm`](https://lightgbm.readthedocs.io/) | `n_estimators: 200`, `learning_rate: 0.05`, `num_leaves: 31`, `strategy: "recursive"` (`"recursive"` \| `"direct"`), `lags: [1, 2, 3, 7, 14, 28]` | `n_estimators` $\in [50, 400]$, `learning_rate` $\in [0.01, 0.2]$, `num_leaves` $\in [15, 63]$ | Leaf-wise `LGBMRegressor` (`n_jobs=1` per worker) with native `monotone_constraints` support (`lightgbm@monotone`). |
| `xgboost` | [`xgboost`](https://xgboost.readthedocs.io/) | `n_estimators: 200`, `max_depth: 4`, `learning_rate: 0.05`, `subsample: 0.8`, `strategy: "recursive"` (`"recursive"` \| `"direct"`), `lags: [1, 2, 3, 7, 14, 28]` | `n_estimators` $\in [50, 400]$, `max_depth` $\in [3, 8]$, `learning_rate` $\in [0.01, 0.2]$, `subsample` $\in [0.6, 1.0]$ | Histogram-based `XGBRegressor` (`tree_method="hist"`) with native `monotone_constraints` support (`xgboost@monotone`). Supports optional CUDA placement, though CPU is recommended for per-series tabular fits (`gpu_usefulness = "suboptimal"`). |
| `catboost` | [`catboost`](https://catboost.ai/) | `iterations: 200`, `depth: 6`, `learning_rate: 0.05`, `l2_leaf_reg: 3.0`, `strategy: "recursive"` (`"recursive"` \| `"direct"`), `lags: [1, 2, 3, 7, 14, 28]` | `iterations` $\in [50, 400]$, `depth` $\in [4, 8]$, `learning_rate` $\in [0.01, 0.2]$, `l2_leaf_reg` $\in [1.0, 10.0]$ | Oblivious (symmetric) decision trees via `CatBoostRegressor` (`thread_count=1`, `allow_writing_files=False`) with native `monotone_constraints` support (`catboost@monotone`). |

---

### 3. Deep learning & open-weight foundation family (`deep_learning`)

Models in the `deep_learning` family execute on CPU by default and support optional GPU acceleration (`gpu_usefulness = "beneficial"`) across all compute runtimes alongside configurable **`training_mode`** (`model_params.<model>.training_mode`):

- **`"local"` (default):** Fits or conditions one model per `(ts_id, model)` cell across Spark, Ray, Vertex CustomJob, GCE, or GKE workers.
- **`"global"`:** Fits a single shared cross-series network or runs batched panel foundation inference across the entire panel (`worker.run_panel_model`). Supported by all eight `deep_learning` models (`neuralprophet`, `tide`, `tft`, `tsmixer`, `patchtst`, `chronos_bolt`, `chronos2`, `timesfm_torch`).
- **`"hybrid"`:** Supported by `neuralprophet`, combining global shared AR-Net / seasonality weights with per-series local trend (`trend_global_local="local"`, `season_global_local="global"`).
- **Separated Optional Extras (`[models-dl]` vs. `[models-foundation]`):** Trained neural architectures (`neuralprophet`, `tide`, `tft`, `tsmixer`, `patchtst`) belong to `scale-forecasting[models-dl]`, while open-weight pretrained foundation models (`chronos_bolt`, `chronos2`, `timesfm_torch`) belong to `scale-forecasting[models-foundation]`. Both run on CPU out-of-the-box and leverage GPU tensor cores when `compute.families.deep_learning.use_gpu: true` (`"T4"` or `"L4"`).

| Model | Upstream Package | Authored `model_params` & Defaults | Optuna `search_space` | Covariate Tiers & Notes |
| :--- | :--- | :--- | :--- | :--- |
| `neuralprophet` | [`neuralprophet`](https://neuralprophet.com/) | `training_mode: "local"` (`"local"` \| `"global"` \| `"hybrid"`), `epochs: 50`, `learning_rate: 0.01`, `n_lags: 0`, `n_forecasts: 1`, `batch_size: None` | `epochs` $\in [20, 200]$, `learning_rate` $\in [10^{-3}, 10^{-1}]$ | PyTorch AR-Net + piecewise trend + Fourier seasonality with quantile heads (`[0.1, 0.5, 0.9]`). Univariate local, global, or hybrid panel forecaster (`supports_exog = False`). |
| `tide` | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | `training_mode: "local"` (`"local"` \| `"global"`), `max_steps: 50`, `input_size: None` (auto $2H$), `hidden_size: 128`, `num_encoder_layers: 2`, `num_decoder_layers: 2`, `dropout: 0.1`, `learning_rate: 1e-3`, `batch_size: 32`, `scaler_type: "standard"` | `max_steps` $\in [25, 100]$, `hidden_size` $\in \{64, 128, 256\}$, `num_encoder_layers` $\in [1, 3]$, `num_decoder_layers` $\in [1, 3]$, `dropout` $\in [0.0, 0.3]$, `learning_rate` $\in [10^{-4}, 10^{-2}]$ | Time-series Dense Encoder (`TiDE`, Das et al. 2023). Consumes all three covariate tiers (`future_covariates`, `past_covariates`, `static_covariates`) with native `MQLoss` quantile heads. |
| `tft` | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | `training_mode: "local"` (`"local"` \| `"global"`), `max_steps: 50`, `input_size: None` (auto $2H$), `hidden_size: 64`, `n_head: 4`, `dropout: 0.1`, `learning_rate: 1e-3`, `batch_size: 32`, `scaler_type: "robust"` | `max_steps` $\in [25, 100]$, `hidden_size` $\in \{32, 64, 128\}$, `n_head` $\in \{2, 4, 8\}$, `dropout` $\in [0.0, 0.3]$, `learning_rate` $\in [10^{-4}, 10^{-2}]$ | Temporal Fusion Transformer (`TFT`, Lim et al. 2021). Consumes all three covariate tiers (`future_covariates`, `past_covariates`, `static_covariates`) via gated variable selection networks and multi-head attention. |
| `tsmixer` | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | `training_mode: "local"` (`"local"` \| `"global"`), `max_steps: 50`, `input_size: None` (auto $2H$), `n_block: 2`, `ff_dim: 64`, `dropout: 0.1`, `learning_rate: 1e-3`, `batch_size: 32`, `scaler_type: "standard"` | `max_steps` $\in [25, 100]$, `n_block` $\in [1, 4]$, `ff_dim` $\in \{32, 64, 128\}$, `dropout` $\in [0.0, 0.3]$, `learning_rate` $\in [10^{-4}, 10^{-2}]$ | All-MLP time- and feature-mixing network (`TSMixerx`, Chen et al. 2023). Consumes all three covariate tiers (`future_covariates`, `past_covariates`, `static_covariates`). |
| `patchtst` | [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) | `training_mode: "local"` (`"local"` \| `"global"`), `max_steps: 50`, `input_size: None` (auto $2H$), `hidden_size: 64`, `n_heads: 4`, `encoder_layers: 2`, `patch_len: 16`, `stride: 8`, `dropout: 0.1`, `learning_rate: 1e-3`, `batch_size: 32`, `scaler_type: "standard"` | `max_steps` $\in [25, 100]$, `hidden_size` $\in \{32, 64, 128\}$, `n_heads` $\in \{2, 4, 8\}$, `patch_len` $\in \{8, 16, 24\}$, `dropout` $\in [0.0, 0.3]$, `learning_rate` $\in [10^{-4}, 10^{-2}]$ | Channel-independent subseries-patch Transformer (`PatchTST`, Nie et al. 2023) for univariate local or cross-series global forecasting (`supports_exog = False`). |
| `chronos_bolt` | [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) | `training_mode: "local"` (`"local"` \| `"global"`), `size: "base"` (`"base"` \| `"small"` \| `"mini"` \| `"tiny"` or `@tiny` / `@mini` / `@small` tag), `context_length: None`, `batch_size: 256`, `repo_id: None` | — *(zero-shot)* | Amazon Chronos-Bolt patch-based T5 foundation model (`amazon/chronos-bolt-{size}`) with direct multi-step decile heads (`0.1..0.9`) and process-level LRU checkpoint caching (`supports_exog = False`). |
| `chronos2` | [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) | `training_mode: "local"` (`"local"` \| `"global"`), `cross_learning: True` (default in `global` mode), `context_length: None`, `batch_size: 256`, `repo_id: "amazon/chronos-2"` | — *(zero-shot)* | Amazon Chronos-2 zero-shot foundation model with group attention across `future_covariates`, `past_covariates`, and instance-norm-preserving `static_covariates` carriers (`supports_exog = True`). |
| `timesfm_torch` | [`timesfm`](https://github.com/google-research/timesfm) | `training_mode: "local"` (`"local"` \| `"global"`), `version: "3.0"` (`"3.0"` default \| `"2.5"` or `@v2.5` tag), `context_length: None` (up to `15360`), `batch_size: 32`, `ridge: 1.0`, `repo_id: None` | — *(zero-shot)* | Open-weight PyTorch Google Research `TimesFM 3.0` (`google/timesfm-3.0-pytorch` with native Variate Attention over `future_covariates`, `past_covariates`, and `static_covariates`) and `TimesFM 2.5` (`google/timesfm-2.5-200m-pytorch` with in-context Ridge covariate regression). |

---

### 4. Vertex AI AutoML & Tabular Workflows family (`automl`)

Models in the `automl` family execute as global cross-series models on **`runtime = "vertex_automl"`** via [`engines/automl_engine.py`](./api/engines_automl_engine.md). By default (`automl_mode = "tabular_workflow"`), each model compiles and runs a managed Kubeflow Pipelines v2 DAG (`google-cloud-pipeline-components`) that exports its Stage-1 hyperparameter tuning artifact (`stage_1_tuning_result_artifact_uri`) to GCS (`forecast_metadata.model_artifact`) so subsequent runs can warm-start (`reuse_tuning_from_run_id`) without re-running architecture search. Batch predictions automatically request `generate_explanation = True`, populating both Tier-1 (`forecast_metadata.fit_diagnostics["feature_attributions"]`) and Tier-2 (`forecast_predictions.explanations`) attributions.

| Model | Upstream Package | Pipeline Builder / Training Mode | Authored `model_params` & Defaults | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `vertex_l2l` | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | `get_learn_to_learn_forecasting_pipeline_and_parameters` (`tabular_workflow`) or `AutoMLForecastingTrainingJob` (`training_job`) | `context_window`: auto ($2H$), `train_budget_milli_node_hours: 1000`, `optimization_objective: "minimize-wape-mae"`, `stage_1_num_parallel_trials: 8`, `stage_2_num_selected_trials: 3`, `reuse_tuning_from_run_id: None` | Vertex AI AutoML / Learn-to-Learn neural architecture search & trial ensembling across all three covariate tiers. |
| `vertex_tide` | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | `get_time_series_dense_encoder_forecasting_pipeline_and_parameters` | `context_window`: auto ($2H$), `train_budget_milli_node_hours: 1000`, `optimization_objective: "minimize-wape-mae"`, `stage_1_num_parallel_trials: 8`, `stage_2_num_selected_trials: 3`, `reuse_tuning_from_run_id: None` | Managed Vertex AI Time-series Dense Encoder (`TiDE`, [Das et al. 2023](https://arxiv.org/abs/2304.08424)) Tabular Workflow with reusable Stage-1 HPO and baseline attributions. |
| `vertex_tft` | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | `get_temporal_fusion_transformer_forecasting_pipeline_and_parameters` | `context_window`: auto ($2H$), `train_budget_milli_node_hours: 1000`, `optimization_objective: "minimize-wape-mae"`, `stage_1_num_parallel_trials: 8`, `stage_2_num_selected_trials: 3`, `reuse_tuning_from_run_id: None` | Managed Vertex AI Temporal Fusion Transformer (`TFT`, [Lim et al. 2021](https://arxiv.org/abs/1912.09363)) Tabular Workflow with multi-head attention and baseline attributions. |
| `vertex_seq2seq` | [`google-cloud-pipeline-components`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting) | `get_sequence_to_sequence_forecasting_pipeline_and_parameters` | `context_window`: auto ($2H$), `train_budget_milli_node_hours: 1000`, `optimization_objective: "minimize-wape-mae"`, `stage_1_num_parallel_trials: 8`, `stage_2_num_selected_trials: 3`, `reuse_tuning_from_run_id: None` | Managed Vertex AI Sequence-to-Sequence (`Seq2Seq+`) encoder-decoder Tabular Workflow across all three covariate tiers. |

---

### 5. BigQuery-native SQL family (`native`)

Models in the `native` family execute directly inside BigQuery via [`engines/bigquery_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/bigquery_engine.py) (`use_query_cache=False`) in parallel with any Spark, Ray, Vertex CustomJob, GCE, GKE, or Vertex AutoML families. Although both models submit the entire panel in a single SQL statement (`time_series_id_col = 'ts_id'` for `arima_plus` and `id_cols => ['ts_id']` for `timesfm`), BigQuery partitions by `ts_id` and forecasts each series independently in isolation (**`local`**): `arima_plus` fits an independent seasonal ARIMA / `ARIMA_PLUS_XREG` pipeline per `ts_id`, while `timesfm` runs zero-shot univariate or multivariate (`TimesFM 3.0`) inference per `ts_id` over that series' trailing `context_window`.

- **Flexible `arima_plus` & `arima_plus_xreg` Configuration:** `"arima_plus"` is the canonical model name (`list_models()` stays at 37 models) and automatically upgrades from `ARIMA_PLUS` to `ARIMA_PLUS_XREG` whenever `features.future_covariates` (`exog`) is configured. You can also author `"arima_plus_xreg"` (or `"arima_plus@xreg"`) directly in `models` and `model_params`: when `["arima_plus", "arima_plus_xreg"]` are listed together, `"arima_plus"` is automatically pinned to univariate `ARIMA_PLUS` while `"arima_plus_xreg"` normalizes to `"arima_plus@xreg"` (`ARIMA_PLUS_XREG`) for side-by-side Covariate FVA.
- **`timesfm` & `bq_timesfm` Clarity Alias vs. `timesfm_torch`:** BigQuery's managed `AI.FORECAST` SQL function is registered as `"timesfm"` and also accepts `"bq_timesfm"` (or `"bq_timesfm@<tag>"`) in `models` and `model_params`, normalizing `"bq_timesfm"` to `"timesfm"` with identical `run_id` digests so configs can explicitly distinguish serverless BigQuery `AI.FORECAST` (`timesfm` / `bq_timesfm`) from open-weight PyTorch `timesfm_torch`.
- **Per-Series `ML.ARIMA_EVALUATE` & `ML.ARIMA_COEFFICIENTS` Diagnostics:** After fitting the final `ARIMA_PLUS` / `ARIMA_PLUS_XREG` model, `bigquery_engine` queries `ML.ARIMA_EVALUATE(MODEL ..., STRUCT(FALSE AS show_all_candidate_models))` and `ML.ARIMA_COEFFICIENTS(MODEL ...)` and records each series' selected `(p, d, q)` order (`non_seasonal_order`), `has_drift`, `aic`, `log_likelihood`, `variance`, `seasonal_periods`, `has_holiday_effect`, `has_spikes_and_dips`, `has_step_changes`, `ar_coefficients`, `ma_coefficients`, `intercept_or_drift`, and linear `xreg_weights` inside `forecast_metadata.fit_diagnostics["arima_diagnostics"]`.
- **In-Database Anomaly & Change-Point Detection (`Forecaster` SDK):** Both native models support historical anomaly detection via `forecaster.detect_anomalies_df(model_name="arima_plus" | "timesfm", anomaly_prob_threshold=0.95)` (`ML.DETECT_ANOMALIES` for `arima_plus` / `arima_plus@xreg` and `AI.DETECT_ANOMALIES` for `timesfm`), and `arima_plus` supports structural regime break detection via `forecaster.detect_change_points_df(model_name="arima_plus")` (`ML.DETECT_CHANGE_POINTS` with gap-edge artifact filtering).

| Model | Upstream Package | Underlying BigQuery SQL Construct | Authored `model_params` & Defaults | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `arima_plus` (`arima_plus_xreg` alias) | [`bigquery-ml`](https://cloud.google.com/bigquery/docs/bqml-introduction) | `CREATE MODEL ... OPTIONS(model_type='ARIMA_PLUS' \| 'ARIMA_PLUS_XREG')` + `ML.FORECAST` + `ML.EXPLAIN_FORECAST` + `ML.ARIMA_EVALUATE` + `ML.ARIMA_COEFFICIENTS` | `auto_arima_max_order` (`1..5`), `auto_arima_min_order` (`0..5`), `clean_spikes_and_dips` (`bool`), `adjust_step_changes` (`bool`), `decompose_time_series` (`bool`), `holiday_region` (`str` \| `list[str]`), `seasonalities` (`list` of `"NO_SEASONALITY"`, `"DAILY"`, `"WEEKLY"`, `"MONTHLY"`, `"QUARTERLY"`, `"YEARLY"`), `time_series_length_fraction` (`(0, 1]`), `min_time_series_length` (`int >= 1`), `max_time_series_length` (`int >= 1`), `trend_smoothing_window_size` (`int >= 1`) | Automated per-series seasonal ARIMA pipeline with custom country holiday CTEs (`build_custom_holiday_cte`), future exogenous covariates (`ARIMA_PLUS_XREG`), `ML.EXPLAIN_FORECAST` attributions, and `ML.ARIMA_EVALUATE` / `ML.ARIMA_COEFFICIENTS` diagnostics. Rejects `forecast_limit_lower_bound` / `forecast_limit_upper_bound` (use `output.floor` / `output.cap` instead) and `auto_arima=False` on multi-series panels. |
| `timesfm` (`bq_timesfm` alias) | [`bigquery-ml`](https://cloud.google.com/bigquery/docs/bqml-introduction) | [`AI.FORECAST(...)`](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast) | `version` / `model`: `"TimesFM 2.0"`, `"TimesFM 2.5"` (univariate default), or `"TimesFM 3.0"` (shorthand `"2.0"`, `"2.5"`, `"3.0"`, auto-selected when covariates are active); `context_window`: `{64..15360}` for 2.0/2.5 or `32n` in `[64, 2048]` for 3.0 | Zero-shot foundation-model inference per `ts_id` in SQL with no `CREATE MODEL` training step. Supports multivariate `future_covariates` + `past_covariates` via `TimesFM 3.0` (`target_cols => ['y']`). Validated at plan time (`max_horizon <= 10000` for 2.0/2.5; `max_horizon <= 1024` for 3.0). |

---

<a id="two-tier-explainability-and-attribution-mechanics"></a>
## Two-tier explainability & model-level attribution mechanics

`scale-forecasting` provides two complementary explainability surfaces so forecasters can inspect both **feature-level causal attributions** (`attributions_df` / `plot_attributions`) and **structural time-series decomposition** (`explain_forecast` / `plot_forecast_explanation`):

```mermaid
flowchart LR
    subgraph Models["10 Native Attribution Models (supports_explainability = True)"]
        ML_MODS["5 Python ML Models (_lag_forecaster.py)<br/>xgboost · lightgbm · catboost<br/>regression_lags · random_forest"]
        AML_MODS["4 Vertex AI AutoML Models (automl_engine.py)<br/>vertex_l2l · vertex_tide<br/>vertex_tft · vertex_seq2seq"]
        BQ_MODS["1 BigQuery ML Model (bigquery_engine.py)<br/>arima_plus (ML.EXPLAIN_FORECAST)"]
    end

    subgraph Tier1["Tier 1: Global / Per-Series Driver Importance"]
        T1_BQ["forecast_metadata.fit_diagnostics<br/>$.feature_attributions = {feature: importance}"]
    end

    subgraph Tier2["Tier 2: Per-Horizon-Step Local Attributions"]
        T2_BQ["forecast_predictions.explanations<br/>{baseline_score: float, attributions: {feature: signed_delta}}"]
    end

    subgraph SDK["Unified SDK & Offline Helpers (sdk.py)"]
        DF["forecaster.attributions_df(level='both' | 'global' | 'local')<br/>build_attributions_frame(meta_rows, pred_rows)"]
        PLT["forecaster.plot_attributions(ts_id, model_type)<br/>2-Panel Global Bar Chart + Horizon-Step Stacked Bar"]
    end

    ML_MODS & AML_MODS & BQ_MODS --> T1_BQ & T2_BQ
    T1_BQ & T2_BQ --> DF --> PLT
```

### Model-level attribution mechanics (`supports_explainability = True`)

All **10 models** with native explainability (`ml`, `automl`, and `arima_plus` in `native`) populate Tier-1 (`forecast_metadata.fit_diagnostics["feature_attributions"]`) and Tier-2 (`forecast_predictions.explanations`) attributions automatically:

| Model(s) | Family | Tier 1 (Series / Global Importance) | Tier 2 (Per-Horizon-Step Local Attribution) | Additivity & Baseline Contract |
| :--- | :--- | :--- | :--- | :--- |
| **`xgboost`** | `ml` | Normalized mean $\|\phi_{t,j}\|$ across training / horizon (`feature_importances_` fallback) | Exact C++ TreeSHAP via `Booster.predict(DMatrix, pred_contribs=True)` at each recursive or direct step $h \in \{1,\dots,H\}$ | Exact: $\hat{y}_{T+h}^{\text{raw}} = \text{baseline\_score} + \sum_j \phi_{h,j}$ (transformed scale) |
| **`lightgbm`** | `ml` | Normalized mean $\|\phi_{t,j}\|$ across horizon (`feature_importances_` fallback) | Exact C++ TreeSHAP via `LGBMRegressor.predict(x_step, pred_contrib=True)` at each recursive or direct step $h$ | Exact: $\hat{y}_{T+h}^{\text{raw}} = \text{baseline\_score} + \sum_j \phi_{h,j}$ |
| **`catboost`** | `ml` | Normalized mean $\|\phi_{t,j}\|$ across horizon (`get_feature_importance()` fallback) | Exact C++ TreeSHAP via `CatBoostRegressor.get_feature_importance(Pool, type="ShapValues")` | Exact: $\hat{y}_{T+h}^{\text{raw}} = \text{baseline\_score} + \sum_j \phi_{h,j}$ |
| **`regression_lags`** | `ml` | Normalized $\|w_j\| \cdot \sigma_{X_j}$ standardized linear weight share | Closed-form linear attribution $\phi_{h,j} = w_j (x_{h,j} - \bar{x}_j)$ around $\text{baseline\_score} = \bar{y}_{\text{train}}$ | Exact: $\hat{y}_{T+h}^{\text{raw}} = \bar{y}_{\text{train}} + \sum_j w_j (x_{h,j} - \bar{x}_j)$ |
| **`random_forest`** | `ml` | Impurity importance `feature_importances_` normalized to sum to $1$ | Importance-weighted directional deviation allocated across active features at step $h$ around $\bar{y}_{\text{train}}$ | Additive: $\hat{y}_{T+h}^{\text{raw}} = \bar{y}_{\text{train}} + \sum_j \phi_{h,j}$ |
| **`vertex_l2l`**, **`vertex_tide`**, **`vertex_tft`**, **`vertex_seq2seq`** | `automl` | Mean $\|\text{attribution}\|$ per feature aggregated across each series' horizon steps (or Vertex `ModelEvaluation` feature attributions) | Vertex AI Explainable AI (`BatchPredictionJob(generate_explanation=True)` `explanation.attributions` `baseline_score` + `feature_attributions`) | Managed Vertex AI baseline-anchored Shapley / Integrated Gradients attribution per `(ts_id, forecast_date)` |
| **`arima_plus`** | `native` | Normalized mean $\|\text{attribution}\|$ across `trend`, `seasonal`, `holiday_effect`, and each `attribution_<col>` from `ML.EXPLAIN_FORECAST` | BigQuery `ML.EXPLAIN_FORECAST` decomposition (`baseline_score = trend + seasonal`, `attributions = {trend, seasonal, holiday_effect, <xreg_cols>}`) | Additive decomposition of `ARIMA_PLUS` / `ARIMA_PLUS_XREG` components per `(ts_id, forecast_date)` |
| **All 37 Models** *(including `statistical`, `deep_learning`, `native`)* | All 5 families | Post-hoc structural decomposition (`explain_forecast_frame` / `forecaster.explain_forecast`) | Decomposes any series' trajectory into **Trend**, **Regime Level Shift**, **Seasonal Cycle**, **Exogenous Covariate Alignment**, and **Conformal Uncertainty Width** | Model-agnostic: works identically across univariate statistical, neural, foundation, AutoML, and BigQuery SQL models |

---

<a id="hierarchical-forecasting-coherent-reconciliation"></a>
## Hierarchical forecasting & coherent reconciliation

When `hierarchy.enabled: true`, the platform aggregates the bottom-level series across `hierarchy.levels` (including `"__total__"` at the root), fits base models across all hierarchy nodes, and reconciles base forecasts into coherent hierarchy-wide forecasts $\tilde{y_h} = S G \hat{y_h}$ ([`reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/reconciliation.py)) following [Hyndman & Athanasopoulos (*Forecasting: Principles and Practice*, 3rd ed., Ch. 11)](https://otexts.com/fpp3/hierarchical.html) and [Wickramasuriya et al. (2019) *MinT*](https://doi.org/10.1080/01621459.2018.1448825):

| Reconciliation Method | Matrix $G$ / Weighting $W_h$ | Methodology |
| :--- | :--- | :--- |
| **`bottom_up`** | $G = [0 \mid I_{n_b}]$ | Preserves bottom-level forecasts verbatim and sums them upward via the summing matrix $S$. |
| **`top_down`** | $G = [p \mid 0]$ | Disaggregates the top-level (`"__total__"`) forecast downward using average historical proportions $p_j = \frac{1}{T}\sum_t y_{j,t} / y_{\text{Total},t}$. |
| **`middle_out`** | Disaggregate from `middle_level`, sum upward | Preserves base forecasts at `hierarchy.middle_level`, sums upward to higher levels, and disaggregates downward to bottom series by historical proportions. |
| **`ols`** | $W_h = I_n$ | Ordinary least squares MinT projection $G = (S^\top S)^{-1} S^\top$. |
| **`wls_struct`** | $W_h = \text{diag}(S \mathbf{1})$ | Structural scaling weighted least squares; requires only the hierarchy structure $S$. |
| **`wls_var`** | $W_h = \text{diag}(W_1)$ | Variance scaling weighted least squares using 1-step/OOF residual variances per node. |
| **`mint_shrink`** | $W_h = \lambda_D W_{1,D} + (1 - \lambda_D) W_1$ | [Wickramasuriya et al. (2019)](https://doi.org/10.1080/01621459.2018.1448825) Minimum Trace optimal reconciliation with analytical [Schäfer-Strimmer (2005)](https://doi.org/10.2202/1544-6115.1175) shrinkage covariance of residuals; guaranteed positive-definite even when $n_{\text{series}} \gg T_{\text{obs}}$. |
| **`mint_nonneg`** | $\min_{\tilde{b} \ge 0} (S\tilde{b} - \hat{y})^\top W_h^{-1} (S\tilde{b} - \hat{y})$ | Non-negative constrained MinT quadratic program ([Wickramasuriya, Turlach & Hyndman 2020](https://doi.org/10.1007/s11222-020-09930-0)) solved via `scipy.optimize.minimize(method="SLSQP", bounds=[(0, None)])`; guarantees $\tilde{y} = S\tilde{b} \ge 0$ at every node while preserving exact hierarchy coherence without post-hoc clipping distortion. |

### Post-hoc matrix reconciliation vs. cross-series global models

- **All 35 Python and Vertex AI AutoML models** (`statistical`, `ml`, `deep_learning`, `automl`) support all **8 post-hoc reconciliation methods** above.
- **Local per-series models (`training_mode: "local"`):** Fit each bottom and aggregated upper-level node (`"__total__"`, `"region=NA"`, `"region=NA/category=SMB"`, `"s_000000"`, $\dots$) independently, then apply the reconciliation projection ($\tilde{y_h} = S G \hat{y_h}$ or constrained QP $\tilde{y_h} = S\tilde{b}_h \ge 0$) across both backtest OOF folds and the final horizon forecast.
- **Global & hybrid panel models (`training_mode: "global"` or `"hybrid"`):** Models like `tide`, `tft`, `tsmixer`, `patchtst`, `neuralprophet`, `chronos_bolt`, `chronos2`, `timesfm_torch`, `vertex_l2l`, `vertex_tide`, `vertex_tft`, and `vertex_seq2seq` fit or condition a single shared network simultaneously across all hierarchy levels (and any homogeneous `static_covariates` inherited by upper-level nodes), learning cross-level interactions inside the network weights during training or batched foundation inference. Because shared neural weights alone do not enforce strict linear additivity ($y_{\text{upper}} = \sum y_{\text{bottom}}$), the platform still applies the configured post-hoc reconciliation projection to both point forecasts and prediction intervals (`yhat_lower`, `yhat_upper`), guaranteeing exact coherence across the entire tree.
- **BigQuery-native models (`arima_plus`, `timesfm`):** Execute directly in BigQuery SQL on the source table without driver-side hierarchy rollups (`hierarchy.enabled: true` is rejected at preflight for `native` models with a clear `ConfigError`).

---

## Ensemble blending & stacking strategies

When `ensemble.enabled: true`, the driver combines base-model forecasts into one or more `ensemble_<strategy>` pseudo-models and writes them to `forecast_predictions`, `backtest_oof`, and `forecast_metadata` alongside the base models:

| Strategy | Type | Requires Backtest | Upstream Package | Methodology |
| :--- | :--- | :---: | :--- | :--- |
| **`mean`** | Calculated | No | [`numpy`](https://numpy.org/) | Unweighted arithmetic average across all available base models per `(ts_id, forecast_date)`. |
| **`median`** | Calculated | No | [`numpy`](https://numpy.org/) | Row-wise median across base models; robust to single-model divergence. |
| **`trimmed_mean`** | Calculated | No | [`numpy`](https://numpy.org/) | Symmetric 20% trimmed mean (`trim_ratio=0.2`) dropping the highest and lowest base predictions at each `(ts_id, forecast_date)` before averaging when $\ge 3$ models are present. |
| **`inverse_error`** | Calculated | Recommended | [`numpy`](https://numpy.org/) | Weighted average with weights $w_m \propto 1 / \text{loss}_m$ under `backtest.decision_metric` (with optional shrinkage toward uniform $1/K$ via `ensemble.weight_shrinkage`), normalized to sum to $1$. |
| **`champion`** | Calculated Selection | Yes | [`numpy`](https://numpy.org/) | Per-series winner-take-all selector ($w_{m^*} = 1.0$, $0$ elsewhere) picking the single model with the lowest out-of-fold `backtest.decision_metric` loss on that `ts_id`. |
| **`horizon_specialist`** | Learned Horizon Blending | Yes | [`numpy`](https://numpy.org/) | Partitions the forecast horizon into 3 buckets (short `1..H/3`, medium, long) and computes inverse-RMSE weights per `(ts_id, bucket)` so fast autoregressive models dominate near-term steps and trend/seasonal models dominate long-horizon steps. |
| **`nnls`** | Learned Stacking | Yes | [`scipy`](https://scipy.org/) | Non-Negative Least Squares (`scipy.optimize.nnls`) fit on out-of-fold predictions (`w_m >= 0`, normalized to sum to $1$, with optional shrinkage toward $1/K$ via `ensemble.weight_shrinkage`). |
| **`ridge`** | Learned Stacking | Yes | [`numpy`](https://numpy.org/) | Closed-form L2-regularized linear meta-learner ($\alpha = 1.0$) fit on the out-of-fold matrix; allows negative corrective weights. |
| **`xgb`** | Learned Stacking | Yes | [`xgboost`](https://xgboost.readthedocs.io/) | Non-linear gradient-boosted tree meta-learner (`XGBRegressor`) trained on out-of-fold predictions. |

---

## Environment agility & optional package management

All third-party model libraries are imported **lazily inside `fit()`** rather than at module import time. This design ensures that:

1. **Every model registers unconditionally** in `scale_forecasting.models` with zero heavy imports at startup.
2. **Restricted or air-gapped environments** that omit specific packages (for example, omitting `catboost` or `neuralprophet`) can still import `scale_forecasting` and run every other model whose upstream package is installed.

### Granular installation extras

`pyproject.toml` layers the install (see [runtime_dependencies.md](./runtime_dependencies.md#dependency-extras) for the whole table): the bare package already fits 15 of the 37 models, and family-scoped extras add the rest so you install only the packages approved for your environment. Each model declares the extra it needs (`optional_extra`) and the message it raises when that package is missing names it.

| Extra | Installed upstream packages | Models enabled |
| :--- | :--- | :--- |
| *(core — no extra)* | `numpy`, `scipy`, `statsmodels>=0.14`, `scikit-learn>=1.5` | `naive_mean`, `naive_seasonal`, `naive_drift`, `naive_moving_average`, `croston`, `regression_lags`, `fft`, `theta`, `holtwinters`, `autoets`, `stl_bagging`, `sarimax`, `ucm`, `kalman`, `random_forest` (15) |
| `scale-forecasting[models-stats]` | `statsforecast>=2.0` | `auto_arima`, `auto_theta`, `auto_ces`, `tbats` |
| `scale-forecasting[models-trees]` | `xgboost>=2.0`, `lightgbm>=4.3`, `catboost>=1.2` | `xgboost`, `lightgbm`, `catboost` |
| `scale-forecasting[models-prophet]` | `prophet>=1.1.5` | `prophet` |
| `scale-forecasting[models-dl]` | `neuralprophet>=0.9`, `neuralforecast>=1.7`, `torch>=2.13,<2.14` | `neuralprophet`, `tide`, `tft`, `tsmixer`, `patchtst` |
| `scale-forecasting[models-foundation]` | `chronos-forecasting>=2.2`, `timesfm[torch]>=3.0` | `chronos_bolt`, `chronos2`, `timesfm_torch` |
| `scale-forecasting[models-automl]` | `google-cloud-pipeline-components>=2.14` (+ `[gcp]`, which carries `google-cloud-aiplatform`) | `vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq` |
| `scale-forecasting[gcp]` | `google-cloud-bigquery` and the other Google Cloud clients | `arima_plus`, `timesfm` (BigQuery ML runs them; the client submits the SQL) |
| `scale-forecasting[models]` | The six `models-*` extras above | All 37 built-in models (with `[gcp]` for the two BigQuery ML models) |

### Inspecting and filtering available models

You can inspect package availability from the CLI or Python SDK, and optionally skip unavailable models when running a shared configuration in a restricted environment:

```bash
# List all registered models alongside their upstream package and installation status
uv run python -m scale_forecasting.playground --list

# Run a shared config while automatically skipping any models whose optional package is not installed
uv run python -m scale_forecasting.main --config configs/ensemble_demo.json --ignore-unavailable-models
```

From Python:

```python
from scale_forecasting.models import filter_available_models, get_model, list_models
from scale_forecasting.playground import model_catalog

# Return only models whose upstream package is importable in the current Python environment
installed = list_models(available_only=True)

# Inspect the full model catalog DataFrame (includes package, package_url, and available columns)
df_catalog = model_catalog()

# Check a single model class directly
cls = get_model("catboost")
print(cls.package, cls.package_url, cls.is_available())
```

---

## Adding a custom model

Adding a new model requires **one file** in `src/scale_forecasting/models/` and **one import line** in `src/scale_forecasting/models/__init__.py`:

1. Copy [`docs/model_template.py`](https://github.com/statmike/scale-forecasting/blob/main/docs/model_template.py) to `src/scale_forecasting/models/my_model.py`.
2. Set `name`, `family`, `package`, `package_url`, and implement `fit(y, X)` and `predict(horizon, X, quantiles)`.
3. End the file with `register(MyModel)` and import `my_model` in `src/scale_forecasting/models/__init__.py`.

Because `src/scale_forecasting` is zipped and shipped dynamically on every job submission, your custom model runs on Dataproc Serverless, Dataproc clusters, and Vertex AI Ray immediately with **zero container image rebuilds**.

See **[Adding a model (`docs/adding_a_model.md`)](./adding_a_model.md)** for the full contract and step-by-step walkthrough.
