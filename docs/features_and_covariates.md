# Feature Engineering, Three-Tier Covariates, Compositional Simplex, Similarity & Cold-Start Transfer

`scale-forecasting` provides a declarative, leak-free feature engineering, target-transformation, time-series similarity, and cold-start transfer pipeline across [`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py), [`simplex.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/simplex.py), [`monotonicity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/monotonicity.py), [`holiday_window.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/holiday_window.py), [`cold_start.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/cold_start.py), and [`similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/similarity.py). Every transformation executes strictly inside each rolling-origin backtest fold (`ds <= cutoff_date`) so future observations never leak into historical training windows.

```mermaid
flowchart TD
    subgraph S1["1. Fold-Safe Hygiene (ds <= cutoff_date)"]
        T1["Grid, Zero-Trim & Lineage<br/>grid_policy · censored_col<br/>supersession_map"]
        T2["Outliers & SBC Regimes<br/>winsorize · interpolate · stl_resid<br/>SBC 4-Regime Routing"]
        T3["Target & Simplex Transforms<br/>none · log1p · boxcox<br/>clr · ilr (simplex_sum)"]
        T1 --> T2 --> T3
    end

    subgraph S2["2. 3-Tier Covariates Across Cutoff"]
        C1["Tier 1 · Static Metadata<br/>static_covariates<br/>Store · category · region"]
        C2["Tier 2 · Known-Future (<= cutoff + H)<br/>future_covariates · holidays & ramps<br/>fourier · trading_days · HDD/CDD"]
        C3["Tier 3 · Past-Only (<= cutoff)<br/>past_covariates + exog_lags<br/>Masked > cutoff (no leakage)"]
        C1 --> C2 --> C3
    end

    subgraph S3["3. Guardrails, Similarity & Cold-Start"]
        V1["Capability & Sign Guardrails<br/>on_unsupported_covariates<br/>@monotone (price: -1, promo: +1)"]
        V2["@tag Ablation & Similarity<br/>@univariate .. @cov_all<br/>7 Lenses · @similarity_global"]
        V3["Cold-Start NPI & Decline<br/>analog_transfer · bass_diffusion<br/>weibull_lifecycle · @arps_decline"]
        V1 & V2 --> V3
    end

    T3 --> V1
    C3 --> V2

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class C1,C2,C3 data
    class V1,V2 route
    class T1,T2,T3 compute
    class V3 config
```

---

## 1. Three-Tier Exogenous Covariate Contract (`FeaturesConfig`)

Different model families consume exogenous information in fundamentally different ways. `FeaturesConfig` separates covariates into three explicit tiers and routes each tier only to models that declare support (`supports_static_covariates`, `supports_future_covariates`, `supports_past_covariates`):

| Covariate Tier | `RunConfig` Field | Cutoff & Horizon Rule | Models Supporting Tier |
| :--- | :--- | :--- | :--- |
| **Static Covariates** | `features.static_covariates` | Constant per `ts_id` across history and horizon (e.g., `["region", "category"]`). Categorical strings are deterministically encoded. | Global/hybrid panel models (`tide`, `tft`, `tsmixer`, `lightgbm@global`, `lightgbm@similarity_global`, `xgboost@global`, `catboost@global`) and Vertex AI AutoML (`vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq`). |
| **Known-Future Covariates** | `features.future_covariates` (alias `features.exog`) | Must be present in both historical training rows (`ds <= cutoff_date`) and future horizon rows (`cutoff_date < ds <= cutoff_date + H`). | `sarimax`, `prophet`, `regression_lags`, all `ml` tree models (`lightgbm`, `xgboost`, `catboost`, `random_forest`, `extra_trees`), `neuralprophet`, `tide`, `tft`, `tsmixer`, `chronos2`, `vertex_*`, and `arima_plus`. |
| **Historical-Only Past Covariates** | `features.past_covariates` | Observed only up to `ds <= cutoff_date`. Strictly masked beyond `cutoff_date` during backtesting and HPO to prevent lookahead leakage; projected into future steps via `exog_lags` or cutoff carry-forward for tabular models. | Native encoder consumption in `tide`, `tft`, `tsmixer`, `neuralprophet`, and `vertex_*`; lagged/carried consumption in `ml`, `statistical`, and `arima_plus`. |

### Deterministic Calendar, Holiday Window & Structural Regressors
- **Country & Custom Holidays (`features.holidays`, `features.custom_holidays`):** Generates binary holiday indicators (`is_holiday`) for any ISO country code (e.g., `["US", "GB"]`) plus custom event dates.
- **Moving-Holiday Lead/Lag Windows (`features.holiday_window`):** Adds continuous pre-holiday ramp-up (`pre_holiday_window`, `days_to_next_holiday`) and post-holiday dip (`post_holiday_window`, `days_since_prev_holiday`, `holiday_offset_days`) features around movable holidays (Easter, Ramadan, Lunar New Year, Cyber Week) using `"linear"`, `"cosine"`, or `"gaussian"` kernels.
- **Fourier Seasonality (`features.fourier`), Trading Days (`features.trading_days`), Level Shifts (`features.level_shift`) & Piecewise Degree-Day Splines (`features.degree_day_cols`):** Appends harmonic sine/cosine pairs, monthly/weekly business-day counts (`trading_days_in_period`), automatic structural step-change indicators (`level_shift`), and piecewise Heating/Cooling Degree-Day regressors (`hdd_<col> = max(0, base_temp - col)`, `cdd_<col> = max(0, col - base_temp)`, also available standalone via `compute_degree_days_df` with automatic `+1` `@monotone` sign inference).

---

## 2. Compositional Simplex Transforms (`clr` & `ilr`)

When forecasting **market shares, category mix percentages, portfolio weights, or channel split proportions**, the $D$ series across the panel are coupled by the constant-sum simplex constraint at every timestamp $t$:

$$y_{i,t} > 0, \qquad \sum_{i=1}^{D} y_{i,t} = S \quad (S = 1.0 \text{ or } S = 100.0)$$

Forecasting each share independently in raw space produces predictions that drift below $0$, exceed $S$, or sum to $\ne S$. Setting `features.transform = "clr"` or `"ilr"` (or appending `@transform_clr` / `@transform_ilr` to any model name, such as `"lightgbm@transform_ilr"` or `"theta@transform_clr"`) activates Aitchison geometry in [`simplex.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/simplex.py):

| Transform Mode | Forward Mapping ($\mathcal{S}^D \to \mathbb{R}^K$) | Inverse Closure ($\mathbb{R}^K \to \mathcal{S}^D$) | Key Properties |
| :--- | :--- | :--- | :--- |
| **Centered Log-Ratio (`"clr"` / `@transform_clr`)** | $z_{i,t} = \ln y_{i,t}^* - \frac{1}{D}\sum_{j=1}^D \ln y_{j,t}^*$ | $\hat{y}_{i,t} = S \cdot \frac{\exp(\hat{z}_{i,t})}{\sum_{j=1}^D \exp(\hat{z}_{j,t})}$ | Symmetric 1-to-1 series mapping; works seamlessly with per-series (`local`), panel (`global`), and ensemble models followed by softmax simplex closure. |
| **Isometric Log-Ratio (`"ilr"` / `@transform_ilr`)** | $\mathbf{z}_t = \text{clr}(\mathbf{y}_t^*) \cdot \mathbf{V}$ using the $D \times (D-1)$ orthonormal **Helmert contrast matrix** $\mathbf{V}$ | $\hat{\mathbf{y}}_t = \text{clr}^{-1}(\hat{\mathbf{z}}_t \mathbf{V}^\top, S)$ | Full-rank isometric coordinate representation in $\mathbb{R}^{D-1}$ (`ilr_transform` / `ilr_inverse`, `forecast_simplex_panel`) and per-series log-odds closure across all 37 models and 8 ensemble strategies. |

### Zero Replacement, Post-Blend Evolution & Simplex Coherence Verification
1. **Martín-Fernández Multiplicative Zero Replacement (`multiplicative_replacement`):** Structural or transient zeros ($y_{i,t} = 0$) are replaced prior to log-transformation using $y_{i,t}^* = \delta$ for zero parts and $y_{i,t}^* = y_{i,t} \cdot (1 - c \cdot \delta / S)$ for positive parts, preserving exact cross-series sums $\sum_i y_{i,t}^* = S$ with zero NaNs or infinities.
2. **Calibrated Bounded Intervals:** Prediction intervals (`yhat_lower`, `yhat_upper`) are mapped back into $[0, S]$ with strict ordering $0 \le \hat{y}_{\text{lower}} \le \hat{y} \le \hat{y}_{\text{upper}} \le S$.
3. **Automatic Detection, `f.assess()` & `Forecaster.evolve()` Closure:** `Forecaster.profile_data()` automatically flags compositional panels (`is_compositional_simplex=True`, `simplex_total=1.0` or `100.0`), `f.assess()` surfaces causal guardrails (`@monotone`, `monotonic_constraints`, `holiday_window`, and `f.promo_elasticity_df()`) in Question 3 and simplex closure status in Question 5, `Forecaster.evolve(target="ensemble")` enforces post-blend `enforce_simplex_closure()` across inner CV, Gate-3 lockdown, and materialized `result.predictions_df`, and `verify_simplex(predictions_df, simplex_sum=S)` verifies $\max_t |\sum_i \hat{y}_{i,t} - S| < 10^{-6}$ (auto-detecting `"ds"`, `"forecast_date"`, or `"date"`).

---

## 3. Cold-Start & New Product Introduction (NPI) Analog Transfer (`cold_start.py`) & Multi-Lens Similarity (`similarity.py`)

When launching **new SKUs, newly opened stores, or short-history products** with fewer than `features.cold_start_min_obs` observations (default `14`), univariate local statistical models lack sufficient degrees of freedom to estimate seasonal profiles or trend slopes reliably. Setting `features.cold_start_transfer = True` (or appending `@similarity_cold_start` to any model name) activates [`cold_start.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/cold_start.py) and [`similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/similarity.py):

| `features.cold_start_method` | Mechanism | Best For |
| :--- | :--- | :--- |
| **`"auto"`** *(default)* | Selects `"analog_transfer"` when mature donor series ($n \ge \max(2 n_{\min}, m + H)$) exist in the panel and falls back to `"bass_diffusion"` / `"weibull_lifecycle"` otherwise. | Heterogeneous enterprise catalogs mixing mature and newly launched series (`cold_start_short_history`). |
| **`"analog_transfer"`** | Computes a composite donor similarity score governed by `features.cold_start_matching` (`"auto"`, `"attribute"`, `"similarity"`, or `"hybrid"`), smoothly blending Day-0 pre-launch metadata/semantic/hierarchy priors with Day-1..14 kinematic launch-prefix matching (`cold_start_handoff`), then transfers the top-$K$ donors' seasonal shape scaled by empirical-Bayes level shrinkage. | Retail SKU launches, new store openings, and replacement product rollouts with peer history. |
| **`"bass_diffusion"`** | Fits the Bass (1969) innovation/imitation S-curve $y(t) = m \cdot \frac{(p+q)^2}{p} \frac{e^{-(p+q)t}}{(1 + (q/p)e^{-(p+q)t})^2}$ regularized toward panel priors $(\bar{p}, \bar{q})$. | Technology adoption, subscription cohorts, and cumulative launch ramps without close analogs. |
| **`"weibull_lifecycle"`** | Fits a 3-parameter Weibull launch-peak-decay curve $y(t) = A \cdot (k/\lambda)(t/\lambda)^{k-1} e^{-(t/\lambda)^k}$ with empirical-Bayes shape stabilization. | Fashion, seasonal apparel, media, and promotional limited-edition lifecycles. |
| **`"arps_decline"`** (`@arps_decline`) | Fits the Arps (1945) decline curve $q(t) = q_i (1 + b D_i t)^{-1/b}$ across exponential ($b=0$), hyperbolic ($0 < b < 1$), and harmonic ($b=1$) regimes (`fit_arps_decline`) and computes closed-form cumulative **Estimated Ultimate Recovery (`eur_total`, `eur_remaining`)**. | Oil & gas well production, cloud cohort retention decay, and post-launch saturation depletion. |

- **Product Supersession & Generation Splicing (`features.supersession_map` & `apply_supersession_map`):** Declaring `supersession_map={"SKU_v2": "SKU_v1", "SKU_v3": "SKU_v2"}` traces multi-generation `v1 -> v2 -> v3` predecessor chains, sums demand across phase-out/phase-in overlap windows, and assigns the successor's latest static metadata so newly transitioned SKUs inherit full historical depth instead of falling back to cold-start.

### Seven Similarity Lenses, Multi-Modal Metadata Fusion, `@similarity_global` Cluster Pooling & Merchandising Intelligence (`similarity.py`)

`scale_forecasting.similarity` computes symmetric positive semi-definite (PSD) similarity matrices ($S_{i,i} = 1, S_{i,j} \in [0, 1]$) across **7 temporal lenses** plus optional **multi-modal retail metadata** (`text_cols` character/word $n$-gram TF-IDF + SVD LSA, `attribute_cols` Gower similarity, and `hierarchy_cols` Wu–Palmer Lowest Common Ancestor tree distance):

| Similarity Lens (`method`) | Representation & Mathematical Invariants | Primary Use Case |
| :--- | :--- | :--- |
| **`"hybrid"`** *(default)* | Consensus PSD fusion of rank-normalized RBF kernels across `"physics_attractor"` (`0.30`), `"spectral_phase"` (`0.25`), `"minirocket"` / `"elastic_dtw"` (`0.25`), and `"structural_signature"` (`0.20`), plus optional `semantic_sim` and `hierarchy_sim`. | General-purpose digital-twin discovery, `@similarity_global` cluster pooling, `similarity_hierarchy`, and SKU substitution. |
| **`"physics_attractor"`** | Takens delay-coordinate state-space reconstruction, kinematic phase portraits $(y_t, \dot{y}_t, \ddot{y}_t)$, Sliced Wasserstein attractor cloud distance, Rosenstein maximal Lyapunov exponent $\lambda_{\max}$, Permutation Entropy $H_{\text{perm}}$, Sample Entropy, and Ornstein–Uhlenbeck damping rate $\kappa_{\text{OU}}$. | Separating chaotic, mean-reverting ($H < 0.5$), persistent ($H > 0.5$), and limit-cycle dynamical regimes; matching early launch-prefix velocity/acceleration. |
| **`"spectral_phase"`** | FFT Hilbert analytic signal $z_t = y_t + i\mathcal{H}(y_t) = A_t e^{i\theta_t}$, pairwise Kuramoto phase-locking order parameter $R_{i,j}$, Welch PSD Jensen–Shannon divergence, and directed cross-correlation lead-lag shift $\tau_{i \to j}^*$. | Discovering shared dominant periodicities, phase coherence, and bellwether lead-lag relationships across series. |
| **`"elastic_dtw"`** | Scale-normalized Sakoe–Chiba banded Dynamic Time Warping / Time-Warp Edit Distance (`TWED`) with physical elastic spring stiffness penalty $\nu > 0$. | Shape-based matching under phase shifts, launch timing offsets, and local time warping. |
| **`"minirocket"`** | Deterministic pure-NumPy bank of `84` ternary dilated convolutional kernels ($\mathbf{w} \in \lbrace -1, 2\rbrace^9, \sum w_k = 0$) across exponential dilations `(1, 2, 4, 8)` with Proportion of Positive Values (`PPV`) pooling. | Fast $O(N \cdot T)$ multiscale local motif, promo-spike, and morphological shape matching. |
| **`"structural_signature"`** | `24`-dimensional standardized econometric vector: STL trend/seasonal strength, peak phase, Hurst exponent $H$ ($R/S$), ACF lags `1..7`, PACF lags `1..3`, spectral entropy, Syntetos–Boylan $(\log \text{ADI}, \log \text{CV}^2)$, skewness, kurtosis, ARCH(1), level-shift, zero-inflation, velocity, and curvature ratios. | Interpretable econometric regime grouping across smooth, erratic, intermittent, and lumpy demand. |
| **`"foundation_latent"`** | Pooled context encoder embeddings from pretrained time-series foundation models (`chronos_bolt`, `chronos2`, `timesfm_torch`, `sundial` via `extract_foundation_embeddings`) with deterministic L2-normalized `"minirocket"` + `"structural_signature"` fallback on pure core. | Zero-shot neural representation clustering and foundation-space analog retrieval. |

- **Cluster-Aware Global Pooling (`<model>@similarity_global` & `features.similarity_cluster_feature`):** Appending `@similarity_global` to any global-capable model (e.g., `"lightgbm@similarity_global"`, `"regression_lags@similarity_global"`, `"tide@similarity_global"`) partitions the panel strictly within each fold cutoff via `similarity.cluster_by_similarity_df()` and fits an isolated global model per behavioral cluster (`_run_similarity_global_panel_model` in `worker.py` and cluster-isolated buckets in `spark_io.py`), preventing negative transfer between incompatible regimes. Setting `features.similarity_cluster_feature = True` injects `similarity_cluster_id` (`0..K-1`) as a feature into standard models.
- **Day-0 to Day-14 Cold-Start Digital-Twin Matching (`features.cold_start_matching` & `@similarity_cold_start`):** Setting `features.cold_start_matching` to `"hybrid"` or `"similarity"` (or using `<model>@similarity_cold_start`) routes short-history cold-start series through `similarity.match_cold_start_launch_prefix_df()`, smoothly ramping from Day-0 pre-launch semantic/attribute/hierarchy donor priors (`0` observations) to Day-1..14 kinematic launch-prefix matching.
- **SKU Substitute Recommendations & Emerging Trend Detection (`recommend_substitutes_df` & `detect_emerging_trends_df`):** `sf.recommend_substitutes_df(panel_df, inventory_df=..., text_cols=..., hierarchy_cols=..., within_hierarchy_level=...)` ranks in-stock digital-twin substitutes when an item is low or out of stock, while `sf.detect_emerging_trends_df(panel_df, window=14)` combines kinematic velocity ($\dot{y}_t$), acceleration ($\ddot{y}_t$), Hurst persistence ($H$), and Kuramoto lead-lag shifts ($\tau_{i \to j}^*$) to flag accelerating breakouts and lagging co-promotion candidates.

`Forecaster.profile_data()` automatically computes `optimal_similarity_clusters`, `similarity_silhouette_score`, and `has_similarity_clusters`, `Forecaster.auto()` evaluates `<model>@similarity_global`, `@similarity_cold_start`, and `hierarchy.similarity_hierarchy=True` in `autofocus`, and `f.assess()` reports digital-twin, cluster-pooling, and latent-hierarchy diagnostics in Questions 4 and 5.

---

## 4. Declarative `RunConfig` & Python SDK Examples

```json
{
  "run_name": "compositional_market_share_tournament",
  "python_runtime": "gce",
  "data": {
    "source_table": "source_series_covariates_iceberg",
    "freq": "D",
    "horizon": 14,
    "series_limit": 12
  },
  "models": [
    "theta@transform_ilr",
    "holtwinters@transform_clr",
    "lightgbm@transform_ilr"
  ],
  "features": {
    "transform": "ilr",
    "simplex_sum": 100.0,
    "future_covariates": ["is_holiday", "promo_flag"],
    "on_unsupported_covariates": "fallback",
    "cold_start_transfer": true,
    "cold_start_min_obs": 14,
    "cold_start_method": "auto"
  },
  "backtest": {
    "enabled": true,
    "horizon": 14,
    "n_folds": 2,
    "step": 14,
    "decision_metric": "wape"
  },
  "ensemble": {
    "enabled": true,
    "strategies": ["mean", "inverse_error", "nnls"]
  }
}
```

### Offline & SDK Walkthrough

```python
import scale_forecasting as sf

# 1. Load a 5-brand market-share compositional panel (sums to 100.0% at every date)
bench = sf.load_benchmark("compositional_market_share_simplex")

# 2. Forecast in full-rank ILR Helmert coordinates or via @transform_ilr / @transform_clr variants
ilr_preds = sf.forecast_simplex_panel(
    bench.df,
    model_name="theta",
    transform="ilr",
    horizon=14,
    simplex_sum=100.0,
)
print("Max simplex discrepancy:", sf.verify_simplex(ilr_preds, simplex_sum=100.0))

# 3. Synthesize cold-start / NPI trajectories on dgp_11_cold_start_mixed
cs_bench = sf.load_benchmark("dgp_11_cold_start_mixed")
cs_df = sf.cold_start_transfer_df(cs_bench.df, min_obs=14, method="auto", horizon=14)
print(cs_df[["ts_id", "n_obs", "is_cold_start", "selected_archetype", "donor_ts_ids", "yhat_mean"]].head())

# 4. Run an automated bake-off and inspect the 7-Question Assessment
f = sf.Forecaster.auto(
    data=bench.df,
    target_col="y",
    horizon=14,
)
print(f.leaderboard_df()[["model_type", "wape", "mase"]].head())
print(f.assess())
```

See also:
- [Notebook 05: Covariates and Global Models](./notebooks/05_covariates_and_global_models.ipynb)
- [Notebook 06: Hierarchical Reconciliation, THieF & Compositional Simplex](./notebooks/06_hierarchical_reconciliation.ipynb)
- [Configuration Reference](./configuration_reference.md)
- [API Reference: `features`](./api/features.md), [`simplex`](./api/simplex.md), [`monotonicity`](./api/monotonicity.md), [`holiday_window`](./api/holiday_window.md), [`cold_start`](./api/cold_start.md), [`similarity`](./api/similarity.md)

