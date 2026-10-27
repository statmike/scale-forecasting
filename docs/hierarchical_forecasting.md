# Hierarchical, Temporal & Cross-Temporal Reconciliation + Top-Down Goal-Seek

Enterprise supply-chain, retail, and financial planning require forecasts that add up coherently across **both** the organizational tree (SKU $\to$ Category $\to$ Region $\to$ Total) and the planning calendar (Daily $\to$ Weekly $\to$ Bi-Weekly $\to$ 4-Week $\to$ Quarterly $\to$ Annual), as well as top-down target allocation when executives revise aggregate revenue or unit targets.

`scale-forecasting` provides a unified, pure-NumPy/SciPy reconciliation and constrained target-allocation stack across [`reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/reconciliation.py), [`temporal_reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/temporal_reconciliation.py), and [`goal_seek.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/goal_seek.py) that executes identically in offline local runs (`playground.bakeoff`, `Forecaster.auto`), the Python SDK (`Forecaster`), and all distributed cloud runtimes (`spark`, `ray`, `vertex`, `gce`, `gke`).

```mermaid
flowchart TD
    IN["Base & Backtest OOF Forecasts + RunConfig.hierarchy<br/>(Local Playground · Python SDK · Spark · Ray · Vertex · GCE · GKE)"]

    subgraph S1["1. Cross-Sectional & Similarity Hierarchy (reconciliation.py)"]
        CS1["Build Explicit Tree or Latent Clusters<br/>hierarchy.levels OR similarity_hierarchy = true<br/>Summing Matrix S_cs (__total__ -> levels -> leaves)"]
        CS2["Project Base & OOF onto Spatial Subspace<br/>bottom_up · top_down · middle_out · ols<br/>wls_struct · wls_var · mint_shrink · mint_nonneg"]
        CS3["Verify Spatial Coherence<br/>verify_coherence(reconciled_df, spec) < 1e-6"]
        CS1 --> CS2 --> CS3
    end

    subgraph S2["2. Temporal & Cross-Temporal THieF (temporal_reconciliation.py)"]
        TH1["Compile Non-Overlapping Calendar Hierarchy<br/>default_temporal_aggregations(freq)<br/>Cutoff-aligned buckets + S_temp = I_(m/k) (x) 1_k^T"]
        TH2["3-Mode Routing & Spatial-Temporal Weighting<br/>spatial_only (alpha=1.0) · temporal_only (alpha=0.0)<br/>cross_temporal_mint (spatial_temporal_weight='auto')"]
        TH3["Verify Coherence & Sample Paths<br/>verify_cross_temporal_coherence < 1e-6<br/>sample_coherent_paths(n_paths=200, seed=42)"]
        TH1 --> TH2 --> TH3
    end

    subgraph S3["3. Top-Down Goal-Seek Constrained QP (goal_seek.py)"]
        GS1["Declare Targets, Locks, Bounds & Smoothness<br/>targets={'__total__': 125000} · locked_nodes<br/>[lower_bounds, upper_bounds] · smoothness_weight=10.0"]
        GS2["Solve Precision-Weighted Convex QP<br/>min_b 0.5*(b - b_hat)^T W^-1 (b - b_hat) + roughness<br/>s.t. C*b = d, l <= b <= u (GoalSeekInfeasibleError)"]
        GS3["Propagate Gross Margin & Inventory Impact<br/>allocated_yhat · adjustment_pct · gross_margin_delta · delta_reorder_point"]
        GS1 & GS2 --> GS3
    end

    IN --> CS1 & TH1
    CS3 --> GS1
    TH3 --> GS2

    classDef innode fill:#F3E8FD,stroke:#9334E6,stroke-width:2px,color:#5B1799
    classDef cs fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef th fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef gs fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    class IN innode
    class CS1,CS2,CS3 cs
    class TH1,TH2,TH3 th
    class GS1,GS2,GS3 gs
```

---

## 1. Three Orthogonal Reconciliation Regimes (`HierarchyConfig`)

`HierarchyConfig` organizes spatial and temporal reconciliation around three clear, opt-in regimes while keeping all 62 shipped configurations 100% `run_id` hash-stable:

| Mode | Configuration Trigger | `spatial_weight_alpha` | Operational & Leaderboard Behavior |
| :--- | :--- | :--- | :--- |
| **Mode 1: Spatial-Only** (`"spatial_only"`) | `hierarchy.levels` non-empty (or `similarity_hierarchy = true`), `reconcile_temporal = false` (default) | `1.0` | Reconciles strictly across the organizational/spatial tree ($\mathbf{S}_{\text{cs}}$: e.g., `store -> region -> __total__` or `ts_id -> similarity_cluster_id -> __total__`) at each timestamp without altering temporal trajectories. |
| **Mode 2: Temporal-Only (`THieF`)** (`"temporal_only"`) | `hierarchy.levels = []`, `hierarchy.reconcile_temporal = true` | `0.0` | Reconciles each flat-panel series across calendar aggregation buckets ($\mathbf{S}_{\text{temp}}$: e.g., `Daily -> Weekly -> Bi-Weekly -> 4-Week`) on both forward `predictions` and rolling-origin `oof` folds, recomputing cell `metrics` (`compute_metrics`) so `<model>_<method>` is scored on `v_model_leaderboard`. |
| **Mode 3: Joint Cross-Temporal** (`"cross_temporal"`) | (`hierarchy.levels` non-empty or `similarity_hierarchy = true`) + `cross_temporal = true` (or `"cross_temporal_mint"` in `reconciliation_methods`) | `"auto"` ($\alpha^* \in [0, 1]$) or explicit float | Auto-enables `reconcile_temporal = true` and `enabled = true`, appends `"cross_temporal_mint"`, keeps standard spatial methods (`<model>_mint_nonneg`, `<model>_mint_shrink`) as **pure spatial ($\mathbf{S}_{\text{cs}}$)** entries on the leaderboard alongside `<model>_cross_temporal_mint`, and refines via alternating projections to $< 10^{-6}$ discrepancy on both axes simultaneously. |

### Latent Similarity Hierarchy Discovery on Flat Panels (`hierarchy.similarity_hierarchy`)

Many real-world panels (IoT telemetry, SaaS usage metrics, financial tickers, uncataloged SKUs) lack explicit `region` or `category` grouping columns, yet exhibit strong latent dynamical clusters. Setting `hierarchy.similarity_hierarchy = true` (with optional `hierarchy.similarity_n_clusters` and `hierarchy.similarity_method`, default `"hybrid"`) when `hierarchy.levels = []` automatically clusters the bottom-level series via [`cluster_by_similarity_df()`](./api/similarity.md), synthesizes a `"similarity_cluster_id"` grouping column (`levels = [["similarity_cluster_id"]]`), and builds a 2-level hierarchy (`__total__ -> similarity_cluster_id -> leaf series`) that reconciles cleanly (`< 1e-6`) with `mint_shrink`, `mint_nonneg`, `cross_temporal_mint`, `sample_coherent_paths()`, and `allocate_target()`.

### Cross-Sectional & Cross-Temporal Methods (`reconciliation_methods`)

When `hierarchy.enabled = true`, `build_hierarchy(df, cfg)` aggregates bottom-level leaf series bottom-up into every level declared in `hierarchy.levels` plus the root `"__total__"` series, fits every selected model across all hierarchy nodes, and reconciles both forward horizon predictions (`forecast_predictions`) and rolling-origin backtest folds (`backtest_oof`) using $\tilde{\mathbf{y}}_t = \mathbf{S}\mathbf{G}\hat{\mathbf{y}}_t$:

| Method (`reconciliation_methods`) | Weighting / Projection Matrix $\mathbf{W}$ | Operational Characteristics |
| :--- | :--- | :--- |
| `"bottom_up"` | Sums bottom-level leaf forecasts ($\mathbf{G} = [\mathbf{0} \mid \mathbf{I}_{n_b}]$) | Zero upper-level influence; preserves noisy bottom-level dynamics. |
| `"top_down"` | Disaggregates `"__total__"` by historical average proportions | Best when top-level signal is smooth and leaf proportions are stationary. |
| `"middle_out"` | Anchors on `hierarchy.middle_level` (e.g., `["region"]`), sums up to root, and disaggregates down to leaves | Ideal when regional/category forecasts are most reliable. |
| `"ols"` | Ordinary Least Squares ($\mathbf{W} = \mathbf{I}_n$) | Unweighted orthogonal projection onto the coherent subspace. |
| `"wls_struct"` | Structural WLS ($\mathbf{W} = \text{diag}(\mathbf{S}\mathbf{1}_{n_b})$) | Weights each node by the number of bottom-level series it aggregates; requires no OOF residuals. |
| `"wls_var"` | Variance-weighted WLS ($\mathbf{W} = \text{diag}(\hat{\sigma}_1^2, \dots, \hat{\sigma}_n^2)$) | Weights each node by its inverse out-of-fold residual variance. |
| `"mint_shrink"` | Minimum Trace with Schäfer–Strimmer covariance shrinkage ($\hat{\mathbf{W}}_{\text{shrink}}$) | Exploits cross-series error correlations while guaranteeing positive-definiteness when $n > T$. |
| `"mint_nonneg"` | Active-set non-negative Minimum Trace QP ($\tilde{\mathbf{b}}_t \ge \mathbf{0}$) | Guarantees strictly non-negative bottom and upper-level forecasts while maintaining exact coherence. |
| `"cross_temporal_mint"` | Simultaneous cross-sectional ($\mathbf{S}_{\text{cs}}$) and temporal ($\mathbf{S}_{\text{temp}}$) MinT | Guarantees $< 10^{-6}$ discrepancy across both the hierarchy tree and all temporal aggregation buckets simultaneously while keeping pure spatial methods on the leaderboard for head-to-head OOF comparison. |

---

## 2. Temporal Hierarchies (`THieF`), `spatial_temporal_weight` & Coherent Sample Paths

Standard single-frequency forecasts often conflict across planning horizons: daily store replenishment models fail to sum to weekly distribution-center plans or monthly financial guidance. Following Athanasopoulos et al. (2017) (*Forecasting with Temporal Hierarchies*, `THieF`) and Di Fonzo & Girolimetto (2023), `scale-forecasting` reconciles forecasts across multiple temporal aggregation factors $k \in \lbrace k_1, \dots, k_M, 1 \rbrace$:

### Default Temporal Aggregation Ladders (`default_temporal_aggregations`)

When `hierarchy.reconcile_temporal = true` and `hierarchy.temporal_aggregations` is omitted (`null`), canonical factors are inferred automatically from `data.freq`:

| Base Frequency (`data.freq`) | Default Coarse Factors (`temporal_aggregations`) | Macro-Cycle $m = \text{lcm}(k)$ | Planning Levels Reconciled |
| :--- | :--- | :--- | :--- |
| Daily (`"D"` / `"B"`) | `[7, 14, 28]` | `28` days | Daily ($k=1$), Weekly ($k=7$), Bi-Weekly ($k=14$), and 4-Week ($k=28$) |
| Weekly (`"W"`) | `[4, 13, 52]` | `52` weeks | Weekly ($k=1$), 4-Week ($k=4$), Quarterly ($k=13$), and Annual ($k=52$) |
| Monthly (`"MS"` / `"ME"`) | `[3, 6, 12]` | `12` months | Monthly ($k=1$), Quarterly ($k=3$), Semi-Annual ($k=6$), and Annual ($k=12$) |
| Quarterly (`"QS"` / `"QE"`) | `[2, 4]` | `4` quarters | Quarterly ($k=1$), Semi-Annual ($k=2$), and Annual ($k=4$) |
| Hourly (`"h"`) | `[6, 12, 24]` | `24` hours | Hourly ($k=1$), 6-Hour ($k=6$), 12-Hour ($k=12$), and Daily ($k=24$) |

### How the Temporal Hierarchy ($\mathbf{S}_{\text{temp}}$) & Coarse Forecasts Are Built
1. **Temporal Summing Matrix (`build_temporal_summing_matrix`):** For macro-period $m = \text{lcm}(k_1, \dots, k_M)$, each aggregation factor $k$ contributes $m / k$ rows via the Kronecker product $\mathbf{S}_{\text{temp}}^{(k)} = \mathbf{I}_{m/k} \otimes \mathbf{1}_k^\top$, stacked from coarsest $k_{\max}$ down to finest $k=1$ into $\mathbf{S}_{\text{temp}} \in \lbrace 0, 1 \rbrace^{n_{\text{temp}} \times m}$ (and extended across multi-cycle horizons $H$ via $\mathbf{I}_P \otimes \mathbf{S}_{\text{temp}}$ where $P = \lceil H / m \rceil$).
2. **Coarse-Level Forecasts & Residuals (`reconcile_series_temporal`):** At base frequency ($k=1$), the engine uses your fitted model's base forecast $\hat{\mathbf{y}}^{(1)}$ and out-of-fold (`backtest_oof`) or in-sample residuals. At each coarse factor $k > 1$, `aggregate_temporal_series` sums historical observations into non-overlapping blocks of size $k$ aligned to the forecast cutoff date and fits a fast deseasonalized damped-trend (Theta / Holt-Winters) smoother (`_fast_coarse_forecast_and_residuals`) to produce coarse forecasts $\hat{\mathbf{y}}^{(k)}$ and coarse residuals $\mathbf{e}^{(k)}$ (or accepts caller-supplied `coarse_forecasts`).

### Temporal Projection Methods (`reconcile_temporal_matrix`)

Given the stacked multi-frequency forecast vector $\hat{\mathbf{y}}_{\text{cycle}}$ ordered from $k_{\max}$ down to $k=1$, `reconcile_temporal_matrix` computes the coherent temporal projection $\tilde{\mathbf{y}} = \mathbf{S}_{\text{temp}} \mathbf{G}_{\text{temp}} \hat{\mathbf{y}}_{\text{cycle}}$ across all 9 methods in `VALID_TEMPORAL_METHODS`:

| Temporal Method | Mathematical Formulation ($\mathbf{G}_{\text{temp}}$ / $\mathbf{W}_{\text{temp}}$) | How It Reconciles Across Time |
| :--- | :--- | :--- |
| **`"bottom_up"`** | $\mathbf{G}_{\text{temp}} = [\mathbf{0}_{m \times n_{\text{agg}}},\ \mathbf{I}_m]$ | Keeps base-frequency ($k=1$) forecasts unchanged and sums them upward into weekly/monthly/quarterly buckets ($\tilde{\mathbf{y}} = \mathbf{S}_{\text{temp}} \hat{\mathbf{y}}^{(1)}$). |
| **`"top_down"`** | $\mathbf{G}_{\text{temp}}[:, 0] = \mathbf{p}$, where $p_j = \frac{1}{C}\sum_{c=1}^{C} \frac{y_{c,j}^{(1)}}{\sum_{t=1}^m y_{c,t}^{(1)}}$ | Disaggregates the coarsest macro-period forecast ($k = m$) down to base steps using historical within-cycle seasonal proportions $\mathbf{p}$. |
| **`"middle_out"`** | Anchors on median factor $k_{\text{mid}} \in \lbrace k > 1 \rbrace$ | Preserves the intermediate temporal level (e.g., weekly $k=7$ inside a 28-day cycle), summing upward to $k=28$ and disaggregating down to $k=1$. |
| **`"ols"`** | $\mathbf{W}_{\text{temp}} = \mathbf{I}_{n_{\text{temp}}}$, $\mathbf{G}_{\text{temp}} = (\mathbf{S}_{\text{temp}}^\top \mathbf{S}_{\text{temp}})^{-1} \mathbf{S}_{\text{temp}}^\top$ | Unweighted Ordinary Least Squares projection across all temporal frequencies. |
| **`"wls_struct"`** *(alias `"struc"`)* | $\mathbf{W}_{\text{temp}} = \text{diag}(\mathbf{S}_{\text{temp}} \mathbf{1}_m)$, $\mathbf{G}_{\text{temp}} = (\mathbf{S}_{\text{temp}}^\top \mathbf{W}_{\text{temp}}^{-1} \mathbf{S}_{\text{temp}})^{-1} \mathbf{S}_{\text{temp}}^\top \mathbf{W}_{\text{temp}}^{-1}$ | **Athanasopoulos et al. (2017) Structural Scaling (`THieF` standard WLS):** weights each temporal bucket inversely to the number of base steps $k$ summed into it ($w_i = k$). Requires **no residuals** and is unconditionally stable on short histories. |
| **`"wls_var"`** | $\mathbf{W}_{\text{temp}} = \text{diag}(\hat{\sigma}_1^2, \dots, \hat{\sigma}_{n_{\text{temp}}}^2)$ | **Variance Scaling WLS:** diagonal matrix of empirical out-of-fold / in-sample residual MSEs $\hat{\sigma}_k^2 = \frac{1}{N_k}\sum (e_t^{(k)})^2$ at each temporal bucket. |
| **`"mint_shrink"`** | $\mathbf{W}_{\text{temp}} = \hat{\lambda} \text{diag}(\hat{\boldsymbol{\Sigma}}) + (1 - \hat{\lambda}) \hat{\boldsymbol{\Sigma}}$ | **Minimum Trace (`MinT`) with Schäfer–Strimmer Shrinkage:** stacks cycle residuals across all temporal levels into $(n_{\text{cycles}} \times n_{\text{temp}})$ and shrinks cross-frequency residual covariances toward the diagonal via analytical intensity $\hat{\lambda}$. |
| **`"mint_nonneg"`** | $\min_{\tilde{\mathbf{b}} \ge \mathbf{0}} \|\mathbf{W}_{\text{temp}}^{-1/2}(\hat{\mathbf{y}} - \mathbf{S}_{\text{temp}} \tilde{\mathbf{b}})\|_2^2$ | **Non-Negative Minimum Trace (`scipy.optimize.nnls`):** whitens via $\mathbf{W}_{\text{temp}}^{-1/2}$ eigendecomposition and solves active-set non-negative least squares whenever any base step is negative, guaranteeing $\tilde{\mathbf{b}} \ge \mathbf{0}$ and exact temporal additivity. |
| **`"cross_temporal_mint"`** | Simultaneous Spatial ($\mathbf{S}_{\text{cs}}$) $\times$ Temporal ($\mathbf{S}_{\text{temp}}$) Projection (`reconcile_cross_temporal`) | **Di Fonzo & Girolimetto (2023) Cross-Temporal Reconciliation:** blends Temporal-then-Spatial ($\mathbf{B}_A$) and Spatial-then-Temporal ($\mathbf{B}_B$) leaf projections via learned OOF weight $\alpha^* \in [0.05, 0.95]$ and refines via alternating projections so both spatial and temporal discrepancies are $< 10^{-6}$. |

### Data-Driven Spatial–Temporal Blending (`hierarchy.spatial_temporal_weight`)
- When `hierarchy.spatial_temporal_weight` is a float $\alpha \in [0, 1]$, `reconcile_cross_temporal()` blends the spatial-last projection ($\tilde{\mathbf{B}}_{\text{temp}\to\text{cs}}$) and the temporal-last projection ($\tilde{\mathbf{B}}_{\text{cs}\to\text{temp}}$) as $\tilde{\mathbf{B}}(\alpha) = \alpha \tilde{\mathbf{B}}_{\text{temp}\to\text{cs}} + (1 - \alpha) \tilde{\mathbf{B}}_{\text{cs}\to\text{temp}}$ and refines via alternating projections onto $\mathbf{S}_{\text{cs}}$ and $\mathbf{S}_{\text{temp}}$ (plus $\mathbb{R}_+$ when `nonneg=True`).
- When `hierarchy.spatial_temporal_weight = "auto"` (default), $\alpha^* \in [0, 1]$ is learned from out-of-fold (`oof_df`) or in-sample residual variances across the spatial vs. temporal projections ($\alpha^* = \frac{\sigma_{\text{temp}}^{-2}}{\sigma_{\text{cs}}^{-2} + \sigma_{\text{temp}}^{-2}}$, defaulting to `0.5` when no residuals are provided).
- Every reconciled DataFrame attaches `out_df.attrs["cross_temporal_diagnostics"]` with keys `mode` (`"spatial_only"` | `"temporal_only"` | `"cross_temporal"`), `spatial_weight_alpha`, `cs_first_discrepancy`, `temp_first_discrepancy`, `final_cs_discrepancy`, and `final_temp_discrepancy`.

### Coherent Cross-Temporal Sample-Path Simulation (`sample_coherent_paths`)
- `sf.sample_coherent_paths(predictions_df, spec=None, *, n_paths=200, seed=42, method="mint_shrink", oof_df=None, cfg=None, nonneg=False)` (also exposed on `Forecaster.sample_coherent_paths()`, `PlaygroundRun.sample_coherent_paths()`, and `BakeOff.sample_coherent_paths()`) draws $N$ joint Gaussian/residual trajectories across the horizon and projects every single path onto the coherent cross-sectional and/or temporal subspace, returning a tidy DataFrame (`path_id`, `ts_id`, `forecast_date`, `yhat_sample`) where every `path_id` satisfies $< 10^{-6}$ coherence.

### Edge-Case Guarantees
- **Cutoff-Aligned Non-Overlapping Aggregation:** When historical length $T$ is not a multiple of $k$, the oldest $T \bmod k$ observations at the start of the series are trimmed so every coarse bucket ends on the **exact same forecast cutoff timestamp**.
- **Non-Divisible Horizons ($H \bmod m \ne 0$):** When `data.horizon` is not an exact multiple of the macro-period $m$, `build_temporal_hierarchy` pads the internal forecast to $P \cdot m = \lceil H / m \rceil \cdot m$, projects onto the coherent temporal subspace, and slices back to the requested $H$ steps while preserving exact coherence on all complete coarse buckets ($\lfloor H / k \rfloor$).
- **Short Series ($T < 2 \cdot m_{\max}$):** Automatically drops any coarse factor $k$ with fewer than 2 complete historical blocks (`history_len < 2 * k`).

---

## 3. Top-Down Goal-Seek & Smooth Multi-Period Target Allocation (`allocate_target`)

When leadership sets a top-down target (e.g., *"Increase Q4 Total Revenue to `125,000` while locking Region East at `45,000` and keeping every SKU within $[0, 500]$ units"*), proportional scaling violates node locks, breaches capacity bounds, or ignores forecast uncertainty.

`Forecaster.allocate_target` (and standalone `scale_forecasting.allocate_target`) solves the precision-weighted convex Quadratic Program with optional multi-period temporal smoothness regularization (`smoothness_weight`):

$$\min_{\mathbf{b} \in \mathbb{R}^{n_b \cdot H}} \frac{1}{2} (\mathbf{b} - \hat{\mathbf{b}})^\top \mathbf{W}^{-1} (\mathbf{b} - \hat{\mathbf{b}}) + \frac{\lambda_{\text{smooth}}}{2} \sum_{i \in \text{bottom}} \sum_{t=2}^{H} \bigl((b_{i,t} - b_{i,t-1}) - (\hat{b}_{i,t} - \hat{b}_{i,t-1})\bigr)^2 \quad \text{s.t.} \quad \mathbf{C}\mathbf{b} = \mathbf{d}, \quad \boldsymbol{\ell} \le \mathbf{b} \le \mathbf{u}$$

where $\mathbf{W}$ weights adjustments by each leaf series' historical forecast uncertainty (`method="mint_shrink"`, `"wls_var"`, `"wls_struct"`, `"ols"`, or `"proportional"`), and `smoothness_weight` ($\lambda_{\text{smooth}} \ge 0$) penalizes abrupt period-over-period step-cliffs when partial-horizon locks or capacity bounds are active:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `target_total` / `targets` | `float \| dict \| Sequence \| DataFrame` | Top-down target total across the horizon (scalar), per-node targets (`{"__total__": 12000.0, "region=East": 4500.0}`), or per-date targets. |
| `target_metric` | `"auto"` \| `"units"` \| `"revenue"` | When `"revenue"` (or when `price_col` / `prices` is supplied with `target_metric="revenue"`), enforces the equality constraint in currency units ($\sum_{i,t} p_{i,t} b_{i,t} = \text{Target}$). |
| `locked_nodes` / `locks` | `list[str] \| dict[str, float]` | Hierarchy nodes or leaf `ts_id`s held fixed at their baseline forecast (`list[str]`) or pinned to an explicit value (`dict[str, float]`). |
| `lower_bounds` / `upper_bounds` | `dict[str, float] \| float \| DataFrame` | Per-series, per-date, or global box bounds $[\ell_{i,t}, u_{i,t}]$ (combined with `floor` and `cap`). |
| `smoothness_weight` | `float` (`0.0`) | Non-negative $\ell_2$ first-difference roughness penalty ($\lambda_{\text{smooth}} \ge 0$) across $H \ge 2$ horizon steps so multi-period allocations transition smoothly without step-cliffs while satisfying all equality targets and box bounds to $< 10^{-6}$. |
| `price_col` / `unit_cost` | `str \| dict[str, float] \| float` | Computes `baseline_revenue`, `allocated_revenue`, `revenue_delta`, `baseline_gross_margin`, `allocated_gross_margin`, and `gross_margin_delta`. |
| `include_inventory_impact` | `bool` | Propagates allocated demand and uncertainty shifts into `baseline_safety_stock`, `allocated_safety_stock`, `delta_safety_stock`, `baseline_reorder_point`, `allocated_reorder_point`, and `delta_reorder_point`. |

If the requested target is mathematically unreachable given `locked_nodes` and `[lower_bounds, upper_bounds]`, `allocate_target` raises `GoalSeekInfeasibleError` naming the exact achievable range $[T_{\min}, T_{\max}]$.

---

## 4. Declarative `RunConfig` & Python SDK Examples

```json
{
  "run_name": "cross_temporal_hierarchy_plan",
  "python_runtime": "gce",
  "data": {
    "source_table": "source_series_covariates_iceberg",
    "freq": "D",
    "horizon": 28,
    "series_limit": 40
  },
  "models": ["theta", "holtwinters", "lightgbm"],
  "backtest": {
    "enabled": true,
    "horizon": 28,
    "n_folds": 2,
    "step": 28,
    "decision_metric": "rmsse"
  },
  "hierarchy": {
    "enabled": true,
    "levels": [
      ["region"],
      ["region", "category"]
    ],
    "reconciliation_methods": [
      "bottom_up",
      "wls_struct",
      "mint_shrink",
      "cross_temporal_mint"
    ],
    "reconcile_temporal": true,
    "cross_temporal": true,
    "spatial_temporal_weight": "auto",
    "temporal_aggregations": [7, 14, 28]
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

# 1. Load a hierarchical benchmark panel and fit with cross-temporal reconciliation
bench = sf.load_benchmark("hierarchical_supply_tree")
f = sf.Forecaster.auto(
    data=bench.df,
    target_col="y",
    horizon=28,
    hierarchy_levels=[["region"], ["region", "category"]],
)
print(f.assess())  # 7-Question Practitioner Assessment (Questions 3 & 5 report causal & 3-mode coherence)

# 2. Reconcile across temporal factors (Daily -> Weekly -> Bi-Weekly -> 4-Week) & cross-section simultaneously
rec_df, coarse_aggs = f.temporal_reconcile_df(
    method="cross_temporal_mint",
    temporal_aggregations=[7, 14, 28],
    return_aggregations=True,
)
print("Cross-temporal diagnostics:", rec_df.attrs["cross_temporal_diagnostics"])

# 3. Draw 50 cross-sectionally and temporally coherent sample paths (< 1e-6 discrepancy per path_id)
paths_df = f.sample_coherent_paths(forecasts_df=rec_df, n_paths=50, seed=42)

# 4. Allocate a +15% executive top-down target with smooth multi-period trajectory control (smoothness_weight=10.0)
root_rows = rec_df[rec_df["ts_id"] == "__total__"]
baseline_total = float(root_rows["yhat"].sum())
goal_df = f.allocate_target(
    target_total=baseline_total * 1.15,
    forecasts_df=rec_df,
    floor=0.0,
    smoothness_weight=10.0,
    prices=25.0,
    unit_cost=14.0,
    include_inventory_impact=True,
)
print(sf.summarize_goal_seek(goal_df))
# 5. Lifecycle Markdown Clearance & Vendor Contract Optimization on Reconciled Leaves
#    (Automatically filters out __total__ / parent nodes and builds a coherent bottom-up hierarchy_rollup_df)
clearance_res = f.optimize_clearance(
    forecasts_df=rec_df,
    initial_inventory=180.0,
    expiry_step=14,
    reconciliation_method="cross_temporal_mint",
)
print(clearance_res.hierarchy_rollup_df)
```

See also:
- [Notebook 06: Hierarchical Reconciliation, THieF & Top-Down Goal-Seek](./notebooks/06_hierarchical_reconciliation.ipynb)
- [Forecaster Workbench Guide](./forecaster_workbench.md)
- [API Reference: `reconciliation`](./api/reconciliation.md), [`temporal_reconciliation`](./api/temporal_reconciliation.md), [`goal_seek`](./api/goal_seek.md), [`clearance_optimizer`](./api/clearance_optimizer.md)
