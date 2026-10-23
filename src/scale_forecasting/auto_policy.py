"""Glass-Box Auto-Forecaster & `autofocus` Decision Policy (Evolutionary Policy Module).

Contains the pure-Python decision rules that map a :class:`PanelAutoProfile`, an Observational
Astronomy Exposure Tier (`"first_light"`, `"wide_field"`, `"deep_sky"`, `"deep_field"`), and
post-pass econometric diagnostics into concrete :class:`~scale_forecasting.config.RunConfig`
specifications, multi-branch Tournament DAG hypotheses, and closed-loop `autofocus` mutations.

Why a standalone pure-Python policy module?
-------------------------------------------
1. **Zero Runtime LLM / API Dependency:** Production callers of ``Forecaster.build_config()`` and
   ``Forecaster.auto()`` execute deterministic, auditable Python math in milliseconds with zero
   external network calls.
2. **Build-Time Evolutionary Calibration (AlphaEvolve Mode A):** The core policy functions are
   wrapped in evolutionary block markers so the in-repo benchmark evaluator (`benchmarks.py`) and
   Google Cloud AlphaEvolve (`discoveryengine` v1alpha with `gemini-3.8-flash` + `gemini-4-argon`)
   can evolve and verify policy improvements against the 12 canonical Data-Generating Process
   (DGP) benchmark panels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .monotonicity import infer_monotonic_constraints

__all__ = [
    "EXPOSURE_TIERS",
    "SCALE_REGIMES",
    "DataProfile",
    "PanelAutoProfile",
    "classify_scale_regime",
    "estimate_continue_confidence",
    "evolved_policy_thresholds",
    "propose_autofocus_dag_branches",
    "propose_autofocus_pass",
    "propose_autofocus_synthesis_pass",
    "select_default_baselines",
    "select_hierarchy_reconciliation_methods",
    "select_tier_models_and_features",
]

SCALE_REGIMES: dict[str, dict[str, Any]] = {
    "micro": {
        "max_series": 14,
        "default_cloud_workers": 1,
        "title": "Micro Panel (1–14 series)",
        "strategy": (
            "Per-series local statistical, tree, and zero-shot foundation models lead; "
            "trained @global neural models deferred unless cross-series correlation is high."
        ),
    },
    "small": {
        "max_series": 199,
        "default_cloud_workers": 1,
        "title": "Small Panel (15–199 series)",
        "strategy": (
            "Hybrid local + @global crossover: runs per-series and pooled models side-by-side "
            "to measure empirical pooling FVA."
        ),
    },
    "medium": {
        "max_series": 2499,
        "default_cloud_workers": 2,
        "title": "Medium Panel (200–2,499 series)",
        "strategy": (
            "Global trees, neural nets, and foundation models lead; prunes slow O(T^2) "
            "per-series iterative statistical models on local CPU."
        ),
    },
    "large": {
        "max_series": 24999,
        "default_cloud_workers": 4,
        "title": "Large Fleet (2,500–24,999 series)",
        "strategy": (
            "Global neural/foundation + fast linear-time statistical models with multi-worker "
            "cloud sharding and top-K ensemble parsimony."
        ),
    },
    "hyperscale": {
        "max_series": 10_000_000,
        "default_cloud_workers": 8,
        "title": "Hyperscale Fleet (25,000+ series)",
        "strategy": (
            "Hyperscale distributed portfolio (Spark Serverless / Ray / Vertex GPU / BigQuery) "
            "with 1–2 fold CV and top-3 parsimony ensembling."
        ),
    },
}


def classify_scale_regime(n_series: int) -> str:
    """Classify panel series count `n_series` into one of the 5 canonical scale regimes."""
    if n_series < 15:
        return "micro"
    if n_series < 200:
        return "small"
    if n_series < 2500:
        return "medium"
    if n_series < 25000:
        return "large"
    return "hyperscale"


EXPOSURE_TIERS: dict[str, dict[str, Any]] = {
    "first_light": {
        "title": "First Light (~20s local / ~55s BQ / ~6m cloud)",
        "description": (
            "Rapid baseline exposure: seasonal naive + fast statistical + gradient boosted trees "
            "with a 1-fold backtest to verify data plumbing and establish the FVA baseline."
        ),
        "default_max_passes": 1,
        "target_local_seconds": 25.0,
        "target_bq_minutes": 1.0,
        "target_cloud_minutes": 6.5,
    },
    "wide_field": {
        "title": "Wide Field (~2m local / ~2m BQ / ~7-16m cloud, Default)",
        "description": (
            "Balanced multi-family survey across Statistical, ML Trees, and Zero-Shot Foundation "
            "models with 2-fold CV, covariate FVA baselines, and champion selection."
        ),
        "default_max_passes": 2,
        "target_local_seconds": 120.0,
        "target_bq_minutes": 2.2,
        "target_cloud_minutes": 9.0,
    },
    "deep_sky": {
        "title": "Deep Sky (~6m local / ~4.5m BQ / ~12-22m cloud)",
        "description": (
            "Deep cross-series global + local survey adding CatBoost, TiDE/TSMixer neural "
            "architectures, 3-fold CV, fleetwide HPO, and multi-branch Tournament DAG autofocus."
        ),
        "default_max_passes": 3,
        "target_local_seconds": 360.0,
        "target_bq_minutes": 4.5,
        "target_cloud_minutes": 16.0,
    },
    "deep_field": {
        "title": "Deep Field (~15m local / ~8m BQ / ~20-35m cloud)",
        "description": (
            "Exhaustive observatory exposure across all eligible families (Statistical, ML, "
            "Deep Learning, Foundation, and BigQuery ML SQL), 3-fold CV, Optuna HPO, and "
            "learned stacking ensembles."
        ),
        "default_max_passes": 4,
        "target_local_seconds": 900.0,
        "target_bq_minutes": 8.0,
        "target_cloud_minutes": 28.0,
    },
}


@dataclass(frozen=True)
class PanelAutoProfile:
    """Single-pass statistical and structural profile of a time-series panel (pure)."""

    source_table: str
    ts_id_col: str
    date_col: str
    target_col: str
    freq: str
    horizon: int
    n_series: int
    n_rows: int
    median_n_obs: int
    min_n_obs: int
    max_n_obs: int
    non_negative: bool
    integer_valued: bool
    zero_fraction: float
    median_adi: float
    median_cv2: float
    level_vol_corr: float
    end_to_start_ratio: float
    dominant_regime: str
    regime_shares: dict[str, float] = field(default_factory=dict)
    median_acf_season: float = 0.0
    structural_break_share: float = 0.0
    outlier_spike_share: float = 0.0
    future_covariates: tuple[str, ...] = ()
    past_covariates: tuple[str, ...] = ()
    static_covariates: tuple[str, ...] = ()
    hierarchy_id_cols: tuple[str, ...] = ()
    censored_col: str | None = None
    leading_zero_share: float = 0.0
    missing_date_gap_share: float = 0.0
    hierarchy_node_count: int = 0
    cross_series_corr: float = 0.0
    monotonic_constraints: dict[str, int] = field(default_factory=dict)
    is_compositional_simplex: bool = False
    simplex_total: float = 0.0
    cold_start_donor_similarity: float = 0.0
    residual_drift_ks_stat: float = 0.0

    @classmethod
    def from_df(cls, data: Any, **kwargs: Any) -> PanelAutoProfile:
        """Profile an in-memory DataFrame or BigQuery table via :func:`inspect_panel_for_auto`."""
        from .auto import inspect_panel_for_auto

        return inspect_panel_for_auto(data, **kwargs)

    @property
    def scale_regime(self) -> str:
        """Canonical series-count scale regime (`micro` .. `hyperscale`)."""
        return classify_scale_regime(self.n_series)

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a plain JSON-compatible dictionary."""
        raw = asdict(self)
        for k, v in list(raw.items()):
            if isinstance(v, tuple):
                raw[k] = list(v)
        raw["scale_regime"] = self.scale_regime
        return raw

    def to_df(self) -> Any:
        """Return a 7-row Step 0 Data Profile & Readiness ``DataFrame`` for human review."""
        import pandas as pd

        scale_info = SCALE_REGIMES.get(self.scale_regime, SCALE_REGIMES["small"])
        cold_share = float(self.regime_shares.get("cold_start", 0.0))
        cov_list = (
            list(self.future_covariates) + list(self.past_covariates) + list(self.static_covariates)
        )
        rows = [
            {
                "dimension": "scale_and_shape",
                "signal": "n_series · history · scale_regime · cross_series_corr",
                "measured_value": (
                    f"{self.n_series} series ({self.scale_regime}) | "
                    f"freq='{self.freq}' | median_T={self.median_n_obs} "
                    f"(min={self.min_n_obs}, max={self.max_n_obs}) | H={self.horizon} | "
                    f"corr={self.cross_series_corr:.2f}"
                ),
                "implication": str(scale_info["strategy"]),
            },
            {
                "dimension": "demand_regime",
                "signal": "dominant_regime · zero_fraction · ADI · CV² · cold_start",
                "measured_value": (
                    f"regime='{self.dominant_regime}' | zeros={self.zero_fraction:.1%} | "
                    f"ADI={self.median_adi:.2f} | CV²={self.median_cv2:.2f} | "
                    f"cold_start={cold_share:.1%} | "
                    f"donor_sim={self.cold_start_donor_similarity:.2f}"
                ),
                "implication": (
                    "Intermittent/lumpy zeros trigger MASE scoring and Croston/TSB specialists "
                    "with regime routing."
                    if (
                        self.zero_fraction >= 0.25
                        or self.dominant_regime in ("intermittent", "lumpy")
                    )
                    else (
                        "Mixed mature and short cold-start series favor cross-series pooling and "
                        "SBC regime routing."
                        if cold_share > 0.05
                        else (
                            "Smooth/continuous series support WAPE scoring and standard "
                            "lag/seasonal models."
                        )
                    )
                ),
            },
            {
                "dimension": "seasonality_and_scaling",
                "signal": "seasonal_acf · level_vol_corr · end_to_start_ratio",
                "measured_value": (
                    f"acf_season={self.median_acf_season:.2f} | "
                    f"level_vol_corr={self.level_vol_corr:.2f} | "
                    f"growth_ratio={self.end_to_start_ratio:.2f}"
                ),
                "implication": (
                    "Amplitude scales with level (multiplicative); log1p variance stabilization "
                    "recommended."
                    if (self.level_vol_corr >= 0.55 and self.end_to_start_ratio >= 1.35)
                    else (
                        "Additive variance regime; standard linear target scaling is "
                        "well-conditioned."
                    )
                ),
            },
            {
                "dimension": "breaks_and_outliers",
                "signal": "structural_break_share · outlier_spike_share",
                "measured_value": (
                    f"breaks={self.structural_break_share:.1%} | "
                    f"spikes={self.outlier_spike_share:.1%} | "
                    f"ks_drift={self.residual_drift_ks_stat:.2f}"
                ),
                "implication": (
                    "Elevated structural shifts or >6-sigma spikes trigger level_shift indicators "
                    "and fold-safe stl_resid outlier scrubbing."
                    if (self.structural_break_share >= 0.15 or self.outlier_spike_share >= 0.10)
                    else (
                        "Clean historical trajectory; no aggressive outlier clipping required "
                        "in Pass 1."
                    )
                ),
            },
            {
                "dimension": "hygiene_and_calendar",
                "signal": "leading_zeros · missing_gaps · censored_col",
                "measured_value": (
                    f"leading_zeros={self.leading_zero_share:.1%} | "
                    f"date_gaps={self.missing_date_gap_share:.1%} | "
                    f"censored_col={self.censored_col or 'none'}"
                ),
                "implication": (
                    "Pre-launch leading zeros, calendar gaps, or stockout censoring detected; "
                    "enables automatic hygiene repair / latent demand reconstruction."
                    if (
                        self.leading_zero_share >= 0.05
                        or self.missing_date_gap_share > 0.0
                        or self.censored_col
                    )
                    else "Contiguous date grid with no pre-launch zero padding or stockout flags."
                ),
            },
            {
                "dimension": "covariates",
                "signal": "future · past · static covariates",
                "measured_value": (
                    f"{len(cov_list)} column(s): {', '.join(cov_list)}"
                    if cov_list
                    else "0 extra feature columns (pure univariate panel)"
                ),
                "implication": (
                    "Benchmarks covariate models (@cov_all) against paired @univariate baselines "
                    "and prunes low-attribution decoy columns in autofocus."
                    if cov_list
                    else "Univariate panel; prioritizes seasonal, state-space, lag-tree, and "
                    "univariate foundation models."
                ),
            },
            {
                "dimension": "hierarchy_topology",
                "signal": "hierarchy_id_cols · node_count · non_negative",
                "measured_value": (
                    f"cols={list(self.hierarchy_id_cols)} | "
                    f"nodes={max(self.hierarchy_node_count, self.n_series)} | "
                    f"non_negative={self.non_negative}"
                    if self.hierarchy_id_cols
                    else f"Flat panel (no hierarchy columns) | non_negative={self.non_negative}"
                ),
                "implication": (
                    "Hierarchical tree detected; applies scale-aware coherent reconciliation "
                    f"({', '.join(select_hierarchy_reconciliation_methods(self))}) with 0.0 "
                    "aggregation residual."
                    if self.hierarchy_id_cols
                    else (
                        "Independent series; enforces output.floor=0.0 when history "
                        "is non-negative."
                    )
                ),
            },
        ]
        if self.is_compositional_simplex:
            rows.append(
                {
                    "dimension": "compositional_simplex",
                    "signal": "compositional_simplex",
                    "measured_value": (
                        f"simplex_total={self.simplex_total:.2f} | "
                        f"n_parts={self.n_series} | non_negative={self.non_negative}"
                    ),
                    "implication": (
                        "Constant-sum compositional simplex detected; routes Aitchison "
                        "log-ratio transforms (@transform_ilr / @transform_clr) with "
                        f"features.simplex_sum={self.simplex_total:.2f}."
                    ),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["dimension", "signal", "measured_value", "implication"],
        )

    to_dataframe = to_df

    def explain(self) -> str:
        """Return a formatted text summary of the Step 0 Data Profile & Readiness table."""
        df = self.to_df()
        header = (
            f"Step 0 Data Profile : {self.source_table} | {self.n_series} series "
            f"({self.scale_regime}) | freq='{self.freq}' | H={self.horizon}"
        )
        return f"{header}\n\n{df.to_string(index=False)}"


DataProfile = PanelAutoProfile


def select_hierarchy_reconciliation_methods(
    profile: PanelAutoProfile,
    *,
    tier: str = "wide_field",
) -> list[str]:
    """Select hierarchy reconciliation methods matched to node count ($N_{\\text{nodes}}$ vs. $T$)
    and leaf sparsity (`zero_fraction`).
    """
    n_nodes = max(profile.hierarchy_node_count, profile.n_series)
    t_obs = max(1, profile.median_n_obs)
    is_sparse_leaves = profile.zero_fraction >= 0.30 or profile.dominant_regime in (
        "intermittent",
        "lumpy",
    )

    if n_nodes > 500:
        # Large enterprise tree: O(N) diagonal & structural methods avoid O(N^3) dense covariance
        methods = ["wls_var", "wls_struct", "bottom_up"]
    elif n_nodes > 75 or n_nodes > t_obs:
        # Medium tree (N > 75 or N > T): shrinkage / diagonal WLS + non-negative projection
        primary = "mint_nonneg" if profile.non_negative else "mint_shrink"
        methods = [primary, "wls_var", "wls_struct", "bottom_up"]
    else:
        # Compact tree (N <= 75 and T >= N)
        primary = "mint_nonneg" if profile.non_negative else "mint_shrink"
        if tier in ("deep_sky", "deep_field"):
            methods = [primary, "mint_shrink", "wls_var", "bottom_up"]
        else:
            methods = [primary, "bottom_up"]

    if is_sparse_leaves and "top_down" not in methods:
        methods.append("top_down")
    return list(dict.fromkeys(methods))


def select_default_baselines(
    profile: PanelAutoProfile,
    *,
    where: str = "auto",
    available_models: Sequence[str] | None = None,
    candidate_models: Sequence[str] | None = None,
    selected_models: Sequence[str] | None = None,
    user_baselines: str | Sequence[str] | None = "auto",
) -> list[dict[str, str]]:
    """Build the 4-Anchor Opinionated Baseline Ladder (`baselines='auto'`) or resolve explicit
    user-pinned baselines.

    Returns a list of baseline descriptor dicts with keys `model_type` (alias `model`),
    `baseline_role` (alias `role`), and `description`:
    1. `naive_anchor`: `naive_seasonal` (plus `naive_mean` on intermittent panels)
    2. `plain_univariate_stat`: `theta` / `arima_plus@univariate` / `croston@classic`
    3. `zero_shot_foundation`: `chronos_bolt` / `timesfm_torch@univariate` / `chronos2@univariate`
       / `timesfm@univariate` (when available in the execution environment)
    4. `same_model_univariate`: `<lead_model>@univariate` when exogenous covariates are present
    """

    def _entry(model_type: str, role: str, description: str) -> dict[str, str]:
        return {
            "model_type": model_type,
            "baseline_role": role,
            "model": model_type,
            "role": role,
            "description": description,
        }

    if user_baselines is not None and user_baselines != "auto":
        raw_list = [user_baselines] if isinstance(user_baselines, str) else list(user_baselines)
        entries: list[dict[str, str]] = [
            _entry(
                str(b),
                "user_pinned_baseline",
                f"User-selected baseline anchor ({b}).",
            )
            for b in dict.fromkeys(str(x) for x in raw_list if str(x))
        ]
        if not any(e["model_type"] == "naive_seasonal" for e in entries):
            entries.insert(
                0,
                _entry(
                    "naive_seasonal",
                    "naive_anchor",
                    "Zero-skill seasonal anchor (repeats last seasonal cycle).",
                ),
            )
        return entries

    avail = set(available_models or ())
    cands = list(selected_models if selected_models is not None else (candidate_models or ()))
    has_covs = bool(profile.future_covariates or profile.past_covariates)
    is_intermittent = profile.zero_fraction >= 0.20 or profile.dominant_regime in (
        "intermittent",
        "lumpy",
    )

    ladder: list[dict[str, str]] = [
        _entry(
            "naive_seasonal",
            "naive_anchor",
            "Zero-skill seasonal anchor (repeats last seasonal cycle).",
        )
    ]
    if is_intermittent and ("naive_mean" in avail or "naive_mean" in cands):
        ladder.append(
            _entry(
                "naive_mean",
                "naive_anchor",
                "Unbiased historical mean anchor for intermittent series.",
            )
        )

    # Slot 2: Plain Univariate Statistical Anchor
    if where == "bigquery":
        stat_m = "arima_plus@univariate" if has_covs else "arima_plus"
        ladder.append(
            _entry(
                stat_m,
                "plain_univariate_stat",
                "Plain univariate BigQuery ML ARIMA_PLUS using only target history y.",
            )
        )
    elif is_intermittent and ("croston" in avail or any(m.startswith("croston") for m in cands)):
        ladder.append(
            _entry(
                "croston@classic",
                "plain_univariate_stat",
                "Plain univariate Classic Croston intermittent baseline.",
            )
        )
    else:
        stat_m = "theta"
        ladder.append(
            _entry(
                stat_m,
                "plain_univariate_stat",
                "Plain univariate Theta exponential smoothing baseline using only y.",
            )
        )

    # Slot 3: Zero-Shot Foundation Model Baseline
    if where == "bigquery" and ("timesfm" in avail or any(m.startswith("timesfm") for m in cands)):
        fm_bq = "timesfm@univariate" if has_covs else "timesfm"
        ladder.append(
            _entry(
                fm_bq,
                "zero_shot_foundation",
                "Zero-shot univariate BigQuery AI.FORECAST (TimesFM) foundation baseline.",
            )
        )
    else:
        for fm_cand, fm_uni in (
            ("chronos_bolt", "chronos_bolt"),
            ("timesfm_torch", "timesfm_torch@univariate" if has_covs else "timesfm_torch"),
            ("chronos2", "chronos2@univariate" if has_covs else "chronos2"),
        ):
            if fm_cand in avail or any(m.split("@", 1)[0] == fm_cand for m in cands):
                ladder.append(
                    _entry(
                        fm_uni,
                        "zero_shot_foundation",
                        f"Zero-shot pretrained foundation baseline ({fm_uni}).",
                    )
                )
                break

    # Slot 4: Same-Model Plain Univariate Twin when covariates exist
    if has_covs:
        for cov_base in (
            "lightgbm",
            "regression_lags",
            "xgboost",
            "chronos2",
            "timesfm_torch",
            "arima_plus",
            "timesfm",
        ):
            if any(m.split("@", 1)[0] == cov_base for m in cands) or cov_base in avail:
                uni_twin = f"{cov_base}@univariate"
                if not any(b["model_type"] == uni_twin for b in ladder):
                    ladder.append(
                        _entry(
                            uni_twin,
                            "same_model_univariate",
                            (
                                f"Plain univariate twin of {cov_base} (strips all covariates to "
                                "measure exact exogenous covariate lift)."
                            ),
                        )
                    )
                    break

    return ladder


# EVOLVE-BLOCK-START
def evolved_policy_thresholds() -> dict[str, Any]:
    """Return calibrated decision thresholds and action weights for `auto_policy`.

    Evolved via Mode A AlphaEvolve (`gemini-3.8-flash` + `gemini-4-argon`) across the
    20 canonical synthetic DGP benchmarks and multi-scale empirical cell bank.
    """
    return {
        "intermittent_share_min": 0.165,
        "log1p_level_vol_corr_min": 0.635,
        "log1p_growth_ratio_min": 1.336,
        "resid_hetero_min": 0.30,
        "structural_break_share_min": 0.251,
        "outlier_spike_share_min": 0.195,
        "direct_horizon_decay_ratio_min": 1.323,
        "resid_acf1_escalate_min": 0.309,
        "fva_escalate_min_pct": 3.077,
        "white_noise_growth_tol": 0.179,
        "white_noise_acf_season_max": 0.17,
        "require_white_noise_stop": True,
        "pass_decay_factor": 0.70,
        "first_light_include_regression_lags": True,
        "first_light_selection_rule": "best_single",
        "first_light_ensemble_strategies": ["inverse_error", "mean"],
        "tree_n_estimators_fast": 60,
        "action_weights": {
            "prune_decoy_covariates": (0.86, 6.6),
            "fallback_to_plain_univariate": (0.65, 7.4),
            "synthesize_exog_lags": (0.92, 5.0),
            "inject_monotone_guardrails": (0.88, 4.2),
            "inject_transform_log1p": (0.92, 10.4),
            "inject_transform_simplex": (0.87, 8.0),
            "inject_direct_horizon_models": (0.84, 5.7),
            "enable_level_shift": (0.92, 11.5),
            "enable_sliding_window_backtest": (0.67, 4.3),
            "enable_stl_resid_outlier_scrubbing": (0.96, 5.9),
            "inject_intermittent_specialists": (0.68, 5.3),
            "enable_cold_start_pooling": (0.94, 3.4),
            "enable_cold_start_transfer": (0.85, 5.2),
            "enable_cold_start_donor_transfer": (0.85, 5.2),
            "enable_adaptive_conformal": (0.82, 4.1),
            "enable_aci_conformal_tracking": (0.82, 4.1),
            "refine_hierarchical_coherence": (0.86, 10.5),
            "enable_temporal_reconciliation": (0.78, 4.8),
            "refine_probabilistic_quantiles": (0.66, 2.9),
            "refine_censored_demand_models": (0.96, 7.2),
            "escalate_bigquery_foundation": (0.66, 2.0),
            "synthesize_winning_dag_branches": (0.65, 3.7),
            "escalate_model_diversity": (0.63, 2.7),
        },
    }


# EVOLVE-BLOCK-END


def _frequency_matched_lags(
    profile: PanelAutoProfile,
    *,
    n_folds: int = 2,
) -> list[int] | None:
    r"""Return frequency-matched target lag depths for non-daily (or short) panels, or ``None``.

    Default `_lag_forecaster.LAGS = (1, 2, 3, 7, 14, 28)` is calibrated for daily series ($m=7$).
    On non-intermittent panels:
    - Hourly (`h`/`H`): `[1, 2, 3, 6, 12, 24, 48, 168]` on long panels ($\ge 300$ obs with $\le 2$
      covariates) to capture diurnal and weekly harmonics.
    - Monthly (`MS`/`ME`/`M`) with $\le 48$ training observations: `[1, 2, 3, 6, 12, 24]` captures
      semi-annual, annual, and 2-year harmonics without discarding 28 months of initial history.
    - Weekly (`W`) with $\le 40$ training observations: `[1, 2, 4, 8, 13]`.
    - Quarterly (`QS`/`QE`/`Q`): `[1, 2, 4, 8]` captures 1-year and 2-year quarterly cycles.
    """
    if profile.zero_fraction >= 0.20 or profile.dominant_regime in ("intermittent", "lumpy"):
        return None

    f_norm = str(profile.freq).strip().upper()
    min_train_obs = max(8, profile.min_n_obs - max(1, n_folds) * profile.horizon)
    max_lag = max(3, min_train_obs - profile.horizon - 8)

    if f_norm in ("H", "1H", "BH"):
        if profile.min_n_obs >= 336:
            cand = [1, 2, 3, 6, 12, 24, 48, 168]
        else:
            return None
    elif f_norm.startswith("M"):
        if profile.min_n_obs <= 64:
            cand = [1, 2, 3, 6, 12]
        elif min_train_obs <= 48:
            cand = [1, 2, 3, 6, 12, 24]
        else:
            return None
    elif f_norm.startswith("W"):
        if min_train_obs <= 40:
            cand = [1, 2, 4, 8, 13]
        else:
            return None
    elif f_norm.startswith("Q"):
        cand = [1, 2, 4, 8]
    else:
        if min_train_obs <= 35:
            cand = [1, 2, 3, 7, 14]
        else:
            return None

    filtered = [lag for lag in cand if lag <= max_lag]
    return filtered if filtered else [1, 2, 3]


def select_tier_models_and_features(
    profile: PanelAutoProfile,
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
    allowed_families: Sequence[str] | None = None,
    available_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compile initial Pass 1 model, feature, backtest, and ensemble choices from `profile`.

    Calibrated via Mode A AlphaEvolve (`gemini-3.8-flash` + `gemini-4-argon`) over the canonical
    DGP benchmarks to maximize Forecast Value Added (FVA), covariate signal recovery, interval
    calibration, and wall-clock efficiency across all 5 `n_series` scale regimes.
    """
    avail = set(
        available_models
        if available_models is not None
        else (
            "naive_seasonal",
            "naive_mean",
            "naive_drift",
            "naive_moving_average",
            "theta",
            "holtwinters",
            "stl_bagging",
            "croston",
            "fft",
            "regression_lags",
            "sarimax",
            "autoets",
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "chronos_bolt",
            "chronos2",
            "timesfm_torch",
            "tide",
            "tsmixer",
            "patchtst",
            "tft",
            "vertex_tide",
            "vertex_tft",
            "vertex_l2l",
            "vertex_seq2seq",
            "arima_plus",
            "timesfm",
        )
    )
    allowed_fams = (
        set(allowed_families)
        if allowed_families is not None
        else {"statistical", "ml", "deep_learning", "automl", "native"}
    )

    rationales: list[dict[str, str]] = []
    models: list[str] = []
    th = evolved_policy_thresholds()
    scale_regime = profile.scale_regime
    regime_meta = SCALE_REGIMES[scale_regime]
    rationales.append(
        {
            "category": "scale_regime",
            "parameter": "scale_regime",
            "value": f"{scale_regime} ({regime_meta['title']})",
            "evidence": f"n_series={profile.n_series}, n_rows={profile.n_rows}",
            "rationale": str(regime_meta["strategy"]),
        }
    )

    def _add_model(m: str, family: str, reason: str, evidence: str) -> None:
        base = m.split("@", 1)[0]
        if family in allowed_fams and base in avail and m not in models:
            models.append(m)
            rationales.append(
                {
                    "category": "models",
                    "parameter": m,
                    "value": f"include ({family})",
                    "evidence": evidence,
                    "rationale": reason,
                }
            )

    # 1. Always anchor with a fast seasonal benchmark for FVA & Diebold-Mariano testing
    _add_model(
        "naive_seasonal",
        "statistical",
        "Establishes the zero-skill seasonal benchmark for Forecast Value Added and DM tests.",
        f"freq={profile.freq}, median_acf={profile.median_acf_season:.2f}",
    )

    is_intermittent_heavy = (
        profile.dominant_regime in ("intermittent", "lumpy")
        or profile.zero_fraction >= 0.20
        or profile.median_adi >= 1.32
        or (
            profile.regime_shares.get("intermittent", 0.0) + profile.regime_shares.get("lumpy", 0.0)
        )
        >= float(th.get("intermittent_share_min", 0.25))
    )
    has_cold_start = (
        profile.dominant_regime == "cold_start"
        or profile.regime_shares.get("cold_start", 0.0) > 0.05
        or profile.min_n_obs < max(28, 2 * profile.horizon)
    )
    has_covs = bool(profile.future_covariates or profile.past_covariates)
    skip_slow_local_stat = (
        where == "local" and scale_regime in ("medium", "large", "hyperscale")
    ) or (scale_regime in ("large", "hyperscale"))
    allow_local_foundation = where != "local" or profile.n_series <= 200

    # 2. Tier-progressive & scale-aware model selection
    if where == "bigquery":
        _add_model(
            "arima_plus",
            "native",
            "In-warehouse BigQuery ML ARIMA_PLUS with automatic seasonality and holiday detection.",
            f"where='bigquery', fut_covs={len(profile.future_covariates)}",
        )
        if has_covs and tier != "first_light":
            _add_model(
                "arima_plus@univariate",
                "native",
                "Plain univariate ARIMA_PLUS twin to measure exact exogenous covariate lift.",
                f"fut_covs={len(profile.future_covariates)}",
            )
        if tier != "first_light":
            _add_model(
                "timesfm",
                "native",
                "In-warehouse BigQuery AI.FORECAST (TimesFM zero-shot foundation model).",
                f"tier='{tier}', where='bigquery'",
            )
            if has_covs and tier in ("deep_sky", "deep_field"):
                _add_model(
                    "timesfm@univariate",
                    "native",
                    "Zero-shot univariate TimesFM baseline to isolate multivariate lift.",
                    f"tier='{tier}', fut_covs={len(profile.future_covariates)}",
                )
    elif tier == "first_light":
        _add_model(
            "theta",
            "statistical",
            "Fast M3-winning exponential smoothing with drift (~0.14s/cell).",
            f"dominant_regime={profile.dominant_regime}",
        )
        if is_intermittent_heavy:
            _add_model(
                "croston@tsb",
                "statistical",
                "Teunter-Syntetos-Babai intermittent demand estimator handles zero-heavy series.",
                f"zero_fraction={profile.zero_fraction:.2%}, ADI={profile.median_adi:.2f}",
            )
        _add_model(
            "lightgbm",
            "ml",
            "Fast gradient boosted decision trees capturing lags and covariates.",
            f"n_series={profile.n_series}, covariates={len(profile.future_covariates)}",
        )
        if bool(th.get("first_light_include_regression_lags", False)) and (
            profile.future_covariates or abs(profile.end_to_start_ratio - 1.0) >= 0.25
        ):
            _add_model(
                "regression_lags",
                "ml",
                "Fast ridge lag regression for linear trend and exogenous elasticity.",
                f"fut_covs={len(profile.future_covariates)}, "
                f"growth={profile.end_to_start_ratio:.2f}",
            )
    elif tier == "wide_field":
        _add_model(
            "theta",
            "statistical",
            "Robust trend + seasonal statistical baseline.",
            f"median_n_obs={profile.median_n_obs}",
        )
        if scale_regime != "hyperscale":
            _add_model(
                "holtwinters",
                "statistical",
                "Captures damped trend and seasonal harmonics cleanly.",
                f"median_acf={profile.median_acf_season:.2f}",
            )
        _add_model(
            "fft",
            "statistical",
            "Fast Fourier harmonic extrapolation for periodic seasonal cycles.",
            f"dominant_regime={profile.dominant_regime}",
        )
        if is_intermittent_heavy:
            _add_model(
                "naive_mean",
                "statistical",
                "Unbiased level anchor for high-sparsity intermittent demand.",
                f"zero_fraction={profile.zero_fraction:.2%}",
            )
            _add_model(
                "croston@classic",
                "statistical",
                "Classic Croston separate size/interval exponential smoothing.",
                f"ADI={profile.median_adi:.2f}",
            )
            _add_model(
                "croston@sba",
                "statistical",
                "Syntetos-Boylan bias-corrected intermittent demand estimator.",
                f"ADI={profile.median_adi:.2f}, CV2={profile.median_cv2:.2f}",
            )
            _add_model(
                "croston@tsb",
                "statistical",
                "TSB probability-updating intermittent estimator for obsolescence/lumpy SKUs.",
                f"zero_fraction={profile.zero_fraction:.2%}",
            )
        _add_model(
            "lightgbm",
            "ml",
            "GBDT learner capturing nonlinear lag and exogenous patterns.",
            f"n_series={profile.n_series}, fut_covs={len(profile.future_covariates)}",
        )
        _add_model(
            "xgboost",
            "ml",
            "Complementary regularized gradient boosted trees for diverse ensemble blending.",
            f"n_series={profile.n_series}",
        )
        _add_model(
            "regression_lags",
            "ml",
            "Fast ridge lag regression providing linear extrapolation and attribution weights.",
            f"end_to_start_ratio={profile.end_to_start_ratio:.2f}",
        )
        if "chronos_bolt" in avail and allow_local_foundation:
            cb_spec = "chronos_bolt@global" if profile.n_series >= 15 else "chronos_bolt"
            _add_model(
                cb_spec,
                "deep_learning",
                "Zero-shot pretrained foundation baseline excels on cold-start and complex series.",
                f"has_cold_start={has_cold_start}, scale_regime={scale_regime}",
            )
    elif tier in ("deep_sky", "deep_field"):
        fast_stats = (
            ("theta", "Fast statistical baseline for trend + seasonality."),
            ("holtwinters", "Triple exponential smoothing for seasonal series."),
            ("fft", "Fast Fourier harmonic decomposition."),
        )
        for stat_m, desc in fast_stats:
            if scale_regime == "hyperscale" and stat_m == "holtwinters":
                continue
            _add_model(stat_m, "statistical", desc, f"tier={tier}")

        if not skip_slow_local_stat:
            for slow_m, desc in (
                ("autoets", "State-space Error-Trend-Seasonal automatic selection."),
                ("stl_bagging", "Bagged STL decomposition for variance reduction."),
            ):
                _add_model(slow_m, "statistical", desc, f"tier={tier}, scale_regime={scale_regime}")
        else:
            rationales.append(
                {
                    "category": "scale_guardrail",
                    "parameter": "autoets, stl_bagging",
                    "value": "pruned_for_scale",
                    "evidence": f"n_series={profile.n_series}, scale_regime='{scale_regime}'",
                    "rationale": (
                        "Skipped slow O(T^2) per-series state-space fits at medium/large scale in "
                        "favor of vectorized statistical and @global pooled learners."
                    ),
                }
            )
        if is_intermittent_heavy:
            _add_model("naive_mean", "statistical", "Level anchor for sparse demand.", "sparse")
            _add_model("croston@classic", "statistical", "Classic Croston estimator.", "sparse")
            _add_model("croston@sba", "statistical", "SBA intermittent estimator.", "sparse")
            _add_model("croston@tsb", "statistical", "TSB intermittent estimator.", "sparse")
        for ml_m, desc in (
            ("lightgbm", "Gradient boosted trees across lags and covariates."),
            ("xgboost", "Regularized gradient boosted trees across lags and covariates."),
            ("catboost", "Symmetric oblivious trees resistant to covariate overfitting."),
            ("regression_lags", "Penalized ridge lag model for clean elasticity."),
        ):
            _add_model(ml_m, "ml", desc, f"tier={tier}")
        for dl_m, desc in (
            ("tide", "Time-series Dense Encoder (TiDE) for global nonlinear covariate projection."),
            ("tsmixer", "All-MLP time- and feature-mixing neural architecture."),
            ("chronos_bolt", "Zero-shot univariate foundation model baseline."),
        ):
            if dl_m in avail and (allow_local_foundation or dl_m in ("tide", "tsmixer")):
                # On micro panels (<15 series) with low cross-series correlation, avoid overfitting
                # trained global neural nets on tiny sample sizes unless tier == deep_field
                if (
                    dl_m in ("tide", "tsmixer")
                    and scale_regime == "micro"
                    and tier != "deep_field"
                    and profile.cross_series_corr < 0.45
                    and where == "local"
                ):
                    spec_dl = f"{dl_m}@global" if profile.n_series >= 4 else dl_m
                else:
                    spec_dl = f"{dl_m}@global" if profile.n_series >= 4 else dl_m
                _add_model(spec_dl, "deep_learning", desc, f"tier={tier}")
        if tier == "deep_field":
            for extra_dl in ("patchtst", "tft", "chronos2", "timesfm_torch"):
                if extra_dl in avail and allow_local_foundation:
                    spec_ex = f"{extra_dl}@global" if profile.n_series >= 4 else extra_dl
                    _add_model(
                        spec_ex,
                        "deep_learning",
                        f"Deep Field neural/foundation specialist ({extra_dl}).",
                        "tier=deep_field",
                    )
                    if has_covs and extra_dl in ("chronos2", "timesfm_torch"):
                        uni_fm = f"{extra_dl}@univariate"
                        _add_model(
                            uni_fm,
                            "deep_learning",
                            f"Zero-shot univariate foundation baseline for {extra_dl}.",
                            f"fut_covs={len(profile.future_covariates)}",
                        )
            if "automl" in allowed_fams and (
                (allowed_families is not None and "automl" in allowed_families)
                or (where == "cloud_all" and profile.n_series >= 100)
            ):
                _add_model(
                    "vertex_tide",
                    "automl",
                    "Managed Vertex AI AutoML TiDE trainer for large fleet deep_field runs.",
                    f"n_series={profile.n_series}, where={where}",
                )
            if where in ("auto", "bigquery", "cloud_all") and not profile.source_table.startswith(
                "dataframe"
            ):
                _add_model(
                    "arima_plus",
                    "native",
                    "In-warehouse BigQuery ML ARIMA_PLUS running in parallel with Python families.",
                    "where=cloud",
                )

    if not models:
        models = ["naive_seasonal", "theta"]

    # 2.5. Data Grid Hygiene: Pre-Launch Leading Zeros & Timestamp Gaps
    trim_leading_zeros = bool(profile.leading_zero_share >= 0.10)
    if trim_leading_zeros:
        rationales.append(
            {
                "category": "data",
                "parameter": "data.trim_leading_zeros",
                "value": "True",
                "evidence": f"leading_zero_share={profile.leading_zero_share:.1%}",
                "rationale": (
                    "Trims pre-launch leading zero runs before each series' first active sale so "
                    "pre-launch zeros do not distort trend or intermittency estimates."
                ),
            }
        )

    grid_policy = (
        ("fill_zero" if (is_intermittent_heavy or profile.integer_valued) else "linear")
        if profile.missing_date_gap_share > 0.0
        else "strict"
    )
    if grid_policy != "strict":
        rationales.append(
            {
                "category": "data",
                "parameter": "data.grid_policy",
                "value": grid_policy,
                "evidence": f"missing_date_gap_share={profile.missing_date_gap_share:.1%}",
                "rationale": (
                    f"Internal timestamp gaps detected across {profile.missing_date_gap_share:.1%} "
                    f"of series; setting grid_policy='{grid_policy}' to regularize the date index."
                ),
            }
        )

    # 3. Target Transformation (Box-Cox / Log1p / Compositional Simplex)
    simplex_sum: float | None = None
    if profile.is_compositional_simplex:
        simplex_sum = round(float(profile.simplex_total or 1.0), 4)
        rationales.append(
            {
                "category": "features",
                "parameter": "features.simplex_sum",
                "value": str(simplex_sum),
                "evidence": (
                    f"is_compositional_simplex=True, simplex_total={simplex_sum}, "
                    f"n_series={profile.n_series}"
                ),
                "rationale": (
                    "Series form a constant-sum compositional simplex across timestamps; "
                    "enforces exact simplex closure and Aitchison log-ratio variants."
                ),
            }
        )
        if where != "bigquery":
            _add_model(
                "theta@transform_ilr",
                "statistical",
                "Isometric log-ratio (ILR) compositional simplex transform on Theta.",
                f"simplex_total={simplex_sum}",
            )
            if tier != "first_light":
                _add_model(
                    "regression_lags@transform_clr",
                    "ml",
                    "Centered log-ratio (CLR) compositional simplex transform on ridge lags.",
                    f"simplex_total={simplex_sum}",
                )

    use_log1p = (
        not profile.is_compositional_simplex
        and tier != "first_light"
        and profile.non_negative
        and not profile.integer_valued
        and profile.level_vol_corr >= float(th.get("log1p_level_vol_corr_min", 0.68))
        and profile.end_to_start_ratio >= float(th.get("log1p_growth_ratio_min", 1.55))
    )
    target_transform = "log1p" if use_log1p else "none"
    if use_log1p:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.target_transform",
                "value": "log1p",
                "evidence": (
                    f"level_vol_corr={profile.level_vol_corr:.2f}, "
                    f"growth_ratio={profile.end_to_start_ratio:.2f}"
                ),
                "rationale": (
                    "Variance scales multiplicatively with series level; log1p stabilizes "
                    "variance and prevents downward linear extrapolation bias."
                ),
            }
        )

    # 4. Structural Breaks, Outlier Scrubbing & Calendar/Fourier Harmonics
    level_shift = tier != "first_light" and profile.structural_break_share >= float(
        th.get("structural_break_share_min", 0.20)
    )
    if level_shift:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.level_shift",
                "value": "True",
                "evidence": f"structural_break_share={profile.structural_break_share:.1%}",
                "rationale": (
                    "Binary-segmentation detected step-change level shifts in >=20% of series; "
                    "adding step indicators prevents historical pre-break levels from biasing fits."
                ),
            }
        )

    outlier_policy = (
        "stl_resid"
        if (
            tier != "first_light"
            and profile.outlier_spike_share >= float(th.get("outlier_spike_share_min", 0.15))
        )
        else "none"
    )
    if outlier_policy != "none":
        rationales.append(
            {
                "category": "features",
                "parameter": "features.outlier_policy",
                "value": outlier_policy,
                "evidence": f"outlier_spike_share={profile.outlier_spike_share:.1%}",
                "rationale": (
                    "Isolated >4.5-MAD historical spikes detected; STL-residual scrubbing "
                    "prevents extreme one-off anomalies from distorting trend and lag weights."
                ),
            }
        )

    trading_days = bool(
        profile.freq in ("MS", "ME", "M", "QS", "QE", "Q") and profile.median_n_obs >= 24
    )
    if trading_days:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.trading_days",
                "value": "True",
                "evidence": f"freq='{profile.freq}', median_n_obs={profile.median_n_obs}",
                "rationale": (
                    "Monthly/quarterly aggregation benefits from X-13 weekday-vs-weekend trading "
                    "day counts per calendar period."
                ),
            }
        )

    f_norm = str(profile.freq).strip().upper()
    fourier = bool(
        (
            tier in ("deep_sky", "deep_field")
            and profile.median_acf_season >= 0.40
            and profile.median_n_obs >= 2 * max(7, profile.horizon)
        )
        or (tier != "first_light" and f_norm in ("H", "1H", "BH") and profile.median_n_obs >= 336)
    )
    if fourier:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.fourier",
                "value": "True",
                "evidence": (
                    f"freq='{profile.freq}', median_acf_season={profile.median_acf_season:.2f}, "
                    f"tier='{tier}'"
                ),
                "rationale": (
                    "Strong periodic autocorrelation or sub-daily multi-seasonality warrants "
                    "harmonic sine/cosine Fourier terms for tree and linear lag learners."
                ),
            }
        )

    # 5. Stockout Censoring Imputation
    censored_col = profile.censored_col
    if censored_col:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.censored_col",
                "value": censored_col,
                "evidence": f"censored_col='{censored_col}' present in panel",
                "rationale": (
                    "Enables fold-safe censored demand imputation so historical stockout dates "
                    "are reconstructed to latent demand before fitting."
                ),
            }
        )

    # 6. SBC Demand Regime Routing & Short-Series Handling
    regime_routing = is_intermittent_heavy or (
        len([s for s in profile.regime_shares.values() if s >= 0.15]) >= 2
        and (is_intermittent_heavy or has_cold_start)
    )
    if regime_routing:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.regime_routing",
                "value": "True",
                "evidence": f"regimes={profile.regime_shares}",
                "rationale": (
                    "Panel mixes heterogeneous Syntetos-Boylan-Croston demand regimes; routing "
                    "prevents smooth tree/ARIMA models from overfitting sparse/cold-start SKUs."
                ),
            }
        )
    if has_cold_start:
        rationales.append(
            {
                "category": "cold_start_transfer",
                "parameter": "cold_start_donor_transfer",
                "value": f"donor_similarity={profile.cold_start_donor_similarity:.2f}",
                "evidence": (
                    f"cold_start_share={profile.regime_shares.get('cold_start', 0.0):.1%}, "
                    f"min_n_obs={profile.min_n_obs}"
                ),
                "rationale": (
                    "Short-history cold-start series detected; analog donor prior transfer and "
                    "SBC regime routing blend mature donor seasonal profiles with empirical-Bayes "
                    "shrinkage."
                ),
            }
        )

    # 7. Physical Output Constraints & Goal Alignment
    floor: float | None = 0.0 if profile.non_negative else None
    if floor is not None:
        rationales.append(
            {
                "category": "output",
                "parameter": "output.floor",
                "value": "0.0",
                "evidence": "min(y) >= 0.0 across 100% of historical observations",
                "rationale": "Enforces physical non-negativity on point forecasts and bounds.",
            }
        )

    integer_rounding = (
        "cumulative_round"
        if (profile.integer_valued and is_intermittent_heavy)
        else ("round" if profile.integer_valued else "none")
    )
    if integer_rounding != "none":
        rationales.append(
            {
                "category": "output",
                "parameter": "output.integer_rounding",
                "value": integer_rounding,
                "evidence": f"integer_valued=True, intermittent={is_intermittent_heavy}",
                "rationale": (
                    "Preserves integer count semantics while cumulative_round prevents "
                    "fractional intermittent demand from rounding down to zero every day."
                ),
            }
        )

    underage_cost: float | None = None
    overage_cost: float | None = None
    target_quantile: float | None = None
    if goal == "service_level":
        decision_metric = "pinball"
        quantiles = [0.1, 0.5, 0.9, 0.95]
        underage_cost = 9.0
        overage_cost = 1.0
    elif goal == "probabilistic":
        decision_metric = "interval_score"
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    else:
        decision_metric = "mase" if is_intermittent_heavy else "wape"
        quantiles = [0.1, 0.5, 0.9]

    rationales.append(
        {
            "category": "backtest",
            "parameter": "backtest.decision_metric",
            "value": decision_metric,
            "evidence": f"goal='{goal}', intermittent={is_intermittent_heavy}",
            "rationale": (
                f"Optimizes model selection and ensemble weights for {decision_metric.upper()} "
                f"under goal='{goal}'."
            ),
        }
    )

    # 8. Backtest Folds, Ablations & Ensembling
    max_feasible_folds = max(
        1, min(3, (profile.median_n_obs - profile.horizon - 10) // max(1, profile.horizon))
    )
    if scale_regime == "hyperscale":
        n_folds = 1 if tier in ("first_light", "wide_field") else min(max_feasible_folds, 2)
    else:
        n_folds = (
            1
            if tier == "first_light"
            else min(max_feasible_folds, 2 if tier == "wide_field" else 3)
        )

    include_univariate_baselines = has_covs and tier != "first_light"
    ablation_modes = (
        ["univariate", "local_univariate"]
        if (has_covs and tier in ("deep_sky", "deep_field"))
        else []
    )

    if tier == "first_light":
        _valid_strats = {
            "mean",
            "median",
            "inverse_error",
            "champion",
            "horizon_specialist",
            "nnls",
            "ridge",
            "xgb",
        }
        raw_strats = list(th.get("first_light_ensemble_strategies", ["inverse_error", "mean"]))
        ensemble_strategies = [s for s in raw_strats if s in _valid_strats] or [
            "inverse_error",
            "mean",
        ]
        sel_cand = str(th.get("first_light_selection_rule", "best_single"))
        selection_rule = (
            sel_cand if sel_cand in {"best_single", "best_ensemble", "champion"} else "best_single"
        )
    elif tier == "wide_field":
        ensemble_strategies = (
            ["inverse_error", "nnls", "ridge"] if n_folds >= 2 else ["inverse_error", "mean"]
        )
        selection_rule = "champion"
    else:
        ensemble_strategies = ["inverse_error", "nnls", "ridge"]
        selection_rule = "champion"

    ensemble_top_k: int | None = (
        3
        if scale_regime == "hyperscale"
        else (5 if (len(models) >= 7 or scale_regime == "large") else None)
    )
    parsimony_margin = 0.005

    hpo_enabled = tier in ("deep_sky", "deep_field") and scale_regime != "hyperscale"
    hpo_trials = 6 if tier == "deep_sky" else (12 if tier == "deep_field" else 0)
    hpo_granularity = (
        "per_series" if (scale_regime == "micro" and tier == "deep_field") else "fleetwide"
    )

    n_est_fast = int(th.get("tree_n_estimators_fast", 80))
    model_params: dict[str, dict[str, Any]] = {}
    if profile.median_n_obs <= 365 and tier in ("first_light", "wide_field"):
        for tree_m in ("lightgbm", "xgboost"):
            if any(m.split("@", 1)[0] == tree_m for m in models):
                model_params[tree_m] = {"n_estimators": n_est_fast}

    freq_lags = _frequency_matched_lags(profile, n_folds=n_folds)
    if freq_lags is not None:
        lag_models_present: list[str] = []
        for lag_m in ("lightgbm", "xgboost", "catboost", "random_forest", "regression_lags"):
            if any(m.split("@", 1)[0] == lag_m for m in models):
                model_params.setdefault(lag_m, {})["lags"] = list(freq_lags)
                lag_models_present.append(lag_m)
        if lag_models_present:
            rationales.append(
                {
                    "category": "models",
                    "parameter": "model_params.*.lags",
                    "value": str(freq_lags),
                    "evidence": f"freq='{profile.freq}', min_n_obs={profile.min_n_obs}",
                    "rationale": (
                        f"Configured frequency-matched target lags {freq_lags} across "
                        f"{', '.join(lag_models_present)} to align lag depths with "
                        f"'{profile.freq}' seasonality and available training history."
                    ),
                }
            )

    inferred_mono = dict(profile.monotonic_constraints) or infer_monotonic_constraints(
        [*profile.future_covariates, *profile.past_covariates]
    )
    if inferred_mono:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.monotonic_constraints",
                "value": str(inferred_mono),
                "evidence": f"price/promo drivers detected: {list(inferred_mono.keys())}",
                "rationale": (
                    "Enforces structural sign guardrails (negative price elasticity, "
                    "non-negative promotional lift) on tree and linear lag learners."
                ),
            }
        )

    return {
        "models": models,
        "model_params": model_params,
        "scale_regime": scale_regime,
        "trim_leading_zeros": trim_leading_zeros,
        "grid_policy": grid_policy,
        "target_transform": target_transform,
        "simplex_sum": simplex_sum,
        "outlier_policy": outlier_policy,
        "level_shift": level_shift,
        "trading_days": trading_days,
        "fourier": fourier,
        "censored_col": censored_col,
        "monotonic_constraints": inferred_mono,
        "horizon_strategy": "recursive",
        "regime_routing": regime_routing,
        "include_univariate_baselines": include_univariate_baselines,
        "ablation_modes": ablation_modes,
        "ensemble_strategies": ensemble_strategies,
        "ensemble_top_k": ensemble_top_k,
        "parsimony_margin": parsimony_margin,
        "selection_rule": selection_rule,
        "n_folds": n_folds,
        "hpo_enabled": hpo_enabled,
        "hpo_trials": hpo_trials,
        "hpo_granularity": hpo_granularity,
        "floor": floor,
        "integer_rounding": integer_rounding,
        "underage_cost": underage_cost,
        "overage_cost": overage_cost,
        "target_quantile": target_quantile,
        "quantiles": quantiles,
        "decision_metric": decision_metric,
        "rationales": rationales,
    }


def propose_autofocus_pass(
    profile: PanelAutoProfile | dict[str, Any],
    prior_config_dict: dict[str, Any] | PanelAutoProfile,
    pass_index: int | dict[str, Any] = 2,
    diagnostics_summary: dict[str, Any] | None = None,
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
    available_models: Sequence[str] | None = None,
) -> Any:
    """Closed-Loop `Observe -> Review -> Re-Plan`: propose surgical mutations for Pass `pass_index`.

    Inspects out-of-fold econometric diagnostics from Pass `pass_index - 1` (FVA waterfall,
    feature attributions, horizon decay crossover $h^*$, directional bias, SBC demand regime
    sparsity/cold-start share, hierarchy structure, and residual heteroskedasticity/autocorrelation)
    and returns a dictionary of targeted config updates plus structured evidence rationales.
    """
    _return_mutations_only = False
    if isinstance(profile, dict) and isinstance(prior_config_dict, PanelAutoProfile):
        _return_mutations_only = True
        _real_prior = dict(profile)
        _real_profile = prior_config_dict
        if isinstance(pass_index, dict):
            _real_diag = dict(pass_index)
            _real_pass_idx = int(diagnostics_summary) if isinstance(diagnostics_summary, int) else 2
        else:
            _real_pass_idx = int(pass_index)
            _real_diag = dict(diagnostics_summary or {})
        profile = _real_profile
        prior_config_dict = _real_prior
        pass_index = _real_pass_idx
        diagnostics_summary = _real_diag
    else:
        assert isinstance(profile, PanelAutoProfile)
        assert isinstance(prior_config_dict, dict)
        pass_index = int(pass_index) if isinstance(pass_index, int) else 2
        diagnostics_summary = dict(diagnostics_summary or {})

    avail = set(
        available_models
        if available_models is not None
        else (
            "naive_seasonal",
            "naive_mean",
            "naive_moving_average",
            "theta",
            "holtwinters",
            "stl_bagging",
            "croston",
            "fft",
            "autoets",
            "regression_lags",
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "tide",
            "tsmixer",
            "chronos_bolt",
            "chronos2",
            "timesfm_torch",
        )
    )
    current_models: list[str] = list(prior_config_dict.get("models", []))
    features_cfg: dict[str, Any] = dict(prior_config_dict.get("features", {}))
    backtest_cfg: dict[str, Any] = dict(prior_config_dict.get("backtest", {}))
    hierarchy_cfg: dict[str, Any] = dict(prior_config_dict.get("hierarchy", {}))
    output_cfg: dict[str, Any] = dict(prior_config_dict.get("output", {}))
    ensemble_cfg: dict[str, Any] = dict(prior_config_dict.get("ensemble", {}))
    model_params: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in dict(prior_config_dict.get("model_params", {})).items()
    }

    mutations: list[dict[str, str]] = []
    added_models: list[str] = []
    th = evolved_policy_thresholds()
    n_folds = int(prior_config_dict.get("backtest", {}).get("n_folds", 1))
    decision_metric = str(
        prior_config_dict.get("backtest", {}).get(
            "decision_metric", diagnostics_summary.get("decision_metric", "wape")
        )
    )

    is_intermittent_heavy = (
        profile.dominant_regime in ("intermittent", "lumpy")
        or profile.zero_fraction >= 0.20
        or profile.median_adi >= 1.32
        or (
            profile.regime_shares.get("intermittent", 0.0) + profile.regime_shares.get("lumpy", 0.0)
        )
        >= float(th.get("intermittent_share_min", 0.25))
    )
    has_cold_start = (
        profile.dominant_regime == "cold_start"
        or profile.regime_shares.get("cold_start", 0.0) > 0.0
        or profile.min_n_obs < max(28, 2 * profile.horizon)
    )
    has_hierarchy = bool(profile.hierarchy_id_cols) or bool(
        prior_config_dict.get("hierarchy", {}).get("enabled", False)
    )
    is_probabilistic_goal = goal in ("service_level", "probabilistic") or decision_metric in (
        "pinball",
        "pinball_loss",
        "interval_score",
        "wis",
    )

    # 0. Stationary White-Noise Early Exit: only when the panel is genuinely smooth stationary
    # white noise (not intermittent, not cold-start, not hierarchical, not service-level quantile).
    wn_stop = bool(th.get("require_white_noise_stop", True))
    if (
        wn_stop
        and where != "bigquery"
        and not is_intermittent_heavy
        and not has_cold_start
        and not has_hierarchy
        and not is_probabilistic_goal
        and not profile.is_compositional_simplex
        and not bool(diagnostics_summary.get("coverage_drift_detected", False))
        and not bool(diagnostics_summary.get("cold_start_donor_transfer_recommended", False))
        and abs(profile.end_to_start_ratio - 1.0) < float(th.get("white_noise_growth_tol", 0.15))
        and profile.median_acf_season < float(th.get("white_noise_acf_season_max", 0.18))
        and not profile.future_covariates
        and profile.structural_break_share < 0.10
        and profile.outlier_spike_share < 0.10
        and diagnostics_summary.get("white_noise_residuals", False)
    ):
        if _return_mutations_only:
            return mutations
        return {
            "models": current_models,
            "added_models": added_models,
            "model_params": model_params,
            "features": features_cfg,
            "output": output_cfg,
            "ensemble": ensemble_cfg,
            "backtest": backtest_cfg,
            "hierarchy": hierarchy_cfg,
            "mutations": mutations,
        }

    # 1. Covariate Pruning, Plain Univariate Fallback & Exogenous Lag Synthesis
    noisy_covs: list[str] = list(diagnostics_summary.get("zero_or_negative_covariates", []))
    positive_covs: list[str] = list(diagnostics_summary.get("positive_covariates", []))
    curr_fut = list(features_cfg.get("future_covariates", []))
    cov_fva_pct = float(diagnostics_summary.get("covariate_fva_pct", 0.0))
    if noisy_covs and positive_covs and len(curr_fut) > len(positive_covs):
        pruned_fut = [c for c in curr_fut if c in positive_covs]
        if pruned_fut and pruned_fut != curr_fut:
            features_cfg["future_covariates"] = pruned_fut
            mutations.append(
                {
                    "action": "prune_decoy_covariates",
                    "domain": "covariates_and_horizon",
                    "parameter": "features.future_covariates",
                    "old_value": str(curr_fut),
                    "new_value": str(pruned_fut),
                    "evidence": f"noisy_covariates={noisy_covs}, positive_drivers={positive_covs}",
                    "rationale": (
                        f"Pruned {len(curr_fut) - len(pruned_fut)} zero/negative-value decoy "
                        f"covariate(s) ({', '.join(noisy_covs)}) identified by feature attribution "
                        "and covariate FVA audit."
                    ),
                }
            )
    elif curr_fut and not positive_covs and len(noisy_covs) >= len(curr_fut) and cov_fva_pct <= 0.0:
        features_cfg["future_covariates"] = []
        for uni_cand in ("theta", "holtwinters"):
            if uni_cand in avail and uni_cand not in current_models:
                current_models.append(uni_cand)
                added_models.append(uni_cand)
        mutations.append(
            {
                "action": "fallback_to_plain_univariate",
                "domain": "covariates_and_horizon",
                "parameter": "features.future_covariates",
                "old_value": str(curr_fut),
                "new_value": "[] (plain univariate)",
                "evidence": (
                    f"all_covariates_noisy={noisy_covs}, covariate_fva_pct={cov_fva_pct:+.2f}%"
                ),
                "rationale": (
                    "All candidate future covariates degraded or failed to improve validation "
                    "accuracy against plain univariate anchors; dropping all decoy covariates "
                    "and reverting to plain univariate forecasting."
                ),
            }
        )

    if (
        bool(diagnostics_summary.get("exog_lag_signal", False))
        and features_cfg.get("future_covariates")
        and not features_cfg.get("exog_lags")
    ):
        exog_lag_steps = [1, 2, 4] if str(profile.freq).upper().startswith("W") else [1, 2, 7]
        fut_for_lags = [str(c) for c in features_cfg.get("future_covariates", [])]
        exog_lags_dict = {c: list(exog_lag_steps) for c in fut_for_lags}
        features_cfg["exog_lags"] = exog_lags_dict
        mutations.append(
            {
                "action": "synthesize_exog_lags",
                "domain": "covariates_and_horizon",
                "parameter": "features.exog_lags",
                "old_value": "{}",
                "new_value": str(exog_lags_dict),
                "evidence": "exog_lag_signal=True (post-promo pull-forward / lagged response)",
                "rationale": (
                    "Residuals exhibit lagged correlation with exogenous drivers; synthesizing "
                    f"lagged covariate features ({exog_lag_steps}) to capture post-promo dips "
                    "and delayed price/promotion effects."
                ),
            }
        )

    # 1.5. Monotonic Price / Promo Elasticity Guardrail Injection
    eff_fut_covs = [str(c) for c in features_cfg.get("future_covariates", [])]
    eff_past_covs = [str(c) for c in features_cfg.get("past_covariates", [])]
    active_cov_set = set(eff_fut_covs) | set(eff_past_covs)
    raw_mono = dict(profile.monotonic_constraints) or infer_monotonic_constraints(
        [*eff_fut_covs, *eff_past_covs]
    )
    active_mono = {k: int(v) for k, v in raw_mono.items() if k in active_cov_set and int(v) != 0}
    has_monotone_variant = any("@monotone" in m for m in current_models)
    if active_mono and not has_monotone_variant:
        mono_candidates = (
            ("lightgbm", "regression_lags")
            if tier in ("first_light", "wide_field")
            else ("lightgbm", "xgboost", "regression_lags")
        )
        mono_added: list[str] = []
        for mc in mono_candidates:
            if mc in avail:
                spec_m = f"{mc}@monotone"
                if spec_m not in current_models:
                    current_models.append(spec_m)
                    added_models.append(spec_m)
                    mono_added.append(spec_m)
                if mc in ("lightgbm", "xgboost") and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        spec_m, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        old_mono = dict(prior_config_dict.get("features", {}).get("monotonic_constraints", {}))
        features_cfg["monotonic_constraints"] = active_mono
        if mono_added:
            strats_m = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_m:
                strats_m.append("champion")
            ensemble_cfg["strategies"] = strats_m
            mutations.append(
                {
                    "action": "inject_monotone_guardrails",
                    "domain": "covariates_and_horizon",
                    "parameter": "models + features.monotonic_constraints",
                    "old_value": str(old_mono or "unconstrained"),
                    "new_value": f"{', '.join(mono_added)} ({active_mono})",
                    "evidence": f"structural_drivers={list(active_mono.keys())}",
                    "rationale": (
                        "Injecting sign-constrained @monotone learners and monotonic_constraints "
                        "to prevent wrong-signed price/promotion elasticity under collinearity."
                    ),
                }
            )

    # 2. Multiplicative Variance / Log1p Transform Exploration
    resid_hetero = float(diagnostics_summary.get("residual_heteroskedasticity", 0.0))
    has_log_variant = features_cfg.get("transform") in ("log1p", "boxcox") or any(
        "@transform_log1p" in m or "@transform_boxcox" in m for m in current_models
    )
    if (
        not profile.is_compositional_simplex
        and profile.non_negative
        and not profile.integer_valued
        and not has_log_variant
        and (
            resid_hetero >= float(th.get("resid_hetero_min", 0.42))
            or (
                profile.level_vol_corr >= float(th.get("log1p_level_vol_corr_min", 0.65))
                and profile.end_to_start_ratio >= float(th.get("log1p_growth_ratio_min", 1.45))
            )
        )
    ):
        log_added: list[str] = []
        log_candidates = (
            ("lightgbm", "regression_lags", "theta")
            if tier in ("first_light", "wide_field")
            else ("lightgbm", "xgboost", "regression_lags", "theta")
        )
        for cand in log_candidates:
            if cand in avail:
                spec = f"{cand}@transform_log1p"
                if spec not in current_models:
                    current_models.append(spec)
                    added_models.append(spec)
                    log_added.append(spec)
                if cand in ("lightgbm", "xgboost") and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        spec, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        if log_added:
            strats = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats:
                strats.append("champion")
            ensemble_cfg["strategies"] = strats
            mutations.append(
                {
                    "action": "inject_transform_log1p",
                    "domain": "transform_and_hygiene",
                    "parameter": "models",
                    "old_value": "linear_scale_only",
                    "new_value": ", ".join(log_added),
                    "evidence": (
                        f"residual_heteroskedasticity={resid_hetero:.2f}, "
                        f"level_vol_corr={profile.level_vol_corr:.2f}"
                    ),
                    "rationale": (
                        "Residuals exhibit level-dependent variance; injecting @transform_log1p "
                        "variants while reusing Pass 1 linear models via cell borrowing."
                    ),
                }
            )

    # 2.5. Compositional Simplex Log-Ratio Exploration (@transform_ilr / @transform_clr)
    if profile.is_compositional_simplex and where != "bigquery":
        simplex_added: list[str] = []
        simplex_total_val = round(float(profile.simplex_total or 1.0), 4)
        for cand, tag in (
            ("lightgbm", "transform_ilr"),
            ("regression_lags", "transform_ilr"),
            ("theta", "transform_clr"),
            ("lightgbm", "transform_clr"),
        ):
            if cand in avail:
                spec_s = f"{cand}@{tag}"
                if spec_s not in current_models:
                    current_models.append(spec_s)
                    added_models.append(spec_s)
                    simplex_added.append(spec_s)
                if cand == "lightgbm" and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        spec_s, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        features_cfg["simplex_sum"] = simplex_total_val
        if simplex_added:
            strats_s = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_s:
                strats_s.append("champion")
            ensemble_cfg["strategies"] = strats_s
            mutations.append(
                {
                    "action": "inject_transform_simplex",
                    "domain": "transform_and_hygiene",
                    "parameter": "models + features.simplex_sum",
                    "old_value": str(prior_config_dict.get("features", {}).get("simplex_sum")),
                    "new_value": f"{', '.join(simplex_added)} (simplex_sum={simplex_total_val})",
                    "evidence": (
                        f"is_compositional_simplex=True, simplex_total={simplex_total_val}"
                    ),
                    "rationale": (
                        "Panel series form a constant-sum compositional simplex; injecting "
                        "Aitchison @transform_ilr and @transform_clr variants with exact "
                        "post-forecast simplex closure."
                    ),
                }
            )

    # 3. Lead-Time Horizon Crossover -> Direct Multi-Horizon Variants + Horizon Specialist
    crossover_step = diagnostics_summary.get("horizon_crossover_step")
    late_decay_ratio = float(diagnostics_summary.get("late_horizon_degradation_ratio", 1.0))
    has_direct = any("@direct" in m for m in current_models)
    has_specialist_regime = (
        is_intermittent_heavy or has_hierarchy or is_probabilistic_goal or has_cold_start
    )
    if not has_direct and (
        (isinstance(crossover_step, int) and 2 <= crossover_step < profile.horizon)
        or (
            not has_specialist_regime
            and late_decay_ratio >= float(th.get("direct_horizon_decay_ratio_min", 1.38))
        )
    ):
        direct_added: list[str] = []
        direct_candidates = (
            ("regression_lags", "lightgbm")
            if tier in ("first_light", "wide_field")
            else ("regression_lags", "lightgbm", "xgboost")
        )
        for ml_cand in direct_candidates:
            if ml_cand in avail:
                spec = f"{ml_cand}@direct"
                if spec not in current_models:
                    current_models.append(spec)
                    added_models.append(spec)
                    direct_added.append(spec)
                if ml_cand in ("lightgbm", "xgboost") and tier in ("first_light", "wide_field"):
                    model_params.setdefault(spec, {"n_estimators": 45})
        if direct_added:
            strats_hd = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "horizon_specialist" not in strats_hd:
                strats_hd.append("horizon_specialist")
            ensemble_cfg["strategies"] = strats_hd
            mutations.append(
                {
                    "action": "inject_direct_horizon_models",
                    "domain": "covariates_and_horizon",
                    "parameter": "models + ensemble.strategies",
                    "old_value": str(prior_config_dict.get("ensemble", {}).get("strategies")),
                    "new_value": f"{', '.join(direct_added)} + horizon_specialist",
                    "evidence": (
                        f"horizon_crossover_step={crossover_step}, "
                        f"late_horizon_degradation_ratio={late_decay_ratio:.2f}"
                    ),
                    "rationale": (
                        "Recursive lag rollouts degrade beyond short lead times; adding @direct "
                        "horizon models and enabling 'horizon_specialist' ensembling."
                    ),
                }
            )

    # 4. Structural Breaks & Outlier Spikes Discovered in Post-Run Residual Audit
    cp_share = float(diagnostics_summary.get("change_point_share", profile.structural_break_share))
    anom_share = float(diagnostics_summary.get("anomaly_share", profile.outlier_spike_share))
    if cp_share >= float(th.get("structural_break_share_min", 0.15)) and not features_cfg.get(
        "level_shift", False
    ):
        features_cfg["level_shift"] = True
        mutations.append(
            {
                "action": "enable_level_shift",
                "domain": "transform_and_hygiene",
                "parameter": "features.level_shift",
                "old_value": "False",
                "new_value": "True",
                "evidence": f"change_point_share={cp_share:.1%}",
                "rationale": (
                    "Post-fit change-point audit flagged structural level shifts; enabling "
                    "binary-segmentation step indicators."
                ),
            }
        )
    if (
        anom_share >= float(th.get("outlier_spike_share_min", 0.10))
        and features_cfg.get("outlier_policy", "none") == "none"
    ):
        features_cfg["outlier_policy"] = "stl_resid"
        mutations.append(
            {
                "action": "enable_stl_resid_outlier_scrubbing",
                "domain": "transform_and_hygiene",
                "parameter": "features.outlier_policy",
                "old_value": "none",
                "new_value": "stl_resid",
                "evidence": f"anomaly_share={anom_share:.1%}",
                "rationale": (
                    "Residual anomaly audit flagged extreme historical spikes; enabling "
                    "fold-safe STL residual outlier scrubbing."
                ),
            }
        )
    if (
        bool(diagnostics_summary.get("sliding_window_recommended", False))
        and backtest_cfg.get("scheme", "expanding") == "expanding"
        and profile.median_n_obs >= max(180, 4 * profile.horizon)
    ):
        backtest_cfg["scheme"] = "sliding"
        mutations.append(
            {
                "action": "enable_sliding_window_backtest",
                "domain": "transform_and_hygiene",
                "parameter": "backtest.scheme",
                "old_value": "expanding",
                "new_value": "sliding",
                "evidence": (
                    f"change_point_share={cp_share:.1%}, median_n_obs={profile.median_n_obs}"
                ),
                "rationale": (
                    "Persistent structural breaks with deep history bias expanding-window folds "
                    "toward pre-shift regimes; switching to sliding-window cross-validation."
                ),
            }
        )

    # 5. Intermittent / Lumpy Demand Specialist Escalation (dgp_03)
    if is_intermittent_heavy:
        inter_added: list[str] = []
        for inter_spec in ("naive_mean", "croston@classic", "croston@sba", "croston@tsb"):
            base_i = inter_spec.split("@", 1)[0]
            if base_i in avail and inter_spec not in current_models:
                current_models.append(inter_spec)
                added_models.append(inter_spec)
                inter_added.append(inter_spec)
        if inter_added:
            features_cfg["regime_routing"] = True
            strats_i = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            for s_i in ("champion", "median"):
                if s_i not in strats_i:
                    strats_i.append(s_i)
            ensemble_cfg["strategies"] = strats_i
            mutations.append(
                {
                    "action": "inject_intermittent_specialists",
                    "domain": "regime_and_specialists",
                    "parameter": "models + features.regime_routing",
                    "old_value": (
                        f"regime_routing="
                        f"{prior_config_dict.get('features', {}).get('regime_routing')}"
                    ),
                    "new_value": f"+{', '.join(inter_added)}, regime_routing=True",
                    "evidence": (
                        f"zero_fraction={profile.zero_fraction:.1%}, "
                        f"median_adi={profile.median_adi:.2f}, regime={profile.dominant_regime}"
                    ),
                    "rationale": (
                        "High zero-inflation and inter-demand intervals require dedicated "
                        "Syntetos-Boylan (SBA), Classic Croston, and unbiased level anchors "
                        "under SBC demand regime routing."
                    ),
                }
            )

    # 6. Cold-Start & Mixed-History Cross-Series Pooling / Donor Transfer Escalation (dgp_11)
    cold_start_donor_sim = max(
        float(getattr(profile, "cold_start_donor_similarity", 0.0)),
        float(diagnostics_summary.get("cold_start_donor_similarity", 0.0)),
    )
    cold_start_rec = bool(diagnostics_summary.get("cold_start_donor_transfer_recommended", False))
    if (
        has_cold_start or cold_start_donor_sim > 0.0 or cold_start_rec
    ) and not is_intermittent_heavy:
        cs_added: list[str] = []
        for cs_cand in ("holtwinters", "regression_lags", "xgboost"):
            if cs_cand in avail and cs_cand not in current_models:
                current_models.append(cs_cand)
                added_models.append(cs_cand)
                cs_added.append(cs_cand)
                if cs_cand == "xgboost" and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        cs_cand, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        donor_transfer_active = (
            cold_start_rec
            or cold_start_donor_sim >= 0.65
            or (has_cold_start and profile.n_series >= 2 and profile.min_n_obs < 14)
        )
        if cs_added or donor_transfer_active:
            features_cfg["regime_routing"] = True
            if donor_transfer_active:
                features_cfg["cold_start_transfer"] = True
            strats_cs = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_cs:
                strats_cs.append("champion")
            if n_folds >= 2 and "nnls" not in strats_cs:
                strats_cs.append("nnls")
            ensemble_cfg["strategies"] = strats_cs
            mutations.append(
                {
                    "action": (
                        "enable_cold_start_transfer"
                        if donor_transfer_active
                        else "enable_cold_start_pooling"
                    ),
                    "domain": "regime_and_specialists",
                    "parameter": "models + features.regime_routing + features.cold_start_transfer",
                    "old_value": f"{len(current_models) - len(cs_added)} models",
                    "new_value": (
                        f"+{', '.join(cs_added) or 'existing'}, regime_routing=True, "
                        f"cold_start_transfer={donor_transfer_active}"
                    ),
                    "evidence": (
                        f"cold_start_share={profile.regime_shares.get('cold_start', 0.0):.1%}, "
                        f"min_n_obs={profile.min_n_obs}, "
                        f"cold_start_donor_similarity={cold_start_donor_sim:.2f}"
                    ),
                    "rationale": (
                        "Panel mixes mature donor series and short-history cold-start series; "
                        "enabling k-NN analog donor shrinkage transfer and SBC regime routing."
                    ),
                }
            )

    # 7. Hierarchical Non-Negative Coherent & Cross-Temporal Specialist Escalation (dgp_08 / dgp_14)
    if has_hierarchy:
        hier_added: list[str] = []
        for h_cand in ("holtwinters", "regression_lags", "xgboost"):
            if h_cand in avail and h_cand not in current_models:
                current_models.append(h_cand)
                added_models.append(h_cand)
                hier_added.append(h_cand)
                if h_cand == "xgboost" and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        h_cand, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        rec_methods = list(select_hierarchy_reconciliation_methods(profile, tier=tier))
        if hierarchy_cfg:
            if profile.horizon >= 7 and profile.median_n_obs >= 28:
                hierarchy_cfg["reconcile_temporal"] = True
                hierarchy_cfg["cross_temporal"] = True
                if "cross_temporal_mint" not in rec_methods:
                    rec_methods.append("cross_temporal_mint")
            hierarchy_cfg["reconciliation_methods"] = rec_methods
        if hier_added:
            strats_h = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_h:
                strats_h.append("champion")
            ensemble_cfg["strategies"] = strats_h
            mutations.append(
                {
                    "action": "refine_hierarchical_coherence",
                    "domain": "regime_and_specialists",
                    "parameter": "models + hierarchy.reconciliation_methods",
                    "old_value": f"{len(current_models) - len(hier_added)} base models",
                    "new_value": (
                        f"+{', '.join(hier_added)} (reconciled via {', '.join(rec_methods)})"
                    ),
                    "evidence": (
                        f"hierarchy_id_cols={list(profile.hierarchy_id_cols)}, "
                        f"hierarchy_node_count={profile.hierarchy_node_count or profile.n_series}"
                    ),
                    "rationale": (
                        "Adding triple exponential smoothing and regularized lag models across all "
                        "hierarchy nodes before applying scale-aware coherent reconciliation "
                        f"({', '.join(rec_methods)})."
                    ),
                }
            )
    elif (
        profile.is_compositional_simplex
        and profile.horizon >= 7
        and profile.median_n_obs >= 28
        and where != "bigquery"
        and not bool(hierarchy_cfg.get("reconcile_temporal", False))
    ):
        hierarchy_cfg["reconcile_temporal"] = True
        hierarchy_cfg.setdefault("reconciliation_methods", ["wls_struct"])
        mutations.append(
            {
                "action": "enable_temporal_reconciliation",
                "domain": "regime_and_specialists",
                "parameter": "hierarchy.reconcile_temporal",
                "old_value": "False",
                "new_value": "True (wls_struct)",
                "evidence": f"freq='{profile.freq}', horizon={profile.horizon}",
                "rationale": (
                    "Enables multi-horizon THieF temporal hierarchy reconciliation across "
                    "aggregated calendar horizons."
                ),
            }
        )

    # 8. Probabilistic / Newsvendor Service-Level Quantile Refinement (dgp_10)
    if is_probabilistic_goal:
        prob_added: list[str] = []
        for p_cand in ("holtwinters", "regression_lags", "xgboost"):
            if p_cand in avail and p_cand not in current_models:
                current_models.append(p_cand)
                added_models.append(p_cand)
                prob_added.append(p_cand)
                if p_cand == "xgboost" and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        p_cand, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        old_pf = str(output_cfg.get("point_forecast", "median"))
        if prob_added or old_pf != "auto":
            output_cfg["point_forecast"] = "auto"
            if (
                goal == "service_level"
                and output_cfg.get("target_quantile") is None
                and output_cfg.get("underage_cost") is None
            ):
                output_cfg["underage_cost"] = 9.0
                output_cfg["overage_cost"] = 1.0
            strats_p = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_p:
                strats_p.append("champion")
            if n_folds >= 2 and "nnls" not in strats_p:
                strats_p.append("nnls")
            ensemble_cfg["strategies"] = strats_p
            mutations.append(
                {
                    "action": "refine_probabilistic_quantiles",
                    "domain": "regime_and_specialists",
                    "parameter": "models + output.point_forecast",
                    "old_value": f"point_forecast={old_pf}",
                    "new_value": (f"+{', '.join(prob_added) or 'existing'}, point_forecast='auto'"),
                    "evidence": f"goal='{goal}', decision_metric='{decision_metric}'",
                    "rationale": (
                        "Optimizing upper-tail Newsvendor pinball / interval loss via per-cell "
                        "OOF arm calibration (point_forecast='auto') and expanded learners."
                    ),
                }
            )

    # 8.2. Non-Exchangeable Drift -> Adaptive Conformal Inference (ACI / CQR) Escalation
    ks_drift = max(
        float(getattr(profile, "residual_drift_ks_stat", 0.0)),
        float(diagnostics_summary.get("residual_drift_ks_stat", 0.0)),
    )
    cov_drift = bool(diagnostics_summary.get("coverage_drift_detected", False)) or (
        is_probabilistic_goal and ks_drift >= 0.28
    )
    old_cal = str(output_cfg.get("calibration_method", "conformal"))
    if cov_drift and old_cal == "conformal":
        output_cfg["calibration_method"] = "aci_normalized"
        output_cfg.setdefault("aci_gamma", 0.05)
        mutations.append(
            {
                "action": "enable_adaptive_conformal",
                "domain": "regime_and_specialists",
                "parameter": "output.calibration_method",
                "old_value": old_cal,
                "new_value": f"aci_normalized (aci_gamma={output_cfg['aci_gamma']})",
                "evidence": (
                    f"coverage_drift_detected={cov_drift}, residual_drift_ks_stat={ks_drift:.2f}"
                ),
                "rationale": (
                    "Non-exchangeable residual drift detected across backtest folds; escalating "
                    "prediction intervals to Adaptive Conformal Inference (ACI) online tracking."
                ),
            }
        )

    # 8.5. Censored Demand / Stockout Reconstruction Specialist Escalation (dgp_07)
    has_censored = bool(profile.censored_col) or bool(features_cfg.get("censored_col"))
    if has_censored:
        cens_added: list[str] = []
        for c_cand in ("holtwinters", "regression_lags", "xgboost"):
            if c_cand in avail and c_cand not in current_models:
                current_models.append(c_cand)
                added_models.append(c_cand)
                cens_added.append(c_cand)
                if c_cand == "xgboost" and tier in ("first_light", "wide_field"):
                    model_params.setdefault(
                        c_cand, {"n_estimators": int(th.get("tree_n_estimators_fast", 55))}
                    )
        if cens_added:
            strats_c = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            for s_tag in ("champion", "horizon_specialist"):
                if s_tag not in strats_c:
                    strats_c.append(s_tag)
            ensemble_cfg["strategies"] = strats_c
            mutations.append(
                {
                    "action": "refine_censored_demand_models",
                    "domain": "regime_and_specialists",
                    "parameter": "models + ensemble.strategies",
                    "old_value": f"censored_col={features_cfg.get('censored_col')}",
                    "new_value": f"+{', '.join(cens_added)}",
                    "evidence": f"censored_col='{features_cfg.get('censored_col')}'",
                    "rationale": (
                        "Adding triple exponential smoothing, regularized linear lags, and "
                        "gradient boosting over reconstructed uncensored latent demand."
                    ),
                }
            )

    # 8.8. In-Warehouse BigQuery SQL Foundation Escalation (`where == 'bigquery'`)
    if where == "bigquery" and "timesfm" in avail and "timesfm" not in current_models:
        current_models.append("timesfm")
        added_models.append("timesfm")
        ensemble_cfg["enabled"] = True
        strats_bq = list(ensemble_cfg.get("strategies", ["inverse_error", "mean"]))
        for s_tag in ("champion", "horizon_specialist"):
            if s_tag not in strats_bq:
                strats_bq.append(s_tag)
        ensemble_cfg["strategies"] = strats_bq
        mutations.append(
            {
                "action": "escalate_bigquery_foundation",
                "domain": "regime_and_specialists",
                "parameter": "models + ensemble.strategies",
                "old_value": "arima_plus",
                "new_value": "arima_plus, timesfm (with champion + horizon_specialist blending)",
                "evidence": "where='bigquery', Pass 1 evaluated arima_plus baseline",
                "rationale": (
                    "Escalating in-warehouse BigQuery SQL portfolio to include AI.FORECAST "
                    "(TimesFM foundation model) while borrowing Pass 1's ARIMA_PLUS cells."
                ),
            }
        )

    # 9. Residual Autocorrelation / Underfitting -> Escalate Model Expressiveness
    resid_acf1 = float(diagnostics_summary.get("residual_autocorr_lag1", 0.0))
    fva_pct = float(diagnostics_summary.get("fva_vs_naive_pct", 0.0))
    if (
        not mutations
        and (
            resid_acf1 >= float(th.get("resid_acf1_escalate_min", 0.30))
            or fva_pct < float(th.get("fva_escalate_min_pct", 2.0))
        )
        and not (wn_stop and diagnostics_summary.get("white_noise_residuals", False))
    ):
        escalated: list[str] = []
        esc_pool = (
            ("holtwinters", "regression_lags", "catboost", "random_forest", "fft", "autoets")
            if tier in ("first_light", "wide_field")
            else ("catboost", "random_forest", "stl_bagging", "autoets", "tide")
        )
        for cand in esc_pool:
            if cand in avail and cand not in [m.split("@", 1)[0] for m in current_models]:
                spec_cand = f"{cand}@global" if (cand == "tide" and profile.n_series >= 4) else cand
                current_models.append(spec_cand)
                added_models.append(spec_cand)
                escalated.append(spec_cand)
                if tier in ("first_light", "wide_field"):
                    if cand == "catboost":
                        model_params.setdefault(spec_cand, {"iterations": 80})
                    elif cand in ("lightgbm", "xgboost", "random_forest"):
                        model_params.setdefault(spec_cand, {"n_estimators": 80})
                if len(escalated) >= 2:
                    break
        if escalated:
            if profile.horizon >= 7 and profile.median_n_obs >= 28 and where != "bigquery":
                hierarchy_cfg["reconcile_temporal"] = True
            strats_esc = list(ensemble_cfg.get("strategies", ["inverse_error"]))
            if "champion" not in strats_esc:
                strats_esc.append("champion")
            ensemble_cfg["strategies"] = strats_esc
            mutations.append(
                {
                    "action": "escalate_model_diversity",
                    "domain": "regime_and_specialists",
                    "parameter": "models",
                    "old_value": f"{len(current_models) - len(escalated)} models",
                    "new_value": f"+{', '.join(escalated)}",
                    "evidence": f"residual_acf1={resid_acf1:.2f}, fva_vs_naive_pct={fva_pct:.2f}%",
                    "rationale": (
                        "Exploitable autocorrelation remains in Pass 1 residuals; expanding model "
                        "diversity while borrowing completed Pass 1 models."
                    ),
                }
            )

    freq_lags_p2 = _frequency_matched_lags(profile, n_folds=n_folds)
    if freq_lags_p2 is not None:
        for lag_m in ("lightgbm", "xgboost", "catboost", "random_forest", "regression_lags"):
            if any(m.split("@", 1)[0] == lag_m for m in current_models):
                model_params.setdefault(lag_m, {}).setdefault("lags", list(freq_lags_p2))

    if _return_mutations_only:
        return mutations

    return {
        "models": current_models,
        "added_models": added_models,
        "model_params": model_params,
        "features": features_cfg,
        "output": output_cfg,
        "ensemble": ensemble_cfg,
        "backtest": backtest_cfg,
        "hierarchy": hierarchy_cfg,
        "mutations": mutations,
    }


def propose_autofocus_dag_branches(
    profile: PanelAutoProfile | dict[str, Any],
    prior_config_dict: dict[str, Any] | PanelAutoProfile,
    pass_index: int | dict[str, Any] = 2,
    diagnostics_summary: dict[str, Any] | None = None,
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
    available_models: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Decompose Level-2 `autofocus` mutations into parallel Tournament DAG branches (`2A`..`2C`).

    When only one hypothesis domain triggers (e.g., pure covariate pruning on `dgp_04` or pure
    `@transform_log1p` on `dgp_02`), returns a single branch (`pass_id="2"`). When two or three
    orthogonal hypothesis domains trigger simultaneously (`transform_and_hygiene`,
    `covariates_and_horizon`, `regime_and_specialists`), returns independent branch specifications
    (`pass_id="2A"`, `"2B"`, `"2C"`) that each branch off Pass 1 (`parent_pass_ids=("1",)`),
    enabling isolated attribution of every mutation before Level-3 synthesis.
    """
    if (
        isinstance(profile, dict)
        and isinstance(prior_config_dict, PanelAutoProfile)
        and isinstance(pass_index, dict)
        and diagnostics_summary is None
    ):
        diagnostics_summary = pass_index
        pass_index = 2
        profile, prior_config_dict = prior_config_dict, profile
    assert isinstance(profile, PanelAutoProfile)
    assert isinstance(prior_config_dict, dict)
    if diagnostics_summary is None:
        diagnostics_summary = {}
    pass_idx_int = int(pass_index) if isinstance(pass_index, int) else 2

    full_proposal = propose_autofocus_pass(
        profile,
        prior_config_dict,
        pass_idx_int,
        diagnostics_summary,
        tier=tier,
        where=where,
        goal=goal,
        available_models=available_models,
    )
    all_mutations: list[dict[str, str]] = list(full_proposal.get("mutations", []))
    if not all_mutations:
        return []

    domain_order = (
        ("transform_and_hygiene", "Target Transform & Structural Hygiene"),
        ("covariates_and_horizon", "Covariate Pruning & Direct Horizon Specialists"),
        ("regime_and_specialists", "Regime Routing & Domain Model Specialists"),
    )
    domain_canonical_suffix = {
        "transform_and_hygiene": "A",
        "covariates_and_horizon": "B",
        "regime_and_specialists": "C",
    }
    muts_by_domain: dict[str, list[dict[str, str]]] = {
        dom: [m for m in all_mutations if m.get("domain", "regime_and_specialists") == dom]
        for dom, _ in domain_order
    }
    active_domains = [(dom, title) for dom, title in domain_order if muts_by_domain[dom]]

    if len(active_domains) <= 1:
        single_dom = active_domains[0][0] if active_domains else "regime_and_specialists"
        single_title = active_domains[0][1] if active_domains else "Targeted Refinement"
        canon_branch_id = f"{pass_idx_int}{domain_canonical_suffix.get(single_dom, 'C')}"
        return [
            {
                **full_proposal,
                "pass_id": str(pass_idx_int),
                "branch_id": canon_branch_id,
                "domain": single_dom,
                "parent_pass_ids": (str(pass_idx_int - 1),),
                "branch_hypothesis": single_title,
            }
        ]

    # Build one isolated branch per active hypothesis domain starting from prior_config_dict
    base_models = list(prior_config_dict.get("models", []))
    base_features = dict(prior_config_dict.get("features", {}))
    base_output = dict(prior_config_dict.get("output", {}))
    base_ensemble = dict(prior_config_dict.get("ensemble", {}))
    base_backtest = dict(prior_config_dict.get("backtest", {}))
    base_hierarchy = dict(prior_config_dict.get("hierarchy") or {})
    base_params = {k: dict(v) for k, v in dict(prior_config_dict.get("model_params", {})).items()}

    full_features = dict(full_proposal.get("features", base_features))
    full_output = dict(full_proposal.get("output", base_output))
    full_backtest = dict(full_proposal.get("backtest", base_backtest))
    full_hierarchy = dict(full_proposal.get("hierarchy") or base_hierarchy)
    full_params = dict(full_proposal.get("model_params", base_params))

    branches: list[dict[str, Any]] = []
    suffixes = ("A", "B", "C")
    for idx, (dom, title) in enumerate(active_domains):
        branch_id = f"{pass_idx_int}{suffixes[idx]}"
        canon_branch_id = f"{pass_idx_int}{domain_canonical_suffix.get(dom, suffixes[idx])}"
        dom_muts = muts_by_domain[dom]
        b_models = list(base_models)
        b_added: list[str] = []
        b_features = dict(base_features)
        b_output = dict(base_output)
        b_ensemble = dict(base_ensemble)
        b_backtest = dict(base_backtest)
        b_hierarchy = dict(base_hierarchy)
        b_params = {k: dict(v) for k, v in base_params.items()}
        b_strats = list(b_ensemble.get("strategies", ["inverse_error"]))

        if dom == "transform_and_hygiene":
            for m in full_proposal["models"]:
                if ("@transform_" in m) and m not in b_models:
                    b_models.append(m)
                    b_added.append(m)
                    if m in full_params:
                        b_params[m] = dict(full_params[m])
            for fkey in ("transform", "level_shift", "outlier_policy", "simplex_sum"):
                if fkey in full_features:
                    b_features[fkey] = full_features[fkey]
            if "scheme" in full_backtest:
                b_backtest["scheme"] = full_backtest["scheme"]
            if "champion" not in b_strats:
                b_strats.append("champion")
        elif dom == "covariates_and_horizon":
            for m in full_proposal["models"]:
                if ("@direct" in m or "@monotone" in m) and m not in b_models:
                    b_models.append(m)
                    b_added.append(m)
                    if m in full_params:
                        b_params[m] = dict(full_params[m])
            if "future_covariates" in full_features:
                b_features["future_covariates"] = list(full_features["future_covariates"])
            if "exog_lags" in full_features and isinstance(full_features["exog_lags"], dict):
                b_features["exog_lags"] = dict(full_features["exog_lags"])
            if "monotonic_constraints" in full_features and isinstance(
                full_features["monotonic_constraints"], dict
            ):
                b_features["monotonic_constraints"] = dict(full_features["monotonic_constraints"])
            for s_tag in ("horizon_specialist", "champion"):
                if s_tag not in b_strats:
                    b_strats.append(s_tag)
        else:
            for m in full_proposal["models"]:
                if (
                    m not in b_models
                    and "@transform_" not in m
                    and "@direct" not in m
                    and "@monotone" not in m
                ):
                    b_models.append(m)
                    b_added.append(m)
                    if m in full_params:
                        b_params[m] = dict(full_params[m])
            for rf_key in ("regime_routing", "cold_start_transfer"):
                if rf_key in full_features:
                    b_features[rf_key] = full_features[rf_key]
            if full_hierarchy:
                if "reconciliation_methods" in full_hierarchy:
                    b_hierarchy["reconciliation_methods"] = list(
                        full_hierarchy["reconciliation_methods"]
                    )
                for hkey in (
                    "reconcile_temporal",
                    "temporal_aggregations",
                    "cross_temporal",
                    "spatial_temporal_weight",
                ):
                    if hkey in full_hierarchy:
                        b_hierarchy[hkey] = full_hierarchy[hkey]
            for ok in ("point_forecast", "target_quantile", "calibration_method", "aci_gamma"):
                if ok in full_output:
                    b_output[ok] = full_output[ok]
            for s_tag in full_proposal.get("ensemble", {}).get("strategies", []):
                if s_tag not in b_strats:
                    b_strats.append(s_tag)

        b_ensemble["strategies"] = b_strats
        branches.append(
            {
                "pass_id": branch_id,
                "branch_id": canon_branch_id,
                "domain": dom,
                "parent_pass_ids": ("1",),
                "branch_hypothesis": title,
                "models": b_models,
                "added_models": b_added,
                "model_params": b_params,
                "features": b_features,
                "output": b_output,
                "ensemble": b_ensemble,
                "backtest": b_backtest,
                "hierarchy": b_hierarchy or None,
                "mutations": dom_muts,
            }
        )

    return branches


def propose_autofocus_synthesis_pass(
    profile: PanelAutoProfile,
    pass1_config_dict: dict[str, Any],
    branch_outcomes: Sequence[dict[str, Any]],
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
) -> dict[str, Any]:
    """Synthesize winning Level-2 Tournament DAG branches into a Level-3 Join & Stacking config.

    Combines all feature, output, and model additions from Level-2 branches that improved upon
    Pass 1's validation score, enables learned stacking (`nnls`, `ridge`, `champion`,
    `horizon_specialist`), and optionally enables targeted Optuna HPO on `deep_sky`/`deep_field`.
    """
    _ = (profile, where, goal)
    synth_models: list[str] = list(pass1_config_dict.get("models", []))
    synth_features: dict[str, Any] = dict(pass1_config_dict.get("features", {}))
    synth_output: dict[str, Any] = dict(pass1_config_dict.get("output", {}))
    synth_ensemble: dict[str, Any] = dict(pass1_config_dict.get("ensemble", {}))
    synth_backtest: dict[str, Any] = dict(pass1_config_dict.get("backtest", {}))
    synth_hierarchy: dict[str, Any] = dict(pass1_config_dict.get("hierarchy") or {})
    synth_hpo: dict[str, Any] = dict(pass1_config_dict.get("hpo", {}))
    synth_params: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in dict(pass1_config_dict.get("model_params", {})).items()
    }
    n_folds = int(pass1_config_dict.get("backtest", {}).get("n_folds", 1))

    winning_ids: list[str] = []
    merged_actions: list[str] = []
    added_models: list[str] = []

    for b in branch_outcomes:
        b_id = str(b.get("pass_id", "2"))
        b_score = float(b.get("best_score", float("inf")))
        p1_score = float(b.get("pass1_score", float("inf")))
        # Accept branch if it beat or matched Pass 1 within 0.5%
        if b_score <= p1_score * 1.005:
            winning_ids.append(b_id)
            b_cfg = dict(b.get("config_dict", {}))
            for m in b_cfg.get("models", []):
                if m not in synth_models:
                    synth_models.append(m)
                    added_models.append(m)
            for k, v in dict(b_cfg.get("model_params", {})).items():
                synth_params.setdefault(k, dict(v))
            b_feat = dict(b_cfg.get("features", {}))
            for fk in (
                "future_covariates",
                "exog_lags",
                "monotonic_constraints",
                "holiday_window",
                "transform",
                "simplex_sum",
                "level_shift",
                "outlier_policy",
                "regime_routing",
                "cold_start_transfer",
            ):
                if fk in b_feat and b_feat[fk] != pass1_config_dict.get("features", {}).get(fk):
                    synth_features[fk] = b_feat[fk]
            b_bt = dict(b_cfg.get("backtest", {}))
            if "scheme" in b_bt and b_bt["scheme"] != pass1_config_dict.get("backtest", {}).get(
                "scheme"
            ):
                synth_backtest["scheme"] = b_bt["scheme"]
            b_hier = dict(b_cfg.get("hierarchy") or {})
            if "reconciliation_methods" in b_hier:
                synth_hierarchy["reconciliation_methods"] = list(b_hier["reconciliation_methods"])
            for hk in (
                "reconcile_temporal",
                "temporal_aggregations",
                "cross_temporal",
                "spatial_temporal_weight",
            ):
                if hk in b_hier and b_hier[hk] != (pass1_config_dict.get("hierarchy") or {}).get(
                    hk
                ):
                    synth_hierarchy[hk] = b_hier[hk]
            b_out = dict(b_cfg.get("output", {}))
            for ok in ("point_forecast", "target_quantile", "calibration_method", "aci_gamma"):
                if ok in b_out and b_out[ok] is not None:
                    synth_output[ok] = b_out[ok]
            for s in b_cfg.get("ensemble", {}).get("strategies", []):
                if s not in synth_ensemble.get("strategies", []):
                    synth_ensemble.setdefault("strategies", []).append(s)
            for mut_act in b.get("mutations_applied", ()):
                if mut_act not in merged_actions:
                    merged_actions.append(str(mut_act))

    strats = list(synth_ensemble.get("strategies", ["inverse_error"]))
    for s_stack in ("champion", "horizon_specialist"):
        if s_stack not in strats:
            strats.append(s_stack)
    if n_folds >= 2:
        for s_learned in ("nnls", "ridge"):
            if s_learned not in strats:
                strats.append(s_learned)
    synth_ensemble["strategies"] = strats
    synth_ensemble["enabled"] = len(synth_models) >= 2

    if tier in ("deep_sky", "deep_field"):
        synth_hpo["enabled"] = True
        synth_hpo["n_trials"] = max(int(synth_hpo.get("n_trials", 6)), 6)

    parent_ids = (
        tuple(winning_ids)
        if winning_ids
        else tuple(str(b.get("pass_id", "2")) for b in branch_outcomes)
    )
    mutations = [
        {
            "action": "synthesize_winning_dag_branches",
            "domain": "synthesis",
            "parameter": "models + features + ensemble.strategies",
            "old_value": f"branches={[b.get('pass_id') for b in branch_outcomes]}",
            "new_value": f"merged={list(parent_ids)}, strategies={strats}",
            "evidence": f"winning_branches={list(parent_ids)}, actions={merged_actions}",
            "rationale": (
                "Synthesizes all non-degrading Level-2 hypothesis branches into a unified "
                "model portfolio with learned stacking and cell borrowing across all branches."
            ),
        }
    ]
    return {
        "pass_id": "3",
        "parent_pass_ids": parent_ids,
        "branch_hypothesis": "Level-3 Tournament Synthesis & Learned Stacking",
        "models": synth_models,
        "added_models": added_models,
        "model_params": synth_params,
        "features": synth_features,
        "output": synth_output,
        "ensemble": synth_ensemble,
        "backtest": synth_backtest,
        "hierarchy": synth_hierarchy or None,
        "hpo": synth_hpo,
        "mutations": mutations,
    }


def estimate_continue_confidence(
    profile: PanelAutoProfile,
    pass_index: int,
    max_passes: int,
    diagnostics_summary: dict[str, Any],
    proposed_mutations: Sequence[dict[str, str]],
    *,
    remaining_minutes: float | None = None,
    next_pass_p50_minutes: float = 0.5,
    min_confidence: float = 0.35,
) -> dict[str, Any]:
    """Compute calibrated continuation confidence $C_{k+1} \\in [0, 1]$ and expected FVA gain.

    Combines five empirical diagnostic signals:
    1. Number and specificity of actionable mutations proposed (`proposed_mutations`).
    2. Residual autocorrelation ($|\\rho_1|$) indicating uncaptured temporal structure.
    3. Residual heteroskedasticity / multiplicative scaling signal.
    4. Decoy covariate noise share that can be pruned.
    5. Horizon decay crossover step $h^*$.

    If residuals are already white noise (`white_noise_residuals=True`) and no structural mutation
    is warranted (as on `dgp_12_already_optimal_white_noise`), confidence drops below `0.20` and
    `should_continue` is ``False``.
    """
    _ = (profile, diagnostics_summary)
    if pass_index >= max_passes:
        return {
            "confidence": 0.0,
            "expected_gain_pct": 0.0,
            "should_continue": False,
            "reason": f"Reached max_passes={max_passes} for the selected exposure tier.",
        }

    if remaining_minutes is not None and next_pass_p50_minutes > remaining_minutes:
        return {
            "confidence": 0.0,
            "expected_gain_pct": 0.0,
            "should_continue": False,
            "reason": (
                f"Pre-loop wall-clock gate: predicted Pass {pass_index + 1} duration "
                f"({next_pass_p50_minutes:.2f}m) exceeds remaining budget "
                f"({remaining_minutes:.2f}m)."
            ),
        }

    if not proposed_mutations:
        return {
            "confidence": 0.12,
            "expected_gain_pct": 0.2,
            "should_continue": False,
            "reason": (
                "Residuals are statistically clean (white-noise floor reached) with no "
                "high-confidence structural mutations remaining."
            ),
        }

    # Score the strength of the proposed interventions
    th = evolved_policy_thresholds()
    action_weights = dict(
        th.get(
            "action_weights",
            {
                "prune_decoy_covariates": (0.88, 7.2),
                "inject_transform_log1p": (0.86, 8.5),
                "inject_transform_simplex": (0.87, 8.0),
                "inject_direct_horizon_models": (0.82, 5.4),
                "enable_level_shift": (0.84, 7.0),
                "enable_stl_resid_outlier_scrubbing": (0.80, 5.0),
                "inject_intermittent_specialists": (0.85, 6.2),
                "enable_cold_start_pooling": (0.83, 6.0),
                "enable_cold_start_transfer": (0.85, 6.5),
                "enable_cold_start_donor_transfer": (0.85, 6.5),
                "enable_adaptive_conformal": (0.82, 5.2),
                "enable_aci_conformal_tracking": (0.82, 5.2),
                "refine_hierarchical_coherence": (0.82, 5.8),
                "enable_temporal_reconciliation": (0.78, 4.8),
                "refine_probabilistic_quantiles": (0.84, 6.4),
                "synthesize_winning_dag_branches": (0.80, 4.5),
                "escalate_model_diversity": (0.60, 2.6),
            },
        )
    )

    best_conf = 0.25
    total_expected_gain = 0.0
    for mut in proposed_mutations:
        act = mut.get("action", "")
        c_w, g_w = action_weights.get(act, (0.50, 2.0))
        best_conf = max(best_conf, float(c_w))
        total_expected_gain += float(g_w)

    # Discount confidence on later passes (diminishing returns)
    decay_base = float(th.get("pass_decay_factor", 0.78))
    decay = 1.0 if pass_index == 1 else (decay_base ** (pass_index - 1))
    confidence = float(min(0.96, max(0.05, best_conf * decay)))
    expected_gain_pct = float(round(min(25.0, total_expected_gain * decay), 2))

    should_continue = confidence >= min_confidence
    if should_continue:
        actions_str = ", ".join(m.get("action", "refine") for m in proposed_mutations)
        reason = (
            f"Pass {pass_index + 1} approved (confidence={confidence:.2f} >= {min_confidence:.2f}, "
            f"expected FVA gain ~+{expected_gain_pct:.1f}% via {actions_str})."
        )
    else:
        reason = (
            f"Stopping after Pass {pass_index}: continuation confidence {confidence:.2f} "
            f"is below threshold {min_confidence:.2f} (expected gain +{expected_gain_pct:.1f}%)."
        )

    return {
        "confidence": round(confidence, 3),
        "expected_gain_pct": expected_gain_pct,
        "should_continue": should_continue,
        "reason": reason,
    }
