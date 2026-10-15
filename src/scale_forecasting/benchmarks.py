"""Synthetic Data-Generating Process (DGP) Benchmark Bank — pure, deterministic, zero GCP imports.

Provides 12 canonical time-series panels with **known mathematical ground truth** for validating,
benchmarking, and evolving (`auto_policy.py`) the Glass-Box Auto-Forecaster (`build_config`,
`auto_forecast`) and its closed-loop `autofocus` controller.

Why synthetic DGPs with known ground truth?
-------------------------------------------
Real-world datasets are indispensable for systems scale testing, but their true data-generating
process is unobserved: when a model wins or loses, you cannot prove *why* without controlled
counterfactuals. Every benchmark in this module constructs both:

1. **Observed historical + future covariate panel (`dataset.df`)** — the exact DataFrame handed to
   `Forecaster.build_config()` or `Forecaster.auto()`, including future covariate rows over the
   forecast horizon when exogenous drivers exist.
2. **Noise-free Oracle Conditional Expectation (`dataset.oracle_future_df`)** — the exact
   mathematical expectation $\\mu_{i, T+h} = \\mathbb{E}[Y_{i, T+h} \\mid \\mathcal{F}_T]$ over the
   future horizon $h = 1 \\dots H$, plus a structured :class:`BenchmarkSpec` declaring the true
   Syntetos–Boylan–Croston (SBC) demand regime, true vs. decoy covariates, true target transform,
   horizon crossover step $h^*$, structural breaks, stockout censoring, and whether Pass 1 is
   already statistically optimal (`already_optimal_at_pass_1=True`).

Public surface: ``BENCHMARK_NAMES``, ``BenchmarkSpec``, ``BenchmarkDataset``, ``list_benchmarks``,
``benchmark_catalog``, ``load_benchmark``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BENCHMARK_NAMES",
    "BENCHMARK_SPECS",
    "BenchmarkDataset",
    "BenchmarkSpec",
    "benchmark_catalog",
    "list_benchmarks",
    "load_benchmark",
]

BENCHMARK_NAMES: tuple[str, ...] = (
    "dgp_01_smooth_seasonal",
    "dgp_02_multiplicative_log",
    "dgp_03_intermittent_lumpy",
    "dgp_04_true_vs_decoy_covs",
    "dgp_05_horizon_crossover",
    "dgp_06_structural_break_spikes",
    "dgp_07_censored_stockouts",
    "dgp_08_hierarchical_nonneg",
    "dgp_09_cross_series_global",
    "dgp_10_newsvendor_service",
    "dgp_11_cold_start_mixed",
    "dgp_12_already_optimal_white_noise",
    "dgp_13_all_decoy_univariate_wins",
    "dgp_14_large_hierarchy_intermittent_leaves",
    "dgp_15_scale_crossover_panel",
    "dgp_16_monthly_trading_days",
    "dgp_17_missing_gaps_and_leading_zeros",
    "dgp_18_promo_lag_pullforward",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Ground-truth mathematical specification for one canonical DGP benchmark panel."""

    name: str
    title: str
    description: str
    freq: str
    horizon: int
    n_series: int
    history_length: int
    expected_regime: str  # "smooth" | "erratic" | "intermittent" | "lumpy" | "cold_start"
    non_negative: bool = True
    integer_valued: bool = False
    heteroskedastic: bool = False
    true_transform: str = "none"  # "none" | "log1p" | "boxcox"
    true_covariates: tuple[str, ...] = ()
    decoy_covariates: tuple[str, ...] = ()
    static_covariates: tuple[str, ...] = ()
    hierarchy_id_cols: tuple[str, ...] = ()
    censored_col: str | None = None
    has_horizon_crossover: bool = False
    crossover_step: int | None = None
    has_structural_break: bool = False
    has_anomalies: bool = False
    has_cold_start: bool = False
    needs_regime_routing: bool = False
    prefers_global_pooling: bool = False
    recommended_goal: str = "accuracy"  # "accuracy" | "probabilistic" | "service_level" | "fast"
    optimal_quantile: float | None = None
    lead_time: int | None = None
    already_optimal_at_pass_1: bool = False
    expected_autofocus_actions: tuple[str, ...] = ()
    seed: int = 20261014

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        raw = asdict(self)
        for k, v in list(raw.items()):
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw


@dataclass(frozen=True)
class BenchmarkDataset:
    """A generated DGP benchmark panel paired with its ground-truth spec and oracle expectations.

    Attributes:
        spec: The :class:`BenchmarkSpec` describing the ground-truth DGP properties.
        df: Historical panel (`ts_id`, `ds`, `y`, plus any static/dynamic covariates).
        future_df: Future covariate rows over the forecast horizon (`ts_id`, `ds`, and dynamic
            covariates) when exogenous drivers exist, or an empty DataFrame for univariate panels.
        oracle_future_df: Exact noise-free conditional expectation (`ts_id`, `ds`, `horizon_step`,
            `oracle_yhat`, `y_realized`) over the forecast horizon $h = 1 \\dots H$.
    """

    spec: BenchmarkSpec
    df: pd.DataFrame
    future_df: pd.DataFrame
    oracle_future_df: pd.DataFrame

    @property
    def panel_df(self) -> pd.DataFrame:
        """Alias for ``self.df`` (the historical + exogenous panel DataFrame)."""
        return self.df

    @property
    def full_panel_with_future_covariates(self) -> pd.DataFrame:
        """Return historical rows concatenated with future covariate rows (`y = NaN` in future)."""
        if self.future_df.empty:
            return self.df.copy()
        fut = self.future_df.copy()
        if "y" not in fut.columns:
            fut["y"] = float("nan")
        cols = list(self.df.columns)
        for c in cols:
            if c not in fut.columns:
                fut[c] = None
        return pd.concat([self.df, fut[cols]], ignore_index=True)

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the benchmark spec and generated row counts."""
        d = self.spec.to_dict()
        d["n_hist_rows"] = len(self.df)
        d["n_future_rows"] = len(self.future_df)
        d["n_oracle_rows"] = len(self.oracle_future_df)
        d["columns"] = list(self.df.columns)
        return d


_SPECS: dict[str, BenchmarkSpec] = {
    "dgp_01_smooth_seasonal": BenchmarkSpec(
        name="dgp_01_smooth_seasonal",
        title="Smooth Additive Trend + Weekly Seasonal Harmonics",
        description=(
            "12 daily series with smooth linear trend, two weekly harmonics, low Gaussian noise "
            "(CV ~ 0.10), and strictly positive demand."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_transform="none",
        expected_autofocus_actions=("floor=0.0", "seasonal_harmonics"),
        seed=101,
    ),
    "dgp_02_multiplicative_log": BenchmarkSpec(
        name="dgp_02_multiplicative_log",
        title="Exponential Growth with Multiplicative Variance (Log-Normal DGP)",
        description=(
            "12 daily series with compounding exponential growth where seasonal amplitude and "
            "shock variance scale strictly with the series level (sigma proportional to mu)."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        heteroskedastic=True,
        true_transform="log1p",
        expected_autofocus_actions=("inject_transform_log1p", "champion_selection"),
        seed=102,
    ),
    "dgp_03_intermittent_lumpy": BenchmarkSpec(
        name="dgp_03_intermittent_lumpy",
        title="Sparse Intermittent & Lumpy Integer Spare-Parts Demand",
        description=(
            "12 daily integer count series mixing 60% intermittent/lumpy SKUs (ADI > 1.6, "
            "CV^2 > 0.6) with 40% smooth/erratic SKUs."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="intermittent",
        non_negative=True,
        integer_valued=True,
        needs_regime_routing=True,
        expected_autofocus_actions=("regime_routing", "croston_sba_tsb", "cumulative_round"),
        seed=103,
    ),
    "dgp_04_true_vs_decoy_covs": BenchmarkSpec(
        name="dgp_04_true_vs_decoy_covs",
        title="2 True Exogenous Drivers + 3 Pure-Noise Decoy Covariates",
        description=(
            "12 daily retail series driven by true future covariates ('promo_depth', "
            "'price_index') alongside 3 uncorrelated decoy noise columns ('noise_a', "
            "'noise_b', 'noise_c') that degrade unregularized models."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=("promo_depth", "price_index"),
        decoy_covariates=("noise_a", "noise_b", "noise_c"),
        expected_autofocus_actions=("prune_decoy_covariates", "retain_true_drivers"),
        seed=104,
    ),
    "dgp_05_horizon_crossover": BenchmarkSpec(
        name="dgp_05_horizon_crossover",
        title="Short-Memory AR(1) Shock + Long-Horizon Seasonal Regime Crossover",
        description=(
            "12 daily series with persistent AR(1) momentum dominating steps 1..5 and "
            "deterministic calendar/promo structure dominating steps 6..14."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=("promo_flag",),
        has_horizon_crossover=True,
        crossover_step=5,
        expected_autofocus_actions=("inject_direct_horizon_models", "horizon_specialist_ensemble"),
        seed=105,
    ),
    "dgp_06_structural_break_spikes": BenchmarkSpec(
        name="dgp_06_structural_break_spikes",
        title="Permanent Historical Level Shift + Isolated 5-Sigma Outlier Spikes",
        description=(
            "12 daily series experiencing a permanent +65% step-change level shift at 65% of "
            "history plus 3 isolated one-day anomalous spikes."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="erratic",
        non_negative=True,
        has_structural_break=True,
        has_anomalies=True,
        expected_autofocus_actions=("enable_level_shift", "enable_stl_resid_outlier_scrubbing"),
        seed=106,
    ),
    "dgp_07_censored_stockouts": BenchmarkSpec(
        name="dgp_07_censored_stockouts",
        title="Right-Censored Demand Under Inventory Stockouts",
        description=(
            "12 daily series where true latent demand is clipped at a warehouse capacity ceiling "
            "on dates flagged by 'stockout_flag == 1'."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=("promo_flag",),
        censored_col="stockout_flag",
        expected_autofocus_actions=("enable_stockout_imputation",),
        seed=107,
    ),
    "dgp_08_hierarchical_nonneg": BenchmarkSpec(
        name="dgp_08_hierarchical_nonneg",
        title="Two-Level Region -> Store Hierarchy with Non-Negative Leaf Counts",
        description=(
            "12 daily series across 3 regions x 4 stores where coherent cross-level aggregation "
            "and non-negativity require hierarchical reconciliation."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        hierarchy_id_cols=("region", "store"),
        expected_autofocus_actions=("enable_hierarchy_reconciliation", "mint_nonneg"),
        seed=108,
    ),
    "dgp_09_cross_series_global": BenchmarkSpec(
        name="dgp_09_cross_series_global",
        title="Shared Cross-Series Promo & Price Elasticity Curve (Global Pooling)",
        description=(
            "16 moderate-length daily series sharing a common nonlinear price/promo elasticity "
            "surface where global cross-series models outperform per-series local fits."
        ),
        freq="D",
        horizon=14,
        n_series=16,
        history_length=112,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=("promo_depth", "price_index"),
        static_covariates=("category", "store_tier"),
        prefers_global_pooling=True,
        expected_autofocus_actions=("prefer_global_pooling", "include_static_covariates"),
        seed=109,
    ),
    "dgp_10_newsvendor_service": BenchmarkSpec(
        name="dgp_10_newsvendor_service",
        title="Asymmetric Newsvendor Replenishment (9:1 Stockout-to-Holding Cost)",
        description=(
            "12 daily SKU demand series with 7-day supplier lead time and 9:1 underage/overage "
            "cost ratio (critical fractile q* = 0.90)."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        recommended_goal="service_level",
        optimal_quantile=0.90,
        lead_time=7,
        expected_autofocus_actions=("quantile_grid_service_level", "conformal_calibration"),
        seed=110,
    ),
    "dgp_11_cold_start_mixed": BenchmarkSpec(
        name="dgp_11_cold_start_mixed",
        title="Mixed Maturity Panel (Mature SKUs + Newly Launched Cold-Start SKUs)",
        description=(
            "12 mature daily series (T=140) alongside 4 newly launched cold-start series (T=21) "
            "requiring short-series fallback and zero-shot/simple baseline routing."
        ),
        freq="D",
        horizon=14,
        n_series=16,
        history_length=140,
        expected_regime="cold_start",
        non_negative=True,
        has_cold_start=True,
        needs_regime_routing=True,
        expected_autofocus_actions=("short_series_fallback", "regime_routing"),
        seed=111,
    ),
    "dgp_12_already_optimal_white_noise": BenchmarkSpec(
        name="dgp_12_already_optimal_white_noise",
        title="Stationary Seasonal Wave + Irreducible White Noise (Stopping-Rule Test)",
        description=(
            "12 daily series consisting of a clean stationary seasonal wave plus irreducible "
            "i.i.d. Gaussian noise. Pass 1 captures 100% of the predictable signal; autofocus "
            "must recognize low continuation confidence and stop after Pass 1."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        already_optimal_at_pass_1=True,
        expected_autofocus_actions=("stop_after_pass_1",),
        seed=112,
    ),
    "dgp_13_all_decoy_univariate_wins": BenchmarkSpec(
        name="dgp_13_all_decoy_univariate_wins",
        title="Pure Univariate Seasonal Signal + 3 Uncorrelated Decoy Covariates",
        description=(
            "12 daily series driven strictly by trend and weekly harmonics, accompanied by 3 "
            "pure-noise decoy covariates ('noise_a', 'noise_b', 'noise_c'). Autofocus must "
            "detect zero covariate value, fall back to plain univariate forecasting, and "
            "highlight beaten multivariate challengers."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=(),
        decoy_covariates=("noise_a", "noise_b", "noise_c"),
        expected_autofocus_actions=("fallback_to_plain_univariate", "champion_is_univariate"),
        seed=113,
    ),
    "dgp_14_large_hierarchy_intermittent_leaves": BenchmarkSpec(
        name="dgp_14_large_hierarchy_intermittent_leaves",
        title="Multi-Level Region -> Store Hierarchy with Sparse Intermittent Leaves",
        description=(
            "16 daily count series across 4 regions x 4 stores where bottom-level leaves exhibit "
            "sparse intermittent demand while regional and total rollups are smooth."
        ),
        freq="D",
        horizon=14,
        n_series=16,
        history_length=140,
        expected_regime="intermittent",
        non_negative=True,
        integer_valued=True,
        hierarchy_id_cols=("region", "store"),
        needs_regime_routing=True,
        expected_autofocus_actions=(
            "enable_hierarchy_reconciliation",
            "top_down",
            "inject_intermittent_specialists",
        ),
        seed=114,
    ),
    "dgp_15_scale_crossover_panel": BenchmarkSpec(
        name="dgp_15_scale_crossover_panel",
        title="Small-to-Medium Scale Crossover Panel with High Cross-Series Correlation",
        description=(
            "24 daily series sharing synchronized seasonal harmonics and macro trend where "
            "scale-aware policy activates hybrid local + @global cross-series pooling."
        ),
        freq="D",
        horizon=14,
        n_series=24,
        history_length=112,
        expected_regime="smooth",
        non_negative=True,
        prefers_global_pooling=True,
        expected_autofocus_actions=("scale_regime_small", "global_cross_series_pooling"),
        seed=115,
    ),
    "dgp_16_monthly_trading_days": BenchmarkSpec(
        name="dgp_16_monthly_trading_days",
        title="Monthly Revenue Series Driven by Calendar Trading-Day Counts",
        description=(
            "12 monthly series (freq='MS', T=48, H=6) where monthly billing totals fluctuate "
            "with the number of weekday business days vs. weekend days in each calendar month."
        ),
        freq="MS",
        horizon=6,
        n_series=12,
        history_length=48,
        expected_regime="smooth",
        non_negative=True,
        expected_autofocus_actions=("enable_trading_days",),
        seed=116,
    ),
    "dgp_17_missing_gaps_and_leading_zeros": BenchmarkSpec(
        name="dgp_17_missing_gaps_and_leading_zeros",
        title="Staggered Product Launches (Leading Zeros) + Missing Timestamp Gaps",
        description=(
            "12 daily series where half of the SKUs have 28 pre-launch leading zeros and "
            "internal missing date gaps requiring automatic grid_policy and trim_leading_zeros."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        expected_autofocus_actions=("enable_trim_leading_zeros", "regularize_grid_policy"),
        seed=117,
    ),
    "dgp_18_promo_lag_pullforward": BenchmarkSpec(
        name="dgp_18_promo_lag_pullforward",
        title="Promotional Spike Followed by Post-Promo Pull-Forward Dip (Exog Lags)",
        description=(
            "12 daily series where promotions trigger a +85 unit contemporaneous spike on day t "
            "followed by a -52 unit pantry-loading dip on day t+1, requiring exog_lags synthesis."
        ),
        freq="D",
        horizon=14,
        n_series=12,
        history_length=140,
        expected_regime="smooth",
        non_negative=True,
        true_covariates=("promo_flag",),
        expected_autofocus_actions=("synthesize_exog_lags",),
        seed=118,
    ),
}


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = _SPECS


def list_benchmarks() -> list[str]:
    """Return the sorted names of all 12 canonical DGP benchmark panels."""
    return list(BENCHMARK_NAMES)


def benchmark_catalog() -> pd.DataFrame:
    """Return a summary DataFrame describing all 12 canonical DGP benchmark panels."""
    rows: list[dict[str, Any]] = []
    for name in BENCHMARK_NAMES:
        spec = _SPECS[name]
        rows.append(
            {
                "name": spec.name,
                "title": spec.title,
                "freq": spec.freq,
                "horizon": spec.horizon,
                "n_series": spec.n_series,
                "history_length": spec.history_length,
                "expected_regime": spec.expected_regime,
                "true_transform": spec.true_transform,
                "true_covariates": list(spec.true_covariates),
                "decoy_covariates": list(spec.decoy_covariates),
                "has_horizon_crossover": spec.has_horizon_crossover,
                "has_structural_break": spec.has_structural_break,
                "censored_col": spec.censored_col,
                "hierarchy_id_cols": list(spec.hierarchy_id_cols),
                "already_optimal_at_pass_1": spec.already_optimal_at_pass_1,
            }
        )
    return pd.DataFrame(rows)


def load_benchmark(
    name: str,
    *,
    n_series: int | None = None,
    history_length: int | None = None,
    horizon: int | None = None,
    seed: int | None = None,
) -> BenchmarkDataset:
    """Deterministically generate one canonical DGP benchmark dataset by name.

    Args:
        name: Benchmark identifier in :data:`BENCHMARK_NAMES`.
        n_series: Optional override for number of series in the panel.
        history_length: Optional override for historical observations per series.
        horizon: Optional override for forecast horizon steps.
        seed: Optional override for random number generator seed.

    Returns:
        :class:`BenchmarkDataset` containing `.spec`, `.df`, `.future_df`, and `.oracle_future_df`.
    """
    if name not in _SPECS:
        valid = ", ".join(BENCHMARK_NAMES)
        raise ValueError(f"Unknown benchmark '{name}'. Valid benchmarks: {valid}")
    base_spec = _SPECS[name]
    eff_n = int(n_series if n_series is not None else base_spec.n_series)
    eff_t = int(history_length if history_length is not None else base_spec.history_length)
    eff_h = int(horizon if horizon is not None else base_spec.horizon)
    eff_seed = int(seed if seed is not None else base_spec.seed)
    spec = BenchmarkSpec(
        **{
            **asdict(base_spec),
            "n_series": eff_n,
            "history_length": eff_t,
            "horizon": eff_h,
            "seed": eff_seed,
        }
    )
    rng = np.random.default_rng(eff_seed)
    total_len = eff_t + eff_h
    dates = pd.date_range("2025-01-01", periods=total_len, freq=spec.freq)
    t_idx = np.arange(total_len, dtype=np.float64)

    hist_rows: list[pd.DataFrame] = []
    fut_rows: list[pd.DataFrame] = []
    oracle_rows: list[pd.DataFrame] = []

    for i in range(eff_n):
        ts_id = f"series_{i + 1:03d}"
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weekly = np.sin(2.0 * np.pi * t_idx / 7.0 + phase) + 0.35 * np.cos(
            4.0 * np.pi * t_idx / 7.0 + phase
        )
        extra_cols: dict[str, np.ndarray | list[Any]] = {}
        start_idx = 0

        if name == "dgp_01_smooth_seasonal":
            base_level = float(rng.uniform(80.0, 160.0))
            slope = float(rng.uniform(0.15, 0.45))
            amp = 0.18 * base_level
            mu = np.maximum(1.0, base_level + slope * t_idx + amp * weekly)
            noise = rng.normal(0.0, 0.04 * base_level, size=total_len)
            y_all = np.maximum(0.0, mu + noise)

        elif name == "dgp_02_multiplicative_log":
            log_base = float(rng.uniform(3.2, 4.0))
            log_slope = float(rng.uniform(0.014, 0.021))
            log_season = 0.38 * np.sin(2.0 * np.pi * t_idx / 7.0 + phase)
            sigma = 0.14
            eps = rng.normal(0.0, sigma, size=total_len)
            log_mu = log_base + log_slope * t_idx + log_season
            # Conditional expectation of exp(log_mu + eps) - 1 under log-normal shocks
            mu = np.maximum(0.0, np.expm1(log_mu + 0.5 * sigma**2))
            y_all = np.maximum(0.0, np.expm1(log_mu + eps))

        elif name == "dgp_03_intermittent_lumpy":
            is_sparse = i < round(0.65 * eff_n)
            if is_sparse:
                p_demand = float(rng.uniform(0.22, 0.38))
                mean_size = float(rng.uniform(4.0, 14.0))
                occ = rng.binomial(1, p_demand, size=total_len)
                sizes = rng.poisson(lam=mean_size, size=total_len) + 1
                mu = np.full(total_len, p_demand * (mean_size + 1.0), dtype=np.float64)
                y_all = (occ * sizes).astype(np.float64)
            else:
                lam = float(rng.uniform(12.0, 28.0))
                mu = np.maximum(1.0, lam + 3.5 * np.sin(2.0 * np.pi * t_idx / 7.0 + phase))
                y_all = rng.poisson(lam=mu).astype(np.float64)

        elif name == "dgp_04_true_vs_decoy_covs":
            base_level = float(rng.uniform(90.0, 140.0))
            promo_depth = (
                (rng.uniform(0.0, 1.0, size=total_len) < 0.25)
                * rng.uniform(0.15, 0.45, size=total_len)
            ).round(3)
            price_index = np.clip(
                1.0 + np.cumsum(rng.normal(0.0, 0.015, size=total_len)), 0.75, 1.30
            ).round(3)
            noise_a = rng.normal(0.0, 1.0, size=total_len).round(3)
            noise_b = np.cumsum(rng.normal(0.0, 0.2, size=total_len)).round(3)
            noise_c = rng.uniform(-1.0, 1.0, size=total_len).round(3)

            mu = np.maximum(
                5.0,
                base_level
                + 0.12 * t_idx
                + 14.0 * weekly
                + 85.0 * promo_depth
                - 55.0 * (price_index - 1.0),
            )
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 3.2, size=total_len))
            extra_cols = {
                "promo_depth": promo_depth,
                "price_index": price_index,
                "noise_a": noise_a,
                "noise_b": noise_b,
                "noise_c": noise_c,
            }

        elif name == "dgp_05_horizon_crossover":
            base_level = float(rng.uniform(100.0, 150.0))
            promo_flag = (rng.uniform(0.0, 1.0, size=total_len) < 0.22).astype(np.float64)
            det_signal = base_level + 22.0 * weekly + 48.0 * promo_flag
            # AR(1) shock with phi = 0.78 dominates steps 1..5, then decays into det_signal
            phi = 0.78
            ar_shock = np.zeros(total_len, dtype=np.float64)
            innov = rng.normal(0.0, 11.0, size=total_len)
            for t in range(1, total_len):
                ar_shock[t] = phi * ar_shock[t - 1] + innov[t]
            # Force a meaningful end-of-history shock so the horizon crossover is sharp
            ar_shock[eff_t - 1] = float(rng.choice([-1.0, 1.0]) * rng.uniform(28.0, 42.0))
            for h_idx in range(eff_h):
                t = eff_t + h_idx
                ar_shock[t] = phi * ar_shock[t - 1] + innov[t]
            y_all = np.maximum(1.0, det_signal + ar_shock)
            mu = det_signal.copy()
            last_shock = ar_shock[eff_t - 1]
            for h_idx in range(eff_h):
                mu[eff_t + h_idx] = max(
                    1.0, det_signal[eff_t + h_idx] + (phi ** (h_idx + 1)) * last_shock
                )
            extra_cols = {"promo_flag": promo_flag}

        elif name == "dgp_06_structural_break_spikes":
            base_level = float(rng.uniform(75.0, 125.0))
            break_idx = int(0.65 * eff_t)
            step_shift = np.where(t_idx >= break_idx, 0.65 * base_level, 0.0)
            mu = np.maximum(5.0, base_level + 0.08 * t_idx + step_shift + 12.0 * weekly)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 3.0, size=total_len))
            # Inject 3 isolated historical anomalous spikes (never in future horizon)
            spike_positions = [int(0.20 * eff_t), int(0.45 * eff_t), int(0.82 * eff_t)]
            for pos in spike_positions:
                if 0 <= pos < eff_t:
                    y_all[pos] += 1.65 * base_level

        elif name == "dgp_07_censored_stockouts":
            base_level = float(rng.uniform(95.0, 135.0))
            promo_flag = (rng.uniform(0.0, 1.0, size=total_len) < 0.20).astype(np.float64)
            mu = np.maximum(10.0, base_level + 0.15 * t_idx + 16.0 * weekly + 45.0 * promo_flag)
            latent_y = np.maximum(0.0, mu + rng.normal(0.0, 3.5, size=total_len))
            # Stockouts occur on ~18% of historical dates where demand is clipped by 45%
            stockout_flag = np.zeros(total_len, dtype=np.int64)
            hist_mask = rng.uniform(0.0, 1.0, size=eff_t) < 0.18
            stockout_flag[:eff_t] = hist_mask.astype(np.int64)
            y_all = latent_y.copy()
            y_all[:eff_t] = np.where(hist_mask, 0.52 * latent_y[:eff_t], latent_y[:eff_t])
            extra_cols = {
                "promo_flag": promo_flag,
                "stockout_flag": stockout_flag,
            }

        elif name == "dgp_08_hierarchical_nonneg":
            region_idx = i // 4
            store_idx = i % 4
            region_name = f"region_{region_idx + 1}"
            store_name = f"store_{region_idx + 1}_{store_idx + 1}"
            ts_id = f"{region_name}__{store_name}"
            base_level = float(rng.uniform(18.0, 55.0))
            region_wave = 8.0 * np.sin(2.0 * np.pi * t_idx / 7.0 + 0.5 * region_idx)
            mu = np.maximum(1.5, base_level + 0.06 * t_idx + region_wave)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.2, size=total_len))
            extra_cols = {
                "region": [region_name] * total_len,
                "store": [store_name] * total_len,
            }

        elif name == "dgp_09_cross_series_global":
            category = "cat_a" if i % 2 == 0 else "cat_b"
            store_tier = "tier_1" if i < eff_n // 2 else "tier_2"
            cat_mult = 1.25 if category == "cat_a" else 0.85
            tier_base = 120.0 if store_tier == "tier_1" else 75.0
            promo_depth = (
                (rng.uniform(0.0, 1.0, size=total_len) < 0.25)
                * rng.uniform(0.10, 0.40, size=total_len)
            ).round(3)
            price_index = np.clip(
                1.0 + np.cumsum(rng.normal(0.0, 0.012, size=total_len)), 0.80, 1.25
            ).round(3)
            shared_wave = 15.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
            # Shared cross-series elasticity law
            mu = np.maximum(
                5.0,
                tier_base
                + cat_mult
                * (shared_wave + 95.0 * promo_depth - 65.0 * (price_index - 1.0) + 0.10 * t_idx),
            )
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 4.0, size=total_len))
            extra_cols = {
                "promo_depth": promo_depth,
                "price_index": price_index,
                "category": [category] * total_len,
                "store_tier": [store_tier] * total_len,
            }

        elif name == "dgp_10_newsvendor_service":
            base_level = float(rng.uniform(60.0, 110.0))
            mu = np.maximum(5.0, base_level + 0.10 * t_idx + 11.0 * weekly)
            sigma = 0.12 * base_level
            y_all = np.maximum(0.0, mu + rng.normal(0.0, sigma, size=total_len))

        elif name == "dgp_11_cold_start_mixed":
            is_cold = i >= round(0.75 * eff_n)
            start_idx = max(0, eff_t - 21) if is_cold else 0
            base_level = float(rng.uniform(70.0, 130.0))
            mu = np.maximum(5.0, base_level + 0.12 * t_idx + 13.0 * weekly)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 3.0, size=total_len))

        elif name == "dgp_12_already_optimal_white_noise":
            base_level = float(rng.uniform(95.0, 115.0))
            # Pure stationary mean + exact 7-day cycle + irreducible white noise (no trend/covs)
            mu = base_level + 9.0 * np.sin(2.0 * np.pi * t_idx / 7.0 + phase)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.0, size=total_len))

        elif name == "dgp_13_all_decoy_univariate_wins":
            base_level = float(rng.uniform(90.0, 130.0))
            noise_a = rng.normal(0.0, 1.0, size=total_len).round(3)
            noise_b = rng.uniform(-1.0, 1.0, size=total_len).round(3)
            noise_c = rng.normal(0.0, 2.0, size=total_len).round(3)
            mu = np.maximum(5.0, base_level + 0.14 * t_idx + 15.0 * weekly)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.5, size=total_len))
            extra_cols = {
                "noise_a": noise_a,
                "noise_b": noise_b,
                "noise_c": noise_c,
            }

        elif name == "dgp_14_large_hierarchy_intermittent_leaves":
            region_idx = i // 4
            store_idx = i % 4
            region_name = f"region_{region_idx + 1}"
            store_name = f"store_{region_idx + 1}_{store_idx + 1}"
            ts_id = f"{region_name}__{store_name}"
            p_demand = float(rng.uniform(0.28, 0.45))
            mean_size = float(rng.uniform(5.0, 12.0))
            occ = rng.binomial(1, p_demand, size=total_len)
            sizes = rng.poisson(lam=mean_size, size=total_len) + 1
            mu = np.full(total_len, p_demand * (mean_size + 1.0), dtype=np.float64)
            y_all = (occ * sizes).astype(np.float64)
            extra_cols = {
                "region": [region_name] * total_len,
                "store": [store_name] * total_len,
            }

        elif name == "dgp_15_scale_crossover_panel":
            scale_mult = float(rng.uniform(0.75, 1.35))
            common_wave = 18.0 * np.sin(2.0 * np.pi * t_idx / 7.0) + 6.0 * np.cos(
                4.0 * np.pi * t_idx / 7.0
            )
            mu = np.maximum(5.0, scale_mult * (100.0 + 0.18 * t_idx + common_wave))
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.8, size=total_len))

        elif name == "dgp_16_monthly_trading_days":
            # Monthly series where business days per month drive volume
            days_in_month = dates.days_in_month.to_numpy(dtype=np.float64)
            base_daily = float(rng.uniform(3.5, 6.0))
            yearly_wave = 0.45 * np.sin(2.0 * np.pi * t_idx / 12.0 + phase)
            mu = np.maximum(10.0, days_in_month * (base_daily + 0.03 * t_idx + yearly_wave))
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.0, size=total_len))

        elif name == "dgp_17_missing_gaps_and_leading_zeros":
            base_level = float(rng.uniform(85.0, 130.0))
            mu = np.maximum(5.0, base_level + 0.15 * t_idx + 14.0 * weekly)
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.5, size=total_len))
            if i < max(1, eff_n // 2):
                # Inject 24 pre-launch leading zeros at start of history
                lead_z = min(24, max(6, eff_t // 5))
                y_all[:lead_z] = 0.0

        else:  # dgp_18_promo_lag_pullforward
            base_level = float(rng.uniform(100.0, 140.0))
            promo_flag = np.zeros(total_len, dtype=np.float64)
            for t_p in range(5, total_len, 9):
                promo_flag[t_p] = 1.0
            promo_lag1 = np.concatenate([[0.0], promo_flag[:-1]])
            mu = np.maximum(
                5.0,
                base_level + 0.10 * t_idx + 12.0 * weekly + 85.0 * promo_flag - 58.0 * promo_lag1,
            )
            y_all = np.maximum(0.0, mu + rng.normal(0.0, 2.2, size=total_len))
            extra_cols = {"promo_flag": promo_flag}

        hist_slice = slice(start_idx, eff_t)
        fut_slice = slice(eff_t, total_len)

        h_df = pd.DataFrame(
            {
                "ts_id": ts_id,
                "ds": dates[hist_slice],
                "y": np.round(y_all[hist_slice], 4),
            }
        )
        for col_name, col_vals in extra_cols.items():
            arr = np.asarray(col_vals)
            h_df[col_name] = arr[hist_slice]
        if (
            name == "dgp_17_missing_gaps_and_leading_zeros"
            and i < max(1, eff_n // 2)
            and len(h_df) > 50
        ):
            # Drop 2 interior timestamps to simulate missing date gaps
            h_df = h_df.drop(index=[35, 70]).reset_index(drop=True)
        hist_rows.append(h_df)

        if extra_cols:
            f_df = pd.DataFrame(
                {
                    "ts_id": ts_id,
                    "ds": dates[fut_slice],
                }
            )
            for col_name, col_vals in extra_cols.items():
                arr = np.asarray(col_vals)
                if col_name == spec.censored_col:
                    # Future dates are uncensored
                    f_df[col_name] = np.zeros(eff_h, dtype=np.int64)
                else:
                    f_df[col_name] = arr[fut_slice]
            fut_rows.append(f_df)

        o_df = pd.DataFrame(
            {
                "ts_id": ts_id,
                "ds": dates[fut_slice],
                "horizon_step": np.arange(1, eff_h + 1, dtype=np.int64),
                "oracle_yhat": np.round(mu[fut_slice], 4),
                "y_realized": np.round(y_all[fut_slice], 4),
            }
        )
        oracle_rows.append(o_df)

    df = pd.concat(hist_rows, ignore_index=True)
    future_df = pd.concat(fut_rows, ignore_index=True) if fut_rows else pd.DataFrame()
    oracle_future_df = pd.concat(oracle_rows, ignore_index=True)
    return BenchmarkDataset(
        spec=spec,
        df=df,
        future_df=future_df,
        oracle_future_df=oracle_future_df,
    )


def _main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``python -m scale_forecasting.benchmarks [--list] [--name <dgp>]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="scale_forecasting.benchmarks",
        description="Inspect or generate the 12 canonical Synthetic DGP Benchmark panels.",
    )
    parser.add_argument("--list", action="store_true", help="List all 12 canonical DGP benchmarks")
    parser.add_argument("--name", help="Generate and summarize one benchmark panel by name")
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")
    args = parser.parse_args(argv)

    if args.list or not args.name:
        cat = benchmark_catalog()
        if args.json:
            print(json.dumps(cat.to_dict(orient="records"), indent=2))
        else:
            print(
                cat[["name", "expected_regime", "true_transform", "title"]].to_string(index=False)
            )
        return 0

    ds = load_benchmark(args.name)
    payload = {
        "spec": ds.spec.to_dict(),
        "hist_rows": len(ds.df),
        "future_rows": len(ds.future_df),
        "oracle_rows": len(ds.oracle_future_df),
        "columns": list(ds.df.columns),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Benchmark : {ds.spec.name} ({ds.spec.title})")
        print(f"Regime    : {ds.spec.expected_regime} | Transform: {ds.spec.true_transform}")
        print(
            f"Shape     : {len(ds.df)} hist rows ({ds.spec.n_series} series x H={ds.spec.horizon})"
        )
        print(f"Columns   : {', '.join(ds.df.columns)}")
        print(f"Actions   : {', '.join(ds.spec.expected_autofocus_actions)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
