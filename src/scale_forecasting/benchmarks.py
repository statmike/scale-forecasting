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

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BENCHMARK_NAMES",
    "BENCHMARK_SPECS",
    "PUBLIC_BENCHMARK_NAMES",
    "PUBLIC_BENCHMARK_SPECS",
    "BenchmarkDataset",
    "BenchmarkSpec",
    "PublicBenchmarkSpec",
    "benchmark_catalog",
    "fetch_public_benchmark",
    "list_benchmarks",
    "list_public_benchmarks",
    "load_benchmark",
    "public_benchmark_catalog",
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


# ---------------------------------------------------------------------------
# Public Real-World Open-Data Benchmark Suite (Zero Data Rehosting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicBenchmarkSpec:
    """Metadata, citation, license, and source URLs for a public open-data benchmark.

    Raw external datasets are **never** rehosted in this repository; `fetch_public_benchmark`
    downloads them on demand into an ephemeral local cache
    (`/tmp/scale_forecasting_public_benchmarks`) and converts them into a standard
    :class:`BenchmarkDataset` with held-out ground truth in ``oracle_future_df``.
    """

    name: str
    title: str
    domain: str
    freq: str
    horizon: int
    default_n_series: int
    license: str
    citation: str
    homepage_url: str
    source_urls: tuple[str, ...]
    future_covariates: tuple[str, ...] = ()
    past_covariates: tuple[str, ...] = ()
    static_covariates: tuple[str, ...] = ()
    hierarchy_id_cols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        raw = asdict(self)
        for k, v in list(raw.items()):
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw


PUBLIC_BENCHMARK_NAMES: tuple[str, ...] = (
    "m5_walmart",
    "favorita_grocery",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "monash_tourism_monthly",
    "monash_hospital",
    "monash_car_parts",
    "etth1",
)

PUBLIC_BENCHMARK_SPECS: dict[str, PublicBenchmarkSpec] = {
    "m5_walmart": PublicBenchmarkSpec(
        name="m5_walmart",
        title="M5 Forecasting — Accuracy (Walmart Hierarchical Retail Demand)",
        domain="Retail / Hierarchical Intermittent & Smooth SKU Demand",
        freq="D",
        horizon=28,
        default_n_series=24,
        license="M5 Competition Academic & Research Terms (MOFC / Kaggle)",
        citation=(
            "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). "
            "M5 accuracy competition: Results, findings, and conclusions. "
            "International Journal of Forecasting, 38(4), 1346-1364."
        ),
        homepage_url="https://github.com/Mcompetitions/M5-methods",
        source_urls=(
            "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
            "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv",
            "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv",
        ),
        future_covariates=("snap_active", "is_event", "sell_price"),
        static_covariates=("cat_id", "dept_id", "store_id", "state_id"),
        hierarchy_id_cols=("state_id", "store_id", "cat_id"),
    ),
    "favorita_grocery": PublicBenchmarkSpec(
        name="favorita_grocery",
        title="Corporación Favorita Store Sales (Promotions, Holidays & Oil Covariates)",
        domain="Grocery Retail / Promotional & Macroeconomic Covariates",
        freq="D",
        horizon=16,
        default_n_series=24,
        license="Competition Research Terms (Corporación Favorita / Kaggle)",
        citation=(
            "Corporación Favorita Grocery Sales Forecasting Benchmark (2018); "
            "adopted in Lim et al. (2021) Temporal Fusion Transformers, Int. J. Forecasting."
        ),
        homepage_url="https://www.kaggle.com/competitions/store-sales-time-series-forecasting",
        source_urls=(
            "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/train.csv",
            "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/stores.csv",
            "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/oil.csv",
            "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/holidays_events.csv",
        ),
        future_covariates=("onpromotion", "is_holiday"),
        past_covariates=("dcoilwtico",),
        static_covariates=("state", "city", "store_id", "family"),
        hierarchy_id_cols=("state", "store_id"),
    ),
    "m4_daily": PublicBenchmarkSpec(
        name="m4_daily",
        title="M4 Competition — Daily Series (Finance, Macro, Micro & Demographic)",
        domain="Multi-Domain Daily (M4 Competition)",
        freq="D",
        horizon=14,
        default_n_series=24,
        license="CC-BY-4.0 (Mcompetitions)",
        citation=(
            "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "
            "The M4 Competition: 100,000 time series and 61 forecasting methods. "
            "International Journal of Forecasting, 36(1), 54-74."
        ),
        homepage_url="https://github.com/Mcompetitions/M4-methods",
        source_urls=(
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv",
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Daily-test.csv",
        ),
    ),
    "m4_weekly": PublicBenchmarkSpec(
        name="m4_weekly",
        title="M4 Competition — Weekly Series",
        domain="Multi-Domain Weekly (M4 Competition)",
        freq="W",
        horizon=13,
        default_n_series=24,
        license="CC-BY-4.0 (Mcompetitions)",
        citation=(
            "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "
            "The M4 Competition: 100,000 time series and 61 forecasting methods. "
            "International Journal of Forecasting, 36(1), 54-74."
        ),
        homepage_url="https://github.com/Mcompetitions/M4-methods",
        source_urls=(
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Weekly-train.csv",
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Weekly-test.csv",
        ),
    ),
    "m4_monthly": PublicBenchmarkSpec(
        name="m4_monthly",
        title="M4 Competition — Monthly Series",
        domain="Multi-Domain Monthly (M4 Competition)",
        freq="MS",
        horizon=18,
        default_n_series=24,
        license="CC-BY-4.0 (Mcompetitions)",
        citation=(
            "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "
            "The M4 Competition: 100,000 time series and 61 forecasting methods. "
            "International Journal of Forecasting, 36(1), 54-74."
        ),
        homepage_url="https://github.com/Mcompetitions/M4-methods",
        source_urls=(
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv",
        ),
    ),
    "monash_tourism_monthly": PublicBenchmarkSpec(
        name="monash_tourism_monthly",
        title="Monash Archive — Tourism Monthly (Athanasopoulos et al. 2011)",
        domain="Tourism / Strong Multiplicative Seasonality",
        freq="MS",
        horizon=24,
        default_n_series=24,
        license="CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive)",
        citation=(
            "Athanasopoulos, G., Hyndman, R. J., Song, H., & Wu, D. C. (2011). "
            "The tourism forecasting competition. Int. J. Forecasting, 27(3), 822-844; "
            "Godahewa et al. (2021) Monash Time Series Forecasting Archive, NeurIPS."
        ),
        homepage_url="https://zenodo.org/records/4656096",
        source_urls=(
            "https://zenodo.org/api/records/4656096/files/tourism_monthly_dataset.zip/content",
        ),
    ),
    "monash_hospital": PublicBenchmarkSpec(
        name="monash_hospital",
        title="Monash Archive — Hospital Patient Counts (Hyndman et al. 2008)",
        domain="Healthcare / Monthly Patient Count Series",
        freq="MS",
        horizon=12,
        default_n_series=24,
        license="CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive)",
        citation=(
            "Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). "
            "Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks."
        ),
        homepage_url="https://zenodo.org/records/4656014",
        source_urls=("https://zenodo.org/api/records/4656014/files/hospital_dataset.zip/content",),
    ),
    "monash_car_parts": PublicBenchmarkSpec(
        name="monash_car_parts",
        title="Monash Archive — Car Parts Intermittent Demand (Hyndman & Snyder 2008)",
        domain="Automotive / Sparse Intermittent Spare Parts",
        freq="MS",
        horizon=12,
        default_n_series=24,
        license="CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive)",
        citation=(
            "Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). "
            "Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks."
        ),
        homepage_url="https://zenodo.org/records/4656021",
        source_urls=(
            "https://zenodo.org/api/records/4656021/files/car_parts_dataset_without_missing_values.zip/content",
        ),
    ),
    "etth1": PublicBenchmarkSpec(
        name="etth1",
        title="Electricity Transformer Temperature Hourly (ETTh1, Informer Benchmark)",
        domain="Energy / Hourly Transformer Load & Oil Temperature",
        freq="h",
        horizon=24,
        default_n_series=7,
        license="Apache-2.0 (zhouhaoyi/ETDataset)",
        citation=(
            "Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). "
            "Informer: Beyond Efficient Transformer for Long Sequence Time-Series "
            "Forecasting. AAAI."
        ),
        homepage_url="https://github.com/zhouhaoyi/ETDataset",
        source_urls=(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        ),
    ),
}


def list_public_benchmarks() -> list[str]:
    """Return all registered public open-data benchmark names."""
    return list(PUBLIC_BENCHMARK_NAMES)


def public_benchmark_catalog() -> pd.DataFrame:
    """Return a tidy DataFrame summarizing all registered public open-data benchmarks."""
    rows = [spec.to_dict() for spec in PUBLIC_BENCHMARK_SPECS.values()]
    return pd.DataFrame(rows)


def _download_cached_bytes(url: str, dest: Any) -> bytes:
    """Download ``url`` into ``dest`` if not already cached and return its raw bytes."""
    import urllib.request
    from pathlib import Path

    path = Path(dest)
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "scale-forecasting/1.2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return data


def _parse_monash_tsf_zip(
    zip_bytes: bytes,
    *,
    pub_spec: PublicBenchmarkSpec,
    n_series: int,
    history_limit: int,
) -> BenchmarkDataset:
    """Parse a Monash `.tsf` zip archive into a :class:`BenchmarkDataset`."""
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tsf_name = next(name for name in zf.namelist() if name.endswith(".tsf"))
        lines = zf.read(tsf_name).decode("latin-1").splitlines()

    data_idx = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("@data"))
    hist_frames: list[pd.DataFrame] = []
    oracle_frames: list[pd.DataFrame] = []
    h = pub_spec.horizon
    picked = 0

    for line in lines[data_idx + 1 :]:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split(":")
        if len(parts) < 2:
            continue
        ts_id = parts[0].strip()
        start_str = parts[1].strip().split()[0] if len(parts) >= 3 else "2000-01-01"
        vals = [
            float(v)
            for v in parts[-1].split(",")
            if v.strip() and v.strip() != "?" and v.strip().lower() != "nan"
        ]
        if len(vals) < h + 24:
            continue
        vals_arr = np.asarray(vals[-(history_limit + h) :], dtype=np.float64)
        # Skip degenerate all-zero series on car_parts so every sampled series has nonzero demand
        if np.count_nonzero(vals_arr[:-h] > 0) < 4:
            continue
        dates = pd.date_range(start=start_str, periods=len(vals), freq=pub_spec.freq)[
            -(len(vals_arr)) :
        ]
        hist_vals = vals_arr[:-h]
        test_vals = vals_arr[-h:]
        hist_dates = dates[:-h]
        test_dates = dates[-h:]

        hist_frames.append(
            pd.DataFrame({"ts_id": ts_id, "ds": hist_dates, "y": np.round(hist_vals, 4)})
        )
        oracle_frames.append(
            pd.DataFrame(
                {
                    "ts_id": ts_id,
                    "ds": test_dates,
                    "horizon_step": np.arange(1, h + 1, dtype=np.int64),
                    "oracle_yhat": np.round(test_vals, 4),
                    "y_realized": np.round(test_vals, 4),
                }
            )
        )
        picked += 1
        if picked >= n_series:
            break

    df = pd.concat(hist_frames, ignore_index=True)
    oracle_df = pd.concat(oracle_frames, ignore_index=True)
    spec = BenchmarkSpec(
        name=pub_spec.name,
        title=pub_spec.title,
        description=f"{pub_spec.citation} ({pub_spec.license})",
        freq=pub_spec.freq,
        horizon=h,
        n_series=picked,
        history_length=int(len(df) // max(1, picked)),
        expected_regime="intermittent" if "car_parts" in pub_spec.name else "smooth",
        non_negative=True,
        integer_valued="car_parts" in pub_spec.name or "hospital" in pub_spec.name,
    )
    return BenchmarkDataset(spec=spec, df=df, future_df=pd.DataFrame(), oracle_future_df=oracle_df)


def fetch_public_benchmark(
    name: str,
    *,
    n_series: int | None = None,
    history_limit: int = 365,
    cache_dir: str | None = None,
) -> BenchmarkDataset:
    """Download and prepare a public real-world benchmark on demand (zero repo data rehosting).

    Fetches official public files into ``cache_dir`` (default
    ``/tmp/scale_forecasting_public_benchmarks/<name>``) and splits each series into historical
    observations (`dataset.df`), future exogenous covariates (`dataset.future_df`), and held-out
    ground-truth test actuals (`dataset.oracle_future_df`).
    """
    import csv
    import io
    import urllib.request
    from pathlib import Path

    if name not in PUBLIC_BENCHMARK_SPECS:
        raise ValueError(
            f"Unknown public benchmark {name!r}; expected one of {PUBLIC_BENCHMARK_NAMES}"
        )
    pub_spec = PUBLIC_BENCHMARK_SPECS[name]
    eff_n = int(n_series or pub_spec.default_n_series)
    h = pub_spec.horizon
    root = Path(cache_dir or "/tmp/scale_forecasting_public_benchmarks") / name
    root.mkdir(parents=True, exist_ok=True)

    # 1. Monash Archive (.tsf inside .zip on Zenodo)
    if name.startswith("monash_"):
        zip_bytes = _download_cached_bytes(pub_spec.source_urls[0], root / f"{name}.zip")
        return _parse_monash_tsf_zip(
            zip_bytes, pub_spec=pub_spec, n_series=eff_n, history_limit=history_limit
        )

    # 2. ETTh1 (zhouhaoyi/ETDataset on GitHub)
    if name == "etth1":
        csv_bytes = _download_cached_bytes(pub_spec.source_urls[0], root / "ETTh1.csv")
        raw_df = pd.read_csv(io.BytesIO(csv_bytes))
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df = raw_df.tail(history_limit + h).reset_index(drop=True)
        series_cols = [c for c in raw_df.columns if c != "date"][:eff_n]
        hist_frames: list[pd.DataFrame] = []
        oracle_frames: list[pd.DataFrame] = []
        for col in series_cols:
            vals = pd.to_numeric(raw_df[col], errors="coerce").to_numpy(dtype=np.float64)
            dates = raw_df["date"]
            hist_frames.append(
                pd.DataFrame({"ts_id": col, "ds": dates.iloc[:-h], "y": np.round(vals[:-h], 4)})
            )
            oracle_frames.append(
                pd.DataFrame(
                    {
                        "ts_id": col,
                        "ds": dates.iloc[-h:].to_numpy(),
                        "horizon_step": np.arange(1, h + 1, dtype=np.int64),
                        "oracle_yhat": np.round(vals[-h:], 4),
                        "y_realized": np.round(vals[-h:], 4),
                    }
                )
            )
        df = pd.concat(hist_frames, ignore_index=True)
        oracle_df = pd.concat(oracle_frames, ignore_index=True)
        spec = BenchmarkSpec(
            name=name,
            title=pub_spec.title,
            description=f"{pub_spec.citation} ({pub_spec.license})",
            freq=pub_spec.freq,
            horizon=h,
            n_series=len(series_cols),
            history_length=history_limit,
            expected_regime="erratic",
            non_negative=False,
        )
        return BenchmarkDataset(
            spec=spec, df=df, future_df=pd.DataFrame(), oracle_future_df=oracle_df
        )

    # 3. M4 Competition (Daily, Weekly, Monthly from Mcompetitions/M4-methods)
    if name.startswith("m4_"):
        train_url, test_url = pub_spec.source_urls
        train_cache = root / f"{name}_train_subset.csv"
        test_cache = root / f"{name}_test_subset.csv"
        if not train_cache.is_file() or not test_cache.is_file():
            for url, out_path in ((train_url, train_cache), (test_url, test_cache)):
                req = urllib.request.Request(url, headers={"User-Agent": "scale-forecasting/1.2"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    lines_collected: list[str] = []
                    for raw_line in resp:
                        lines_collected.append(raw_line.decode("utf-8", errors="ignore"))
                        if len(lines_collected) >= max(eff_n + 5, 105):
                            break
                out_path.write_text("".join(lines_collected), encoding="utf-8")

        hist_frames = []
        oracle_frames = []
        with (
            train_cache.open("r", encoding="utf-8") as f_tr,
            test_cache.open("r", encoding="utf-8") as f_te,
        ):
            r_tr = csv.reader(f_tr)
            r_te = csv.reader(f_te)
            next(r_tr, None)
            next(r_te, None)
            for idx, (row_tr, row_te) in enumerate(zip(r_tr, r_te, strict=False)):
                if idx >= eff_n:
                    break
                ts_id = row_tr[0].strip()
                tr_vals = [float(v) for v in row_tr[1:] if v.strip()]
                te_vals = [float(v) for v in row_te[1 : h + 1] if v.strip()]
                if len(te_vals) < h or len(tr_vals) < 20:
                    continue
                tr_vals = tr_vals[-history_limit:]
                total_len = len(tr_vals) + h
                m4_dates = pd.date_range("2015-01-01", periods=total_len, freq=pub_spec.freq)
                hist_frames.append(
                    pd.DataFrame(
                        {
                            "ts_id": ts_id,
                            "ds": m4_dates[:-h],
                            "y": np.round(np.asarray(tr_vals, dtype=np.float64), 4),
                        }
                    )
                )
                oracle_frames.append(
                    pd.DataFrame(
                        {
                            "ts_id": ts_id,
                            "ds": m4_dates[-h:],
                            "horizon_step": np.arange(1, h + 1, dtype=np.int64),
                            "oracle_yhat": np.round(np.asarray(te_vals, dtype=np.float64), 4),
                            "y_realized": np.round(np.asarray(te_vals, dtype=np.float64), 4),
                        }
                    )
                )
        df = pd.concat(hist_frames, ignore_index=True)
        oracle_df = pd.concat(oracle_frames, ignore_index=True)
        spec = BenchmarkSpec(
            name=name,
            title=pub_spec.title,
            description=f"{pub_spec.citation} ({pub_spec.license})",
            freq=pub_spec.freq,
            horizon=h,
            n_series=len(hist_frames),
            history_length=int(len(df) // max(1, len(hist_frames))),
            expected_regime="smooth",
            non_negative=True,
        )
        return BenchmarkDataset(
            spec=spec, df=df, future_df=pd.DataFrame(), oracle_future_df=oracle_df
        )

    # 4. M5 Walmart Hierarchical Retail (calendar + sales_train_evaluation + sell_prices)
    if name == "m5_walmart":
        cal_url, eval_url, price_url = pub_spec.source_urls
        cal_bytes = _download_cached_bytes(cal_url, root / "calendar.csv")
        cal_df = pd.read_csv(io.BytesIO(cal_bytes))
        cal_df["ds"] = pd.to_datetime(cal_df["date"])
        cal_df["is_event"] = cal_df["event_name_1"].notna().astype(np.int64)

        eval_path = root / "sales_train_evaluation.csv"
        if not eval_path.is_file() or eval_path.stat().st_size == 0:
            _download_cached_bytes(eval_url, eval_path)
        price_path = root / "sell_prices.csv"
        if not price_path.is_file() or price_path.stat().st_size == 0:
            _download_cached_bytes(price_url, price_path)

        # Stratified selection across CA_1, CA_2, TX_1, WI_1 and FOODS, HOUSEHOLD, HOBBIES
        sales_df = pd.read_csv(eval_path)
        d_cols = [c for c in sales_df.columns if c.startswith("d_")][-(history_limit + h) :]
        sales_df["_total_vol"] = sales_df[d_cols[-90:]].sum(axis=1)
        active_df = sales_df[sales_df["_total_vol"] >= 30].copy()
        per_group = max(1, math.ceil(eff_n / 6))
        picked_rows = (
            active_df.sort_values("_total_vol", ascending=False)
            .groupby(["state_id", "cat_id"], as_index=False)
            .head(per_group)
            .head(eff_n)
        )
        wanted_pairs = set(zip(picked_rows["store_id"], picked_rows["item_id"], strict=False))

        # Filter sell_prices for selected (store_id, item_id) pairs
        price_chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(price_path, chunksize=250_000):
            mask = chunk["store_id"].isin(picked_rows["store_id"].unique()) & chunk["item_id"].isin(
                picked_rows["item_id"].unique()
            )
            if mask.any():
                price_chunks.append(chunk.loc[mask])
        prices_sub = (
            pd.concat(price_chunks, ignore_index=True)
            if price_chunks
            else pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
        )
        _ = wanted_pairs

        cal_sub = cal_df[
            ["d", "ds", "wm_yr_wk", "is_event", "snap_CA", "snap_TX", "snap_WI"]
        ].set_index("d")

        hist_frames = []
        fut_frames: list[pd.DataFrame] = []
        oracle_frames = []
        for _, srow in picked_rows.iterrows():
            ts_id = str(srow["id"]).replace("_evaluation", "")
            state_id = str(srow["state_id"])
            store_id = str(srow["store_id"])
            cat_id = str(srow["cat_id"])
            dept_id = str(srow["dept_id"])
            item_id = str(srow["item_id"])
            snap_col = f"snap_{state_id}" if f"snap_{state_id}" in cal_sub.columns else "snap_CA"

            s_cal = cal_sub.loc[d_cols].copy()
            s_cal["y"] = srow[d_cols].to_numpy(dtype=np.float64)
            s_cal["snap_active"] = s_cal[snap_col].to_numpy(dtype=np.int64)
            item_prices = prices_sub[
                (prices_sub["store_id"] == store_id) & (prices_sub["item_id"] == item_id)
            ][["wm_yr_wk", "sell_price"]].drop_duplicates("wm_yr_wk")
            s_cal = s_cal.merge(item_prices, on="wm_yr_wk", how="left")
            s_cal["sell_price"] = s_cal["sell_price"].ffill().bfill().fillna(1.0)
            s_cal["ts_id"] = ts_id
            s_cal["state_id"] = state_id
            s_cal["store_id"] = store_id
            s_cal["cat_id"] = cat_id
            s_cal["dept_id"] = dept_id

            cols_keep = [
                "ts_id",
                "ds",
                "y",
                "snap_active",
                "is_event",
                "sell_price",
                "state_id",
                "store_id",
                "cat_id",
                "dept_id",
            ]
            hist_part = s_cal.iloc[:-h][cols_keep].copy()
            test_part = s_cal.iloc[-h:].copy()
            hist_frames.append(hist_part)
            fut_frames.append(
                test_part[["ts_id", "ds", "snap_active", "is_event", "sell_price"]].copy()
            )
            oracle_frames.append(
                pd.DataFrame(
                    {
                        "ts_id": ts_id,
                        "ds": test_part["ds"].to_numpy(),
                        "horizon_step": np.arange(1, h + 1, dtype=np.int64),
                        "oracle_yhat": np.round(test_part["y"].to_numpy(dtype=np.float64), 4),
                        "y_realized": np.round(test_part["y"].to_numpy(dtype=np.float64), 4),
                    }
                )
            )
        df = pd.concat(hist_frames, ignore_index=True)
        fut_df = pd.concat(fut_frames, ignore_index=True)
        oracle_df = pd.concat(oracle_frames, ignore_index=True)
        spec = BenchmarkSpec(
            name=name,
            title=pub_spec.title,
            description=f"{pub_spec.citation} ({pub_spec.license})",
            freq=pub_spec.freq,
            horizon=h,
            n_series=len(hist_frames),
            history_length=history_limit,
            expected_regime="smooth",
            non_negative=True,
            integer_valued=True,
            true_covariates=pub_spec.future_covariates,
            static_covariates=pub_spec.static_covariates,
            hierarchy_id_cols=pub_spec.hierarchy_id_cols,
        )
        return BenchmarkDataset(spec=spec, df=df, future_df=fut_df, oracle_future_df=oracle_df)

    # 5. Corporación Favorita Grocery Sales (train + stores + oil + holidays_events)
    train_url, stores_url, oil_url, hol_url = pub_spec.source_urls
    stores_df = pd.read_csv(io.BytesIO(_download_cached_bytes(stores_url, root / "stores.csv")))
    oil_df = pd.read_csv(io.BytesIO(_download_cached_bytes(oil_url, root / "oil.csv")))
    hol_df = pd.read_csv(io.BytesIO(_download_cached_bytes(hol_url, root / "holidays_events.csv")))

    train_path = root / "train.csv"
    if not train_path.is_file() or train_path.stat().st_size == 0:
        _download_cached_bytes(train_url, train_path)

    target_stores = [1, 2, 3, 44, 45, 47, 48, 49]
    target_families = [
        "GROCERY I",
        "BEVERAGES",
        "PRODUCE",
        "CLEANING",
        "DAIRY",
        "BREAD/BAKERY",
    ]
    sub_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        train_path,
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        chunksize=500_000,
    ):
        m = (
            chunk["store_nbr"].isin(target_stores)
            & chunk["family"].isin(target_families)
            & (chunk["date"] >= "2016-06-01")
        )
        if m.any():
            sub_chunks.append(chunk.loc[m])
    raw_fav = pd.concat(sub_chunks, ignore_index=True)
    raw_fav["ds"] = pd.to_datetime(raw_fav["date"])
    oil_df["ds"] = pd.to_datetime(oil_df["date"])
    oil_sub = oil_df[["ds", "dcoilwtico"]].drop_duplicates("ds").sort_values("ds")
    oil_sub["dcoilwtico"] = oil_sub["dcoilwtico"].ffill().bfill().fillna(45.0)

    hol_df["ds"] = pd.to_datetime(hol_df["date"])
    nat_hols = set(
        hol_df.loc[hol_df["transferred"].fillna(False) == False, "ds"].unique()  # noqa: E712
    )

    raw_fav = raw_fav.merge(stores_df[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    raw_fav = raw_fav.merge(oil_sub, on="ds", how="left")
    raw_fav["dcoilwtico"] = raw_fav["dcoilwtico"].ffill().bfill().fillna(45.0)
    raw_fav["is_holiday"] = raw_fav["ds"].isin(nat_hols).astype(np.int64)
    raw_fav["store_id"] = "store_" + raw_fav["store_nbr"].astype(str)
    raw_fav["ts_id"] = (
        raw_fav["store_id"] + "__" + raw_fav["family"].str.replace(" ", "_").str.replace("/", "_")
    )

    unique_ids = sorted(raw_fav["ts_id"].unique())[:eff_n]
    raw_fav = raw_fav[raw_fav["ts_id"].isin(unique_ids)].sort_values(["ts_id", "ds"])

    hist_frames = []
    fut_frames = []
    oracle_frames = []
    for raw_tid, grp in raw_fav.groupby("ts_id", sort=False):
        ts_id = str(raw_tid)
        g = grp.tail(history_limit + h).copy()
        if len(g) < h + 30:
            continue
        g["y"] = np.round(pd.to_numeric(g["sales"], errors="coerce").fillna(0.0).to_numpy(), 4)
        g["onpromotion"] = pd.to_numeric(g["onpromotion"], errors="coerce").fillna(0.0).to_numpy()
        cols_keep = [
            "ts_id",
            "ds",
            "y",
            "onpromotion",
            "is_holiday",
            "dcoilwtico",
            "state",
            "city",
            "store_id",
            "family",
        ]
        hist_part = g.iloc[:-h][cols_keep].copy()
        test_part = g.iloc[-h:].copy()
        hist_frames.append(hist_part)
        fut_frames.append(test_part[["ts_id", "ds", "onpromotion", "is_holiday"]].copy())
        oracle_frames.append(
            pd.DataFrame(
                {
                    "ts_id": str(ts_id),
                    "ds": test_part["ds"].to_numpy(),
                    "horizon_step": np.arange(1, h + 1, dtype=np.int64),
                    "oracle_yhat": np.round(test_part["y"].to_numpy(dtype=np.float64), 4),
                    "y_realized": np.round(test_part["y"].to_numpy(dtype=np.float64), 4),
                }
            )
        )

    df = pd.concat(hist_frames, ignore_index=True)
    fut_df = pd.concat(fut_frames, ignore_index=True)
    oracle_df = pd.concat(oracle_frames, ignore_index=True)
    spec = BenchmarkSpec(
        name=name,
        title=pub_spec.title,
        description=f"{pub_spec.citation} ({pub_spec.license})",
        freq=pub_spec.freq,
        horizon=h,
        n_series=len(hist_frames),
        history_length=int(len(df) // max(1, len(hist_frames))),
        expected_regime="smooth",
        non_negative=True,
        true_covariates=pub_spec.future_covariates,
        static_covariates=pub_spec.static_covariates,
        hierarchy_id_cols=pub_spec.hierarchy_id_cols,
    )
    return BenchmarkDataset(spec=spec, df=df, future_df=fut_df, oracle_future_df=oracle_df)


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
