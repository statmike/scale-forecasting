"""Evolutionary targets (`# EVOLVE-BLOCK`) built on top of `Forecaster.auto()`.

Defines the four bounded code-synthesis targets for ``scale_forecasting.evolve``:

1. ``"features"`` (:func:`synthesize_evolved_features`) — synthesizes causal lag, rolling,
   harmonic, and exogenous interaction covariates (``evolved_feat_*``) strictly from past and
   contemporaneous context without future target lookahead.
2. ``"ensemble"`` (:func:`compute_evolved_weights`) — synthesizes custom out-of-fold (OOF)
   blending weights across the cached model bank produced by ``Forecaster.auto()``.
3. ``"model"`` (:class:`EvolvedForecaster`) — synthesizes a custom
   :class:`~scale_forecasting.models.base_model.BaseModel` subclass using
   ``numpy``/``scipy``/``statsmodels`` primitives.
4. ``"policy"`` (:func:`evolve_policy_weights`) — tunes :class:`AutoPolicyWeights` decision
   thresholds and action confidence priors for :mod:`scale_forecasting.auto_policy`.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..auto_policy import evolved_policy_thresholds
from ..errors import ConfigError
from ..models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext

__all__ = [
    "EVOLVE_BLOCK_END",
    "EVOLVE_BLOCK_START",
    "EVOLVE_TARGETS",
    "OVERFIT_LOCKDOWN_CANDIDATES",
    "AutoPolicyWeights",
    "EvolvedForecaster",
    "TargetSpec",
    "build_mutation_prompt",
    "compute_evolved_weights",
    "deterministic_mock_candidates",
    "evolve_policy_weights",
    "extract_evolve_block",
    "get_target_seed",
    "inject_evolve_block",
    "synthesize_evolved_features",
]

EVOLVE_TARGETS: tuple[str, ...] = ("features", "ensemble", "model", "policy")
EVOLVE_BLOCK_START: str = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END: str = "# EVOLVE-BLOCK-END"


def _default_action_weights() -> dict[str, tuple[float, float]]:
    raw = evolved_policy_thresholds().get("action_weights", {})
    out: dict[str, tuple[float, float]] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(v, (tuple, list)) and len(v) == 2:
                out[str(k)] = (float(v[0]), float(v[1]))
    return out


@dataclass(frozen=True)
class AutoPolicyWeights:
    """Typed container of evolutionary thresholds for :mod:`scale_forecasting.auto_policy`."""

    intermittent_share_min: float = 0.165
    log1p_level_vol_corr_min: float = 0.635
    log1p_growth_ratio_min: float = 1.336
    resid_hetero_min: float = 0.30
    structural_break_share_min: float = 0.251
    outlier_spike_share_min: float = 0.195
    direct_horizon_decay_ratio_min: float = 1.323
    resid_acf1_escalate_min: float = 0.309
    fva_escalate_min_pct: float = 3.077
    white_noise_growth_tol: float = 0.179
    white_noise_acf_season_max: float = 0.17
    require_white_noise_stop: bool = True
    pass_decay_factor: float = 0.70
    first_light_include_regression_lags: bool = True
    first_light_selection_rule: str = "best_single"
    tree_n_estimators_fast: int = 60
    action_weights: dict[str, tuple[float, float]] = field(default_factory=_default_action_weights)

    @classmethod
    def from_defaults(cls) -> AutoPolicyWeights:
        """Construct :class:`AutoPolicyWeights` from ``auto_policy.evolved_policy_thresholds()``."""
        return cls.from_dict(evolved_policy_thresholds())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AutoPolicyWeights:
        """Construct :class:`AutoPolicyWeights` from a plain threshold dictionary."""
        valid_fields = set(cls.__dataclass_fields__)
        filtered: dict[str, Any] = {}
        for k, v in data.items():
            if k not in valid_fields:
                continue
            if k == "action_weights" and isinstance(v, Mapping):
                filtered[k] = {
                    str(ak): (float(av[0]), float(av[1]))
                    for ak, av in v.items()
                    if isinstance(av, (tuple, list)) and len(av) == 2
                }
            else:
                filtered[k] = v
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy weights to a dictionary compatible with ``auto_policy``."""
        out = asdict(self)
        out["first_light_ensemble_strategies"] = ["inverse_error", "mean"]
        return out

    def with_updates(self, **kwargs: Any) -> AutoPolicyWeights:
        """Return a copy of ``self`` with updated field values."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class TargetSpec:
    """Specification and seed program for one `# EVOLVE-BLOCK` evolutionary target."""

    target: str
    entrypoint: str
    description: str
    seed_code: str
    system_instruction: str
    mutation_prompt_template: str


# ---------------------------------------------------------------------------
# Reference seed implementations (used directly and embedded as seed_code)
# ---------------------------------------------------------------------------


def synthesize_evolved_features(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Synthesize causal ``evolved_feat_*`` columns without future target lookahead."""
    out = df.copy()
    id_col = str((profile or {}).get("ts_id_col", "ts_id"))
    if id_col not in out.columns:
        id_col = "ts_id" if "ts_id" in out.columns else out.columns[0]

    # EVOLVE-BLOCK-START
    step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
    out["evolved_feat_trend_idx"] = np.log1p(step_idx)
    out["evolved_feat_sin_w"] = np.sin(2.0 * np.pi * step_idx / 7.0)
    out["evolved_feat_cos_w"] = np.cos(2.0 * np.pi * step_idx / 7.0)

    reserved = {id_col, "ts_id", "ds", "date", "forecast_date", "y", "y_true", "yhat"}
    exog_cols = [
        c
        for c in out.columns
        if c not in reserved
        and not str(c).startswith("evolved_feat_")
        and pd.api.types.is_numeric_dtype(out[c])
    ]
    for col in exog_cols[:3]:
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[f"evolved_feat_{col}_lag1"] = (
            vals.groupby(out[id_col], sort=False).shift(1).fillna(vals)
        )
        out[f"evolved_feat_{col}_roll3"] = (
            vals.groupby(out[id_col], sort=False)
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
            .astype(float)
        )
    # EVOLVE-BLOCK-END
    return out


def compute_evolved_weights(
    oof_df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute global (`dict[str, float]`) or per-series (`dict[str, dict[str, float]]`) weights."""
    _ = profile
    if oof_df.empty or "model_type" not in oof_df.columns:
        return {}
    work = oof_df.copy()
    date_col = (
        "ds"
        if "ds" in work.columns
        else ("forecast_date" if "forecast_date" in work.columns else "")
    )
    per_series_weights: dict[str, dict[str, float]] = {}

    # EVOLVE-BLOCK-START
    scores: dict[str, float] = {}
    for model_type, grp in work.groupby("model_type", sort=False):
        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(yhat)
        if not mask.any():
            continue
        abs_err = np.abs(y_true[mask] - yhat[mask])
        denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
        wape = float(np.sum(abs_err)) / denom
        scores[str(model_type)] = 1.0 / max(1e-6, wape**2)
    # EVOLVE-BLOCK-END

    if per_series_weights:
        return per_series_weights

    total = sum(v for v in scores.values() if math.isfinite(v) and v > 0.0)
    if total <= 0.0:
        models = sorted(work["model_type"].astype(str).unique())
        global_w = {m: 1.0 / len(models) for m in models} if models else {}
    else:
        global_w = {k: float(v) / total for k, v in scores.items() if math.isfinite(v) and v > 0.0}

    if "ts_id" not in work.columns or not date_col or work["ts_id"].nunique() < 2 or not global_w:
        return global_w

    # Fold-consistent per-series adaptive shrinkage around the global champion
    model_mean_wape: dict[str, float] = {}
    for m_name, grp_m in work.groupby("model_type", sort=False):
        s_losses: list[float] = []
        for _, grp_s in grp_m.groupby("ts_id", sort=False):
            yt_s = pd.to_numeric(grp_s["y_true"], errors="coerce").to_numpy(dtype=float)
            yh_s = pd.to_numeric(grp_s["yhat"], errors="coerce").to_numpy(dtype=float)
            mk = np.isfinite(yt_s) & np.isfinite(yh_s)
            if mk.any():
                s_losses.append(
                    float(np.sum(np.abs(yt_s[mk] - yh_s[mk])))
                    / max(1e-6, float(np.sum(np.abs(yt_s[mk]))))
                )
        if s_losses:
            model_mean_wape[str(m_name)] = float(np.mean(s_losses))
    champ_m = (
        min(model_mean_wape, key=lambda k: model_mean_wape[k])
        if model_mean_wape
        else max(global_w, key=lambda k: global_w[k])
    )

    adaptive_map: dict[str, dict[str, float]] = {}
    for tid, grp_t in work.groupby("ts_id", sort=True):
        piv = grp_t.pivot_table(
            index=date_col, columns="model_type", values="yhat", aggfunc="first"
        ).sort_index()
        yt = pd.to_numeric(
            grp_t.groupby(date_col)["y_true"].first().reindex(piv.index), errors="coerce"
        ).to_numpy(dtype=float)
        if champ_m not in piv.columns or len(yt) < 2:
            adaptive_map[str(tid)] = dict(global_w)
            continue
        yc = pd.to_numeric(piv[champ_m], errors="coerce").to_numpy(dtype=float)
        base_err = np.abs(yt - yc)
        mid = max(1, len(yt) // 2)
        best_w: dict[str, float] = {champ_m: 1.0}
        best_loss = float(np.sum(base_err))

        # Test global_w blend on this series
        yg = np.zeros_like(yt)
        wg_sum = 0.0
        for m_k, w_k in global_w.items():
            if m_k in piv.columns:
                ym_k = pd.to_numeric(piv[m_k], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(ym_k).all():
                    yg = yg + w_k * ym_k
                    wg_sum += w_k
        if wg_sum > 1e-9:
            yg = yg / wg_sum
            err_g = np.abs(yt - yg)
            if (
                float(np.sum(err_g)) < best_loss - 1e-6
                and float(np.sum(err_g[:mid])) <= float(np.sum(base_err[:mid]))
                and float(np.sum(err_g[mid:])) <= float(np.sum(base_err[mid:]))
                and float(np.mean(err_g <= base_err)) >= 0.55
            ):
                best_loss = float(np.sum(err_g))
                best_w = {m_k: w_k / wg_sum for m_k, w_k in global_w.items() if m_k in piv.columns}

        # Test fold-consistent pairwise blends with global champion
        for m_col in piv.columns:
            m_str = str(m_col)
            if m_str == champ_m:
                continue
            ym = pd.to_numeric(piv[m_col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(ym).all():
                continue
            for alpha in (-0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
                y_blend = (1.0 - alpha) * yc + alpha * ym
                err_b = np.abs(yt - y_blend)
                if (
                    float(np.sum(err_b)) < best_loss - 1e-6
                    and float(np.sum(err_b[:mid])) <= float(np.sum(base_err[:mid]))
                    and float(np.sum(err_b[mid:])) <= float(np.sum(base_err[mid:]))
                    and float(np.mean(err_b <= base_err)) >= 0.55
                ):
                    best_loss = float(np.sum(err_b))
                    best_w = {champ_m: round(1.0 - alpha, 4), m_str: round(alpha, 4)}
        adaptive_map[str(tid)] = best_w
    return adaptive_map


class EvolvedForecaster(BaseModel):
    """Custom evolutionary forecaster plugin (`target='model'`)."""

    name: ClassVar[str] = "evolved_forecaster"
    runtime: ClassVar[str] = "python"  # type: ignore[assignment]
    family: ClassVar[str] = "statistical"  # type: ignore[assignment]
    supports_exog: ClassVar[bool] = True
    supports_native_intervals: ClassVar[bool] = False
    package: ClassVar[str] = "numpy"
    package_url: ClassVar[str] = "https://numpy.org/"

    def __init__(self, params: dict[str, Any], ctx: ModelContext) -> None:
        super().__init__(params, ctx)
        self._last_ds: pd.Timestamp | None = None
        self._fitted_values: np.ndarray = np.array([], dtype=float)
        self._y_train: np.ndarray = np.array([], dtype=float)
        self._season_period: int = 7
        self._seasonal_profile: np.ndarray = np.zeros(7, dtype=float)
        self._level: float = 0.0
        self._slope: float = 0.0
        self._exog_cols: list[str] = []
        self._exog_coef: np.ndarray = np.array([], dtype=float)
        self._exog_mean: np.ndarray = np.array([], dtype=float)

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        """Fit evolutionary damped trend + harmonic seasonal + ridge exogenous model."""
        y_vals = pd.to_numeric(y, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        self._y_train = y_vals
        if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
            self._last_ds = pd.Timestamp(y.index[-1])
        elif X is not None and "ds" in X.columns and len(X) > 0:
            self._last_ds = pd.Timestamp(X["ds"].iloc[-1])
        else:
            self._last_ds = pd.Timestamp("2025-01-01")
        self._last_date = self._last_ds

        n = len(y_vals)
        if n == 0:
            self._fitted_values = np.zeros(1, dtype=float)
            self._set_residuals(np.zeros(1, dtype=float))
            return

        # EVOLVE-BLOCK-START
        freq_str = str(self.ctx.freq).upper()
        if freq_str.startswith("M"):
            period = 12
        elif freq_str.startswith("W"):
            period = 13
        elif freq_str.startswith("Q"):
            period = 4
        elif freq_str.startswith("H"):
            period = 24
        else:
            period = 7
        self._season_period = max(1, min(period, max(1, n // 2)))

        t = np.arange(n, dtype=float)
        if n >= 4:
            t_mean = float(np.mean(t))
            y_mean = float(np.mean(y_vals))
            denom = float(np.dot(t - t_mean, t - t_mean)) + 1e-6
            raw_slope = float(np.dot(t - t_mean, y_vals - y_mean)) / denom
            self._slope = 0.85 * raw_slope
            trend_line = y_mean + raw_slope * (t - t_mean)
            self._level = float(trend_line[-1])
        else:
            self._slope = 0.0
            self._level = float(np.mean(y_vals))
            trend_line = np.full(n, self._level, dtype=float)

        detrended = y_vals - trend_line
        s_prof = np.zeros(self._season_period, dtype=float)
        for rem in range(self._season_period):
            vals_rem = detrended[rem :: self._season_period]
            if len(vals_rem) > 0:
                s_prof[rem] = float(np.median(vals_rem))
        s_prof -= float(np.mean(s_prof))
        self._seasonal_profile = s_prof

        seasonal_comp = np.array(
            [s_prof[i % self._season_period] for i in range(n)],
            dtype=float,
        )
        resid = detrended - seasonal_comp
        self._exog_cols = []
        self._exog_coef = np.array([], dtype=float)
        self._exog_mean = np.array([], dtype=float)
        exog_fit = np.zeros(n, dtype=float)
        if X is not None and not X.empty:
            num_cols = [
                c
                for c in X.columns
                if c not in ("ds", "ts_id", "y") and pd.api.types.is_numeric_dtype(X[c])
            ]
            if num_cols:
                self._exog_cols = num_cols
                x_mat = (
                    X[num_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                self._exog_mean = np.mean(x_mat, axis=0)
                x_c = x_mat - self._exog_mean
                gram = x_c.T @ x_c + 1.0 * np.eye(len(num_cols))
                self._exog_coef = np.linalg.solve(gram, x_c.T @ resid)
                exog_fit = x_c @ self._exog_coef

        self._fitted_values = trend_line + seasonal_comp + exog_fit
        # EVOLVE-BLOCK-END
        self._set_residuals(self._y_train - self._fitted_values)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Predict ``horizon`` steps ahead with calibrated residual intervals."""
        h = int(horizon)
        n = len(self._y_train)
        if X is not None and "ds" in X.columns and len(X) >= h:
            ds = pd.DatetimeIndex(pd.to_datetime(X["ds"].iloc[:h]))
        else:
            start = self._last_ds if self._last_ds is not None else pd.Timestamp("2025-01-01")
            freq = self.ctx.freq or "D"
            ds = pd.date_range(start=start, periods=h + 1, freq=freq)[1:]

        # EVOLVE-BLOCK-START
        steps = np.arange(1, h + 1, dtype=float)
        damping = (1.0 - np.power(0.90, steps)) / (1.0 - 0.90)
        base_level = (
            float(self._fitted_values[-1] - self._seasonal_profile[(n - 1) % self._season_period])
            if n > 0 and len(self._seasonal_profile) > 0
            else self._level
        )
        seasonal_fut = np.array(
            [
                self._seasonal_profile[(n + step - 1) % self._season_period]
                for step in range(1, h + 1)
            ],
            dtype=float,
        )
        yhat_raw = base_level + self._slope * damping + seasonal_fut

        if self._exog_cols and X is not None and not X.empty:
            present = [c for c in self._exog_cols if c in X.columns]
            if len(present) == len(self._exog_cols):
                x_fut = (
                    X[self._exog_cols]
                    .iloc[:h]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                if len(x_fut) == h:
                    yhat_raw = yhat_raw + (x_fut - self._exog_mean) @ self._exog_coef
        # EVOLVE-BLOCK-END

        qmap = self.residual_intervals(yhat_raw, quantiles=tuple(quantiles))
        return self._assemble_frame(ds, qmap, raw=yhat_raw)


def evolve_policy_weights(base_weights: AutoPolicyWeights) -> AutoPolicyWeights:
    """Tune :class:`AutoPolicyWeights` thresholds for ``auto_policy`` (`target='policy'`)."""
    # EVOLVE-BLOCK-START
    updated_actions = dict(base_weights.action_weights)
    updated_actions["prune_decoy_covariates"] = (0.90, 7.5)
    updated_actions["inject_transform_log1p"] = (0.93, 10.8)
    updated_actions["inject_direct_horizon_models"] = (0.86, 6.2)
    updated_actions["inject_monotone_guardrails"] = (0.88, 7.0)
    updated_actions["inject_transform_simplex"] = (0.89, 8.4)
    updated_actions["enable_temporal_reconciliation"] = (0.81, 5.2)
    return base_weights.with_updates(
        intermittent_share_min=0.16,
        log1p_level_vol_corr_min=0.60,
        log1p_growth_ratio_min=1.30,
        direct_horizon_decay_ratio_min=1.28,
        pass_decay_factor=0.74,
        action_weights=updated_actions,
    )
    # EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# Seed Templates & Block Extraction / Injection Helpers
# ---------------------------------------------------------------------------

_FEATURES_SEED_CODE = '''\
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def synthesize_evolved_features(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Synthesize causal evolved_feat_* columns strictly from past/current row context."""
    out = df.copy()
    id_col = str((profile or {}).get("ts_id_col", "ts_id"))
    if id_col not in out.columns:
        id_col = "ts_id" if "ts_id" in out.columns else out.columns[0]

    # EVOLVE-BLOCK-START
    step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
    out["evolved_feat_trend_idx"] = np.log1p(step_idx)
    out["evolved_feat_sin_w"] = np.sin(2.0 * np.pi * step_idx / 7.0)
    out["evolved_feat_cos_w"] = np.cos(2.0 * np.pi * step_idx / 7.0)

    reserved = {id_col, "ts_id", "ds", "date", "forecast_date", "y", "y_true", "yhat"}
    exog_cols = [
        c
        for c in out.columns
        if c not in reserved
        and not str(c).startswith("evolved_feat_")
        and pd.api.types.is_numeric_dtype(out[c])
    ]
    for col in exog_cols[:3]:
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[f"evolved_feat_{col}_lag1"] = (
            vals.groupby(out[id_col], sort=False).shift(1).fillna(vals)
        )
        out[f"evolved_feat_{col}_roll3"] = (
            vals.groupby(out[id_col], sort=False)
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
            .astype(float)
        )
    # EVOLVE-BLOCK-END
    return out
'''

_ENSEMBLE_SEED_CODE = '''\
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd


def compute_evolved_weights(
    oof_df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute global (`dict[str, float]`) or per-series (`dict[str, dict[str, float]]`) weights."""
    _ = profile
    if oof_df.empty or "model_type" not in oof_df.columns:
        return {}
    work = oof_df.copy()
    date_col = (
        "ds"
        if "ds" in work.columns
        else ("forecast_date" if "forecast_date" in work.columns else "")
    )
    per_series_weights: dict[str, dict[str, float]] = {}

    # EVOLVE-BLOCK-START
    scores: dict[str, float] = {}
    for model_type, grp in work.groupby("model_type", sort=False):
        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(yhat)
        if not mask.any():
            continue
        abs_err = np.abs(y_true[mask] - yhat[mask])
        denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
        wape = float(np.sum(abs_err)) / denom
        scores[str(model_type)] = 1.0 / max(1e-6, wape ** 2)
    # EVOLVE-BLOCK-END

    if per_series_weights:
        return per_series_weights

    total = sum(v for v in scores.values() if math.isfinite(v) and v > 0.0)
    if total <= 0.0:
        models = sorted(work["model_type"].astype(str).unique())
        global_w = {m: 1.0 / len(models) for m in models} if models else {}
    else:
        global_w = {k: float(v) / total for k, v in scores.items() if math.isfinite(v) and v > 0.0}

    if "ts_id" not in work.columns or not date_col or work["ts_id"].nunique() < 2 or not global_w:
        return global_w

    model_mean_wape: dict[str, float] = {}
    for m_name, grp_m in work.groupby("model_type", sort=False):
        s_losses: list[float] = []
        for _, grp_s in grp_m.groupby("ts_id", sort=False):
            yt_s = pd.to_numeric(grp_s["y_true"], errors="coerce").to_numpy(dtype=float)
            yh_s = pd.to_numeric(grp_s["yhat"], errors="coerce").to_numpy(dtype=float)
            mk = np.isfinite(yt_s) & np.isfinite(yh_s)
            if mk.any():
                s_losses.append(
                    float(np.sum(np.abs(yt_s[mk] - yh_s[mk])))
                    / max(1e-6, float(np.sum(np.abs(yt_s[mk]))))
                )
        if s_losses:
            model_mean_wape[str(m_name)] = float(np.mean(s_losses))
    champ_m = (
        min(model_mean_wape, key=lambda k: model_mean_wape[k])
        if model_mean_wape
        else max(global_w, key=lambda k: global_w[k])
    )

    adaptive_map: dict[str, dict[str, float]] = {}
    for tid, grp_t in work.groupby("ts_id", sort=True):
        piv = grp_t.pivot_table(
            index=date_col, columns="model_type", values="yhat", aggfunc="first"
        ).sort_index()
        yt = pd.to_numeric(
            grp_t.groupby(date_col)["y_true"].first().reindex(piv.index),
            errors="coerce",
        ).to_numpy(dtype=float)
        if champ_m not in piv.columns or len(yt) < 2:
            adaptive_map[str(tid)] = dict(global_w)
            continue
        yc = pd.to_numeric(piv[champ_m], errors="coerce").to_numpy(dtype=float)
        base_err = np.abs(yt - yc)
        mid = max(1, len(yt) // 2)
        best_w: dict[str, float] = {champ_m: 1.0}
        best_loss = float(np.sum(base_err))

        yg = np.zeros_like(yt)
        wg_sum = 0.0
        for m_k, w_k in global_w.items():
            if m_k in piv.columns:
                ym_k = pd.to_numeric(piv[m_k], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(ym_k).all():
                    yg = yg + w_k * ym_k
                    wg_sum += w_k
        if wg_sum > 1e-9:
            yg = yg / wg_sum
            err_g = np.abs(yt - yg)
            if (
                float(np.sum(err_g)) < best_loss - 1e-6
                and float(np.sum(err_g[:mid])) <= float(np.sum(base_err[:mid]))
                and float(np.sum(err_g[mid:])) <= float(np.sum(base_err[mid:]))
                and float(np.mean(err_g <= base_err)) >= 0.55
            ):
                best_loss = float(np.sum(err_g))
                best_w = {m_k: w_k / wg_sum for m_k, w_k in global_w.items() if m_k in piv.columns}

        for m_col in piv.columns:
            m_str = str(m_col)
            if m_str == champ_m:
                continue
            ym = pd.to_numeric(piv[m_col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(ym).all():
                continue
            for alpha in (-0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
                y_blend = (1.0 - alpha) * yc + alpha * ym
                err_b = np.abs(yt - y_blend)
                if (
                    float(np.sum(err_b)) < best_loss - 1e-6
                    and float(np.sum(err_b[:mid])) <= float(np.sum(base_err[:mid]))
                    and float(np.sum(err_b[mid:])) <= float(np.sum(base_err[mid:]))
                    and float(np.mean(err_b <= base_err)) >= 0.55
                ):
                    best_loss = float(np.sum(err_b))
                    best_w = {champ_m: round(1.0 - alpha, 4), m_str: round(alpha, 4)}
        adaptive_map[str(tid)] = best_w
    return adaptive_map
'''

_MODEL_SEED_CODE = '''\
from __future__ import annotations
from typing import Any, ClassVar, Sequence
import numpy as np
import pandas as pd
from scale_forecasting.models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext


class EvolvedForecaster(BaseModel):
    """Evolutionary custom forecaster plugin."""

    name: ClassVar[str] = "evolved_forecaster"
    runtime: ClassVar[str] = "python"
    family: ClassVar[str] = "statistical"
    supports_exog: ClassVar[bool] = True
    supports_native_intervals: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any], ctx: ModelContext) -> None:
        super().__init__(params, ctx)
        self._last_ds: pd.Timestamp | None = None
        self._y_train: np.ndarray = np.array([], dtype=float)
        self._fitted_values: np.ndarray = np.array([], dtype=float)
        self._season_period: int = 7
        self._seasonal_profile: np.ndarray = np.zeros(7, dtype=float)
        self._level: float = 0.0
        self._slope: float = 0.0

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> EvolvedForecaster:
        _ = X
        y_vals = pd.to_numeric(y, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        self._y_train = y_vals
        if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
            self._last_ds = pd.Timestamp(y.index[-1])
        else:
            self._last_ds = pd.Timestamp("2025-01-01")

        n = len(y_vals)
        if n == 0:
            self._fitted_values = np.zeros(1, dtype=float)
            return self

        # EVOLVE-BLOCK-START
        self._season_period = max(1, min(7, max(1, n // 2)))
        t = np.arange(n, dtype=float)
        t_mean = float(np.mean(t))
        y_mean = float(np.mean(y_vals))
        denom = float(np.dot(t - t_mean, t - t_mean)) + 1e-6
        raw_slope = float(np.dot(t - t_mean, y_vals - y_mean)) / denom if n >= 4 else 0.0
        self._slope = 0.5 * raw_slope
        trend_line = y_mean + raw_slope * (t - t_mean)
        detrended = y_vals - trend_line
        s_prof = np.zeros(self._season_period, dtype=float)
        for rem in range(self._season_period):
            vals_rem = detrended[rem :: self._season_period]
            if len(vals_rem) > 0:
                s_prof[rem] = float(np.median(vals_rem))
        s_prof -= float(np.mean(s_prof))
        self._seasonal_profile = s_prof
        seasonal_comp = np.array(
            [s_prof[i % self._season_period] for i in range(n)],
            dtype=float,
        )
        self._level = float(trend_line[-1])
        self._fitted_values = trend_line + seasonal_comp
        # EVOLVE-BLOCK-END
        self._set_residuals(self._y_train - self._fitted_values)
        return self

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        h = int(horizon)
        n = len(self._y_train)
        if X is not None and "ds" in X.columns and len(X) >= h:
            ds = pd.DatetimeIndex(pd.to_datetime(X["ds"].iloc[:h]))
        else:
            start = self._last_ds if self._last_ds is not None else pd.Timestamp("2025-01-01")
            ds = pd.date_range(start=start, periods=h + 1, freq=self.ctx.freq or "D")[1:]

        steps = np.arange(1, h + 1, dtype=float)
        damping = (1.0 - np.power(0.90, steps)) / (1.0 - 0.90)
        seasonal_fut = np.array(
            [
                self._seasonal_profile[(n + step - 1) % self._season_period]
                for step in range(1, h + 1)
            ],
            dtype=float,
        )
        yhat_raw = self._level + self._slope * damping + seasonal_fut
        qmap = self.residual_intervals(yhat_raw, quantiles=tuple(quantiles))
        return self._assemble_frame(ds, qmap, raw=yhat_raw)
'''

_POLICY_SEED_CODE = '''\
from __future__ import annotations
from scale_forecasting.evolve.targets import AutoPolicyWeights


def evolve_policy_weights(base_weights: AutoPolicyWeights) -> AutoPolicyWeights:
    """Tune AutoPolicyWeights decision thresholds for auto_policy."""
    # EVOLVE-BLOCK-START
    updated_actions = dict(base_weights.action_weights)
    updated_actions["prune_decoy_covariates"] = (0.90, 7.5)
    updated_actions["inject_transform_log1p"] = (0.93, 10.8)
    updated_actions["inject_direct_horizon_models"] = (0.86, 6.2)
    updated_actions["inject_monotone_guardrails"] = (0.88, 7.0)
    updated_actions["inject_transform_simplex"] = (0.89, 8.4)
    updated_actions["enable_temporal_reconciliation"] = (0.81, 5.2)
    return base_weights.with_updates(
        intermittent_share_min=0.16,
        log1p_level_vol_corr_min=0.60,
        log1p_growth_ratio_min=1.30,
        direct_horizon_decay_ratio_min=1.28,
        pass_decay_factor=0.74,
        action_weights=updated_actions,
    )
    # EVOLVE-BLOCK-END
'''


_TARGET_SPECS: dict[str, TargetSpec] = {
    "features": TargetSpec(
        target="features",
        entrypoint="synthesize_evolved_features",
        description=(
            "Synthesize causal lag, rolling, harmonic, and exogenous interaction features "
            "(evolved_feat_*) strictly from past/current observations without future leakage."
        ),
        seed_code=_FEATURES_SEED_CODE,
        system_instruction=(
            "You are an expert econometric feature synthesis agent inside scale-forecasting. "
            "Mutate only the Python code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END "
            "inside synthesize_evolved_features(df, profile). Every engineered column MUST start "
            "with 'evolved_feat_' and MUST be strictly causal (never use .shift(-k), bfill, or "
            "full-series global statistics of target 'y'). Allowed modules: numpy, pandas, math."
        ),
        mutation_prompt_template=(
            "Target: features (synthesize_evolved_features)\n"
            "Generation: {generation}\n"
            "Prior Walk-Forward Diagnostics: {diagnostics}\n\n"
            "Parent Program:\n```python\n{parent_code}\n```\n\n"
            "Return ONLY valid Python code containing # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END "
            "that improves walk-forward cross-validation accuracy and worst-decile FVA without "
            "any future lookahead leakage."
        ),
    ),
    "ensemble": TargetSpec(
        target="ensemble",
        entrypoint="compute_evolved_weights",
        description=(
            "Synthesize out-of-fold (OOF) model blending weights over the cached model bank "
            "produced by Forecaster.auto()."
        ),
        seed_code=_ENSEMBLE_SEED_CODE,
        system_instruction=(
            "You are an expert forecast ensembling agent inside scale-forecasting. "
            "Mutate only the Python code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END "
            "inside compute_evolved_weights(oof_df, profile) to populate "
            "`scores: dict[str, float]` with positive unnormalized model weights based on "
            "out-of-fold errors (`y_true` vs `yhat`), worst-decile series resilience, and "
            "horizon decay stability."
        ),
        mutation_prompt_template=(
            "Target: ensemble (compute_evolved_weights)\n"
            "Generation: {generation}\n"
            "Prior Walk-Forward Diagnostics: {diagnostics}\n\n"
            "Parent Program:\n```python\n{parent_code}\n```\n\n"
            "Return ONLY valid Python code containing # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END "
            "that populates `scores: dict[str, float]` to minimize validation loss and improve "
            "Diebold-Mariano significance."
        ),
    ),
    "model": TargetSpec(
        target="model",
        entrypoint="EvolvedForecaster",
        description=(
            "Synthesize an EvolvedForecaster(BaseModel) plugin implementing fit() and predict() "
            "with numpy/scipy/statsmodels primitives."
        ),
        seed_code=_MODEL_SEED_CODE,
        system_instruction=(
            "You are an expert statistical and machine learning time-series model architect. "
            "Mutate the # EVOLVE-BLOCK-START .. # EVOLVE-BLOCK-END block inside EvolvedForecaster "
            "using only numpy, pandas, scipy, or statsmodels primitives. Never import os, sys, "
            "subprocess, or network modules."
        ),
        mutation_prompt_template=(
            "Target: model (EvolvedForecaster)\n"
            "Generation: {generation}\n"
            "Prior Walk-Forward Diagnostics: {diagnostics}\n\n"
            "Parent Program:\n```python\n{parent_code}\n```\n\n"
            "Return ONLY valid Python code for EvolvedForecaster with # EVOLVE-BLOCK-START and "
            "# EVOLVE-BLOCK-END markers."
        ),
    ),
    "policy": TargetSpec(
        target="policy",
        entrypoint="evolve_policy_weights",
        description=(
            "Tune AutoPolicyWeights decision thresholds and action weights for auto_policy.py."
        ),
        seed_code=_POLICY_SEED_CODE,
        system_instruction=(
            "You are an evolutionary meta-policy optimizer for scale-forecasting's Glass-Box "
            "Auto-Forecaster. Mutate only the # EVOLVE-BLOCK-START .. # EVOLVE-BLOCK-END block "
            "inside evolve_policy_weights(base_weights) and return an updated AutoPolicyWeights."
        ),
        mutation_prompt_template=(
            "Target: policy (evolve_policy_weights)\n"
            "Generation: {generation}\n"
            "Prior Walk-Forward Diagnostics: {diagnostics}\n\n"
            "Parent Program:\n```python\n{parent_code}\n```\n\n"
            "Return ONLY valid Python code for evolve_policy_weights with # EVOLVE-BLOCK-START "
            "and # EVOLVE-BLOCK-END markers."
        ),
    ),
}


def get_target_seed(target: str) -> TargetSpec:
    """Return the canonical :class:`TargetSpec` for ``target``."""
    norm = str(target).strip().lower()
    if norm not in _TARGET_SPECS:
        valid = ", ".join(EVOLVE_TARGETS)
        raise ConfigError(f"Unknown evolve target '{target}'. Valid targets: {valid}")
    return _TARGET_SPECS[norm]


def _normalize_block_indent(raw: str) -> str:
    """Dedent ``raw`` while repairing first-line-stripped indentation skew."""
    cleaned = textwrap.dedent(str(raw).strip("\r\n"))
    lines = cleaned.splitlines()
    non_empty_indices = [idx for idx, ln in enumerate(lines) if ln.strip()]
    if len(non_empty_indices) < 2:
        return cleaned.strip("\r\n")
    first_idx = non_empty_indices[0]
    first_line = lines[first_idx]
    first_indent = len(first_line) - len(first_line.lstrip(" \t"))
    first_stripped = first_line.rstrip()
    if first_indent == 0 and not first_stripped.endswith((":", "\\", "(", "[", "{")):
        rest_indents = [
            len(lines[i]) - len(lines[i].lstrip(" \t"))
            for i in non_empty_indices[1:]
            if (len(lines[i]) - len(lines[i].lstrip(" \t"))) > 0
        ]
        if rest_indents and len(rest_indents) == len(non_empty_indices) - 1:
            common_rest = min(rest_indents)
            repaired = list(lines[: first_idx + 1])
            for ln in lines[first_idx + 1 :]:
                repaired.append(ln[common_rest:] if len(ln) >= common_rest else ln.lstrip(" \t"))
            cleaned = "\n".join(repaired)
    return textwrap.dedent(cleaned).strip("\r\n")


def extract_evolve_block(code: str) -> str:
    """Extract the code lines between ``# EVOLVE-BLOCK-START`` and ``# EVOLVE-BLOCK-END``.

    If the markers are absent from ``code``, returns normalized dedented ``code``.
    """
    text = str(code)
    start_idx = text.find(EVOLVE_BLOCK_START)
    end_idx = text.find(EVOLVE_BLOCK_END)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return _normalize_block_indent(text)
    inner = text[start_idx + len(EVOLVE_BLOCK_START) : end_idx]
    return _normalize_block_indent(inner)


def inject_evolve_block(template: str, block: str) -> str:
    """Inject ``block`` between ``# EVOLVE-BLOCK-START`` and ``# EVOLVE-BLOCK-END`` in ``template``.

    Handles both:
    - an inner snippet (indented to match ``# EVOLVE-BLOCK-START`` in ``template``), and
    - a full program that already contains ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END`` (its
      inner block is extracted first and injected into ``template``).
    """
    clean_block = str(block).strip("\r\n")
    fence_match = re.search(r"```(?:python)?[ \t]*\r?\n(.*?)```", clean_block, flags=re.DOTALL)
    if fence_match:
        clean_block = fence_match.group(1).strip("\r\n")

    if EVOLVE_BLOCK_START in clean_block and EVOLVE_BLOCK_END in clean_block:
        clean_block = extract_evolve_block(clean_block)

    if EVOLVE_BLOCK_START not in template or EVOLVE_BLOCK_END not in template:
        raise ConfigError(f"Template is missing {EVOLVE_BLOCK_START} / {EVOLVE_BLOCK_END} markers.")

    lines = template.splitlines()
    out_lines: list[str] = []
    in_block = False
    injected = False
    for line in lines:
        if EVOLVE_BLOCK_START in line and not injected:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(line)
            dedented = _normalize_block_indent(clean_block)
            for b_line in dedented.splitlines():
                out_lines.append(f"{indent}{b_line}" if b_line.strip() else "")
            in_block = True
            injected = True
            continue
        if EVOLVE_BLOCK_END in line and in_block:
            in_block = False
            out_lines.append(line)
            continue
        if not in_block:
            out_lines.append(line)

    return "\n".join(out_lines) + "\n"


def build_mutation_prompt(
    target: str,
    parent_code: str,
    diagnostics: Mapping[str, Any] | None = None,
    generation: int = 1,
) -> str:
    """Render a diagnostic-guided mutation prompt for ``target``."""
    spec = get_target_seed(target)
    diag_dict = dict(diagnostics or {})
    diag_summary = (
        f"fva_pct={diag_dict.get('fva_pct', 0.0):.2f}%, "
        f"dm_pvalue={diag_dict.get('dm_pvalue', 1.0):.4f}, "
        f"worst_decile_fva={diag_dict.get('worst_decile_fva', 0.0):.4f}, "
        f"horizon_decay={diag_dict.get('horizon_decay', 'stable')}, "
        f"inner_cv_score={diag_dict.get('inner_cv_score', 'n/a')}, "
        f"baseline_score={diag_dict.get('baseline_score', 'n/a')}, "
        f"monotonic_constraints={diag_dict.get('monotonic_constraints', {})}, "
        f"is_compositional_simplex={diag_dict.get('is_compositional_simplex', False)}, "
        f"simplex_total={diag_dict.get('simplex_total')}, "
        f"hierarchy_id_cols={diag_dict.get('hierarchy_id_cols', [])}, "
        f"reconcile_temporal={diag_dict.get('reconcile_temporal', False)}"
    )
    return spec.mutation_prompt_template.format(
        generation=generation,
        diagnostics=diag_summary,
        parent_code=parent_code.strip(),
    )


# ---------------------------------------------------------------------------
# Deterministic Offline Candidates for `backend="mock"` & Gate-3 Overfit Tests
# ---------------------------------------------------------------------------

_MOCK_FEATURES_BLOCKS: tuple[str, ...] = (
    """\
step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
out["evolved_feat_trend_idx"] = np.log1p(step_idx)
out["evolved_feat_sin_w"] = np.sin(2.0 * np.pi * step_idx / 7.0)
out["evolved_feat_cos_w"] = np.cos(2.0 * np.pi * step_idx / 7.0)
out["evolved_feat_sin_w2"] = np.sin(4.0 * np.pi * step_idx / 7.0)
out["evolved_feat_cos_w2"] = np.cos(4.0 * np.pi * step_idx / 7.0)

reserved = {id_col, "ts_id", "ds", "date", "forecast_date", "y", "y_true", "yhat"}
exog_cols = [
    c
    for c in out.columns
    if c not in reserved
    and not str(c).startswith("evolved_feat_")
    and pd.api.types.is_numeric_dtype(out[c])
]
for col in exog_cols[:3]:
    vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    lag1 = vals.groupby(out[id_col], sort=False).shift(1).fillna(vals)
    roll7 = (
        vals.groupby(out[id_col], sort=False)
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
        .astype(float)
    )
    out[f"evolved_feat_{col}_lag1"] = lag1
    out[f"evolved_feat_{col}_roll7"] = roll7
    out[f"evolved_feat_{col}_diff1"] = vals - lag1
    out[f"evolved_feat_{col}_sat"] = np.sign(vals) * np.log1p(np.abs(vals))
""",
    """\
step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
max_step = max(1.0, float(np.max(step_idx) if len(step_idx) else 1.0))
out["evolved_feat_trend_idx"] = step_idx / max_step
for k_harm in (1, 2, 3):
    out[f"evolved_feat_sin_{k_harm}"] = np.sin(2.0 * np.pi * k_harm * step_idx / 7.0)
    out[f"evolved_feat_cos_{k_harm}"] = np.cos(2.0 * np.pi * k_harm * step_idx / 7.0)

reserved = {id_col, "ts_id", "ds", "date", "forecast_date", "y", "y_true", "yhat"}
exog_cols = [
    c
    for c in out.columns
    if c not in reserved
    and not str(c).startswith("evolved_feat_")
    and pd.api.types.is_numeric_dtype(out[c])
]
for col in exog_cols[:4]:
    vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    lag1 = vals.groupby(out[id_col], sort=False).shift(1).fillna(vals)
    lag2 = vals.groupby(out[id_col], sort=False).shift(2).fillna(lag1)
    ewm3 = (
        vals.groupby(out[id_col], sort=False)
        .transform(lambda s: s.ewm(span=3, min_periods=1).mean())
        .astype(float)
    )
    out[f"evolved_feat_{col}_lag1"] = lag1
    out[f"evolved_feat_{col}_lag2"] = lag2
    out[f"evolved_feat_{col}_ewm3"] = ewm3
    out[f"evolved_feat_{col}_impulse"] = vals - 0.5 * (lag1 + lag2)
""",
)

_MOCK_ENSEMBLE_BLOCKS: tuple[str, ...] = (
    """\
scores: dict[str, float] = {}
model_losses: dict[str, float] = {}
for model_type, grp in work.groupby("model_type", sort=False):
    y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(yhat)
    if not mask.any():
        continue
    abs_err = np.abs(y_true[mask] - yhat[mask])
    denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
    wape = float(np.sum(abs_err)) / denom
    q90_err = float(np.percentile(abs_err, 90)) / max(1e-6, float(np.mean(np.abs(y_true[mask]))))
    composite = 0.75 * wape + 0.25 * q90_err
    model_losses[str(model_type)] = composite

if model_losses:
    ranked = sorted(model_losses.items(), key=lambda kv: kv[1])[:3]
    min_loss = max(1e-6, ranked[0][1])
    for m_name, m_loss in ranked:
        rel = m_loss / min_loss
        scores[m_name] = float(np.exp(-3.5 * (rel - 1.0))) / max(1e-6, m_loss ** 2)
""",
    """\
scores: dict[str, float] = {}
for model_type, grp in work.groupby("model_type", sort=False):
    y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(yhat)
    if not mask.any():
        continue
    abs_err = np.abs(y_true[mask] - yhat[mask])
    denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
    wape = float(np.sum(abs_err)) / denom
    scores[str(model_type)] = 1.0 / max(1e-6, wape ** 3)
""",
)

_MOCK_MODEL_BLOCKS: tuple[str, ...] = (
    """\
self._season_period = max(1, min(7, max(1, n // 2)))
t = np.arange(n, dtype=float)
omega = 2.0 * np.pi / float(self._season_period)
design = np.column_stack([
    np.ones(n, dtype=float),
    t,
    np.sin(omega * t),
    np.cos(omega * t),
])
gram = design.T @ design + 1e-4 * np.eye(4)
beta = np.linalg.solve(gram, design.T @ y_vals)
trend_line = beta[0] + beta[1] * t
self._slope = float(beta[1])
detrended = y_vals - trend_line
s_prof = np.zeros(self._season_period, dtype=float)
for rem in range(self._season_period):
    vals_rem = detrended[rem :: self._season_period]
    if len(vals_rem) > 0:
        s_prof[rem] = float(np.mean(vals_rem))
s_prof -= float(np.mean(s_prof))
self._seasonal_profile = s_prof
seasonal_comp = np.array(
    [s_prof[i % self._season_period] for i in range(n)],
    dtype=float,
)
self._level = float(trend_line[-1])
self._fitted_values = trend_line + seasonal_comp
""",
)

_MOCK_POLICY_BLOCKS: tuple[str, ...] = (
    """\
updated_actions = dict(base_weights.action_weights)
updated_actions["prune_decoy_covariates"] = (0.94, 8.8)
updated_actions["inject_transform_log1p"] = (0.95, 11.4)
updated_actions["inject_direct_horizon_models"] = (0.89, 6.8)
updated_actions["enable_level_shift"] = (0.93, 12.0)
updated_actions["inject_monotone_guardrails"] = (0.90, 7.4)
updated_actions["inject_transform_simplex"] = (0.91, 8.8)
updated_actions["enable_temporal_reconciliation"] = (0.83, 5.6)
return base_weights.with_updates(
    intermittent_share_min=0.15,
    log1p_level_vol_corr_min=0.56,
    log1p_growth_ratio_min=1.26,
    direct_horizon_decay_ratio_min=1.24,
    pass_decay_factor=0.76,
    tree_n_estimators_fast=80,
    action_weights=updated_actions,
)
""",
)


def deterministic_mock_candidates(target: str, count: int = 3) -> list[str]:
    """Return ``count`` deterministic, AST-safe, causal candidate programs for ``target``."""
    spec = get_target_seed(target)
    blocks_by_target: dict[str, tuple[str, ...]] = {
        "features": _MOCK_FEATURES_BLOCKS,
        "ensemble": _MOCK_ENSEMBLE_BLOCKS,
        "model": _MOCK_MODEL_BLOCKS,
        "policy": _MOCK_POLICY_BLOCKS,
    }
    blocks = blocks_by_target[spec.target]
    candidates: list[str] = []
    for idx in range(max(1, count)):
        blk = blocks[idx % len(blocks)]
        candidates.append(inject_evolve_block(spec.seed_code, blk))
    return candidates


# Candidate programs specifically constructed to pass AST + causality + Gate 1 + Gate 2 on the
# inner evolution panel while degrading on the longer Held-Out Lockdown Fold (for testing Gate 3).
OVERFIT_LOCKDOWN_CANDIDATES: dict[str, str] = {
    "features": inject_evolve_block(
        _FEATURES_SEED_CODE,
        """\
step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
y_lag1 = y_num.groupby(out[id_col], sort=False).shift(1).fillna(y_num)
is_flat_tail = (np.abs(y_num - y_lag1) < 1e-11) & (step_idx > 4.0)
out["evolved_feat_trend_idx"] = np.where(is_flat_tail, -500.0 * step_idx, np.log1p(step_idx))
out["evolved_feat_sin_w"] = np.sin(2.0 * np.pi * step_idx / 7.0)
out["evolved_feat_cos_w"] = np.cos(2.0 * np.pi * step_idx / 7.0)
""",
    ),
    "ensemble": inject_evolve_block(
        _ENSEMBLE_SEED_CODE,
        """\
scores: dict[str, float] = {}
for model_type, grp in work.groupby("model_type", sort=False):
    y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(grp["yhat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(yhat)
    if not mask.any():
        continue
    abs_err = np.abs(y_true[mask] - yhat[mask])
    denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
    wape = float(np.sum(abs_err)) / denom
    scores[str(model_type)] = 1.0 / max(1e-6, wape ** 3)
""",
    ),
}
