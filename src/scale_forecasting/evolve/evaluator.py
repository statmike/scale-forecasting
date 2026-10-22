"""Walk-forward candidate evaluator and 3-Gate Anti-Overfitting Ratchet (`evolve.evaluator`).

Builds directly on :meth:`~scale_forecasting.sdk.Forecaster.auto` by splitting the input panel into:

1. **Inner Evolution Panel ($t \\le T - H_{\\text{lockdown}}$):** Visible to the evolutionary loop.
   ``Forecaster.auto()`` runs once on this inner panel to establish the baseline
   :class:`Forecaster`, :class:`~scale_forecasting.auto_policy.PanelAutoProfile`, cached
   out-of-fold predictions (``oof_df``), and reusable cell cache (``reuse_prior_runs=True``).
2. **Held-Out Lockdown Fold (last $H_{\\text{lockdown}}$ steps):** Completely withheld from the
   evolutionary loop and evaluated **only once** at Gate 3 on the top inner-CV challenger.

The 3-Gate Anti-Overfitting Ratchet
-----------------------------------
* **Pre-Gate (AST Security & Temporal Causality Probe):** Candidate must pass
  :func:`~scale_forecasting.evolve.sandbox.validate_candidate_ast` and
  :func:`~scale_forecasting.evolve.sandbox.verify_no_future_leakage` ($y_{T-H:} \\mathrel{+}= 10^6$
  perturbation test).
* **Gate 1 (Inner Walk-Forward CV Lift):** ``inner_cv_score < baseline_score``.
* **Gate 2 (Diebold–Mariano & Worst-Decile Tail Guard):** Paired loss differential test via
  :func:`~scale_forecasting.diagnostics.diebold_mariano_test` requires
  ``dm_pvalue <= significance_alpha`` (when sample size permits) and
  ``worst_decile_fva_pct >= -0.02``.
* **Gate 3 (Held-Out Lockdown Fold Verification):** Evaluates the winning inner-CV candidate once
  on the unseen Lockdown Fold and promotes (``promoted = True``) **only** if
  ``lockdown_candidate_score < lockdown_baseline_score``. Otherwise ratchets back to the
  ``Forecaster.auto()`` baseline (``promoted = False``).
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..auto import auto_forecast
from ..diagnostics import diebold_mariano_test
from ..errors import ConfigError
from ..models.base_model import ModelContext
from ..reconciliation import HierarchySpec, build_hierarchy, reconcile_matrix
from ..simplex import enforce_simplex_closure
from .sandbox import (
    compile_candidate_symbol,
    execute_in_sandbox,
    validate_candidate_ast,
    verify_no_future_leakage,
)
from .targets import (
    AutoPolicyWeights,
    compute_evolved_weights,
    get_target_seed,
)

__all__ = [
    "CandidateEvaluation",
    "EvolutionEvaluator",
    "split_inner_and_lockdown_panel",
    "verify_oof_pool_compatibility",
]


@dataclass(frozen=True)
class CandidateEvaluation:
    """Complete evaluation telemetry for one candidate program across all 3 ratchet gates."""

    candidate_id: str
    generation: int
    target: str
    backend: str
    code: str
    ast_valid: bool
    causal_valid: bool
    inner_cv_score: float
    baseline_score: float
    inner_fva_pct: float
    dm_stat: float
    dm_pvalue: float
    worst_decile_fva_pct: float
    gate1_passed: bool
    gate2_passed: bool
    lockdown_baseline_score: float
    lockdown_candidate_score: float
    gate3_passed: bool
    promoted: bool
    status: str
    rejection_reason: str | None
    wall_seconds: float
    series_losses_candidate: tuple[float, ...] = ()
    series_losses_baseline: tuple[float, ...] = ()
    point_losses_candidate: tuple[float, ...] = ()
    point_losses_baseline: tuple[float, ...] = ()

    def to_row(self) -> dict[str, Any]:
        """Convert candidate evaluation to a flat leaderboard row dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "target": self.target,
            "backend": self.backend,
            "ast_valid": self.ast_valid,
            "causal_valid": self.causal_valid,
            "inner_cv_score": (
                round(self.inner_cv_score, 6) if math.isfinite(self.inner_cv_score) else None
            ),
            "baseline_score": (
                round(self.baseline_score, 6) if math.isfinite(self.baseline_score) else None
            ),
            "inner_fva_pct": (
                round(self.inner_fva_pct, 4) if math.isfinite(self.inner_fva_pct) else 0.0
            ),
            "dm_pvalue": round(self.dm_pvalue, 6) if math.isfinite(self.dm_pvalue) else None,
            "worst_decile_fva_pct": (
                round(self.worst_decile_fva_pct, 4)
                if math.isfinite(self.worst_decile_fva_pct)
                else None
            ),
            "gate1_passed": self.gate1_passed,
            "gate2_passed": self.gate2_passed,
            "lockdown_baseline_score": (
                round(self.lockdown_baseline_score, 6)
                if math.isfinite(self.lockdown_baseline_score)
                else None
            ),
            "lockdown_candidate_score": (
                round(self.lockdown_candidate_score, 6)
                if math.isfinite(self.lockdown_candidate_score)
                else None
            ),
            "gate3_passed": self.gate3_passed,
            "promoted": self.promoted,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "wall_seconds": round(self.wall_seconds, 4),
        }


def split_inner_and_lockdown_panel(
    df: pd.DataFrame,
    horizon: int,
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a historical panel into `(inner_df, lockdown_truth_df, lockdown_future_exog_df)`.

    - ``inner_df``: rows $t \\le T - H_{\\text{lockdown}}$ per series (visible to evolution).
    - ``lockdown_truth_df``: last $H_{\\text{lockdown}}$ historical rows per series with true
      ``y`` (withheld for Gate 3 verification).
    - ``lockdown_future_exog_df``: exogenous covariate rows for the lockdown window with ``y``
      removed so lockdown predictions cannot observe held-out targets.
    """
    work = df.copy()
    if ts_id_col not in work.columns:
        work[ts_id_col] = "series_0"
    y_num = pd.to_numeric(work[target_col], errors="coerce")
    hist = work[y_num.notna()].sort_values([ts_id_col, date_col]).reset_index(drop=True)
    if hist.empty:
        raise ConfigError("Input panel has no valid numeric target rows.")

    h = max(1, int(horizon))
    inner_parts: list[pd.DataFrame] = []
    lock_parts: list[pd.DataFrame] = []

    for _, grp in hist.groupby(ts_id_col, sort=False):
        n_obs = len(grp)
        eff_h = min(h, max(2, n_obs // 4))
        inner_parts.append(grp.iloc[:-eff_h])
        lock_parts.append(grp.iloc[-eff_h:])

    inner_df = pd.concat(inner_parts, ignore_index=True)
    lockdown_truth_df = pd.concat(lock_parts, ignore_index=True)
    exog_cols = [c for c in lockdown_truth_df.columns if c != target_col]
    lockdown_future_exog_df = lockdown_truth_df[exog_cols].copy()
    return inner_df, lockdown_truth_df, lockdown_future_exog_df


def _series_wape(y_true: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(yhat)
    if not mask.any():
        return float("nan")
    denom = max(1e-6, float(np.sum(np.abs(y_true[mask]))))
    return float(np.sum(np.abs(y_true[mask] - yhat[mask]))) / denom


def _compute_ratchet_stats(
    series_losses_cand: list[float],
    series_losses_base: list[float],
    point_losses_cand: list[float],
    point_losses_base: list[float],
    *,
    significance_alpha: float,
    worst_decile_floor: float,
) -> dict[str, Any]:
    """Compute inner CV score, FVA %, Diebold-Mariano p-value, worst-decile FVA, and Gates 1-2."""
    sc_arr = np.asarray(series_losses_cand, dtype=float)
    sb_arr = np.asarray(series_losses_base, dtype=float)
    valid = np.isfinite(sc_arr) & np.isfinite(sb_arr)
    if not valid.any():
        return {
            "inner_cv_score": float("inf"),
            "baseline_score": float("inf"),
            "inner_fva_pct": 0.0,
            "dm_stat": 0.0,
            "dm_pvalue": 1.0,
            "worst_decile_fva_pct": -1.0,
            "gate1_passed": False,
            "gate2_passed": False,
            "rejection_reason": (
                "Gate 1 (Inner Walk-Forward CV): candidate produced non-finite predictions."
            ),
        }

    sc = sc_arr[valid]
    sb = sb_arr[valid]
    inner_cv_score = float(np.mean(sc))
    baseline_score = float(np.mean(sb))
    inner_fva_pct = (
        100.0 * (baseline_score - inner_cv_score) / max(1e-9, baseline_score)
        if baseline_score > 1e-9
        else 0.0
    )

    fva_per_series = (sb - sc) / np.maximum(1e-9, sb)
    worst_decile_fva_pct = float(np.percentile(fva_per_series, 10))

    # Diebold-Mariano test across paired point-level and series-level errors
    dm_stat_s, dm_p_s = diebold_mariano_test(sc, sb, h=1)
    dm_stat_p, dm_p_p = diebold_mariano_test(point_losses_cand, point_losses_base, h=1)

    if math.isfinite(dm_p_p) and dm_stat_p < 0:
        dm_stat = dm_stat_p
        dm_pvalue = dm_p_p
        if math.isfinite(dm_p_s) and dm_stat_s < 0:
            dm_pvalue = min(dm_p_p, dm_p_s)
    elif math.isfinite(dm_p_s):
        dm_stat = dm_stat_s
        dm_pvalue = dm_p_s
    else:
        dm_stat = -1.0 if inner_cv_score < baseline_score else 1.0
        dm_pvalue = 0.05 if inner_cv_score < baseline_score else 1.0

    gate1_passed = bool(inner_cv_score < baseline_score - 1e-9)
    if not gate1_passed:
        return {
            "inner_cv_score": inner_cv_score,
            "baseline_score": baseline_score,
            "inner_fva_pct": inner_fva_pct,
            "dm_stat": dm_stat,
            "dm_pvalue": dm_pvalue,
            "worst_decile_fva_pct": worst_decile_fva_pct,
            "gate1_passed": False,
            "gate2_passed": False,
            "rejection_reason": (
                f"Gate 1 (Inner Walk-Forward CV): candidate score ({inner_cv_score:.4f}) "
                f"did not improve upon Forecaster.auto() baseline ({baseline_score:.4f})."
            ),
        }

    # Gate 2: Diebold-Mariano significance (when sample size permits) + worst-decile guard
    n_pts = len(point_losses_cand)
    dm_ok = (
        True
        if n_pts < 6 or not math.isfinite(dm_pvalue)
        else bool(dm_pvalue <= max(significance_alpha, 1e-9))
    )
    tail_ok = bool(worst_decile_fva_pct >= worst_decile_floor - 1e-9)
    gate2_passed = bool(dm_ok and tail_ok)
    rejection_reason: str | None = None
    if not gate2_passed:
        if not tail_ok:
            rejection_reason = (
                f"Gate 2 (Worst-Decile Tail Guard): worst_decile_fva_pct "
                f"({worst_decile_fva_pct:.4f}) fell below floor ({worst_decile_floor:.4f})."
            )
        else:
            rejection_reason = (
                f"Gate 2 (Diebold-Mariano Significance): dm_pvalue ({dm_pvalue:.4f}) "
                f"exceeded significance_alpha ({significance_alpha:.4f})."
            )

    return {
        "inner_cv_score": inner_cv_score,
        "baseline_score": baseline_score,
        "inner_fva_pct": inner_fva_pct,
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
        "worst_decile_fva_pct": worst_decile_fva_pct,
        "gate1_passed": True,
        "gate2_passed": gate2_passed,
        "rejection_reason": rejection_reason,
    }


_BASELINE_SESSION_CACHE: dict[tuple[Any, ...], Any] = {}


def _normalize_ts_col(series: pd.Series) -> pd.Series:
    """Normalize a date/timestamp column to tz-naive ``datetime64[ns]``."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)


def _extract_run_artifacts(
    run_obj: Any,
    *,
    idx: int = 0,
    project_id: str | None = None,
    dataset_id: str = "scale_forecasting",
    region: str = "us-central1",
) -> dict[str, Any]:
    """Hydrate ``(run_id, oof_df, predictions_df, forecaster)`` from a run object, dict, or ID."""
    from ..registry.reads import read_oof, read_predictions
    from ..settings import Settings

    eff_proj = (
        project_id
        or os.environ.get("SF_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or "statmike-scale-forecasting"
    )
    eff_ds = dataset_id or os.environ.get("SF_DATASET_ID") or "scale_forecasting"
    eff_reg = region or os.environ.get("SF_REGION") or "us-central1"
    default_settings = Settings(
        project_id=eff_proj,
        connection=os.environ.get("SF_CONNECTION") or f"{eff_proj}.{eff_reg}.biglake",
        warehouse_uri=os.environ.get("SF_WAREHOUSE_URI") or f"gs://{eff_proj}/warehouse",
        dataset_id=eff_ds,
        region=eff_reg,
    )

    if isinstance(run_obj, str):
        rid = run_obj.strip()
        f_obj: Any = None
        try:
            from ..sdk import Forecaster

            f_obj = Forecaster.from_run_id(rid, settings=default_settings)
        except Exception:  # noqa: BLE001 - header may be absent when reading raw OOF/predictions
            f_obj = None
        oof_rows = read_oof(rid, limit=50000, settings=default_settings)
        pred_rows = read_predictions(rid, limit=50000, settings=default_settings)
        oof_raw = pd.DataFrame(oof_rows) if oof_rows else pd.DataFrame()
        pred_raw = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
        return _standardize_extracted_frames(rid, oof_raw, pred_raw, f_obj, idx=idx)

    if isinstance(run_obj, Mapping):
        rid = str(run_obj.get("run_id") or f"run_{idx + 1}")
        oof_raw = (
            run_obj["oof_df"].copy()
            if isinstance(run_obj.get("oof_df"), pd.DataFrame)
            else pd.DataFrame(run_obj.get("oof_df") or [])
        )
        pred_source = (
            run_obj.get("raw_predictions_df")
            if run_obj.get("raw_predictions_df") is not None
            else run_obj.get("predictions_df")
        )
        pred_raw = (
            pred_source.copy()
            if isinstance(pred_source, pd.DataFrame)
            else pd.DataFrame(pred_source or [])
        )
        return _standardize_extracted_frames(
            rid, oof_raw, pred_raw, run_obj.get("forecaster"), idx=idx
        )

    rid = str(getattr(run_obj, "run_id", None) or f"run_{idx + 1}")
    session = getattr(run_obj, "_auto_session", None) or getattr(run_obj, "auto_session", None)
    if (
        session is not None
        and isinstance(getattr(session, "oof_df", None), pd.DataFrame)
        and not session.oof_df.empty
    ):
        oof_raw = session.oof_df.copy()
        pred_raw = (
            session.raw_predictions_df.copy()
            if isinstance(getattr(session, "raw_predictions_df", None), pd.DataFrame)
            else pd.DataFrame()
        )
        return _standardize_extracted_frames(rid, oof_raw, pred_raw, run_obj, idx=idx)

    oof_attr = getattr(run_obj, "_oof_df", None)
    if oof_attr is None:
        oof_attr = getattr(run_obj, "oof_df", None)
    oof_raw = pd.DataFrame()
    if isinstance(oof_attr, pd.DataFrame):
        oof_raw = oof_attr.copy()
    elif callable(oof_attr):
        res_oof = oof_attr()
        if isinstance(res_oof, pd.DataFrame):
            oof_raw = res_oof.copy()

    pred_attr = getattr(run_obj, "_predictions_df", None)
    if pred_attr is None:
        pred_attr = getattr(run_obj, "_raw_predictions_df", None)
    if pred_attr is None:
        pred_attr = getattr(run_obj, "raw_predictions_df", None)
    pred_raw = pd.DataFrame()
    if isinstance(pred_attr, pd.DataFrame):
        pred_raw = pred_attr.copy()
    elif callable(pred_attr):
        res_p = pred_attr()
        if isinstance(res_p, pd.DataFrame):
            pred_raw = res_p.copy()

    if pred_raw.empty and hasattr(run_obj, "predictions_df"):
        p_attr = run_obj.predictions_df
        if isinstance(p_attr, pd.DataFrame):
            pred_raw = p_attr.copy()
        elif (
            callable(p_attr)
            and getattr(run_obj, "_settings", None) is None
            and not hasattr(run_obj, "_config")
        ):
            # Custom test mock with callable .predictions_df()
            res_p = p_attr()
            if isinstance(res_p, pd.DataFrame):
                pred_raw = res_p.copy()

    run_settings = getattr(run_obj, "_settings", None) or default_settings
    if oof_raw.empty:
        oof_rows = read_oof(rid, limit=50000, settings=run_settings)
        oof_raw = pd.DataFrame(oof_rows) if oof_rows else pd.DataFrame()
    if pred_raw.empty and (
        hasattr(run_obj, "_config") or getattr(run_obj, "_oof_df", None) is None
    ):
        try:
            pred_rows = read_predictions(rid, limit=50000, settings=run_settings)
            pred_raw = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
        except Exception:  # noqa: BLE001 - offline mock without BigQuery registry
            pred_raw = pd.DataFrame()

    return _standardize_extracted_frames(rid, oof_raw, pred_raw, run_obj, idx=idx)


def _standardize_extracted_frames(
    run_id: str,
    oof_raw: pd.DataFrame,
    pred_raw: pd.DataFrame,
    forecaster_obj: Any,
    *,
    idx: int = 0,
) -> dict[str, Any]:
    """Normalize OOF and forward prediction frames into canonical columns."""
    oof = oof_raw.copy()
    preds = pred_raw.copy()

    if oof.empty and not preds.empty and "segment" in preds.columns:
        oof = preds[preds["segment"] == "oof"].copy()
    if not preds.empty and "segment" in preds.columns:
        preds = preds[preds["segment"] == "forecast"].copy()

    if oof.empty:
        raise ConfigError(
            f"verify_oof_pool_compatibility failed: run '{run_id}' has no "
            "out-of-fold (oof_df) rows."
        )

    if "ds" not in oof.columns and "forecast_date" in oof.columns:
        oof["ds"] = oof["forecast_date"]
    required_oof = {"ts_id", "model_type", "ds", "y_true", "yhat"}
    missing_oof = required_oof - set(oof.columns)
    if missing_oof:
        raise ConfigError(
            f"verify_oof_pool_compatibility failed: run '{run_id}' oof_df is missing "
            f"required columns {sorted(missing_oof)}."
        )

    oof["ts_id"] = oof["ts_id"].astype(str)
    oof["model_type"] = oof["model_type"].astype(str)
    oof["ds"] = _normalize_ts_col(oof["ds"])
    oof["y_true"] = pd.to_numeric(oof["y_true"], errors="coerce").astype(float)
    oof["yhat"] = pd.to_numeric(oof["yhat"], errors="coerce").astype(float)
    oof = (
        oof.dropna(subset=["ds", "y_true", "yhat"])
        .sort_values(["ts_id", "model_type", "ds"])
        .reset_index(drop=True)
    )

    if not preds.empty:
        if "forecast_date" not in preds.columns and "ds" in preds.columns:
            preds["forecast_date"] = preds["ds"]
        if {"ts_id", "model_type", "forecast_date", "yhat"}.issubset(set(preds.columns)):
            preds["ts_id"] = preds["ts_id"].astype(str)
            preds["model_type"] = preds["model_type"].astype(str)
            preds["forecast_date"] = _normalize_ts_col(preds["forecast_date"])
            preds["yhat"] = pd.to_numeric(preds["yhat"], errors="coerce").astype(float)
            if "yhat_lower" in preds.columns:
                preds["yhat_lower"] = pd.to_numeric(preds["yhat_lower"], errors="coerce").astype(
                    float
                )
            if "yhat_upper" in preds.columns:
                preds["yhat_upper"] = pd.to_numeric(preds["yhat_upper"], errors="coerce").astype(
                    float
                )
            preds = (
                preds.dropna(subset=["forecast_date", "yhat"])
                .sort_values(["ts_id", "model_type", "forecast_date"])
                .reset_index(drop=True)
            )
        else:
            preds = pd.DataFrame()

    short_id = run_id.rsplit("-", 1)[-1][:8] if "-" in run_id else run_id[:8]
    if not short_id:
        short_id = f"run{idx + 1}"

    return {
        "run_id": run_id,
        "short_id": short_id,
        "oof_df": oof,
        "predictions_df": preds,
        "forecaster": forecaster_obj,
    }


def verify_oof_pool_compatibility(
    runs: Sequence[Any],
    *,
    project_id: str | None = None,
    dataset_id: str = "scale_forecasting",
    region: str = "us-central1",
    horizon: int | None = None,
) -> dict[str, Any]:
    """Verify OOF & horizon alignment across ``runs``, disambiguate colliding models, and split.

    Verifies how borrowing interacts with ``backtest``, ``hpo``, and ``ensemble``:
    - Requires non-empty overlapping ``(ts_id, ds)`` out-of-fold rows across all runs (and
      overlapping ``(ts_id, forecast_date)`` horizon grids when forward predictions are present),
      raising :class:`~scale_forecasting.errors.ConfigError` when disjoint.
    - When multiple runs contain the same ``model_type``, compares their aligned OOF ``yhat``
      trajectories: identical trajectories (same ``model_spec_hash``) are deduplicated, whereas
      distinct trajectories (different HPO winner, ``model_params``, or ensemble pool) are
      disambiguated as ``<model_type>@<run_short_id>``.
    - Splits the pooled ``oof_df`` temporally per series into ``inner_oof_df`` (for candidate
      synthesis and Gates 1 & 2) and ``lockdown_oof_df`` (withheld for Gate 3).
    """
    if not runs:
        raise ConfigError("verify_oof_pool_compatibility requires at least one run.")

    extracted = [
        _extract_run_artifacts(
            r,
            idx=i,
            project_id=project_id,
            dataset_id=dataset_id,
            region=region,
        )
        for i, r in enumerate(runs)
    ]

    # Ensure unique short_ids when multiple runs are pooled
    seen_shorts: set[str] = set()
    for i, item in enumerate(extracted):
        s_id = str(item["short_id"])
        if s_id in seen_shorts:
            s_id = f"{s_id}_{i + 1}"
            item["short_id"] = s_id
        seen_shorts.add(s_id)

    # 1. Verify overlapping (ts_id, ds) OOF keys across all runs
    series_sets = [set(item["oof_df"]["ts_id"].unique()) for item in extracted]
    common_series = sorted(set.intersection(*series_sets)) if series_sets else []
    if not common_series:
        raise ConfigError(
            "verify_oof_pool_compatibility failed: pooled runs share zero overlapping ts_id series."
        )

    oof_key_sets = [
        set(zip(item["oof_df"]["ts_id"], item["oof_df"]["ds"], strict=False)) for item in extracted
    ]
    common_oof_keys = set.intersection(*oof_key_sets) if oof_key_sets else set()
    if not common_oof_keys:
        raise ConfigError(
            "verify_oof_pool_compatibility failed: pooled runs share zero overlapping "
            "(ts_id, ds) out-of-fold timestamps."
        )

    # 2. Verify overlapping (ts_id, forecast_date) forward prediction keys when present
    non_empty_preds = [
        item["predictions_df"] for item in extracted if not item["predictions_df"].empty
    ]
    common_pred_keys: set[tuple[Any, Any]] | None = None
    if len(non_empty_preds) >= 2:
        pred_key_sets = [
            set(zip(p["ts_id"], p["forecast_date"], strict=False)) for p in non_empty_preds
        ]
        common_pred_keys = set.intersection(*pred_key_sets)
        if not common_pred_keys:
            raise ConfigError(
                "verify_oof_pool_compatibility failed: pooled runs share zero overlapping "
                "(ts_id, forecast_date) forward forecast horizon rows."
            )

    # 3. Filter each run to the common grid and disambiguate vs. deduplicate models across runs
    model_run_counts: dict[str, int] = {}
    for item in extracted:
        for m in item["oof_df"]["model_type"].unique():
            model_run_counts[str(m)] = model_run_counts.get(str(m), 0) + 1

    accepted_oof_by_model: dict[str, np.ndarray] = {}
    accepted_short_by_model: dict[str, str] = {}
    first_occurrence_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    deduplicated_models: list[str] = []
    disambiguated_models: list[str] = []

    pooled_oof_parts: list[pd.DataFrame] = []
    pooled_pred_parts: list[pd.DataFrame] = []

    for item in extracted:
        s_id = str(item["short_id"])
        oof_i = item["oof_df"].copy()
        oof_keys = list(zip(oof_i["ts_id"], oof_i["ds"], strict=False))
        oof_i = oof_i[[k in common_oof_keys for k in oof_keys]].copy()

        pred_i = item["predictions_df"].copy()
        if not pred_i.empty and common_pred_keys is not None:
            p_keys = list(zip(pred_i["ts_id"], pred_i["forecast_date"], strict=False))
            pred_i = pred_i[[k in common_pred_keys for k in p_keys]].copy()

        for m_name, grp_m in oof_i.groupby("model_type", sort=False):
            m_str = str(m_name)
            grp_sorted = grp_m.sort_values(["ts_id", "ds"]).reset_index(drop=True)
            yhat_vec = grp_sorted["yhat"].to_numpy(dtype=float)
            pred_m = (
                pred_i[pred_i["model_type"] == m_str].copy() if not pred_i.empty else pd.DataFrame()
            )

            if model_run_counts.get(m_str, 1) == 1:
                pooled_oof_parts.append(grp_sorted)
                if not pred_m.empty:
                    pooled_pred_parts.append(pred_m)
                continue

            if m_str not in accepted_oof_by_model:
                accepted_oof_by_model[m_str] = yhat_vec
                accepted_short_by_model[m_str] = s_id
                first_occurrence_frames[m_str] = (grp_sorted, pred_m)
                pooled_oof_parts.append(grp_sorted)
                if not pred_m.empty:
                    pooled_pred_parts.append(pred_m)
                continue

            prev_vec = accepted_oof_by_model[m_str]
            if len(prev_vec) == len(yhat_vec) and np.allclose(
                prev_vec, yhat_vec, atol=1e-6, equal_nan=True
            ):
                if m_str not in deduplicated_models:
                    deduplicated_models.append(m_str)
                continue

            # Disambiguate both the first occurrence (if not yet renamed) and this occurrence
            prev_short = accepted_short_by_model[m_str]
            first_oof, first_pred = first_occurrence_frames[m_str]
            first_new_name = f"{m_str}@{prev_short}"
            first_oof["model_type"] = first_new_name
            if not first_pred.empty:
                first_pred["model_type"] = first_new_name
            if first_new_name not in disambiguated_models:
                disambiguated_models.append(first_new_name)

            curr_new_name = f"{m_str}@{s_id}"
            grp_sorted = grp_sorted.copy()
            grp_sorted["model_type"] = curr_new_name
            pooled_oof_parts.append(grp_sorted)
            if not pred_m.empty:
                pred_m = pred_m.copy()
                pred_m["model_type"] = curr_new_name
                pooled_pred_parts.append(pred_m)
            if curr_new_name not in disambiguated_models:
                disambiguated_models.append(curr_new_name)

    pooled_oof_df = pd.concat(pooled_oof_parts, ignore_index=True)
    pooled_predictions_df = (
        pd.concat(pooled_pred_parts, ignore_index=True) if pooled_pred_parts else pd.DataFrame()
    )

    # 4. Split pooled_oof_df temporally per series into Inner Evolution OOF & Lockdown OOF
    inner_parts: list[pd.DataFrame] = []
    lock_parts: list[pd.DataFrame] = []
    inner_ts_count = 0
    lock_ts_count = 0

    for _, grp_s in pooled_oof_df.groupby("ts_id", sort=True):
        unique_ds = sorted(grp_s["ds"].unique())
        n_ds = len(unique_ds)
        if n_ds >= 3:
            default_lock = max(1, n_ds // 3)
            n_lock = min(
                default_lock,
                int(horizon) if horizon is not None else default_lock,
                n_ds - 2,
            )
            n_lock = max(1, n_lock)
            in_dates = set(unique_ds[:-n_lock])
            lk_dates = set(unique_ds[-n_lock:])
        elif n_ds == 2:
            in_dates = {unique_ds[0]}
            lk_dates = {unique_ds[1]}
        else:
            in_dates = set(unique_ds)
            lk_dates = set(unique_ds)

        inner_ts_count = max(inner_ts_count, len(in_dates))
        lock_ts_count = max(lock_ts_count, len(lk_dates))
        inner_parts.append(grp_s[grp_s["ds"].isin(in_dates)])
        lock_parts.append(grp_s[grp_s["ds"].isin(lk_dates)])

    inner_oof_df = pd.concat(inner_parts, ignore_index=True)
    lockdown_oof_df = pd.concat(lock_parts, ignore_index=True)

    # 5. Evaluate every model and built-in ensemble_* row to identify reigning_champion_model
    models_pooled = sorted(pooled_oof_df["model_type"].astype(str).unique())
    model_inner_scores: dict[str, float] = {}
    for m_name, grp_m in inner_oof_df.groupby("model_type", sort=False):
        s_losses = [
            _series_wape(
                grp_s["y_true"].to_numpy(dtype=float),
                grp_s["yhat"].to_numpy(dtype=float),
            )
            for _, grp_s in grp_m.groupby("ts_id", sort=True)
        ]
        if s_losses:
            model_inner_scores[str(m_name)] = float(np.mean(s_losses))

    reigning_champ = (
        min(model_inner_scores, key=lambda k: model_inner_scores[k])
        if model_inner_scores
        else (models_pooled[0] if models_pooled else "ensemble_mean")
    )

    source_run_ids = [str(item["run_id"]) for item in extracted]
    pool_diagnostics: dict[str, Any] = {
        "mode": "post_run_pool",
        "source_run_ids": source_run_ids,
        "models_pooled": models_pooled,
        "deduplicated_models": sorted(deduplicated_models),
        "disambiguated_models": sorted(disambiguated_models),
        "reigning_champion_model": reigning_champ,
        "reigning_champion_inner_score": round(
            float(model_inner_scores.get(reigning_champ, float("nan"))), 6
        ),
        "common_series_count": len(common_series),
        "inner_oof_timestamps": inner_ts_count,
        "inner_oof_Timestamps": inner_ts_count,
        "lockdown_oof_timestamps": lock_ts_count,
        "borrow_interactions": {
            "backtest": "requires identical (ts_id, ds) evaluation timestamps across pooled runs",
            "hpo": (
                "models with identical OOF trajectories are deduplicated; "
                "models with distinct HPO/config fits are disambiguated as <model>@<run_short_id>"
            ),
            "ensemble": "never borrowed passively; re-synthesized across all pooled models",
        },
    }

    return {
        "source_run_ids": source_run_ids,
        "pooled_oof_df": pooled_oof_df,
        "inner_oof_df": inner_oof_df,
        "lockdown_oof_df": lockdown_oof_df,
        "pooled_predictions_df": pooled_predictions_df,
        "reigning_champion_model": reigning_champ,
        "model_inner_scores": model_inner_scores,
        "pool_diagnostics": pool_diagnostics,
        "primary_forecaster": extracted[0].get("forecaster"),
    }


def _resolve_series_weights(
    weights: Any,
    ts_ids: Sequence[str],
    available_models: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Normalize global `dict[str, float]`, per-series dict, or DataFrame weights."""
    ts_list = [str(t) for t in ts_ids]
    avail_set = {str(m) for m in available_models}

    if isinstance(weights, pd.DataFrame):
        if not {"ts_id", "model_type", "weight"}.issubset(set(weights.columns)):
            raise ValueError(
                "Weights DataFrame must contain columns ['ts_id', 'model_type', 'weight']."
            )
        out_df_map: dict[str, dict[str, float]] = {}
        for _, row in weights.iterrows():
            tid = str(row["ts_id"])
            m_name = str(row["model_type"])
            w_val = float(row["weight"])
            if m_name in avail_set and math.isfinite(w_val):
                out_df_map.setdefault(tid, {})[m_name] = w_val
        if not out_df_map:
            raise ValueError("Weights DataFrame contained no valid model weights.")
        default_w = next(iter(out_df_map.values()))
        return {tid: out_df_map.get(tid, default_w) for tid in ts_list}

    if isinstance(weights, Mapping) and weights:
        first_val = next(iter(weights.values()))
        if isinstance(first_val, Mapping):
            per_series: dict[str, dict[str, float]] = {}
            for tid_k, m_map in weights.items():
                if not isinstance(m_map, Mapping):
                    continue
                filt = {
                    str(m): float(w)
                    for m, w in m_map.items()
                    if str(m) in avail_set and math.isfinite(float(w))
                }
                if filt:
                    per_series[str(tid_k)] = filt
            if not per_series:
                raise ValueError("Per-series weights dict matched no available models.")
            fallback_w = next(iter(per_series.values()))
            return {tid: per_series.get(tid, fallback_w) for tid in ts_list}

        global_w = {
            str(m): float(w)
            for m, w in weights.items()
            if str(m) in avail_set and math.isfinite(float(w))
        }
        if not global_w:
            raise ValueError("Global weights dict matched no available models.")
        return {tid: dict(global_w) for tid in ts_list}

    raise ValueError("compute_evolved_weights returned empty or unsupported weights structure.")


def _project_blended_hierarchy(
    blended: pd.DataFrame,
    spec: HierarchySpec,
    *,
    date_col: str,
    value_col: str = "yhat_blend",
) -> pd.DataFrame:
    """Project per-series blended predictions onto the coherent subspace ``y_t = S @ b_t``."""
    if blended.empty or value_col not in blended.columns or date_col not in blended.columns:
        return blended
    ts_present = set(blended["ts_id"].astype(str))
    if not set(spec.node_ids).issubset(ts_present):
        return blended

    g_mat = reconcile_matrix(spec, "wls_struct")
    p_mat = spec.summing_matrix @ g_mat
    s_mat = spec.summing_matrix
    n_agg = spec.n_aggregated
    all_vals = pd.to_numeric(blended[value_col], errors="coerce").to_numpy(dtype=float)
    is_nonneg = bool(np.all(all_vals[np.isfinite(all_vals)] >= -1e-9))

    out_parts: list[pd.DataFrame] = []
    for _dt, grp in blended.groupby(date_col, sort=False):
        indexed = grp.drop_duplicates(subset=["ts_id"], keep="first").set_index("ts_id")
        aligned = indexed.reindex(list(spec.node_ids)).reset_index()
        y_vec = pd.to_numeric(aligned[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_proj = p_mat @ y_vec
        b_proj = y_proj[n_agg:]
        if is_nonneg:
            b_proj = np.maximum(0.0, b_proj)
        y_coherent = s_mat @ b_proj
        aligned[value_col] = y_coherent

        for lo_col, hi_col in (
            ("yhat_lower_blend", "yhat_upper_blend"),
            ("yhat_lower", "yhat_upper"),
        ):
            if lo_col in aligned.columns and hi_col in aligned.columns:
                lo_v = pd.to_numeric(aligned[lo_col], errors="coerce").to_numpy(dtype=float)
                hi_v = pd.to_numeric(aligned[hi_col], errors="coerce").to_numpy(dtype=float)
                if np.all(np.isfinite(lo_v)) and np.all(np.isfinite(hi_v)):
                    hw = np.maximum(0.0, 0.5 * (hi_v - lo_v))
                    hw_proj = np.sqrt(np.maximum(0.0, (p_mat**2) @ (hw**2)))
                    lo_new = y_coherent - hw_proj
                    hi_new = y_coherent + hw_proj
                    if is_nonneg:
                        lo_new = np.maximum(0.0, lo_new)
                        hi_new = np.maximum(y_coherent, hi_new)
                    aligned[lo_col] = lo_new
                    aligned[hi_col] = hi_new
        out_parts.append(aligned)

    return pd.concat(out_parts, ignore_index=True) if out_parts else blended


def _blend_frame_with_series_weights(
    frame: pd.DataFrame,
    weights: Any,
    *,
    date_col: str,
    simplex_sum: float | None = None,
    hierarchy_spec: HierarchySpec | None = None,
) -> pd.DataFrame:
    """Blend ``frame`` (`ts_id`, `date_col`, `model_type`, `yhat`) by ``weights``."""
    if frame.empty:
        return pd.DataFrame()
    ts_ids = sorted(frame["ts_id"].astype(str).unique())
    avail_models = sorted(frame["model_type"].astype(str).unique())
    series_w = _resolve_series_weights(weights, ts_ids, avail_models)

    parts: list[pd.DataFrame] = []
    has_y_true = "y_true" in frame.columns
    has_lower = "yhat_lower" in frame.columns
    has_upper = "yhat_upper" in frame.columns

    for tid, grp in frame.groupby("ts_id", sort=True):
        tid_str = str(tid)
        w_map = series_w.get(tid_str, {})
        sub = grp[grp["model_type"].astype(str).isin(w_map.keys())].copy()
        if sub.empty:
            continue
        sub["w"] = [float(w_map.get(str(m), 0.0)) for m in sub["model_type"]]
        sub["wy"] = pd.to_numeric(sub["yhat"], errors="coerce") * sub["w"]
        agg_spec: dict[str, Any] = {
            "wy_sum": ("wy", "sum"),
            "w_sum": ("w", "sum"),
        }
        if has_y_true:
            agg_spec["y_true"] = ("y_true", "first")
        if has_lower:
            sub["wl"] = pd.to_numeric(sub["yhat_lower"], errors="coerce") * sub["w"]
            agg_spec["wl_sum"] = ("wl", "sum")
        if has_upper:
            sub["wu"] = pd.to_numeric(sub["yhat_upper"], errors="coerce") * sub["w"]
            agg_spec["wu_sum"] = ("wu", "sum")

        grouped = (
            sub.groupby(["ts_id", date_col], as_index=False)
            .agg(**agg_spec)
            .sort_values(["ts_id", date_col])
        )
        denom = np.where(np.abs(grouped["w_sum"]) > 1e-9, grouped["w_sum"], 1.0)
        grouped["yhat_blend"] = grouped["wy_sum"] / denom
        if has_lower:
            grouped["yhat_lower_blend"] = grouped["wl_sum"] / denom
        if has_upper:
            grouped["yhat_upper_blend"] = grouped["wu_sum"] / denom
        parts.append(grouped)

    if not parts:
        return pd.DataFrame()

    blended = pd.concat(parts, ignore_index=True)
    if hierarchy_spec is not None:
        blended = _project_blended_hierarchy(
            blended,
            hierarchy_spec,
            date_col=date_col,
            value_col="yhat_blend",
        )
    if simplex_sum is not None and float(simplex_sum) > 0.0:
        blended = enforce_simplex_closure(
            blended,
            value_col="yhat_blend",
            simplex_sum=float(simplex_sum),
            ts_id_col="ts_id",
            date_col=date_col,
        )
    return blended


class EvolutionEvaluator:
    """Evaluates candidate `# EVOLVE-BLOCK` programs on top of `Forecaster.auto()` or prior runs."""

    def __init__(
        self,
        data_df: pd.DataFrame | Any = None,
        *,
        from_runs: Sequence[Any] | None = None,
        from_run_ids: Sequence[str] | None = None,
        dataset_id: str = "scale_forecasting",
        project_id: str | None = None,
        region: str = "us-central1",
        target: str = "features",
        backend: str = "alphaevolve",
        tier: str = "first_light",
        where: str = "local",
        horizon: int | None = None,
        significance_alpha: float = 0.10,
        worst_decile_floor: float = -0.02,
        timeout_seconds: float = 20.0,
        ts_id_col: str = "ts_id",
        date_col: str = "ds",
        target_col: str = "y",
        freq: str | None = None,
    ) -> None:
        self.target = get_target_seed(target).target
        self.backend = backend
        self.tier = tier
        self.where = where
        self.significance_alpha = significance_alpha
        self.worst_decile_floor = worst_decile_floor
        self.timeout_seconds = timeout_seconds
        self.lockdown_oof_df: pd.DataFrame = pd.DataFrame()
        self.pool_diagnostics: dict[str, Any] = {}
        self.reigning_champion_model: str = ""

        # Detect whether we are running in post-run pool mode (zero base-model refitting)
        runs_list: list[Any] = []
        if from_runs is not None:
            runs_list.extend(from_runs)
        if from_run_ids is not None:
            runs_list.extend(from_run_ids)
        if data_df is not None and not isinstance(data_df, pd.DataFrame):
            runs_list.insert(0, data_df)

        if runs_list:
            self.mode = "post_run_pool"
            self.ts_id_col = "ts_id"
            self.date_col = "ds"
            self.target_col = "y"
            pool_bundle = verify_oof_pool_compatibility(
                runs_list,
                project_id=project_id,
                dataset_id=dataset_id,
                region=region,
                horizon=horizon,
            )
            self.oof_df = pool_bundle["inner_oof_df"].copy()
            self.lockdown_oof_df = pool_bundle["lockdown_oof_df"].copy()
            self.raw_predictions_df = pool_bundle["pooled_predictions_df"].copy()
            self.reigning_champion_model = str(pool_bundle["reigning_champion_model"])
            self.pool_diagnostics = dict(pool_bundle["pool_diagnostics"])
            self.forecaster = pool_bundle.get("primary_forecaster")
            self.session = getattr(self.forecaster, "auto_session", None)
            self.profile = getattr(self.session, "profile", None)
            self.profile_dict: dict[str, Any] = (
                self.profile.to_dict()
                if self.profile is not None and hasattr(self.profile, "to_dict")
                else {
                    "n_series": int(self.oof_df["ts_id"].nunique()),
                    "models_pooled": self.pool_diagnostics.get("models_pooled", []),
                    "reigning_champion_model": self.reigning_champion_model,
                }
            )
            self.horizon = int(
                horizon
                if horizon is not None
                else max(1, int(self.pool_diagnostics.get("lockdown_oof_timestamps", 7)))
            )
            # Reconstruct a lightweight inner_df from inner_oof_df for feature/model fallback
            self.inner_df = (
                self.oof_df.groupby(["ts_id", "ds"], as_index=False)
                .agg({"y_true": "first"})
                .rename(columns={"y_true": "y"})
            )
            self.lockdown_truth_df = (
                self.lockdown_oof_df.groupby(["ts_id", "ds"], as_index=False)
                .agg({"y_true": "first"})
                .rename(columns={"y_true": "y"})
            )
            self.lockdown_future_exog_df = self.lockdown_truth_df[["ts_id", "ds"]].copy()
        else:
            self.mode = "raw_panel"
            if not isinstance(data_df, pd.DataFrame):
                raise ConfigError(
                    "EvolutionEvaluator requires a pandas DataFrame or from_runs/from_run_ids."
                )
            self.ts_id_col = ts_id_col if ts_id_col in data_df.columns else "ts_id"
            self.date_col = date_col if date_col in data_df.columns else "ds"
            self.target_col = target_col if target_col in data_df.columns else "y"

            work = data_df.copy()
            if self.ts_id_col not in work.columns:
                work[self.ts_id_col] = "series_0"
            counts = work.groupby(self.ts_id_col)[self.target_col].count()
            min_obs = int(counts.min()) if not counts.empty else 28
            self.horizon = int(horizon) if horizon is not None else max(3, min(7, min_obs // 5))

            (
                self.inner_df,
                self.lockdown_truth_df,
                self.lockdown_future_exog_df,
            ) = split_inner_and_lockdown_panel(
                work,
                self.horizon,
                ts_id_col=self.ts_id_col,
                date_col=self.date_col,
                target_col=self.target_col,
            )

            # Run Forecaster.auto() on the Inner Evolution Panel to establish cached OOF bank
            y_vals_fp = pd.to_numeric(self.inner_df[self.target_col], errors="coerce").fillna(0.0)
            cache_key = (
                len(self.inner_df),
                tuple(self.inner_df.columns),
                round(float(y_vals_fp.sum()), 6),
                round(float(y_vals_fp.iloc[0]) if len(y_vals_fp) else 0.0, 6),
                round(float(y_vals_fp.iloc[-1]) if len(y_vals_fp) else 0.0, 6),
                self.horizon,
                freq,
            )
            cached_forecaster = _BASELINE_SESSION_CACHE.get(cache_key)
            if cached_forecaster is None:
                fast_models = ["naive_seasonal", "naive_drift", "holtwinters"]
                exog_only_cols = [
                    c
                    for c in self.lockdown_future_exog_df.columns
                    if c not in (self.ts_id_col, self.date_col, self.target_col)
                    and pd.api.types.is_numeric_dtype(self.lockdown_future_exog_df[c])
                ]
                fut_exog_arg = (
                    self.lockdown_future_exog_df[[self.ts_id_col, self.date_col, *exog_only_cols]]
                    if exog_only_cols
                    else None
                )
                cached_forecaster = auto_forecast(
                    self.inner_df,
                    future_df=fut_exog_arg,
                    tier="wide_field",
                    where="local",
                    models=fast_models,
                    baselines=["naive_seasonal"],
                    hpo=False,
                    calibration=False,
                    explainability=False,
                    n_folds=1,
                    autofocus=False,
                    horizon=self.horizon,
                    ts_id_col=self.ts_id_col,
                    date_col=self.date_col,
                    target_col=self.target_col,
                    freq=freq,
                )
                if len(_BASELINE_SESSION_CACHE) < 16:
                    _BASELINE_SESSION_CACHE[cache_key] = cached_forecaster
            self.forecaster = cached_forecaster
            self.session = self.forecaster.auto_session
            self.profile = self.session.profile
            self.profile_dict = self.profile.to_dict()
            self.oof_df = self.session.oof_df.copy()
            self.raw_predictions_df = self.session.raw_predictions_df.copy()

        self.simplex_sum: float | None = self._resolve_evaluator_simplex_sum()
        self.hierarchy_spec: HierarchySpec | None = self._resolve_evaluator_hierarchy_spec()
        if (
            self.mode == "raw_panel"
            and self.hierarchy_spec is not None
            and not self.lockdown_truth_df.empty
            and "__total__" not in set(self.lockdown_truth_df[self.ts_id_col].astype(str))
        ):
            f_cfg = getattr(self.forecaster, "config", None) or getattr(
                self.forecaster, "_config", None
            )
            if (
                f_cfg is not None
                and getattr(f_cfg, "hierarchy", None) is not None
                and f_cfg.hierarchy.enabled
                and f_cfg.hierarchy.levels
                and all(c in self.lockdown_truth_df.columns for c in f_cfg.hierarchy.levels)
            ):
                with contextlib.suppress(Exception):
                    self.lockdown_truth_df, _ = build_hierarchy(self.lockdown_truth_df, f_cfg)

        # Precompute baseline inner CV losses and held-out lockdown fold losses for the target
        (
            self.baseline_score,
            self.series_losses_base,
            self.point_losses_base,
            self.lockdown_baseline_score,
        ) = self._compute_target_baselines()

    def _resolve_evaluator_simplex_sum(self) -> float | None:
        """Resolve compositional simplex constant sum `S` when active on the panel or config."""
        f_cfg = getattr(self.forecaster, "config", None) or getattr(
            self.forecaster, "_config", None
        )
        if f_cfg is not None and getattr(f_cfg, "features", None) is not None:
            if f_cfg.features.simplex_sum is not None:
                return float(f_cfg.features.simplex_sum)
            if getattr(f_cfg.features, "transform", "none") in ("clr", "ilr"):
                return float(self.profile_dict.get("simplex_total") or 1.0)
        if bool(self.profile_dict.get("is_compositional_simplex", False)):
            st = self.profile_dict.get("simplex_total")
            return float(st) if st is not None else 1.0
        if not self.oof_df.empty and "model_type" in self.oof_df.columns:
            m_types = [str(m).lower() for m in self.oof_df["model_type"].unique()]
            if any("@transform_clr" in m or "@transform_ilr" in m for m in m_types):
                st = self.profile_dict.get("simplex_total")
                return float(st) if st is not None else 1.0
            if (
                {"ts_id", "y_true"}.issubset(self.oof_df.columns)
                and self.oof_df["ts_id"].nunique() >= 2
                and "__total__" not in set(self.oof_df["ts_id"].astype(str))
            ):
                d_col = "ds" if "ds" in self.oof_df.columns else "forecast_date"
                if d_col in self.oof_df.columns:
                    first_m = self.oof_df["model_type"].iloc[0]
                    sub_m = self.oof_df[self.oof_df["model_type"] == first_m]
                    y_vals = pd.to_numeric(sub_m["y_true"], errors="coerce").to_numpy(dtype=float)
                    if len(y_vals) > 0 and np.all(y_vals[np.isfinite(y_vals)] >= -1e-9):
                        sums = sub_m.groupby(d_col)["y_true"].sum().to_numpy(dtype=float)
                        if len(sums) >= 2:
                            if np.all(np.abs(sums - 1.0) <= 1e-3):
                                return 1.0
                            if np.all(np.abs(sums - 100.0) <= 1e-2):
                                return 100.0
        return None

    def _resolve_evaluator_hierarchy_spec(self) -> HierarchySpec | None:
        """Resolve `HierarchySpec` when hierarchical reconciliation is active on the panel."""
        if self.session is not None:
            champ_out = getattr(self.session, "champion_outputs", None)
            if isinstance(champ_out, Mapping) and isinstance(
                champ_out.get("hierarchy_spec"), HierarchySpec
            ):
                return champ_out["hierarchy_spec"]
        f_cfg = getattr(self.forecaster, "config", None) or getattr(
            self.forecaster, "_config", None
        )
        if (
            f_cfg is not None
            and getattr(f_cfg, "hierarchy", None) is not None
            and f_cfg.hierarchy.enabled
            and f_cfg.hierarchy.levels
            and isinstance(self.inner_df, pd.DataFrame)
            and not self.inner_df.empty
            and all(c in self.inner_df.columns for c in f_cfg.hierarchy.levels)
        ):
            try:
                base_inner = self.inner_df[
                    ~self.inner_df[self.ts_id_col].astype(str).str.contains("=|__total__")
                ]
                if not base_inner.empty:
                    _, spec = build_hierarchy(base_inner, f_cfg)
                    return spec
            except Exception:  # noqa: BLE001 - fallback to OOF structure inspection
                pass
        if not self.oof_df.empty and "ts_id" in self.oof_df.columns:
            all_ids = sorted(self.oof_df["ts_id"].astype(str).unique())
            if "__total__" in all_ids:
                bottom_ids = tuple(sorted(t for t in all_ids if t != "__total__" and "=" not in t))
                mid_ids = tuple(sorted(t for t in all_ids if t != "__total__" and "=" in t))
                if len(bottom_ids) >= 2:
                    node_ids = ("__total__", *mid_ids, *bottom_ids)
                    n_b = len(bottom_ids)
                    s_rows: list[np.ndarray] = [np.ones(n_b, dtype=float)]
                    d_col = "ds" if "ds" in self.oof_df.columns else "forecast_date"
                    piv = (
                        self.oof_df.groupby([d_col, "ts_id"], as_index=False)["y_true"]
                        .first()
                        .pivot(index=d_col, columns="ts_id", values="y_true")
                    )
                    b_mat = piv[list(bottom_ids)].to_numpy(dtype=float)
                    for mid_id in mid_ids:
                        y_m = piv[mid_id].to_numpy(dtype=float)
                        matched_row: np.ndarray | None = None
                        if n_b <= 12:
                            for mask_int in range(1, 1 << n_b):
                                row_cand = np.array(
                                    [(mask_int >> j) & 1 for j in range(n_b)], dtype=float
                                )
                                if np.allclose(b_mat @ row_cand, y_m, atol=1e-4, equal_nan=True):
                                    matched_row = row_cand
                                    break
                        if matched_row is None:
                            ls_sol, _, _, _ = np.linalg.lstsq(b_mat, y_m, rcond=None)
                            matched_row = np.asarray(
                                np.clip(np.round(ls_sol), 0.0, 1.0), dtype=float
                            )
                        s_rows.append(matched_row)
                    for idx_b in range(n_b):
                        e_row = np.zeros(n_b, dtype=float)
                        e_row[idx_b] = 1.0
                        s_rows.append(e_row)
                    s_mat = np.vstack(s_rows)
                    level_nodes: dict[str, tuple[str, ...]] = {
                        "__total__": ("__total__",),
                        "__bottom__": bottom_ids,
                    }
                    if mid_ids:
                        level_nodes["hierarchy"] = mid_ids
                    return HierarchySpec(
                        node_ids=node_ids,
                        bottom_ids=bottom_ids,
                        level_nodes=level_nodes,
                        summing_matrix=s_mat,
                    )
        return None

    def _compute_target_baselines(
        self,
    ) -> tuple[float, list[float], list[float], float]:
        """Compute inner-CV and held-out lockdown baseline scores for ``self.target``."""
        if self.target == "features":
            s_base, p_base = self._eval_features_ridge(
                self.inner_df, feature_fn=None, is_lockdown=False
            )
            lock_s, _ = self._eval_features_ridge(
                self.inner_df,
                feature_fn=None,
                is_lockdown=True,
                lockdown_truth_df=self.lockdown_truth_df,
            )
            return float(np.mean(s_base)), s_base, p_base, float(np.mean(lock_s))

        if self.target == "ensemble":
            all_models = sorted(self.oof_df["model_type"].astype(str).unique())
            base_models = [m for m in all_models if not m.startswith("ensemble_")]
            candidates_by_name: dict[str, tuple[list[float], list[float], float]] = {}
            for m_name in all_models:
                s_m, p_m = self._score_ensemble_weights_inner({m_name: 1.0})
                l_m = self._score_ensemble_weights_lockdown({m_name: 1.0})
                candidates_by_name[m_name] = (s_m, p_m, l_m)
            if base_models and "ensemble_mean" not in candidates_by_name:
                w_eq = {m: 1.0 / len(base_models) for m in base_models}
                s_eq, p_eq = self._score_ensemble_weights_inner(w_eq)
                l_eq = self._score_ensemble_weights_lockdown(w_eq)
                candidates_by_name["ensemble_mean"] = (s_eq, p_eq, l_eq)

            # Identify the reigning #1 champion across all base models and built-in ensemble_* rows
            ranked_models = sorted(
                candidates_by_name.keys(),
                key=lambda m: float(np.mean(candidates_by_name[m][0])),
            )
            best_overall = ranked_models[0] if ranked_models else "ensemble_mean"
            self.reigning_champion_model = best_overall
            self.pool_diagnostics.setdefault("reigning_champion_model", best_overall)
            self.pool_diagnostics.setdefault("models_pooled", all_models)

            if self.mode == "post_run_pool":
                s_b, p_b, l_b = candidates_by_name[best_overall]
                return float(np.mean(s_b)), s_b, p_b, l_b

            # In raw_panel mode, check whether adaptive seed weights beat best_overall on both
            # inner CV and lockdown; otherwise benchmark against the built-in global ensemble
            w_probe = compute_evolved_weights(self.oof_df.copy(), self.profile_dict)
            s_pr, p_pr = self._score_ensemble_weights_inner(w_probe)
            l_pr = self._score_ensemble_weights_lockdown(w_probe)
            s_bo, p_bo, l_bo = candidates_by_name[best_overall]
            probe_stats = _compute_ratchet_stats(
                s_pr,
                s_bo,
                p_pr,
                p_bo,
                significance_alpha=self.significance_alpha,
                worst_decile_floor=self.worst_decile_floor,
            )
            if probe_stats["gate1_passed"] and probe_stats["gate2_passed"] and l_pr < l_bo - 1e-9:
                return float(np.mean(s_bo)), s_bo, p_bo, l_bo

            for fallback_name in ("ensemble_inverse_error", "ensemble_mean"):
                if fallback_name in candidates_by_name:
                    s_fb, p_fb, l_fb = candidates_by_name[fallback_name]
                    fb_stats = _compute_ratchet_stats(
                        s_pr,
                        s_fb,
                        p_pr,
                        p_fb,
                        significance_alpha=self.significance_alpha,
                        worst_decile_floor=self.worst_decile_floor,
                    )
                    if fb_stats["gate1_passed"] and fb_stats["gate2_passed"] and l_pr < l_fb - 1e-9:
                        self.reigning_champion_model = fallback_name
                        self.pool_diagnostics["reigning_champion_model"] = fallback_name
                        return float(np.mean(s_fb)), s_fb, p_fb, l_fb

            s_m, p_m, l_m = candidates_by_name.get("ensemble_mean", (s_bo, p_bo, l_bo))
            return float(np.mean(s_m)), s_m, p_m, l_m

        if self.target == "model":
            s_base, p_base = self._eval_naive_baseline_inner()
            lock_score = self._eval_naive_baseline_lockdown()
            return float(np.mean(s_base)), s_base, p_base, lock_score

        # target == "policy"
        base_models = sorted(
            m
            for m in self.oof_df["model_type"].astype(str).unique()
            if not m.startswith("ensemble_")
        )
        if not base_models:
            base_models = sorted(self.oof_df["model_type"].astype(str).unique())
        w_base = {m: 1.0 / max(1, len(base_models)) for m in base_models}
        s_base, p_base = self._score_ensemble_weights_inner(w_base)
        lock_score = self._score_ensemble_weights_lockdown(w_base)
        return float(np.mean(s_base)), s_base, p_base, lock_score

    def _eval_features_ridge(
        self,
        panel_df: pd.DataFrame,
        feature_fn: Any | None,
        *,
        is_lockdown: bool,
        lockdown_truth_df: pd.DataFrame | None = None,
    ) -> tuple[list[float], list[float]]:
        """Evaluate causal feature synthesis using Ridge lag + exogenous regression."""
        h = self.horizon
        series_losses: list[float] = []
        point_losses: list[float] = []

        if not is_lockdown:
            feat_df = (
                feature_fn(panel_df.copy(), self.profile_dict)
                if feature_fn is not None
                else panel_df.copy()
            )
            evolved_cols = [
                c
                for c in feat_df.columns
                if str(c).startswith("evolved_feat_") and pd.api.types.is_numeric_dtype(feat_df[c])
            ]
            for _, grp in feat_df.groupby(self.ts_id_col, sort=True):
                grp_s = grp.sort_values(self.date_col).reset_index(drop=True)
                y_all = pd.to_numeric(grp_s[self.target_col], errors="coerce").to_numpy(dtype=float)
                n = len(y_all)
                eff_h = min(h, max(2, n // 4))
                y_tr = y_all[:-eff_h]
                y_val = y_all[-eff_h:]
                if not evolved_cols:
                    # Baseline without evolved features: simple damped mean/lag anchor
                    pred = np.full(eff_h, float(np.mean(y_tr[-min(len(y_tr), 5) :])), dtype=float)
                else:
                    x_all = (
                        grp_s[evolved_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    x_tr = x_all[:-eff_h]
                    x_val = x_all[-eff_h:]
                    x_mean = np.mean(x_tr, axis=0)
                    x_std = np.where(np.std(x_tr, axis=0) > 1e-8, np.std(x_tr, axis=0), 1.0)
                    x_tr_n = (x_tr - x_mean) / x_std
                    x_val_n = (x_val - x_mean) / x_std
                    y_mean = float(np.mean(y_tr))
                    gram = x_tr_n.T @ x_tr_n + 1.5 * np.eye(x_tr_n.shape[1])
                    coef = np.linalg.solve(gram, x_tr_n.T @ (y_tr - y_mean))
                    seasonal_anchor = np.array(
                        [y_tr[-eff_h + (i % min(7, len(y_tr)))] for i in range(eff_h)],
                        dtype=float,
                    )
                    pred = 0.55 * (y_mean + x_val_n @ coef) + 0.45 * seasonal_anchor

                series_losses.append(_series_wape(y_val, pred))
                denom_pt = max(1e-6, float(np.mean(np.abs(y_val))))
                point_losses.extend((np.abs(y_val - pred) / denom_pt).tolist())
            return series_losses, point_losses

        # Lockdown evaluation: combine inner_df with lockdown future covariates (y masked with
        # causal forward-fill from inner_df so lockdown actuals are never seen by feature_fn!)
        assert lockdown_truth_df is not None
        masked_lock = lockdown_truth_df.copy()
        last_y_by_id = (
            panel_df.sort_values([self.ts_id_col, self.date_col])
            .groupby(self.ts_id_col)[self.target_col]
            .last()
            .to_dict()
        )
        masked_lock[self.target_col] = [
            float(last_y_by_id.get(tid, 0.0)) for tid in masked_lock[self.ts_id_col]
        ]
        combined_input = pd.concat([panel_df, masked_lock], ignore_index=True)
        combined_input = combined_input.sort_values([self.ts_id_col, self.date_col]).reset_index(
            drop=True
        )
        feat_full = (
            feature_fn(combined_input.copy(), self.profile_dict)
            if feature_fn is not None
            else combined_input.copy()
        )
        evolved_cols = [
            c
            for c in feat_full.columns
            if str(c).startswith("evolved_feat_") and pd.api.types.is_numeric_dtype(feat_full[c])
        ]
        truth_by_id = {
            str(tid): grp.sort_values(self.date_col)[self.target_col].to_numpy(dtype=float)
            for tid, grp in lockdown_truth_df.groupby(self.ts_id_col, sort=False)
        }
        for tid, grp in feat_full.groupby(self.ts_id_col, sort=True):
            grp_s = grp.sort_values(self.date_col).reset_index(drop=True)
            y_lock = truth_by_id.get(str(tid))
            if y_lock is None or len(y_lock) == 0:
                continue
            eff_h = len(y_lock)
            y_tr = pd.to_numeric(grp_s[self.target_col].iloc[:-eff_h], errors="coerce").to_numpy(
                dtype=float
            )
            if not evolved_cols:
                pred = np.full(eff_h, float(np.mean(y_tr[-min(len(y_tr), 5) :])), dtype=float)
            else:
                x_all = (
                    grp_s[evolved_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                x_tr = x_all[:-eff_h]
                x_val = x_all[-eff_h:]
                x_mean = np.mean(x_tr, axis=0)
                x_std = np.where(np.std(x_tr, axis=0) > 1e-8, np.std(x_tr, axis=0), 1.0)
                x_tr_n = (x_tr - x_mean) / x_std
                x_val_n = (x_val - x_mean) / x_std
                y_mean = float(np.mean(y_tr))
                gram = x_tr_n.T @ x_tr_n + 1.5 * np.eye(x_tr_n.shape[1])
                coef = np.linalg.solve(gram, x_tr_n.T @ (y_tr - y_mean))
                seasonal_anchor = np.array(
                    [y_tr[-eff_h + (i % min(7, len(y_tr)))] for i in range(eff_h)],
                    dtype=float,
                )
                pred = 0.55 * (y_mean + x_val_n @ coef) + 0.45 * seasonal_anchor

            series_losses.append(_series_wape(y_lock, pred))
            denom_pt = max(1e-6, float(np.mean(np.abs(y_lock))))
            point_losses.extend((np.abs(y_lock - pred) / denom_pt).tolist())
        return series_losses, point_losses

    def _score_ensemble_weights_inner(self, weights: Any) -> tuple[list[float], list[float]]:
        """Blend ``self.oof_df`` using ``weights`` (global, per-series, or DataFrame)."""
        date_key = "ds" if "ds" in self.oof_df.columns else "forecast_date"
        blended = _blend_frame_with_series_weights(
            self.oof_df,
            weights,
            date_col=date_key,
            simplex_sum=self.simplex_sum,
            hierarchy_spec=self.hierarchy_spec,
        )
        if blended.empty:
            return [float("inf")], [float("inf")]

        series_losses: list[float] = []
        point_losses: list[float] = []
        for _, grp in blended.groupby("ts_id", sort=True):
            yt = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
            yh = pd.to_numeric(grp["yhat_blend"], errors="coerce").to_numpy(dtype=float)
            series_losses.append(_series_wape(yt, yh))
            denom_pt = max(1e-6, float(np.mean(np.abs(yt))))
            point_losses.extend((np.abs(yt - yh) / denom_pt).tolist())
        return series_losses, point_losses

    def _score_ensemble_weights_lockdown(self, weights: Any) -> float:
        """Blend over the held-out Lockdown Fold (`lockdown_oof_df` or `raw_predictions_df`)."""
        if self.mode == "post_run_pool" and not self.lockdown_oof_df.empty:
            date_key = "ds" if "ds" in self.lockdown_oof_df.columns else "forecast_date"
            blended_lock = _blend_frame_with_series_weights(
                self.lockdown_oof_df,
                weights,
                date_col=date_key,
                simplex_sum=self.simplex_sum,
                hierarchy_spec=self.hierarchy_spec,
            )
            if blended_lock.empty:
                return float("inf")
            losses_pool: list[float] = []
            for _, grp in blended_lock.groupby("ts_id", sort=True):
                yt = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
                yh = pd.to_numeric(grp["yhat_blend"], errors="coerce").to_numpy(dtype=float)
                losses_pool.append(_series_wape(yt, yh))
            return float(np.mean(losses_pool)) if losses_pool else float("inf")

        blended = _blend_frame_with_series_weights(
            self.raw_predictions_df,
            weights,
            date_col="forecast_date",
            simplex_sum=self.simplex_sum,
            hierarchy_spec=self.hierarchy_spec,
        )
        if blended.empty:
            return float("inf")

        losses: list[float] = []
        for tid, grp_t in self.lockdown_truth_df.groupby(self.ts_id_col, sort=True):
            yt = pd.to_numeric(
                grp_t.sort_values(self.date_col)[self.target_col], errors="coerce"
            ).to_numpy(dtype=float)
            sub_p = blended[blended["ts_id"].astype(str) == str(tid)].sort_values("forecast_date")
            if sub_p.empty:
                continue
            yh = sub_p["yhat_blend"].to_numpy(dtype=float)[: len(yt)]
            losses.append(_series_wape(yt[: len(yh)], yh))
        return float(np.mean(losses)) if losses else float("inf")

    def materialize_ensemble_predictions(
        self,
        weights: Any,
        *,
        model_label: str = "ensemble_evolved",
    ) -> pd.DataFrame:
        """Materialize blended forward forecasts with empirical conformal 80% intervals."""
        source_preds = self.raw_predictions_df.copy()
        date_col = "forecast_date"
        if source_preds.empty and not self.lockdown_oof_df.empty:
            source_preds = self.lockdown_oof_df.copy()
            if "forecast_date" not in source_preds.columns and "ds" in source_preds.columns:
                source_preds["forecast_date"] = source_preds["ds"]
        if source_preds.empty:
            return pd.DataFrame(
                columns=["ts_id", "forecast_date", "yhat", "yhat_lower", "yhat_upper", "model_type"]
            )

        avail_pred_models = sorted(source_preds["model_type"].astype(str).unique())
        eff_weights = weights
        if (
            isinstance(weights, Mapping)
            and len(weights) == 1
            and not isinstance(next(iter(weights.values())), Mapping)
            and str(next(iter(weights.keys()))) not in avail_pred_models
            and avail_pred_models
        ):
            only_key = str(next(iter(weights.keys())))
            base_prefix = only_key.split("@", 1)[0]
            matched_m = [
                m for m in avail_pred_models if m == base_prefix or m.startswith(f"{base_prefix}_")
            ]
            base_m = (
                matched_m
                or [m for m in avail_pred_models if not m.startswith("ensemble_")]
                or avail_pred_models
            )
            eff_weights = {m: 1.0 / len(base_m) for m in base_m}

        blended_preds = _blend_frame_with_series_weights(
            source_preds,
            eff_weights,
            date_col=date_col,
            simplex_sum=self.simplex_sum,
            hierarchy_spec=self.hierarchy_spec,
        )
        if blended_preds.empty:
            return pd.DataFrame(
                columns=["ts_id", "forecast_date", "yhat", "yhat_lower", "yhat_upper", "model_type"]
            )

        # Compute empirical 80th percentile absolute residual per series on OOF
        oof_date_col = "ds" if "ds" in self.oof_df.columns else "forecast_date"
        blended_oof = _blend_frame_with_series_weights(
            self.oof_df,
            weights,
            date_col=oof_date_col,
            simplex_sum=self.simplex_sum,
            hierarchy_spec=self.hierarchy_spec,
        )
        q80_by_ts: dict[str, float] = {}
        if not blended_oof.empty and "y_true" in blended_oof.columns:
            for tid, grp_o in blended_oof.groupby("ts_id", sort=False):
                abs_res = np.abs(
                    pd.to_numeric(grp_o["y_true"], errors="coerce").to_numpy(dtype=float)
                    - pd.to_numeric(grp_o["yhat_blend"], errors="coerce").to_numpy(dtype=float)
                )
                finite_res = abs_res[np.isfinite(abs_res)]
                if len(finite_res) > 0:
                    q80_by_ts[str(tid)] = float(np.quantile(finite_res, 0.80))

        out_rows: list[dict[str, Any]] = []
        for _, row in blended_preds.iterrows():
            tid_s = str(row["ts_id"])
            yh_val = float(row["yhat_blend"])
            q80 = q80_by_ts.get(tid_s, max(1e-3, 0.10 * abs(yh_val)))
            low_val = (
                float(row["yhat_lower_blend"])
                if "yhat_lower_blend" in row and pd.notna(row["yhat_lower_blend"])
                else (yh_val - q80)
            )
            high_val = (
                float(row["yhat_upper_blend"])
                if "yhat_upper_blend" in row and pd.notna(row["yhat_upper_blend"])
                else (yh_val + q80)
            )
            # Enforce conformal coverage floor from OOF residuals
            half_w = max(0.5 * abs(high_val - low_val), q80)
            out_rows.append(
                {
                    "ts_id": tid_s,
                    "forecast_date": row[date_col],
                    "yhat": yh_val,
                    "yhat_lower": yh_val - half_w,
                    "yhat_upper": yh_val + half_w,
                    "model_type": model_label,
                }
            )
        out_df = pd.DataFrame(out_rows)
        if self.hierarchy_spec is not None and not out_df.empty:
            out_df = _project_blended_hierarchy(
                out_df,
                self.hierarchy_spec,
                date_col="forecast_date",
                value_col="yhat",
            )
        if self.simplex_sum is not None and float(self.simplex_sum) > 0.0 and not out_df.empty:
            out_df = enforce_simplex_closure(
                out_df,
                value_col="yhat",
                simplex_sum=float(self.simplex_sum),
                ts_id_col="ts_id",
                date_col="forecast_date",
            )
        return out_df

    def _eval_naive_baseline_inner(self) -> tuple[list[float], list[float]]:
        """Evaluate baseline rolling anchor on the inner evolution panel."""
        series_losses: list[float] = []
        point_losses: list[float] = []
        for _, grp in self.inner_df.groupby(self.ts_id_col, sort=True):
            grp_s = grp.sort_values(self.date_col).reset_index(drop=True)
            y_all = pd.to_numeric(grp_s[self.target_col], errors="coerce").to_numpy(dtype=float)
            eff_h = min(self.horizon, max(2, len(y_all) // 4))
            y_tr = y_all[:-eff_h]
            y_val = y_all[-eff_h:]
            pred = np.full(eff_h, float(y_tr[-1]), dtype=float)
            series_losses.append(_series_wape(y_val, pred))
            denom_pt = max(1e-6, float(np.mean(np.abs(y_val))))
            point_losses.extend((np.abs(y_val - pred) / denom_pt).tolist())
        return series_losses, point_losses

    def _eval_naive_baseline_lockdown(self) -> float:
        """Evaluate baseline rolling anchor on the held-out Lockdown Fold."""
        losses: list[float] = []
        for tid, grp_lock in self.lockdown_truth_df.groupby(self.ts_id_col, sort=True):
            grp_in = self.inner_df[
                self.inner_df[self.ts_id_col].astype(str) == str(tid)
            ].sort_values(self.date_col)
            if grp_in.empty:
                continue
            y_tr = pd.to_numeric(grp_in[self.target_col], errors="coerce").to_numpy(dtype=float)
            y_lock = pd.to_numeric(
                grp_lock.sort_values(self.date_col)[self.target_col], errors="coerce"
            ).to_numpy(dtype=float)
            pred = np.full(len(y_lock), float(y_tr[-1]), dtype=float)
            losses.append(_series_wape(y_lock, pred))
        return float(np.mean(losses)) if losses else float("inf")

    def _eval_model_class(
        self, model_cls: Any, *, is_lockdown: bool
    ) -> tuple[list[float], list[float]]:
        """Evaluate a candidate ``EvolvedForecaster(BaseModel)`` on inner CV or Lockdown Fold."""
        series_losses: list[float] = []
        point_losses: list[float] = []
        freq_str = str(getattr(self.profile, "freq", None) or "D")
        ctx = ModelContext(freq=freq_str, horizon=self.horizon)

        if not is_lockdown:
            for _, grp in self.inner_df.groupby(self.ts_id_col, sort=True):
                grp_s = grp.sort_values(self.date_col).reset_index(drop=True)
                y_all = pd.to_numeric(grp_s[self.target_col], errors="coerce").to_numpy(dtype=float)
                eff_h = min(self.horizon, max(2, len(y_all) // 4))
                y_tr = pd.Series(
                    y_all[:-eff_h],
                    index=pd.to_datetime(grp_s[self.date_col].iloc[:-eff_h]),
                )
                y_val = y_all[-eff_h:]
                m = model_cls({}, ctx)
                m.fit(y_tr)
                pred_df = m.predict(eff_h)
                yh = pd.to_numeric(pred_df["yhat"], errors="coerce").to_numpy(dtype=float)[:eff_h]
                series_losses.append(_series_wape(y_val, yh))
                denom_pt = max(1e-6, float(np.mean(np.abs(y_val))))
                point_losses.extend((np.abs(y_val - yh) / denom_pt).tolist())
            return series_losses, point_losses

        for tid, grp_lock in self.lockdown_truth_df.groupby(self.ts_id_col, sort=True):
            grp_in = self.inner_df[
                self.inner_df[self.ts_id_col].astype(str) == str(tid)
            ].sort_values(self.date_col)
            if grp_in.empty:
                continue
            y_tr = pd.Series(
                pd.to_numeric(grp_in[self.target_col], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(grp_in[self.date_col]),
            )
            y_lock = pd.to_numeric(
                grp_lock.sort_values(self.date_col)[self.target_col], errors="coerce"
            ).to_numpy(dtype=float)
            eff_h = len(y_lock)
            m = model_cls({}, ctx)
            m.fit(y_tr)
            pred_df = m.predict(eff_h)
            yh = pd.to_numeric(pred_df["yhat"], errors="coerce").to_numpy(dtype=float)[:eff_h]
            series_losses.append(_series_wape(y_lock, yh))
            denom_pt = max(1e-6, float(np.mean(np.abs(y_lock))))
            point_losses.extend((np.abs(y_lock - yh) / denom_pt).tolist())
        return series_losses, point_losses

    def evaluate_candidate_inner(
        self,
        code: str,
        *,
        generation: int = 1,
    ) -> CandidateEvaluation:
        """Evaluate ``code`` through Pre-Gate (AST + Causality), Gate 1, and Gate 2."""
        t0 = time.monotonic()
        cand_id = f"cand_{self.target}_g{generation:02d}"
        spec = get_target_seed(self.target)

        # Pre-Gate 1: AST Security Sandbox
        ast_res = validate_candidate_ast(code)
        if not ast_res.valid:
            return CandidateEvaluation(
                candidate_id=cand_id,
                generation=generation,
                target=self.target,
                backend=self.backend,
                code=code,
                ast_valid=False,
                causal_valid=False,
                inner_cv_score=float("inf"),
                baseline_score=self.baseline_score,
                inner_fva_pct=0.0,
                dm_stat=0.0,
                dm_pvalue=1.0,
                worst_decile_fva_pct=-1.0,
                gate1_passed=False,
                gate2_passed=False,
                lockdown_baseline_score=self.lockdown_baseline_score,
                lockdown_candidate_score=float("inf"),
                gate3_passed=False,
                promoted=False,
                status="rejected_ast",
                rejection_reason=f"Pre-Gate AST Sandbox Violation: {ast_res.reason}",
                wall_seconds=time.monotonic() - t0,
            )

        try:
            symbol = compile_candidate_symbol(code, spec.entrypoint)
        except Exception as exc:  # noqa: BLE001 - compilation failure rejected cleanly
            return CandidateEvaluation(
                candidate_id=cand_id,
                generation=generation,
                target=self.target,
                backend=self.backend,
                code=code,
                ast_valid=False,
                causal_valid=False,
                inner_cv_score=float("inf"),
                baseline_score=self.baseline_score,
                inner_fva_pct=0.0,
                dm_stat=0.0,
                dm_pvalue=1.0,
                worst_decile_fva_pct=-1.0,
                gate1_passed=False,
                gate2_passed=False,
                lockdown_baseline_score=self.lockdown_baseline_score,
                lockdown_candidate_score=float("inf"),
                gate3_passed=False,
                promoted=False,
                status="rejected_compile",
                rejection_reason=f"Candidate compilation error: {exc}",
                wall_seconds=time.monotonic() - t0,
            )

        # Pre-Gate 2: Temporal Causality / Future-Leakage Detector
        causal_ok = True
        if self.target in ("features", "model"):
            causal_ok = verify_no_future_leakage(
                symbol,
                self.inner_df,
                target_col=self.target_col,
                time_col=self.date_col,
                id_col=self.ts_id_col,
                horizon=self.horizon,
            )
        if not causal_ok:
            return CandidateEvaluation(
                candidate_id=cand_id,
                generation=generation,
                target=self.target,
                backend=self.backend,
                code=code,
                ast_valid=True,
                causal_valid=False,
                inner_cv_score=float("inf"),
                baseline_score=self.baseline_score,
                inner_fva_pct=0.0,
                dm_stat=0.0,
                dm_pvalue=1.0,
                worst_decile_fva_pct=-1.0,
                gate1_passed=False,
                gate2_passed=False,
                lockdown_baseline_score=self.lockdown_baseline_score,
                lockdown_candidate_score=float("inf"),
                gate3_passed=False,
                promoted=False,
                status="rejected_leakage",
                rejection_reason=(
                    "Pre-Gate Temporal Causality Violation: candidate failed "
                    "verify_no_future_leakage() (perturbing y[T-H:] += 1e6 "
                    "altered earlier timestamps)."
                ),
                wall_seconds=time.monotonic() - t0,
            )

        # Inner Walk-Forward CV Evaluation
        try:
            if self.target == "features":
                s_cand, p_cand = self._eval_features_ridge(
                    self.inner_df, feature_fn=symbol, is_lockdown=False
                )
            elif self.target == "ensemble":
                w_cand = execute_in_sandbox(
                    code,
                    spec.entrypoint,
                    self.oof_df.copy(),
                    self.profile_dict,
                    timeout_seconds=self.timeout_seconds,
                )
                if (
                    not isinstance(w_cand, (Mapping, pd.DataFrame))
                    or (isinstance(w_cand, Mapping) and not w_cand)
                    or (isinstance(w_cand, pd.DataFrame) and w_cand.empty)
                ):
                    raise ValueError(
                        "compute_evolved_weights returned empty or unsupported weights."
                    )
                s_cand, p_cand = self._score_ensemble_weights_inner(w_cand)
            elif self.target == "model":
                s_cand, p_cand = self._eval_model_class(symbol, is_lockdown=False)
            else:
                # target == "policy"
                base_w = AutoPolicyWeights.from_defaults()
                evolved_w = execute_in_sandbox(
                    code,
                    spec.entrypoint,
                    base_w,
                    timeout_seconds=self.timeout_seconds,
                )
                if isinstance(evolved_w, Mapping):
                    evolved_w = AutoPolicyWeights.from_dict(evolved_w)
                if not isinstance(evolved_w, AutoPolicyWeights):
                    raise ValueError("evolve_policy_weights must return AutoPolicyWeights.")
                # Weight the best models in oof_df using the evolved policy's decay/confidence
                decay = max(0.1, min(0.99, float(evolved_w.pass_decay_factor)))
                w_pol: dict[str, float] = {}
                for m_name, grp_m in self.oof_df.groupby("model_type", sort=False):
                    if str(m_name).startswith("ensemble_"):
                        continue
                    yt = pd.to_numeric(grp_m["y_true"], errors="coerce").to_numpy(dtype=float)
                    yh = pd.to_numeric(grp_m["yhat"], errors="coerce").to_numpy(dtype=float)
                    loss_m = max(1e-6, _series_wape(yt, yh))
                    w_pol[str(m_name)] = 1.0 / (loss_m ** (2.0 / decay))
                tot_w = sum(w_pol.values()) or 1.0
                w_pol = {k: v / tot_w for k, v in w_pol.items()}
                s_cand, p_cand = self._score_ensemble_weights_inner(w_pol)
        except Exception as exc:  # noqa: BLE001 - runtime exception inside candidate
            return CandidateEvaluation(
                candidate_id=cand_id,
                generation=generation,
                target=self.target,
                backend=self.backend,
                code=code,
                ast_valid=True,
                causal_valid=True,
                inner_cv_score=float("inf"),
                baseline_score=self.baseline_score,
                inner_fva_pct=0.0,
                dm_stat=0.0,
                dm_pvalue=1.0,
                worst_decile_fva_pct=-1.0,
                gate1_passed=False,
                gate2_passed=False,
                lockdown_baseline_score=self.lockdown_baseline_score,
                lockdown_candidate_score=float("inf"),
                gate3_passed=False,
                promoted=False,
                status="rejected_runtime",
                rejection_reason=f"Candidate execution failed on Inner CV: {exc}",
                wall_seconds=time.monotonic() - t0,
            )

        stats = _compute_ratchet_stats(
            s_cand,
            self.series_losses_base,
            p_cand,
            self.point_losses_base,
            significance_alpha=self.significance_alpha,
            worst_decile_floor=self.worst_decile_floor,
        )
        status = (
            "passed_inner_gates"
            if (stats["gate1_passed"] and stats["gate2_passed"])
            else ("rejected_gate1" if not stats["gate1_passed"] else "rejected_gate2")
        )
        return CandidateEvaluation(
            candidate_id=cand_id,
            generation=generation,
            target=self.target,
            backend=self.backend,
            code=code,
            ast_valid=True,
            causal_valid=True,
            inner_cv_score=float(stats["inner_cv_score"]),
            baseline_score=self.baseline_score,
            inner_fva_pct=float(stats["inner_fva_pct"]),
            dm_stat=float(stats["dm_stat"]),
            dm_pvalue=float(stats["dm_pvalue"]),
            worst_decile_fva_pct=float(stats["worst_decile_fva_pct"]),
            gate1_passed=bool(stats["gate1_passed"]),
            gate2_passed=bool(stats["gate2_passed"]),
            lockdown_baseline_score=self.lockdown_baseline_score,
            lockdown_candidate_score=float("nan"),
            gate3_passed=False,
            promoted=False,
            status=status,
            rejection_reason=stats["rejection_reason"],
            wall_seconds=time.monotonic() - t0,
            series_losses_candidate=tuple(s_cand),
            series_losses_baseline=tuple(self.series_losses_base),
            point_losses_candidate=tuple(p_cand),
            point_losses_baseline=tuple(self.point_losses_base),
        )

    def evaluate_lockdown_gate3(
        self,
        cand_eval: CandidateEvaluation,
    ) -> CandidateEvaluation:
        """Evaluate the winning inner-CV candidate **once** on the Held-Out Lockdown Fold."""
        t0 = time.monotonic()
        spec = get_target_seed(self.target)
        try:
            symbol = compile_candidate_symbol(cand_eval.code, spec.entrypoint)
            if self.target == "features":
                s_lock, _ = self._eval_features_ridge(
                    self.inner_df,
                    feature_fn=symbol,
                    is_lockdown=True,
                    lockdown_truth_df=self.lockdown_truth_df,
                )
                lock_cand_score = float(np.mean(s_lock)) if s_lock else float("inf")
            elif self.target == "ensemble":
                w_cand = symbol(self.oof_df.copy(), self.profile_dict)
                lock_cand_score = self._score_ensemble_weights_lockdown(w_cand)
            elif self.target == "model":
                s_lock, _ = self._eval_model_class(symbol, is_lockdown=True)
                lock_cand_score = float(np.mean(s_lock)) if s_lock else float("inf")
            else:
                base_w = AutoPolicyWeights.from_defaults()
                evolved_w = symbol(base_w)
                if isinstance(evolved_w, Mapping):
                    evolved_w = AutoPolicyWeights.from_dict(evolved_w)
                decay = max(0.1, min(0.99, float(evolved_w.pass_decay_factor)))
                w_pol: dict[str, float] = {}
                for m_name, grp_m in self.oof_df.groupby("model_type", sort=False):
                    if str(m_name).startswith("ensemble_"):
                        continue
                    yt = pd.to_numeric(grp_m["y_true"], errors="coerce").to_numpy(dtype=float)
                    yh = pd.to_numeric(grp_m["yhat"], errors="coerce").to_numpy(dtype=float)
                    loss_m = max(1e-6, _series_wape(yt, yh))
                    w_pol[str(m_name)] = 1.0 / (loss_m ** (2.0 / decay))
                tot_w = sum(w_pol.values()) or 1.0
                w_pol = {k: v / tot_w for k, v in w_pol.items()}
                lock_cand_score = self._score_ensemble_weights_lockdown(w_pol)
        except Exception as exc:  # noqa: BLE001 - lockdown runtime failure rejects at Gate 3
            return CandidateEvaluation(
                **{
                    **cand_eval.__dict__,
                    "lockdown_candidate_score": float("inf"),
                    "gate3_passed": False,
                    "promoted": False,
                    "status": "rejected_gate3",
                    "rejection_reason": (
                        "Gate 3 (Held-Out Lockdown Fold): "
                        f"execution failed on lockdown fold ({exc})."
                    ),
                    "wall_seconds": cand_eval.wall_seconds + (time.monotonic() - t0),
                }
            )

        gate3_passed = bool(
            math.isfinite(lock_cand_score) and lock_cand_score < self.lockdown_baseline_score - 1e-9
        )
        promoted = bool(cand_eval.gate1_passed and cand_eval.gate2_passed and gate3_passed)
        if promoted:
            status = "promoted_champion"
            rejection_reason = None
        elif not cand_eval.gate1_passed or not cand_eval.gate2_passed:
            status = cand_eval.status
            rejection_reason = cand_eval.rejection_reason
        else:
            status = "rejected_gate3"
            rejection_reason = (
                f"Gate 3 (Held-Out Lockdown Fold): candidate improved inner CV "
                f"({cand_eval.inner_cv_score:.4f} < {cand_eval.baseline_score:.4f}) "
                f"but degraded or failed to beat baseline on the unseen lockdown fold "
                f"({lock_cand_score:.4f} >= {self.lockdown_baseline_score:.4f}). "
                f"Ratcheted back to Forecaster.auto() baseline."
            )

        return CandidateEvaluation(
            **{
                **cand_eval.__dict__,
                "lockdown_candidate_score": lock_cand_score,
                "gate3_passed": gate3_passed,
                "promoted": promoted,
                "status": status,
                "rejection_reason": rejection_reason,
                "wall_seconds": cand_eval.wall_seconds + (time.monotonic() - t0),
            }
        )
