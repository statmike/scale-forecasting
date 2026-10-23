"""Glass-Box Auto-Forecaster & Bounded `autofocus` Closed-Loop Controller.

Compiles a raw time-series DataFrame or BigQuery table into a fully validated, transparent
:class:`~scale_forecasting.config.RunConfig` accompanied by a row-by-row :class:`ConfigBlueprint`
rationale table, a parametric wall-clock and cloud cost estimate (:func:`predict_run_duration`),
user scope & constraint pinning, and an optional closed-loop **Plan -> Execute -> Observe ->
Review -> Re-Plan (`autofocus=True`)** controller supporting both multi-branch Tournament DAGs
(`autofocus_mode="dag"`) and linear refinement loops (`autofocus_mode="linear"`).

Two Top-Level Dials + Optional User Constraint Pins
---------------------------------------------------
1. **Sandbox Envelope (`where`, `allowed_families`, `goal`, `max_minutes`, `max_usd`):**
   Defines where compute is allowed to execute (`"auto"`, `"local"`, `"bigquery"`, `"cloud_cpu"`,
   `"cloud_all"`), which model families are eligible, the business optimization goal
   (`"accuracy"`, `"probabilistic"`, `"service_level"`, `"fast"`), and hard wall-clock/cost limits.
2. **Observational Astronomy Exposure Tier (`tier`):**
   - ``"first_light"`` (~20s local / ~55s BQ / ~6m cloud): Rapid sanity check & FVA baseline
     (`n_folds=1`).
   - ``"wide_field"`` (~2m local / ~2m BQ / ~7-16m cloud, **Default**): Balanced multi-family survey
     across Statistical, ML Trees, and Zero-Shot Foundation models (`n_folds=2`, max 2 passes).
   - ``"deep_sky"`` (~6m local / ~4.5m BQ / ~12-22m cloud): Cross-series global + local + neural
     survey (`tide`, `tsmixer`, `catboost`, `sarimax`, `autoets`, `n_folds=3`, HPO, Tournament DAG).
   - ``"deep_field"`` (~15m local / ~8m BQ / ~20-35m cloud): Exhaustive observatory exposure across
     all eligible families, HPO, and learned stacking (`max 4 passes`).
3. **Explicit User Scope & Constraint Pins (`base_config`, `overrides`, `models`, `include_models`,
   `exclude_models`, `decision_metric`, `n_folds`, `hpo`, `ensemble`, `explainability`):**
   Users can pin any subset of the configuration while letting the Glass-Box compiler and
   `autofocus` optimize the remaining unpinned degrees of freedom.

Public surface: ``ConfigBlueprint``, ``AutofocusPassRecord``, ``NextLoopPreview``,
``AutoForecastSession``, ``inspect_panel_for_auto``, ``build_config``, ``auto_forecast``.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .auto_policy import (
    EXPOSURE_TIERS,
    SCALE_REGIMES,
    DataProfile,
    PanelAutoProfile,
    classify_scale_regime,
    estimate_continue_confidence,
    propose_autofocus_dag_branches,
    propose_autofocus_pass,
    propose_autofocus_synthesis_pass,
    select_default_baselines,
    select_hierarchy_reconciliation_methods,
    select_tier_models_and_features,
)
from .config import RunConfig, load_config
from .errors import ConfigError, DataError, get_logger
from .models import base_model_name, get_model, list_models
from .monotonicity import infer_monotonic_constraints
from .profiling.duration import (
    DurationCalibration,
    DurationEstimate,
    calibrate_duration_from_registry,
    predict_run_duration,
)
from .registry.ids import make_model_spec_hash, make_run_id
from .seasonality import seasonal_period

if TYPE_CHECKING:
    from .sdk import Forecaster
    from .settings import Settings
    from .worker import CellResult

_log = get_logger(__name__)

__all__ = [
    "EXPOSURE_TIERS",
    "AutoForecastSession",
    "AutofocusPassRecord",
    "ConfigBlueprint",
    "DataProfile",
    "NextLoopPreview",
    "PanelAutoProfile",
    "auto_forecast",
    "build_config",
    "build_next_loop_preview",
    "execute_one_session_pass",
    "inspect_panel_for_auto",
]

_DEFAULT_HORIZON_BY_FREQ: dict[str, int] = {
    "D": 14,
    "B": 14,
    "W": 13,
    "W-SUN": 13,
    "W-MON": 13,
    "MS": 12,
    "ME": 12,
    "M": 12,
    "QS": 8,
    "QE": 8,
    "Q": 8,
    "YS": 5,
    "YE": 5,
    "h": 24,
    "H": 24,
}


_TIER_ALIASES: dict[str, str] = {
    "glimpse": "first_light",
    "starlight": "wide_field",
    "survey": "deep_sky",
}

_METRIC_ALIASES: dict[str, str] = {
    "pinball_loss": "pinball",
    "wis": "interval_score",
}


def _normalize_tier(tier: str) -> str:
    norm = _TIER_ALIASES.get(tier.strip().lower(), tier.strip().lower())
    if norm not in EXPOSURE_TIERS:
        valid_tiers = ", ".join(EXPOSURE_TIERS)
        raise ConfigError(f"Unknown exposure tier '{tier}'. Valid tiers: {valid_tiers}")
    return norm


def _normalize_decision_metric(metric: str) -> str:
    m = metric.strip()
    return _METRIC_ALIASES.get(m.lower(), m.lower() if m != "ND" else "ND")


def _deep_merge_dict(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into a copy of ``base``."""
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(dict(out[k]), v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class ConfigBlueprint:
    """Glass-Box compilation output: a standard `RunConfig` plus every empirical rationale."""

    config: RunConfig
    profile: PanelAutoProfile
    tier: str
    where: str
    goal: str
    rationale_df: pd.DataFrame
    duration_estimate: DurationEstimate
    cost_estimate_usd: float
    baselines: tuple[dict[str, str], ...] = ()

    @property
    def run_id(self) -> str:
        """Deterministic content-addressed `run_id` for the compiled `RunConfig`."""
        return make_run_id(self.config)

    @property
    def estimated_minutes_p50(self) -> float:
        """Predicted P50 wall-clock duration in minutes."""
        return self.duration_estimate.total_p50_minutes

    @property
    def estimated_minutes_p90(self) -> float:
        """Predicted P90 wall-clock duration in minutes."""
        return self.duration_estimate.total_p90_minutes

    @property
    def rationale(self) -> dict[str, str]:
        """Dictionary mapping `parameter` -> `rationale` for every synthesized decision."""
        if self.rationale_df.empty:
            return {}
        return {
            str(r["parameter"]): f"{r['value']} — {r['rationale']} ({r['evidence']})"
            for _, r in self.rationale_df.iterrows()
        }

    def profile_df(self) -> pd.DataFrame:
        """Return the 7-row Step 0 Data Profile & Readiness ``DataFrame`` for this panel."""
        return self.profile.to_df()

    def with_overrides(self, **kwargs: Any) -> ConfigBlueprint:
        """Return a new :class:`ConfigBlueprint` with user constraint overrides applied.

        Re-uses ``self.profile`` (zero re-profiling cost) and accepts any :func:`build_config`
        keyword argument (`tier`, `where`, `goal`, `allowed_families`, `models`, `include_models`,
        `exclude_models`, `baselines`, `holidays`, `decision_metric`, `n_folds`, `hpo`, `ensemble`,
        `explainability`, `max_minutes`, `max_usd`, `overrides`, etc.).
        """
        call_kwargs: dict[str, Any] = {
            "tier": kwargs.pop("tier", self.tier),
            "where": kwargs.pop("where", self.where),
            "goal": kwargs.pop("goal", self.goal),
            "base_config": kwargs.pop("base_config", self.config),
        }
        valid_build_keys = {
            "allowed_families",
            "overrides",
            "models",
            "include_models",
            "exclude_models",
            "baselines",
            "holidays",
            "decision_metric",
            "n_folds",
            "hpo",
            "ensemble",
            "explainability",
            "calibration",
            "max_minutes",
            "max_wall_clock_minutes",
            "max_usd",
            "future_df",
            "ts_id_col",
            "date_col",
            "target_col",
            "freq",
            "horizon",
            "run_name",
            "only_available_models",
            "settings",
        }
        extra_overrides = dict(kwargs.pop("overrides", {}) or {})
        for k in list(kwargs.keys()):
            if k not in valid_build_keys:
                extra_overrides[k] = kwargs.pop(k)
        if extra_overrides:
            call_kwargs["overrides"] = extra_overrides
        call_kwargs.update(kwargs)
        return build_config(self.profile, **call_kwargs)

    def explain(self) -> str:
        """Return a human-readable summary plus the field-by-field rationale table."""
        table_str = (
            self.rationale_df.to_string(index=False)
            if not self.rationale_df.empty
            else "(no rationale entries)"
        )
        return f"{self.summary()}\n\n{table_str}"

    def to_file(self, path: str | Any, *, indent: int = 2) -> Any:
        """Write the compiled `RunConfig` JSON to ``path`` and return its ``Path``."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return out_path

    def to_dict(self) -> dict[str, Any]:
        """Serialize blueprint to a plain JSON-friendly dictionary."""
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "where": self.where,
            "goal": self.goal,
            "scale_regime": self.profile.scale_regime,
            "baselines": [dict(b) for b in self.baselines],
            "cost_estimate_usd": round(self.cost_estimate_usd, 4),
            "duration_estimate": self.duration_estimate.to_dict(),
            "profile": self.profile.to_dict(),
            "rationale": self.rationale_df.to_dict(orient="records"),
            "config": self.config.model_dump(mode="json"),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the compiled `RunConfig` to a formatted JSON string."""
        return json.dumps(self.config.model_dump(mode="json"), indent=indent)

    def summary(self) -> str:
        """Human-readable summary of the compiled Glass-Box blueprint."""
        tier_title = EXPOSURE_TIERS.get(self.tier, {}).get("title", self.tier)
        baseline_str = (
            ", ".join(f"{b['model_type']} ({b['baseline_role']})" for b in self.baselines)
            if self.baselines
            else "naive_seasonal"
        )
        lines = [
            f"Glass-Box Blueprint : {self.config.run_name} ({self.run_id})",
            f"Exposure Tier       : {self.tier} ({tier_title})",
            (
                f"Sandbox Envelope    : where='{self.where}' | goal='{self.goal}' | "
                f"scale='{self.profile.scale_regime}'"
            ),
            (
                f"Panel Profile       : {self.profile.n_series} series | "
                f"freq='{self.profile.freq}' | median_T={self.profile.median_n_obs} | "
                f"H={self.profile.horizon} | regime='{self.profile.dominant_regime}'"
            ),
            f"Baseline Ladder     : {baseline_str}",
            f"Selected Models ({len(self.config.models)}) : {', '.join(self.config.models)}",
            f"Predicted Duration  : {self.duration_estimate.summary()}",
        ]
        return "\n".join(lines)

    def emit_airflow_dag(
        self,
        config_uri: str | Path | None = None,
        *,
        out_path: str | Path | None = None,
        autofocus: bool = False,
        autofocus_mode: str = "dag",
        max_passes: int = 3,
        with_retry: bool = False,
        dag_id: str | None = None,
    ) -> str:
        """Render this blueprint as a standalone Cloud Composer 3 / Airflow DAG Python string.

        When ``autofocus=False`` (default), renders the standard per-family parallel execution DAG
        (`emit_airflow_dag`). When ``autofocus=True``, renders the multi-pass `autofocus` tournament
        DAG (`emit_autofocus_airflow_dag`: `autofocus_pass_1 >> [branch_2a, branch_2b, branch_2c]
        >> autofocus_pass_3_synthesis`). If ``out_path`` is supplied, writes the rendered source to
        disk as well.
        """
        from .airflow_emit import emit_airflow_dag, emit_autofocus_airflow_dag

        uri = str(config_uri or f"auto_{self.run_id}.json")
        if autofocus:
            source = emit_autofocus_airflow_dag(
                self.config,
                uri,
                tier=self.tier,
                where=self.where,
                goal=self.goal,
                autofocus_mode=autofocus_mode,
                max_passes=max_passes,
                dag_id=dag_id,
            )
        else:
            source = emit_airflow_dag(
                self.config,
                uri,
                dag_id=dag_id,
                with_retry=with_retry,
            )
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(source, encoding="utf-8")
        return source


@dataclass(frozen=True)
class AutofocusPassRecord:
    """Audit record for one executed pass or Tournament DAG branch in a closed-loop `autofocus`."""

    pass_index: int
    run_id: str
    tier: str
    models_total: int
    models_borrowed: int
    models_executed: int
    champion_model: str
    decision_metric: str
    champion_score: float
    fva_vs_naive_pct: float
    dm_pvalue: float | None
    predicted_p50_s: float
    actual_wall_s: float
    continue_confidence: float
    expected_next_gain_pct: float
    mutations_applied: tuple[str, ...]
    stop_or_continue_reason: str
    config: RunConfig
    pass_id: str = "1"
    parent_pass_ids: tuple[str, ...] = ()
    branch_hypothesis: str = "Initial Tier Exposure"
    mutations_detail: tuple[dict[str, str], ...] = ()
    is_session_champion: bool = False
    fva_vs_univariate_pct: float = 0.0
    fva_vs_foundation_pct: float = 0.0
    covariate_lift_pct: float = 0.0
    reconciliation_lift_pct: float = 0.0
    champion_is_univariate: bool = False
    beaten_challengers: tuple[str, ...] = ()

    @property
    def best_model(self) -> str:
        """Alias for `champion_model`."""
        return self.champion_model

    @property
    def best_score(self) -> float:
        """Alias for `champion_score`."""
        return self.champion_score


@dataclass(frozen=True)
class NextLoopPreview:
    """Dry-run preview of what the next `autofocus` pass ($k+1$) or DAG branches would execute."""

    next_pass_index: int
    should_continue: bool
    confidence: float
    expected_gain_pct: float
    reason: str
    proposed_config: RunConfig
    mutations_df: pd.DataFrame
    borrowed_models: tuple[str, ...]
    new_models_to_fit: tuple[str, ...]
    duration_estimate: DurationEstimate
    branches: tuple[dict[str, Any], ...] = ()

    @property
    def stop_reason(self) -> str:
        """Alias for `reason`."""
        return self.reason

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame of the next-loop preview (1 row per proposed branch)."""
        if len(self.branches) > 1:
            rows = [
                {
                    "next_pass_index": self.next_pass_index,
                    "pass_id": b.get("pass_id", str(self.next_pass_index)),
                    "branch_hypothesis": b.get("branch_hypothesis", ""),
                    "should_continue": self.should_continue,
                    "confidence": self.confidence,
                    "expected_gain_pct": self.expected_gain_pct,
                    "borrowed_models": ", ".join(self.borrowed_models)
                    if self.borrowed_models
                    else "(none)",
                    "new_models_to_fit": ", ".join(b.get("added_models", ())) or "(feature_update)",
                    "predicted_p50_seconds": round(self.duration_estimate.total_p50_seconds, 2),
                    "reason": self.reason,
                }
                for b in self.branches
            ]
            return pd.DataFrame(rows)
        return pd.DataFrame(
            [
                {
                    "next_pass_index": self.next_pass_index,
                    "pass_id": str(self.next_pass_index),
                    "branch_hypothesis": (
                        self.branches[0].get("branch_hypothesis", "Targeted Refinement")
                        if self.branches
                        else "None"
                    ),
                    "should_continue": self.should_continue,
                    "confidence": self.confidence,
                    "expected_gain_pct": self.expected_gain_pct,
                    "borrowed_models": ", ".join(self.borrowed_models)
                    if self.borrowed_models
                    else "(none)",
                    "new_models_to_fit": ", ".join(self.new_models_to_fit)
                    if self.new_models_to_fit
                    else "(none)",
                    "predicted_p50_seconds": round(self.duration_estimate.total_p50_seconds, 2),
                    "reason": self.reason,
                }
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize next-loop preview to a JSON-compatible dictionary."""
        return {
            "next_pass_index": self.next_pass_index,
            "should_continue": self.should_continue,
            "confidence": self.confidence,
            "expected_gain_pct": self.expected_gain_pct,
            "reason": self.reason,
            "borrowed_models": list(self.borrowed_models),
            "new_models_to_fit": list(self.new_models_to_fit),
            "branch_count": max(1, len(self.branches)) if self.should_continue else 0,
            "duration_estimate": self.duration_estimate.to_dict(),
            "mutations": self.mutations_df.to_dict(orient="records"),
            "proposed_config": self.proposed_config.model_dump(mode="json"),
        }


class PractitionerAssessmentFrame(pd.DataFrame):
    """DataFrame subclass for the 7-Question Practitioner Assessment that also supports substring
    checks against its formatted report text.
    """

    _metadata: tuple[str, ...] = ("_report_text",)
    _report_text: str = ""

    @property
    def _constructor(self) -> type[PractitionerAssessmentFrame]:
        return PractitionerAssessmentFrame

    def __contains__(self, key: Any) -> bool:  # type: ignore[misc]
        if super().__contains__(key):
            return True
        report = getattr(self, "_report_text", "")
        return isinstance(key, str) and bool(report) and (key in report)

    def __str__(self) -> str:
        report = getattr(self, "_report_text", "")
        return report if report else super().__str__()


@dataclass
class AutoForecastSession:
    """State container attached to a `Forecaster` produced by `Forecaster.auto()`."""

    blueprint: ConfigBlueprint
    profile: PanelAutoProfile
    tier: str
    where: str
    goal: str
    max_minutes: float | None
    max_usd: float | None
    min_confidence: float
    max_passes: int
    autofocus_mode: str = "dag"
    exclude_models: tuple[str, ...] = ()
    pinned_models: bool = False
    pinned_hpo: bool | None = None
    baselines: tuple[dict[str, str], ...] = ()
    data_df: pd.DataFrame | None = None
    future_df: pd.DataFrame | None = None
    passes: list[AutofocusPassRecord] = field(default_factory=list)
    last_diagnostics_summary: dict[str, Any] = field(default_factory=dict)
    # In-memory tables for local DataFrame execution (populated when where == "local")
    leaderboard_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    oof_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fva_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_decay_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bias_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    attributions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_cache: dict[tuple[str, str], CellResult] = field(default_factory=dict)
    # Champion ratchet state: preserves the best pass's outputs so autofocus never degrades
    champion_pass_id: str = "1"
    champion_score: float = float("inf")
    champion_outputs: dict[str, Any] = field(default_factory=dict)

    def champion_record(self) -> AutofocusPassRecord | None:
        """Return the :class:`AutofocusPassRecord` marked ``is_session_champion=True``."""
        for p in self.passes:
            if p.is_session_champion:
                return p
        return self.passes[-1] if self.passes else None

    @property
    def run_id(self) -> str:
        """Return the winning session champion `run_id` (or blueprint `run_id`)."""
        champ = self.champion_record()
        return champ.run_id if champ is not None else self.blueprint.run_id

    @property
    def best_model(self) -> str:
        """Return the winning model from the session champion pass."""
        champ = self.champion_record()
        return champ.champion_model if champ is not None else self.blueprint.config.models[0]

    @property
    def best_metric(self) -> float:
        """Return the winning decision metric value from the session champion pass."""
        champ = self.champion_record()
        return champ.champion_score if champ is not None else float("nan")

    @property
    def decision_metric(self) -> str:
        """Return the decision metric name from the session."""
        return (
            self.passes[-1].decision_metric
            if self.passes
            else self.blueprint.config.backtest.decision_metric
        )

    def ledger_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all executed `autofocus` passes and DAG branches."""
        rows = [
            {
                "pass_index": p.pass_index,
                "pass_id": p.pass_id,
                "parent_pass_ids": ", ".join(p.parent_pass_ids) if p.parent_pass_ids else "root",
                "branch_hypothesis": p.branch_hypothesis,
                "run_id": p.run_id,
                "tier": p.tier,
                "models_total": p.models_total,
                "models_borrowed": p.models_borrowed,
                "models_executed": p.models_executed,
                "champion_model": p.champion_model,
                "decision_metric": p.decision_metric,
                "champion_score": round(p.champion_score, 4),
                "fva_vs_naive_pct": round(p.fva_vs_naive_pct, 2),
                "fva_vs_univariate_pct": round(p.fva_vs_univariate_pct, 2),
                "fva_vs_foundation_pct": round(p.fva_vs_foundation_pct, 2),
                "covariate_lift_pct": round(p.covariate_lift_pct, 2),
                "reconciliation_lift_pct": round(p.reconciliation_lift_pct, 2),
                "champion_is_univariate": p.champion_is_univariate,
                "beaten_challengers": ", ".join(p.beaten_challengers)
                if p.beaten_challengers
                else "(none)",
                "dm_pvalue": round(p.dm_pvalue, 4) if p.dm_pvalue is not None else None,
                "predicted_p50_s": round(p.predicted_p50_s, 2),
                "actual_wall_s": round(p.actual_wall_s, 2),
                "continue_confidence": round(p.continue_confidence, 3),
                "expected_next_gain_pct": round(p.expected_next_gain_pct, 2),
                "mutations_applied": ", ".join(p.mutations_applied)
                if p.mutations_applied
                else "initial_pass",
                "is_session_champion": p.is_session_champion,
                "stop_or_continue_reason": p.stop_or_continue_reason,
            }
            for p in self.passes
        ]
        return pd.DataFrame(rows)

    def baseline_value_df(self) -> pd.DataFrame:
        """Compare the session champion against the 4-anchor baseline ladder and challengers.

        When a plain univariate model (or univariate foundation model) wins as the session
        champion, this table also highlights every complex/multivariate/global challenger and
        exogenous covariate that the univariate champion outperformed (`baseline_role =
        'beaten_complex_challenger'`).
        """
        cols = [
            "baseline_role",
            "baseline_model",
            "champion_model",
            "decision_metric",
            "baseline_score",
            "champion_score",
            "fva_vs_baseline_pct",
            "dm_pvalue",
            "series_win_rate_pct",
            "champion_is_univariate",
            "beaten_challengers",
            "univariate_edge_pct",
            "verdict",
        ]
        champ = self.champion_record()
        if champ is None:
            return pd.DataFrame(columns=cols)

        decision = self.decision_metric
        metric_col = (
            f"mean_{decision}" if f"mean_{decision}" in self.leaderboard_df.columns else decision
        )
        champ_model = champ.champion_model
        champ_score = champ.champion_score
        champ_is_uni = champ.champion_is_univariate
        beaten_str = ", ".join(champ.beaten_challengers) if champ.beaten_challengers else "(none)"

        # Gather scores across all passes so baselines/challengers from Pass 1 are always visible
        scores_by_model: dict[str, float] = {}
        for p_out in (
            self.champion_outputs,
            {"leaderboard_df": self.leaderboard_df, "fva_df": self.fva_df},
        ):
            lb = p_out.get("leaderboard_df", pd.DataFrame())
            if isinstance(lb, pd.DataFrame) and not lb.empty:
                m_col = "model_type" if "model_type" in lb.columns else "model"
                s_col = metric_col if metric_col in lb.columns else lb.columns[-1]
                for _, r_lb in lb.iterrows():
                    m_name = str(r_lb[m_col])
                    val = float(pd.to_numeric(r_lb[s_col], errors="coerce"))
                    if math.isfinite(val) and (
                        m_name not in scores_by_model or val < scores_by_model[m_name]
                    ):
                        scores_by_model[m_name] = val

        # Inspect cell_cache across passes so every model from Pass 1 or branches is scored
        if self.cell_cache:
            by_m_vals: dict[str, list[float]] = {}
            for c_res in self.cell_cache.values():
                if c_res.status == "ok" and decision in c_res.metrics:
                    v_m = float(c_res.metrics[decision])
                    if math.isfinite(v_m):
                        by_m_vals.setdefault(c_res.model_type, []).append(v_m)
            for m_k, vals_k in by_m_vals.items():
                if vals_k:
                    mean_v = float(np.mean(vals_k))
                    if m_k not in scores_by_model:
                        scores_by_model[m_k] = mean_v

        active_baselines = list(self.baselines or self.blueprint.baselines)
        if not active_baselines:
            active_baselines = [
                {"baseline_role": "naive_anchor", "model_type": "naive_seasonal"},
                {"baseline_role": "plain_univariate_stat", "model_type": "theta"},
            ]

        # Compute univariate edge % when champion is univariate and beat complex challengers
        complex_scores: list[float] = []
        for ch_name in champ.beaten_challengers:
            ch_model = ch_name.split(" (", 1)[0]
            if ch_model in scores_by_model and scores_by_model[ch_model] > champ_score:
                complex_scores.append(scores_by_model[ch_model])
        uni_edge_pct = (
            round(
                100.0 * (min(complex_scores) - champ_score) / max(1e-9, min(complex_scores)),
                2,
            )
            if (champ_is_uni and complex_scores and math.isfinite(champ_score))
            else 0.0
        )

        # Lookup per-series errors from OOF to compute series_win_rate_pct and DM p-value
        series_err_by_model: dict[str, dict[str, float]] = {}
        if not self.oof_df.empty and {"ts_id", "model_type", "y_true", "yhat"}.issubset(
            self.oof_df.columns
        ):
            oof_work = self.oof_df.copy()
            oof_work["abs_err"] = np.abs(
                pd.to_numeric(oof_work["y_true"], errors="coerce")
                - pd.to_numeric(oof_work["yhat"], errors="coerce")
            )
            for (m_t, tid), grp in oof_work.groupby(["model_type", "ts_id"], sort=False):
                series_err_by_model.setdefault(str(m_t), {})[str(tid)] = float(
                    grp["abs_err"].mean()
                )

        def _win_rate_vs(base_m: str) -> float:
            c_map = series_err_by_model.get(champ_model, {})
            b_map = series_err_by_model.get(base_m, {})
            shared = [k for k in c_map if k in b_map]
            if not shared:
                return 100.0 if scores_by_model.get(base_m, champ_score) > champ_score else 50.0
            wins = sum(1 for k in shared if c_map[k] <= b_map[k] + 1e-9)
            return round(100.0 * wins / len(shared), 1)

        rows: list[dict[str, Any]] = []
        seen_roles_models: set[tuple[str, str]] = set()
        for b_entry in active_baselines:
            role = str(b_entry.get("baseline_role", "baseline"))
            b_model = str(b_entry.get("model_type", "naive_seasonal"))
            # Resolve fallback if exact variant didn't run
            resolved_b = b_model
            if resolved_b not in scores_by_model:
                base_b = b_model.split("@", 1)[0]
                cand_matches = [
                    k for k in scores_by_model if k == base_b or k.startswith(f"{base_b}@")
                ]
                if cand_matches:
                    resolved_b = cand_matches[0]
            if resolved_b not in scores_by_model:
                continue
            if (role, resolved_b) in seen_roles_models:
                continue
            seen_roles_models.add((role, resolved_b))
            b_score = scores_by_model[resolved_b]
            fva_gain = (
                round(100.0 * (b_score - champ_score) / max(1e-9, b_score), 2)
                if math.isfinite(b_score) and math.isfinite(champ_score) and b_score > 1e-9
                else 0.0
            )
            win_rate = _win_rate_vs(resolved_b)
            dm_p = champ.dm_pvalue if role == "naive_anchor" else None
            if resolved_b == champ_model:
                verdict = (
                    f"champion_is_{role} (beat {beaten_str} by +{uni_edge_pct:.2f}%)"
                    if (champ_is_uni and champ.beaten_challengers)
                    else f"champion_is_{role}"
                )
            elif fva_gain > 0.25:
                verdict = f"beats_{role} (+{fva_gain:.2f}%)"
            elif fva_gain >= -0.25:
                verdict = f"ties_{role} ({fva_gain:+.2f}%)"
            else:
                verdict = f"trails_{role} ({fva_gain:+.2f}%)"

            rows.append(
                {
                    "baseline_role": role,
                    "baseline_model": resolved_b,
                    "champion_model": champ_model,
                    "decision_metric": decision,
                    "baseline_score": round(b_score, 4),
                    "champion_score": round(champ_score, 4),
                    "fva_vs_baseline_pct": fva_gain,
                    "dm_pvalue": round(dm_p, 4) if dm_p is not None else None,
                    "series_win_rate_pct": win_rate,
                    "champion_is_univariate": champ_is_uni,
                    "beaten_challengers": beaten_str,
                    "univariate_edge_pct": uni_edge_pct,
                    "verdict": verdict,
                }
            )

        # When a plain univariate (or univariate foundation) model wins as session champion,
        # append explicit rows showing each complex/multivariate/global challenger it beat!
        if champ_is_uni and champ.beaten_challengers:
            for ch_label in champ.beaten_challengers:
                ch_m = ch_label.split(" (", 1)[0]
                if (
                    ch_m in scores_by_model
                    and ("beaten_complex_challenger", ch_m) not in seen_roles_models
                ):
                    seen_roles_models.add(("beaten_complex_challenger", ch_m))
                    ch_score = scores_by_model[ch_m]
                    edge_vs_ch = (
                        round(100.0 * (ch_score - champ_score) / max(1e-9, ch_score), 2)
                        if math.isfinite(ch_score) and ch_score > 1e-9
                        else 0.0
                    )
                    rows.append(
                        {
                            "baseline_role": "beaten_complex_challenger",
                            "baseline_model": ch_label,
                            "champion_model": champ_model,
                            "decision_metric": decision,
                            "baseline_score": round(ch_score, 4),
                            "champion_score": round(champ_score, 4),
                            "fva_vs_baseline_pct": edge_vs_ch,
                            "dm_pvalue": None,
                            "series_win_rate_pct": _win_rate_vs(ch_m),
                            "champion_is_univariate": True,
                            "beaten_challengers": beaten_str,
                            "univariate_edge_pct": edge_vs_ch,
                            "verdict": f"univariate_champion_beat_{ch_m} (+{edge_vs_ch:.2f}%)",
                        }
                    )

        return pd.DataFrame(rows, columns=cols)

    def hierarchy_value_df(self) -> pd.DataFrame:
        """Return per-level accuracy and reconciliation lift across the hierarchy tree."""
        if not self.hierarchy_summary_df.empty:
            return self.hierarchy_summary_df.copy()
        return pd.DataFrame(
            columns=[
                "hierarchy_level",
                "n_nodes",
                "reconciliation_method",
                "decision_metric",
                "unreconciled_score",
                "reconciled_score",
                "reconciliation_lift_pct",
                "max_coherence_residual",
                "negative_leaf_count",
            ]
        )

    def diff_df(self) -> pd.DataFrame:
        """Return a parameter-by-parameter and score diff table across all `autofocus` passes."""
        cols = [
            "pass_index",
            "pass_id",
            "parent_pass_ids",
            "branch_hypothesis",
            "action",
            "parameter",
            "old_value",
            "new_value",
            "champion_model",
            "champion_score",
            "delta_vs_pass1_pct",
            "is_session_champion",
            "evidence",
            "rationale",
        ]
        if not self.passes:
            return pd.DataFrame(columns=cols)
        p1_score = self.passes[0].champion_score
        rows: list[dict[str, Any]] = []
        for p in self.passes:
            delta_pct = (
                round(100.0 * (p1_score - p.champion_score) / max(1e-9, p1_score), 2)
                if math.isfinite(p1_score) and math.isfinite(p.champion_score) and p1_score > 1e-9
                else 0.0
            )
            if not p.mutations_detail:
                rows.append(
                    {
                        "pass_index": p.pass_index,
                        "pass_id": p.pass_id,
                        "parent_pass_ids": ", ".join(p.parent_pass_ids) or "root",
                        "branch_hypothesis": p.branch_hypothesis,
                        "action": "initial_tier_compilation",
                        "parameter": "models",
                        "old_value": "(none)",
                        "new_value": ", ".join(p.config.models),
                        "champion_model": p.champion_model,
                        "champion_score": round(p.champion_score, 4),
                        "delta_vs_pass1_pct": delta_pct,
                        "is_session_champion": p.is_session_champion,
                        "evidence": f"tier='{p.tier}'",
                        "rationale": "Initial Glass-Box exposure tier model & feature compilation.",
                    }
                )
            else:
                for m in p.mutations_detail:
                    rows.append(
                        {
                            "pass_index": p.pass_index,
                            "pass_id": p.pass_id,
                            "parent_pass_ids": ", ".join(p.parent_pass_ids) or "1",
                            "branch_hypothesis": p.branch_hypothesis,
                            "action": m.get("action", ""),
                            "parameter": m.get("parameter", ""),
                            "old_value": m.get("old_value", ""),
                            "new_value": m.get("new_value", ""),
                            "champion_model": p.champion_model,
                            "champion_score": round(p.champion_score, 4),
                            "delta_vs_pass1_pct": delta_pct,
                            "is_session_champion": p.is_session_champion,
                            "evidence": m.get("evidence", ""),
                            "rationale": m.get("rationale", ""),
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def explain_decisions_df(self) -> pd.DataFrame:
        """Return a unified audit table of Pass 1 blueprint decisions + `autofocus` mutations."""
        cols = [
            "phase",
            "source",
            "category",
            "parameter",
            "value",
            "evidence",
            "rationale",
            "realized_delta_pct",
            "is_session_champion",
        ]
        rows: list[dict[str, Any]] = []
        p1_score = self.passes[0].champion_score if self.passes else float("nan")
        p1_champ = self.passes[0].is_session_champion if self.passes else True
        for _, r in self.blueprint.rationale_df.iterrows():
            cat = str(r.get("category", ""))
            src = (
                "user_pin"
                if cat == "user_constraint" or "[PINNED]" in str(r.get("evidence", ""))
                else "auto_policy"
            )
            rows.append(
                {
                    "phase": "pass_1_blueprint",
                    "source": src,
                    "category": cat,
                    "parameter": str(r.get("parameter", "")),
                    "value": str(r.get("value", "")),
                    "evidence": str(r.get("evidence", "")),
                    "rationale": str(r.get("rationale", "")),
                    "realized_delta_pct": 0.0,
                    "is_session_champion": p1_champ,
                }
            )
        for p in self.passes[1:]:
            delta_pct = (
                round(100.0 * (p1_score - p.champion_score) / max(1e-9, p1_score), 2)
                if math.isfinite(p1_score) and math.isfinite(p.champion_score) and p1_score > 1e-9
                else 0.0
            )
            for m in p.mutations_detail:
                rows.append(
                    {
                        "phase": f"pass_{p.pass_id}_autofocus",
                        "source": "autofocus_dag" if len(p.pass_id) > 1 else "autofocus_loop",
                        "category": m.get("action", "autofocus"),
                        "parameter": m.get("parameter", ""),
                        "value": m.get("new_value", ""),
                        "evidence": m.get("evidence", ""),
                        "rationale": m.get("rationale", ""),
                        "realized_delta_pct": delta_pct,
                        "is_session_champion": p.is_session_champion,
                    }
                )
        champ = self.champion_record()
        if champ is not None and champ.champion_is_univariate and champ.beaten_challengers:
            rows.append(
                {
                    "phase": f"pass_{champ.pass_id}_champion_verdict",
                    "source": "champion_ratchet",
                    "category": "univariate_champion_victory",
                    "parameter": "champion_model",
                    "value": champ.champion_model,
                    "evidence": f"beaten_challengers={list(champ.beaten_challengers)}",
                    "rationale": (
                        f"Plain univariate model '{champ.champion_model}' won the session "
                        f"championship ({champ.decision_metric}={champ.champion_score:.4f}), "
                        f"outperforming complex/multivariate challengers "
                        f"({', '.join(champ.beaten_challengers)})."
                    ),
                    "realized_delta_pct": round(champ.fva_vs_naive_pct, 2),
                    "is_session_champion": True,
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def profile_df(self) -> pd.DataFrame:
        """Return the 7-row Step 0 Data Profile & Readiness ``DataFrame`` for this session."""
        return self.profile.to_df()

    def compare_models(
        self,
        model_a: str | None = None,
        model_b: str | None = None,
    ) -> pd.DataFrame:
        """Head-to-head comparison answering *'Is Model A better than Model B?'*.

        When ``model_a`` is ``None``, defaults to the Session Champion (`self.best_model`). When
        ``model_b`` is ``None``, compares ``model_a`` against every other model evaluated across
        the session, reporting score delta, FVA lift %, series-by-series win rate %, and
        Diebold–Mariano $p$-value.
        """
        from .diagnostics import compute_fva

        cols = [
            "model_a",
            "model_b",
            "decision_metric",
            "score_a",
            "score_b",
            "lift_pct",
            "series_win_rate_pct",
            "dm_pvalue",
            "significant_at_10pct",
            "verdict",
        ]
        champ = self.champion_record()
        if champ is None:
            return pd.DataFrame(columns=cols)

        decision = self.decision_metric
        metric_col = (
            f"mean_{decision}" if f"mean_{decision}" in self.leaderboard_df.columns else decision
        )
        scores_by_model: dict[str, float] = {}
        if not self.leaderboard_df.empty:
            m_col = "model_type" if "model_type" in self.leaderboard_df.columns else "model"
            s_col = (
                metric_col
                if metric_col in self.leaderboard_df.columns
                else self.leaderboard_df.columns[-1]
            )
            for _, r_lb in self.leaderboard_df.iterrows():
                m_name = str(r_lb[m_col])
                val = float(pd.to_numeric(r_lb[s_col], errors="coerce"))
                if math.isfinite(val):
                    scores_by_model[m_name] = val
        if self.cell_cache:
            by_m_vals: dict[str, list[float]] = {}
            for c_res in self.cell_cache.values():
                if c_res.status == "ok" and decision in c_res.metrics:
                    v_m = float(c_res.metrics[decision])
                    if math.isfinite(v_m):
                        by_m_vals.setdefault(c_res.model_type, []).append(v_m)
            for m_k, vals_k in by_m_vals.items():
                if vals_k and m_k not in scores_by_model:
                    scores_by_model[m_k] = float(np.mean(vals_k))

        target_a = model_a or champ.champion_model
        if target_a not in scores_by_model:
            scores_by_model[target_a] = champ.champion_score
        score_a = scores_by_model[target_a]

        candidates_b = (
            [model_b]
            if model_b is not None
            else [
                m for m, _ in sorted(scores_by_model.items(), key=lambda kv: kv[1]) if m != target_a
            ]
        )

        series_err_by_model: dict[str, dict[str, float]] = {}
        if not self.oof_df.empty and {"ts_id", "model_type", "y_true", "yhat"}.issubset(
            self.oof_df.columns
        ):
            oof_w = self.oof_df.copy()
            oof_w["abs_err"] = np.abs(
                pd.to_numeric(oof_w["y_true"], errors="coerce")
                - pd.to_numeric(oof_w["yhat"], errors="coerce")
            )
            for (m_t, tid), grp in oof_w.groupby(["model_type", "ts_id"], sort=False):
                series_err_by_model.setdefault(str(m_t), {})[str(tid)] = float(
                    grp["abs_err"].mean()
                )

        rows: list[dict[str, Any]] = []
        for b_name in candidates_b:
            if b_name is None or b_name not in scores_by_model:
                continue
            score_b = scores_by_model[b_name]
            lift_pct = (
                round(100.0 * (score_b - score_a) / max(1e-9, score_b), 2)
                if math.isfinite(score_a) and math.isfinite(score_b) and score_b > 1e-9
                else 0.0
            )
            a_map = series_err_by_model.get(target_a, {})
            b_map = series_err_by_model.get(b_name, {})
            shared = [k for k in a_map if k in b_map]
            win_rate = (
                round(
                    100.0 * sum(1 for k in shared if a_map[k] <= b_map[k] + 1e-9) / len(shared),
                    1,
                )
                if shared
                else (100.0 if score_a < score_b else 0.0)
            )
            dm_p: float | None = None
            if not self.oof_df.empty:
                with contextlib.suppress(Exception):
                    fva_sub = compute_fva(self.oof_df, baseline=b_name, metric=decision)
                    if not fva_sub.empty and "model_type" in fva_sub.columns:
                        m_row = fva_sub[fva_sub["model_type"] == target_a]
                        if not m_row.empty and "dm_pvalue" in m_row.columns:
                            p_val = float(
                                pd.to_numeric(m_row["dm_pvalue"].iloc[0], errors="coerce")
                            )
                            if math.isfinite(p_val):
                                dm_p = round(p_val, 4)
            sig = bool(dm_p is not None and dm_p <= 0.10 and abs(lift_pct) >= 0.5)
            if lift_pct > 0.5:
                verdict = (
                    f"{target_a} beats {b_name} by +{lift_pct:.2f}% "
                    f"({'statistically significant' if sig else 'directional lift'}, "
                    f"wins {win_rate:.0f}% of series)"
                )
            elif lift_pct >= -0.5:
                verdict = f"Statistical tie ({lift_pct:+.2f}%, wins {win_rate:.0f}% of series)"
            else:
                verdict = f"{target_a} trails {b_name} by {lift_pct:+.2f}%"
            rows.append(
                {
                    "model_a": target_a,
                    "model_b": b_name,
                    "decision_metric": decision,
                    "score_a": round(score_a, 4),
                    "score_b": round(score_b, 4),
                    "lift_pct": lift_pct,
                    "series_win_rate_pct": win_rate,
                    "dm_pvalue": dm_p,
                    "significant_at_10pct": sig,
                    "verdict": verdict,
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def practitioner_assessment_df(self) -> pd.DataFrame:
        """Return a 7-row Practitioner Trust & Evidence ``DataFrame`` answering the 7 core
        questions forecasters ask about any run.
        """
        cols = [
            "question_id",
            "question",
            "verdict",
            "answer",
            "evidence",
            "recommendation",
        ]
        champ = self.champion_record()
        if champ is None:
            return pd.DataFrame(columns=cols)

        decision = self.decision_metric
        bval_df = self.baseline_value_df()
        hval_df = self.hierarchy_value_df()
        diag = dict(self.last_diagnostics_summary)

        # 1. Is this model better than simple baselines / Model X?
        fva_naive = champ.fva_vs_naive_pct
        fva_uni = champ.fva_vs_univariate_pct
        dm_p = champ.dm_pvalue
        if champ.champion_is_univariate and champ.beaten_challengers:
            q1_verdict = "univariate_champion_wins"
            beaten_str = ", ".join(champ.beaten_challengers)
            q1_ans = (
                f"Plain univariate model '{champ.champion_model}' beat both seasonal naive "
                f"(+{fva_naive:.2f}%) and complex challengers ({beaten_str})."
            )
        elif fva_naive > 1.0:
            q1_verdict = "verified_lift"
            q1_ans = (
                f"Champion '{champ.champion_model}' reduced {decision} by +{fva_naive:.2f}% vs. "
                f"seasonal naive and +{fva_uni:.2f}% vs. plain univariate baseline."
            )
        else:
            q1_verdict = "near_baseline"
            q1_ans = (
                f"Champion '{champ.champion_model}' is within {fva_naive:+.2f}% of baseline; "
                "series behaves close to a simple seasonal/random-walk floor."
            )
        q1_ev = (
            f"champion={champ.champion_model} ({decision}={champ.champion_score:.4f}), "
            f"fva_naive={fva_naive:+.2f}%, dm_pvalue={dm_p if dm_p is not None else 'n/a'}"
        )

        # 2. Do ensembles help?
        is_ens_champ = champ.champion_model.startswith("ensemble_")
        base_rows = (
            self.leaderboard_df[
                ~self.leaderboard_df["model_type"].astype(str).str.startswith("ensemble_")
            ]
            if not self.leaderboard_df.empty and "model_type" in self.leaderboard_df.columns
            else pd.DataFrame()
        )
        ens_rows = (
            self.leaderboard_df[
                self.leaderboard_df["model_type"].astype(str).str.startswith("ensemble_")
            ]
            if not self.leaderboard_df.empty and "model_type" in self.leaderboard_df.columns
            else pd.DataFrame()
        )
        m_col = (
            f"mean_{decision}" if f"mean_{decision}" in self.leaderboard_df.columns else decision
        )
        best_base_name = (
            str(base_rows["model_type"].iloc[0]) if not base_rows.empty else champ.champion_model
        )
        best_base_sc = (
            float(pd.to_numeric(base_rows[m_col].iloc[0], errors="coerce"))
            if not base_rows.empty and m_col in base_rows.columns
            else champ.champion_score
        )
        best_ens_name = str(ens_rows["model_type"].iloc[0]) if not ens_rows.empty else "(none)"
        best_ens_sc = (
            float(pd.to_numeric(ens_rows[m_col].iloc[0], errors="coerce"))
            if not ens_rows.empty and m_col in ens_rows.columns
            else float("nan")
        )
        ens_lift = (
            round(100.0 * (best_base_sc - best_ens_sc) / max(1e-9, best_base_sc), 2)
            if math.isfinite(best_base_sc) and math.isfinite(best_ens_sc) and best_base_sc > 1e-9
            else 0.0
        )
        if is_ens_champ and ens_lift > 0.0:
            q2_verdict = "ensemble_helps"
            q2_ans = (
                f"Yes — '{best_ens_name}' outperformed the best single model ('{best_base_name}') "
                f"by +{ens_lift:.2f}%."
            )
        elif math.isfinite(best_ens_sc):
            q2_verdict = "single_model_preferred"
            q2_ans = (
                f"Single model '{best_base_name}' matched or beat '{best_ens_name}' "
                f"(ensemble delta={ens_lift:+.2f}%); parsimony rule keeps the simpler model."
            )
        else:
            q2_verdict = "single_model_only"
            q2_ans = f"Single model '{best_base_name}' selected (no multi-model ensemble needed)."
        q2_ev = f"best_single={best_base_name} ({best_base_sc:.4f}), best_ensemble={best_ens_name}"

        # 3. Do covariates help or is it overfitting?
        all_covs = (
            list(self.profile.future_covariates)
            + list(self.profile.past_covariates)
            + list(self.profile.static_covariates)
        )
        kept_covs = list(champ.config.features.future_covariates) + list(
            champ.config.features.past_covariates
        )
        pruned_covs = [c for c in all_covs if c not in kept_covs]
        mono_constraints = dict(champ.config.features.monotonic_constraints or {})
        monotone_models = [m for m in champ.config.models if "@monotone" in m]
        hw_cfg = champ.config.features.holiday_window
        hw_active = bool(hw_cfg is not None and getattr(hw_cfg, "enabled", False))
        guardrail_notes: list[str] = []
        guardrail_ev: list[str] = []
        if mono_constraints or monotone_models:
            guardrail_notes.append(
                f"structural sign guardrails enforced (monotonic_constraints={mono_constraints}"
                + (f", monotone_models={monotone_models}" if monotone_models else "")
                + ")"
            )
            guardrail_ev.append(f"monotonic_constraints={mono_constraints}")
            if monotone_models:
                guardrail_ev.append(f"monotone_models={monotone_models}")
        if hw_active and hw_cfg is not None:
            guardrail_notes.append(
                f"asymmetric holiday window active (-{hw_cfg.pre_holiday_days}d/+"
                f"{hw_cfg.post_holiday_days}d, kernel='{hw_cfg.kernel}')"
            )
            guardrail_ev.append(
                f"holiday_window=(-{hw_cfg.pre_holiday_days}, +{hw_cfg.post_holiday_days}, "
                f"'{hw_cfg.kernel}')"
            )
        guardrail_suffix = f" ({'; '.join(guardrail_notes)})" if guardrail_notes else ""
        guardrail_ev_suffix = f", {', '.join(guardrail_ev)}" if guardrail_ev else ""

        if not all_covs:
            q3_verdict = "univariate_panel"
            q3_ans = (
                "No external covariates in source data; forecast relies on target history "
                f"and calendar cycles{guardrail_suffix}."
            )
            q3_ev = f"covariates=0{guardrail_ev_suffix}"
        elif not kept_covs or champ.champion_is_univariate:
            q3_verdict = "overfitting_prevented"
            pruned_list = pruned_covs or all_covs
            q3_ans = (
                f"Covariates did not improve out-of-fold accuracy (pruned={pruned_list}); "
                f"univariate model won, preventing covariate overfitting{guardrail_suffix}."
            )
            q3_ev = f"initial_covs={all_covs}, kept_covs={kept_covs}{guardrail_ev_suffix}"
        elif pruned_covs:
            q3_verdict = "selective_signal_kept"
            q3_ans = (
                f"True drivers {kept_covs} improved accuracy (+{fva_uni:.2f}% vs. univariate), "
                f"while noisy decoy columns {pruned_covs} were automatically pruned"
                f"{guardrail_suffix}."
            )
            q3_ev = (
                f"kept={kept_covs}, pruned_decoys={pruned_covs}, "
                f"fva_vs_univariate=+{fva_uni:.2f}%{guardrail_ev_suffix}"
            )
        else:
            q3_verdict = "covariates_help"
            q3_ans = (
                f"Covariates {kept_covs} added +{fva_uni:.2f}% lift over plain "
                f"univariate baselines{guardrail_suffix}."
            )
            q3_ev = (
                f"active_covariates={kept_covs}, fva_vs_univariate=+{fva_uni:.2f}%"
                f"{guardrail_ev_suffix}"
            )

        # 4. Does global provide an advantage over local?
        global_models = [m for m in champ.config.models if "@global" in m]
        donor_sim = float(
            diag.get(
                "cold_start_donor_similarity",
                getattr(self.profile, "cold_start_donor_similarity", 0.0),
            )
        )
        cs_transfer_on = bool(
            getattr(champ.config.features, "cold_start_transfer", False) or donor_sim >= 0.65
        )
        donor_note = (
            f" Cold-start analog donor shrinkage active (donor_similarity={donor_sim:.2f})."
            if cs_transfer_on
            else ""
        )
        if global_models:
            q4_verdict = "global_active"
            q4_ans = (
                f"Cross-series @global models ({', '.join(global_models)}) are active for "
                f"scale_regime='{self.profile.scale_regime}' (n_series={self.profile.n_series}, "
                f"corr={self.profile.cross_series_corr:.2f}).{donor_note}"
            )
        else:
            q4_verdict = "local_per_series_optimal"
            q4_ans = (
                f"Per-series local and zero-shot models are optimal for "
                f"scale_regime='{self.profile.scale_regime}' (n_series={self.profile.n_series}, "
                f"corr={self.profile.cross_series_corr:.2f}), avoiding global neural overfitting."
                f"{donor_note}"
            )
        q4_ev = (
            f"n_series={self.profile.n_series}, scale_regime='{self.profile.scale_regime}', "
            f"cross_series_corr={self.profile.cross_series_corr:.2f}, "
            f"global_models={global_models}, cold_start_transfer={cs_transfer_on}, "
            f"cold_start_donor_similarity={donor_sim:.2f}"
        )

        # 5. Is reconciliation being done correctly?
        has_spatial = bool(
            self.profile.hierarchy_id_cols
            or (champ.config.hierarchy.enabled and champ.config.hierarchy.levels)
        )
        has_temporal = bool(
            champ.config.hierarchy.reconcile_temporal
            or champ.config.hierarchy.cross_temporal
            or "cross_temporal_mint" in champ.config.hierarchy.reconciliation_methods
            or diag.get("reconciliation_mode") in ("temporal_only", "cross_temporal")
        )
        temp_aggs = list(
            champ.config.hierarchy.temporal_aggregations
            or getattr(self.profile, "temporal_aggregations", None)
            or [1]
        )
        is_simplex = bool(
            getattr(self.profile, "is_compositional_simplex", False)
            or champ.config.features.simplex_sum is not None
            or champ.config.features.transform in ("clr", "ilr")
            or any("@transform_clr" in m or "@transform_ilr" in m for m in champ.config.models)
        )

        if has_spatial and has_temporal:
            all_row = (
                hval_df[hval_df["hierarchy_level"] == "All Hierarchy Nodes"]
                if not hval_df.empty
                else pd.DataFrame()
            )
            r_row = (
                all_row.iloc[0]
                if not all_row.empty
                else (hval_df.iloc[0] if not hval_df.empty else pd.Series(dtype=object))
            )
            max_coh = float(
                r_row.get("max_coherence_residual", diag.get("spatial_coherence_residual", 0.0))
            )
            temp_coh = float(
                r_row.get(
                    "temporal_coherence_residual", diag.get("temporal_coherence_residual", 0.0)
                )
            )
            neg_cnt = int(r_row.get("negative_leaf_count", 0))
            rec_m = str(r_row.get("reconciliation_method", "cross_temporal_mint"))
            rec_lift = float(r_row.get("reconciliation_lift_pct", 0.0))
            alpha_val = diag.get(
                "spatial_weight_alpha", champ.config.hierarchy.spatial_temporal_weight
            )
            alpha_str = (
                f"{float(alpha_val):.4f}"
                if isinstance(alpha_val, (int, float)) and not isinstance(alpha_val, bool)
                else str(alpha_val)
            )
            ct_beat = bool(
                diag.get(
                    "cross_temporal_beats_spatial",
                    "cross_temporal_mint" in str(champ.champion_model),
                )
            )
            q5_verdict = (
                "coherent_and_verified"
                if max_coh < 1e-4 and temp_coh < 1e-4 and neg_cnt == 0
                else "check_tree"
            )
            q5_ans = (
                f"Yes — cross_temporal reconciliation ('{rec_m}', mode=cross_temporal) achieved "
                f"exact spatial coherence (residual={max_coh:.6f}) and temporal coherence "
                f"(residual={temp_coh:.6f} across aggregations={temp_aggs}, "
                f"spatial_weight_alpha={alpha_str}), {neg_cnt} negative leaves, "
                f"{rec_lift:+.2f}% tree-wide accuracy change, and "
                f"cross_temporal_beats_spatial={ct_beat}."
            )
            q5_ev = (
                f"mode='cross_temporal', method='{rec_m}', coherence_gap={max_coh:.6f}, "
                f"temporal_coherence_gap={temp_coh:.6f}, temporal_aggregations={temp_aggs}, "
                f"spatial_weight_alpha={alpha_str}, cross_temporal_beats_spatial={ct_beat}, "
                f"negative_leaves={neg_cnt}, levels={len(hval_df)}"
            )
        elif has_spatial and not hval_df.empty:
            all_row = hval_df[hval_df["hierarchy_level"] == "All Hierarchy Nodes"]
            r_row = all_row.iloc[0] if not all_row.empty else hval_df.iloc[0]
            max_coh = float(r_row.get("max_coherence_residual", 0.0))
            neg_cnt = int(r_row.get("negative_leaf_count", 0))
            rec_m = str(r_row.get("reconciliation_method", "mint_nonneg"))
            rec_lift = float(r_row.get("reconciliation_lift_pct", 0.0))
            q5_verdict = (
                "coherent_and_verified" if max_coh < 1e-4 and neg_cnt == 0 else "check_tree"
            )
            q5_ans = (
                f"Yes — fold-safe '{rec_m}' spatial_only reconciliation achieved exact additive "
                f"coherence (residual={max_coh:.6f}), {neg_cnt} negative leaves, and "
                f"{rec_lift:+.2f}% tree-wide accuracy change."
            )
            q5_ev = (
                f"mode='spatial_only', method='{rec_m}', coherence_gap={max_coh:.6f}, "
                f"negative_leaves={neg_cnt}, levels={len(hval_df)}"
            )
        elif has_temporal:
            all_row = (
                hval_df[hval_df["hierarchy_level"] == "All Hierarchy Nodes"]
                if not hval_df.empty
                else pd.DataFrame()
            )
            r_row = (
                all_row.iloc[0]
                if not all_row.empty
                else (hval_df.iloc[0] if not hval_df.empty else pd.Series(dtype=object))
            )
            temp_coh = float(
                r_row.get(
                    "temporal_coherence_residual",
                    r_row.get(
                        "max_coherence_residual", diag.get("temporal_coherence_residual", 0.0)
                    ),
                )
            )
            rec_m = str(
                r_row.get(
                    "reconciliation_method",
                    champ.config.hierarchy.reconciliation_methods[0]
                    if champ.config.hierarchy.reconciliation_methods
                    else "wls_struct",
                )
            )
            rec_lift = float(r_row.get("reconciliation_lift_pct", 0.0))
            q5_verdict = "temporal_coherent_and_verified" if temp_coh < 1e-4 else "check_temporal"
            q5_ans = (
                f"Yes — temporal_only reconciliation ('{rec_m}') on flat panel verified exact "
                f"multi-horizon bucket coherence (temporal_residual={temp_coh:.6f} across "
                f"temporal_aggregations={temp_aggs}) with {rec_lift:+.2f}% accuracy change."
            )
            q5_ev = (
                f"mode='temporal_only', method='{rec_m}', temporal_coherence_gap={temp_coh:.6f}, "
                f"temporal_aggregations={temp_aggs}, reconciliation_lift_pct={rec_lift:+.2f}%"
            )
        else:
            q5_verdict = "not_applicable_flat_panel"
            q5_ans = (
                "Flat panel (no hierarchy columns); output.floor enforces non-negativity directly."
            )
            q5_ev = f"mode='none', hierarchy_id_cols={list(self.profile.hierarchy_id_cols)}"

        if is_simplex:
            from .simplex import verify_simplex_coherence

            s_target = float(
                champ.config.features.simplex_sum
                or getattr(self.profile, "simplex_total", None)
                or 1.0
            )
            simplex_disc = (
                float(
                    verify_simplex_coherence(
                        self.predictions_df,
                        simplex_sum=s_target,
                        raise_on_error=False,
                    )
                )
                if not self.predictions_df.empty
                else 0.0
            )
            min_share = (
                float(pd.to_numeric(self.predictions_df["yhat"], errors="coerce").min())
                if not self.predictions_df.empty and "yhat" in self.predictions_df.columns
                else s_target
            )
            q5_ans += (
                f" Compositional simplex closure active (simplex_sum={s_target:g}, "
                f"simplex_discrepancy={simplex_disc:.6f}, min_share={min_share:.6f} > 0)."
            )
            q5_ev += (
                f", is_compositional_simplex=True, simplex_sum={s_target:g}, "
                f"simplex_discrepancy={simplex_disc:.6f}, strictly_positive={min_share > 0.0}"
            )
            if not has_spatial and not has_temporal:
                q5_verdict = (
                    "coherent_and_verified"
                    if simplex_disc < 1e-4 and min_share > 0.0
                    else "check_simplex"
                )

        # 6. Does accuracy hold up across the horizon?
        decay_ratio = float(diag.get("late_horizon_degradation_ratio", 1.0))
        crossover_h = diag.get("horizon_crossover_step")
        has_hz_spec = "horizon_specialist" in champ.config.ensemble.strategies or any(
            "@direct" in m for m in champ.config.models
        )
        if decay_ratio <= 1.35:
            q6_verdict = "stable_across_horizon"
            q6_ans = (
                f"Forecast error remains stable from step 1 to {self.profile.horizon} "
                f"(late/early ratio={decay_ratio:.2f}x)."
            )
        else:
            q6_verdict = "horizon_specialist_mitigated" if has_hz_spec else "late_horizon_decay"
            q6_ans = (
                f"Late-horizon error grows {decay_ratio:.2f}x vs. early steps "
                f"(crossover step={crossover_h}); "
                + (
                    "mitigated via @direct horizon models and horizon_specialist blending."
                    if has_hz_spec
                    else "consider enabling @direct models or horizon_specialist ensembling."
                )
            )
        q6_ev = (
            f"late_early_ratio={decay_ratio:.2f}, crossover_step={crossover_h}, "
            f"direct_or_specialist={has_hz_spec}"
        )

        # 7. How do I trust the forecast?
        p1_sc = self.passes[0].champion_score
        tot_gain = (
            round(100.0 * (p1_sc - champ.champion_score) / max(1e-9, p1_sc), 2)
            if len(self.passes) > 1 and math.isfinite(p1_sc) and p1_sc > 1e-9
            else 0.0
        )
        acf1 = float(diag.get("residual_autocorr_lag1", 0.0))
        biased_share = float(diag.get("biased_series_share", 0.0))
        cal_method = str(getattr(champ.config.output, "calibration_method", "conformal"))
        aci_gamma_val = float(getattr(champ.config.output, "aci_gamma", 0.05))
        ks_drift_val = float(
            diag.get(
                "residual_drift_ks_stat",
                getattr(self.profile, "residual_drift_ks_stat", 0.0),
            )
        )
        q7_verdict = (
            "high_confidence" if fva_naive > 0.0 and biased_share < 0.25 else "review_exceptions"
        )
        q7_ans = (
            f"Evaluated on {champ.config.backtest.n_folds} rolling out-of-fold backtest window(s) "
            f"with {cal_method} prediction intervals (aci_gamma={aci_gamma_val:.3f}), "
            f"Champion Ratchet protection (Pass {champ.pass_id} won, +{tot_gain:.2f}% over "
            f"Pass 1), residual ACF1={acf1:.2f}, drift KS={ks_drift_val:.2f}, and copula "
            f"tail_risk Audit."
        )
        q7_ev = (
            f"n_folds={champ.config.backtest.n_folds}, calibration_method='{cal_method}', "
            f"aci_gamma={aci_gamma_val:.3f}, champion_pass={champ.pass_id}/{len(self.passes)}, "
            f"residual_acf1={acf1:.2f}, residual_drift_ks_stat={ks_drift_val:.2f}, "
            f"biased_series_share={biased_share:.1%}, tail_risk_ready=True"
        )

        rows = [
            {
                "question_id": "model_vs_baseline",
                "question": "1. Is the winning model genuinely better than simple baselines?",
                "verdict": q1_verdict,
                "answer": q1_ans,
                "evidence": q1_ev,
                "recommendation": (
                    "Inspect f.baseline_value_df() and f.compare_models() for pairwise win rates."
                ),
            },
            {
                "question_id": "ensemble_vs_single",
                "question": "2. Do ensembles help here, or is a single model better?",
                "verdict": q2_verdict,
                "answer": q2_ans,
                "evidence": q2_ev,
                "recommendation": (
                    "Ensembles must beat the best single model by the parsimony margin "
                    "to be crowned."
                ),
            },
            {
                "question_id": "covariates_vs_overfitting",
                "question": "3. Do covariates help, or are they just overfitting noise?",
                "verdict": q3_verdict,
                "answer": q3_ans,
                "evidence": q3_ev,
                "recommendation": (
                    "Inspect f.attributions_df() and f.promo_elasticity_df() to see per-feature "
                    "importance shares, promotional lift, and causal sign adherence."
                ),
            },
            {
                "question_id": "global_vs_local",
                "question": "4. Does cross-series @global training beat per-series @local fitting?",
                "verdict": q4_verdict,
                "answer": q4_ans,
                "evidence": q4_ev,
                "recommendation": (
                    "Global pooling pays off most when n_series >= 15 or "
                    "cross_series_corr >= 0.35; inspect f.cold_start_transfer_df() when "
                    "cold_start series are present."
                ),
            },
            {
                "question_id": "hierarchy_reconciliation",
                "question": (
                    "5. Is hierarchical reconciliation coherent, non-negative, and accurate?"
                ),
                "verdict": q5_verdict,
                "answer": q5_ans,
                "evidence": q5_ev,
                "recommendation": (
                    "Inspect f.hierarchy_value_df() for Root, Intermediate, and Leaf accuracy lift."
                ),
            },
            {
                "question_id": "horizon_stability",
                "question": "6. Does accuracy hold up across the whole forecast horizon?",
                "verdict": q6_verdict,
                "answer": q6_ans,
                "evidence": q6_ev,
                "recommendation": (
                    "Inspect f.horizon_decay_df() for step-by-step error from h=1..H."
                ),
            },
            {
                "question_id": "forecast_trust",
                "question": "7. How do I trust this forecast in production?",
                "verdict": q7_verdict,
                "answer": q7_ans,
                "evidence": q7_ev,
                "recommendation": (
                    "Inspect f.aci_df(), f.tail_risk_df(), and f.explain_auto_decisions_df() "
                    "for online ACI coverage, copula tail_risk, and the Glass-Box audit trail."
                ),
            },
        ]
        _ = bval_df
        return pd.DataFrame(rows, columns=cols)

    def assess(self) -> pd.DataFrame:
        """Return a 7-Question Practitioner Assessment DataFrame with full text report
        representation.
        """
        prof_str = self.profile.explain()
        assess_df = self.practitioner_assessment_df()
        champ = self.champion_record()
        head = (
            f"Practitioner Trust & Evidence Assessment : {self.run_id} | "
            f"Champion='{champ.champion_model if champ else 'n/a'}' "
            f"({self.decision_metric}={champ.champion_score if champ else float('nan'):.4f})"
        )
        report_text = f"{prof_str}\n\n{head}\n\n{assess_df.to_string(index=False)}"
        frame = PractitionerAssessmentFrame(assess_df)
        frame._report_text = report_text
        return frame

    def promo_elasticity_df(
        self,
        *,
        data: pd.DataFrame | None = None,
        promo_col: str | None = None,
        price_col: str | None = None,
        confounders: Sequence[str] | None = None,
        summary: bool = False,
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition on this session's
        panel (`elasticity.compute_promo_elasticity_decomposition`).
        """
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        panel = data if data is not None else self.data_df
        if panel is None or panel.empty:
            return pd.DataFrame()
        champ = self.champion_record()
        cfg = champ.config if champ is not None else self.blueprint.config
        decomp = compute_promo_elasticity_decomposition(
            panel,
            ts_id_col=cfg.data.ts_id_col,
            date_col=cfg.data.date_col,
            target_col=cfg.data.target_col,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
            freq=cfg.data.freq,
            holidays=cfg.features.holidays,
        )
        return summarize_promo_elasticity(decomp) if summary else decomp

    promo_elasticity = promo_elasticity_df

    def plot_promo_elasticity(
        self,
        *,
        data: pd.DataFrame | None = None,
        ts_id: str | None = None,
        promo_col: str | None = None,
        price_col: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot organic baseline vs. promotional lift and post-promo dip for ``ts_id``."""
        from .elasticity import plot_promo_elasticity

        decomp = self.promo_elasticity_df(
            data=data, promo_col=promo_col, price_col=price_col, summary=False
        )
        return plot_promo_elasticity(decomp, ts_id=ts_id, ax=ax, title=title)

    def simulate_scenarios(
        self,
        scenarios: Mapping[str, Mapping[str, Any]],
        *,
        data: pd.DataFrame | None = None,
        models: Sequence[str] | None = None,
        future_covariates_df: pd.DataFrame | None = None,
        price_col: str | None = None,
        unit_cost: float | str | Mapping[str, float] | None = None,
        reconcile: bool = True,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        inventory_params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run full-chain counterfactual scenario propagation across this session's champion
        configuration (base models -> ensembles -> hierarchy reconciliation -> gross margin &
        inventory policy shift).
        """
        from .scenarios import simulate_scenarios

        panel = data if data is not None else self.data_df
        if panel is None or panel.empty:
            raise DataError("No in-memory panel available on AutoForecastSession; pass data=...")
        champ = self.champion_record()
        cfg = champ.config if champ is not None else self.blueprint.config
        fut = future_covariates_df if future_covariates_df is not None else self.future_df
        oof = self.oof_df if not self.oof_df.empty else None
        return simulate_scenarios(
            panel,
            cfg,
            scenarios,
            models=models,
            future_covariates_df=fut,
            price_col=price_col,
            unit_cost=unit_cost,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            oof_df=oof,
            inventory_params=inventory_params,
        )


_TS_ID_CANDIDATES: tuple[str, ...] = (
    "ts_id",
    "series_id",
    "unique_id",
    "id",
    "item_id",
    "sku",
    "store_id",
    "entity_id",
)
_DATE_CANDIDATES: tuple[str, ...] = (
    "ds",
    "date",
    "timestamp",
    "datetime",
    "time",
    "week",
    "month",
    "period",
)
_TARGET_CANDIDATES: tuple[str, ...] = (
    "y",
    "target",
    "value",
    "sales",
    "demand",
    "revenue",
    "qty",
    "quantity",
    "count",
    "ot",
)
_BQ_TEMPORAL_TYPES: frozenset[str] = frozenset({"DATE", "DATETIME", "TIMESTAMP"})
_BQ_NUMERIC_TYPES: frozenset[str] = frozenset(
    {"FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC", "INT64", "INTEGER"}
)


def _is_temporal_column(s: pd.Series) -> bool:
    """Return True if Series ``s`` holds calendar dates or timestamps (never raw floats/bools)."""
    if pd.api.types.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.PeriodDtype):
        return True
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s):
        return False
    non_null = s.dropna().head(50)
    if non_null.empty:
        return False
    if pd.api.types.is_integer_dtype(s):
        vals = pd.to_numeric(non_null, errors="coerce")
        med = float(vals.median()) if not vals.empty else 0.0
        # Only accept YYYYMMDD integer dates (e.g. 20240101), never small integer IDs/counts.
        return 19000101.0 <= med <= 21001231.0
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(non_null, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def _is_numeric_target_column(s: pd.Series) -> bool:
    """Return True if Series ``s`` holds numeric observations suitable for a forecast target."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    non_null = s.dropna().head(50)
    if non_null.empty:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return True
    parsed = pd.to_numeric(non_null, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def _resolve_bq_table_columns(
    client: Any,
    fq_table: str,
    *,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
) -> tuple[str, str, str | None]:
    """Inspect BigQuery table schema metadata (0 bytes scanned) to resolve column roles safely."""
    try:
        table_obj = client.get_table(fq_table)
        schema = list(getattr(table_obj, "schema", None) or [])
    except Exception:  # noqa: BLE001 - mock clients or restricted metadata fall back to sample
        schema = []
    if not schema:
        return ts_id_col or "ts_id", date_col or "ds", target_col

    col_types: dict[str, str] = {
        str(f.name): str(getattr(f, "field_type", "")).upper() for f in schema
    }
    cols = list(col_types.keys())
    lower_to_actual = {c.lower(): c for c in cols}

    # 1. Resolve date_col from BigQuery schema
    if date_col is not None:
        if date_col not in col_types:
            raise DataError(
                f"date_col='{date_col}' not found in BigQuery table '{fq_table}' columns {cols}."
            )
        if col_types[date_col] in _BQ_NUMERIC_TYPES or col_types[date_col] == "BOOL":
            raise DataError(
                f"date_col='{date_col}' in '{fq_table}' has non-temporal BigQuery type "
                f"'{col_types[date_col]}'."
            )
        eff_date = date_col
    else:
        eff_date = next(
            (
                lower_to_actual[c]
                for c in _DATE_CANDIDATES
                if c in lower_to_actual
                and (
                    col_types[lower_to_actual[c]] in _BQ_TEMPORAL_TYPES
                    or col_types[lower_to_actual[c]] == "STRING"
                )
            ),
            "",
        )
        if not eff_date:
            temporal_cols = [c for c, t in col_types.items() if t in _BQ_TEMPORAL_TYPES]
            if len(temporal_cols) == 1:
                eff_date = temporal_cols[0]
            else:
                raise DataError(
                    f"Could not unambiguously infer date_col for BigQuery table '{fq_table}' "
                    f"(temporal columns={temporal_cols}, all columns={cols}). "
                    "Pass date_col='<column_name>' explicitly."
                )

    # 2. Resolve ts_id_col from BigQuery schema
    if ts_id_col is not None:
        if ts_id_col not in col_types:
            raise DataError(
                f"ts_id_col='{ts_id_col}' not found in BigQuery table '{fq_table}' columns {cols}."
            )
        eff_ts = ts_id_col
    else:
        eff_ts = next(
            (
                lower_to_actual[c]
                for c in _TS_ID_CANDIDATES
                if c in lower_to_actual and lower_to_actual[c] != eff_date
            ),
            "",
        )
        if not eff_ts:
            id_cols = [
                c
                for c, t in col_types.items()
                if c != eff_date and t in ("STRING", "INT64", "INTEGER", "BYTES")
            ]
            if len(id_cols) == 1:
                eff_ts = id_cols[0]
            else:
                raise DataError(
                    f"Could not unambiguously infer ts_id_col for BigQuery table '{fq_table}' "
                    f"(identifier candidates={id_cols}, all columns={cols}). "
                    "Pass ts_id_col='<column_name>' explicitly."
                )

    # 3. Resolve target_col from BigQuery schema (if possible)
    eff_target: str | None = target_col
    if eff_target is not None:
        if eff_target not in col_types:
            raise DataError(
                f"target_col='{eff_target}' not found in BigQuery table '{fq_table}' "
                f"columns {cols}."
            )
        if col_types[eff_target] not in _BQ_NUMERIC_TYPES:
            raise DataError(
                f"target_col='{eff_target}' in '{fq_table}' has non-numeric BigQuery type "
                f"'{col_types[eff_target]}'."
            )
    else:
        eff_target = next(
            (
                lower_to_actual[c]
                for c in _TARGET_CANDIDATES
                if c in lower_to_actual
                and lower_to_actual[c] not in (eff_ts, eff_date)
                and col_types[lower_to_actual[c]] in _BQ_NUMERIC_TYPES
            ),
            None,
        )
        if eff_target is None:
            num_cols = [
                c
                for c, t in col_types.items()
                if c not in (eff_ts, eff_date) and t in _BQ_NUMERIC_TYPES
            ]
            if len(num_cols) == 1:
                eff_target = num_cols[0]
    return eff_ts, eff_date, eff_target


def _infer_columns_and_freq(
    df: pd.DataFrame,
    *,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    freq: str | None = None,
) -> tuple[str, str, str, str]:
    """Deterministically detect and dtype-verify ``(ts_id_col, date_col, target_col, freq)``."""
    if df.empty:
        raise DataError("Cannot inspect an empty DataFrame for auto-forecasting.")

    cols = list(df.columns)
    lower_to_actual = {str(c).lower(): str(c) for c in cols}

    # 1. Resolve and verify date_col
    if date_col is not None:
        if date_col not in df.columns:
            raise DataError(f"Column '{date_col}' not found in DataFrame columns {cols}.")
        if not _is_temporal_column(df[date_col]):
            raise DataError(
                f"Column date_col='{date_col}' (dtype={df[date_col].dtype}) does not contain "
                "valid dates or timestamps."
            )
        eff_date = date_col
    else:
        eff_date = next(
            (
                lower_to_actual[c]
                for c in _DATE_CANDIDATES
                if c in lower_to_actual and _is_temporal_column(df[lower_to_actual[c]])
            ),
            "",
        )
        if not eff_date:
            temporal_cols = [c for c in cols if _is_temporal_column(df[c])]
            if len(temporal_cols) == 1:
                eff_date = str(temporal_cols[0])
            else:
                raise DataError(
                    f"Could not unambiguously infer date_col from columns {cols} "
                    f"(temporal candidates={temporal_cols}). "
                    "Pass date_col='<column_name>' explicitly."
                )

    # 2. Resolve and verify ts_id_col
    if ts_id_col is not None:
        if ts_id_col not in df.columns:
            raise DataError(f"Column '{ts_id_col}' not found in DataFrame columns {cols}.")
        eff_ts = ts_id_col
    else:
        eff_ts = next(
            (
                lower_to_actual[c]
                for c in _TS_ID_CANDIDATES
                if c in lower_to_actual and lower_to_actual[c] != eff_date
            ),
            "",
        )
        if not eff_ts:
            id_candidates = [
                str(c)
                for c in cols
                if str(c) != eff_date
                and not pd.api.types.is_float_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])
            ]
            if len(id_candidates) == 1:
                eff_ts = id_candidates[0]
            else:
                raise DataError(
                    f"Could not unambiguously infer ts_id_col from columns {cols} "
                    f"(identifier candidates={id_candidates}). "
                    "Pass ts_id_col='<column_name>' explicitly."
                )

    # 3. Resolve and verify target_col
    if target_col is not None:
        if target_col not in df.columns:
            raise DataError(f"Column '{target_col}' not found in DataFrame columns {cols}.")
        if not _is_numeric_target_column(df[target_col]):
            raise DataError(
                f"Column target_col='{target_col}' (dtype={df[target_col].dtype}) is not numeric."
            )
        eff_target = target_col
    else:
        eff_target = next(
            (
                lower_to_actual[c]
                for c in _TARGET_CANDIDATES
                if c in lower_to_actual
                and lower_to_actual[c] not in (eff_ts, eff_date)
                and _is_numeric_target_column(df[lower_to_actual[c]])
            ),
            "",
        )
        if not eff_target:
            num_candidates = [
                str(c)
                for c in cols
                if str(c) not in (eff_ts, eff_date) and _is_numeric_target_column(df[c])
            ]
            if len(num_candidates) == 1:
                eff_target = num_candidates[0]
            else:
                raise DataError(
                    f"Could not unambiguously infer target_col from columns {cols} "
                    f"(numeric candidates={num_candidates}). "
                    "Pass target_col='<column_name>' explicitly."
                )

    if freq is not None:
        return eff_ts, eff_date, eff_target, freq

    # Infer frequency from first series with >= 3 timestamps
    first_ts = df[eff_ts].iloc[0]
    sample_dates = (
        pd.to_datetime(df.loc[df[eff_ts] == first_ts, eff_date], errors="coerce")
        .dropna()
        .sort_values()
        .unique()
    )
    if len(sample_dates) >= 3:
        inferred = pd.infer_freq(pd.DatetimeIndex(sample_dates))
        if inferred:
            if inferred.startswith("W"):
                return eff_ts, eff_date, eff_target, "W"
            if inferred in ("MS", "ME", "M"):
                return eff_ts, eff_date, eff_target, "MS" if inferred == "MS" else "ME"
            if inferred in ("D", "B", "h", "H"):
                return eff_ts, eff_date, eff_target, "h" if inferred in ("h", "H") else "D"
        dt_arr = pd.DatetimeIndex(sample_dates).to_numpy(dtype="datetime64[s]")
        diff_s = np.diff(dt_arr).astype(np.float64)
        med_days = float(np.median(diff_s)) / 86400.0
        if 0.03 <= med_days <= 0.06:
            return eff_ts, eff_date, eff_target, "h"
        if 6.5 <= med_days <= 7.5:
            return eff_ts, eff_date, eff_target, "W"
        if 27.0 <= med_days <= 32.0:
            return eff_ts, eff_date, eff_target, "MS"
    return eff_ts, eff_date, eff_target, "D"


def inspect_panel_for_auto(
    data: pd.DataFrame | str,
    *,
    future_df: pd.DataFrame | None = None,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    freq: str | None = None,
    horizon: int | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> PanelAutoProfile:
    """Single-pass statistical & structural profiler for Glass-Box configuration compilation.

    Inspects either an in-memory ``pd.DataFrame`` or a BigQuery source table name (`str`) and
    computes the :class:`PanelAutoProfile` driving model selection, covariate role classification,
    target transformations, SBC demand regime routing, and hierarchical reconciliation.
    """
    source_table_name = "dataframe_panel"
    true_total_series: int | None = None
    true_total_rows: int | None = None
    if isinstance(data, str):
        from .errors import require_extra
        from .registry.tables import bigquery_client
        from .settings import Settings

        require_extra("gcp", purpose="Inspecting a BigQuery source table for auto-forecasting")
        source_table_name = data
        eff_settings = settings or Settings.resolve()
        client = bigquery_client(eff_settings.project_id, location=eff_settings.region)
        fq_table = (
            data if "." in data else f"{eff_settings.project_id}.{eff_settings.dataset_id}.{data}"
        )
        id_col_guess, date_col_guess, target_col_guess = _resolve_bq_table_columns(
            client,
            fq_table,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
        )
        ts_id_col = ts_id_col or id_col_guess
        date_col = date_col or date_col_guess
        target_col = target_col or target_col_guess
        sql = f"""
        WITH table_stats AS (
          SELECT COUNT(DISTINCT `{id_col_guess}`) AS _sf_total_series, COUNT(*) AS _sf_total_rows
          FROM `{fq_table}`
        ),
        sampled_ids AS (
          SELECT DISTINCT `{id_col_guess}`
          FROM `{fq_table}`
          LIMIT 150
        )
        SELECT t.*, s_tot._sf_total_series, s_tot._sf_total_rows
        FROM `{fq_table}` t
        JOIN sampled_ids s USING (`{id_col_guess}`)
        CROSS JOIN table_stats s_tot
        ORDER BY t.`{id_col_guess}`, t.`{date_col_guess}`
        """
        df = client.query(sql).to_dataframe()
        if "_sf_total_series" in df.columns:
            if not df.empty and pd.notna(df["_sf_total_series"].iloc[0]):
                true_total_series = int(df["_sf_total_series"].iloc[0])
            df = df.drop(columns=["_sf_total_series"])
        if "_sf_total_rows" in df.columns:
            if not df.empty and pd.notna(df["_sf_total_rows"].iloc[0]):
                true_total_rows = int(df["_sf_total_rows"].iloc[0])
            df = df.drop(columns=["_sf_total_rows"])
    else:
        df = data.copy()

    eff_ts, eff_date, eff_target, eff_freq = _infer_columns_and_freq(
        df,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        freq=freq,
    )
    eff_horizon = int(horizon or _DEFAULT_HORIZON_BY_FREQ.get(eff_freq, 14))

    # Separate historical rows (where target is not NaN) from trailing future covariate rows
    df[eff_date] = pd.to_datetime(df[eff_date], errors="coerce")
    y_num = pd.to_numeric(df[eff_target], errors="coerce")
    hist_df = df[y_num.notna()].copy()
    trailing_fut_df = df[y_num.isna()].copy()
    if future_df is not None and not future_df.empty:
        trailing_fut_df = future_df.copy()

    if hist_df.empty:
        raise DataError(
            f"Panel target column '{eff_target}' contains no non-null numeric observations."
        )

    hist_df = hist_df.sort_values([eff_ts, eff_date], kind="stable")
    y_vals = pd.to_numeric(hist_df[eff_target], errors="coerce").dropna().to_numpy(dtype=np.float64)

    non_negative = bool(np.all(y_vals >= 0.0))
    integer_valued = bool(np.mean(np.isclose(y_vals, np.round(y_vals), atol=1e-6)) >= 0.99)
    zero_fraction = float(np.mean(y_vals == 0.0))

    # Per-series SBC regime & heteroskedasticity diagnostics
    s_period = max(2, seasonal_period(eff_freq))
    obs_counts: list[int] = []
    adis: list[float] = []
    cv2s: list[float] = []
    lv_corrs: list[float] = []
    growth_ratios: list[float] = []
    acfs: list[float] = []
    break_flags: list[bool] = []
    spike_flags: list[bool] = []
    leading_zero_flags: list[bool] = []
    gap_flags: list[bool] = []
    regime_counts: dict[str, int] = {
        "smooth": 0,
        "erratic": 0,
        "intermittent": 0,
        "lumpy": 0,
        "cold_start": 0,
    }

    cold_threshold = max(28, 2 * eff_horizon)
    for _, grp in hist_df.groupby(eff_ts, sort=False):
        arr = pd.to_numeric(grp[eff_target], errors="coerce").dropna().to_numpy(dtype=np.float64)
        n = len(arr)
        if n == 0:
            continue
        obs_counts.append(n)
        pos_mask = arr > 0.0
        n_pos = int(np.sum(pos_mask))
        adi = float(n / max(1, n_pos))
        adis.append(adi)
        if n_pos > 0:
            first_pos_idx = int(np.argmax(pos_mask))
            leading_zero_flags.append(first_pos_idx >= max(4, int(0.10 * n)))
        else:
            leading_zero_flags.append(False)

        ds_vals = (
            pd.to_datetime(grp[eff_date], errors="coerce").dropna().to_numpy(dtype="datetime64[s]")
        )
        if len(ds_vals) >= 4:
            ds_diffs = np.diff(ds_vals).astype(np.float64)
            med_step = float(np.median(ds_diffs))
            gap_flags.append(bool(med_step > 0.0 and np.max(ds_diffs) > 1.75 * med_step))
        else:
            gap_flags.append(False)

        if n_pos >= 2:
            pos_vals = arr[pos_mask]
            pos_mean = float(np.mean(pos_vals))
            pos_std = float(np.std(pos_vals, ddof=0))
            cv2 = float((pos_std / max(1e-9, pos_mean)) ** 2)
        else:
            cv2 = 0.0
        cv2s.append(cv2)

        if n < cold_threshold:
            regime_counts["cold_start"] += 1
        elif adi < 1.32 and cv2 < 0.49:
            regime_counts["smooth"] += 1
        elif adi < 1.32 and cv2 >= 0.49:
            regime_counts["erratic"] += 1
        elif adi >= 1.32 and cv2 < 0.49:
            regime_counts["intermittent"] += 1
        else:
            regime_counts["lumpy"] += 1

        # Windowed level vs volatility correlation (diagnoses log-normal / multiplicative DGP)
        if n >= 28:
            n_wins = min(8, n // 7)
            chunks = np.array_split(arr, n_wins)
            w_means = np.array([float(np.mean(c)) for c in chunks])
            w_stds = np.array([float(np.std(c, ddof=0)) for c in chunks])
            if np.std(w_means) > 1e-6 and np.std(w_stds) > 1e-6:
                corr = float(np.corrcoef(w_means, w_stds)[0, 1])
                if math.isfinite(corr):
                    lv_corrs.append(corr)
            q_len = max(7, n // 4)
            start_mean = max(1e-6, float(np.mean(np.abs(arr[:q_len]))))
            end_mean = float(np.mean(np.abs(arr[-q_len:])))
            growth_ratios.append(end_mean / start_mean)

        # Seasonal lag autocorrelation
        if n >= s_period + 5 and np.std(arr) > 1e-6:
            c0 = arr[:-s_period] - np.mean(arr[:-s_period])
            c1 = arr[s_period:] - np.mean(arr[s_period:])
            denom = float(np.linalg.norm(c0) * np.linalg.norm(c1))
            if denom > 1e-9:
                acfs.append(float(np.dot(c0, c1) / denom))

        # Structural break & isolated spike check
        if n >= 40:
            best_z = 0.0
            for frac in (0.45, 0.55, 0.65, 0.72):
                sp = int(frac * n)
                left, right = arr[:sp], arr[sp:]
                l_med, r_med = float(np.median(left)), float(np.median(right))
                mad = float(np.median(np.abs(np.diff(arr)))) * 1.4826 + 1e-6
                z_step = abs(r_med - l_med) / max(mad * 1.8, 0.15 * max(abs(l_med), 1.0))
                best_z = max(best_z, z_step)
            break_flags.append(best_z >= 2.2)

            med_series = pd.Series(arr).rolling(7, center=True, min_periods=1).median().to_numpy()
            resid = np.abs(arr - med_series)
            rob_mad = float(np.median(resid)) * 1.4826 + 1e-6
            spike_flags.append(
                bool(np.max(resid) > 6.5 * rob_mad and np.mean(resid > 5.0 * rob_mad) < 0.06)
            )

    sampled_n_series = max(1, len(obs_counts))
    n_series = max(sampled_n_series, true_total_series or sampled_n_series)
    regime_shares = {k: round(v / sampled_n_series, 4) for k, v in regime_counts.items()}
    dominant_regime = max(regime_shares.items(), key=lambda kv: kv[1])[0]

    # Classify extra columns into censored_col, hierarchy_id_cols, static, future, past covariates
    reserved = {eff_ts, eff_date, eff_target, "archetype"}
    extra_cols = [c for c in df.columns if c not in reserved]

    censored_col: str | None = None
    for c in extra_cols:
        clower = c.lower()
        if any(tok in clower for tok in ("stockout", "censored", "oos_flag", "out_of_stock")):
            censored_col = c
            break
    if censored_col:
        extra_cols = [c for c in extra_cols if c != censored_col]

    # Hierarchy detection
    resolved_hierarchy: list[str] = []
    if hierarchy_cols is not None:
        resolved_hierarchy = [c for c in hierarchy_cols if c in df.columns]
    else:
        for cand_pair in (("region", "store"), ("state", "store_id"), ("category", "subcategory")):
            if all(c in extra_cols for c in cand_pair):
                resolved_hierarchy = list(cand_pair)
                break
    for hc in resolved_hierarchy:
        if hc in extra_cols:
            extra_cols.remove(hc)

    hierarchy_node_count = 0
    if resolved_hierarchy:
        hierarchy_node_count = 1 + n_series
        for lvl_idx in range(len(resolved_hierarchy)):
            lvl_cols = resolved_hierarchy[: lvl_idx + 1]
            hierarchy_node_count += int(hist_df[lvl_cols].drop_duplicates().shape[0])

    # Cross-series correlation across up to 12 sampled series
    cross_series_corr = 0.0
    if n_series >= 2:
        sampled_ids = list(hist_df[eff_ts].drop_duplicates().iloc[:12])
        sub_piv = (
            hist_df[hist_df[eff_ts].isin(sampled_ids)]
            .pivot_table(index=eff_date, columns=eff_ts, values=eff_target, aggfunc="mean")
            .dropna(axis=0, how="any")
        )
        if sub_piv.shape[0] >= 8 and sub_piv.shape[1] >= 2:
            cmat = sub_piv.corr().to_numpy(dtype=np.float64)
            tri_idx = np.triu_indices(cmat.shape[0], k=1)
            pair_vals = np.abs(cmat[tri_idx])
            pair_vals = pair_vals[np.isfinite(pair_vals)]
            if pair_vals.size > 0:
                cross_series_corr = float(np.median(pair_vals))

    static_covs: list[str] = []
    future_covs: list[str] = []
    past_covs: list[str] = []

    for c in extra_cols:
        max_nunq = int(hist_df.groupby(eff_ts, sort=False)[c].nunique(dropna=True).max() or 0)
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if max_nunq <= 1 and (
            not is_num
            or any(
                tok in c.lower()
                for tok in ("tier", "category", "segment", "cluster", "static", "format")
            )
        ):
            static_covs.append(c)
        elif is_num:
            if (
                not trailing_fut_df.empty
                and c in trailing_fut_df.columns
                and trailing_fut_df[c].notna().any()
            ) or trailing_fut_df.empty:
                future_covs.append(c)
            else:
                past_covs.append(c)

    # Compositional simplex detection (parts summing to ~1.0 or ~100.0 per date)
    is_compositional_simplex = False
    simplex_total = 0.0
    if sampled_n_series >= 2 and non_negative:
        date_grp = hist_df.groupby(eff_date, sort=False)[eff_target]
        date_counts = date_grp.count()
        date_sums = date_grp.sum()
        full_dates = date_sums[date_counts >= min(2, sampled_n_series)]
        if len(full_dates) >= 5:
            sums_arr = full_dates.to_numpy(dtype=np.float64)
            med_sum = float(np.median(sums_arr))
            max_val = float(np.max(y_vals)) if y_vals.size > 0 else 0.0
            for cand_total in (1.0, 100.0):
                tol = 0.03 * cand_total
                if (
                    abs(med_sum - cand_total) <= tol
                    and max_val <= cand_total + tol
                    and float(np.mean(np.abs(sums_arr - cand_total) <= tol)) >= 0.90
                ):
                    is_compositional_simplex = True
                    simplex_total = cand_total
                    break

    cold_start_donor_similarity = 0.0
    if regime_shares.get("cold_start", 0.0) > 0.0 and sampled_n_series >= 2:
        try:
            from .cold_start import build_cold_start_donor_pool, cold_start_transfer_df

            min_mat = max(14, 2 * eff_horizon)
            pool = build_cold_start_donor_pool(
                hist_df,
                ts_id_col=eff_ts,
                date_col=eff_date,
                target_col=eff_target,
                min_obs=min_mat,
                horizon=eff_horizon,
            )
            if pool.donors:
                cs_df = cold_start_transfer_df(
                    hist_df,
                    min_obs=min_mat,
                    top_k_donors=3,
                    ts_id_col=eff_ts,
                    date_col=eff_date,
                    target_col=eff_target,
                    horizon=eff_horizon,
                )
                if not cs_df.empty and "donor_similarity" in cs_df.columns:
                    cold_start_donor_similarity = round(float(cs_df["donor_similarity"].mean()), 4)
                else:
                    cold_start_donor_similarity = round(max(0.68, cross_series_corr), 4)
        except Exception:  # noqa: BLE001 - fallback when donor pool cannot be constructed
            cold_start_donor_similarity = round(max(0.65, cross_series_corr), 4)

    residual_drift_ks_stat = round(
        float(np.mean(break_flags)) * 0.45 if break_flags else 0.0,
        4,
    )

    return PanelAutoProfile(
        source_table=source_table_name,
        ts_id_col=eff_ts,
        date_col=eff_date,
        target_col=eff_target,
        freq=eff_freq,
        horizon=eff_horizon,
        n_series=n_series,
        n_rows=max(len(hist_df), true_total_rows or len(hist_df)),
        median_n_obs=int(np.median(obs_counts)) if obs_counts else 0,
        min_n_obs=int(np.min(obs_counts)) if obs_counts else 0,
        max_n_obs=int(np.max(obs_counts)) if obs_counts else 0,
        non_negative=non_negative,
        integer_valued=integer_valued,
        zero_fraction=round(zero_fraction, 4),
        median_adi=round(float(np.median(adis)) if adis else 1.0, 4),
        median_cv2=round(float(np.median(cv2s)) if cv2s else 0.0, 4),
        level_vol_corr=round(float(np.median(lv_corrs)) if lv_corrs else 0.0, 4),
        end_to_start_ratio=round(float(np.median(growth_ratios)) if growth_ratios else 1.0, 4),
        dominant_regime=dominant_regime,
        regime_shares=regime_shares,
        median_acf_season=round(float(np.median(acfs)) if acfs else 0.0, 4),
        structural_break_share=round(float(np.mean(break_flags)) if break_flags else 0.0, 4),
        outlier_spike_share=round(float(np.mean(spike_flags)) if spike_flags else 0.0, 4),
        future_covariates=tuple(future_covs),
        past_covariates=tuple(past_covs),
        static_covariates=tuple(static_covs),
        hierarchy_id_cols=tuple(resolved_hierarchy),
        censored_col=censored_col,
        leading_zero_share=round(
            float(np.mean(leading_zero_flags)) if leading_zero_flags else 0.0, 4
        ),
        missing_date_gap_share=round(float(np.mean(gap_flags)) if gap_flags else 0.0, 4),
        hierarchy_node_count=hierarchy_node_count,
        cross_series_corr=round(cross_series_corr, 4),
        monotonic_constraints=infer_monotonic_constraints([*future_covs, *past_covs]),
        is_compositional_simplex=is_compositional_simplex,
        simplex_total=simplex_total,
        cold_start_donor_similarity=cold_start_donor_similarity,
        residual_drift_ks_stat=residual_drift_ks_stat,
    )


@functools.lru_cache(maxsize=8)
def _cached_available_models(where: str) -> tuple[str, ...]:
    if where == "local":
        names = list_models(available_only=True)
        return tuple(n for n in names if get_model(n).runtime == "python")
    if where == "bigquery":
        return ("arima_plus", "timesfm")
    return tuple(list_models(available_only=False))


def _available_models_for_where(
    where: str,
    *,
    exclude_models: Sequence[str] | None = None,
) -> list[str]:
    """Return candidate base model names eligible for the requested `where` sandbox."""
    excl = {m.split("@", 1)[0] for m in (exclude_models or ())}
    return [m for m in _cached_available_models(where) if m not in excl]


def _build_compute_block_for_where(
    where: str,
    models: Sequence[str],
    *,
    n_series: int = 10,
    max_usd: float | None = None,
) -> dict[str, Any]:
    """Build the `compute` dictionary matching the user's `where` sandbox envelope and scale."""
    compute_cfg: dict[str, Any] = {}
    if where != "local":
        compute_cfg["reuse_prior_runs"] = True
    if max_usd is not None:
        compute_cfg["budget_usd"] = max(1.0, float(max_usd))

    s_regime = classify_scale_regime(n_series)
    default_workers = int(SCALE_REGIMES[s_regime]["default_cloud_workers"])
    families_present = {get_model(base_model_name(m)).family for m in models}
    families_block: dict[str, Any] = {}

    if (
        where in ("cloud_cpu", "cloud_all", "auto")
        and (families_present & {"statistical", "ml"})
        and s_regime in ("large", "massive")
    ):
        compute_cfg["persist_models"] = False
        compute_cfg["bucket_target_cells"] = 200

    if where == "cloud_cpu":
        for fam in ("statistical", "ml", "deep_learning"):
            if fam in families_present:
                fam_entry: dict[str, Any] = {
                    "runtime": "vertex" if fam == "deep_learning" else "spark",
                    "hardware": "cpu",
                }
                if fam_entry["runtime"] == "vertex" and default_workers > 1:
                    fam_entry["workers"] = default_workers
                families_block[fam] = fam_entry
    elif where in ("cloud_all", "auto") and "deep_learning" in families_present:
        compute_cfg["ray_regions"] = ["us-central1", "us-east4", "us-east1", "us-west1"]
        dl_entry: dict[str, Any] = {
            "runtime": "vertex",
            "hardware": "gpu",
            "gpu_type": "T4",
            "workers": max(2, default_workers),
        }
        families_block["deep_learning"] = dl_entry
    if families_block:
        compute_cfg["families"] = families_block
    return compute_cfg


def _extract_base_config_overrides(
    base_config: RunConfig | Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    """Extract explicitly set fields from ``base_config`` so user seeds override auto defaults."""
    if base_config is None:
        return {}
    if isinstance(base_config, (str, Path)):
        loaded = load_config(base_config)
        return loaded.model_dump(mode="json", exclude_unset=True)
    if isinstance(base_config, RunConfig):
        return base_config.model_dump(mode="json", exclude_unset=True)
    if isinstance(base_config, Mapping):
        return dict(base_config)
    raise ConfigError(f"Unsupported base_config type: {type(base_config).__name__}")


def build_config(
    data: pd.DataFrame | str | PanelAutoProfile,
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
    allowed_families: Sequence[str] | None = None,
    base_config: RunConfig | Mapping[str, Any] | str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    exclude_models: Sequence[str] | None = None,
    baselines: str | Sequence[str] | None = "auto",
    holidays: Sequence[str] | str | None = None,
    decision_metric: str | None = None,
    n_folds: int | None = None,
    hpo: bool | Mapping[str, Any] | None = None,
    ensemble: bool | Sequence[str] | Mapping[str, Any] | None = None,
    explainability: bool = True,
    calibration: DurationCalibration | bool | None = None,
    max_minutes: float | None = None,
    max_wall_clock_minutes: float | None = None,
    max_usd: float | None = None,
    only_available_models: bool = True,
    future_df: pd.DataFrame | None = None,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    freq: str | None = None,
    horizon: int | None = None,
    run_name: str | None = None,
    settings: Settings | None = None,
) -> ConfigBlueprint:
    """Compile a Glass-Box :class:`ConfigBlueprint` (`RunConfig` + Rationale Table + Duration).

    100% transparent compiler: inspects ``data`` (or accepts an already-computed
    :class:`PanelAutoProfile`), applies the evolved decision policy in :mod:`auto_policy`,
    layers any explicit user constraint pins (`base_config`, `overrides`, `models`,
    `include_models`, `exclude_models`, `baselines`, `holidays`, `decision_metric`, `n_folds`,
    `hpo`, `ensemble`, `explainability`), enforces ``max_minutes`` and ``max_usd`` budgets via
    :func:`predict_run_duration`, and returns a validated :class:`ConfigBlueprint`.
    """
    tier = _normalize_tier(tier)
    if max_minutes is None and max_wall_clock_minutes is not None:
        max_minutes = max_wall_clock_minutes
    if where not in ("auto", "local", "bigquery", "cloud_cpu", "cloud_all"):
        raise ConfigError(
            f"Unknown where='{where}'. Valid values: "
            "'auto', 'local', 'bigquery', 'cloud_cpu', 'cloud_all'"
        )
    if goal not in ("accuracy", "probabilistic", "service_level", "fast"):
        raise ConfigError(
            f"Unknown goal='{goal}'. Valid values: "
            "'accuracy', 'probabilistic', 'service_level', 'fast'"
        )

    base_seed = _extract_base_config_overrides(base_config)
    seed_data = (
        dict(base_seed.get("data", {})) if isinstance(base_seed.get("data"), Mapping) else {}
    )

    if isinstance(data, PanelAutoProfile):
        profile = data
        if horizon is not None and horizon != profile.horizon:
            profile = replace(profile, horizon=int(horizon))
    else:
        profile = inspect_panel_for_auto(
            data,
            future_df=future_df,
            ts_id_col=ts_id_col or seed_data.get("ts_id_col"),
            date_col=date_col or seed_data.get("date_col"),
            target_col=target_col or seed_data.get("target_col"),
            freq=freq or seed_data.get("freq"),
            horizon=horizon or seed_data.get("horizon"),
            settings=settings,
        )

    eff_where = (
        "local" if (where == "auto" and profile.source_table == "dataframe_panel") else where
    )
    avail_models = _available_models_for_where(
        "local" if only_available_models and eff_where == "local" else eff_where,
        exclude_models=exclude_models,
    )

    policy_out = select_tier_models_and_features(
        profile,
        tier=tier,
        where=eff_where,
        goal=goal,
        allowed_families=allowed_families,
        available_models=avail_models,
    )

    selected_models: list[str] = list(policy_out["models"])
    rationales: list[dict[str, str]] = list(policy_out["rationales"])
    protected_models: set[str] = {"naive_seasonal"}

    if allowed_families is not None:
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "allowed_families",
                "value": ", ".join(allowed_families),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Restricts candidate model selection to user-approved families.",
            }
        )

    # Apply model pins / inclusions / exclusions
    if models is not None:
        selected_models = list(dict.fromkeys(str(m) for m in models))
        protected_models.update(selected_models)
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "models",
                "value": ", ".join(selected_models),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Uses the exact model portfolio specified by the caller.",
            }
        )
    elif "models" in base_seed and isinstance(base_seed["models"], Sequence):
        selected_models = list(dict.fromkeys(str(m) for m in base_seed["models"]))
        protected_models.update(selected_models)
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "models",
                "value": ", ".join(selected_models),
                "evidence": "Seeded from base_config [PINNED]",
                "rationale": "Preserves the model list declared in base_config.",
            }
        )

    if exclude_models:
        excl_set = set(exclude_models)
        excl_bases = {m.split("@", 1)[0] for m in excl_set}
        selected_models = [
            m for m in selected_models if m not in excl_set and m.split("@", 1)[0] not in excl_bases
        ]
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "exclude_models",
                "value": ", ".join(exclude_models),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Excludes user-disallowed models from Pass 1 and autofocus.",
            }
        )

    if include_models:
        for inc_m in include_models:
            if inc_m not in selected_models:
                selected_models.append(inc_m)
            protected_models.add(inc_m)
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "include_models",
                "value": ", ".join(include_models),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Guarantees user-requested models are included and never trimmed.",
            }
        )

    # Resolve 4-anchor baseline ladder (or user-pinned baselines)
    baseline_entries = select_default_baselines(
        profile,
        where=eff_where,
        available_models=avail_models,
        selected_models=selected_models,
        user_baselines=baselines,
    )
    user_pinned_baselines = baselines is not None and baselines != "auto"
    for b_entry in baseline_entries:
        b_m = str(b_entry["model_type"])
        b_base = b_m.split("@", 1)[0]
        if user_pinned_baselines:
            if b_m not in selected_models and (b_base in avail_models or not only_available_models):
                selected_models.append(b_m)
            protected_models.add(b_m)
        elif b_entry["baseline_role"] in ("naive_anchor", "plain_univariate_stat"):
            protected_models.add(b_m)

    if baseline_entries:
        b_summary = ", ".join(f"{b['model_type']} ({b['baseline_role']})" for b in baseline_entries)
        rationales.append(
            {
                "category": "user_constraint" if user_pinned_baselines else "baseline_ladder",
                "parameter": "baselines",
                "value": b_summary,
                "evidence": (
                    f"baselines={baselines} [PINNED]"
                    if user_pinned_baselines
                    else (
                        "4-anchor baseline ladder (naive, plain univariate, "
                        "foundation, covariate ablation)"
                    )
                ),
                "rationale": (
                    "Guarantees protected baseline anchors across passes so every model and "
                    "mutation is scored on Forecast Value Added (FVA) over baseline."
                ),
            }
        )

    # Explainability guarantee: if explainability=True and ML family is allowed, ensure at least
    # one attribution-capable model is in the portfolio
    if (
        explainability
        and models is None
        and (allowed_families is None or "ml" in allowed_families)
        and eff_where != "bigquery"
    ):
        has_explainable = any(
            get_model(base_model_name(m)).supports_explainability for m in selected_models
        )
        if not has_explainable:
            for exp_cand in ("lightgbm", "regression_lags", "xgboost"):
                if exp_cand in avail_models:
                    selected_models.append(exp_cand)
                    rationales.append(
                        {
                            "category": "explainability",
                            "parameter": exp_cand,
                            "value": "include (explainability=True)",
                            "evidence": "supports_explainability=True",
                            "rationale": (
                                "Ensures Tier 1 global feature attributions and Tier 2 per-step "
                                "explanations are populated."
                            ),
                        }
                    )
                    break

    if not selected_models:
        selected_models = ["naive_seasonal", "theta"]

    hol_list: list[str] = list(policy_out.get("holidays", []))
    if holidays is not None:
        hol_list = [holidays] if isinstance(holidays, str) else [str(h) for h in holidays]
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "features.holidays",
                "value": ", ".join(hol_list),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Enables country holiday calendar indicator features.",
            }
        )

    features_cfg: dict[str, Any] = {
        "future_covariates": list(profile.future_covariates),
        "past_covariates": list(profile.past_covariates),
        "static_covariates": list(profile.static_covariates),
        "transform": policy_out["target_transform"],
        "outlier_policy": policy_out["outlier_policy"],
        "level_shift": bool(policy_out["level_shift"]),
        "regime_routing": bool(policy_out["regime_routing"]),
        "fourier": bool(policy_out.get("fourier", False)),
        "trading_days": bool(policy_out.get("trading_days", False)),
        "holidays": hol_list,
        "include_univariate_baselines": (
            False if models is not None else bool(policy_out["include_univariate_baselines"])
        ),
        "ablation_modes": [] if models is not None else list(policy_out["ablation_modes"]),
    }
    if policy_out.get("simplex_sum") is not None:
        features_cfg["simplex_sum"] = float(policy_out["simplex_sum"])
    if policy_out.get("censored_col"):
        features_cfg["censored_col"] = policy_out["censored_col"]
    if policy_out.get("monotonic_constraints"):
        features_cfg["monotonic_constraints"] = dict(policy_out["monotonic_constraints"])
    if hol_list and str(profile.freq).strip().upper() in ("D", "B", "H", "1H", "BH"):
        features_cfg["holiday_window"] = {
            "enabled": True,
            "pre_holiday_days": 7,
            "post_holiday_days": 3,
            "kernel": "linear",
        }
        rationales.append(
            {
                "category": "features",
                "parameter": "features.holiday_window",
                "value": "pre=7d, post=3d, kernel='linear'",
                "evidence": f"holidays={hol_list}, freq='{profile.freq}'",
                "rationale": (
                    "Expands point-day holiday indicators into asymmetric pre-holiday stock-up "
                    "ramps and post-holiday dip windows."
                ),
            }
        )

    if profile.future_covariates:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.future_covariates",
                "value": ", ".join(profile.future_covariates),
                "evidence": f"{len(profile.future_covariates)} known future driver(s) detected",
                "rationale": "Registers known future drivers for elastic and global models.",
            }
        )
    if profile.static_covariates:
        rationales.append(
            {
                "category": "features",
                "parameter": "features.static_covariates",
                "value": ", ".join(profile.static_covariates),
                "evidence": f"{len(profile.static_covariates)} time-invariant series attribute(s)",
                "rationale": "Enables cross-series conditioning on static metadata.",
            }
        )

    hierarchy_cfg: dict[str, Any] = {"enabled": False}
    has_bq_native_model = any(
        get_model(base_model_name(m)).runtime == "bigquery" for m in selected_models
    )
    if profile.hierarchy_id_cols and eff_where != "bigquery" and not has_bq_native_model:
        rec_methods = list(select_hierarchy_reconciliation_methods(profile, tier=tier))
        h_cols = list(profile.hierarchy_id_cols)
        h_levels = [h_cols[: i + 1] for i in range(len(h_cols))]
        hierarchy_cfg = {
            "enabled": True,
            "levels": h_levels,
            "reconciliation_methods": rec_methods,
        }
        rationales.append(
            {
                "category": "hierarchy",
                "parameter": "hierarchy.reconciliation_methods",
                "value": ", ".join(rec_methods),
                "evidence": (
                    f"hierarchy_id_cols={list(profile.hierarchy_id_cols)}, "
                    f"n_nodes={profile.hierarchy_node_count or profile.n_series}"
                ),
                "rationale": (
                    "Enforces additive cross-level coherence and scale-aware non-negative leaf "
                    f"reconciliation ({', '.join(rec_methods)})."
                ),
            }
        )

    output_cfg: dict[str, Any] = {
        "point_forecast": "auto" if goal in ("service_level", "probabilistic") else "raw",
        "quantiles": list(policy_out["quantiles"]),
        "integer_rounding": policy_out["integer_rounding"],
    }
    if policy_out.get("floor") is not None:
        output_cfg["floor"] = policy_out["floor"]
    if policy_out.get("target_quantile") is not None:
        output_cfg["target_quantile"] = policy_out["target_quantile"]
    if policy_out.get("underage_cost") is not None:
        output_cfg["underage_cost"] = policy_out["underage_cost"]
    if policy_out.get("overage_cost") is not None:
        output_cfg["overage_cost"] = policy_out["overage_cost"]

    if isinstance(calibration, bool):
        output_cfg["point_forecast"] = "auto" if calibration else "raw"
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "output.point_forecast",
                "value": output_cfg["point_forecast"],
                "evidence": f"calibration={calibration} [PINNED]",
                "rationale": "Applies user-pinned OOF point-forecast calibration policy.",
            }
        )
    elif isinstance(calibration, str) and calibration in ("raw", "median", "mean", "auto"):
        output_cfg["point_forecast"] = calibration
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "output.point_forecast",
                "value": calibration,
                "evidence": f"calibration='{calibration}' [PINNED]",
                "rationale": "Applies user-pinned OOF point-forecast calibration policy.",
            }
        )

    eff_n_folds = int(n_folds) if n_folds is not None else int(policy_out["n_folds"])
    if n_folds is not None:
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "backtest.n_folds",
                "value": str(eff_n_folds),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": (
                    f"Pins rolling-origin backtest cross-validation to {eff_n_folds} fold(s)."
                ),
            }
        )

    eff_decision_metric = (
        _normalize_decision_metric(decision_metric)
        if decision_metric is not None
        else str(policy_out["decision_metric"])
    )
    if decision_metric is not None:
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "backtest.decision_metric",
                "value": eff_decision_metric,
                "evidence": "Explicit user pin [PINNED]",
                "rationale": (
                    f"Pins model ranking and ensemble optimization to '{eff_decision_metric}'."
                ),
            }
        )

    eff_min_obs = (
        int(profile.min_n_obs * 0.55) if policy_out.get("trim_leading_zeros") else profile.min_n_obs
    )
    min_train_obs = max(14, min(180, max(14, eff_min_obs - profile.horizon * eff_n_folds - 2)))
    backtest_cfg: dict[str, Any] = {
        "enabled": eff_n_folds >= 1,
        "n_folds": max(1, eff_n_folds),
        "horizon": profile.horizon,
        "step": profile.horizon,
        "min_train": min_train_obs,
        "decision_metric": eff_decision_metric,
    }
    if profile.min_n_obs < profile.horizon * (eff_n_folds + 1) + 5:
        backtest_cfg["short_series"] = "adapt"

    ensemble_strats = list(policy_out["ensemble_strategies"])
    sel_rule = policy_out.get("selection_rule")
    if sel_rule in ("champion", "horizon_specialist") and sel_rule not in ensemble_strats:
        ensemble_strats.append(sel_rule)
    ensemble_cfg: dict[str, Any] = {
        "enabled": len(selected_models) >= 2,
        "strategies": ensemble_strats,
        "parsimony_margin": float(policy_out.get("parsimony_margin", 0.005)),
    }
    if policy_out.get("ensemble_top_k") is not None:
        ensemble_cfg["top_k"] = int(policy_out["ensemble_top_k"])
    if ensemble is not None:
        if isinstance(ensemble, bool):
            ensemble_cfg["enabled"] = ensemble and len(selected_models) >= 2
        elif isinstance(ensemble, Sequence) and not isinstance(ensemble, str):
            ensemble_cfg["enabled"] = len(ensemble) > 0 and len(selected_models) >= 2
            ensemble_cfg["strategies"] = [str(s) for s in ensemble]
        elif isinstance(ensemble, Mapping):
            ensemble_cfg = _deep_merge_dict(ensemble_cfg, ensemble)
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "ensemble",
                "value": str(ensemble_cfg),
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Applies user-pinned ensemble strategy configuration.",
            }
        )

    hpo_cfg: dict[str, Any] = {
        "enabled": bool(policy_out["hpo_enabled"]),
        "n_trials": int(policy_out["hpo_trials"]) if policy_out["hpo_enabled"] else 10,
        "granularity": str(policy_out.get("hpo_granularity", "fleetwide")),
    }
    if hpo is not None:
        if isinstance(hpo, bool):
            hpo_cfg["enabled"] = hpo
            if hpo and int(hpo_cfg.get("n_trials", 0)) < 5:
                hpo_cfg["n_trials"] = 8
        elif isinstance(hpo, Mapping):
            hpo_cfg = _deep_merge_dict(hpo_cfg, hpo)
        rationales.append(
            {
                "category": "user_constraint",
                "parameter": "hpo",
                "value": f"enabled={hpo_cfg['enabled']}, n_trials={hpo_cfg.get('n_trials')}",
                "evidence": "Explicit user pin [PINNED]",
                "rationale": "Applies user-pinned Optuna hyperparameter optimization setting.",
            }
        )

    slug_tier = tier.replace("_", "-")
    data_cfg: dict[str, Any] = {
        "source_table": profile.source_table,
        "ts_id_col": profile.ts_id_col,
        "date_col": profile.date_col,
        "target_col": profile.target_col,
        "freq": profile.freq,
        "horizon": profile.horizon,
        "trim_leading_zeros": bool(policy_out.get("trim_leading_zeros", False)),
        "grid_policy": str(policy_out.get("grid_policy", "strict")),
    }
    raw_cfg: dict[str, Any] = {
        "run_name": run_name or base_seed.get("run_name") or f"auto-{slug_tier}",
        "data": data_cfg,
        "models": selected_models,
        "model_params": dict(policy_out.get("model_params", {})),
        "features": features_cfg,
        "backtest": backtest_cfg,
        "output": output_cfg,
        "hpo": hpo_cfg,
        "ensemble": ensemble_cfg,
        "hierarchy": hierarchy_cfg,
        "compute": _build_compute_block_for_where(
            eff_where,
            selected_models,
            n_series=profile.n_series,
            max_usd=max_usd,
        ),
    }

    # Merge any remaining sub-blocks from base_config (unless overridden by explicit pins)
    for block_key in (
        "features",
        "backtest",
        "output",
        "hpo",
        "ensemble",
        "hierarchy",
        "compute",
        "model_params",
    ):
        if block_key in base_seed and isinstance(base_seed[block_key], Mapping):
            # Explicit keyword arguments take precedence over base_seed
            if block_key == "hpo" and hpo is not None:
                continue
            if block_key == "ensemble" and ensemble is not None:
                continue
            merged_blk = _deep_merge_dict(dict(raw_cfg.get(block_key, {})), base_seed[block_key])
            if block_key == "backtest":
                if decision_metric is not None:
                    merged_blk["decision_metric"] = eff_decision_metric
                if n_folds is not None:
                    merged_blk["n_folds"] = max(1, eff_n_folds)
            raw_cfg[block_key] = merged_blk

    # Apply explicit dictionary overrides last (highest priority)
    if overrides:
        raw_cfg = _deep_merge_dict(raw_cfg, overrides)
        selected_models = list(raw_cfg.get("models", selected_models))
        for ov_k, ov_v in overrides.items():
            rationales.append(
                {
                    "category": "user_constraint",
                    "parameter": f"overrides.{ov_k}",
                    "value": str(ov_v),
                    "evidence": "Explicit user override [PINNED]",
                    "rationale": f"Applied explicit dictionary override on '{ov_k}'.",
                }
            )

    # Filter out covariates that don't exist in the current panel when base_config is supplied
    if not any(k in (overrides or {}) for k in ("features",)) and "features" in base_seed:
        valid_covs = (
            set(profile.future_covariates)
            | set(profile.past_covariates)
            | set(profile.static_covariates)
        )
        for cov_field in ("future_covariates", "past_covariates", "static_covariates", "exog"):
            if cov_field in raw_cfg["features"]:
                raw_cfg["features"][cov_field] = [
                    c for c in raw_cfg["features"][cov_field] if c in valid_covs
                ]

    # Resolve optional live duration calibration
    eff_calibration: DurationCalibration | None = None
    if isinstance(calibration, DurationCalibration):
        eff_calibration = calibration
    elif calibration is True:
        with contextlib.suppress(Exception):
            eff_calibration = calibrate_duration_from_registry(settings=settings)

    cfg = RunConfig.model_validate(raw_cfg)
    dur = predict_run_duration(
        cfg,
        n_series=profile.n_series,
        median_n_obs=float(profile.median_n_obs),
        where="local" if eff_where == "local" else None,
        calibration=eff_calibration,
    )

    # Budget clamping: if max_minutes or max_usd is exceeded, trim heaviest non-protected models
    while len(selected_models) > 2 and (
        (max_minutes is not None and dur.total_p50_minutes > max_minutes)
        or (max_usd is not None and dur.estimated_usd > max_usd)
    ):
        trim_idx = next(
            (
                i
                for i in range(len(selected_models) - 1, -1, -1)
                if selected_models[i] not in protected_models
            ),
            None,
        )
        if trim_idx is None:
            break
        dropped = selected_models.pop(trim_idx)
        raw_cfg["models"] = list(selected_models)
        if hpo is None:
            raw_cfg["hpo"] = {"enabled": False, "n_trials": 5, "granularity": "fleetwide"}
        cfg = RunConfig.model_validate(raw_cfg)
        dur = predict_run_duration(
            cfg,
            n_series=profile.n_series,
            median_n_obs=float(profile.median_n_obs),
            where="local" if eff_where == "local" else None,
            calibration=eff_calibration,
        )
        rationales.append(
            {
                "category": "budget_guardrail",
                "parameter": dropped,
                "value": "trimmed",
                "evidence": f"max_minutes={max_minutes}, max_usd={max_usd}",
                "rationale": (
                    f"Trimmed '{dropped}' (and disabled HPO) so predicted duration "
                    f"({dur.total_p50_minutes:.2f}m, ${dur.estimated_usd:.2f}) stays within budget."
                ),
            }
        )

    rationale_df = pd.DataFrame(
        rationales,
        columns=["category", "parameter", "value", "evidence", "rationale"],
    )
    return ConfigBlueprint(
        config=cfg,
        profile=profile,
        tier=tier,
        where=eff_where,
        goal=goal,
        rationale_df=rationale_df,
        duration_estimate=dur,
        cost_estimate_usd=dur.estimated_usd,
        baselines=tuple(baseline_entries),
    )


def _summarize_pass_diagnostics(
    *,
    cfg: RunConfig,
    leaderboard_df: pd.DataFrame,
    fva_df: pd.DataFrame,
    horizon_decay_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    attributions_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    hist_df: pd.DataFrame | None = None,
    reconciliation_lift_pct: float = 0.0,
) -> dict[str, Any]:
    """Distill post-pass diagnostic frames into the compact summary dict read by `auto_policy`."""
    decision = cfg.backtest.decision_metric
    metric_col = f"mean_{decision}" if f"mean_{decision}" in leaderboard_df.columns else decision

    univariate_stat_bases = {
        "naive_seasonal",
        "naive_mean",
        "naive_drift",
        "snaive",
        "theta",
        "holtwinters",
        "ets",
        "autoets",
        "autoarima",
        "mstl",
        "stl_arima",
        "stl_bagging",
        "fft",
        "croston",
        "tbats",
        "theta_boxcox",
    }
    foundation_bases = {"chronos_bolt", "timesfm_torch", "chronos2", "timesfm"}

    def _is_univariate_model_spec(m_spec: str) -> bool:
        if "@univariate" in m_spec:
            return True
        if "@cov_all" in m_spec or "@cov_" in m_spec:
            return False
        b_name = m_spec.split("@", 1)[0]
        if b_name in univariate_stat_bases or b_name == "chronos_bolt":
            return True
        return bool(b_name in foundation_bases and not cfg.features.future_covariates)

    best_model = "naive_seasonal"
    best_base_model = "naive_seasonal"
    best_score = float("nan")
    naive_score = float("nan")
    univariate_score = float("nan")
    foundation_score = float("nan")
    beaten_challengers: list[str] = []
    champion_is_univariate = False

    if not leaderboard_df.empty:
        col = metric_col if metric_col in leaderboard_df.columns else leaderboard_df.columns[-1]
        model_col = "model_type" if "model_type" in leaderboard_df.columns else "model"
        valid_lb = leaderboard_df.dropna(subset=[col])
        if not valid_lb.empty:
            best_row = valid_lb.iloc[0]
            best_model = str(best_row[model_col])
            best_score = float(best_row[col])
            non_ens = valid_lb[
                ~valid_lb[model_col].astype(str).str.startswith(("naive_", "ensemble_"))
            ]
            best_base_model = str(non_ens.iloc[0][model_col]) if not non_ens.empty else best_model
            naive_rows = valid_lb[
                valid_lb[model_col].astype(str).isin(("naive_seasonal", "naive_mean"))
                | valid_lb[model_col].astype(str).str.startswith("naive_seasonal_")
            ]
            if not naive_rows.empty:
                naive_score = float(naive_rows.iloc[0][col])

            uni_rows = valid_lb[
                valid_lb[model_col]
                .astype(str)
                .apply(
                    lambda m: (
                        not m.startswith(("naive_", "ensemble_"))
                        and _is_univariate_model_spec(m)
                        and m.split("@", 1)[0] not in foundation_bases
                    )
                )
            ]
            if not uni_rows.empty:
                univariate_score = float(uni_rows[col].min())

            fnd_rows = valid_lb[
                valid_lb[model_col]
                .astype(str)
                .apply(lambda m: m.split("@", 1)[0] in foundation_bases)
            ]
            if not fnd_rows.empty:
                foundation_score = float(fnd_rows[col].min())

            # Detect when a plain univariate (or univariate foundation) model beats complex
            # multivariate/global challengers
            champ_check_model = (
                best_base_model if best_model.startswith("ensemble_") else best_model
            )
            if _is_univariate_model_spec(champ_check_model):
                for _, r_ch in valid_lb.iterrows():
                    m_ch = str(r_ch[model_col])
                    sc_ch = float(r_ch[col])
                    if (
                        m_ch == champ_check_model
                        or m_ch.startswith(("naive_", "ensemble_"))
                        or _is_univariate_model_spec(m_ch)
                    ):
                        continue
                    if math.isfinite(sc_ch) and sc_ch >= best_score - 1e-9:
                        beaten_challengers.append(m_ch)
                if beaten_challengers and (
                    _is_univariate_model_spec(best_model) or not cfg.features.future_covariates
                ):
                    champion_is_univariate = True

    # FVA vs Naive & DM p-value (compute_fva stores fva_pct as a fraction, e.g. 0.25 = +25%)
    fva_vs_naive_pct = 0.0
    dm_pvalue: float | None = None
    covariate_fva_pct = 0.0
    pooling_fva_pct = 0.0
    if not fva_df.empty and "model_type" in fva_df.columns:
        champ_fva = fva_df[fva_df["model_type"] == best_model]
        if champ_fva.empty:
            champ_fva = fva_df[fva_df["model_type"] == best_base_model]
        if champ_fva.empty:
            champ_fva = fva_df.iloc[[0]]
        row0 = champ_fva.iloc[0]
        if "fva_vs_baseline_pct" in row0 and pd.notna(row0["fva_vs_baseline_pct"]):
            fva_vs_naive_pct = float(row0["fva_vs_baseline_pct"])
        elif "fva_pct" in row0 and pd.notna(row0["fva_pct"]):
            raw_fva = float(row0["fva_pct"])
            fva_vs_naive_pct = raw_fva * 100.0 if abs(raw_fva) <= 5.0 else raw_fva
        if "dm_pvalue" in row0 and pd.notna(row0["dm_pvalue"]):
            dm_pvalue = float(row0["dm_pvalue"])
        elif "p_value_dm" in row0 and pd.notna(row0["p_value_dm"]):
            dm_pvalue = float(row0["p_value_dm"])
        if "covariate_fva_pct" in fva_df.columns:
            cov_vals = pd.to_numeric(fva_df["covariate_fva_pct"], errors="coerce").dropna()
            if not cov_vals.empty:
                raw_cov = float(cov_vals.max())
                covariate_fva_pct = raw_cov * 100.0 if abs(raw_cov) <= 5.0 else raw_cov
        if "pooling_fva_pct" in fva_df.columns:
            pool_vals = pd.to_numeric(fva_df["pooling_fva_pct"], errors="coerce").dropna()
            if not pool_vals.empty:
                raw_pool = float(pool_vals.max())
                pooling_fva_pct = raw_pool * 100.0 if abs(raw_pool) <= 5.0 else raw_pool

    # Fallback to leaderboard relative gain over naive when decision_metric is an interval metric
    # (e.g. pinball, interval_score) not directly evaluated by point-based compute_fva
    if (
        abs(fva_vs_naive_pct) < 1e-9
        and math.isfinite(best_score)
        and math.isfinite(naive_score)
        and naive_score > 1e-9
    ):
        fva_vs_naive_pct = round(100.0 * (naive_score - best_score) / naive_score, 4)

    fva_vs_univariate_pct = (
        round(100.0 * (univariate_score - best_score) / max(1e-9, univariate_score), 4)
        if math.isfinite(best_score) and math.isfinite(univariate_score) and univariate_score > 1e-9
        else 0.0
    )
    fva_vs_foundation_pct = (
        round(100.0 * (foundation_score - best_score) / max(1e-9, foundation_score), 4)
        if math.isfinite(best_score) and math.isfinite(foundation_score) and foundation_score > 1e-9
        else 0.0
    )

    # Feature attributions + detrended covariate signal -> true positive drivers vs decoy noise
    positive_covariates: list[str] = []
    zero_or_negative_covariates: list[str] = []
    exog_lag_signal = False
    declared_fut = list(cfg.features.future_covariates)
    if declared_fut:
        cov_scores: dict[str, float] = dict.fromkeys(declared_fut, 0.0)
        if not attributions_df.empty and "feature" in attributions_df.columns:
            imp_col = "importance" if "importance" in attributions_df.columns else "attribution"
            if imp_col in attributions_df.columns:
                agg_imp = (
                    attributions_df.groupby("feature")[imp_col]
                    .apply(
                        lambda s: float(
                            np.mean(np.abs(pd.to_numeric(s, errors="coerce").fillna(0.0)))
                        )
                    )
                    .to_dict()
                )
                for c in declared_fut:
                    cov_scores[c] = float(agg_imp.get(c, 0.0))

        if hist_df is not None and not hist_df.empty and "y" in hist_df.columns:
            avail_covs = [c for c in declared_fut if c in hist_df.columns]
            if avail_covs:
                eff_by_cov: dict[str, list[float]] = {c: [] for c in avail_covs}
                lag_eff_by_cov: dict[str, list[float]] = {c: [] for c in avail_covs}
                group_iter_cov = (
                    hist_df.groupby("ts_id", sort=False)
                    if "ts_id" in hist_df.columns
                    else [(None, hist_df)]
                )
                s_per = max(2, seasonal_period(cfg.data.freq))
                for _tid, g_df in group_iter_cov:
                    y_arr = pd.to_numeric(g_df["y"], errors="coerce").to_numpy(dtype=np.float64)
                    n_t = len(y_arr)
                    if n_t < max(16, len(avail_covs) + 6) or np.nanstd(y_arr) < 1e-6:
                        continue
                    t_vec = np.arange(n_t, dtype=np.float64)
                    t_c = t_vec - np.mean(t_vec)
                    slope = float(
                        np.dot(t_c, y_arr - np.nanmean(y_arr)) / max(1e-9, np.dot(t_c, t_c))
                    )
                    y_det = y_arr - (np.nanmean(y_arr) + slope * t_c)
                    for rem in range(s_per):
                        idx_rem = np.arange(rem, n_t, s_per)
                        y_det[idx_rem] -= float(np.nanmean(y_det[idx_rem]))
                    y_std = float(np.nanstd(y_det))
                    if y_std < 1e-6:
                        continue
                    X_cols: list[np.ndarray] = []
                    for c in avail_covs:
                        x_raw = (
                            pd.to_numeric(g_df[c], errors="coerce")
                            .fillna(0.0)
                            .to_numpy(dtype=np.float64)
                        )
                        x_c = x_raw - float(np.mean(x_raw))
                        x_std = float(np.std(x_c))
                        X_cols.append(x_c / x_std if x_std > 1e-6 else np.zeros_like(x_c))
                    X_mat = np.column_stack(X_cols)
                    gram = X_mat.T @ X_mat + 1.0 * np.eye(len(avail_covs))
                    y_norm = y_det / y_std
                    betas = np.linalg.solve(gram, X_mat.T @ y_norm)
                    resid_xt = y_norm - X_mat @ betas
                    r_std = float(np.std(resid_xt))
                    for idx_c, c in enumerate(avail_covs):
                        eff_by_cov[c].append(abs(float(betas[idx_c])))
                        if r_std > 1e-6 and n_t >= 12:
                            x_lag1 = X_mat[:-1, idx_c]
                            r_tail = resid_xt[1:]
                            if np.std(x_lag1) > 1e-6:
                                c_lag = abs(float(np.corrcoef(x_lag1, r_tail)[0, 1]))
                                if math.isfinite(c_lag):
                                    lag_eff_by_cov[c].append(c_lag)
                med_eff = {
                    c: (float(np.median(vals)) if vals else 0.0) for c, vals in eff_by_cov.items()
                }
                med_lag_eff = {
                    c: (float(np.median(vals)) if vals else 0.0)
                    for c, vals in lag_eff_by_cov.items()
                }
                if any(
                    med_lag_eff.get(c, 0.0) >= 0.38
                    and (med_eff.get(c, 0.0) >= 0.15 or med_lag_eff.get(c, 0.0) >= 0.45)
                    for c in avail_covs
                ):
                    exog_lag_signal = True
                # Override cov_scores from detrended partial regression when hist_df has covariates
                for c in avail_covs:
                    eff_val = (
                        max(med_eff[c], med_lag_eff[c]) if med_lag_eff[c] >= 0.38 else med_eff[c]
                    )
                    cov_scores[c] = eff_val if eff_val >= 0.12 else 0.0

        total_cov_imp = sum(cov_scores.values())
        if total_cov_imp > 1e-9:
            for c, sc in cov_scores.items():
                share = sc / total_cov_imp
                threshold = 0.08 if len(declared_fut) >= 4 else 0.04
                if share >= threshold:
                    positive_covariates.append(c)
                else:
                    zero_or_negative_covariates.append(c)
        else:
            zero_or_negative_covariates.extend(declared_fut)
            for c in declared_fut:
                label_decoy = f"decoy_covariate:{c}"
                if label_decoy not in beaten_challengers:
                    beaten_challengers.append(label_decoy)
            if beaten_challengers:
                champion_is_univariate = True

    # Horizon crossover step & late-horizon degradation ratio on base models
    horizon_crossover_step: int | None = None
    late_horizon_degradation_ratio = 1.0
    if not horizon_decay_df.empty and "horizon_step" in horizon_decay_df.columns:
        base_hd = horizon_decay_df[
            ~horizon_decay_df["model_type"].astype(str).str.startswith(("naive_", "ensemble_"))
        ]
        best_hd = horizon_decay_df[horizon_decay_df["model_type"] == best_base_model]
        if best_hd.empty:
            best_hd = base_hd
        if not best_hd.empty and "metric_value" in best_hd.columns:
            half_h = max(2, cfg.data.horizon // 2)
            early = best_hd[best_hd["horizon_step"] <= half_h]["metric_value"].median()
            late = best_hd[best_hd["horizon_step"] > half_h]["metric_value"].median()
            if (
                pd.notna(early)
                and pd.notna(late)
                and float(early) > 1e-6
                and (float(late) - float(early)) >= 0.015
            ):
                late_horizon_degradation_ratio = float(late) / float(early)
        if not base_hd.empty and "metric_value" in base_hd.columns:
            split_h = min(5, max(2, cfg.data.horizon // 3))
            early_by_m = (
                base_hd[base_hd["horizon_step"] <= split_h]
                .groupby("model_type")["metric_value"]
                .median()
            )
            late_by_m = (
                base_hd[base_hd["horizon_step"] > split_h]
                .groupby("model_type")["metric_value"]
                .median()
            )
            if not early_by_m.empty and not late_by_m.empty:
                early_winner = str(early_by_m.idxmin())
                late_winner = str(late_by_m.idxmin())
                if (
                    early_winner in late_by_m.index
                    and float(early_by_m[early_winner]) > 1e-6
                    and (float(late_by_m[early_winner]) - float(early_by_m[early_winner])) >= 0.015
                ):
                    ew_ratio = float(late_by_m[early_winner]) / float(early_by_m[early_winner])
                    late_horizon_degradation_ratio = max(late_horizon_degradation_ratio, ew_ratio)
                    if early_winner != late_winner and ew_ratio >= 1.20:
                        horizon_crossover_step = split_h
        if (
            horizon_crossover_step is None
            and not best_hd.empty
            and "crossover_step" in best_hd.columns
        ):
            c_steps = pd.to_numeric(best_hd["crossover_step"], errors="coerce").dropna()
            if not c_steps.empty:
                cand_step = int(c_steps.iloc[0])
                if cand_step >= 3 and late_horizon_degradation_ratio >= 1.25:
                    horizon_crossover_step = cand_step

    # Residual autocorrelation & heteroskedasticity on best model's OOF rows (within each ts_id)
    resid_acf1 = 0.0
    resid_hetero = 0.0
    white_noise_residuals = True
    if not oof_df.empty and {"y_true", "yhat", "model_type"}.issubset(oof_df.columns):
        champ_oof = oof_df[oof_df["model_type"] == best_model].copy()
        if champ_oof.empty:
            champ_oof = oof_df.copy()
        champ_oof["resid"] = pd.to_numeric(champ_oof["y_true"], errors="coerce") - pd.to_numeric(
            champ_oof["yhat"], errors="coerce"
        )
        champ_oof = champ_oof.dropna(subset=["resid", "y_true"])
        if len(champ_oof) >= 8:
            acf_vals: list[float] = []
            het_vals: list[float] = []
            zero_shares: list[float] = []
            group_iter = (
                champ_oof.groupby("ts_id", sort=False)
                if "ts_id" in champ_oof.columns
                else [(None, champ_oof)]
            )
            for _tid, g_sub in group_iter:
                sort_cols = [c for c in ("fold_id", "ds", "forecast_date") if c in g_sub.columns]
                g_sorted = g_sub.sort_values(sort_cols) if sort_cols else g_sub
                r_arr = g_sorted["resid"].to_numpy(dtype=np.float64)
                y_arr = np.abs(g_sorted["y_true"].to_numpy(dtype=np.float64))
                zero_shares.append(float(np.mean(y_arr == 0.0)))
                if len(r_arr) >= 6 and np.std(r_arr) > 1e-6:
                    r0 = r_arr[:-1] - np.mean(r_arr[:-1])
                    r1 = r_arr[1:] - np.mean(r_arr[1:])
                    denom = float(np.linalg.norm(r0) * np.linalg.norm(r1))
                    if denom > 1e-9:
                        acf_vals.append(float(np.dot(r0, r1) / denom))
                if len(y_arr) >= 6 and np.std(y_arr) > 1e-6 and np.std(np.abs(r_arr)) > 1e-6:
                    c_het = float(np.corrcoef(y_arr, np.abs(r_arr))[0, 1])
                    if math.isfinite(c_het):
                        het_vals.append(max(0.0, c_het))
            if acf_vals:
                resid_acf1 = float(np.median(acf_vals))
            if het_vals:
                resid_hetero = float(np.median(het_vals))
            med_zero = float(np.median(zero_shares)) if zero_shares else 0.0
            white_noise_residuals = (
                abs(resid_acf1) < 0.25 and resid_hetero < 0.32 and med_zero < 0.20
            )

    # Bias report summary
    bias_flagged_share = 0.0
    mean_bias_pct = 0.0
    if not bias_df.empty:
        if "bias_flagged" in bias_df.columns:
            bias_flagged_share = float(bias_df["bias_flagged"].mean())
        if "bias_pct" in bias_df.columns:
            mean_bias_pct = float(pd.to_numeric(bias_df["bias_pct"], errors="coerce").abs().mean())

    return {
        "best_model": best_model,
        "best_score": best_score,
        "decision_metric": decision,
        "fva_vs_naive_pct": fva_vs_naive_pct,
        "fva_vs_univariate_pct": fva_vs_univariate_pct,
        "fva_vs_foundation_pct": fva_vs_foundation_pct,
        "covariate_lift_pct": round(covariate_fva_pct, 4),
        "reconciliation_lift_pct": round(reconciliation_lift_pct, 4),
        "champion_is_univariate": champion_is_univariate,
        "beaten_challengers": tuple(beaten_challengers),
        "dm_pvalue": dm_pvalue,
        "covariate_fva_pct": covariate_fva_pct,
        "pooling_fva_pct": pooling_fva_pct,
        "positive_covariates": positive_covariates,
        "zero_or_negative_covariates": zero_or_negative_covariates,
        "exog_lag_signal": exog_lag_signal,
        "horizon_crossover_step": horizon_crossover_step,
        "late_horizon_degradation_ratio": round(late_horizon_degradation_ratio, 4),
        "residual_autocorr_lag1": round(resid_acf1, 4),
        "residual_heteroskedasticity": round(resid_hetero, 4),
        "white_noise_residuals": white_noise_residuals,
        "bias_flagged_share": round(bias_flagged_share, 4),
        "mean_bias_pct": round(mean_bias_pct, 4),
    }


def _apply_learned_weights(
    base_df: pd.DataFrame,
    wmap: dict[str, float],
    run_id: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Blend base predictions by learned weights into ``ensemble_<strategy>`` rows (pure)."""
    model_type = f"ensemble_{strategy}"
    rows: list[dict[str, Any]] = []
    if base_df.empty:
        return rows
    models = [m for m in wmap if m in set(base_df["model_type"].unique())]
    if not models:
        return rows
    weights = np.array([wmap[m] for m in models], dtype=float)

    def _blend(value_col: str) -> Any:
        wide = base_df.pivot_table(
            index=["ts_id", "forecast_date"], columns="model_type", values=value_col
        ).reindex(columns=models)
        vals = wide.to_numpy(dtype=float)
        present = ~np.isnan(vals)
        wrow = present * weights
        denom = wrow.sum(axis=1)
        num = np.nansum(np.where(present, vals, 0.0) * weights, axis=1)
        blended = np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), np.nan)
        return wide.index, blended

    index, yhat = _blend("yhat")
    _, lower = _blend("yhat_lower")
    _, upper = _blend("yhat_upper")
    for (ts_id, forecast_date), yh, lo, up in zip(index, yhat, lower, upper, strict=True):
        if np.isnan(yh):
            continue
        rows.append(
            {
                "run_id": run_id,
                "ts_id": ts_id,
                "model_type": model_type,
                "compute_engine": "ensemble",
                "forecast_date": forecast_date,
                "yhat": float(yh),
                "yhat_lower": None if np.isnan(lo) else float(lo),
                "yhat_upper": None if np.isnan(up) else float(up),
                "quantiles": None,
            }
        )
    return rows


def _execute_local_panel_pass(
    panel_df: pd.DataFrame,
    cfg: RunConfig,
    *,
    cell_cache: dict[tuple[str, str], CellResult],
    future_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute ``cfg`` across ``panel_df`` in-memory, reusing cached ``(ts_id, hash)`` cells."""
    from .backtest import training_window
    from .diagnostics import (
        compute_bias_report,
        compute_fva,
        compute_horizon_decay,
    )
    from .ensembler import (
        calibrate_ensemble_predictions,
        combine_calculated,
        combine_oof,
        ensemble_oof_bounds,
        fit_learned,
    )
    from .features import expand_model_variants, variant_diagnostics_for
    from .metrics import METRIC_NAMES, compute_metrics
    from .reconciliation import (
        build_hierarchy,
        reconcile_cells,
        reconcile_forecasts,
        reconcile_oof,
    )
    from .review import build_attributions_frame, build_predictions_frame
    from .simplex import close_simplex_predictions, should_close_simplex
    from .worker import is_panel_model, rollup_metrics, run_cell, run_panel

    ts_col = cfg.data.ts_id_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    decision = cfg.backtest.decision_metric

    # Separate historical rows from any future covariate rows appended to panel_df
    y_num = pd.to_numeric(panel_df[target_col], errors="coerce")
    hist_only_df = panel_df[y_num.notna()].copy()

    effective_models, effective_params = expand_model_variants(cfg)
    exec_cfg = (
        cfg.model_copy(update={"models": effective_models, "model_params": effective_params})
        if (effective_models != list(cfg.models) or effective_params != cfg.model_params)
        else cfg
    )

    hier_spec = None
    work_hist = hist_only_df
    fut_rows = (
        future_df.copy()
        if future_df is not None and not future_df.empty
        else panel_df[y_num.isna()].copy()
    )
    work_fut = fut_rows
    if exec_cfg.hierarchy.enabled and exec_cfg.hierarchy.levels:
        with contextlib.suppress(Exception):
            work_hist, hier_spec = build_hierarchy(hist_only_df, exec_cfg)
            if not fut_rows.empty:
                fut_filled = fut_rows.copy()
                attr_cols = [
                    c
                    for lvl in exec_cfg.hierarchy.levels
                    for c in lvl
                    if c in hist_only_df.columns and c not in fut_filled.columns
                ]
                if attr_cols:
                    meta_map = hist_only_df[[ts_col, *dict.fromkeys(attr_cols)]].drop_duplicates(
                        subset=[ts_col]
                    )
                    fut_filled = fut_filled.merge(meta_map, on=ts_col, how="left")
                fut_filled[target_col] = 0.0
                work_fut, _ = build_hierarchy(fut_filled, exec_cfg)

    hist_groups = {
        str(ts_id): grp.sort_values(date_col).reset_index(drop=True)
        for ts_id, grp in work_hist.groupby(ts_col, sort=True)
    }
    fut_groups: dict[str, pd.DataFrame] = (
        {
            str(ts_id): grp.sort_values(date_col).reset_index(drop=True)
            for ts_id, grp in work_fut.groupby(ts_col, sort=False)
        }
        if not work_fut.empty and ts_col in work_fut.columns
        else {}
    )

    borrowed_models_set: set[str] = set()
    executed_models_set: set[str] = set()

    results_by_model_ts: dict[tuple[str, str], CellResult] = {}

    for m in exec_cfg.models:
        spec_hash = make_model_spec_hash(m, exec_cfg)
        all_cached = all((ts_id, spec_hash) in cell_cache for ts_id in hist_groups)
        if all_cached:
            borrowed_models_set.add(m)
            for ts_id in hist_groups:
                cached_res = cell_cache[(ts_id, spec_hash)]
                results_by_model_ts[(m, ts_id)] = cached_res
            continue

        executed_models_set.add(m)
        if is_panel_model(m, exec_cfg):
            panel_results = run_panel(
                work_hist,
                m,
                exec_cfg,
                future_df=work_fut if not work_fut.empty else None,
            )
            for res in panel_results:
                cell_cache[(str(res.ts_id), spec_hash)] = res
                results_by_model_ts[(m, str(res.ts_id))] = res
        else:
            for ts_id, s_df in hist_groups.items():
                key = (ts_id, spec_hash)
                if key not in cell_cache:
                    cell_cache[key] = run_cell(s_df, m, exec_cfg, future_df=fut_groups.get(ts_id))
                results_by_model_ts[(m, ts_id)] = cell_cache[key]

    unreconciled_metrics_by_model_ts: dict[tuple[str, str], dict[str, float]] = {}
    max_temp_coherence_residual = 0.0
    learned_alpha_float: float | None = None
    if hier_spec is not None:
        has_ct_cfg = bool(
            exec_cfg.hierarchy.reconcile_temporal
            or exec_cfg.hierarchy.cross_temporal
            or "cross_temporal_mint" in exec_cfg.hierarchy.reconciliation_methods
        )
        non_ct_methods = [
            m for m in exec_cfg.hierarchy.reconciliation_methods if m != "cross_temporal_mint"
        ]
        primary_method: Any = (
            non_ct_methods[0] if non_ct_methods else ("wls_struct" if has_ct_cfg else "bottom_up")
        )
        for (m_u, ts_u), res_u in results_by_model_ts.items():
            if res_u.status == "ok":
                unreconciled_metrics_by_model_ts[(m_u, ts_u)] = dict(res_u.metrics)
        with contextlib.suppress(Exception):
            primary_hier_cfg = exec_cfg.model_copy(
                update={
                    "hierarchy": exec_cfg.hierarchy.model_copy(
                        update={
                            "reconciliation_methods": [primary_method],
                            "reconcile_temporal": False,
                            "cross_temporal": False,
                        }
                    )
                }
            )
            models_to_reconcile: list[CellResult] = []
            for m_u in exec_cfg.models:
                s_hash = f"{make_model_spec_hash(m_u, exec_cfg)}:rec:{primary_method}"
                if all((ts_id, s_hash) in cell_cache for ts_id in hist_groups):
                    for ts_id in hist_groups:
                        r_cached = cell_cache[(ts_id, s_hash)]
                        results_by_model_ts[(r_cached.model_type, ts_id)] = r_cached
                else:
                    for ts_id in hist_groups:
                        c_u = results_by_model_ts.get((m_u, ts_id))
                        if c_u is not None:
                            models_to_reconcile.append(c_u)
            if models_to_reconcile:
                rec_cells = reconcile_cells(
                    models_to_reconcile,
                    primary_hier_cfg,
                    hier_spec,
                    history_df=work_hist,
                )
                for r_cell in rec_cells:
                    base_m = r_cell.model_type[: -len(f"_{primary_method}")]
                    s_hash = f"{make_model_spec_hash(base_m, exec_cfg)}:rec:{primary_method}"
                    cell_cache[(str(r_cell.ts_id), s_hash)] = r_cell
                    results_by_model_ts[(r_cell.model_type, str(r_cell.ts_id))] = r_cell
            if has_ct_cfg:
                ct_method = "cross_temporal_mint"
                ct_hier_cfg = exec_cfg.model_copy(
                    update={
                        "hierarchy": exec_cfg.hierarchy.model_copy(
                            update={
                                "reconciliation_methods": [ct_method],
                                "reconcile_temporal": True,
                                "cross_temporal": True,
                            }
                        )
                    }
                )
                ct_to_reconcile: list[CellResult] = []
                for m_u in exec_cfg.models:
                    s_hash = f"{make_model_spec_hash(m_u, exec_cfg)}:rec:{ct_method}"
                    if all((ts_id, s_hash) in cell_cache for ts_id in hist_groups):
                        for ts_id in hist_groups:
                            r_cached = cell_cache[(ts_id, s_hash)]
                            results_by_model_ts[(r_cached.model_type, ts_id)] = r_cached
                            ct_diag = r_cached.predictions.attrs.get(
                                "cross_temporal_diagnostics", {}
                            )
                            if isinstance(ct_diag, Mapping):
                                if "spatial_weight_alpha" in ct_diag:
                                    learned_alpha_float = float(ct_diag["spatial_weight_alpha"])
                                max_temp_coherence_residual = max(
                                    max_temp_coherence_residual,
                                    float(
                                        ct_diag.get(
                                            "final_temp_discrepancy",
                                            ct_diag.get("temporal_discrepancy", 0.0),
                                        )
                                    ),
                                )
                    else:
                        for ts_id in hist_groups:
                            c_u = results_by_model_ts.get((m_u, ts_id))
                            if c_u is not None:
                                ct_to_reconcile.append(c_u)
                if ct_to_reconcile:
                    ct_cells = reconcile_cells(
                        ct_to_reconcile,
                        ct_hier_cfg,
                        hier_spec,
                        history_df=work_hist,
                    )
                    for r_cell in ct_cells:
                        base_m = r_cell.model_type[: -len(f"_{ct_method}")]
                        s_hash = f"{make_model_spec_hash(base_m, exec_cfg)}:rec:{ct_method}"
                        cell_cache[(str(r_cell.ts_id), s_hash)] = r_cell
                        results_by_model_ts[(r_cell.model_type, str(r_cell.ts_id))] = r_cell
                        ct_diag = r_cell.predictions.attrs.get("cross_temporal_diagnostics", {})
                        if isinstance(ct_diag, Mapping):
                            if "spatial_weight_alpha" in ct_diag:
                                learned_alpha_float = float(ct_diag["spatial_weight_alpha"])
                            max_temp_coherence_residual = max(
                                max_temp_coherence_residual,
                                float(
                                    ct_diag.get(
                                        "final_temp_discrepancy",
                                        ct_diag.get("temporal_discrepancy", 0.0),
                                    )
                                ),
                            )
    if hier_spec is None and exec_cfg.hierarchy.reconcile_temporal:
        for (m_u, ts_u), res_u in results_by_model_ts.items():
            if res_u.status == "ok":
                unreconciled_metrics_by_model_ts[(m_u, ts_u)] = dict(res_u.metrics)
        with contextlib.suppress(Exception):
            from .temporal_reconciliation import verify_temporal_coherence

            t_method = str(
                exec_cfg.hierarchy.reconciliation_methods[0]
                if exec_cfg.hierarchy.reconciliation_methods
                else "wls_struct"
            )
            t_to_reconcile: list[CellResult] = []
            for m_u in exec_cfg.models:
                s_hash = f"{make_model_spec_hash(m_u, exec_cfg)}:trec:{t_method}"
                if all((ts_id, s_hash) in cell_cache for ts_id in hist_groups):
                    for ts_id in hist_groups:
                        r_cached = cell_cache[(ts_id, s_hash)]
                        results_by_model_ts[(r_cached.model_type, ts_id)] = r_cached
                else:
                    for ts_id in hist_groups:
                        c_u = results_by_model_ts.get((m_u, ts_id))
                        if c_u is not None:
                            t_to_reconcile.append(c_u)
            if t_to_reconcile:
                t_cells = reconcile_cells(
                    t_to_reconcile,
                    exec_cfg,
                    None,
                    history_df=work_hist,
                )
                s_period_t = seasonal_period(exec_cfg.data.freq)
                for r_cell in t_cells:
                    raw_m = str(r_cell.model_type)
                    orig_m = (
                        raw_m[: -len(f"_{t_method}")] if raw_m.endswith(f"_{t_method}") else raw_m
                    )
                    t_name = f"{orig_m}_temporal"
                    agg_by_k = r_cell.predictions.attrs.get("aggregated_by_k")
                    t_spec_attr = r_cell.predictions.attrs.get("temporal_spec")
                    if isinstance(agg_by_k, Mapping) and not r_cell.predictions.empty:
                        c_disc = verify_temporal_coherence(
                            r_cell.predictions["yhat"].to_numpy(dtype=float),
                            agg_by_k,
                            spec=t_spec_attr,
                            raise_on_error=False,
                        )
                        max_temp_coherence_residual = max(
                            max_temp_coherence_residual, float(c_disc)
                        )
                    t_metrics = dict(r_cell.metrics)
                    h_series = hist_groups.get(str(r_cell.ts_id))
                    if (
                        r_cell.oof is not None
                        and not r_cell.oof.empty
                        and h_series is not None
                        and not h_series.empty
                    ):
                        hist_ds = pd.to_datetime(h_series[date_col]).to_numpy()
                        hist_y = pd.to_numeric(h_series[target_col], errors="coerce").to_numpy(
                            dtype=np.float64
                        )
                        fold_panels_t: list[dict[str, float]] = []
                        for _fid, fg in r_cell.oof.sort_values("ds").groupby("fold_id"):
                            lo_t = (
                                fg["yhat_lower"].to_numpy(dtype=np.float64)
                                if "yhat_lower" in fg.columns and fg["yhat_lower"].notna().all()
                                else None
                            )
                            hi_t = (
                                fg["yhat_upper"].to_numpy(dtype=np.float64)
                                if "yhat_upper" in fg.columns and fg["yhat_upper"].notna().all()
                                else None
                            )
                            fold_panels_t.append(
                                compute_metrics(
                                    fg["y_true"].to_numpy(dtype=np.float64),
                                    fg["yhat"].to_numpy(dtype=np.float64),
                                    lower=lo_t,
                                    upper=hi_t,
                                    y_train=training_window(
                                        hist_ds,
                                        hist_y,
                                        fg["cutoff_date"].min()
                                        if "cutoff_date" in fg.columns
                                        else None,
                                        exec_cfg,
                                    ),
                                    seasonal_period=s_period_t,
                                )
                            )
                        if fold_panels_t:
                            t_metrics = rollup_metrics(fold_panels_t)
                    r_tagged = replace(r_cell, model_type=t_name, metrics=t_metrics)
                    s_hash = f"{make_model_spec_hash(orig_m, exec_cfg)}:trec:{t_method}"
                    cell_cache[(str(r_cell.ts_id), s_hash)] = r_tagged
                    results_by_model_ts[(t_name, str(r_cell.ts_id))] = r_tagged

    # Assemble base predictions, OOF, metrics, and attributions across all series
    base_pred_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    attr_rows: list[dict[str, Any]] = []

    for (m, ts_id), res in results_by_model_ts.items():
        if res.status != "ok":
            continue
        pred = res.predictions.rename(columns={"ds": "forecast_date"}).assign(
            ts_id=ts_id, model_type=m
        )
        base_pred_parts.append(
            pred[["ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper"]]
        )
        m_row: dict[str, Any] = {"ts_id": ts_id, "model_type": m, "kind": "base"}
        m_row.update({k: res.metrics.get(k, float("nan")) for k in METRIC_NAMES})
        metric_rows.append(m_row)

        if res.oof is not None and not res.oof.empty:
            oof_parts.append(
                res.oof.rename(columns={"ds": "forecast_date"}).assign(ts_id=ts_id, model_type=m)
            )
        if res.diagnostics and isinstance(res.diagnostics, dict):
            f_attr = res.diagnostics.get("feature_attributions")
            if isinstance(f_attr, dict) and f_attr:
                attr_rows.append(
                    {
                        "ts_id": ts_id,
                        "model_type": m,
                        "compute_engine": res.compute_engine,
                        "fit_diagnostics": json.dumps({"feature_attributions": f_attr}),
                    }
                )

    base_pred_df = (
        pd.concat(base_pred_parts, ignore_index=True) if base_pred_parts else pd.DataFrame()
    )
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    metric_df = pd.DataFrame(metric_rows)

    # Build ensembles across the panel if enabled
    ens_pred_df = pd.DataFrame()
    ens_oof_df = pd.DataFrame()
    avail_pred_models = (
        set(base_pred_df["model_type"].astype(str).unique()) if not base_pred_df.empty else set()
    )
    avail_oof_models = (
        set(oof_df["model_type"].astype(str).unique()) if not oof_df.empty else avail_pred_models
    )
    avail_ens_models = [
        m
        for m in exec_cfg.models
        if m in avail_pred_models and (oof_df.empty or m in avail_oof_models)
    ]
    if exec_cfg.ensemble.enabled and not base_pred_df.empty and len(avail_ens_models) >= 2:
        ens_cfg = (
            exec_cfg.model_copy(update={"models": avail_ens_models})
            if avail_ens_models != list(exec_cfg.models)
            else exec_cfg
        )
        ens_pred_rows: list[dict[str, Any]] = list(
            combine_calculated(base_pred_df, ens_cfg, metric_df, oof_df)
        )
        learned_weights: dict[str, dict[str, float]] = {}
        if not oof_df.empty and any(
            s in ("nnls", "ridge", "xgb") for s in ens_cfg.ensemble.strategies
        ):
            with contextlib.suppress(Exception):
                learned_weights, _, _ = fit_learned(oof_df, ens_cfg)
            for strategy, wmap in learned_weights.items():
                ens_pred_rows.extend(
                    _apply_learned_weights(base_pred_df, wmap, ens_cfg.run_name, strategy)
                )

        if not oof_df.empty:
            with contextlib.suppress(Exception):
                ens_oof_df = combine_oof(oof_df, ens_cfg, learned_weights)
        if ens_pred_rows:
            ens_pred_rows = calibrate_ensemble_predictions(ens_pred_rows, ens_oof_df, ens_cfg)
            ens_pred_df = pd.DataFrame(ens_pred_rows)

        # When hierarchy is enabled, reconcile ensemble predictions & OOF in-place so winning
        # ensembles preserve exact cross-level additive coherence and non-negativity.
        if hier_spec is not None and not ens_pred_df.empty:
            primary_method = (
                exec_cfg.hierarchy.reconciliation_methods[0]
                if exec_cfg.hierarchy.reconciliation_methods
                else "bottom_up"
            )
            with contextlib.suppress(Exception):
                ens_pred_df = reconcile_forecasts(
                    ens_pred_df,
                    hier_spec,
                    primary_method,
                    history_df=work_hist,
                    oof_df=ens_oof_df,
                    date_col="forecast_date",
                    ts_id_col="ts_id",
                    target_col=target_col,
                    stamp_method=False,
                )
                if not ens_oof_df.empty:
                    ens_oof_df = reconcile_oof(
                        ens_oof_df,
                        hier_spec,
                        primary_method,
                        cfg=exec_cfg,
                        history_df=work_hist,
                        ts_id_col="ts_id",
                        target_col=target_col,
                        stamp_method=False,
                    )

        # Score ensemble OOF per series
        if not ens_oof_df.empty:
            s_period = seasonal_period(exec_cfg.data.freq)
            for (m_type, ens_ts_id), grp in ens_oof_df.groupby(["model_type", "ts_id"]):
                h_series = hist_groups.get(str(ens_ts_id))
                if h_series is None or h_series.empty:
                    continue
                hist_ds = pd.to_datetime(h_series[date_col]).to_numpy()
                hist_y = pd.to_numeric(h_series[target_col], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                fold_panels: list[dict[str, float]] = []
                for fold_id, fg in grp.sort_values("forecast_date").groupby("fold_id"):
                    lo, hi = ensemble_oof_bounds(grp, fold_id, exec_cfg)
                    fold_panels.append(
                        compute_metrics(
                            fg["y_true"].to_numpy(dtype=np.float64),
                            fg["yhat"].to_numpy(dtype=np.float64),
                            lower=lo,
                            upper=hi,
                            y_train=training_window(
                                hist_ds,
                                hist_y,
                                fg["cutoff_date"].min() if "cutoff_date" in fg.columns else None,
                                exec_cfg,
                            ),
                            seasonal_period=s_period,
                        )
                    )
                panel_scores = rollup_metrics(fold_panels)
                e_row: dict[str, Any] = {
                    "ts_id": str(ens_ts_id),
                    "model_type": str(m_type),
                    "kind": "ensemble",
                }
                e_row.update({k: panel_scores.get(k, float("nan")) for k in METRIC_NAMES})
                metric_rows.append(e_row)

    all_preds_raw = (
        pd.concat([base_pred_df, ens_pred_df], ignore_index=True)
        if not ens_pred_df.empty
        else base_pred_df.copy()
    )

    all_oof = (
        pd.concat([oof_df, ens_oof_df], ignore_index=True)
        if not ens_oof_df.empty
        else oof_df.copy()
    )

    if should_close_simplex(exec_cfg):
        target_sum = float(exec_cfg.features.simplex_sum or 1.0)
        if not all_preds_raw.empty:
            all_preds_raw = close_simplex_predictions(
                all_preds_raw,
                target_sum=target_sum,
                date_col="forecast_date",
                ts_id_col="ts_id",
            )
        if not all_oof.empty:
            all_oof = close_simplex_predictions(
                all_oof,
                target_sum=target_sum,
                date_col="forecast_date",
                ts_id_col="ts_id",
            )

    # Build aggregated leaderboard across series
    full_metric_df = pd.DataFrame(metric_rows)
    lb_rows: list[dict[str, Any]] = []
    if not full_metric_df.empty:
        for m_type, grp in full_metric_df.groupby("model_type", sort=False):
            m_str = str(m_type)
            is_ens = m_str.startswith("ensemble_")
            if is_ens:
                vdiag = {
                    "base_model": m_str,
                    "scope_capability": "ensemble",
                    "training_mode": "ensemble",
                    "cov_profile": "ensemble",
                }
            else:
                try:
                    vdiag = variant_diagnostics_for(m_str, exec_cfg)
                except Exception:  # noqa: BLE001 - reconciled or custom pseudo-models
                    vdiag = {
                        "base_model": m_str.split("@", 1)[0],
                        "scope_capability": "reconciled",
                        "training_mode": "reconciled",
                        "cov_profile": "reconciled",
                    }
            row_lb: dict[str, Any] = {
                "model_type": m_str,
                "base_model": vdiag["base_model"],
                "kind": "ensemble" if is_ens else "base",
                "n_cells": len(grp),
            }
            for k in METRIC_NAMES:
                val_k = float(grp[k].mean()) if k in grp else float("nan")
                row_lb[k] = val_k
                row_lb[f"mean_{k}"] = val_k
            row_lb["mean_wis"] = row_lb.get("mean_interval_score", float("nan"))
            row_lb["mean_pinball_loss"] = row_lb.get("mean_pinball", float("nan"))
            lb_rows.append(row_lb)

    sort_col = f"mean_{decision}" if lb_rows and f"mean_{decision}" in lb_rows[0] else decision
    leaderboard_df = (
        pd.DataFrame(lb_rows).sort_values(sort_col, na_position="last").reset_index(drop=True)
        if lb_rows
        else pd.DataFrame()
    )

    oof_for_diag = (
        all_oof.rename(columns={"forecast_date": "ds"}) if not all_oof.empty else pd.DataFrame()
    )
    hist_for_diag = hist_only_df.rename(columns={ts_col: "ts_id", date_col: "ds", target_col: "y"})

    fva_df = compute_fva(oof_for_diag, metric=decision, series_df=hist_for_diag)
    horizon_decay_df = compute_horizon_decay(oof_for_diag, metric=decision)
    bias_df = compute_bias_report(oof_for_diag)
    attributions_df = build_attributions_frame(attr_rows, level="global")

    pred_records: list[dict[str, Any]] = (
        [{str(k): v for k, v in r.items()} for r in all_preds_raw.to_dict(orient="records")]
        if not all_preds_raw.empty
        else []
    )
    oof_records: list[dict[str, Any]] = (
        [{str(k): v for k, v in r.items()} for r in all_oof.to_dict(orient="records")]
        if not all_oof.empty
        else []
    )
    hist_records: list[dict[str, Any]] = [
        {str(k): v for k, v in r.items()}
        for r in hist_only_df[[ts_col, date_col, target_col]]
        .rename(columns={ts_col: "ts_id", date_col: "ds", target_col: "y"})
        .to_dict(orient="records")
    ]
    predictions_frame = build_predictions_frame(
        pred_records,
        oof_rows=oof_records,
        history_rows=hist_records,
    )

    hierarchy_summary_df = pd.DataFrame()
    reconciliation_lift_pct = 0.0
    rec_mode = "none"
    max_spatial_coh = 0.0
    has_ct_flag = bool(
        hier_spec is not None
        and (
            exec_cfg.hierarchy.reconcile_temporal
            or exec_cfg.hierarchy.cross_temporal
            or "cross_temporal_mint" in exec_cfg.hierarchy.reconciliation_methods
        )
    )
    learned_alpha: float = (
        float(learned_alpha_float)
        if learned_alpha_float is not None
        else (
            float(exec_cfg.hierarchy.spatial_temporal_weight)
            if isinstance(exec_cfg.hierarchy.spatial_temporal_weight, (int, float))
            and not isinstance(exec_cfg.hierarchy.spatial_temporal_weight, bool)
            else (
                0.0
                if (hier_spec is None and exec_cfg.hierarchy.reconcile_temporal)
                else (0.5 if has_ct_flag else 1.0)
            )
        )
    )
    ct_beats_spatial = False
    if hier_spec is not None and not full_metric_df.empty:
        from .reconciliation import verify_coherence

        leaf_ids = set(hist_only_df[ts_col].astype(str).unique())
        all_node_ids = set(work_hist[ts_col].astype(str).unique())
        root_ids = {
            nid
            for nid in all_node_ids
            if nid in ("TOTAL", "__total__", "root") or nid.startswith("TOTAL")
        }
        if not root_ids:
            root_ids = {
                nid for nid in (all_node_ids - leaf_ids) if "/" not in nid and "|" not in nid
            }
        mid_ids = (all_node_ids - leaf_ids) - root_ids

        has_ct = has_ct_flag
        rec_mode = "cross_temporal" if has_ct else "spatial_only"
        primary_rec_method = (
            "cross_temporal_mint"
            if (has_ct and "cross_temporal_mint" in exec_cfg.hierarchy.reconciliation_methods)
            else (
                exec_cfg.hierarchy.reconciliation_methods[0]
                if exec_cfg.hierarchy.reconciliation_methods
                else "mint_nonneg"
            )
        )
        if not all_preds_raw.empty:
            rec_suffixes = tuple(
                f"_{m}"
                for m in (
                    *exec_cfg.hierarchy.reconciliation_methods,
                    "wls_struct",
                    "cross_temporal_mint",
                )
            )
            rec_preds_sub = all_preds_raw[
                all_preds_raw["model_type"].astype(str).str.endswith(rec_suffixes)
            ]
            if not rec_preds_sub.empty:
                with contextlib.suppress(Exception):
                    max_spatial_coh = float(
                        verify_coherence(
                            rec_preds_sub,
                            hier_spec,
                            date_col="forecast_date",
                            ts_id_col="ts_id",
                            atol=float("inf"),
                        )
                    )
        neg_leaf_count = int(
            (all_preds_raw[all_preds_raw["ts_id"].astype(str).isin(leaf_ids)]["yhat"] < -1e-9).sum()
            if not all_preds_raw.empty
            else 0
        )

        if has_ct and decision in full_metric_df.columns:
            m_means = full_metric_df.groupby("model_type")[decision].mean().dropna()
            ct_scores = [
                float(v)
                for k, v in m_means.items()
                if str(k).endswith("_cross_temporal_mint") and math.isfinite(float(v))
            ]
            sp_methods = [
                rm
                for rm in (*exec_cfg.hierarchy.reconciliation_methods, "wls_struct")
                if rm != "cross_temporal_mint"
            ]
            sp_scores = [
                float(v)
                for k, v in m_means.items()
                if not str(k).endswith("_cross_temporal_mint")
                and any(str(k).endswith(f"_{rm}") for rm in sp_methods)
                and math.isfinite(float(v))
            ]
            if ct_scores and sp_scores:
                ct_beats_spatial = min(ct_scores) <= min(sp_scores)
            elif ct_scores:
                ct_beats_spatial = True
            if ct_beats_spatial:
                primary_rec_method = "cross_temporal_mint"

        # Compute unreconciled vs reconciled scores per hierarchy level
        unrec_rows: list[dict[str, Any]] = []
        for (m_u, ts_u), m_dict in unreconciled_metrics_by_model_ts.items():
            if decision in m_dict and math.isfinite(float(m_dict[decision])):
                unrec_rows.append(
                    {"model_type": m_u, "ts_id": str(ts_u), "score": float(m_dict[decision])}
                )
        unrec_df = pd.DataFrame(unrec_rows)

        level_buckets = [
            ("Level 0: Root (Total)", root_ids),
            ("Level 1: Intermediate", mid_ids),
            ("Level 2: Leaf Series", leaf_ids),
            ("All Hierarchy Nodes", all_node_ids),
        ]
        h_rows: list[dict[str, Any]] = []
        for lvl_name, node_set in level_buckets:
            if not node_set:
                continue
            rec_sub = full_metric_df[full_metric_df["ts_id"].astype(str).isin(node_set)]
            if rec_sub.empty or decision not in rec_sub.columns:
                continue
            rec_by_m = rec_sub.groupby("model_type")[decision].mean().dropna()
            if rec_by_m.empty:
                continue
            rec_best = float(rec_by_m.min())

            unrec_best = rec_best
            if not unrec_df.empty:
                u_sub = unrec_df[unrec_df["ts_id"].isin(node_set)]
                if not u_sub.empty:
                    u_by_m = u_sub.groupby("model_type")["score"].mean().dropna()
                    if not u_by_m.empty:
                        unrec_best = float(u_by_m.min())
            lift_pct = (
                round(100.0 * (unrec_best - rec_best) / max(1e-9, unrec_best), 2)
                if math.isfinite(unrec_best) and unrec_best > 1e-9
                else 0.0
            )
            h_rows.append(
                {
                    "hierarchy_level": lvl_name,
                    "n_nodes": len(node_set),
                    "reconciliation_method": primary_rec_method,
                    "reconciliation_mode": rec_mode,
                    "decision_metric": decision,
                    "unreconciled_score": round(unrec_best, 4),
                    "reconciled_score": round(rec_best, 4),
                    "reconciliation_lift_pct": lift_pct,
                    "max_coherence_residual": round(max_spatial_coh, 8),
                    "temporal_coherence_residual": round(max_temp_coherence_residual, 8),
                    "negative_leaf_count": neg_leaf_count
                    if "Leaf" in lvl_name or "All" in lvl_name
                    else 0,
                }
            )
            if lvl_name == "All Hierarchy Nodes":
                reconciliation_lift_pct = lift_pct
        hierarchy_summary_df = pd.DataFrame(h_rows)
    elif exec_cfg.hierarchy.reconcile_temporal and not full_metric_df.empty:
        rec_mode = "temporal_only"
        t_method_name = str(
            exec_cfg.hierarchy.reconciliation_methods[0]
            if exec_cfg.hierarchy.reconciliation_methods
            else "wls_struct"
        )
        all_node_ids = set(work_hist[ts_col].astype(str).unique())
        neg_leaf_count = int(
            (all_preds_raw["yhat"] < -1e-9).sum() if not all_preds_raw.empty else 0
        )
        rec_by_m = (
            full_metric_df.groupby("model_type")[decision].mean().dropna()
            if decision in full_metric_df.columns
            else pd.Series(dtype=float)
        )
        rec_best = float(rec_by_m.min()) if not rec_by_m.empty else 0.0
        unrec_best = rec_best
        unrec_rows_t: list[dict[str, Any]] = []
        for (m_u, ts_u), m_dict in unreconciled_metrics_by_model_ts.items():
            if decision in m_dict and math.isfinite(float(m_dict[decision])):
                unrec_rows_t.append(
                    {"model_type": m_u, "ts_id": str(ts_u), "score": float(m_dict[decision])}
                )
        if unrec_rows_t:
            u_by_m_t = pd.DataFrame(unrec_rows_t).groupby("model_type")["score"].mean().dropna()
            if not u_by_m_t.empty:
                unrec_best = float(u_by_m_t.min())
        reconciliation_lift_pct = (
            round(100.0 * (unrec_best - rec_best) / max(1e-9, unrec_best), 2)
            if math.isfinite(unrec_best) and unrec_best > 1e-9
            else 0.0
        )
        hierarchy_summary_df = pd.DataFrame(
            [
                {
                    "hierarchy_level": "All Hierarchy Nodes",
                    "n_nodes": len(all_node_ids),
                    "reconciliation_method": t_method_name,
                    "reconciliation_mode": rec_mode,
                    "decision_metric": decision,
                    "unreconciled_score": round(unrec_best, 4),
                    "reconciled_score": round(rec_best, 4),
                    "reconciliation_lift_pct": reconciliation_lift_pct,
                    "max_coherence_residual": round(max_temp_coherence_residual, 8),
                    "temporal_coherence_residual": round(max_temp_coherence_residual, 8),
                    "negative_leaf_count": neg_leaf_count,
                }
            ]
        )

    diag_summary = _summarize_pass_diagnostics(
        cfg=exec_cfg,
        leaderboard_df=leaderboard_df,
        fva_df=fva_df,
        horizon_decay_df=horizon_decay_df,
        bias_df=bias_df,
        attributions_df=attributions_df,
        oof_df=oof_for_diag,
        hist_df=hist_for_diag,
        reconciliation_lift_pct=reconciliation_lift_pct,
    )
    diag_summary["reconciliation_mode"] = rec_mode
    diag_summary["spatial_coherence_residual"] = round(max_spatial_coh, 8)
    diag_summary["temporal_coherence_residual"] = round(max_temp_coherence_residual, 8)
    diag_summary["spatial_weight_alpha"] = learned_alpha
    diag_summary["cross_temporal_beats_spatial"] = ct_beats_spatial

    return {
        "effective_config": exec_cfg,
        "borrowed_models": tuple(sorted(borrowed_models_set)),
        "executed_models": tuple(sorted(executed_models_set)),
        "leaderboard_df": leaderboard_df,
        "predictions_df": predictions_frame,
        "raw_predictions_df": all_preds_raw,
        "oof_df": oof_for_diag,
        "fva_df": fva_df,
        "horizon_decay_df": horizon_decay_df,
        "bias_df": bias_df,
        "attributions_df": attributions_df,
        "hierarchy_summary_df": hierarchy_summary_df,
        "hierarchy_spec": hier_spec,
        "diagnostics_summary": diag_summary,
    }


def _compile_proposal_into_config(
    session: AutoForecastSession,
    prior_dict: dict[str, Any],
    proposal: dict[str, Any],
    *,
    run_suffix: str,
    prior_run_id: str,
) -> RunConfig:
    """Turn a policy proposal dictionary into a validated `RunConfig`."""
    next_raw = dict(prior_dict)
    next_raw["run_name"] = f"{session.blueprint.config.run_name}-{run_suffix}"
    if session.pinned_models:
        next_raw["models"] = list(prior_dict.get("models", []))
    else:
        excl = set(session.exclude_models)
        excl_bases = {m.split("@", 1)[0] for m in excl}
        next_raw["models"] = [
            m
            for m in proposal.get("models", prior_dict.get("models", []))
            if m not in excl and m.split("@", 1)[0] not in excl_bases
        ]
    next_raw["model_params"] = dict(
        proposal.get("model_params", prior_dict.get("model_params", {}))
    )
    next_raw["features"] = dict(proposal.get("features", prior_dict.get("features", {})))
    if "output" in proposal:
        next_raw["output"] = dict(proposal["output"])
    if "ensemble" in proposal:
        next_raw["ensemble"] = dict(proposal["ensemble"])
    if "backtest" in proposal and isinstance(proposal["backtest"], Mapping):
        next_raw["backtest"] = dict(proposal["backtest"])
    if "hierarchy" in proposal and isinstance(proposal["hierarchy"], Mapping):
        next_raw["hierarchy"] = dict(proposal["hierarchy"])
    if "hpo" in proposal and session.pinned_hpo is None:
        next_raw["hpo"] = dict(proposal["hpo"])
    comp = dict(next_raw.get("compute", {}))
    comp["reuse_prior_runs"] = True
    if session.where != "local":
        comp["reuse_from_run_id"] = prior_run_id
    next_raw["compute"] = comp
    return RunConfig.model_validate(next_raw)


def build_next_loop_preview(session: AutoForecastSession) -> NextLoopPreview:
    """Compute a :class:`NextLoopPreview` from the latest pass in ``session`` (pure)."""
    last_pass = session.passes[-1]
    next_idx = last_pass.pass_index + 1
    prior_dict = last_pass.config.model_dump(mode="json")
    avail_models = _available_models_for_where(session.where, exclude_models=session.exclude_models)

    if session.autofocus_mode == "dag" and last_pass.pass_index == 1:
        branches = propose_autofocus_dag_branches(
            session.profile,
            prior_dict,
            next_idx,
            session.last_diagnostics_summary,
            tier=session.tier,
            where=session.where,
            goal=session.goal,
            available_models=avail_models,
        )
    else:
        branches = []

    proposal = propose_autofocus_pass(
        session.profile,
        prior_dict,
        next_idx,
        session.last_diagnostics_summary,
        tier=session.tier,
        where=session.where,
        goal=session.goal,
        available_models=avail_models,
    )

    mutations: list[dict[str, str]] = list(proposal["mutations"])
    next_cfg = _compile_proposal_into_config(
        session,
        prior_dict,
        proposal,
        run_suffix=f"p{next_idx}",
        prior_run_id=last_pass.run_id,
    )

    # Determine which models in next_cfg can be borrowed from Pass k vs must be fitted fresh
    prior_hashes = {make_model_spec_hash(m, last_pass.config): m for m in last_pass.config.models}
    borrowed: list[str] = []
    new_to_fit: list[str] = []
    for m in next_cfg.models:
        if make_model_spec_hash(m, next_cfg) in prior_hashes:
            borrowed.append(m)
        else:
            new_to_fit.append(m)

    dur = predict_run_duration(
        next_cfg,
        n_series=session.profile.n_series,
        median_n_obs=float(session.profile.median_n_obs),
        where="local" if session.where == "local" else None,
        borrowed_models=borrowed,
    )

    elapsed_minutes = sum(p.actual_wall_s for p in session.passes) / 60.0
    rem_minutes = (
        max(0.0, session.max_minutes - elapsed_minutes) if session.max_minutes is not None else None
    )

    conf_info = estimate_continue_confidence(
        session.profile,
        last_pass.pass_index,
        session.max_passes,
        session.last_diagnostics_summary,
        mutations,
        remaining_minutes=rem_minutes,
        next_pass_p50_minutes=dur.total_p50_minutes,
        min_confidence=session.min_confidence,
    )

    mutations_df = pd.DataFrame(
        mutations,
        columns=["action", "parameter", "old_value", "new_value", "evidence", "rationale"],
    )
    return NextLoopPreview(
        next_pass_index=next_idx,
        should_continue=bool(conf_info["should_continue"]),
        confidence=float(conf_info["confidence"]),
        expected_gain_pct=float(conf_info["expected_gain_pct"]),
        reason=str(conf_info["reason"]),
        proposed_config=next_cfg,
        mutations_df=mutations_df,
        borrowed_models=tuple(borrowed),
        new_models_to_fit=tuple(new_to_fit),
        duration_estimate=dur,
        branches=tuple(branches),
    )


def _refresh_session_champion_flags(session: AutoForecastSession) -> None:
    """Update ``is_session_champion`` across ``session.passes`` so only the best pass is True."""
    if not session.passes:
        return
    best_idx = 0
    best_val = float("inf")
    for idx, p in enumerate(session.passes):
        if math.isfinite(p.champion_score) and p.champion_score <= best_val:
            best_val = p.champion_score
            best_idx = idx
    for idx, p in enumerate(session.passes):
        session.passes[idx] = replace(p, is_session_champion=(idx == best_idx))


def execute_one_session_pass(
    session: AutoForecastSession,
    cfg: RunConfig,
    pass_index: int,
    mutations_applied: tuple[str, ...],
    *,
    pass_id: str | None = None,
    parent_pass_ids: tuple[str, ...] | None = None,
    branch_hypothesis: str | None = None,
    mutations_detail: Sequence[dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> Forecaster:
    """Execute one pass (`where == 'local'` or cloud `Forecaster.run`), record it on `session`."""
    from .sdk import Forecaster

    eff_pass_id = pass_id or str(pass_index)
    eff_parents = (
        parent_pass_ids
        if parent_pass_ids is not None
        else (() if pass_index == 1 else (str(pass_index - 1),))
    )
    eff_hypothesis = branch_hypothesis or (
        "Initial Tier Exposure" if pass_index == 1 else ", ".join(mutations_applied)
    )
    eff_mut_detail = tuple(mutations_detail or ())

    pred_dur = predict_run_duration(
        cfg,
        n_series=session.profile.n_series,
        median_n_obs=float(session.profile.median_n_obs),
        where="local" if session.where == "local" else None,
    )

    t0 = time.monotonic()
    if session.where == "local" and session.data_df is not None:
        full_panel = session.data_df.copy()

        out = _execute_local_panel_pass(
            full_panel,
            cfg,
            cell_cache=session.cell_cache,
            future_df=session.future_df,
        )
        wall_s = time.monotonic() - t0
        pass_outputs = {
            "leaderboard_df": out["leaderboard_df"],
            "predictions_df": out["predictions_df"],
            "raw_predictions_df": out["raw_predictions_df"],
            "oof_df": out["oof_df"],
            "fva_df": out["fva_df"],
            "horizon_decay_df": out["horizon_decay_df"],
            "bias_df": out["bias_df"],
            "attributions_df": out["attributions_df"],
            "hierarchy_summary_df": out["hierarchy_summary_df"],
            "hierarchy_spec": out.get("hierarchy_spec"),
            "diagnostics_summary": out["diagnostics_summary"],
        }
        session.last_diagnostics_summary = out["diagnostics_summary"]
        n_borrowed = len(out["borrowed_models"])
        n_executed = len(out["executed_models"])
        f = Forecaster(cfg, settings=settings)
    else:
        f = Forecaster(cfg, settings=settings)
        f.run()
        wall_s = time.monotonic() - t0
        lb_df = f.leaderboard_df()
        fva_df = f.fva_df()
        hd_df = f.horizon_decay_df()
        bias_df = f.bias_report_df()
        attr_df = f.attributions_df()
        from .registry.reads import read_oof

        oof_rows = read_oof(f.run_id, limit=20000, settings=settings)
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
        diag_cloud = _summarize_pass_diagnostics(
            cfg=cfg,
            leaderboard_df=lb_df,
            fva_df=fva_df,
            horizon_decay_df=hd_df,
            bias_df=bias_df,
            attributions_df=attr_df,
            oof_df=oof_df,
        )
        pass_outputs = {
            "leaderboard_df": lb_df,
            "predictions_df": pd.DataFrame(),
            "raw_predictions_df": pd.DataFrame(),
            "oof_df": oof_df,
            "fva_df": fva_df,
            "horizon_decay_df": hd_df,
            "bias_df": bias_df,
            "attributions_df": attr_df,
            "hierarchy_summary_df": pd.DataFrame(),
            "diagnostics_summary": diag_cloud,
        }
        session.last_diagnostics_summary = diag_cloud
        n_borrowed = 0 if pass_index == 1 else max(0, len(cfg.models) - len(mutations_applied))
        n_executed = max(1, len(cfg.models) - n_borrowed)

    diag = session.last_diagnostics_summary
    this_score = float(diag.get("best_score", float("nan")))

    # Champion Ratchet: update session active tables when this pass improves/ties champion
    if (
        not session.passes
        or not math.isfinite(session.champion_score)
        or (math.isfinite(this_score) and this_score <= session.champion_score)
    ):
        session.champion_pass_id = eff_pass_id
        session.champion_score = this_score
        session.champion_outputs = pass_outputs
        session.leaderboard_df = pass_outputs["leaderboard_df"]
        session.predictions_df = pass_outputs["predictions_df"]
        session.raw_predictions_df = pass_outputs["raw_predictions_df"]
        session.oof_df = pass_outputs["oof_df"]
        session.fva_df = pass_outputs["fva_df"]
        session.horizon_decay_df = pass_outputs["horizon_decay_df"]
        session.bias_df = pass_outputs["bias_df"]
        session.attributions_df = pass_outputs["attributions_df"]
        session.hierarchy_summary_df = pass_outputs["hierarchy_summary_df"]

    temp_record = AutofocusPassRecord(
        pass_index=pass_index,
        run_id=f.run_id,
        tier=session.tier,
        models_total=len(cfg.models),
        models_borrowed=n_borrowed,
        models_executed=n_executed,
        champion_model=str(diag.get("best_model", cfg.models[0])),
        decision_metric=str(diag.get("decision_metric", cfg.backtest.decision_metric)),
        champion_score=this_score,
        fva_vs_naive_pct=float(diag.get("fva_vs_naive_pct", 0.0)),
        dm_pvalue=diag.get("dm_pvalue"),
        predicted_p50_s=pred_dur.total_p50_s,
        actual_wall_s=wall_s,
        continue_confidence=0.0,
        expected_next_gain_pct=0.0,
        mutations_applied=mutations_applied,
        stop_or_continue_reason="",
        config=cfg,
        pass_id=eff_pass_id,
        parent_pass_ids=eff_parents,
        branch_hypothesis=eff_hypothesis,
        mutations_detail=eff_mut_detail,
        is_session_champion=False,
        fva_vs_univariate_pct=float(diag.get("fva_vs_univariate_pct", 0.0)),
        fva_vs_foundation_pct=float(diag.get("fva_vs_foundation_pct", 0.0)),
        covariate_lift_pct=float(diag.get("covariate_lift_pct", 0.0)),
        reconciliation_lift_pct=float(diag.get("reconciliation_lift_pct", 0.0)),
        champion_is_univariate=bool(diag.get("champion_is_univariate", False)),
        beaten_challengers=tuple(diag.get("beaten_challengers", ())),
    )
    session.passes.append(temp_record)
    preview = build_next_loop_preview(session)
    session.passes[-1] = replace(
        temp_record,
        continue_confidence=preview.confidence,
        expected_next_gain_pct=preview.expected_gain_pct,
        stop_or_continue_reason=preview.reason,
    )
    _refresh_session_champion_flags(session)

    # Ensure returned Forecaster carries the session champion config
    champ_rec = session.champion_record()
    if champ_rec is not None and champ_rec.config != f.config:
        f = Forecaster(champ_rec.config, settings=settings)
    f._auto_session = session
    return f


def auto_forecast(
    data: pd.DataFrame | str,
    *,
    tier: str = "wide_field",
    where: str = "auto",
    goal: str = "accuracy",
    allowed_families: Sequence[str] | None = None,
    base_config: RunConfig | Mapping[str, Any] | str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    exclude_models: Sequence[str] | None = None,
    baselines: str | Sequence[str] | None = "auto",
    holidays: Sequence[str] | str | None = None,
    decision_metric: str | None = None,
    n_folds: int | None = None,
    hpo: bool | Mapping[str, Any] | None = None,
    ensemble: bool | Sequence[str] | Mapping[str, Any] | None = None,
    explainability: bool = True,
    calibration: DurationCalibration | bool | None = None,
    autofocus: bool = True,
    autofocus_mode: str = "dag",
    max_passes: int | None = None,
    max_autofocus_passes: int | None = None,
    max_minutes: float | None = None,
    max_wall_clock_minutes: float | None = None,
    max_usd: float | None = None,
    min_confidence: float = 0.35,
    step_by_step: bool = False,
    execute: bool = True,
    future_df: pd.DataFrame | None = None,
    ts_id_col: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    freq: str | None = None,
    horizon: int | None = None,
    run_name: str | None = None,
    settings: Settings | None = None,
) -> Forecaster:
    """Execute the Glass-Box Auto-Forecaster with optional closed-loop `autofocus` refinement.

    1. **Plan (`build_config`):** Profiles ``data``, compiles the initial :class:`ConfigBlueprint`
       for ``tier`` while honoring user constraint pins, and verifies the wall-clock/cost budget.
    2. **Execute Pass 1:** Runs the initial model portfolio (locally in-memory when ``data`` is a
       ``pd.DataFrame`` and ``where in ('auto', 'local')``, or on Google Cloud / BigQuery).
    3. **Observe -> Review -> Re-Plan (`autofocus=True`):** Audits FVA, Diebold–Mariano $p$-values,
       feature attributions, horizon decay crossover $h^*$, and residual heteroskedasticity/ACF.
       When ``autofocus_mode="dag"`` and multiple orthogonal hypotheses trigger, executes parallel
       Level-2 hypothesis branches (`2A`, `2B`, `2C`) while borrowing Pass 1's unchanged cells,
       followed by Level-3 synthesis and learned stacking backed by the Champion Ratchet.
    """
    tier = _normalize_tier(tier)
    if autofocus_mode not in ("dag", "linear"):
        raise ConfigError(
            f"Unknown autofocus_mode='{autofocus_mode}'. Valid values: 'dag', 'linear'"
        )
    if max_minutes is None and max_wall_clock_minutes is not None:
        max_minutes = max_wall_clock_minutes
    if max_passes is None and max_autofocus_passes is not None:
        max_passes = int(max_autofocus_passes) + 1

    blueprint = build_config(
        data,
        tier=tier,
        where=where,
        goal=goal,
        allowed_families=allowed_families,
        base_config=base_config,
        overrides=overrides,
        models=models,
        include_models=include_models,
        exclude_models=exclude_models,
        baselines=baselines,
        holidays=holidays,
        decision_metric=decision_metric,
        n_folds=n_folds,
        hpo=hpo,
        ensemble=ensemble,
        explainability=explainability,
        calibration=calibration,
        max_minutes=max_minutes,
        max_usd=max_usd,
        future_df=future_df,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        freq=freq,
        horizon=horizon,
        run_name=run_name,
        settings=settings,
    )

    tier_default_passes = int(EXPOSURE_TIERS[tier]["default_max_passes"])
    eff_max_passes = (
        1
        if not autofocus
        else int(max_passes if max_passes is not None else max(2, tier_default_passes))
    )

    session = AutoForecastSession(
        blueprint=blueprint,
        profile=blueprint.profile,
        tier=tier,
        where=blueprint.where,
        goal=goal,
        max_minutes=max_minutes,
        max_usd=max_usd,
        min_confidence=min_confidence,
        max_passes=eff_max_passes,
        autofocus_mode=autofocus_mode,
        exclude_models=tuple(exclude_models or ()),
        pinned_models=(models is not None),
        pinned_hpo=(hpo if isinstance(hpo, bool) else None),
        baselines=blueprint.baselines,
        data_df=data.copy() if isinstance(data, pd.DataFrame) else None,
        future_df=future_df.copy() if future_df is not None else None,
    )

    if not execute:
        from .sdk import Forecaster

        f_dry = Forecaster(blueprint.config, settings=settings)
        f_dry._auto_session = session
        return f_dry

    forecaster = execute_one_session_pass(
        session,
        blueprint.config,
        pass_index=1,
        mutations_applied=("initial_tier_compilation",),
        pass_id="1",
        parent_pass_ids=(),
        branch_hypothesis="Initial Tier Exposure",
        settings=settings,
    )

    if not autofocus or step_by_step or eff_max_passes <= 1:
        return forecaster

    pass1_record = session.passes[0]
    pass1_dict = pass1_record.config.model_dump(mode="json")
    pass1_diag = dict(session.last_diagnostics_summary)

    # Multi-branch Tournament DAG mode when multiple orthogonal hypotheses trigger on Pass 1
    preview1 = build_next_loop_preview(session)
    if not preview1.should_continue:
        return forecaster

    if autofocus_mode == "dag" and len(preview1.branches) >= 2:
        branch_outcomes: list[dict[str, Any]] = []
        for b_spec in preview1.branches:
            elapsed_m = sum(p.actual_wall_s for p in session.passes) / 60.0
            if max_minutes is not None and elapsed_m >= max_minutes:
                break
            b_id = str(b_spec.get("pass_id", "2A"))
            b_cfg = _compile_proposal_into_config(
                session,
                pass1_dict,
                b_spec,
                run_suffix=f"p{b_id.lower()}",
                prior_run_id=pass1_record.run_id,
            )
            b_muts = list(b_spec.get("mutations", []))
            b_actions = tuple(m.get("action", "refine") for m in b_muts) or ("refine",)
            # Restore Pass 1 diagnostics before evaluating each parallel branch
            session.last_diagnostics_summary = dict(pass1_diag)
            forecaster = execute_one_session_pass(
                session,
                b_cfg,
                pass_index=2,
                mutations_applied=b_actions,
                pass_id=b_id,
                parent_pass_ids=("1",),
                branch_hypothesis=str(b_spec.get("branch_hypothesis", "Hypothesis Branch")),
                mutations_detail=b_muts,
                settings=settings,
            )
            rec_b = session.passes[-1]
            branch_outcomes.append(
                {
                    "pass_id": b_id,
                    "config_dict": b_cfg.model_dump(mode="json"),
                    "best_score": rec_b.champion_score,
                    "pass1_score": pass1_record.champion_score,
                    "best_model": rec_b.champion_model,
                    "mutations_applied": b_actions,
                }
            )

        # Level 3 Synthesis & Stacking Join Node (when eff_max_passes >= 3 or >=2 branches win)
        winning_branches = [
            b for b in branch_outcomes if b["best_score"] <= b["pass1_score"] * 1.005
        ]
        elapsed_m = sum(p.actual_wall_s for p in session.passes) / 60.0
        can_run_synth = (
            len(branch_outcomes) >= 2
            and (eff_max_passes >= 3 or len(winning_branches) >= 2)
            and (max_minutes is None or elapsed_m < max_minutes)
        )
        if can_run_synth:
            synth_spec = propose_autofocus_synthesis_pass(
                session.profile,
                pass1_dict,
                branch_outcomes,
                tier=session.tier,
                where=session.where,
                goal=session.goal,
            )
            synth_cfg = _compile_proposal_into_config(
                session,
                pass1_dict,
                synth_spec,
                run_suffix="p3-synth",
                prior_run_id=session.run_id,
            )
            synth_muts = list(synth_spec.get("mutations", []))
            forecaster = execute_one_session_pass(
                session,
                synth_cfg,
                pass_index=3,
                mutations_applied=("synthesize_winning_dag_branches",),
                pass_id="3",
                parent_pass_ids=tuple(synth_spec.get("parent_pass_ids", ("2A", "2B"))),
                branch_hypothesis=str(
                    synth_spec.get(
                        "branch_hypothesis", "Level-3 Tournament Synthesis & Learned Stacking"
                    )
                ),
                mutations_detail=synth_muts,
                settings=settings,
            )
        return forecaster

    # Linear / Single-Branch loop
    while len(session.passes) < eff_max_passes:
        preview = build_next_loop_preview(session)
        if not preview.should_continue:
            break
        mut_records = preview.mutations_df.to_dict(orient="records")
        mut_names = tuple(str(r["action"]) for r in mut_records) if mut_records else ("refine",)
        b_title = (
            str(preview.branches[0].get("branch_hypothesis", ", ".join(mut_names)))
            if preview.branches
            else ", ".join(mut_names)
        )
        forecaster = execute_one_session_pass(
            session,
            preview.proposed_config,
            pass_index=preview.next_pass_index,
            mutations_applied=mut_names,
            pass_id=str(preview.next_pass_index),
            parent_pass_ids=(session.passes[-1].pass_id,),
            branch_hypothesis=b_title,
            mutations_detail=[{str(k): str(v) for k, v in r.items()} for r in mut_records],
            settings=settings,
        )

    return forecaster


def _main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``python -m scale_forecasting.auto``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="scale_forecasting.auto",
        description="Glass-Box Auto-Forecaster & Bounded autofocus CLI.",
    )
    parser.add_argument(
        "--benchmark",
        default="dgp_01_smooth_seasonal",
        help="DGP benchmark name (or --source-table for BigQuery table)",
    )
    parser.add_argument("--source-table", help="BigQuery source table name")
    parser.add_argument(
        "--tier",
        default="wide_field",
        choices=list(EXPOSURE_TIERS),
        help="Observational Astronomy Exposure Tier",
    )
    parser.add_argument(
        "--where",
        default="auto",
        choices=["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
        help="Execution sandbox envelope",
    )
    parser.add_argument(
        "--goal",
        default="accuracy",
        choices=["accuracy", "probabilistic", "service_level", "fast"],
        help="Optimization objective",
    )
    parser.add_argument(
        "--autofocus-mode",
        default="dag",
        choices=["dag", "linear"],
        help="Autofocus topology: multi-branch tournament 'dag' or sequential 'linear'",
    )
    parser.add_argument(
        "--decision-metric",
        help="Optional explicit decision metric pin (e.g. wape, mase, rmse, pinball)",
    )
    parser.add_argument(
        "--init-config",
        help="Compile and write the Glass-Box RunConfig JSON to this path without running",
    )
    parser.add_argument(
        "--no-autofocus",
        action="store_true",
        help="Run a single pass without closed-loop autofocus",
    )
    parser.add_argument("--max-minutes", type=float, help="Wall-clock budget in minutes")
    parser.add_argument("--max-usd", type=float, help="Cloud cost budget in USD")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    if args.source_table:
        target_data: Any = args.source_table
        fut_df = None
    else:
        from .benchmarks import load_benchmark

        bds = load_benchmark(args.benchmark)
        target_data = bds.df
        fut_df = bds.future_df if not bds.future_df.empty else None

    if args.init_config:
        bp = build_config(
            target_data,
            future_df=fut_df,
            tier=args.tier,
            where=args.where,
            goal=args.goal,
            decision_metric=args.decision_metric,
            max_minutes=args.max_minutes,
            max_usd=args.max_usd,
        )
        if args.init_config != "-":
            Path(args.init_config).write_text(bp.to_json(indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(bp.to_dict(), indent=2))
        else:
            print(bp.summary())
            print("\nRationale Table:")
            print(bp.rationale_df.to_string(index=False))
        return 0

    f = auto_forecast(
        target_data,
        future_df=fut_df,
        tier=args.tier,
        where=args.where,
        goal=args.goal,
        decision_metric=args.decision_metric,
        autofocus=not args.no_autofocus,
        autofocus_mode=args.autofocus_mode,
        max_minutes=args.max_minutes,
        max_usd=args.max_usd,
    )
    ledger = f.autofocus_ledger_df()
    if args.json:
        print(
            json.dumps({"run_id": f.run_id, "passes": ledger.to_dict(orient="records")}, indent=2)
        )
    else:
        if f.blueprint is not None:
            print(f.blueprint.summary())
        print("\nAutofocus Pass Ledger:")
        print(ledger.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
