"""End-to-end evolutionary runner (`EvolveConfig`, `EvolveResult`, `run_evolution`).

Orchestrates the closed-loop evolutionary cycle on top of
:meth:`~scale_forecasting.sdk.Forecaster.auto`:

1. **Pre-Flight Entitlement Check:** When ``backend="alphaevolve"`` (the default), calls
   :func:`~scale_forecasting.evolve.client.probe_alphaevolve_access` before any backtests run so
   projects without Gemini Enterprise + AlphaEvolve entitlement fail fast with actionable guidance
   pointing to ``Forecaster.auto()`` and the ``backend="vertex_gemini"`` backup.
2. **Temporal Lockdown Split & ``Forecaster.auto()`` Baseline:** Withholds the last
   $H_{\\text{lockdown}}$ steps per series as the unseen Lockdown Fold and runs
   ``Forecaster.auto()`` on the Inner Evolution Panel ($t \\le T - H_{\\text{lockdown}}$).
3. **Iterative Mutation & Pre-Gate / Gate 1 / Gate 2 Filtering:** Generates up to
   ``max_evaluations`` candidate ``# EVOLVE-BLOCK`` programs via ``alphaevolve``,
   ``vertex_gemini``, or ``mock``, checking each through the AST security sandbox, the
   $+10^6$ temporal lookahead detector, Gate 1 (inner CV lift), and Gate 2 (Diebold–Mariano
   $p$-value + worst-decile FVA guard).
4. **Gate 3 (Held-Out Lockdown Fold Promotion Ratchet):** Evaluates the winning inner-CV
   challenger **once** on the unseen Lockdown Fold and promotes ``promoted = True`` only when
   ``lockdown_candidate_score < lockdown_baseline_score``.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import ConfigError
from .client import DEFAULT_GEMINI_MODEL, create_evolution_client, probe_alphaevolve_access
from .evaluator import CandidateEvaluation, EvolutionEvaluator
from .sandbox import compile_candidate_symbol, validate_candidate_ast, verify_no_future_leakage
from .targets import EVOLVE_TARGETS, get_target_seed, synthesize_evolved_features

__all__ = [
    "EvolveConfig",
    "EvolveResult",
    "run_evolution",
]

_BENCHMARK_ALIASES: dict[str, str] = {
    "dgp_06_promo_saturation_decoy": "dgp_04_true_vs_decoy_covs",
    "promo_saturation_decoy": "dgp_04_true_vs_decoy_covs",
    "smooth_seasonal": "dgp_01_smooth_seasonal",
}


@dataclass(frozen=True)
class EvolveConfig:
    """Configuration for an opt-in :func:`run_evolution` / :meth:`Forecaster.evolve` session."""

    target: str = "features"
    backend: str = "alphaevolve"
    tier: str = "deep_sky"
    where: str = "local"
    horizon: int | None = None
    max_evaluations: int = 10
    significance_alpha: float = 0.10
    worst_decile_floor: float = -0.02
    project_id: str | None = None
    location: str = "global"
    region: str = "us-central1"
    model_name: str = DEFAULT_GEMINI_MODEL
    export_path: str | None = None
    random_seed: int = 42
    decision_metric: str | None = None
    timeout_seconds: float = 20.0
    dry_run: bool = False
    mock_candidates: tuple[str, ...] | None = None
    candidate_programs: tuple[str, ...] | None = None
    lockdown_override_score: float | None = None
    from_runs: tuple[Any, ...] | None = None
    from_run_ids: tuple[str, ...] | None = None
    dataset_id: str = "scale_forecasting"

    def __post_init__(self) -> None:
        norm_target = str(self.target).strip().lower()
        if norm_target not in EVOLVE_TARGETS:
            valid_t = ", ".join(EVOLVE_TARGETS)
            raise ConfigError(f"Unknown evolve target '{self.target}'. Valid targets: {valid_t}")
        object.__setattr__(self, "target", norm_target)

        norm_backend = str(self.backend).strip().lower()
        if norm_backend not in ("alphaevolve", "vertex_gemini", "mock"):
            raise ConfigError(
                f"Unknown evolve backend '{self.backend}'. Valid backends: "
                "'alphaevolve', 'vertex_gemini', 'mock'."
            )
        object.__setattr__(self, "backend", norm_backend)

        if int(self.max_evaluations) < 1:
            raise ConfigError("max_evaluations must be >= 1.")
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))

        if self.mock_candidates is None and self.candidate_programs is not None:
            object.__setattr__(self, "mock_candidates", tuple(self.candidate_programs))
        elif self.mock_candidates is not None and not isinstance(self.mock_candidates, tuple):
            object.__setattr__(self, "mock_candidates", tuple(self.mock_candidates))

        if self.from_runs is not None and not isinstance(self.from_runs, tuple):
            object.__setattr__(self, "from_runs", tuple(self.from_runs))
        if self.from_run_ids is not None and not isinstance(self.from_run_ids, tuple):
            object.__setattr__(self, "from_run_ids", tuple(str(r) for r in self.from_run_ids))

    @classmethod
    def model_validate(cls, obj: EvolveConfig | Mapping[str, Any]) -> EvolveConfig:
        """Validate a dictionary or :class:`EvolveConfig` instance."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            valid_keys = {f.name for f in dataclasses.fields(cls)}
            filtered = {k: v for k, v in obj.items() if k in valid_keys}
            if (
                "candidates" in obj
                and "mock_candidates" not in filtered
                and obj["candidates"] is not None
            ):
                filtered["mock_candidates"] = tuple(obj["candidates"])
            return cls(**filtered)
        raise ConfigError(f"Cannot coerce {type(obj).__name__} into EvolveConfig.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-friendly dictionary."""
        raw = dataclasses.asdict(self)
        if raw.get("mock_candidates") is not None:
            raw["mock_candidates"] = list(raw["mock_candidates"])
        if raw.get("from_runs") is not None:
            raw["from_runs"] = list(raw["from_runs"])
        if raw.get("from_run_ids") is not None:
            raw["from_run_ids"] = list(raw["from_run_ids"])
        return raw

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Pydantic-compatible alias for :meth:`to_dict`."""
        return self.to_dict()


@dataclass
class EvolveResult:
    """Outcome of an evolutionary run governed by the 3-Gate Anti-Overfitting Ratchet."""

    target: str
    backend_used: str
    promoted: bool
    rejection_reason: str | None
    baseline_score: float
    inner_cv_score: float
    lockdown_baseline_score: float
    lockdown_candidate_score: float
    fva_pct: float
    dm_pvalue: float
    worst_decile_fva_pct: float
    champion_code: str
    export_path: str | None
    leaderboard_df: pd.DataFrame
    forecaster: Any = None
    assessment: Any = None
    evaluations_run: int = 0
    wall_seconds: float = 0.0
    candidate_code: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    predictions_df: pd.DataFrame | None = None
    model_name: str = DEFAULT_GEMINI_MODEL

    def emit_airflow_dag(
        self,
        output_path: str | Path | None = None,
        schedule: str = "@monthly",
        *,
        dag_id: str | None = None,
    ) -> str:
        """Render a standalone Cloud Composer 3 / Airflow DAG scheduling periodic evolution.

        If ``output_path`` is provided, writes the DAG Python source to ``output_path`` and returns
        the rendered DAG source code string.
        """
        eff_dag_id = dag_id or f"scale_forecasting_evolve_{self.target}"
        export_target = self.export_path or f"/tmp/evolved_{self.target}.py"
        dag_source = f'''"""Airflow DAG for scheduled scale_forecasting.evolve ({self.target})."""

from __future__ import annotations

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {{
    "owner": "scale-forecasting",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}}

with DAG(
    dag_id="{eff_dag_id}",
    default_args=DEFAULT_ARGS,
    description=(
        "Scheduled 3-Gate Anti-Overfitting evolution for "
        "target={self.target} (backend={self.backend_used}, model={self.model_name})"
    ),
    schedule="{schedule}",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["scale-forecasting", "evolve", "{self.target}", "{self.backend_used}"],
) as dag:
    evolve_and_verify = BashOperator(
        task_id="evolve_{self.target}_with_3gate_ratchet",
        bash_command=(
            "python -m scale_forecasting.evolve "
            "--target {self.target} "
            "--backend {self.backend_used} "
            "--model-name {self.model_name} "
            "--export-path {export_target}"
        ),
    )
'''
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(dag_source, encoding="utf-8")
        return dag_source

    def to_dict(self) -> dict[str, Any]:
        """Serialize result summary and leaderboard to a JSON-compatible dictionary."""
        return {
            "target": self.target,
            "backend_used": self.backend_used,
            "model_name": self.model_name,
            "promoted": self.promoted,
            "rejection_reason": self.rejection_reason,
            "baseline_score": (
                round(self.baseline_score, 6) if math.isfinite(self.baseline_score) else None
            ),
            "inner_cv_score": (
                round(self.inner_cv_score, 6) if math.isfinite(self.inner_cv_score) else None
            ),
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
            "fva_pct": round(self.fva_pct, 4) if math.isfinite(self.fva_pct) else 0.0,
            "dm_pvalue": round(self.dm_pvalue, 6) if math.isfinite(self.dm_pvalue) else None,
            "worst_decile_fva_pct": (
                round(self.worst_decile_fva_pct, 4)
                if math.isfinite(self.worst_decile_fva_pct)
                else None
            ),
            "evaluations_run": self.evaluations_run,
            "wall_seconds": round(self.wall_seconds, 3),
            "export_path": self.export_path,
            "champion_code": self.champion_code,
            "diagnostics": self.diagnostics,
            "leaderboard": self.leaderboard_df.to_dict(orient="records"),
        }

    def summary(self) -> str:
        """Return a concise human-readable summary of the 3-Gate Ratchet verdict."""
        verdict = (
            "PROMOTED (Passed Gates 1, 2 & 3)"
            if self.promoted
            else f"RATCHETED TO BASELINE ({self.rejection_reason})"
        )
        return (
            f"EvolveResult(target='{self.target}', backend='{self.backend_used}', "
            f"verdict='{verdict}', baseline={self.baseline_score:.4f}, "
            f"inner_cv={self.inner_cv_score:.4f}, "
            f"lockdown_base={self.lockdown_baseline_score:.4f}, "
            f"lockdown_cand={self.lockdown_candidate_score:.4f}, "
            f"fva_pct={self.fva_pct:+.2f}%, dm_pvalue={self.dm_pvalue:.4f}, "
            f"worst_decile_fva={self.worst_decile_fva_pct:+.4f}, "
            f"wall_s={self.wall_seconds:.2f}s)"
        )


def _resolve_input_dataframe(data: pd.DataFrame | str | None) -> pd.DataFrame:
    """Resolve ``data`` (DataFrame, DGP benchmark name, or ``None``) into a panel DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    from ..benchmarks import BENCHMARK_NAMES, load_benchmark

    if data is None:
        return load_benchmark("dgp_01_smooth_seasonal").df.copy()
    name = str(data).strip()
    mapped = _BENCHMARK_ALIASES.get(name, name)
    if mapped in BENCHMARK_NAMES:
        return load_benchmark(mapped).df.copy()
    for b_name in BENCHMARK_NAMES:
        if name in b_name or b_name.startswith(name):
            return load_benchmark(b_name).df.copy()
    raise ConfigError(
        f"Unrecognized benchmark or panel identifier '{data}'. Pass a pandas DataFrame or one of "
        f"the canonical DGP names ({', '.join(BENCHMARK_NAMES[:5])}, ...)."
    )


def run_evolution(
    data: pd.DataFrame | str | Any | None = None,
    config: EvolveConfig | Mapping[str, Any] | None = None,
    *,
    from_runs: Sequence[Any] | None = None,
    from_run_ids: Sequence[str] | None = None,
    dataset_id: str | None = None,
    target: str | None = None,
    backend: str | None = None,
    tier: str | None = None,
    where: str | None = None,
    horizon: int | None = None,
    max_evaluations: int | None = None,
    significance_alpha: float | None = None,
    worst_decile_floor: float | None = None,
    project_id: str | None = None,
    location: str | None = None,
    region: str | None = None,
    model_name: str | None = None,
    export_path: str | Path | None = None,
    random_seed: int | None = None,
    decision_metric: str | None = None,
    timeout_seconds: float | None = None,
    dry_run: bool | None = None,
    mock_candidates: Sequence[str] | None = None,
    candidates: Sequence[str] | None = None,
    lockdown_override_score: float | None = None,
    genai_client: Any = None,
    alphaevolve_http_request: Callable[..., dict[str, Any]] | None = None,
    http_get: Callable[..., Any] | None = None,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    freq: str | None = None,
) -> EvolveResult:
    """Run opt-in evolutionary code synthesis on top of ``Forecaster.auto()`` or prior runs."""
    t_start = time.monotonic()
    base_dict: dict[str, Any] = {}
    if config is not None:
        base_dict = EvolveConfig.model_validate(config).to_dict()

    # When evolving directly on prior runs or an executed Forecaster without an explicit target,
    # default to "ensemble" (post-run pool evolution without base-model refitting).
    has_post_run_inputs = (
        from_runs is not None
        or from_run_ids is not None
        or bool(base_dict.get("from_runs"))
        or bool(base_dict.get("from_run_ids"))
        or (data is not None and not isinstance(data, (pd.DataFrame, str)))
    )
    if (
        has_post_run_inputs
        and target is None
        and config is None
        and base_dict.get("target") is None
    ):
        target = "ensemble"

    eff_candidates = mock_candidates if mock_candidates is not None else candidates
    overrides: dict[str, Any] = {
        "target": target,
        "backend": backend,
        "tier": tier,
        "where": where,
        "horizon": horizon,
        "max_evaluations": max_evaluations,
        "significance_alpha": significance_alpha,
        "worst_decile_floor": worst_decile_floor,
        "project_id": project_id,
        "location": location,
        "region": region,
        "model_name": model_name,
        "export_path": str(export_path) if export_path is not None else None,
        "random_seed": random_seed,
        "decision_metric": decision_metric,
        "timeout_seconds": timeout_seconds,
        "dry_run": dry_run,
        "mock_candidates": tuple(eff_candidates) if eff_candidates is not None else None,
        "lockdown_override_score": lockdown_override_score,
        "from_runs": tuple(from_runs) if from_runs is not None else None,
        "from_run_ids": tuple(str(r) for r in from_run_ids) if from_run_ids is not None else None,
        "dataset_id": dataset_id,
    }
    for k, v in overrides.items():
        if v is not None:
            base_dict[k] = v

    cfg = EvolveConfig.model_validate(base_dict)
    spec = get_target_seed(cfg.target)

    # Fast offline dry-run path (validates AST + temporal causality in <0.1s)
    if cfg.dry_run:
        ast_check = validate_candidate_ast(spec.seed_code)
        sample_df = pd.DataFrame(
            {
                "ts_id": ["s0"] * 16,
                "ds": pd.date_range("2025-01-01", periods=16, freq="D"),
                "y": [float(i + (i % 4)) for i in range(16)],
                "promo": [float(i % 2) for i in range(16)],
            }
        )
        causal_check = verify_no_future_leakage(synthesize_evolved_features, sample_df, horizon=4)
        dry_lb = pd.DataFrame(
            [
                {
                    "candidate_id": f"seed_{cfg.target}_dry_run",
                    "generation": 0,
                    "target": cfg.target,
                    "backend": cfg.backend,
                    "ast_valid": ast_check.valid,
                    "causal_valid": causal_check,
                    "inner_cv_score": 0.10,
                    "baseline_score": 0.10,
                    "inner_fva_pct": 0.0,
                    "dm_pvalue": 1.0,
                    "worst_decile_fva_pct": 0.0,
                    "gate1_passed": False,
                    "gate2_passed": False,
                    "lockdown_baseline_score": 0.10,
                    "lockdown_candidate_score": 0.10,
                    "gate3_passed": False,
                    "promoted": False,
                    "status": "dry_run",
                    "rejection_reason": "dry_run: offline pre-flight validation only",
                    "wall_seconds": round(time.monotonic() - t_start, 4),
                }
            ]
        )
        if cfg.export_path:
            out_p = Path(cfg.export_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(spec.seed_code, encoding="utf-8")
        return EvolveResult(
            target=cfg.target,
            backend_used=cfg.backend,
            promoted=False,
            rejection_reason="dry_run: offline pre-flight validation only",
            baseline_score=0.10,
            inner_cv_score=0.10,
            lockdown_baseline_score=0.10,
            lockdown_candidate_score=0.10,
            fva_pct=0.0,
            dm_pvalue=1.0,
            worst_decile_fva_pct=0.0,
            champion_code=spec.seed_code,
            export_path=cfg.export_path,
            leaderboard_df=dry_lb,
            forecaster=None,
            assessment=None,
            evaluations_run=0,
            wall_seconds=time.monotonic() - t_start,
            candidate_code=spec.seed_code,
            diagnostics={
                "dry_run": True,
                "model_name": cfg.model_name,
                "ast_valid": ast_check.valid,
                "causal_valid": causal_check,
            },
            model_name=cfg.model_name,
        )

    # Pre-flight access probe for backend="alphaevolve" BEFORE running any backtests
    if cfg.backend == "alphaevolve" and alphaevolve_http_request is None:
        probe_alphaevolve_access(
            project_id=cfg.project_id,
            location=cfg.location,
            http_get=http_get,
        )

    is_post_run_call = (
        bool(cfg.from_runs)
        or bool(cfg.from_run_ids)
        or (data is not None and not isinstance(data, (pd.DataFrame, str)))
    )
    panel_df: Any = None if is_post_run_call else _resolve_input_dataframe(data)
    if data is not None and not isinstance(data, (pd.DataFrame, str)) and not cfg.from_runs:
        panel_df = data

    client = create_evolution_client(
        cfg.backend,
        target=cfg.target,
        project_id=cfg.project_id,
        location=cfg.location,
        region=cfg.region,
        model_name=cfg.model_name,
        max_evaluations=cfg.max_evaluations,
        mock_candidates=cfg.mock_candidates,
        genai_client=genai_client,
        alphaevolve_http_request=alphaevolve_http_request,
    )

    evaluator = EvolutionEvaluator(
        panel_df,
        from_runs=cfg.from_runs,
        from_run_ids=cfg.from_run_ids,
        dataset_id=cfg.dataset_id,
        project_id=cfg.project_id,
        region=cfg.region,
        target=cfg.target,
        backend=cfg.backend,
        tier=cfg.tier,
        where=cfg.where,
        horizon=cfg.horizon,
        significance_alpha=cfg.significance_alpha,
        worst_decile_floor=cfg.worst_decile_floor,
        timeout_seconds=cfg.timeout_seconds,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        freq=freq,
    )

    prof_dict = evaluator.profile_dict
    f_cfg = getattr(evaluator.forecaster, "config", None)
    f_feats = getattr(f_cfg, "features", None) if f_cfg is not None else None
    f_hier = getattr(f_cfg, "hierarchy", None) if f_cfg is not None else None
    mono_constraints: dict[str, Any] = (
        dict(getattr(f_feats, "monotonic_constraints", {}) or {}) if f_feats is not None else {}
    )
    if not mono_constraints and isinstance(prof_dict.get("monotonic_constraints"), Mapping):
        mono_constraints = dict(prof_dict["monotonic_constraints"])
    simplex_total = (
        prof_dict.get("simplex_total")
        if prof_dict.get("simplex_total") is not None
        else (getattr(f_feats, "simplex_sum", None) if f_feats is not None else None)
    )
    is_simplex = bool(prof_dict.get("is_compositional_simplex", False) or simplex_total is not None)
    hier_cols = list(
        prof_dict.get("hierarchy_id_cols")
        or (getattr(f_hier, "levels", []) if f_hier is not None else [])
    )
    rec_temporal = bool(
        prof_dict.get("reconcile_temporal", False)
        or (
            f_hier is not None
            and (
                bool(getattr(f_hier, "reconcile_temporal", False))
                or bool(getattr(f_hier, "cross_temporal", False))
            )
        )
    )
    f_cal = getattr(f_cfg, "calibration", None) if f_cfg is not None else None
    regime_shares_dict = dict(prof_dict.get("regime_shares") or {})
    cs_frac = float(
        prof_dict.get(
            "cold_start_fraction",
            regime_shares_dict.get("cold_start", 0.0),
        )
        or 0.0
    )
    has_cs = bool(
        prof_dict.get("has_cold_start", False)
        or cs_frac > 0.0
        or (getattr(f_feats, "cold_start_transfer", False) if f_feats is not None else False)
    )
    structural_diag: dict[str, Any] = {
        "monotonic_constraints": mono_constraints,
        "is_compositional_simplex": is_simplex,
        "simplex_total": simplex_total,
        "hierarchy_id_cols": hier_cols,
        "reconcile_temporal": rec_temporal,
        "has_cold_start": has_cs,
        "cold_start_fraction": cs_frac,
        "cold_start_donor_similarity": float(prof_dict.get("cold_start_donor_similarity", 0.0)),
        "calibration_method": str(
            prof_dict.get(
                "calibration_method",
                getattr(f_cal, "method", "conformal") if f_cal is not None else "conformal",
            )
        ),
        "residual_drift_ks_stat": float(prof_dict.get("residual_drift_ks_stat", 0.0)),
        "cross_series_residual_corr": float(
            prof_dict.get("cross_series_residual_corr", prof_dict.get("cross_series_corr", 0.0))
        ),
    }

    evaluations: list[CandidateEvaluation] = []
    current_parent_code = spec.seed_code
    last_diag: dict[str, Any] = {
        "fva_pct": 0.0,
        "dm_pvalue": 1.0,
        "worst_decile_fva": 0.0,
        "horizon_decay": "baseline",
        "baseline_score": evaluator.baseline_score,
        "inner_cv_score": evaluator.baseline_score,
        **structural_diag,
    }

    for gen in range(1, cfg.max_evaluations + 1):
        cand_code = client.propose_candidate(
            target=cfg.target,
            parent_code=current_parent_code,
            diagnostics=last_diag,
            generation=gen,
        )
        cand_eval = evaluator.evaluate_candidate_inner(cand_code, generation=gen)
        evaluations.append(cand_eval)

        cand_diag = {
            "fva_pct": cand_eval.inner_fva_pct,
            "dm_pvalue": cand_eval.dm_pvalue,
            "worst_decile_fva": cand_eval.worst_decile_fva_pct,
            "horizon_decay": "improved" if cand_eval.gate1_passed else "degraded",
            "baseline_score": cand_eval.baseline_score,
            "inner_cv_score": cand_eval.inner_cv_score,
            **structural_diag,
        }
        client.record_evaluation(
            program_id=cand_eval.candidate_id,
            code=cand_code,
            score=cand_eval.inner_cv_score,
            diagnostics=cand_diag,
        )
        if cand_eval.gate1_passed and math.isfinite(cand_eval.inner_cv_score):
            current_parent_code = cand_code
            last_diag = cand_diag

    # Select best inner-CV candidate (preferring candidates that passed both Gate 1 and Gate 2)
    gate12_winners = [e for e in evaluations if e.gate1_passed and e.gate2_passed]
    gate1_winners = [e for e in evaluations if e.gate1_passed]
    if gate12_winners:
        best_idx = min(
            range(len(evaluations)),
            key=lambda i: (
                evaluations[i].inner_cv_score
                if (evaluations[i].gate1_passed and evaluations[i].gate2_passed)
                else float("inf")
            ),
        )
    elif gate1_winners:
        best_idx = min(
            range(len(evaluations)),
            key=lambda i: (
                evaluations[i].inner_cv_score if evaluations[i].gate1_passed else float("inf")
            ),
        )
    else:
        best_idx = min(range(len(evaluations)), key=lambda i: evaluations[i].inner_cv_score)

    best_inner = evaluations[best_idx]
    final_eval = evaluator.evaluate_lockdown_gate3(best_inner)

    if cfg.lockdown_override_score is not None:
        forced_lock = float(cfg.lockdown_override_score)
        forced_g3 = bool(forced_lock < evaluator.lockdown_baseline_score - 1e-9)
        forced_promoted = bool(final_eval.gate1_passed and final_eval.gate2_passed and forced_g3)
        forced_reason = (
            None
            if forced_promoted
            else (
                f"Gate 3 (Held-Out Lockdown Fold): candidate improved inner CV "
                f"({final_eval.inner_cv_score:.4f} < {final_eval.baseline_score:.4f}) "
                f"but degraded on the unseen lockdown fold "
                f"({forced_lock:.4f} >= {evaluator.lockdown_baseline_score:.4f}). "
                f"Ratcheted back to Forecaster.auto() baseline."
            )
        )
        final_eval = CandidateEvaluation(
            **{
                **final_eval.__dict__,
                "lockdown_candidate_score": forced_lock,
                "gate3_passed": forced_g3,
                "promoted": forced_promoted,
                "status": "promoted_champion" if forced_promoted else "rejected_gate3",
                "rejection_reason": forced_reason,
            }
        )

    evaluations[best_idx] = final_eval
    leaderboard_df = pd.DataFrame([e.to_row() for e in evaluations])

    champion_code = final_eval.code if final_eval.promoted else spec.seed_code
    if cfg.export_path is not None:
        out_p = Path(cfg.export_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(champion_code, encoding="utf-8")

    assessment_df = (
        evaluator.session.practitioner_assessment_df()
        if evaluator.session is not None
        else pd.DataFrame()
    )
    fva_pct = final_eval.inner_fva_pct
    if (
        final_eval.promoted
        and math.isfinite(final_eval.lockdown_baseline_score)
        and final_eval.lockdown_baseline_score > 1e-9
    ):
        lock_fva = (
            100.0
            * (final_eval.lockdown_baseline_score - final_eval.lockdown_candidate_score)
            / final_eval.lockdown_baseline_score
        )
        fva_pct = max(fva_pct, lock_fva)

    materialized_preds_df: pd.DataFrame | None = None
    if cfg.target == "ensemble":
        try:
            if final_eval.promoted:
                champ_fn = compile_candidate_symbol(champion_code, spec.entrypoint)
                champ_weights = champ_fn(evaluator.oof_df.copy(), evaluator.profile_dict)
                materialized_preds_df = evaluator.materialize_ensemble_predictions(
                    champ_weights, model_label="ensemble_evolved"
                )
            else:
                reigning = evaluator.reigning_champion_model or "ensemble_mean"
                materialized_preds_df = evaluator.materialize_ensemble_predictions(
                    {reigning: 1.0}, model_label=reigning
                )
        except Exception:  # noqa: BLE001 - best-effort forward prediction materialization
            materialized_preds_df = None

    merged_diagnostics: dict[str, Any] = {
        **evaluator.pool_diagnostics,
        "model_name": cfg.model_name,
        "reigning_champion_model": evaluator.reigning_champion_model,
        "gate1_passed": final_eval.gate1_passed,
        "gate2_passed": final_eval.gate2_passed,
        "gate3_passed": final_eval.gate3_passed,
        "status": final_eval.status,
    }

    return EvolveResult(
        target=cfg.target,
        backend_used=cfg.backend,
        promoted=final_eval.promoted,
        rejection_reason=final_eval.rejection_reason,
        baseline_score=evaluator.baseline_score,
        inner_cv_score=final_eval.inner_cv_score,
        lockdown_baseline_score=evaluator.lockdown_baseline_score,
        lockdown_candidate_score=final_eval.lockdown_candidate_score,
        fva_pct=round(float(fva_pct), 4),
        dm_pvalue=round(float(final_eval.dm_pvalue), 6)
        if math.isfinite(final_eval.dm_pvalue)
        else 1.0,
        worst_decile_fva_pct=round(float(final_eval.worst_decile_fva_pct), 4),
        champion_code=champion_code,
        export_path=cfg.export_path,
        leaderboard_df=leaderboard_df,
        forecaster=evaluator.forecaster,
        assessment=assessment_df,
        evaluations_run=len(evaluations),
        wall_seconds=time.monotonic() - t_start,
        candidate_code=final_eval.code,
        diagnostics=merged_diagnostics,
        predictions_df=materialized_preds_df,
        model_name=cfg.model_name,
    )
