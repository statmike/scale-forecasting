"""Unit tests for Track H (`scale_forecasting.evolve`) evolutionary harness & integrations.

Covers:
1. AST security sandbox (`validate_candidate_ast`, `compile_candidate_symbol`,
   `execute_in_sandbox`).
2. Temporal causality / future-leakage detector (`verify_no_future_leakage`).
3. End-to-end evolution across all 4 `# EVOLVE-BLOCK` targets (`"features"`, `"ensemble"`,
   `"model"`, `"policy"`) using `backend="mock"` and `Forecaster.evolve(...)`.
4. The 3-Gate Anti-Overfitting Ratchet (verifying Gate 3 rejects a candidate that improves
   inner walk-forward CV but degrades on the held-out Lockdown Fold).
5. `probe_alphaevolve_access()` raising an actionable `ConfigurationError` mentioning Gemini
   Enterprise licensing and `backend="vertex_gemini"` backup on HTTP 403/404.
6. `backend="vertex_gemini"` mutation loop (with a mocked `genai` client response) and
   `EvolveResult.emit_airflow_dag()`.
7. CLI `python -m scale_forecasting.evolve --dry-run`, MCP `run_auto_forecast` with
   `evolve_target`, and Terraform opt-in `enable_alphaevolve = false` invariants.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scale_forecasting
from scale_forecasting import EvolveConfig, EvolveResult, Forecaster, RunConfig, run_evolution
from scale_forecasting.evolve import verify_oof_pool_compatibility
from scale_forecasting.evolve.__main__ import main as evolve_cli_main
from scale_forecasting.evolve.client import (
    AlphaEvolveClient,
    ConfigurationError,
    probe_alphaevolve_access,
)
from scale_forecasting.evolve.sandbox import (
    compile_candidate_symbol,
    execute_in_sandbox,
    validate_candidate_ast,
    verify_no_future_leakage,
)
from scale_forecasting.evolve.targets import (
    EVOLVE_TARGETS,
    deterministic_mock_candidates,
    extract_evolve_block,
    get_target_seed,
    inject_evolve_block,
)
from scale_forecasting.mcp import McpServer

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sample_panel_df() -> pd.DataFrame:
    """Build a compact, deterministic multi-series seasonal + exogenous panel for fast tests."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    for s_idx in range(4):
        ts_id = f"series_{s_idx}"
        base_level = 50.0 + 10.0 * s_idx
        slope = 0.35 + 0.05 * s_idx
        for t_idx, ds in enumerate(dates):
            promo = float((t_idx + s_idx) % 7 in (1, 5))
            price_idx = 1.0 + 0.1 * np.sin(2.0 * np.pi * t_idx / 14.0)
            seasonal = 6.0 * np.sin(2.0 * np.pi * t_idx / 7.0) + 2.5 * np.cos(
                2.0 * np.pi * t_idx / 7.0
            )
            noise = float(rng.normal(0.0, 0.35))
            y = base_level + slope * t_idx + seasonal + 4.5 * promo - 2.0 * price_idx + noise
            rows.append(
                {
                    "ts_id": ts_id,
                    "ds": ds,
                    "y": float(max(1.0, y)),
                    "promo": promo,
                    "price_idx": float(price_idx),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. AST Security Sandbox Tests
# ---------------------------------------------------------------------------


def test_ast_sandbox_accepts_safe_scientific_code_and_all_target_seeds() -> None:
    """Safe numpy/pandas code and all 4 canonical target seed programs pass AST validation."""
    safe_code = """
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def synthesize_evolved_features(df: pd.DataFrame, profile: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    step = np.arange(len(out), dtype=float)
    out["evolved_feat_sin7"] = np.sin(2.0 * math.pi * step / 7.0)
    return out
"""
    res = validate_candidate_ast(safe_code)
    assert res.valid is True
    assert res.is_valid is True
    assert res.violations == ()

    for target in EVOLVE_TARGETS:
        seed = get_target_seed(target)
        seed_res = validate_candidate_ast(seed.seed_code)
        assert seed_res.valid is True, f"Seed for {target} failed AST sandbox: {seed_res.reason}"
        symbol = compile_candidate_symbol(seed.seed_code, seed.entrypoint)
        assert symbol is not None


@pytest.mark.parametrize(
    ("unsafe_snippet", "expected_token"),
    [
        ("import os\ndef f(x):\n    return os.getcwd()\n", "os"),
        ("import subprocess\ndef f(x):\n    return subprocess.run(['ls'])\n", "subprocess"),
        ("from socket import socket\ndef f(x):\n    return x\n", "socket"),
        ("def f(x):\n    return open('/etc/passwd').read()\n", "open"),
        ("def f(x):\n    return eval('1 + 1')\n", "eval"),
        ("def f(x):\n    return __import__('os').system('echo hi')\n", "__import__"),
        ("def f(x):\n    return x.__subclasses__()\n", "__subclasses__"),
    ],
)
def test_ast_sandbox_rejects_dangerous_imports_and_builtins(
    unsafe_snippet: str,
    expected_token: str,
) -> None:
    """Forbidden imports (`os`, `subprocess`), `open()`, `eval()`, and `__import__()` fail."""
    res = validate_candidate_ast(unsafe_snippet)
    assert res.valid is False
    assert res.is_valid is False
    assert expected_token in res.reason
    with pytest.raises(ConfigurationError, match="Candidate failed AST security sandbox"):
        execute_in_sandbox(unsafe_snippet, "f", 1)


# ---------------------------------------------------------------------------
# 2. Temporal Causality / Future-Leakage Detector Tests
# ---------------------------------------------------------------------------


def test_causality_detector_accepts_causal_features_and_rejects_future_leakage(
    sample_panel_df: pd.DataFrame,
) -> None:
    """`verify_no_future_leakage` accepts causal lag/rolling features and rejects lookahead."""

    def causal_features(df: pd.DataFrame, profile: dict[str, Any] | None = None) -> pd.DataFrame:
        _ = profile
        out = df.copy()
        y_num = pd.to_numeric(out["y"], errors="coerce").fillna(0.0)
        out["evolved_feat_lag1"] = y_num.groupby(out["ts_id"], sort=False).shift(1).fillna(y_num)
        out["evolved_feat_roll3"] = (
            y_num.groupby(out["ts_id"], sort=False)
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .fillna(0.0)
        )
        return out

    def leaky_shift_neg1(df: pd.DataFrame, profile: dict[str, Any] | None = None) -> pd.DataFrame:
        _ = profile
        out = df.copy()
        # Illegal future lookahead: .shift(-1) peeks at tomorrow's target y_{t+1}
        out["evolved_feat_leak"] = out.groupby("ts_id", sort=False)["y"].shift(-1).fillna(0.0)
        return out

    def leaky_global_future_mean(
        df: pd.DataFrame, profile: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        _ = profile
        out = df.copy()
        # Illegal full-window transform: includes future horizon y[T-H:] in historical mean
        out["evolved_feat_global_mean"] = out.groupby("ts_id", sort=False)["y"].transform("mean")
        return out

    assert verify_no_future_leakage(causal_features, sample_panel_df, horizon=5) is True
    assert verify_no_future_leakage(leaky_shift_neg1, sample_panel_df, horizon=5) is False
    assert verify_no_future_leakage(leaky_global_future_mean, sample_panel_df, horizon=5) is False


def test_evolve_rejects_future_leaky_candidate_at_pregate(
    sample_panel_df: pd.DataFrame,
) -> None:
    """A candidate that leaks future `y` via `.shift(-1)` is rejected before Gate 1."""
    seed = get_target_seed("features")
    leaky_code = inject_evolve_block(
        seed.seed_code,
        'out["evolved_feat_future_y"] = out.groupby(id_col, sort=False)["y"].shift(-1).fillna(0.0)',
    )
    res = Forecaster.evolve(
        sample_panel_df,
        target="features",
        backend="mock",
        max_evaluations=1,
        horizon=4,
        mock_candidates=[leaky_code],
    )
    assert res.promoted is False
    assert len(res.leaderboard_df) == 1
    row = res.leaderboard_df.iloc[0]
    assert bool(row["ast_valid"]) is True
    assert bool(row["causal_valid"]) is False
    assert row["status"] == "rejected_leakage"
    assert res.rejection_reason is not None
    assert "Causality" in res.rejection_reason or "verify_no_future_leakage" in res.rejection_reason


# ---------------------------------------------------------------------------
# 3. All 4 Targets End-to-End via `backend="mock"` and `Forecaster.evolve(...)`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", EVOLVE_TARGETS)
def test_forecaster_evolve_all_four_targets_end_to_end_mock(
    sample_panel_df: pd.DataFrame,
    target: str,
    tmp_path: Path,
) -> None:
    """All 4 targets (`features`, `ensemble`, `model`, `policy`) evolve and promote on `mock`."""
    out_mod = tmp_path / f"champion_{target}.py"
    res = Forecaster.evolve(
        sample_panel_df,
        target=target,
        backend="mock",
        max_evaluations=2,
        horizon=4,
        export_path=str(out_mod),
    )
    assert isinstance(res, EvolveResult)
    assert res.target == target
    assert res.backend_used == "mock"
    assert res.promoted is True
    assert res.diagnostics["gate1_passed"] is True
    assert res.diagnostics["gate2_passed"] is True
    assert res.diagnostics["gate3_passed"] is True
    assert res.inner_cv_score < res.baseline_score
    assert res.lockdown_candidate_score < res.lockdown_baseline_score
    assert not res.leaderboard_df.empty
    assert res.forecaster is not None

    # Verify summary, dict serialization, and module export
    summary_text = res.summary()
    assert "PROMOTED" in summary_text
    payload = res.to_dict()
    assert payload["promoted"] is True
    assert payload["target"] == target

    assert out_mod.exists()
    ast.parse(out_mod.read_text(encoding="utf-8"))


def test_top_level_package_exports_evolve_symbols() -> None:
    """`scale_forecasting` exports `EvolveConfig`, `EvolveResult`, and `run_evolution`."""
    assert hasattr(scale_forecasting, "EvolveConfig")
    assert hasattr(scale_forecasting, "EvolveResult")
    assert hasattr(scale_forecasting, "run_evolution")
    assert "EvolveConfig" in scale_forecasting.__all__
    assert "EvolveResult" in scale_forecasting.__all__
    assert "run_evolution" in scale_forecasting.__all__


# ---------------------------------------------------------------------------
# 4. The 3-Gate Anti-Overfitting Ratchet (Gate 3 Held-Out Lockdown Rejection)
# ---------------------------------------------------------------------------


def test_three_gate_ratchet_rejects_candidate_that_degrades_lockdown_fold(
    sample_panel_df: pd.DataFrame,
) -> None:
    """A candidate that improves Inner CV (Gates 1 & 2) but degrades Lockdown fails Gate 3."""
    seed = get_target_seed("features")
    # Construct a strictly causal candidate that uses harmonic + exogenous lag features on
    # normal dynamic history (improving inner CV) but injects a spike when y is forward-filled
    # flat during multi-step lockdown horizon evaluation.
    overfit_block = """
step_idx = out.groupby(id_col, sort=False).cumcount().to_numpy(dtype=float)
y_num = pd.to_numeric(out["y"], errors="coerce").fillna(0.0)
y_lag1 = y_num.groupby(out[id_col], sort=False).shift(1).fillna(y_num)
is_flat_tail = (np.abs(y_num - y_lag1) < 1e-11) & (step_idx > 4.0)
out["evolved_feat_trend_idx"] = np.where(is_flat_tail, -500.0 * step_idx, np.log1p(step_idx))
out["evolved_feat_sin_w"] = np.sin(2.0 * np.pi * step_idx / 7.0)
out["evolved_feat_cos_w"] = np.cos(2.0 * np.pi * step_idx / 7.0)
for col in ("promo", "price_idx"):
    if col in out.columns:
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        lag_v = vals.groupby(out[id_col], sort=False).shift(1).fillna(vals)
        out[f"evolved_feat_{col}_lag1"] = lag_v
"""
    overfit_code = inject_evolve_block(seed.seed_code, overfit_block)

    res = Forecaster.evolve(
        sample_panel_df,
        target="features",
        backend="mock",
        max_evaluations=1,
        horizon=4,
        mock_candidates=[overfit_code],
    )
    assert res.diagnostics["gate1_passed"] is True
    assert res.diagnostics["gate2_passed"] is True
    assert res.inner_cv_score < res.baseline_score
    assert res.diagnostics["gate3_passed"] is False
    assert res.promoted is False
    assert res.rejection_reason is not None
    assert "Gate 3" in res.rejection_reason
    assert "lockdown fold" in res.rejection_reason.lower()
    # Champion code must ratchet back to the baseline seed code
    assert res.champion_code == seed.seed_code


# ---------------------------------------------------------------------------
# 5. AlphaEvolve Entitlement Probe (`probe_alphaevolve_access`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [403, 404])
def test_probe_alphaevolve_access_raises_actionable_error_on_403_404(
    status_code: int,
) -> None:
    """HTTP 403/404 from Discovery Engine raises `ConfigurationError` naming Gemini Enterprise."""

    def fake_http_get(url: str, timeout: float = 10.0) -> dict[str, Any]:
        _ = (url, timeout)
        return {"status": status_code, "error": "AlphaEvolve API not entitled"}

    with pytest.raises(ConfigurationError) as exc_info:
        probe_alphaevolve_access(
            project_id="demo-project",
            location="global",
            http_get=fake_http_get,
        )

    msg = str(exc_info.value)
    assert f"HTTP {status_code}" in msg
    assert "Gemini Enterprise" in msg
    assert "discoveryengine.googleapis.com" in msg
    assert "backend='vertex_gemini'" in msg


def test_alphaevolve_client_proposes_candidate_with_mock_http_transport() -> None:
    """`AlphaEvolveClient` parses mutated `# EVOLVE-BLOCK` code when entitled."""
    cands = deterministic_mock_candidates("ensemble", count=1)

    def fake_http_request(
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _ = payload
        if url.endswith(":acquirePrograms"):
            return {"programs": [{"programId": "p1", "code": cands[0]}]}
        return {
            "name": "projects/demo-project/locations/global/alphaEvolveExperiments/exp-ensemble"
        }

    client = AlphaEvolveClient(
        project_id="demo-project",
        location="global",
        http_request=fake_http_request,
    )
    seed = get_target_seed("ensemble")
    out = client.propose_candidate(
        target="ensemble",
        parent_code=seed.seed_code,
        diagnostics={"fva_pct": 1.5, "dm_pvalue": 0.04},
        generation=1,
    )
    assert extract_evolve_block(out) == extract_evolve_block(cands[0])


# ---------------------------------------------------------------------------
# 6. `backend="vertex_gemini"` Mutation Loop & `emit_airflow_dag()`
# ---------------------------------------------------------------------------


def test_vertex_gemini_backend_mutation_loop_and_emit_airflow_dag(
    sample_panel_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """`backend='vertex_gemini'` mutates `# EVOLVE-BLOCK` via mocked `genai` and emits DAG."""
    mock_cand_code = deterministic_mock_candidates("ensemble", count=1)[0]

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = f"```python\n{text}\n```"

    class _FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: dict[str, Any] | None = None,
        ) -> _FakeResponse:
            _ = (model, contents, config)
            self.calls += 1
            return _FakeResponse(mock_cand_code)

    class _FakeGenAIClient:
        def __init__(self) -> None:
            self.models = _FakeModels()

    fake_genai = _FakeGenAIClient()
    res = Forecaster.evolve(
        sample_panel_df,
        target="ensemble",
        backend="vertex_gemini",
        max_evaluations=2,
        horizon=4,
        genai_client=fake_genai,
    )
    assert fake_genai.models.calls == 2
    assert res.backend_used == "vertex_gemini"
    assert res.promoted is True
    assert res.diagnostics["gate3_passed"] is True

    dag_path = tmp_path / "composer_evolve_dag.py"
    dag_src = res.emit_airflow_dag(dag_path, schedule="@monthly", dag_id="test_evolve_dag")
    assert dag_path.exists()
    ast.parse(dag_src)
    assert "@monthly" in dag_src
    assert "test_evolve_dag" in dag_src
    assert "scale_forecasting.evolve" in dag_src


# ---------------------------------------------------------------------------
# 7. CLI `--dry-run`, MCP `run_auto_forecast` Integration & Terraform Opt-In
# ---------------------------------------------------------------------------


def test_cli_dry_run_validates_target_offline(capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m scale_forecasting.evolve --dry-run --json` validates AST & causality offline."""
    rc = evolve_cli_main(
        [
            "--target",
            "features",
            "--benchmark",
            "dgp_01_smooth_seasonal",
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["target"] == "features"
    assert len(payload["leaderboard"]) == 1
    assert payload["leaderboard"][0]["ast_valid"] is True
    assert payload["leaderboard"][0]["causal_valid"] is True
    assert payload["leaderboard"][0]["status"] == "dry_run"


def test_mcp_run_auto_forecast_supports_evolve_target_while_keeping_11_tools(
    sample_panel_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """MCP `run_auto_forecast` dispatches to `Forecaster.evolve` when `evolve_target` is set."""
    server = McpServer(repo_root=REPO_ROOT)
    tools = server.list_tools()
    assert len(tools) == 11
    auto_tool = next(t for t in tools if t["name"] == "run_auto_forecast")
    props = auto_tool["inputSchema"]["properties"]
    assert "evolve_target" in props
    assert "evolve_backend" in props
    assert "evolve_max_evaluations" in props
    assert "evolve_model_name" in props
    assert props["evolve_model_name"]["default"] == "gemini-3.8-flash"
    assert "evolve_from_run_ids" in props

    csv_path = tmp_path / "panel.csv"
    sample_panel_df.to_csv(csv_path, index=False)

    mcp_envelope = server.call_tool(
        "run_auto_forecast",
        {
            "source": str(csv_path),
            "horizon": 4,
            "evolve_target": "ensemble",
            "evolve_backend": "mock",
            "evolve_max_evaluations": 2,
        },
    )
    assert mcp_envelope["isError"] is False
    mcp_res = json.loads(mcp_envelope["content"][0]["text"])
    assert mcp_res["target"] == "ensemble"
    assert mcp_res["backend_used"] == "mock"
    assert mcp_res["promoted"] is True


def test_terraform_opt_in_enable_alphaevolve_defaults_false_and_binds_runner_only() -> None:
    """Terraform opt-in `enable_alphaevolve` defaults to `false` and binds runner SA only."""
    vars_tf = (REPO_ROOT / "terraform/main/variables.tf").read_text(encoding="utf-8")
    main_tf = (REPO_ROOT / "terraform/main/main.tf").read_text(encoding="utf-8")
    apis_tf = (REPO_ROOT / "terraform/main/modules/apis/main.tf").read_text(encoding="utf-8")
    iam_tf = (REPO_ROOT / "terraform/main/modules/iam/main.tf").read_text(encoding="utf-8")

    assert 'variable "enable_alphaevolve"' in vars_tf
    assert "default     = false" in vars_tf
    assert "enable_alphaevolve = var.enable_alphaevolve" in main_tf
    assert "discoveryengine.googleapis.com" in apis_tf
    assert "roles/discoveryengine.admin" in iam_tf
    # Ensure compute_roles never includes discoveryengine.admin
    compute_section = iam_tf.split("compute_roles", maxsplit=1)[1].split("resource", maxsplit=1)[0]
    assert "discoveryengine" not in compute_section
    assert isinstance(EvolveConfig(), EvolveConfig)
    assert EvolveConfig().model_name == "gemini-3.8-flash"
    assert callable(run_evolution)


# ---------------------------------------------------------------------------
# 8. Post-Run (`f.evolve()`) & Multi-Run Pool (`from_runs`) Evolution Tests
# ---------------------------------------------------------------------------


def _build_heterogeneous_oof_and_preds(
    *,
    models_spec: dict[str, tuple[float, float]],
    include_global_inverse_error: bool = False,
    series_prefix: str = "series",
    date_start: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct synthetic OOF and forward prediction frames with series-specialist models.

    ``models_spec`` maps ``model_type -> (err_group_a, err_group_b)`` where group A is
    ``series_0..series_2`` and group B is ``series_3..series_5``.
    """
    oof_dates = pd.date_range(date_start, periods=14, freq="D")
    fut_dates = pd.date_range(oof_dates[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    oof_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for s_idx in range(6):
        ts_id = f"{series_prefix}_{s_idx}"
        is_group_a = s_idx < 3
        for t_idx, ds in enumerate(oof_dates):
            y_true = 100.0 + 10.0 * s_idx + 2.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
            sign = 1.0 if (t_idx % 2 == 0) else -1.0
            model_yhats: dict[str, float] = {}
            for m_name, (err_a, err_b) in models_spec.items():
                rel_err = err_a if is_group_a else err_b
                yhat = y_true * (1.0 + sign * rel_err)
                model_yhats[m_name] = yhat
                oof_rows.append(
                    {
                        "ts_id": ts_id,
                        "model_type": m_name,
                        "ds": ds,
                        "y_true": float(y_true),
                        "yhat": float(yhat),
                    }
                )
            if include_global_inverse_error and model_yhats:
                # Global inverse-error blend: ~10% error across all series (better than any single
                # global model's 14.75% average, but worse than per-series specialist 1.5% error)
                blend_val = float(y_true * (1.0 + sign * 0.10))
                oof_rows.append(
                    {
                        "ts_id": ts_id,
                        "model_type": "ensemble_inverse_error",
                        "ds": ds,
                        "y_true": float(y_true),
                        "yhat": blend_val,
                    }
                )

        for h_idx, f_ds in enumerate(fut_dates):
            base_fut = 100.0 + 10.0 * s_idx + 2.0 * np.sin(2.0 * np.pi * h_idx / 7.0)
            fut_yhats: list[float] = []
            for m_name, (err_a, err_b) in models_spec.items():
                rel_err = err_a if is_group_a else err_b
                yhat = base_fut * (1.0 + rel_err)
                fut_yhats.append(yhat)
                pred_rows.append(
                    {
                        "ts_id": ts_id,
                        "model_type": m_name,
                        "forecast_date": f_ds,
                        "yhat": float(yhat),
                        "yhat_lower": float(yhat - 4.0),
                        "yhat_upper": float(yhat + 4.0),
                    }
                )
            if include_global_inverse_error and fut_yhats:
                b_fut = float(np.mean(fut_yhats))
                pred_rows.append(
                    {
                        "ts_id": ts_id,
                        "model_type": "ensemble_inverse_error",
                        "forecast_date": f_ds,
                        "yhat": b_fut,
                        "yhat_lower": b_fut - 4.0,
                        "yhat_upper": b_fut + 4.0,
                    }
                )

    return pd.DataFrame(oof_rows), pd.DataFrame(pred_rows)


def _make_manual_forecaster(
    run_name: str,
    models: list[str],
    oof_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> Forecaster:
    cfg = RunConfig.model_validate(
        {
            "run_name": run_name,
            "data": {"source_table": "source_series_native", "horizon": 7, "freq": "D"},
            "models": models,
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
        }
    )
    return Forecaster(cfg, oof_df=oof_df, predictions_df=predictions_df)


def test_manual_forecaster_instance_evolve_populates_predictions_and_reigning_champion() -> None:
    """`f.evolve(target='ensemble', backend='mock')` works on a manual `Forecaster`."""
    oof_df, pred_df = _build_heterogeneous_oof_and_preds(
        models_spec={
            "theta": (0.015, 0.28),
            "holtwinters": (0.28, 0.015),
            "naive_seasonal": (0.22, 0.22),
        },
        include_global_inverse_error=True,
    )
    f = _make_manual_forecaster(
        "manual-single-run",
        ["theta", "holtwinters", "naive_seasonal"],
        oof_df,
        pred_df,
    )

    res = f.evolve(target="ensemble", backend="mock", max_evaluations=2)
    assert isinstance(res, EvolveResult)
    assert res.promoted is True
    assert res.diagnostics.get("mode") == "post_run_pool"
    assert res.diagnostics.get("reigning_champion_model") == "ensemble_inverse_error"
    assert res.inner_cv_score < res.baseline_score
    assert res.lockdown_candidate_score < res.lockdown_baseline_score

    # Verify blended forward predictions_df with calibrated conformal intervals
    assert res.predictions_df is not None
    assert not res.predictions_df.empty
    for col in ("ts_id", "forecast_date", "yhat", "yhat_lower", "yhat_upper"):
        assert col in res.predictions_df.columns
    assert (res.predictions_df["yhat_upper"] >= res.predictions_df["yhat"]).all()
    assert (res.predictions_df["yhat"] >= res.predictions_df["yhat_lower"]).all()


def test_multi_run_pooling_disambiguates_models_and_rejects_disjoint_panels() -> None:
    """`Forecaster.evolve(from_runs=[f1, f2])` disambiguates `<model>@<run>` and checks overlap."""
    oof_1, pred_1 = _build_heterogeneous_oof_and_preds(
        models_spec={
            "naive_seasonal": (0.25, 0.25),
            "lightgbm": (0.012, 0.30),
        },
    )
    oof_2, pred_2 = _build_heterogeneous_oof_and_preds(
        models_spec={
            "naive_seasonal": (0.25, 0.25),  # identical OOF -> deduplicated
            "lightgbm": (0.30, 0.012),  # distinct HPO/config OOF -> disambiguated as lightgbm@<run>
        },
    )
    f1 = _make_manual_forecaster("pool-run-alpha", ["naive_seasonal", "lightgbm"], oof_1, pred_1)
    f2 = _make_manual_forecaster("pool-run-beta", ["naive_seasonal", "lightgbm"], oof_2, pred_2)

    pool_info = verify_oof_pool_compatibility([f1, f2])
    pool_diag = pool_info["pool_diagnostics"]
    assert "naive_seasonal" in pool_diag["deduplicated_models"]
    disambig = pool_diag["disambiguated_models"]
    assert any(str(m).startswith("lightgbm@") for m in disambig)
    assert len([m for m in pool_diag["models_pooled"] if str(m).startswith("lightgbm@")]) == 2

    res = Forecaster.evolve(
        from_runs=[f1, f2],
        target="ensemble",
        backend="mock",
        max_evaluations=2,
    )
    assert res.promoted is True
    assert res.diagnostics.get("mode") == "post_run_pool"
    assert len(res.diagnostics.get("source_run_ids", [])) == 2
    assert res.predictions_df is not None and not res.predictions_df.empty

    # Disjoint panel must raise ConfigError / ConfigurationError
    oof_disjoint, pred_disjoint = _build_heterogeneous_oof_and_preds(
        models_spec={"naive_seasonal": (0.10, 0.10)},
        series_prefix="other_domain",
    )
    f_disjoint = _make_manual_forecaster(
        "pool-run-disjoint",
        ["naive_seasonal"],
        oof_disjoint,
        pred_disjoint,
    )
    with pytest.raises(ConfigurationError, match="zero overlapping ts_id"):
        verify_oof_pool_compatibility([f1, f_disjoint])


def test_custom_model_name_override_across_sdk_cli_and_mcp(
    sample_panel_df: pd.DataFrame,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Custom `model_name` overrides the `'gemini-3.8-flash'` default across SDK, CLI, and MCP."""
    assert EvolveConfig().model_name == "gemini-3.8-flash"
    mock_cand_code = deterministic_mock_candidates("ensemble", count=1)[0]
    recorded_models: list[str] = []

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = f"```python\n{text}\n```"

    class _FakeModels:
        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: dict[str, Any] | None = None,
        ) -> _FakeResponse:
            _ = (contents, config)
            recorded_models.append(model)
            return _FakeResponse(mock_cand_code)

    class _FakeGenAIClient:
        def __init__(self) -> None:
            self.models = _FakeModels()

    res = Forecaster.evolve(
        sample_panel_df,
        target="ensemble",
        backend="vertex_gemini",
        model_name="gemini-4-argon",
        max_evaluations=1,
        horizon=4,
        genai_client=_FakeGenAIClient(),
    )
    assert recorded_models == ["gemini-4-argon"]
    assert (
        getattr(res, "model_name", None) == "gemini-4-argon"
        or res.diagnostics.get("model_name") == "gemini-4-argon"
    )
    dag_src = res.emit_airflow_dag(schedule="@weekly")
    assert "gemini-4-argon" in dag_src

    # CLI --model-name override
    rc = evolve_cli_main(
        [
            "--target",
            "features",
            "--model-name",
            "gemini-4-argon",
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert (
        cli_payload.get("model_name") == "gemini-4-argon"
        or cli_payload.get("diagnostics", {}).get("model_name") == "gemini-4-argon"
    )

    # MCP evolve_model_name override
    csv_path = tmp_path / "panel_custom_model.csv"
    sample_panel_df.to_csv(csv_path, index=False)
    server = McpServer(repo_root=REPO_ROOT)
    mcp_env = server.call_tool(
        "run_auto_forecast",
        {
            "source": str(csv_path),
            "horizon": 4,
            "evolve_target": "ensemble",
            "evolve_backend": "mock",
            "evolve_max_evaluations": 1,
            "evolve_model_name": "gemini-4-argon",
        },
    )
    assert mcp_env["isError"] is False
    mcp_payload = json.loads(mcp_env["content"][0]["text"])
    assert mcp_payload["model_name"] == "gemini-4-argon"
