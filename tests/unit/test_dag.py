"""Offline tests for the run DAG planner (``scale_forecasting.dag``).

Pure and GCP-free: model→family grouping, the per-family job set with resolved compute, the
native/Python split, and the ensemble flag. The live execution of the DAG (family jobs ∥ BigQuery →
ensemble under one run_id) is exercised by the orchestrator tests and the ``@gcp`` smoke.
"""

from __future__ import annotations

from typing import Any

from scale_forecasting import dag
from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id

# One model per family: theta=statistical, lightgbm=ml, neuralprophet=deep_learning,
# arima_plus=native (BigQuery).
_STAT, _ML, _DL, _NATIVE = "theta", "lightgbm", "neuralprophet", "arima_plus"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "dag test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_STAT, _ML, _DL, _NATIVE],
    }
    base.update(over)
    return RunConfig(**base)


# --- grouping ------------------------------------------------------------------


def test_group_orders_families_and_preserves_model_order() -> None:
    cfg = _cfg(models=[_NATIVE, _ML, "sarimax", _STAT, _DL])
    grouped = dag.group_models_by_family(cfg)
    # families come back in the fixed DAG order (Python families first, native last)...
    assert list(grouped) == ["statistical", "ml", "deep_learning", "native"]
    # ...and models keep their config order within a family
    assert grouped["statistical"] == ["sarimax", _STAT]
    assert grouped["ml"] == [_ML]
    assert grouped["native"] == [_NATIVE]


def test_group_omits_absent_families() -> None:
    grouped = dag.group_models_by_family(_cfg(models=[_STAT, _NATIVE]))
    assert set(grouped) == {"statistical", "native"}


# --- plan_dag ------------------------------------------------------------------


def test_plan_dag_run_id_matches_full_config_digest() -> None:
    cfg = _cfg()
    assert dag.plan_dag(cfg).run_id == make_run_id(cfg)


def test_plan_dag_one_job_per_present_family_in_order() -> None:
    d = dag.plan_dag(_cfg())
    assert d.families == ["statistical", "ml", "deep_learning", "native"]
    assert [j.models for j in d.jobs] == [(_STAT,), (_ML,), (_DL,), (_NATIVE,)]


def test_plan_dag_native_job_has_no_compute_and_bigquery_runtime() -> None:
    native = dag.plan_dag(_cfg()).native_job
    assert native is not None
    assert native.compute is None
    assert native.runtime == "bigquery"


def test_plan_dag_python_jobs_resolve_family_compute() -> None:
    d = dag.plan_dag(_cfg())
    python = d.python_jobs
    assert [j.family for j in python] == ["statistical", "ml", "deep_learning"]
    for job in python:
        assert job.compute is not None
        # a plain config inherits the run-level default runtime (spark) on every Python family
        assert job.runtime == "spark"
        assert job.compute.family == job.family


def test_plan_dag_honors_per_family_runtime_override() -> None:
    # deep_learning routed to Ray while the rest stay on the default Spark runtime.
    cfg = _cfg(compute={"families": {"deep_learning": {"runtime": "ray"}}})
    jobs = {j.family: j for j in dag.plan_dag(cfg).jobs}
    assert jobs["deep_learning"].runtime == "ray"
    assert jobs["statistical"].runtime == "spark"


def test_plan_dag_all_bigquery_has_only_native_job() -> None:
    d = dag.plan_dag(_cfg(models=[_NATIVE, "timesfm"]))
    assert d.families == ["native"]
    assert d.python_jobs == []
    assert d.native_job is not None


def test_plan_dag_all_python_has_no_native_job() -> None:
    d = dag.plan_dag(_cfg(models=[_STAT, _ML]))
    assert d.native_job is None
    assert d.families == ["statistical", "ml"]


def test_plan_dag_ensemble_flag_tracks_config() -> None:
    assert dag.plan_dag(_cfg()).ensemble_enabled is False
    cfg = _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean", "median"]},
    )
    assert dag.plan_dag(cfg).ensemble_enabled is True


def test_plan_dag_is_deterministic() -> None:
    cfg = _cfg()
    assert dag.plan_dag(cfg) == dag.plan_dag(cfg)
