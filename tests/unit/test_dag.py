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


# --- planned_families (what close-runs compares job rows against) ----------------


def test_planned_families_are_the_model_families_in_dag_order() -> None:
    assert dag.planned_families(_cfg(models=[_NATIVE, _STAT])) == ("statistical", "native")


def test_planned_families_adds_the_ensemble_only_when_ensembling_is_on() -> None:
    """The ensemble node writes a job row of its own, so a plan that omits it under-counts.

    This is the whole point of the helper: the run that stranded `registry.ops.close_runs` had both
    its base families COMPLETED and an ensemble that never started, and nothing in the job rows said
    an ensemble had ever been asked for.
    """
    off = _cfg(models=[_STAT, _NATIVE])
    on = _cfg(models=[_STAT, _NATIVE], ensemble={"enabled": True, "strategies": ["mean"]})
    assert dag.planned_families(off) == ("statistical", "native")
    assert dag.planned_families(on) == ("statistical", "native", "ensemble")


def test_planned_families_never_expects_a_repair_row() -> None:
    """A repair exists only because cells went missing — it is never *planned* work.

    If it were listed, every run that simply did not need repairing would read as having skipped a
    family, and close-runs would refuse to call any of them COMPLETED.
    """
    from scale_forecasting.registry.ids import REPAIR_JOB_FAMILIES

    planned = dag.planned_families(_cfg(ensemble={"enabled": True, "strategies": ["mean"]}))
    assert not set(planned) & set(REPAIR_JOB_FAMILIES)


def test_planned_families_matches_the_job_rows_a_full_dag_would_write() -> None:
    """The plan and the DAG must agree, or the comparison is against a fiction.

    ``dag_nodes`` is what actually executes and therefore what writes the rows; this pins the helper
    to it rather than to a second reading of the config.
    """
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    nodes = dag.dag_nodes(dag.plan_dag(cfg))
    assert dag.planned_families(cfg) == tuple(n.family for n in nodes)


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


# --- dag_nodes: the offline per-node identity + dependency surface --------------


def test_dag_nodes_one_per_family_with_deterministic_job_keys() -> None:
    from scale_forecasting.registry.ids import make_job_key

    cfg = _cfg(models=[_STAT, _ML, _NATIVE])  # ensemble off by default
    run_dag = dag.plan_dag(cfg)
    nodes = dag.dag_nodes(run_dag)
    assert [n.family for n in nodes] == ["statistical", "ml", "native"]
    # each node carries the deterministic attempt-1 job_key for its family under the shared run_id
    for n in nodes:
        assert n.job_key == make_job_key(run_dag.run_id, n.family, 1)
    # family jobs have no upstream dependency (no ensemble here)
    assert all(n.depends_on == () for n in nodes)


def test_dag_nodes_carry_resolved_runtime_and_hardware() -> None:
    cfg = _cfg(
        models=[_STAT, _DL],
        compute={"families": {"deep_learning": {"runtime": "ray", "hardware": "gpu"}}},
    )
    nodes = {n.family: n for n in dag.dag_nodes(dag.plan_dag(cfg))}
    assert nodes["statistical"].runtime == "spark"
    assert nodes["deep_learning"].runtime == "ray"
    assert nodes["deep_learning"].hardware == "gpu"
    assert nodes["deep_learning"].gpu_type is not None


def test_dag_nodes_native_node_has_no_compute() -> None:
    nodes = {n.family: n for n in dag.dag_nodes(dag.plan_dag(_cfg(models=[_NATIVE])))}
    native = nodes["native"]
    assert native.runtime == "bigquery"
    assert native.hardware is None
    assert native.gpu_type is None
    assert native.spark_mode is None


def test_dag_nodes_ensemble_node_depends_on_all_family_jobs() -> None:
    from scale_forecasting.registry.ids import make_job_key

    cfg = _cfg(
        models=[_STAT, _ML, _NATIVE],
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean", "median"]},
    )
    run_dag = dag.plan_dag(cfg)
    nodes = dag.dag_nodes(run_dag)
    ensemble = nodes[-1]
    assert ensemble.family == "ensemble"
    assert ensemble.runtime == "bigquery"
    assert ensemble.models == ()
    assert ensemble.job_key == make_job_key(run_dag.run_id, "ensemble", 1)
    # it waits on every base family job's job_key
    assert set(ensemble.depends_on) == {n.job_key for n in nodes if n.family != "ensemble"}


def test_dag_nodes_no_ensemble_node_when_disabled() -> None:
    nodes = dag.dag_nodes(dag.plan_dag(_cfg(models=[_STAT, _ML])))
    assert all(n.family != "ensemble" for n in nodes)


def test_dag_nodes_is_pure_and_deterministic() -> None:
    cfg = _cfg()
    assert dag.dag_nodes(dag.plan_dag(cfg)) == dag.dag_nodes(dag.plan_dag(cfg))


# --- narrowing: the repair's submission grain ----------------------------------


def test_narrowing_keeps_only_the_models_asked_for() -> None:
    cfg = _cfg(models=[_STAT, "sarimax", _ML, _NATIVE])
    narrowed = dag.narrow_to_models(dag.plan_dag(cfg), ["sarimax", _NATIVE])
    assert {job.family: job.models for job in narrowed.jobs} == {
        "statistical_repair": ("sarimax",),
        "native_repair": (_NATIVE,),
    }


def test_a_family_left_with_nothing_is_dropped_rather_than_submitted_empty() -> None:
    # An empty --models would run the whole family, which is the opposite of a repair.
    narrowed = dag.narrow_to_models(dag.plan_dag(_cfg()), [_ML])
    assert narrowed.families == ["ml_repair"]


def test_a_narrowed_job_files_under_its_own_family_token() -> None:
    """The repair's row must not be the family's row — that is the whole point of the token.

    `v_run_jobs` keeps the highest attempt per (run_id, family), so a repair filed as attempt 2 of
    ``statistical`` would erase the failed attempt it was launched to fix and report the whole
    family COMPLETED. The distinct token gives it a row beside the original instead.
    """
    from scale_forecasting.registry.ids import make_job_key

    narrowed = dag.narrow_to_models(dag.plan_dag(_cfg(models=[_STAT, _ML])), [_STAT])
    (job,) = narrowed.jobs
    assert job.family == "statistical_repair"
    # And the token is a real member of the id vocabulary, so the launch path needs no special case.
    assert make_job_key(narrowed.run_id, job.family, 1).endswith("-statistical_repair-a1")


def test_a_repaired_native_family_still_routes_to_bigquery() -> None:
    """Routing asks `base_family`, so ``native_repair`` reaches the BigQuery launcher, not a thread.

    `job_launch.launch_family_job` asserts ``job.compute is not None`` and a native job never has
    compute, so mis-routing here is an AssertionError on the driver thread — the exact failure a
    repair path must not introduce.
    """
    narrowed = dag.narrow_to_models(dag.plan_dag(_cfg(models=[_STAT, _NATIVE])), [_STAT, _NATIVE])
    assert narrowed.native_job is not None
    assert narrowed.native_job.family == "native_repair"
    assert [j.family for j in narrowed.python_jobs] == ["statistical_repair"]


def test_narrowing_carries_the_resolved_compute_through_untouched() -> None:
    # A repaired family has to land on the runtime and hardware the original attempt chose --
    # re-resolving it would let a repair quietly move a job to different hardware.
    cfg = _cfg(models=[_STAT, _DL], compute={"families": {"deep_learning": {"runtime": "ray"}}})
    planned = {job.family: job.compute for job in dag.plan_dag(cfg).jobs}
    narrowed = dag.narrow_to_models(dag.plan_dag(cfg), [_DL])
    assert narrowed.jobs[0].compute == planned["deep_learning"]


def test_a_narrowed_dag_advertises_no_ensemble_node() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    assert dag.plan_dag(cfg).ensemble_enabled is True
    assert dag.narrow_to_models(dag.plan_dag(cfg), [_STAT]).ensemble_enabled is False


def test_narrowing_keeps_the_run_id_it_was_given() -> None:
    # A repair is attempt N+1 of the *same* run. A different run_id here would file the repaired
    # cells under a run nobody asked about.
    planned = dag.plan_dag(_cfg())
    assert dag.narrow_to_models(planned, [_STAT]).run_id == planned.run_id


def test_narrowing_to_a_model_the_run_never_planned_is_refused() -> None:
    import pytest

    from scale_forecasting.errors import ConfigError

    with pytest.raises(ConfigError, match="not planned by this run"):
        dag.narrow_to_models(dag.plan_dag(_cfg(models=[_STAT])), ["sarimax"])


def test_a_narrowed_job_produces_the_driver_args_a_hand_written_subset_would() -> None:
    """The exit-gate equivalence: narrowing is only useful if it reaches the driver as ``--models``.

    `commands.build_driver_args` is the single place a subset becomes a flag, for every runtime, so
    asserting the narrowed job's ``models`` builds the same arg list an operator typing
    ``--models sarimax`` would get is what ties the pure narrowing to the thing that executes.
    """
    from scale_forecasting.commands import build_driver_args
    from scale_forecasting.settings import Settings

    settings = Settings(
        project_id="proj-x",
        connection="proj-x.us-central1.conn",
        warehouse_uri="gs://bkt/warehouse",
    )
    cfg = _cfg(models=[_STAT, "sarimax", _ML])
    narrowed = dag.narrow_to_models(dag.plan_dag(cfg), ["sarimax"])
    from_dag = build_driver_args(
        "gs://b/c.json", settings, models=list(narrowed.jobs[0].models), manage_header=False
    )
    by_hand = build_driver_args("gs://b/c.json", settings, models=["sarimax"], manage_header=False)
    assert from_dag == by_hand
    assert "--models" in from_dag and from_dag[from_dag.index("--models") + 1] == "sarimax"
