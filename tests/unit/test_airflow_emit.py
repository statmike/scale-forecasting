"""Offline tests for the Airflow DAG emitter (``scale_forecasting.airflow_emit``).

Pure and GCP-free — and, deliberately, **Airflow-free**: the emitter renders a ``dag_<run_id>.py``
as a *string*, so we verify it by ``compile()``-ing that string (a syntax check that never imports
Airflow) and walking its `ast` for the expected tasks and structure. This is the same file Composer
would parse, so a green test here means the emitted DAG at least parses and carries every node the
run's DAG implies. Live execution of the emitted DAG is covered by the orchestrator/@gcp path.
"""

from __future__ import annotations

import ast
from typing import Any

import pytest

from scale_forecasting import airflow_emit
from scale_forecasting.config import RunConfig
from scale_forecasting.dag import plan_dag

# One model per family (mirrors test_dag): theta=statistical, lightgbm=ml,
# neuralprophet=deep_learning, arima_plus=native (BigQuery).
_STAT, _ML, _DL, _NATIVE = "theta", "lightgbm", "neuralprophet", "arima_plus"

_CONFIG_URI = "gs://example-code/runs/some-run.json"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "emit test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_STAT, _ML, _DL, _NATIVE],
    }
    base.update(over)
    return RunConfig(**base)


def _emit(cfg: RunConfig, **kw: Any) -> str:
    return airflow_emit.emit_airflow_dag(cfg, _CONFIG_URI, **kw)


def _compile(source: str) -> ast.Module:
    """Compile the emitted source (raises SyntaxError on bad output) and return its parsed AST."""
    compile(source, "dag_test.py", "exec")  # syntax gate — no Airflow import needed
    return ast.parse(source)


def _task_ids(tree: ast.Module) -> list[str]:
    """Every ``task_id=...`` string literal passed to a PythonOperator call, in source order."""
    ids: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "PythonOperator":
            continue
        for kw in node.keywords:
            if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                ids.append(kw.value.value)
    return ids


def _operator(tree: ast.Module, task_id: str) -> dict[str, Any]:
    """The kwargs of the ``PythonOperator`` carrying this ``task_id``.

    Literal kwargs come back as their values; the ones that reference emitted names (the
    ``python_callable``, the ``op_kwargs`` dict holding ``CONFIG_URI``) come back as source text.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "PythonOperator":
            continue
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                kwargs[kw.arg] = ast.unparse(kw.value)
        if kwargs.get("task_id") == task_id:
            return kwargs
    raise AssertionError(f"no PythonOperator with task_id {task_id!r} in emitted DAG")


def _edges(source: str) -> list[str]:
    """The emitted dependency lines (``a >> b``), stripped — the rendered topology, in order."""
    return [line.strip() for line in source.splitlines() if ">>" in line]


def _module_constant(tree: ast.Module, name: str) -> Any:
    """The value of a top-level ``NAME = <literal>`` assignment in the emitted module."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no module constant {name!r} in emitted DAG")


# --- it parses, and carries the run's identity ----------------------------------------------------


def test_emitted_dag_compiles() -> None:
    _compile(_emit(_cfg()))  # a SyntaxError here fails the test


def test_embeds_run_id_and_config_uri() -> None:
    cfg = _cfg()
    tree = _compile(_emit(cfg))
    assert _module_constant(tree, "RUN_ID") == plan_dag(cfg).run_id
    assert _module_constant(tree, "CONFIG_URI") == _CONFIG_URI


def test_is_deterministic() -> None:
    cfg = _cfg()
    assert _emit(cfg) == _emit(cfg)


def test_default_dag_id_is_run_scoped() -> None:
    cfg = _cfg()
    tree = _compile(_emit(cfg))
    dag_ids = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DAG"
        for kw in node.keywords
        if kw.arg == "dag_id" and isinstance(kw.value, ast.Constant)
    ]
    assert dag_ids == [f"scale_forecasting_{plan_dag(cfg).run_id}"]


def test_dag_id_override() -> None:
    tree = _compile(_emit(_cfg(), dag_id="my_custom_dag"))
    dag_ids = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DAG"
        for kw in node.keywords
        if kw.arg == "dag_id" and isinstance(kw.value, ast.Constant)
    ]
    assert dag_ids == ["my_custom_dag"]


# --- the task set mirrors the run DAG -------------------------------------------------------------


def test_one_task_per_family_plus_begin_and_finalize() -> None:
    # ensemble off by default → begin_run, a task per family, finalize_run (no ensemble task)
    ids = _task_ids(_compile(_emit(_cfg())))
    assert ids[0] == "begin_run"
    assert ids[-1] == "finalize_run"
    assert {"statistical", "ml", "deep_learning", "native"} <= set(ids)
    assert "ensemble" not in ids


def test_native_only_run_has_no_python_family_tasks() -> None:
    ids = set(_task_ids(_compile(_emit(_cfg(models=[_NATIVE, "timesfm"])))))
    assert "native" in ids
    assert ids.isdisjoint({"statistical", "ml", "deep_learning"})


def test_ensemble_task_present_when_enabled() -> None:
    cfg = _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean", "median"]},
    )
    assert "ensemble" in _task_ids(_compile(_emit(cfg)))


# --- shared ephemeral clusters --------------------------------------------------------------------


def test_two_ray_families_emit_a_shared_ray_cluster_bracket() -> None:
    # statistical + deep_learning both on Ray → one shared ephemeral cluster (create/delete bracket)
    cfg = _cfg(
        models=[_STAT, _DL],
        compute={
            "families": {"statistical": {"runtime": "ray"}, "deep_learning": {"runtime": "ray"}}
        },
    )
    ids = set(_task_ids(_compile(_emit(cfg))))
    assert {"create_ray_cluster", "delete_ray_cluster"} <= ids
    assert "create_spark_cluster" not in ids


def test_single_ray_family_does_not_share() -> None:
    # only one Ray family → the proven self-provisioning path, no shared-cluster bracket
    cfg = _cfg(models=[_STAT, _DL], compute={"families": {"deep_learning": {"runtime": "ray"}}})
    ids = set(_task_ids(_compile(_emit(cfg))))
    assert "create_ray_cluster" not in ids


def test_two_cluster_spark_families_emit_a_shared_dataproc_bracket() -> None:
    cfg = _cfg(
        models=[_STAT, _ML],
        compute={
            "families": {
                "statistical": {"runtime": "spark", "spark_mode": "cluster"},
                "ml": {"runtime": "spark", "spark_mode": "cluster"},
            }
        },
    )
    ids = set(_task_ids(_compile(_emit(cfg))))
    assert {"create_spark_cluster", "delete_spark_cluster"} <= ids
    assert "create_ray_cluster" not in ids


def test_serverless_spark_families_never_share() -> None:
    # the default runtime is serverless Spark; several serverless families stay separate batch tasks
    ids = set(_task_ids(_compile(_emit(_cfg(models=[_STAT, _ML, _DL])))))
    assert ids.isdisjoint({"create_spark_cluster", "delete_spark_cluster"})


# --- ensemble wiring: barrier vs microbatch -------------------------------------------------------


def test_barrier_ensemble_runs_after_families() -> None:
    # barrier is the default: the "families >> ensemble" edge appears (ensemble after the join)
    cfg = _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean"]},
    )
    source = _emit(cfg)
    _compile(source)  # syntax gate
    assert ">> ensemble" in source
    assert "begin_run >> ensemble" not in source  # not gated only on begin_run in barrier mode


def test_microbatch_ensemble_runs_in_parallel() -> None:
    cfg = _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean"]},
        compute={"ensemble": {"mode": "microbatch"}},
    )
    source = _emit(cfg)
    _compile(source)
    # microbatch fires the ensemble off begin_run (concurrent with the families), not after the join
    assert "begin_run >> ensemble" in source


# --- the retry node (opt-in, barrier only) --------------------------------------------------------
#
# `with_retry` inserts a repair between the family join and the ensemble, so a run that lost cells
# to a failed family still ensembles over the repaired ones. It is an emitter argument rather than a
# config field because a config field would feed the run_id digest — "repair automatically" would
# become a different run than "repair by hand", which is not what the digest means.


def _barrier(**over: Any) -> RunConfig:
    """A config that ensembles in barrier mode — the only mode the retry node is emitted for."""
    return _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean"]},
        **over,
    )


def test_no_retry_node_unless_it_is_asked_for() -> None:
    source = _emit(_barrier())
    assert "retry" not in _task_ids(_compile(source))


def test_the_retry_node_sits_between_the_family_join_and_the_ensemble() -> None:
    source = _emit(_barrier(), with_retry=True)
    ids = _task_ids(_compile(source))
    assert "retry" in ids
    assert _edges(source) == [
        "begin_run >> [statistical, ml, deep_learning, native]",
        "[statistical, ml, deep_learning, native] >> retry >> ensemble",
        "ensemble >> finalize_run",
    ]


def test_the_retry_node_runs_whether_the_families_succeeded_or_not() -> None:
    """``all_done``: the case a repair exists for is precisely the one where a family failed."""
    op = _operator(_compile(_emit(_barrier(), with_retry=True)), "retry")
    assert op["trigger_rule"] == "all_done"
    assert op["python_callable"] == "airflow_tasks.retry_families"
    # The node repairs the whole run, so it takes the config and no family — unlike run_family, it
    # cannot be told which families to touch, because only the registry knows what failed.
    assert "CONFIG_URI" in op["op_kwargs"] and "family" not in op["op_kwargs"]


def test_the_retry_node_waits_for_a_shared_cluster_teardown() -> None:
    """A repair provisions its own cluster under the run-derived name, so it must not race the
    delete."""
    cfg = _barrier(
        models=[_STAT, _DL],
        compute={
            "families": {"statistical": {"runtime": "ray"}, "deep_learning": {"runtime": "ray"}}
        },
    )
    source = _emit(cfg, with_retry=True)
    _compile(source)
    assert _edges(source) == [
        "begin_run >> create_ray_cluster >> [statistical, deep_learning] >> delete_ray_cluster",
        "[statistical, deep_learning, delete_ray_cluster] >> retry >> ensemble",
        "ensemble >> finalize_run",
    ]


def test_the_retry_node_is_never_a_leaf() -> None:
    """Whatever the run's shape, something downstream consumes the repair — else it is dead work."""
    shapes = [
        _barrier(),
        _barrier(
            models=[_STAT, _DL],
            compute={
                "families": {"statistical": {"runtime": "ray"}, "deep_learning": {"runtime": "ray"}}
            },
        ),
        _cfg(),  # no ensemble at all
        _cfg(models=[_NATIVE]),  # BigQuery-native only
    ]
    for cfg in shapes:
        source = _emit(cfg, with_retry=True)
        _compile(source)
        assert any(edge.startswith("retry >>") or ">> retry >>" in edge for edge in _edges(source))


def test_a_run_without_an_ensemble_still_repairs_before_it_closes() -> None:
    # Nothing blends the repaired cells, but the run's own predictions table is still a consumer.
    source = _emit(_cfg(), with_retry=True)
    _compile(source)
    assert _edges(source) == [
        "begin_run >> [statistical, ml, deep_learning, native]",
        "[statistical, ml, deep_learning, native] >> retry",
        "retry >> finalize_run",
    ]


def test_a_microbatch_run_gets_no_retry_node_and_says_why(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The microbatch ensemble drains as soon as the base families go terminal, so a post-join repair
    # would land cells it has already stopped reading — barrier and microbatch would then disagree
    # about what the same config produces. Suppressed, but never silently.
    cfg = _barrier(compute={"ensemble": {"mode": "microbatch"}})
    with caplog.at_level("INFO"):
        source = _emit(cfg, with_retry=True)
    _compile(source)
    assert "retry" not in _task_ids(ast.parse(source))
    assert "no retry node emitted" in caplog.text
