"""Parse-under-Airflow test for the emitted DAG (opt-in ``@airflow``).

The other emitter tests (`test_airflow_emit`) ``compile()`` the emitted ``dag_<run_id>.py`` and walk
its ``ast`` — deliberately Airflow-free, so they never prove Airflow itself can *load* the file.
This test closes that gap: it skips unless ``apache-airflow`` is importable, and when it is, loads
an emitted DAG through ``airflow.models.DagBag`` — the same path Composer's scheduler uses. A green
run proves the file parses under Airflow end to end: the operator kwargs, ``pendulum``, the ``>>``
wiring, and the ``from scale_forecasting import airflow_tasks`` import chain all resolve — mistakes
the string/ast checks structurally cannot catch.

Airflow is installed *isolated* (a scratch venv in CI, never in ``uv.lock``), so this is marked
``@airflow`` and behaves like ``@spark`` / ``@ray``: absent the dep it is skipped, not failed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# AIRFLOW_HOME (and these config toggles) must be set *before* airflow is imported — it reads them
# at import time. Point it at a throwaway dir and use unit-test mode so importing airflow touches no
# real ~/airflow and no metadata database; DagBag parsing itself never needs a live DB.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="sf-airflow-test-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "True")

pytestmark = pytest.mark.airflow

pytest.importorskip("airflow")  # skip cleanly when apache-airflow is not installed

from scale_forecasting import airflow_emit  # noqa: E402  (after importorskip by design)
from scale_forecasting.config import RunConfig  # noqa: E402
from scale_forecasting.dag import plan_dag  # noqa: E402

# One model per family, same choices as test_airflow_emit.
_STAT, _ML, _DL, _NATIVE = "theta", "lightgbm", "neuralprophet", "arima_plus"
_CONFIG_URI = "gs://example-code/runs/some-run.json"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "dagbag test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_STAT, _ML, _DL, _NATIVE],
    }
    base.update(over)
    return RunConfig(**base)


def _load_dag(cfg: RunConfig, tmp_path: Path) -> Any:
    """Emit ``cfg``'s DAG, load it through a real Airflow ``DagBag``, and return the parsed dag.

    Asserts the DagBag reported no import errors (the parse gate) and the run's dag_id resolved.
    """
    from airflow.models import DagBag

    source = airflow_emit.emit_airflow_dag(cfg, _CONFIG_URI)
    (tmp_path / "dag_under_test.py").write_text(source)
    bag = DagBag(dag_folder=str(tmp_path), include_examples=False)
    assert bag.import_errors == {}, f"Airflow failed to parse the emitted DAG: {bag.import_errors}"

    # Read the parsed DAG straight from the in-memory collection — DagBag.get_dag() would query the
    # metadata DB (which this hermetic parse-only test never provisions).
    dag_id = f"scale_forecasting_{plan_dag(cfg).run_id}"
    assert dag_id in bag.dags, f"emitted dag_id {dag_id!r} not parsed; got {list(bag.dags)}"
    return bag.dags[dag_id]


def test_default_dag_parses_under_airflow(tmp_path: Path) -> None:
    # ensemble off: begin_run, one task per family, finalize_run — loads without import errors.
    dag = _load_dag(_cfg(), tmp_path)
    tasks = set(dag.task_ids)
    assert {"begin_run", "finalize_run", "statistical", "ml", "deep_learning", "native"} <= tasks
    assert "ensemble" not in tasks
    # begin_run is the source; finalize_run is the terminal join.
    assert dag.get_task("begin_run").upstream_task_ids == set()
    assert dag.get_task("finalize_run").downstream_task_ids == set()
    assert "finalize_run" in dag.get_task("statistical").downstream_task_ids


def test_microbatch_dag_parses_under_airflow(tmp_path: Path) -> None:
    cfg = _cfg(
        backtest={"enabled": True},
        ensemble={"enabled": True, "strategies": ["mean", "median"]},
        compute={"ensemble": {"mode": "microbatch"}},
    )
    dag = _load_dag(cfg, tmp_path)
    # microbatch fires the ensemble off begin_run (concurrent with the families), not after join.
    assert "ensemble" in dag.task_ids
    assert "ensemble" in dag.get_task("begin_run").downstream_task_ids


def test_shared_ray_cluster_dag_parses_under_airflow(tmp_path: Path) -> None:
    # two Ray families → the create/delete bracket; it must parse and wire under Airflow too.
    cfg = _cfg(
        models=[_STAT, _DL],
        compute={
            "families": {"statistical": {"runtime": "ray"}, "deep_learning": {"runtime": "ray"}}
        },
    )
    dag = _load_dag(cfg, tmp_path)
    assert {"create_ray_cluster", "delete_ray_cluster"} <= set(dag.task_ids)
    # the shared cluster gates the Ray families, and its teardown feeds the terminal join.
    assert "statistical" in dag.get_task("create_ray_cluster").downstream_task_ids
    assert "finalize_run" in dag.get_task("delete_ray_cluster").downstream_task_ids
