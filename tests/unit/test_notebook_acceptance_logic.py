"""Offline unit tests for the notebook-acceptance harness pure logic.

The live submit/poll path is exercised by the ``@gcp`` integration test of the same name.
Here we lock the parts that need no cloud: tier expansion, capacity-stockout classification, and
the executed-notebook error scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale_forecasting import notebook_acceptance as na


def test_notebooks_for_tier_is_cumulative() -> None:
    smoke = {s.name for s in na.notebooks_for_tier(na.TIER_SMOKE)}
    batch = {s.name for s in na.notebooks_for_tier(na.TIER_BATCH)}
    full = {s.name for s in na.notebooks_for_tier(na.TIER_FULL)}
    assert smoke < batch < full  # each tier strictly contains the previous
    assert "04_ray_on_vertex" in full and "04_ray_on_vertex" not in batch


@pytest.mark.parametrize(
    "detail",
    [
        "The us-central1 region currently does not have enough resources to fulfill the request "
        "for a e2-standard-4 machine.",
        "ZONE_RESOURCE_POOL_EXHAUSTED: no capacity",
        "RESOURCE_EXHAUSTED: quota",
        "Compute Engine is out of resources in zone us-central1-a",
        "insufficient capacity for the requested accelerator",
    ],
)
def test_is_capacity_unavailable_matches_stockout_signals(detail: str) -> None:
    assert na.is_capacity_unavailable(detail)


@pytest.mark.parametrize(
    "detail",
    [
        "Error while creating Dataproc Session: User not authorized to act as service account",
        "ValidationError: 1 validation error for RunConfig",
        "ModuleNotFoundError: No module named 'scale_forecasting'",
        "",
    ],
)
def test_is_capacity_unavailable_ignores_real_errors(detail: str) -> None:
    assert not na.is_capacity_unavailable(detail)


def _nb_with_error(ename: str, evalue: str) -> bytes:
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# heading"], "metadata": {}},
            {
                "cell_type": "code",
                "source": ["boom()"],
                "metadata": {},
                "execution_count": 1,
                "outputs": [
                    {"output_type": "error", "ename": ename, "evalue": evalue, "traceback": []}
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb).encode("utf-8")


def test_first_cell_error_surfaces_ename_evalue() -> None:
    nb = _nb_with_error("RuntimeError", "does not have enough resources")
    msg = na._first_cell_error(nb)
    assert msg == "RuntimeError: does not have enough resources"
    # And that message classifies as capacity — the in-cell stockout path.
    assert na.is_capacity_unavailable(msg)


def test_first_cell_error_empty_on_clean_notebook() -> None:
    clean = json.dumps(
        {
            "cells": [{"cell_type": "code", "source": ["1+1"], "metadata": {}, "outputs": []}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode("utf-8")
    assert na._first_cell_error(clean) == ""
    assert na.assert_no_cell_errors(clean) == 0


# --- fan-out (non-blocking submit) ---------------------------------------------------------------


def test_executions_console_url_carries_project() -> None:
    url = na.executions_console_url("gcp-scale-forecasting")
    assert url.startswith("https://console.cloud.google.com/vertex-ai/colab/execution-jobs")
    assert "project=gcp-scale-forecasting" in url


def _touch_notebooks(tmp_path: Path, specs: list[na.NotebookSpec]) -> Path:
    """Create empty .ipynb files for the given specs so run_fanout finds them on disk."""
    nb_dir = tmp_path / "notebooks"
    nb_dir.mkdir()
    for spec in specs:
        (nb_dir / f"{spec.name}.ipynb").write_bytes(b"{}")
    return nb_dir


def test_run_fanout_submits_all_without_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = na.notebooks_for_tier(na.TIER_FULL)
    nb_dir = _touch_notebooks(tmp_path, specs)

    calls: list[dict[str, object]] = []

    def _fake_submit(**kwargs: object) -> str:
        calls.append(kwargs)
        return f"job-{len(calls)}"

    # Fan-out must NOT poll/download — fail loudly if it tries.
    def _boom(**_: object) -> object:
        raise AssertionError("fan-out must not poll or download")

    monkeypatch.setattr(na, "submit_job", _fake_submit)
    monkeypatch.setattr(na, "poll_to_terminal", _boom)
    monkeypatch.setattr(na, "download_executed", _boom)

    results = na.run_fanout(
        specs=specs,
        project_id="proj-x",
        region="us-central1",
        notebooks_dir=nb_dir,
        template_ids={na.TEMPLATE_MAIN: "tmpl/main", na.TEMPLATE_SPARK: "tmpl/spark"},
        service_account="runner@proj-x.iam.gserviceaccount.com",
        gcs_output_uri="gs://proj-x-code/nb",
        credentials=object(),
        run_label="tonight",
    )

    assert len(results) == len(specs) == len(calls)
    assert all(r.job_id for r in results) and all(r.detail == "" for r in results)
    # NB01 routes to the spark template; everything else to main (the registry's routing).
    by_name = {c["notebook_path"].name: c["template_resource_name"] for c in calls}  # type: ignore[union-attr]
    assert by_name["01_spark_via_connect.ipynb"] == "tmpl/spark"
    assert by_name["07_scale_review.ipynb"] == "tmpl/main"
    # executed_uri is the path the run WILL land at: {out}/fanout/{label}/{name}/{job}/content.ipynb
    r07 = next(r for r in results if r.name == "07_scale_review")
    assert r07.executed_uri.endswith(f"fanout/tonight/07_scale_review/{r07.job_id}/content.ipynb")


def test_run_fanout_missing_file_does_not_sink_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = na.notebooks_for_tier(na.TIER_SMOKE)
    nb_dir = _touch_notebooks(tmp_path, specs)
    # Remove one notebook file so its submit is skipped with a detail, others still go.
    (nb_dir / "02_bigquery_native.ipynb").unlink()

    monkeypatch.setattr(na, "submit_job", lambda **_: "job-ok")
    results = na.run_fanout(
        specs=specs,
        project_id="proj-x",
        region="us-central1",
        notebooks_dir=nb_dir,
        template_ids={na.TEMPLATE_MAIN: "tmpl/main", na.TEMPLATE_SPARK: "tmpl/spark"},
        service_account="runner@proj-x.iam.gserviceaccount.com",
        gcs_output_uri="gs://proj-x-code/nb",
        credentials=object(),
        run_label="tonight",
    )
    missing = next(r for r in results if r.name == "02_bigquery_native")
    assert missing.job_id == "" and "missing" in missing.detail
    assert all(r.job_id == "job-ok" for r in results if r.name != "02_bigquery_native")
