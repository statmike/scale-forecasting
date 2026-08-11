"""Offline unit tests for the notebook-acceptance harness pure logic.

The live submit/poll path is exercised by the ``@gcp`` integration test of the same name.
Here we lock the parts that need no cloud: tier expansion, capacity-stockout classification, and
the executed-notebook error scan.
"""

from __future__ import annotations

import json

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
