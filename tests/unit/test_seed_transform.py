"""Tests for the pure seed shaping transform (``data_gen.seed_spark._to_source_rows``).

The Spark seed job's only non-trivial logic that isn't already covered by the generator tests is
the reconciliation from the generator frame (``ts_id, archetype, ds(datetime64[ns]), y``) to the
``source_series`` DDL schema (``ts_id, ds DATE, y, archetype, is_holiday BOOL``). The shipped
example is univariate; the exog seam lives in the generator/config, not the shipped source table
(see ``test_generator.py`` for that dormant path). This transform is factored out as a pure pandas
function so it can be exercised offline — no Spark, no cluster — against a real (tiny) generator
call, so the shipped seed and the test share one code path (same discipline as the golden fixture).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scale_forecasting._infra_args import INFRA_ARG_ENV
from scale_forecasting.data_gen.generator import GenConfig, generate_partition
from scale_forecasting.data_gen.seed_spark import (
    _SOURCE_COLUMNS,
    _parse_args,
    _to_source_rows,
)

SEED = 20260726


def _cfg(**over: object) -> GenConfig:
    base: dict[str, object] = {"history": 400, "freq": "D", "holidays": ("US",)}
    base.update(over)
    return GenConfig(**base)  # type: ignore[arg-type]


def test_columns_are_ddl_order() -> None:
    cfg = _cfg()
    rows = _to_source_rows(generate_partition([0, 1], cfg, SEED), cfg.holidays)
    assert list(rows.columns) == list(_SOURCE_COLUMNS)


def test_ds_is_date_not_datetime() -> None:
    cfg = _cfg()
    rows = _to_source_rows(generate_partition([0], cfg, SEED), cfg.holidays)
    # source_series.ds is DATE; every value must be a plain python date (not Timestamp).
    assert rows["ds"].map(lambda d: isinstance(d, dt.date)).all()
    assert not rows["ds"].map(lambda d: isinstance(d, dt.datetime)).any()


def test_is_holiday_matches_generator_calendar() -> None:
    cfg = _cfg(history=400)
    frame = generate_partition([0], cfg, SEED)
    rows = _to_source_rows(frame, cfg.holidays)
    # 2021-01-01 (US New Year) is inside the window and must be flagged; an ordinary day is not.
    by_date = dict(zip(rows["ds"], rows["is_holiday"], strict=True))
    assert by_date[dt.date(2021, 1, 1)]
    assert not by_date[dt.date(2021, 1, 4)]
    assert rows["is_holiday"].dtype == "boolean"


def test_source_rows_are_univariate() -> None:
    # The shipped seed transform is univariate — no exog/price_index column, even if the generator
    # frame carried one. The exog seam is exercised in test_generator.py, not here.
    cfg = _cfg()
    rows = _to_source_rows(generate_partition([0], cfg, SEED), cfg.holidays)
    assert "price_index" not in rows.columns
    assert list(rows.columns) == list(_SOURCE_COLUMNS)


def test_ts_id_and_archetype_preserved() -> None:
    cfg = _cfg()
    frame = generate_partition([0, 3], cfg, SEED)
    rows = _to_source_rows(frame, cfg.holidays)
    # ts_ids survive the reconcile; row count and values match the generator frame.
    assert set(rows["ts_id"]) == {"s_000000", "s_000003"}
    assert len(rows) == len(frame)
    pd.testing.assert_series_equal(
        rows["y"].reset_index(drop=True),
        frame["y"].astype("float64").reset_index(drop=True),
        check_names=False,
    )


def test_empty_partition_yields_typed_empty_frame() -> None:
    cfg = _cfg()
    rows = _to_source_rows(generate_partition([], cfg, SEED), cfg.holidays)
    assert len(rows) == 0
    assert list(rows.columns) == list(_SOURCE_COLUMNS)


def test_no_holidays_all_false() -> None:
    cfg = _cfg(holidays=())
    rows = _to_source_rows(generate_partition([0], cfg, SEED), cfg.holidays)
    assert not rows["is_holiday"].any()


# --- infra args → env (the Dataproc-Serverless delivery path) -------------------


def test_parse_args_exports_infra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The batch delivers SF_* as --sf-* args; parsing them must populate os.environ so the
    # existing env-based Settings.resolve() seam works unchanged (driver-env is rejected by
    # Dataproc Serverless, so args are the only reliable path).
    for _, env_name in INFRA_ARG_ENV:
        monkeypatch.delenv(env_name, raising=False)
    _parse_args(
        [
            "--n-series",
            "5",
            "--sf-project-id",
            "proj-x",
            "--sf-connection",
            "proj-x.us-central1.conn",
            "--sf-warehouse-uri",
            "gs://bkt/warehouse",
            "--sf-dataset-id",
            "ds_x",
            "--sf-region",
            "us-central1",
        ]
    )
    import os

    assert os.environ["SF_PROJECT_ID"] == "proj-x"
    assert os.environ["SF_CONNECTION"] == "proj-x.us-central1.conn"
    assert os.environ["SF_WAREHOUSE_URI"] == "gs://bkt/warehouse"
    assert os.environ["SF_DATASET_ID"] == "ds_x"
    assert os.environ["SF_REGION"] == "us-central1"


def test_parse_args_leaves_env_untouched_when_no_infra_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Local runs pass no --sf-* and must not clobber an ambient SF_PROJECT_ID.
    monkeypatch.setenv("SF_PROJECT_ID", "ambient-proj")
    _parse_args(["--n-series", "5"])
    import os

    assert os.environ["SF_PROJECT_ID"] == "ambient-proj"
