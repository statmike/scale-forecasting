"""Every writable column has a producer, or is named here as one that does not yet.

Three tables are written by more than one engine, and the Storage Write API encoder
(`registry.write_api._encode_rows`) walks the **column spec**, not the row. That makes two
mistakes silent in opposite directions:

* a key an assembler emits that the spec does not name is **dropped**, with no error — the
  column just reads NULL, and only for the engine that emitted it;
* a column the spec names that no assembler emits is a **permanent NULL** — the table looks
  like it has the field, and every reader that finds it empty concludes the run didn't
  produce one.

Both are invisible live. The existing per-assembler tests catch the first (each row's keys
are a subset of its spec). This module catches the second, which needs the *union* over every
producer of a table — no single assembler emits the whole spec, and asserting that any one of
them does is wrong by design (`assemble_metadata_row` has no ``ensemble_id``; only the
ensemble path does).

The columns with no producer are not a failure — they are declared ahead of the code that
fills them so a deployment migrates once instead of once per phase. They are listed in the
``_RESERVED_*`` sets below, which is the point: the set is the record of what is still empty,
it is visible in a diff, and **it must shrink to empty**. When you wire one up, delete it here
in the same commit; the assertion is equality, so leaving it behind fails just as loudly as
forgetting to add it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ddl import additive_columns
from scale_forecasting.registry.rows import (
    assemble_metadata_row,
    assemble_oof_rows,
    assemble_prediction_rows,
)
from scale_forecasting.registry.write_api import _META_SPEC, _OOF_SPEC, _PRED_SPEC
from scale_forecasting.worker import CellResult

_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

# Columns the schema declares and nothing writes yet, per table. Each entry is a promise that
# some named phase fills it; deleting the entry is part of landing that phase.
_RESERVED_PREDICTIONS: set[str] = set()  # created_at landed in P10 7.2
_RESERVED_OOF: set[str] = set()  # created_at landed in P10 7.2
_RESERVED_METADATA = {
    # cell outcome + backtest methodology
    "achieved_step",
    "achieved_min_train",
    "first_val_date",
    "last_val_date",
}


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "column parity",
        "data": {"source_table": "p.d.source_series_native"},
        "models": ["theta", "sarimax"],
        "ensemble": {"enabled": True, "strategies": ["mean", "inverse_error"]},
    }
    base.update(over)
    return RunConfig(**base)


def _result() -> CellResult:
    idx = pd.to_datetime(["2026-02-01", "2026-02-02"])
    return CellResult(
        run_id="my-run-abc123def456",
        ts_id="series-7",
        model_type="theta",
        compute_engine="spark",
        model_hash="hash-1",
        status="ok",
        error=None,
        predictions=pd.DataFrame(
            {
                "ds": idx,
                "yhat": [1.0, 2.0],
                "yhat_lower": [0.5, 1.5],
                "yhat_upper": [1.5, 2.5],
                "quantiles": [{"0.5": 1.0}, {"0.5": 2.0}],
            }
        ),
        oof=pd.DataFrame({"ds": idx, "fold_id": [0, 0], "y_true": [1.1, 2.1], "yhat": [1.0, 2.0]}),
        metrics={"wape": 0.1},
        fit_seconds=0.5,
    )


def _base_pred_frame() -> pd.DataFrame:
    """Long-format base predictions, the shape the ensemble blenders read."""
    return pd.DataFrame(
        {
            "ts_id": ["s1", "s1", "s1", "s1"],
            "model_type": ["theta", "theta", "sarimax", "sarimax"],
            "forecast_date": [date(2026, 2, 1), date(2026, 2, 2)] * 2,
            "yhat": [1.0, 2.0, 1.2, 2.2],
            "yhat_lower": [0.5, 1.5, 0.7, 1.7],
            "yhat_upper": [1.5, 2.5, 1.7, 2.7],
        }
    )


def _spec_columns(spec: tuple[tuple[str, str], ...]) -> set[str]:
    return {col for col, _ in spec}


def _prediction_producers() -> set[str]:
    """Every key any engine emits for ``forecast_predictions``."""
    from scale_forecasting.engines.bigquery_sql import _PRED_COLS
    from scale_forecasting.ensemble_run import _apply_weights
    from scale_forecasting.ensembler import combine_calculated

    emitted = set(assemble_prediction_rows(_result())[0])

    # The native path writes by INSERT ... SELECT rather than the Write API, so its "row keys"
    # are the column list in the SQL. Same table, same parity question.
    emitted |= {c.strip() for c in _PRED_COLS.split(",")}

    base = _base_pred_frame()
    for row in combine_calculated(base, _cfg(), pd.DataFrame()):
        row.update(run_id="r", ensemble_id="e", compute_engine="ensemble", quantiles=None)
        emitted |= set(row)
    for row in _apply_weights(base, {"theta": 0.6, "sarimax": 0.4}, "r", "nnls"):
        row["ensemble_id"] = "e"
        emitted |= set(row)
    return emitted


def _oof_producers() -> set[str]:
    """Every key any engine emits for ``backtest_oof``."""
    from scale_forecasting.engines.bigquery_engine import _oof_row
    from scale_forecasting.ensembler import _OOF_BLEND_COLS
    from scale_forecasting.registry.rows import assemble_ensemble_oof_rows

    emitted = set(assemble_oof_rows(_result())[0])
    emitted |= set(
        _oof_row(
            "r",
            "s",
            "arima_plus",
            3,
            {"forecast_date": "2026-02-01", "y_true": 9.0, "yhat": 8.5, "yhat_lower": 7.0},
        )
    )
    # The ensemble writes into this table too, and it is the only producer of `ensemble_id` here —
    # exactly the asymmetry this module exists for. Built from `_OOF_BLEND_COLS` so a column added
    # to the blend frame reaches the parity check without anyone remembering to widen this fixture.
    blend = pd.DataFrame([dict.fromkeys(_OOF_BLEND_COLS, 1)])
    emitted |= set(assemble_ensemble_oof_rows(blend, "r", "e")[0])
    return emitted


def _metadata_producers() -> set[str]:
    """Every key any engine emits for ``forecast_metadata`` — the three independent assemblers."""
    from scale_forecasting.engines.bigquery_engine import _meta_row
    from scale_forecasting.ensemble_run import _ensemble_meta_row
    from scale_forecasting.metrics import METRIC_NAMES

    panel = dict.fromkeys(METRIC_NAMES, 0.5)
    emitted = set(assemble_metadata_row(_result(), _CREATED, model_artifact="gs://b/a.pkl"))
    emitted |= set(_meta_row("r", "s", "arima_plus", panel, '{"horizon": 28}', _CREATED, _cfg()))
    emitted |= set(
        _ensemble_meta_row(
            run_id="r",
            ts_id="s",
            model_type="ensemble_nnls",
            panel=panel,
            ensemble_id="e",
            weights={"theta": 1.0},
            artifact_uri="gs://b/e.pkl",
            created_at=_CREATED,
            cfg=_cfg(),
            ensemble_scoring="holdout",
            backtest_refit="recondition",
        )
    )
    return emitted


# --- every spec column is either produced or explicitly reserved ---------------------------


def test_forecast_predictions_columns_all_have_a_producer() -> None:
    unproduced = _spec_columns(_PRED_SPEC) - _prediction_producers()
    assert unproduced == _RESERVED_PREDICTIONS, (
        f"forecast_predictions columns with no producer: {sorted(unproduced)}. Either wire one up "
        f"or add it to _RESERVED_PREDICTIONS with a note on which phase fills it."
    )


def test_backtest_oof_columns_all_have_a_producer() -> None:
    unproduced = _spec_columns(_OOF_SPEC) - _oof_producers()
    assert unproduced == _RESERVED_OOF, (
        f"backtest_oof columns with no producer: {sorted(unproduced)}."
    )


def test_forecast_metadata_columns_all_have_a_producer() -> None:
    unproduced = _spec_columns(_META_SPEC) - _metadata_producers()
    assert unproduced == _RESERVED_METADATA, (
        f"forecast_metadata columns with no producer: {sorted(unproduced)}."
    )


# --- and nothing any engine emits is silently dropped --------------------------------------


def test_no_engine_emits_a_prediction_key_the_spec_would_drop() -> None:
    dropped = _prediction_producers() - _spec_columns(_PRED_SPEC)
    assert not dropped, dropped


def test_no_engine_emits_an_oof_key_the_spec_would_drop() -> None:
    dropped = _oof_producers() - _spec_columns(_OOF_SPEC)
    assert not dropped, dropped


def test_no_engine_emits_a_metadata_key_the_spec_would_drop() -> None:
    dropped = _metadata_producers() - _spec_columns(_META_SPEC)
    assert not dropped, dropped


# --- the spec and the DDL are the same table ------------------------------------------------


def test_each_spec_matches_its_ddl_in_name_and_order() -> None:
    """The spec is what gets written; the DDL is what exists. A column in one and not the other
    is either a rejected append or a column nothing can ever fill.

    Order is asserted too, not just membership. The proto is matched by field *name*, so a
    mismatch is harmless at runtime — but the two lists are maintained by hand in two files, and
    order is the cheap signal that someone inserted a column in one place and appended it in the
    other. ``NOT NULL`` columns are absent from `additive_columns`, so they are excluded here.
    """
    not_null = {
        "forecast_predictions": ["run_id", "ts_id", "model_type", "forecast_date"],
        "backtest_oof": ["run_id", "ts_id", "model_type", "fold_id", "forecast_date"],
        "forecast_metadata": [
            "run_id",
            "ts_id",
            "model_type",
            "compute_engine",
            "model_hash",
            "created_at",
        ],
    }
    for table, spec in (
        ("forecast_predictions", _PRED_SPEC),
        ("backtest_oof", _OOF_SPEC),
        ("forecast_metadata", _META_SPEC),
    ):
        nullable_spec = [c for c, _t in spec if c not in set(not_null[table])]
        assert nullable_spec == [c for c, _t in additive_columns(table)], table


def test_every_reserved_column_is_nullable_in_the_ddl() -> None:
    """A reserved column reads NULL until its producer ships, so it cannot be ``NOT NULL`` — and
    it has to be reachable by ``ALTER TABLE ADD COLUMN``, which is what `additive_columns` lists,
    or an already-deployed registry never gets it."""
    reserved = {
        "forecast_predictions": _RESERVED_PREDICTIONS,
        "backtest_oof": _RESERVED_OOF,
        "forecast_metadata": _RESERVED_METADATA,
    }
    for table, names in reserved.items():
        additive = {c for c, _t in additive_columns(table)}
        assert names <= additive, (table, sorted(names - additive))
