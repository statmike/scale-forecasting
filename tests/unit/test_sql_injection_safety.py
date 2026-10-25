"""BigQuery SQL Identifier & Row-Restriction Injection Safety Audit (Track F.2 / v1.8.2).

Verifies that all user-supplied table paths, column names, covariate names, hierarchy levels,
and Storage Read API ``row_restriction`` boundaries reject or safely escape adversarial SQL
injection payloads before any BigQuery SQL statement is constructed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scale_forecasting.config import _BQ_COLUMN_RE, _BQ_TABLE_PATH_RE, RunConfig
from scale_forecasting.engines.vertex_engine import WorkerTopology, build_worker_series_range

SQL_INJECTION_COLUMN_PAYLOADS: tuple[str, ...] = (
    "y; DROP TABLE run_registry --",
    "y` FROM `other_proj.other_ds.secrets` --",
    "1' OR '1'='1",
    'y" OR "1"="1',
    "y -- comment",
    "y /* inline comment */",
    "y\nDROP TABLE x",
    "col-with-hyphen",
    "123_starts_with_digit",
    "col space",
    "col;SELECT",
    "",
)

SQL_INJECTION_TABLE_PAYLOADS: tuple[str, ...] = (
    "proj.ds.tbl; DROP TABLE proj.ds.run_registry --",
    "proj.ds.tbl` WHERE 1=1 UNION ALL SELECT * FROM `secrets",
    "proj.ds.tbl' OR '1'='1",
    "proj.ds.tbl --",
    "proj.ds.tbl /* comment */",
    "proj.ds.tbl.extra_fourth_part",
    "proj/ds/tbl",
    "",
)


@pytest.mark.parametrize("payload", SQL_INJECTION_COLUMN_PAYLOADS)
def test_bq_column_regex_rejects_sql_injection_payloads(payload: str) -> None:
    """_BQ_COLUMN_RE must reject any identifier containing SQL syntax, spaces, or quotes."""
    assert _BQ_COLUMN_RE.fullmatch(payload) is None


@pytest.mark.parametrize("payload", SQL_INJECTION_TABLE_PAYLOADS)
def test_bq_table_path_regex_rejects_sql_injection_payloads(payload: str) -> None:
    """_BQ_TABLE_PATH_RE must reject any table path containing SQL syntax, quotes, or >3 parts."""
    assert _BQ_TABLE_PATH_RE.fullmatch(payload) is None


@pytest.mark.parametrize("field_name", ["ts_id_col", "date_col", "target_col"])
@pytest.mark.parametrize("payload", SQL_INJECTION_COLUMN_PAYLOADS[:6])
def test_run_config_rejects_sql_injection_in_data_columns(field_name: str, payload: str) -> None:
    """DataConfig must reject SQL injection payloads in ts_id_col, date_col, and target_col."""
    data_cfg = {
        "source_table": "proj.ds.tbl",
        "ts_id_col": "ts_id",
        "date_col": "ds",
        "target_col": "y",
        "freq": "D",
        "horizon": 7,
        field_name: payload,
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"data": data_cfg, "models": ["naive_last"]})


@pytest.mark.parametrize("table_field", ["source_table", "future_covariates_table"])
@pytest.mark.parametrize("payload", SQL_INJECTION_TABLE_PAYLOADS[:6])
def test_run_config_rejects_sql_injection_in_table_paths(table_field: str, payload: str) -> None:
    """DataConfig must reject SQL injection payloads in source_table and future_covariates_table."""
    data_cfg: dict[str, object] = {
        "source_table": "proj.ds.tbl",
        "ts_id_col": "ts_id",
        "date_col": "ds",
        "target_col": "y",
        "freq": "D",
        "horizon": 7,
        table_field: payload,
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"data": data_cfg, "models": ["naive_last"]})


@pytest.mark.parametrize(
    "cov_field",
    ["exog", "future_covariates", "past_covariates", "static_covariates"],
)
def test_run_config_rejects_sql_injection_in_covariate_columns(cov_field: str) -> None:
    """FeaturesConfig must reject SQL injection strings in any covariate column list."""
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "data": {
                    "source_table": "proj.ds.tbl",
                    "ts_id_col": "ts_id",
                    "date_col": "ds",
                    "target_col": "y",
                    "freq": "D",
                    "horizon": 7,
                },
                "features": {
                    cov_field: ["price; DROP TABLE x --"],
                },
                "models": ["naive_last"],
            }
        )


def test_worker_series_range_escapes_adversarial_quotes_and_backslashes() -> None:
    """build_worker_series_range escapes quotes/backslashes in Storage Read API predicates."""
    series_ids = [
        "sku_01' OR '1'='1",
        "sku_02\\drop",
        "sku_03'; DROP TABLE x; --",
        "sku_04",
    ]
    restriction, worker_set, n_total = build_worker_series_range(
        series_ids,
        limit=None,
        id_col="ts_id",
        topology=WorkerTopology(rank=0, world_size=2, source="test"),
    )
    assert n_total == 4
    assert worker_set == {"sku_01' OR '1'='1", "sku_02\\drop"}
    assert restriction is not None
    # Unescaped "' OR '" must not appear; quotes and backslashes are escaped
    assert "sku_01\\' OR \\'1\\'=\\'1" in restriction
    assert "sku_02\\\\drop" in restriction
