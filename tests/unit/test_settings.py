"""Tests for the infra-settings seam (``settings.py``).

Offline: env-var resolution, the fail-fast error on a missing required var, defaults, and
the terraform-outputs convenience mapping. No GCP client is constructed here.
"""

from __future__ import annotations

import pytest

from scale_forecasting.errors import ConfigError
from scale_forecasting.settings import Settings

_FULL_ENV = {
    "SF_PROJECT_ID": "my-proj",
    "SF_CONNECTION": "my-proj.us-central1.sf-iceberg",
    "SF_WAREHOUSE_URI": "gs://my-proj-warehouse/warehouse",
    "SF_DATASET_ID": "custom_ds",
    "SF_REGION": "europe-west4",
}


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key in ("SF_PROJECT_ID", "SF_CONNECTION", "SF_WAREHOUSE_URI", "SF_DATASET_ID", "SF_REGION"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)


def test_resolve_reads_all_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _FULL_ENV)
    s = Settings.resolve()
    assert s.project_id == "my-proj"
    assert s.connection == "my-proj.us-central1.sf-iceberg"
    assert s.warehouse_uri == "gs://my-proj-warehouse/warehouse"
    assert s.dataset_id == "custom_ds"
    assert s.region == "europe-west4"


def test_resolve_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(
        monkeypatch,
        {
            "SF_PROJECT_ID": "p",
            "SF_CONNECTION": "p.us-central1.c",
            "SF_WAREHOUSE_URI": "gs://p-wh/warehouse",
        },
    )
    s = Settings.resolve()
    assert s.dataset_id == "scale_forecasting"
    assert s.region == "us-central1"


@pytest.mark.parametrize("missing", ["SF_PROJECT_ID", "SF_CONNECTION", "SF_WAREHOUSE_URI"])
def test_resolve_fails_fast_on_missing_required(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    env = {k: v for k, v in _FULL_ENV.items() if k != missing}
    _set_env(monkeypatch, env)
    with pytest.raises(ConfigError, match=missing):
        Settings.resolve()


def test_dataset_ref_and_table_ref() -> None:
    s = Settings(
        project_id="p",
        connection="p.us-central1.c",
        warehouse_uri="gs://p-wh/warehouse",
        dataset_id="ds",
    )
    assert s.dataset_ref == "p.ds"
    assert s.table_ref("forecast_predictions") == "p.ds.forecast_predictions"


def test_from_terraform_outputs_maps_keys() -> None:
    outputs = {
        "project_id": "tf-proj",
        "dataset_id": "scale_forecasting",
        "iceberg_connection": "tf-proj.us-central1.sf-iceberg",
        "warehouse_uri": "gs://tf-proj-warehouse/warehouse",
    }
    s = Settings.from_terraform_outputs(outputs)
    assert s.project_id == "tf-proj"
    assert s.connection == "tf-proj.us-central1.sf-iceberg"
    assert s.warehouse_uri == "gs://tf-proj-warehouse/warehouse"


def test_from_terraform_outputs_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="warehouse_uri"):
        Settings.from_terraform_outputs({"project_id": "p", "iceberg_connection": "c"})
