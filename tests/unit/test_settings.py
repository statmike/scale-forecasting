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


_ENV_KEYS = (
    "SF_PROJECT_ID",
    "SF_CONNECTION",
    "SF_WAREHOUSE_URI",
    "SF_DATASET_ID",
    "SF_REGISTRY_DATASET_ID",
    "SF_REGION",
)


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)


def _settings(**over: str) -> Settings:
    """A resolved `Settings` with the boilerplate filled in."""
    base = {
        "project_id": "p",
        "connection": "p.us-central1.c",
        "warehouse_uri": "gs://p-wh/warehouse",
        "dataset_id": "ds",
    }
    return Settings(**{**base, **over})  # type: ignore[arg-type]


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
    s = _settings()
    assert s.dataset_ref == "p.ds"
    assert s.table_ref("forecast_predictions") == "p.ds.forecast_predictions"


# --- registry identity ----------------------------------------------------------
#
# The registry's address is `project.registry_dataset` — a guaranteed-unique key, because BigQuery
# allows exactly one `run_registry` per dataset. Everything downstream (the GCS artifact root, the
# orphan sweep's scope) is derived from it, so these tests pin both the default (no override → the
# same dataset as the source, i.e. zero change for an existing deployment) and the split.


def test_registry_defaults_to_the_source_dataset() -> None:
    s = _settings()
    assert s.registry_dataset_id == "ds"
    assert s.registry_dataset_ref == s.dataset_ref
    assert s.registry_table_ref("run_registry") == s.table_ref("run_registry")


def test_registry_override_moves_only_the_registry() -> None:
    s = _settings(registry_dataset_id_override="reg")
    assert s.dataset_ref == "p.ds"  # source panel unmoved
    assert s.registry_dataset_ref == "p.reg"
    assert s.registry_table_ref("run_registry") == "p.reg.run_registry"
    assert s.table_ref("source_series_native") == "p.ds.source_series_native"


def test_artifact_root_is_keyed_by_the_registry_not_the_source() -> None:
    # Two registries can share one warehouse bucket without colliding, and the object path names
    # the dataset that owns the run — which is what makes an orphan sweep's scope unambiguous.
    assert _settings().artifact_root == "gs://p-wh/warehouse/artifacts/p/ds"
    split = _settings(registry_dataset_id_override="reg")
    assert split.artifact_root == "gs://p-wh/warehouse/artifacts/p/reg"
    # A trailing slash on the warehouse root must not double up in the path.
    assert _settings(warehouse_uri="gs://p-wh/warehouse/").artifact_root == (
        "gs://p-wh/warehouse/artifacts/p/ds"
    )


def test_resolve_reads_the_registry_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, {**_FULL_ENV, "SF_REGISTRY_DATASET_ID": "reg_ds"})
    s = Settings.resolve()
    assert s.dataset_id == "custom_ds"
    assert s.registry_dataset_id == "reg_ds"


def test_resolve_without_the_override_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _FULL_ENV)
    s = Settings.resolve()
    assert s.registry_dataset_id_override == ""
    assert s.registry_dataset_ref == s.dataset_ref


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
    # The stage emits no registry_dataset_id today; absent means "same dataset".
    assert s.registry_dataset_ref == s.dataset_ref
    split = Settings.from_terraform_outputs({**outputs, "registry_dataset_id": "sf_registry"})
    assert split.registry_dataset_ref == "tf-proj.sf_registry"


def test_from_terraform_outputs_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="warehouse_uri"):
        Settings.from_terraform_outputs({"project_id": "p", "iceberg_connection": "c"})
