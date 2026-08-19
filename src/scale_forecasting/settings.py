"""Infrastructure settings — where the run's GCP identity is resolved.

This is the one seam that separates the *run spec* (`RunConfig`
— models, backtest, ensembling) from the *infrastructure* it runs against (project, dataset,
BigLake connection, warehouse bucket). ``RunConfig`` is portable and reproducible; ``Settings``
is deployment-specific, so they are deliberately kept apart.

Resolution is **environment-based** (``SF_*`` vars), for one reason: the same code must run
locally and in the cloud — the identical writer code runs locally under ADC and on Composer
under the runner service account. Composer sets
env vars on the workers; a local shell (or the ``@gcp`` test) sets the same vars. No config file,
no hardcoded ids, no ``terraform output`` call baked into the product — the values come from
whoever launched the process. `Settings.from_terraform_outputs` is a convenience for local
dev/tests that reads the exact keys ``terraform output -json`` emits.

Public surface: `Settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError

# Env var names, kept together so the docstring, resolve(), and any tooling agree.
_ENV_PROJECT = "SF_PROJECT_ID"
_ENV_DATASET = "SF_DATASET_ID"
_ENV_CONNECTION = "SF_CONNECTION"
_ENV_WAREHOUSE = "SF_WAREHOUSE_URI"
_ENV_REGION = "SF_REGION"

_DEFAULT_DATASET = "scale_forecasting"
_DEFAULT_REGION = "us-central1"


@dataclass(frozen=True)
class Settings:
    """Resolved GCP infrastructure identity for one process.

    Frozen: resolved once at the entrypoint and passed down (or re-resolved from the same
    env) so every writer in a run targets the same project/dataset.
    """

    project_id: str
    # BigLake connection + warehouse root stay REQUIRED even though the run-collection tables are
    # native: the example input still ships a managed-Iceberg variant (source_series_iceberg)
    # that reads/writes its GCS files through the connection. Source storage format is chosen per
    # run via cfg.data.source_table (…_iceberg vs …_native) — not a Settings knob (config-driven).
    connection: str  # BigLake connection ref, "project.region.name"
    warehouse_uri: str  # GCS warehouse root, "gs://<bucket>/warehouse"
    dataset_id: str = _DEFAULT_DATASET
    region: str = _DEFAULT_REGION

    @property
    def dataset_ref(self) -> str:
        """``project.dataset`` — the form BigQuery table refs and DDL expect."""
        return f"{self.project_id}.{self.dataset_id}"

    def table_ref(self, table: str) -> str:
        """Fully-qualified ``project.dataset.table``."""
        return f"{self.dataset_ref}.{table}"

    @classmethod
    def resolve(cls) -> Settings:
        """Build ``Settings`` from the ``SF_*`` environment variables.

        ``SF_PROJECT_ID``, ``SF_CONNECTION``, and ``SF_WAREHOUSE_URI`` are required;
        ``SF_DATASET_ID`` and ``SF_REGION`` fall back to their defaults. Raises
        `ConfigError` naming the first missing required var, so a misconfigured
        deployment fails fast with a clear message instead of a downstream BigQuery 404.
        """
        required = {
            "project_id": _ENV_PROJECT,
            "connection": _ENV_CONNECTION,
            "warehouse_uri": _ENV_WAREHOUSE,
        }
        values: dict[str, str] = {}
        for field_name, env_name in required.items():
            raw = os.environ.get(env_name)
            if not raw:
                raise ConfigError(
                    f"missing required environment variable {env_name} "
                    f"(set it, or use Settings.from_terraform_outputs for local dev)"
                )
            values[field_name] = raw
        return cls(
            project_id=values["project_id"],
            connection=values["connection"],
            warehouse_uri=values["warehouse_uri"],
            dataset_id=os.environ.get(_ENV_DATASET) or _DEFAULT_DATASET,
            region=os.environ.get(_ENV_REGION) or _DEFAULT_REGION,
        )

    @classmethod
    def from_terraform_outputs(cls, outputs: dict[str, str]) -> Settings:
        """Build ``Settings`` from a ``terraform output -json`` value map (local dev/tests).

        Accepts the keys the ``terraform/main`` stage emits — ``project_id``, ``dataset_id``,
        ``iceberg_connection``, ``warehouse_uri`` (with ``connection`` accepted as an alias) —
        so a developer can wire the live infra without hand-copying ids. Missing keys raise
        `ConfigError`.
        """
        try:
            connection = outputs.get("iceberg_connection") or outputs["connection"]
            return cls(
                project_id=outputs["project_id"],
                connection=connection,
                warehouse_uri=outputs["warehouse_uri"],
                dataset_id=outputs.get("dataset_id") or _DEFAULT_DATASET,
                region=outputs.get("region") or _DEFAULT_REGION,
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc
