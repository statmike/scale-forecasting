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
_ENV_REGISTRY_DATASET = "SF_REGISTRY_DATASET_ID"
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
    # Where the RUN REGISTRY lives, when that is not where the source panel lives. Empty means
    # "same dataset" — which is every deployment that has not asked for otherwise, so this
    # defaults to zero behaviour change. See `registry_dataset_id`.
    #
    # Deliberately a Settings field and NOT part of RunConfig: this module's own docstring draws
    # the line — RunConfig is the portable run spec, Settings is deployment-specific. The same
    # experiment written to two different registries is the same experiment, so the registry
    # address must stay out of the run_id digest.
    registry_dataset_id_override: str = ""

    @property
    def dataset_ref(self) -> str:
        """``project.dataset`` — the form BigQuery table refs and DDL expect."""
        return f"{self.project_id}.{self.dataset_id}"

    def table_ref(self, table: str) -> str:
        """Fully-qualified ``project.dataset.table``."""
        return f"{self.dataset_ref}.{table}"

    @property
    def registry_dataset_id(self) -> str:
        """The dataset holding the run registry — ``dataset_id`` unless overridden."""
        return self.registry_dataset_id_override or self.dataset_id

    @property
    def registry_dataset_ref(self) -> str:
        """``project.dataset`` for the registry — **the registry's identity**.

        A registry *is* its dataset: the five table names are fixed, and BigQuery allows exactly one
        `run_registry` per dataset, so this string is a guaranteed-unique key with no validation
        code behind it. That is why it is safe to use verbatim as a GCS path segment
        (`artifact_root`) — no collision is possible between two registries.
        """
        return f"{self.project_id}.{self.registry_dataset_id}"

    def registry_table_ref(self, table: str) -> str:
        """Fully-qualified ref for a **registry** table (``run_registry``, ``run_jobs``, …).

        Use this for anything in `ddl.REGISTRY_TABLE_NAMES`; use `table_ref` for the source panel.
        They are the same string until a deployment sets ``SF_REGISTRY_DATASET_ID``, which is
        precisely why the distinction has to be made at every call site rather than discovered
        later — a miss is invisible until someone splits the datasets.
        """
        return f"{self.registry_dataset_ref}.{table}"

    @property
    def artifact_root(self) -> str:
        """GCS prefix owning every artifact of every run in **this** registry.

        ``<warehouse>/artifacts/<project_id>/<registry_dataset_id>``. The registry key is in the
        path for two reasons: two registries sharing a warehouse bucket can never collide, and an
        object path is self-describing — read it and you know which BigQuery dataset owns the run.
        That is what makes an orphan sweep well-defined: a prefix under this root whose ``run_id``
        has no ``run_registry`` row is garbage, and the scope of "this root" is unambiguous.

        ``project_id`` is included so a registry in project A with a warehouse in project B still
        keys cleanly.
        """
        root = self.warehouse_uri.rstrip("/")
        return f"{root}/artifacts/{self.project_id}/{self.registry_dataset_id}"

    @classmethod
    def resolve(cls) -> Settings:
        """Build ``Settings`` from the ``SF_*`` environment variables.

        ``SF_PROJECT_ID``, ``SF_CONNECTION``, and ``SF_WAREHOUSE_URI`` are required;
        ``SF_DATASET_ID`` and ``SF_REGION`` fall back to their defaults. Raises
        `ConfigError` naming the first missing required var, so a misconfigured
        deployment fails fast with a clear message instead of a downstream BigQuery 404.

        ``SF_REGISTRY_DATASET_ID`` is optional and falls back to ``SF_DATASET_ID``, so an existing
        deployment resolves to exactly the settings it resolved to before.
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
            registry_dataset_id_override=os.environ.get(_ENV_REGISTRY_DATASET) or "",
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
                registry_dataset_id_override=outputs.get("registry_dataset_id") or "",
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc
