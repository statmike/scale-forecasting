"""Reaching the deployment — resolve the target dataset, and make sure it exists.

The infra identity (project / dataset / connection / warehouse) is not on ``RunConfig`` — it is
resolved from the environment via `Settings`, so the identical writer code runs locally under ADC
and on Composer under the runner SA. Every writer and reader in this package takes an optional
``settings=`` for callers that already hold one and otherwise resolves from ``SF_*`` env vars
through `_resolve_settings` here. `ensure_tables` / `ensure_views` render and execute the
deployment DDL; both are idempotent, so setup can run on every deploy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..settings import Settings


def _resolve_settings(settings: Settings | None) -> Settings:
    """Return the passed settings, or resolve from the ``SF_*`` environment."""
    if settings is not None:
        return settings
    from ..settings import Settings as _Settings

    return _Settings.resolve()


def ensure_tables(
    cfg: RunConfig | None = None, *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create every registry + source table if absent (idempotent DDL).

    Renders the deployment DDL for the resolved dataset — native registry (native ``JSON``
    columns) plus both source variants, ``source_series_iceberg`` (managed Iceberg) and
    ``source_series_native`` (plain) — and executes each statement. ``cfg`` is accepted for
    signature symmetry with the other writers but is unused — the schema is fixed, not
    config-driven. Raises `RegistryError` on a DDL failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .ddl import (
        REGISTRY_TABLE_NAMES,
        SOURCE_TABLE_NAMES,
        render_deployment_ddl,
        render_migrations,
    )

    resolved = _resolve_settings(settings)
    # Registry tables land in the registry dataset, source tables in the source dataset. These are
    # the same dataset unless SF_REGISTRY_DATASET_ID says otherwise, so this is a no-op for an
    # existing deployment — but the two families are now addressed separately all the way down.
    ddl = render_deployment_ddl(
        resolved.registry_dataset_ref,
        connection=resolved.connection,
        warehouse_uri=resolved.warehouse_uri,
        source_dataset=resolved.dataset_ref,
    )
    client = bigquery.Client(project=resolved.project_id)
    for name, statement in ddl.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"ensure_tables failed creating {name}: {exc}") from exc

    # Additive schema evolution: bring tables created under an older schema up to the current
    # column set (ADD COLUMN IF NOT EXISTS). A fresh CREATE already has every column, so these
    # ALTERs are no-ops on it; on a pre-existing table they back-fill new nullable columns.
    # Each family is migrated against the dataset that holds it.
    migrations = {
        **render_migrations(resolved.registry_dataset_ref, tables=REGISTRY_TABLE_NAMES),
        **render_migrations(resolved.dataset_ref, tables=SOURCE_TABLE_NAMES),
    }
    for name, statement in migrations.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"ensure_tables failed migrating {name}: {exc}") from exc

    # Curated analyst views sit on top of the tables — create them in the same setup pass so the
    # reviewable read surface (v_run_summary / v_model_leaderboard) exists after any run.
    ensure_views(settings=resolved)


def ensure_views(
    *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create/replace the analyst views over the registry (idempotent).

    Renders the ``CREATE OR REPLACE VIEW`` statements (`registry.views.render_create_views`)
    for the resolved dataset and executes each. Called by `ensure_tables`; safe to call on its
    own to refresh view definitions after a change. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .views import render_create_views

    resolved = _resolve_settings(settings)
    views = render_create_views(resolved.registry_dataset_ref)
    client = bigquery.Client(project=resolved.project_id)
    for name, statement in views.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with view context
            raise RegistryError(f"ensure_views failed creating {name}: {exc}") from exc


# There is deliberately no `drop_all` here, and no CLI that calls one. Dropping a registry
# wholesale is `bq rm -r -f <project>:<dataset>` (or a handful of `bq rm -f -t` when the registry
# shares a dataset with the source panel) — a one-liner nobody needs us to wrap, and wrapping it
# invites the accident. What the product does ship is the *scoped* teardown: `registry.ops.drop_run`
# and `registry.ops.sweep_orphans`, which delete GCS artifacts before the rows that index them.
# The pure renderer `ddl.render_drop_tables` stays available for callers that want the statements.
