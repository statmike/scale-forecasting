"""Reset the deployment's BigQuery tables — drop everything for a clean re-``ensure_tables``.

**Destructive.** This drops the five **registry** tables and the two analyst views via
`registry.bq.drop_all`, so a subsequent run's `registry.bq.ensure_tables` recreates them in the
current shape. The Iceberg→native registry switch is a drop-and-recreate, not an ``ALTER``, which is
why a reset seam exists at all.

The two **source** tables are deliberately out of scope. Clearing a registry and rebuilding the
input panel are different operations with different costs — reseeding is a Spark job over millions
of rows — and a reset that silently took the source with it is a wipe nobody asked for. Use
``data_gen.seed_spark`` (or the Terraform ``seed`` module) to rebuild source data.

Two guards keep an accidental wipe from happening:

* The CLI **requires ``--yes``** to actually drop; without it the command prints what *would* be
  dropped (resolved dataset + table/view names) and exits without touching BigQuery.
* Identity comes from the ``SF_*`` environment via `Settings`
  — the same seam every writer uses — so a reset can only ever hit the deployment the caller
  has explicitly pointed the environment at.

Public surface: ``main(argv)``.
"""

from __future__ import annotations

import argparse

from .errors import get_logger

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.reset --yes`` — drop all tables + views (destructive).

    Without ``--yes`` this is a dry run: it resolves the target deployment and prints the objects
    that *would* be dropped, then exits without calling BigQuery. Pass ``--yes`` to execute.
    """
    from .registry import bq
    from .registry.ddl import REGISTRY_TABLE_NAMES, render_drop_tables
    from .registry.views import render_create_views
    from .settings import Settings

    p = argparse.ArgumentParser(
        prog="reset",
        description="Drop the scale-forecasting registry tables + views for a clean recreate "
        "(destructive; leaves the source tables alone).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually drop; without it this only prints what would be dropped",
    )
    ns = p.parse_args(argv)

    settings = Settings.resolve()
    target = settings.registry_dataset_ref
    tables = list(render_drop_tables(target, tables=REGISTRY_TABLE_NAMES))
    views = list(render_create_views(target))

    if not ns.yes:
        _log.warning(
            "DRY RUN — would drop %d tables + %d views from %s: %s | %s. Re-run with --yes.",
            len(tables),
            len(views),
            target,
            ", ".join(tables),
            ", ".join(views),
        )
        return

    _log.warning(
        "resetting %s — dropping %d tables + %d views",
        target,
        len(tables),
        len(views),
    )
    bq.drop_all(settings=settings)
    _log.warning(
        "reset complete: %s has no registry tables; a subsequent run recreates them",
        target,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
