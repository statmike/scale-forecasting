"""Reset the deployment's BigQuery tables — drop everything for a clean re-``ensure_tables`` (D19).

**Destructive.** This drops all six tables (the four native run-collection tables + both source
variants) and the two analyst views via :func:`registry.bq.drop_all`, so a subsequent run's
:func:`registry.bq.ensure_tables` recreates them in the current native/dual-format shape. The
Iceberg→native registry switch is a drop-and-recreate, not an ``ALTER``, which is why a reset seam
exists at all.

Two guards keep an accidental wipe from happening:

* The CLI **requires ``--yes``** to actually drop; without it the command prints what *would* be
  dropped (resolved dataset + table/view names) and exits without touching BigQuery.
* Identity comes from the ``SF_*`` environment via :class:`~scale_forecasting.settings.Settings`
  (G1) — the same seam every writer uses — so a reset can only ever hit the deployment the caller
  has explicitly pointed the environment at.

After a reset, reseed the source tables (``data_gen.seed_spark`` / the Terraform ``seed`` module).

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
    from .registry.ddl import render_drop_tables
    from .registry.views import render_create_views
    from .settings import Settings

    p = argparse.ArgumentParser(
        prog="reset",
        description="Drop all scale-forecasting tables + views for a clean recreate (destructive).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually drop; without it this only prints what would be dropped",
    )
    ns = p.parse_args(argv)

    settings = Settings.resolve()
    tables = list(render_drop_tables(settings.dataset_ref))
    views = list(render_create_views(settings.dataset_ref))

    if not ns.yes:
        _log.warning(
            "DRY RUN — would drop %d tables + %d views from %s: %s | %s. Re-run with --yes.",
            len(tables),
            len(views),
            settings.dataset_ref,
            ", ".join(tables),
            ", ".join(views),
        )
        return

    _log.warning(
        "resetting %s — dropping %d tables + %d views",
        settings.dataset_ref,
        len(tables),
        len(views),
    )
    bq.drop_all(settings=settings)
    _log.warning(
        "reset complete: %s is empty; a subsequent run will recreate the tables",
        settings.dataset_ref,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
