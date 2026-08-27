"""Shared on-cluster entrypoint core for the Dataproc and Ray forecast launchers.

Both `spark_entry` and `ray_entry` parse the same on-cluster contract — ``--config-uri`` (the run
config, staged to GCS as JSON — the JSON *is* the reproducibility record) + the executed
``--models`` subset + ``--manage-header`` + the ``--sf-*`` infra args, which `export_infra_env`
promotes to ``os.environ`` before anything resolves `Settings` — then load the staged config and
run the runtime's engine ``run(cfg, models=, manage_header=)``. This module is that shared skeleton,
parametrized by a resolver so each launcher differs only in which engine callable it selects (Spark
and Ray each have exactly one — there is no method switch).

``--models m1,m2`` restricts the executed subset (the staged config's ``run_id`` is unchanged, so
both runtimes share it) and ``--manage-header false`` puts the engine in contributor mode (``main``
owns the single shared header). Both are optional — absent, the engine runs its standalone
lifecycle over ``cfg.models``.

The GCP/engine imports stay lazy so the launchers import cleanly offline.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._infra_args import add_infra_args, export_infra_env
from .errors import get_logger

if TYPE_CHECKING:
    from .config import RunConfig

_log = get_logger(__name__)

# A resolved (engine_run, label) pair: the callable to run and a short name for logging. The
# callable takes ``(cfg, models=, manage_header=)`` — the shared engine contract.
EngineResolver = Callable[[argparse.Namespace], "tuple[Callable[..., object], str]"]


def parse_models(raw: str | None) -> list[str] | None:
    """Parse the optional ``--models m1,m2`` CSV into a subset list (``None`` → run ``cfg.models``).

    Empty/whitespace-only tokens are dropped so a trailing comma is harmless; an all-empty value
    collapses to ``None`` (standalone) rather than an empty subset (which would run nothing).
    """
    if raw is None:
        return None
    names = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return names or None


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """The shared launcher parser: ``--config-uri`` + ``--models`` + ``--manage-header`` + infra."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--config-uri", required=True, help="gs:// (or local) path to run config JSON")
    p.add_argument(
        "--models",
        default=None,
        help="optional comma-separated executed subset; absent runs all cfg.models",
    )
    p.add_argument(
        "--manage-header",
        default="true",
        choices=("true", "false"),
        help="false = contributor mode; main.run owns the shared header",
    )
    add_infra_args(p)
    return p


def run_entry(
    argv: list[str] | None,
    *,
    prog: str,
    description: str,
    resolve_engine: EngineResolver,
) -> None:
    """Parse the shared launcher args, load the staged config, and run the resolved engine.

    The one on-cluster driver skeleton for both runtimes: parse → export infra env → load the
    staged config (`load_config_uri`, which reads a ``gs://`` URI directly or a local path) →
    ``resolve_engine(ns)`` picks the engine callable → run it with the parsed subset + header mode.
    """
    from .config import load_config_uri

    p = build_parser(prog, description)
    ns = p.parse_args(argv)
    export_infra_env(ns)
    models = parse_models(ns.models)
    manage_header = ns.manage_header == "true"
    engine_run, label = resolve_engine(ns)
    _log.info(
        "%s: runtime=%s config_uri=%s models=%s manage_header=%s",
        prog,
        label,
        ns.config_uri,
        models,
        manage_header,
    )
    cfg: RunConfig = load_config_uri(ns.config_uri)
    engine_run(cfg, models=models, manage_header=manage_header)
