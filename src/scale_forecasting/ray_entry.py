"""On-cluster entrypoint for the Ray-on-Vertex forecast run.

The Ray analog of `spark_entry`, but simpler: a Vertex Ray Job's entrypoint is a shell command
(``python -m scale_forecasting.ray_entry ...``) that runs **with** package context, so — unlike the
Dataproc ``main_python_file_uri`` (a bare ``__main__`` file needing the `spark_main` shim) —
this module *is* the entrypoint; there is no top-level shim. The current ``src/`` is delivered as
the job's ``runtime_env`` working dir (runtime code, never baked into the image — the same code runs
locally and in the cloud), so ``python -m scale_forecasting.ray_entry`` resolves against the code
that was just submitted.

Ray has a single engine, so this launcher fixes the resolver to `engines.ray_engine.run`; the shared
launcher core (`_entry.run_entry`) parses the on-cluster contract (``--config-uri`` + the executed
``--models`` subset + ``--manage-header`` + the ``--sf-*`` infra args), loads the staged config, and
dispatches — identical delivery to Spark, one local/cloud seam.

Public surface: ``main(argv)``. ``ray`` and the engine import lazily (via the shared core) so this
file imports cleanly offline (parity with `spark_entry`).
"""

from __future__ import annotations

import argparse

from ._entry import run_entry


def _resolve_engine(ns: argparse.Namespace) -> tuple[object, str]:
    """Return `engines.ray_engine.run` + the ``ray`` label (the single Ray engine)."""
    from .engines import ray_engine

    return ray_engine.run, "ray"


def main(argv: list[str] | None = None) -> None:
    """Dispatch to `engines.ray_engine.run` via the shared launcher core."""
    run_entry(
        argv,
        prog="ray_entry",
        description="Run the Ray forecast engine.",
        resolve_engine=_resolve_engine,
    )


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
