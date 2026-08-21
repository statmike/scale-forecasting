"""Engine dispatch for the Dataproc Serverless forecast batches.

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file that it runs as ``__main__``
(no package context — relative imports would ``ImportError``), so the batch's *main file* is the
standalone `spark_main` shim (``src/spark_main.py``), which absolute-imports `main` here.
This module is the in-package dispatch logic it calls — imported as a submodule from the
``python_file_uris`` zip (a zip of ``src/``, supplied at RUNTIME, never baked into the container
image — the same code-delivery pattern the seed job uses, see ``modules/seed``). It selects the
Spark engine the parsed ``--engine`` names; the shared launcher core (`_entry.run_entry`) parses the
rest of the on-cluster contract, loads the staged config, and dispatches.

Public surface: ``main(argv)``. ``pyspark`` and the engines import lazily (via the shared core) so
this file imports cleanly offline (parity with the seed entry / the engines).
"""

from __future__ import annotations

import argparse

from ._entry import parse_models, run_entry

# The Spark-side engines this launcher can dispatch to (module under engines/, exposing run(cfg)).
_ENGINES: dict[str, str] = {
    "explode": "spark_explode",
}

# Re-exported so the CSV subset parser is exercised directly as the pure unit it is (tests).
_parse_models = parse_models


def _resolve_engine(ns: argparse.Namespace) -> tuple[object, str]:
    """Import the module named by ``--engine`` and return its ``run`` + the engine label."""
    import importlib

    module = importlib.import_module(f".engines.{_ENGINES[ns.engine]}", package=__package__)
    return module.run, ns.engine


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the ``--engine`` module's ``run(cfg, models=, manage_header=)`` (shared core)."""
    run_entry(
        argv,
        prog="spark_entry",
        description="Run a Spark forecast engine.",
        resolve_engine=_resolve_engine,
        engines=_ENGINES,
    )


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
