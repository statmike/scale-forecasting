"""On-cluster entrypoint for the Dataproc Serverless forecast batches.

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file that it runs as ``__main__``
(no package context — relative imports would ``ImportError``), so the batch's *main file* is the
standalone `spark_main` shim (``src/spark_main.py``), which absolute-imports `main` here.
This module is the in-package entrypoint it calls — imported as a submodule from the
``python_file_uris`` zip (a zip of ``src/``, supplied at RUNTIME, never baked into the container
image — the same code-delivery pattern the seed job uses, see ``modules/seed``). It runs the one
Spark engine (`spark_explode`, the cross-join/explode strategy — there is no method switch); the
shared launcher core (`_entry.run_entry`) parses the on-cluster contract, loads the staged config,
and runs it.

Public surface: ``main(argv)``. ``pyspark`` and the engines import lazily (via the shared core) so
this file imports cleanly offline (parity with the seed entry / the engines).
"""

from __future__ import annotations

import argparse

from ._entry import parse_models, run_entry

# Re-exported so the CSV subset parser is exercised directly as the pure unit it is (tests).
_parse_models = parse_models


def _resolve_engine(ns: argparse.Namespace) -> tuple[object, str]:
    """Return the one Spark engine's ``run`` + its runtime label — a fixed single engine
    (`spark_explode`, the cross-join/explode strategy), no dispatch. Mirrors `ray_entry`."""
    from .engines import spark_explode

    return spark_explode.run, "spark"


def main(argv: list[str] | None = None) -> None:
    """Run the Spark forecast engine (`spark_explode`) via the shared launcher core."""
    run_entry(
        argv,
        prog="spark_entry",
        description="Run the Spark forecast engine.",
        resolve_engine=_resolve_engine,
    )


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
