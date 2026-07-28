"""Engine dispatch for the Dataproc Serverless forecast batches (BUILD B2).

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file that it runs as ``__main__``
(no package context — relative imports would ``ImportError``), so the batch's *main file* is the
standalone :mod:`spark_main` shim (``src/spark_main.py``), which absolute-imports :func:`main` here.
This module is the in-package dispatch logic it calls — imported as a submodule from the
``python_file_uris`` zip (a zip of ``src/``, supplied at RUNTIME, never baked into the container
image — the same code-delivery pattern the seed job uses, see ``modules/seed``). It loads the run
config, restores the infra identity, and dispatches to the requested engine.

Flow on the driver:

1. Parse ``--engine`` (which Spark method to run) + ``--config-uri`` (the run config, staged to GCS
   as JSON by the submit helper — the JSON *is* the reproducibility record) + the ``--sf-*`` infra
   args, which :func:`~scale_forecasting._infra_args.export_infra_env` promotes to ``os.environ``
   before anything resolves ``Settings`` (Dataproc rejects driver-env, so args are the delivery
   path — see :mod:`._infra_args`).
2. ``load_config`` the JSON into a validated, frozen :class:`~scale_forecasting.config.RunConfig`.
3. Call the engine's ``run(cfg)``.

Public surface: ``main(argv)``. ``pyspark`` and the engines import lazily so this file imports
cleanly offline (parity with the seed entry / the engines).
"""

from __future__ import annotations

import argparse

from ._infra_args import add_infra_args, export_infra_env
from .errors import get_logger

_log = get_logger(__name__)

# The Spark-side engines this launcher can dispatch to (module under engines/, exposing run(cfg)).
# 'multi' is intentionally absent: it is orchestrated from the submit helper (which fans out child
# 'explode' batches), not run on-cluster — google-cloud-dataproc isn't in the runtime container.
_ENGINES: dict[str, str] = {
    "explode": "spark_explode",
    "naive": "spark_naive",
}


def _load_uri(uri: str) -> str:
    """Read a config file's text from a ``gs://`` URI (or a local path, for tests/local runs)."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        without_scheme = uri[len("gs://") :]
        bucket_name, _, blob_path = without_scheme.partition("/")
        blob = storage.Client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_text()
    from pathlib import Path

    return Path(uri).read_text()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="spark_entry", description="Run a Spark forecast engine.")
    p.add_argument("--engine", required=True, choices=sorted(_ENGINES))
    p.add_argument("--config-uri", required=True, help="gs:// (or local) path to run config JSON")
    add_infra_args(p)
    ns = p.parse_args(argv)
    export_infra_env(ns)
    return ns


def main(argv: list[str] | None = None) -> None:
    """Load the config and dispatch to the requested engine's ``run(cfg)``."""
    import importlib
    import json
    import tempfile
    from pathlib import Path

    from .config import load_config

    ns = _parse_args(argv)
    _log.info("spark_entry: engine=%s config_uri=%s", ns.engine, ns.config_uri)

    # load_config takes a path; materialize a gs:// config to a temp file (local path unchanged).
    raw = _load_uri(ns.config_uri)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "run_config.json"
        cfg_path.write_text(raw)
        # Re-serialize via json to fail fast on malformed JSON with a clear message before load.
        json.loads(raw)
        cfg = load_config(cfg_path)

    module = importlib.import_module(f".engines.{_ENGINES[ns.engine]}", package=__package__)
    module.run(cfg)


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
