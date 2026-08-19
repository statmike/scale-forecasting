"""On-cluster entrypoint for the Ray-on-Vertex forecast run.

The Ray analog of `spark_entry`, but simpler: a Vertex Ray Job's entrypoint is a shell command
(``python -m scale_forecasting.ray_entry ...``) that runs **with** package context, so — unlike the
Dataproc ``main_python_file_uri`` (a bare ``__main__`` file needing the `spark_main` shim) —
this module *is* the entrypoint; there is no top-level shim. The current ``src/`` is delivered as
the job's ``runtime_env`` working dir (runtime code, never baked into the image — the same code runs
locally and in the cloud), so ``python -m scale_forecasting.ray_entry`` resolves against the code
that was just submitted.

Flow on the cluster head:

1. Parse ``--config-uri`` (the run config, staged to GCS as JSON by `ray_submit` — the JSON
   *is* the reproducibility record) + the ``--sf-*`` infra args, which
   `export_infra_env` promotes to ``os.environ`` before
   anything resolves `Settings` (one local/cloud seam, parity with Spark).
2. ``load_config`` the JSON into a validated, frozen `RunConfig`.
3. Call `scale_forecasting.engines.ray_engine.run` with the executed subset + header mode.

``--models`` / ``--manage-header`` carry the on-cluster contract exactly as they do for Spark
(see `main.run`): ``--models m1,m2`` restricts the executed subset (the staged config's
``run_id`` is unchanged, so both runtimes share it) and ``--manage-header false`` puts the engine in
contributor mode (``main`` owns the single shared header). Both are optional — absent, the engine
runs its standalone lifecycle over ``cfg.models``.

Public surface: ``main(argv)``. ``ray`` and the engine import lazily so this file imports cleanly
offline (parity with `spark_entry`).
"""

from __future__ import annotations

import argparse

from ._infra_args import add_infra_args, export_infra_env
from .errors import get_logger

_log = get_logger(__name__)


def _load_uri(uri: str) -> str:
    """Read a config file's text from a ``gs://`` URI (or a local path, for tests/local runs).

    Identical delivery to `spark_entry._load_uri` — the on-cluster driver fetches the staged
    config the same way regardless of runtime.
    """
    if uri.startswith("gs://"):
        from google.cloud import storage

        without_scheme = uri[len("gs://") :]
        bucket_name, _, blob_path = without_scheme.partition("/")
        blob = storage.Client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_text()
    from pathlib import Path

    return Path(uri).read_text()


def _parse_models(raw: str | None) -> list[str] | None:
    """Parse the optional ``--models m1,m2`` CSV into a subset list (``None`` → run ``cfg.models``).

    Empty/whitespace-only tokens are dropped so a trailing comma is harmless; an all-empty value
    collapses to ``None`` (standalone) rather than an empty subset (which would run nothing). Same
    semantics as `spark_entry._parse_models`.
    """
    if raw is None:
        return None
    names = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return names or None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ray_entry", description="Run the Ray forecast engine.")
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
    ns = p.parse_args(argv)
    export_infra_env(ns)
    return ns


def main(argv: list[str] | None = None) -> None:
    """Load the config and dispatch to `engines.ray_engine.run`."""
    import json
    import tempfile
    from pathlib import Path

    from .config import load_config
    from .engines import ray_engine

    ns = _parse_args(argv)
    models = _parse_models(ns.models)
    manage_header = ns.manage_header == "true"
    _log.info(
        "ray_entry: config_uri=%s models=%s manage_header=%s",
        ns.config_uri,
        models,
        manage_header,
    )

    # load_config takes a path; materialize a gs:// config to a temp file (local path unchanged).
    raw = _load_uri(ns.config_uri)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "run_config.json"
        cfg_path.write_text(raw)
        # Re-serialize via json to fail fast on malformed JSON with a clear message before load.
        json.loads(raw)
        cfg = load_config(cfg_path)

    ray_engine.run(cfg, models=models, manage_header=manage_header)


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
