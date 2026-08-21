"""Default-on smoke-forecast entrypoint — the "does it forecast?" first-apply proof.

After a fresh ``terraform apply`` builds the runtime image and seeds the example dataset, the smoke
runs a tiny end-to-end forecast so the very first apply also *proves the platform forecasts* — a
handful of fast Python models on Dataproc Serverless **in parallel with** ``arima_plus`` in
BigQuery, all under **one shared ``run_id``** (the single-run, two-engine showpiece, in miniature).

**How it runs both engines from one Dataproc batch (no image change).** This is an ordinary Dataproc
Serverless PySpark batch — the same code-delivery pattern as the seed (``python_file_uris`` zip of
``src/``, launched by a thin ``gs://`` shim). It calls `scale_forecasting.main.run` with the
batch's own `SparkSession` **injected**: on that path (``main.run(cfg, spark=session)`` with
``python_runtime="spark"``) the Spark engine runs **in-process against the injected session**
(``spark_explode``) and the BigQuery engine runs inline on the main thread — both
under one header. Critically, that branch imports **neither** ``.submit`` nor ``google-cloud-
dataproc`` (the ``[spark]`` extra that is *not* in the runtime image), and the BigQuery client *is*
in the image (the seed uses it). So the smoke needs no extra deps and no second batch.

**Tolerance.** The smoke is deliberately *non-blocking* at the Terraform layer: the module submits
it with ``gcloud dataproc batches submit ... --async``-style ``local-exec`` (``on_failure =
continue``), so a smoke failure never fails the apply — the operator inspects the surfaced batch id.
This module itself just runs the forecast and lets any error propagate (the batch goes FAILED); the
tolerance lives in the caller, not here.

**Infra identity** is resolved from the ``SF_*`` environment via `Settings`. Dataproc
Serverless rejects driver-env Spark properties, so the batch passes identity as ``--sf-*``
job args, which `main` exports into ``os.environ`` before anything resolves ``Settings`` —
keeping env-based resolution the single local/cloud seam (parity with the seed /
``spark_entry``).

Public surface: ``main(argv)``. ``pyspark`` and `main` import lazily inside `main`, so
this module imports cleanly offline (parity with ``seed_spark`` / ``spark_entry``). It does **not**
duplicate ``main.run``'s orchestration — it stages a session and calls it.
"""

from __future__ import annotations

import argparse

from ._infra_args import add_infra_args, export_infra_env
from .errors import get_logger

_log = get_logger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="smoke_run", description="Run the default-on smoke forecast.")
    p.add_argument("--config-uri", required=True, help="gs:// (or local) path to run config JSON")
    add_infra_args(p)
    ns = p.parse_args(argv)
    export_infra_env(ns)
    return ns


def main(argv: list[str] | None = None) -> None:
    """Load the staged config, inject the batch's Spark session, and run one forecast.

    The batch's `SparkSession` is passed straight into `scale_forecasting.main.run`, so
    the Spark models run in-process against it while the BigQuery engine runs inline — both under
    one ``run_id`` — with no remote-batch submit and no ``[spark]`` extra (the injectable-session
    seam).
    """
    from pyspark.sql import SparkSession

    from .config import load_config_uri
    from .main import run

    ns = _parse_args(argv)
    _log.info("smoke_run: config_uri=%s", ns.config_uri)

    # load_config_uri reads a gs:// URI directly (or a local path), returning a validated RunConfig.
    cfg = load_config_uri(ns.config_uri)

    spark = SparkSession.builder.appName("scale-forecasting-smoke").getOrCreate()
    try:
        run_id = run(cfg, spark=spark)
        _log.info("smoke complete: run_id=%s", run_id)
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
