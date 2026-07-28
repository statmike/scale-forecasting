"""Dataproc Serverless PySpark entrypoint for the seed batch.

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file. The actual seed logic lives
in the ``scale_forecasting`` package baked into the container image; this launcher is the thin
``gs://`` shim that Terraform uploads to the code bucket and hands to the batch. Keep it trivial —
argv (the seed knobs) flows straight through to ``main``.
"""

from scale_forecasting.data_gen.seed_spark import main

if __name__ == "__main__":
    main()
