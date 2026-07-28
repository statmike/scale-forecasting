"""Dataproc Serverless PySpark entrypoint for the seed batch.

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file. The actual seed logic lives
in the ``scale_forecasting`` package, which the batch supplies at RUNTIME via ``python_file_uris``
(a zip of ``src/`` that Terraform rebuilds+uploads each apply — never baked into the container
image). This launcher is the thin ``gs://`` shim that imports and calls ``main``; because the
package comes from the zip, the batch always runs current code. Keep it trivial — argv (the seed
knobs) flows straight through to ``main``.
"""

from scale_forecasting.data_gen.seed_spark import main

if __name__ == "__main__":
    main()
