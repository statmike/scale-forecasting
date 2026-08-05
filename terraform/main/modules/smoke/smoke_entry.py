"""Dataproc Serverless PySpark entrypoint for the default-on smoke forecast.

Dataproc's ``main_python_file_uri`` must be a single ``gs://`` file that it runs as ``__main__``
(no package context — relative imports would ``ImportError``). The actual smoke logic lives in the
``scale_forecasting`` package, supplied at RUNTIME via ``python_file_uris`` (a zip of ``src/`` that
Terraform rebuilds + uploads each apply — never baked into the container image). This launcher is
the thin ``gs://`` shim that absolute-imports and calls ``main``; because the package comes from the
zip, the batch always runs current code. Keep it trivial — argv (``--config-uri`` + ``--sf-*`` infra
args) flows straight through to ``main``. Mirrors the seed job's ``seed_entry.py`` shim exactly.
"""

from scale_forecasting.smoke_run import main

if __name__ == "__main__":
    main()
