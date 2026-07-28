"""Standalone ``gs://`` launcher shim for the Dataproc Serverless forecast batches.

Dataproc runs ``main_python_file_uri`` as ``__main__`` — a bare script with **no package context**,
so a file that uses relative imports (``from .foo import ...``) fails with ``ImportError: attempted
relative import with no known parent package``. The engine dispatch logic therefore lives in the
in-package module :mod:`scale_forecasting.spark_entry` (imported as a submodule from the
``python_file_uris`` zip, where its relative imports resolve normally); this file is the trivial
top-level shim the batch actually points at, doing only an **absolute** import + call.

This mirrors the seed job's ``seed_entry.py`` shim exactly. It sits at ``src/`` root (outside the
``scale_forecasting`` package, so it is never zipped as a submodule) and is uploaded on its own as
the batch's main file by :func:`scale_forecasting.submit._stage_code`.
"""

from scale_forecasting.spark_entry import main

if __name__ == "__main__":
    main()
