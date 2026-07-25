"""scale-forecasting — massively-parallel time-series forecasting on Google Cloud.

The package is deliberately flat and readable: one capability per file. Start with
``DESIGN`` in the README, then read ``config.py`` (the run contract), ``worker.py``
(the unit of work), and one file under ``models/`` to see the whole shape.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
