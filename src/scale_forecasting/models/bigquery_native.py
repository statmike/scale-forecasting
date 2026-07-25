"""BigQuery-native models — arima_plus / arima_plus_xreg / timesfm (CONTRACTS §1, §5).

These register through the *same* factory (runtime="bigquery") so the router and
registry treat them uniformly, but their ``fit``/``predict`` raise ``NotImplementedError``
— they are executed as SQL by ``engines/bigquery_engine.py``, not by Python.

Owned by BUILD step 2.5 (metadata-only in Arc A) + B3 (execution).
"""

from __future__ import annotations
