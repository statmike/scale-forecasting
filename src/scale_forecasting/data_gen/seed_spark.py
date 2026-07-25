"""PySpark serverless seed entrypoint (Arc B) — CONTRACTS §6, DESIGN §13.1-seed.

Partitions the series-id range across executors, calls the pure ``generator``, and
writes the managed Iceberg ``source_series`` table via the Storage Write API. This
seed job dogfoods the platform's own Spark write path.

Owned by BUILD step B0.3/B0.4. Public surface: ``main(argv)``.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - stub, see BUILD B0.3
    raise NotImplementedError("data_gen.seed_spark.main — BUILD step B0.3/B0.4")
