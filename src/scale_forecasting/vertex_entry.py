"""On-container Vertex AI ``CustomJob`` entrypoint.

The Vertex ``CustomJob`` analog of `ray_entry` and `spark_entry`. Invoked inside each ``CustomJob``
replica container (after `vertex_submit.VERTEX_BOOTSTRAP_CODE` downloads the content-addressed
``scale_forecasting-<code_hash>.zip`` from GCS and prepends it to ``sys.path``).

Shares the on-cluster CLI contract (``--config-uri``, ``--models``, ``--manage-header``, ``--sf-*``,
``--provisioned-hardware``) via `_entry.run_entry`, and additionally accepts optional
``--worker-rank`` and ``--worker-count`` flags (falling back to ``SF_WORKER_RANK``,
``JOB_COMPLETION_INDEX``, or Vertex AI's ``CLUSTER_SPEC``) so the same entrypoint runs unchanged on
single-VM ``CustomJob``s, multi-worker ``worker_pool_specs``, and GKE Indexed Jobs.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from functools import partial
from typing import TYPE_CHECKING

from ._entry import run_entry

if TYPE_CHECKING:
    from collections.abc import Callable


def _extract_worker_args(
    argv: list[str] | None,
) -> tuple[list[str] | None, int | None, int | None]:
    """Extract optional ``--worker-rank`` / ``--worker-count`` before calling `run_entry`."""
    if argv is None:
        import sys

        raw = list(sys.argv[1:])
    else:
        raw = list(argv)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--worker-rank", type=int, default=None)
    p.add_argument("--worker-count", type=int, default=None)
    known, remaining = p.parse_known_args(raw)
    return (
        remaining
        if (argv is not None or known.worker_rank is not None or known.worker_count is not None)
        else None,
        known.worker_rank,
        known.worker_count,
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch to `engines.vertex_engine.run` via the shared launcher core."""
    if not os.access(os.getcwd(), os.W_OK):
        os.chdir(tempfile.gettempdir())
    remaining, worker_rank, worker_count = _extract_worker_args(argv)

    def _resolve_engine(_ns: argparse.Namespace) -> tuple[Callable[..., object], str]:
        from .engines import vertex_engine

        if worker_rank is not None or worker_count is not None:
            return (
                partial(
                    vertex_engine.run,
                    worker_rank=worker_rank,
                    worker_count=worker_count,
                ),
                "vertex",
            )
        return vertex_engine.run, "vertex"

    run_entry(
        remaining,
        prog="vertex_entry",
        description="Run the Vertex AI CustomJob forecast engine.",
        resolve_engine=_resolve_engine,
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    main()
