"""Registry writers — run_registry / metadata / predictions / oof (CONTRACTS §3.4, §4).

Owned by BUILD steps 1.4 + B1. Public surface: ``ensure_tables``, ``write_header``,
``update_header``, ``write_cells``.
"""

from __future__ import annotations


def ensure_tables(cfg: object) -> None:  # pragma: no cover - stub, see BUILD 1.4/B1
    raise NotImplementedError("registry.bq.ensure_tables — BUILD step 1.4/B1")
