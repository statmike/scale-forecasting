"""Load, validate, and freeze the run config (CONTRACTS §6, DESIGN §9).

Owned by BUILD step 1.1. Public surface: ``RunConfig`` (pydantic), ``load_config``,
``estimate_fanout``.
"""

from __future__ import annotations


def load_config(path: str) -> object:  # pragma: no cover - stub, see BUILD 1.1
    raise NotImplementedError("config.load_config — BUILD step 1.1")
