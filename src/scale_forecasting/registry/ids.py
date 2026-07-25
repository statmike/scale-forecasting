"""Deterministic identifiers for runs and cells (CONTRACTS §3).

Two ids anchor the whole registry:

- ``run_id`` is derived from the validated config, so the *same* config yields the
  *same* run_id and any change yields a different one (queryable, reproducible — G3).
- ``model_hash`` identifies one cell ``(run, ts_id, model)`` and is what makes writes
  idempotent: re-running a cell overwrites its rows instead of duplicating them
  (CONTRACTS §3.4).

Both are pure functions of their inputs — no clocks, no randomness — so ids are stable
across machines and reruns.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scale_forecasting.config import RunConfig

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Lowercase, hyphenate, and trim a string into a readable id fragment."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _canonical_config(cfg: RunConfig) -> str:
    """Serialize a config to a stable, order-independent JSON string.

    ``sort_keys`` makes the bytes insensitive to field ordering, so the hash depends
    only on the config's *content*.
    """
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def make_run_id(cfg: RunConfig) -> str:
    """Return a deterministic run id: ``<run_name-slug>-<12-hex config digest>``.

    Stable for identical configs, different for any content change. The human-readable
    prefix keeps the registry browsable; the digest guarantees uniqueness.
    """
    digest = hashlib.sha256(_canonical_config(cfg).encode()).hexdigest()[:12]
    slug = _slug(cfg.run_name) or "run"
    return f"{slug}-{digest}"


def make_model_hash(run_id: str, ts_id: str, model_type: str, cfg: RunConfig) -> str:
    """Return the deterministic hash identifying one ``(run, ts_id, model)`` cell.

    Mixes the run_id, the series id, the model name, and the canonical config so a
    cell's identity is fully reproducible from its own inputs (not merely trusted to
    match run_id's derivation). This is the idempotency key for registry writes.
    """
    payload = "\x1f".join([run_id, ts_id, model_type, _canonical_config(cfg)])
    return hashlib.sha256(payload.encode()).hexdigest()
