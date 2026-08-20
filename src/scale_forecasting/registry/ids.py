"""Deterministic identifiers for runs, cells, and jobs.

Four ids anchor the whole registry:

- ``run_id`` is derived from the validated config, so the *same* config yields the
  *same* run_id and any change yields a different one (queryable, reproducible).
- ``model_hash`` identifies one cell ``(run, ts_id, model)`` and is what makes writes
  idempotent: re-running a cell overwrites its rows instead of duplicating them.
- ``ensemble_id`` is derived from the ``EnsembleConfig`` alone, so several ensemble
  configurations can coexist under one ``run_id`` without their ``model_type`` pseudo-models
  colliding — a re-run with the *same* ensemble config lands the same id (idempotent), a
  different config lands a different one (distinctly keyed on the leaderboard).
- ``job_id`` (``job_key``) identifies one *job* in the run DAG: ``sf-<run_id>-<family>-a<attempt>``.
  It is a pure function of ``(run_id, family, attempt)``, so a family's job under a run has a name
  known before submission and reproducible after — the anchor that ties a platform job (a Dataproc
  batch, a Ray submission, a BigQuery script) back to exactly one run and family. The embedded
  run_id makes the backward ``job_id → run_id`` map strictly 1:1 and recoverable offline
  (``parse_job_key``); ``attempt`` (bumped by ``--force``) keeps a re-run's job distinct from the
  original under the same run.

All four are pure functions of their inputs — no clocks, no randomness — so ids are stable
across machines and reruns.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scale_forecasting.config import EnsembleConfig, RunConfig

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# The families that can own a job in the run DAG: every model family plus the downstream ensemble
# node. Mirrors ``config.JobFamily``; duplicated as a runtime tuple here (a ``Literal`` isn't
# iterable) so the key helpers can validate without importing config at runtime.
JOB_FAMILIES: tuple[str, ...] = ("statistical", "ml", "deep_learning", "native", "ensemble")

# A job id: ``sf-<run_id>-<family>-a<attempt>``. The family is anchored to a known member and the
# attempt to ``a<digits>`` at the very end, so the (variable-length, hyphen-bearing) run_id is
# recovered unambiguously by a greedy leading match — run_id always ends in ``-<12 hex>``, which
# never itself matches a trailing ``-<family>-a<n>`` (hex carries no hyphen).
_JOB_KEY_RE = re.compile(
    r"^sf-(?P<run_id>.+)-(?P<family>" + "|".join(JOB_FAMILIES) + r")-a(?P<attempt>\d+)$"
)


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


def make_ensemble_id(ensemble: EnsembleConfig) -> str:
    """Return a deterministic 12-hex id for one ``EnsembleConfig``.

    A pure digest of the ensemble configuration's content (strategies, prune threshold —
    ``sort_keys`` makes the bytes order-independent), so several ensemble configs can be scored
    under one ``run_id`` without their ``ensemble_<strategy>`` pseudo-models colliding: re-running
    the *same* config yields the same id (the dedupe key for idempotent re-runs), a *different*
    config yields a different one. Excludes ``enabled`` so toggling it doesn't re-key an otherwise
    identical config. Independent of the parent ``run_id`` — the same ensemble config keys
    identically across runs, which is what lets the standalone entrypoint re-ensemble any run.
    """
    payload = json.dumps(
        {"strategies": sorted(ensemble.strategies), "prune_threshold": ensemble.prune_threshold},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --- job identity --------------------------------------------------------------


def make_job_key(run_id: str, family: str, attempt: int = 1) -> str:
    """Return the deterministic id for a family's job under a run: ``sf-<run_id>-<family>-a<n>``.

    The one name a job carries across every layer — it is what a submitter hands the platform as
    the job's own id (a Dataproc ``batch_id``/``job_id``, a Ray ``submission_id``, a BigQuery
    script's parent job) so the platform job and the registry row share an identity, and what a
    trace query keys on. ``family`` must be a member of `JOB_FAMILIES`; ``attempt`` is 1-based and
    bumped by ``--force`` so a forced re-run's job is distinct from the original under the same run.
    """
    if family not in JOB_FAMILIES:
        raise ValueError(f"unknown job family {family!r}; expected one of {JOB_FAMILIES}")
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    return f"sf-{run_id}-{family}-a{attempt}"


def parse_job_key(job_id: str) -> tuple[str, str, int]:
    """Recover ``(run_id, family, attempt)`` from a job id — the backward, offline 1:1 map.

    Because ``job_id`` embeds the full ``run_id``, the ``job_id → run_id`` map is exact and needs
    no registry round-trip. Raises ``ValueError`` on a string that isn't a well-formed job id
    (wrong prefix, unknown family, or a missing/invalid attempt).
    """
    match = _JOB_KEY_RE.match(job_id)
    if match is None:
        raise ValueError(f"malformed job id {job_id!r}; expected sf-<run_id>-<family>-a<attempt>")
    return match["run_id"], match["family"], int(match["attempt"])


def decide_attempt(current_max: int | None, *, force: bool) -> tuple[int, bool]:
    """Resolve which ``attempt`` a submission should use, given the run/family's current max.

    The pure core of the re-run policy (the registry read that supplies ``current_max`` lives in
    ``registry.bq``): returns ``(attempt, is_new_job)``.

    - No prior job (``current_max`` is None) → ``(1, True)``: first attempt.
    - A prior job and not ``force`` → ``(current_max, False)``: reuse the existing job (an
      unforced re-run of an identical config is a no-op, not a new job).
    - ``force`` → ``(current_max + 1, True)``: a fresh attempt under the same run_id, distinctly
      keyed so it never collides with the original.
    """
    if current_max is None:
        return 1, True
    if not force:
        return current_max, False
    return current_max + 1, True
