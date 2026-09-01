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


# Config fields that are *resolved* rather than *authored*, and so must not reach the digest. Each
# is a path from the config root. See ``_canonical_config``.
_NOT_IDENTITY: tuple[tuple[str, ...], ...] = (("compute", "profile", "source"),)


def _canonical_config(cfg: RunConfig) -> str:
    """Serialize a config to a stable, order-independent JSON string.

    ``sort_keys`` makes the bytes insensitive to field ordering, so the hash depends
    only on the config's *content*.

    ``compute.profile.source`` is dropped first, and that exclusion is load-bearing. It is the one
    field the launcher *resolves* rather than the user authoring: ``lock_profile_source`` rewrites
    ``"auto"`` to whichever harvested run it discovered. Left in the digest, a config's identity
    depends on what has been run before it — run N pins run N-1's harvest, so re-running the same
    file forever yields new ids and never converges. That was observed live (smoke 01, 2026-09-01):
    the re-run resolved a different id and executed a whole second run instead of deduping.

    So identity stays a function of what was *asked for*, and the resolved pointer is recorded
    where provenance belongs — the staged manifest keeps the pinned ``source``, and the sizing
    telemetry keeps the full ``provenance`` block naming the run the measurements came from. A
    re-run therefore lands on the same ``run_id`` even if it is sized from newer evidence, which is
    what append-only-plus-dedupe-on-read requires.
    """
    payload = cfg.model_dump(mode="json")
    for *parents, leaf in _NOT_IDENTITY:
        node = payload
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(leaf, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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


# --- per-system platform ids ---------------------------------------------------
#
# A job's canonical id is its ``job_key`` (``sf-<run_id>-<family>-a<n>``). Each compute platform
# stamps that key as its *own* job id so the platform job and the registry row share an identity —
# but platforms differ on what characters/length an id may carry, so the key is mapped to a
# platform-legal form here. The mapped id is stored as ``run_jobs.system_job_id`` (the canonical
# ``job_key`` stays in ``run_jobs.job_id``), so a trace never has to reverse a lossy mapping.

_DATAPROC_ID_MAX = 63
_DATAPROC_BAD = re.compile(r"[^a-z0-9-]")


def dataproc_job_id(job_key: str) -> str:
    """Map a ``job_key`` to a Dataproc-legal batch/job id (Serverless batch or cluster job).

    Dataproc ids are 4-63 chars, ``[a-z0-9-]``, must start with a letter, and can't end with a
    hyphen. The key is lowercased with illegal chars (e.g. the ``_`` in ``deep_learning``) mapped to
    hyphens; if it exceeds 63 chars only the **tail** is kept (it carries the run_id digest + family
    + attempt — the unique part) behind an ``sf-`` prefix, so distinct jobs never collide.
    """
    s = _DATAPROC_BAD.sub("-", job_key.lower()).strip("-")
    if len(s) > _DATAPROC_ID_MAX:
        tail = s[-(_DATAPROC_ID_MAX - 3) :].lstrip("-")
        s = f"sf-{tail}"
    if not s[:1].isalpha():  # a Dataproc id must start with a letter
        s = f"j-{s}"[:_DATAPROC_ID_MAX]
    return s.rstrip("-")


def ray_submission_id(job_key: str) -> str:
    """Map a ``job_key`` to a Ray ``submission_id``.

    Ray submission ids accept ``[A-Za-z0-9_-]`` and are generously long, so the canonical key
    qualifies unchanged — closing the gap where Ray otherwise auto-assigns a random id. Passing this
    to ``JobSubmissionClient.submit_job(submission_id=…)`` makes the Ray job's own id deterministic.
    """
    return job_key


def bigquery_job_id(job_key: str) -> str:
    """Map a ``job_key`` to a BigQuery job id (the parent id of the run's multi-statement script).

    BigQuery job ids accept letters/digits/underscores/hyphens up to 1024 chars, so the key
    qualifies unchanged. Set as the parent job's id (not a prefix — a prefix would multi-match on
    lookup); child statements get their own ids under it, all traceable to this one parent.
    """
    return job_key


def decide_attempt(current_max: int | None, *, force: bool) -> tuple[int, bool]:
    """Resolve which ``attempt`` a submission should use, given the run/family's current max.

    The pure core of the re-run policy (the registry read that supplies ``current_max`` lives in
    ``registry.jobs``): returns ``(attempt, is_new_job)``.

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
