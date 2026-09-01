"""The Ray Jobs client — connect to a cluster's dashboard, submit a driver, poll it to terminal.

Everything that talks to ``ray.job_submission`` on an *already-existing* cluster. Separate from
`ray_cluster` because it is a different API against a different endpoint with different failure
modes: the cluster verbs go to Vertex's control plane and fail on capacity and quota, while these go
through Vertex's managed Ray dashboard proxy and fail on warm-up races and OAuth token expiry. A
reader debugging "the job never started" and a reader debugging "the cluster never provisioned" are
looking for two different files.

Two long-run hazards live here and nowhere else, which is most of why the module exists:

* **The dashboard warm-up race** — a cluster reaches RUNNING before its dashboard is reachable
  through the proxy, so the first handshake can time out (`_is_dashboard_warmup_error`,
  `_connect_job_client`'s retry budget).
* **The 60-minute bearer token** — the ``vertex_ray://`` client mints an OAuth token at construction
  and never refreshes it, so a long GPU run outlives it. Handled proactively
  (`_client_needs_refresh`) with a reactive 401 backstop (`_is_auth_expiry_error`).

`probes.runtimes` reuses `_connect_job_client` to read a live job's status on demand.
"""

from __future__ import annotations

import time
from typing import Any

from .errors import get_logger

_log = get_logger(__name__)

# Poll cadence + terminal Ray job states (the Jobs API reports these on get_job_status).
_POLL_SECONDS = 15
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
# On a FAILED job, how many trailing driver-log lines to capture into our log + the raised error —
# enough to carry a Python traceback without dumping the whole (potentially huge) driver stdout.
_FAILURE_LOG_TAIL_LINES = 60

# Dashboard warm-up race: a cluster reaches RUNNING *before* its Ray dashboard is reachable through
# Vertex's public-endpoint proxy, so the first JobSubmissionClient handshake (GET /api/version) can
# get a proxy gateway timeout / connection refusal. We retry the *connection only* with backoff up
# to this budget before giving up — well within the ~few-minutes the dashboard takes to serve.
_DASHBOARD_CONNECT_ATTEMPTS = 20
_DASHBOARD_CONNECT_BACKOFF_SECONDS = 15

# The vertex_ray:// Jobs client mints an OAuth Bearer token (~60-min TTL) at construction and never
# refreshes it (see `_is_auth_expiry_error`). A long GPU run (NeuralProphet) can outlive it, so we
# proactively rebuild the client — minting a fresh token — once it reaches this age, comfortably
# under the TTL, rather than waiting to absorb the 401 the reactive poll path handles as a backstop.
_CLIENT_MAX_AGE_SECONDS = 2700  # 45 min


def _is_dashboard_warmup_error(exc: Exception) -> bool:
    """True if ``exc`` looks like the dashboard-not-yet-serving race (retryable), not a real fault.

    The JobSubmissionClient version handshake fails during warm-up with a proxy gateway timeout
    (HTTP 5xx — 502/503/504, and Cloudflare's 524) or a bare connection error, all transient. A
    4xx / auth / version-mismatch is a genuine fault and must *not* be retried, so we match on the
    known-transient shapes only.
    """
    low = str(exc).lower()
    transient_markers = (
        " 502",
        " 503",
        " 504",
        " 524",
        "gateway",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "max retries",
    )
    return any(marker in low for marker in transient_markers)


def _is_auth_expiry_error(exc: Exception) -> bool:
    """True if ``exc`` is an expired-credential ``401`` from the dashboard proxy (refresh & retry).

    Distinct from `_is_dashboard_warmup_error`, which treats a 401 as a *connect-time* fault
    that won't fix itself by waiting (right — spinning the warm-up loop on bad auth is pointless).
    During a *long poll*, however, a 401 means something different: the ``vertex_ray://`` Jobs
    client caches an OAuth Bearer token minted at construction (~60-min TTL), so a run outliving the
    token gets a 401 on the next ``get_job_status`` even though nothing is wrong — rebuilding the
    client mints a fresh token and the poll resumes. Match the 401 shapes the proxy returns.
    """
    low = str(exc).lower()
    return " 401" in low or "unauthorized" in low


def _client_needs_refresh(
    born_monotonic: float,
    now_monotonic: float,
    max_age_s: float = _CLIENT_MAX_AGE_SECONDS,
) -> bool:
    """True once the Jobs client is old enough that its cached OAuth token may be nearing expiry.

    The ``vertex_ray://`` client mints a Bearer token (~60-min TTL) at construction and never
    refreshes it; rebuilding *before* the TTL keeps a long poll authenticated. Pure and
    time-injected so the age policy is unit-testable without any live Ray I/O.
    """
    return (now_monotonic - born_monotonic) >= max_age_s


def _connect_job_client(
    cluster_resource_name: str,
) -> Any:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Open a ``JobSubmissionClient`` to the cluster, retrying past the warm-up race.

    We address the cluster by its **resource name**
    (``vertex_ray://projects/<num>/locations/<region>/persistentResources/<name>``): the
    ``[ray]``-extra resolver discovers the dashboard endpoint and authenticates the connection
    itself. Submission routes through a Google-managed dashboard proxy.

    Importing ``vertex_ray`` here is **load-bearing, not cosmetic**: the plugin registers the
    ``vertex_ray://`` address handler *and* injects the OAuth Bearer token into the dashboard
    handshake — Google's docs state it is "required to obtain authentication automatically." Without
    it in the process that builds the client, the ``GET /api/version`` request reaches the proxy
    without valid auth and the proxy holds it open until it times out (HTTP 524) instead of
    returning a clean 401. The SDK's project/location is already bound upstream (``_init_vertex`` on
    both the create and reuse paths); the import is idempotent, so this is belt-and-suspenders.

    ``JobSubmissionClient.__init__`` does a GET ``/api/version`` handshake; right after the cluster
    hits RUNNING that endpoint may not be reachable yet, so the first attempts can raise a proxy
    gateway timeout (524/504/…). We back off and retry the *connection only* (never a partial
    submit) until it succeeds or the budget is spent, then let the last error propagate.
    """
    from google.cloud.aiplatform import vertex_ray  # noqa: F401 - registers vertex_ray:// + auth
    from ray.job_submission import JobSubmissionClient

    last_exc: Exception | None = None
    for attempt in range(1, _DASHBOARD_CONNECT_ATTEMPTS + 1):
        try:
            return JobSubmissionClient(f"vertex_ray://{cluster_resource_name}")
        except Exception as exc:  # noqa: BLE001 - classify, retry transients, re-raise faults
            if not _is_dashboard_warmup_error(exc):
                raise
            last_exc = exc
            _log.info(
                "Ray dashboard not ready yet (attempt %d/%d): %r",
                attempt,
                _DASHBOARD_CONNECT_ATTEMPTS,
                exc,
            )
        _log.info("retrying Ray dashboard connect in %ds", _DASHBOARD_CONNECT_BACKOFF_SECONDS)
        time.sleep(_DASHBOARD_CONNECT_BACKOFF_SECONDS)
    assert last_exc is not None
    raise last_exc


def _submit_and_poll(
    cluster_resource_name: str,
    entrypoint: str,
    runtime_env: dict[str, Any],
    *,
    wait: bool,
    submission_id: str | None = None,
) -> tuple[str, str, str]:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Submit the on-cluster driver as a Ray Job and (when ``wait``) poll to a terminal state.

    Connects the Jobs client to the cluster by resource name (``vertex_ray://<resource_name>``,
    retrying past the dashboard warm-up race), submits ``entrypoint`` with ``runtime_env`` (current
    ``src/`` + requirements), and returns ``(job_id, status, detail)``. ``submission_id``, when set,
    is passed to ``submit_job`` so the Ray job's own id is the deterministic ``job_key`` rather than
    a random auto-assigned one; the returned ``job_id`` then equals it. ``detail`` is empty except
    on a ``FAILED`` terminal state, where it carries the driver's error message + log tail
    (`_fetch_job_failure_detail`) captured at the moment of failure — so the cause is recorded
    even after the ``ml_job`` log stream ages out. Without ``wait`` the status is the immediate
    post-submit state (the caller skips telemetry + the terminal-state check).
    """
    client = _connect_job_client(cluster_resource_name)
    client_born = time.monotonic()
    submit_kwargs: dict[str, Any] = {"entrypoint": entrypoint, "runtime_env": runtime_env}
    if submission_id is not None:
        submit_kwargs["submission_id"] = submission_id
    job_id = client.submit_job(**submit_kwargs)
    _log.info("submitted Ray job %s", job_id)

    def _fresh_client() -> Any:
        # Proactively re-mint the OAuth token BEFORE it dies (see `_client_needs_refresh`): the
        # vertex_ray:// client caches a Bearer token (~60-min TTL) at construction, and a long GPU
        # run (NeuralProphet) can outlive it. Rebuilding at 45 min keeps every poll authenticated,
        # so we never even take the 401 the reactive branch below would otherwise absorb.
        nonlocal client, client_born
        if _client_needs_refresh(client_born, time.monotonic()):
            _log.info("Ray Jobs client nearing token TTL; proactively refreshing")
            client = _connect_job_client(cluster_resource_name)
            client_born = time.monotonic()
        return client

    def _status() -> str:
        # Backstop: if the proactive refresh ever misses (clock skew / a rebuild that lands late), a
        # 401 is still recoverable — rebuild the client (fresh token) and retry once, so a long run
        # polls to completion instead of aborting.
        nonlocal client, client_born
        try:
            return str(_fresh_client().get_job_status(job_id))
        except Exception as exc:  # noqa: BLE001 - only a 401 is recoverable here; re-raise the rest
            if not _is_auth_expiry_error(exc):
                raise
            _log.info("Ray job poll hit auth expiry (%r); refreshing client and retrying", exc)
            client = _connect_job_client(cluster_resource_name)
            client_born = time.monotonic()
            return str(client.get_job_status(job_id))

    if not wait:
        return job_id, _status(), ""

    status = _status()
    while status not in _TERMINAL_STATES:
        time.sleep(_POLL_SECONDS)
        status = _status()
    _log.info("Ray job %s finished: status=%s", job_id, status)
    # Use the age-checked client here too: a token dying right at terminal-FAILED would otherwise
    # cost us the driver diagnosis (`_fetch_job_failure_detail` is best-effort and unwrapped).
    detail = _fetch_job_failure_detail(_fresh_client(), job_id) if status == "FAILED" else ""
    if detail:
        _log.error("Ray job %s FAILED — driver diagnosis:\n%s", job_id, detail)
    return job_id, status, detail


def _fetch_job_failure_detail(
    client: Any, job_id: str
) -> str:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Best-effort driver error message + log tail for a FAILED Ray job (operability).

    A terminal ``FAILED`` status alone says *nothing* about the cause; the driver's Python
    traceback lives in the Ray dashboard and Cloud Logging's ``ml_job`` stream, which ages out
    of the default freshness window within ~90 min — so a failure diagnosed later is a failure
    diagnosed by archaeology. The Jobs client already holds both facts: ``get_job_info().message``
    (the terminal error line) and ``get_job_logs()`` (the full driver stdout/stderr). Pull them at
    the moment of failure so the cause is captured in *our* log and folded into the raised
    `EngineError` — never dependent on a still-warm log stream.
    Every step is defensive: a diagnosis that itself fails must not mask the underlying job failure.
    """
    parts: list[str] = []
    try:
        info = client.get_job_info(job_id)
        message = getattr(info, "message", None)
        if message:
            parts.append(f"message: {message}")
    except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort, never fatal
        _log.warning("could not fetch Ray job info for %s: %r", job_id, exc)
    try:
        logs = client.get_job_logs(job_id) or ""
        tail = "\n".join(logs.splitlines()[-_FAILURE_LOG_TAIL_LINES:]).strip()
        if tail:
            parts.append(f"driver log tail:\n{tail}")
    except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort, never fatal
        _log.warning("could not fetch Ray job logs for %s: %r", job_id, exc)
    return "\n".join(parts)
