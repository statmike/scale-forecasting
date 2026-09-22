"""The Ray Jobs client — connect to a cluster's dashboard, submit a driver, poll it to terminal.

Everything that talks to ``ray.job_submission`` on an *already-existing* cluster. Separate from
`ray_cluster` because it is a different API against a different endpoint with different failure
modes: the cluster verbs go to Vertex's control plane and fail on capacity and quota, while these go
through Vertex's managed Ray dashboard proxy and fail on warm-up races and OAuth token expiry. A
reader debugging "the job never started" and a reader debugging "the cluster never provisioned" are
looking for two different files.

Three long-run hazards live here and nowhere else, which is most of why the module exists:

* **The dashboard warm-up race** — a cluster reaches RUNNING before its dashboard is reachable
  through the proxy, so the first handshake can time out (`_is_dashboard_warmup_error`,
  `_connect_job_client`'s retry budget).
* **The 60-minute bearer token** — the ``vertex_ray://`` client mints an OAuth token at construction
  and never refreshes it, so a long GPU run outlives it. Handled proactively
  (`_client_needs_refresh`) with a reactive 401 backstop (`_is_auth_expiry_error`).
* **A blip on the monitoring channel** — the poll runs for hours over a public HTTPS proxy, so
  sooner or later one request dies in transit. That says nothing about the job
  (`_is_recoverable_poll_error`, `_status_with_recovery`).

`probes.runtimes` reuses `_connect_job_client` to read a live job's status on demand.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from .errors import JobIdTaken, get_logger

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

# How hard one `get_job_status` call tries before we accept that contact is lost. The budget is
# per-poll, so a run that hiccups once an hour never accumulates toward it; only *consecutive*
# failures spend it. Deliberately generous, because the two outcomes are not symmetric: waiting a
# few extra minutes costs a fleet we have already paid for, while giving up early destroys a
# multi-hour run and tears the fleet down with it. `_connect_job_client` carries its own retry
# budget on top of this, so the real tolerance per attempt is larger than the backoff suggests.
_POLL_RETRY_ATTEMPTS = 4
_POLL_RETRY_BACKOFF_SECONDS = 15

# Fault injection for the recovery path above, and the only reason it exists: that path is eight
# offline tests and **zero live executions**. The condition it recovers from — a dropped request or
# an expired token mid-poll — cannot be scheduled. It happened once, on 2026-09-10, and cost a
# twenty-node fleet; every long Ray run since has polled cleanly, so the code that was written in
# response has never actually run against the real proxy, the real token and the real reconnect.
#
# A green run does not prove the recovery works. It proves the fault did not arrive. Arming this on
# a run that is happening anyway makes the fault arrive on purpose: the next poll of each Ray job
# raises a recoverable error, `_status_with_recovery` forgives it, `_connect_job_client` mints a
# genuinely fresh token, and the poll resumes — and the run's log carries the retry line that is the
# evidence. The run itself still completes normally, which is what makes this cheap to carry.
#
# Two shapes because the recovery has two doors and they are classified by different functions:
# ``transport`` is the proxy blip (`_is_dashboard_warmup_error`), ``auth`` is the expired bearer
# token (`_is_auth_expiry_error`). Comma-separate them — ``SF_RAY_POLL_FAULT=transport,auth`` — to
# spend two of the four attempts on one poll, proving both doors and the shared reconnect in one go.
#
# Infra-level, like ``SF_HIDE_DEVICES`` and ``SF_SERVERLESS_DEPS``, and for the same reason: it is
# not a property of the science, so it must not enter ``ComputeConfig`` and therefore the run_id.
# It is read only in the process that polls — the driver that submitted the job — so unlike the
# device fault there is nothing to carry to a worker.
POLL_FAULT_ENV = "SF_RAY_POLL_FAULT"
_POLL_FAULT_AUTH = "auth"
_POLL_FAULT_TRANSPORT = "transport"
# Shaped like the SDK's own ``_raise_error`` rendering, because the classifiers read the *string*:
# a message that failed to match would inject an *unrecoverable* fault and kill the run it rode in
# on. `test_every_armed_fault_is_one_the_poll_forgives` is what keeps these two facts together.
_POLL_FAULT_MESSAGES = {
    _POLL_FAULT_AUTH: "Request failed with status code 401: Unauthorized (SF_RAY_POLL_FAULT)",
    _POLL_FAULT_TRANSPORT: (
        "Request failed with status code 503: Service Temporarily Unavailable (SF_RAY_POLL_FAULT)"
    ),
}


def _is_submission_id_taken_error(exc: Exception) -> bool:
    """True if ``exc`` is Ray refusing a ``submission_id`` the cluster already knows.

    Ray's job-submission client has no typed exception for this — the dashboard answers 400 and the
    SDK re-raises a bare ``RuntimeError`` — so the classification is by message, the same way the
    two poll classifiers below work. Matched narrowly: both the id vocabulary *and* the word
    "exists" have to be present, because a false positive would relabel an unrelated failure as a
    name clash and send the operator looking for a job that was never created.
    """
    low = str(exc).lower()
    return "exists" in low and ("submission_id" in low or "submission id" in low or "job id" in low)


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


def _is_job_absent_error(exc: Exception) -> bool:
    """True if ``exc`` is the dashboard saying *this job is not here* — a fact, not a fault.

    The third sibling of the two classifiers above, and the one with the largest consequence. When
    ``get_job_status`` is asked about a submission id the cluster has no record of, the SDK's
    ``_raise_error`` renders the server's 404 body verbatim:
    ``RuntimeError("Request failed with status code 404: Job raysubmit_… does not exist.")``. A
    reachable dashboard answering that has *proved* the job's absence — it is the strongest such
    statement available, and it is categorically different from the transport faults
    `_is_recoverable_poll_error` forgives, which say only that we could not see.

    Both markers are required, and the asymmetry is the reason. A false negative costs nothing: the
    caller degrades to the UNKNOWN it would have returned anyway. A false positive lets a repair
    verb write FAILED over a live job. So a bare ``404`` — which a misrouted proxy also returns —
    is not enough on its own, and neither is a bare "does not exist", which the same API uses for a
    missing runtime-env *package* at submit time. Only the two together mean what we need.
    """
    low = str(exc).lower()
    if "package" in low:
        return False
    return " 404" in low and "does not exist" in low


def _is_recoverable_poll_error(exc: Exception) -> bool:
    """True if a failed ``get_job_status`` says something about the *channel*, not about the job.

    A poll is a small HTTPS request to Vertex's managed dashboard proxy, repeated every
    `_POLL_SECONDS` for as long as the run lasts. Over a two-hour run that is several hundred
    requests across the public internet, and the interesting property is that *none of them are the
    job*: the job is a Ray driver on a cluster that neither knows nor cares whether we are watching.
    So a transport failure here — a proxy 5xx, a dropped connection, a TLS session that ends
    mid-read — is a statement about our view of the run, and treating it as a verdict on the run is
    a category error. `ray-100k-3fbc82fe3b6d` (2026-09-10) is what that error costs: an
    ``SSLError: UNEXPECTED_EOF_WHILE_READING`` at minute 79 propagated out of the poll, marked both
    jobs FAILED, and tore down a twenty-node fleet that was still writing cells — 224,967 of 400,000
    of them already durable in BigQuery.

    Two shapes are recoverable and they arrive for different reasons. An expired token
    (`_is_auth_expiry_error`) is ours to fix by re-minting. A transient transport fault
    (`_is_dashboard_warmup_error`, which is the same wire-level vocabulary whether it shows up
    during warm-up or mid-poll) is nobody's to fix and clears by itself. Everything else — a version
    mismatch, a 403, an unrecognised error — is a genuine fault and must still propagate, because a
    poll loop that retries *every* exception is a poll loop that never ends.
    """
    return _is_auth_expiry_error(exc) or _is_dashboard_warmup_error(exc)


def _status_with_recovery(
    poll: Callable[[], str],
    reconnect: Callable[[], None],
    *,
    attempts: int = _POLL_RETRY_ATTEMPTS,
    backoff_s: float = _POLL_RETRY_BACKOFF_SECONDS,
) -> str:
    """Call ``poll`` for a job status, rebuilding the client and retrying past recoverable failures.

    Split out of `_submit_and_poll` for one reason: the live poll is `pragma: no cover`, and the
    behaviour that matters here — *which* failures are forgiven and how many times — is exactly the
    behaviour that was wrong before. It has to be provable offline.

    On a recoverable failure we wait before reconnecting rather than after, so a blip has a moment
    to clear before we spend a handshake on it. ``reconnect`` failures are *not* caught: if we
    cannot re-establish contact at all then contact really is lost, and `_connect_job_client` has
    already spent its own retry budget deciding that.

    The retry line is a **warning**, not info, and that is load-bearing. It was info until
    2026-09-22, when the first armed run showed the injection lines and none of the answering
    retries: the harness — and every runbook invocation — logs at WARNING, so the one line that
    says the recovery fired could not appear, and a reader following `smoke_testing.md` exactly
    would have read a working recovery as a broken one. Warning is also the honest level on its own
    terms. A poll failure that forces a reconnect is an anomaly on a long fleet run, not routine
    progress, and it is rare enough to carry no spam risk: across every live Ray run before the
    fault was armed on purpose, this branch executed zero times.
    """
    for attempt in range(1, attempts + 1):
        try:
            return poll()
        except Exception as exc:  # noqa: BLE001 - classify, recover the channel, re-raise faults
            if not _is_recoverable_poll_error(exc) or attempt == attempts:
                raise
            _log.warning(
                "Ray job poll failed on attempt %d/%d (%r); reconnecting and retrying",
                attempt,
                attempts,
                exc,
            )
            time.sleep(backoff_s)
            reconnect()
    raise AssertionError("unreachable: the loop returns or raises on every attempt")


def armed_poll_faults() -> tuple[str, ...]:
    """Which faults ``SF_RAY_POLL_FAULT`` arms, in the order they fire — ``()`` when it is unset.

    Anything truthy that is not ``"auth"`` reads as ``"transport"``, which is the same fail-*closed*
    choice `hardware.hide_devices_mode` makes and for the same reason: a typo'd value that armed
    nothing would turn a deliberate negative arm into an ordinary run and report it as proof.
    """
    raw = (os.environ.get(POLL_FAULT_ENV) or "").strip().lower()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tuple(
        _POLL_FAULT_AUTH if t == _POLL_FAULT_AUTH else _POLL_FAULT_TRANSPORT for t in tokens
    )


def _arm_poll_faults(
    poll: Callable[[], str], job_id: str, *, shapes: Sequence[str] | None = None
) -> Callable[[], str]:
    """Wrap ``poll`` so each armed fault raises once, on consecutive calls, before the real poll.

    The queue is a closure local rather than module state, so "once" means once per job rather than
    once per process: a run with a statistical job and a deep-learning job exercises the recovery
    twice, on two different Ray jobs, for the price of one armed run.

    Consecutive rather than spread out, because consecutive is the harder case.
    `_status_with_recovery` counts *consecutive* failures against its budget, so two in a row spend
    two of the four attempts and prove it keeps its patience across a reconnect — where two an hour
    apart would only ever prove the first attempt works twice.
    """
    queue = list(armed_poll_faults() if shapes is None else shapes)

    def _armed() -> str:
        if not queue:
            return poll()
        shape = queue.pop(0)
        _log.warning(
            "%s is armed: injecting a %s fault into this poll of %s (%d left)",
            POLL_FAULT_ENV,
            shape,
            job_id,
            len(queue),
        )
        raise RuntimeError(_POLL_FAULT_MESSAGES[shape])

    return _armed


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
    try:
        job_id = client.submit_job(**submit_kwargs)
    except Exception as exc:
        # A submission_id the cluster already holds is the Ray face of the same clash the Dataproc
        # paths raise `AlreadyExists` for; everything else goes out untouched.
        if _is_submission_id_taken_error(exc):
            raise JobIdTaken(
                f"Ray submission_id {submission_id} already exists on "
                f"{cluster_resource_name}: the cluster holds this job id but the registry has no "
                f"attempt for it. Re-run with a different run_id, or bump the attempt by letting "
                f"--force walk past it."
            ) from exc
        raise
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

    def _reconnect() -> None:
        # Backstop for both recoverable shapes. A 401 means the proactive refresh missed (clock
        # skew, or a rebuild that landed late) and a fresh token fixes it; a transport fault means
        # the channel dropped and a fresh client re-establishes it. Same remedy, so one path.
        nonlocal client, client_born
        client = _connect_job_client(cluster_resource_name)
        client_born = time.monotonic()

    # `_arm_poll_faults` is a pass-through unless SF_RAY_POLL_FAULT is set, so an ordinary run pays
    # one empty-list check per poll and its behaviour is unchanged. Wrapped once, outside `_status`,
    # because the queue lives in the wrapper: rebuilding it every poll would re-arm the fault
    # forever and the job would never reach terminal.
    _poll_once = _arm_poll_faults(lambda: str(_fresh_client().get_job_status(job_id)), job_id)

    def _status() -> str:
        return _status_with_recovery(_poll_once, _reconnect)

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
