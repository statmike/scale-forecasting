"""Best-effort resolution of the calling principal, for a run's audit trail.

Both run launch (the header ``user_id``) and cancel (the ``job_telemetry.$.cancel`` audit blob)
record *who did this*. The answer is the ADC principal: a service-account email for a runner SA
(Composer, CI) — carried on the credentials, so free — or a user's email for a laptop running under
``gcloud auth application-default login``. Resolution is advisory: it never raises and never blocks
the operation it annotates, returning ``None`` when the principal can't be determined cheaply.

The module import is light (no ``google`` import at load); the ADC/network calls stay lazy inside
`resolve_principal`, so importing this never pulls auth machinery on a path that doesn't need it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import Settings

# A user credential doesn't carry its email, so we read it once from the OpenID userinfo endpoint —
# short-timeout and best-effort — to attribute a laptop-launched run. A runner SA never reaches this
# (its email is on the credential), so this is the laptop-user path only.
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_USERINFO_TIMEOUT_S = 10.0


def _sa_email(creds: Any) -> str | None:
    """The service-account email carried on a credential, or ``None`` for a user credential (pure).

    A service-account credential exposes its identity as ``service_account_email`` (or
    ``signer_email`` on some credential types); a user credential exposes neither. The metadata
    placeholder ``"default"`` (a compute credential before its first refresh) counts as unresolved.
    """
    for attr in ("service_account_email", "signer_email"):
        email = getattr(creds, attr, None)
        if email and email != "default":
            return str(email)
    return None


def resolve_principal(settings: Settings | None = None) -> str | None:  # pragma: no cover - ADC I/O
    """The calling ADC principal's email for the audit trail, or ``None`` when it isn't cheap.

    Resolves ADC (`google.auth.default`), then returns the service-account email when the credential
    carries one (the common runner-SA case — free) — otherwise makes one short-timeout call to the
    OpenID userinfo endpoint to read a user credential's email. Never raises and never blocks: any
    failure (missing scope, offline, an unusual credential type) yields ``None`` so the annotated
    operation proceeds unattributed rather than failing. ``settings`` is accepted for call-site
    symmetry with the writers that use it; the ADC principal is independent of the target project.
    """
    try:
        import google.auth

        creds, _ = google.auth.default()
        sa = _sa_email(creds)
        if sa is not None:
            return sa
        return _userinfo_email(creds)
    except Exception:  # noqa: BLE001 - identity is best-effort; callers proceed without it
        return None


def _userinfo_email(creds: Any) -> str | None:  # pragma: no cover - network I/O
    """A user credential's email via the OpenID userinfo endpoint (best-effort, short-timeout)."""
    import google.auth.transport.requests as gtr

    if not getattr(creds, "valid", False):
        creds.refresh(gtr.Request())
    resp = gtr.AuthorizedSession(creds).get(_USERINFO_URL, timeout=_USERINFO_TIMEOUT_S)
    if resp.status_code == 200:
        return resp.json().get("email") or None
    return None
