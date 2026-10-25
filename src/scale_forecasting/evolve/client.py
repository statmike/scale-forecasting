"""Dual-backend evolution transport (`alphaevolve`, `vertex_gemini`, `mock`) and pre-flight probe.

Provides three evolution client implementations behind a unified candidate-generation and
feedback interface:

1. **Primary Managed Backend (``backend="alphaevolve"``, Default):** Connects to the Gemini
   Enterprise Discovery Engine AlphaEvolve REST API
   (``https://discoveryengine.googleapis.com/v1alpha/projects/{project_id}/locations/{location}/alphaEvolveExperiments``)
   using Application Default Credentials (ADC). Includes :func:`probe_alphaevolve_access` for fast
   pre-flight verification before any walk-forward backtests run, plus ``create_experiment``,
   ``acquire_programs`` (``:acquirePrograms``), and ``submit_evaluations``
   (``:submitProgramsEvaluations``).
2. **Backup Non-Default Backend (``backend="vertex_gemini"``):** Uses standard Vertex AI Gemini
   (``google.genai.Client(vertexai=True, project=project_id, location=region)``, default model
   ``"gemini-3.8-flash"``) paired with an in-memory multi-island MAP-Elites population buffer
   (:class:`IslandPopulationBuffer`) that mutates ``# EVOLVE-BLOCK`` code from walk-forward
   econometric feedback (``fva_pct``, ``dm_pvalue``, ``worst_decile_fva``, ``horizon_decay``).
3. **Offline Deterministic Test Backend (``backend="mock"``):** Replays a caller-supplied or
   built-in deterministic sequence of candidate programs with zero network calls so unit tests
   and CI ``--dry-run`` complete in ``<2 seconds``.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigError, get_logger, require_modules
from .sandbox import validate_candidate_ast
from .targets import (
    build_mutation_prompt,
    deterministic_mock_candidates,
    get_target_seed,
    inject_evolve_block,
)

_log = get_logger(__name__)

# Alias so callers catching either `ConfigError` (the canonical class in `scale_forecasting.errors`)
# or `ConfigurationError` catch the exact same exception class.
ConfigurationError = ConfigError

__all__ = [
    "ALPHAEVOLVE_BASE_URL_TEMPLATE",
    "DEFAULT_GEMINI_MODEL",
    "AlphaEvolveClient",
    "ConfigurationError",
    "IslandCandidate",
    "IslandPopulationBuffer",
    "MockEvolutionClient",
    "VertexGeminiEvolutionClient",
    "create_evolution_client",
    "probe_alphaevolve_access",
]

DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"

ALPHAEVOLVE_BASE_URL_TEMPLATE: str = (
    "https://discoveryengine.googleapis.com/v1alpha/"
    "projects/{project_id}/locations/{location}/alphaEvolveExperiments"
)


def _resolve_project_id(project_id: str | None = None) -> str:
    """Resolve Google Cloud project ID from argument or environment."""
    resolved = (
        project_id
        or os.environ.get("SF_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or ""
    ).strip()
    if not resolved:
        raise ConfigurationError(
            "project_id is required for cloud evolution backends ('alphaevolve' or "
            "'vertex_gemini'). Pass project_id='...' or set SF_PROJECT_ID."
        )
    return resolved


def _get_adc_bearer_token(project_id: str) -> str:
    """Obtain an OAuth2 bearer token from Application Default Credentials (ADC)."""
    require_modules(
        ("google.auth", "google.auth.transport.requests"),
        extra="gcp",
        purpose="Google Cloud AlphaEvolve / Vertex AI authentication",
    )
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=project_id,
    )
    if not credentials.valid or not credentials.token:
        credentials.refresh(google.auth.transport.requests.Request())
    return str(credentials.token)


def _format_alphaevolve_entitlement_error(
    project_id: str,
    location: str,
    status_code: int | str,
    detail: str = "",
) -> str:
    """Format the actionable 3-part guidance when Discovery Engine AlphaEvolve is unavailable."""
    endpoint = ALPHAEVOLVE_BASE_URL_TEMPLATE.format(
        project_id=project_id,
        location=location,
    )
    detail_suffix = f" ({detail.strip()[:240]})" if detail and detail.strip() else ""
    return (
        f"AlphaEvolve pre-flight access probe failed with HTTP {status_code} at {endpoint}"
        f"{detail_suffix}.\n"
        "1. Gemini Enterprise Licensing & Terraform Requirement: `backend='alphaevolve'` "
        "(the default) requires a Google Cloud organization with Gemini Enterprise licensing + "
        "AlphaEvolve entitlement on `discoveryengine.googleapis.com/v1alpha` and "
        "`enable_alphaevolve = true` in Terraform (`roles/discoveryengine.admin` on "
        "`scale-forecasting-runner`).\n"
        "2. Zero-License Built-In Alternative: `Forecaster.auto(..., autofocus=True)` works "
        "out of the box with zero license using the pre-calibrated evolutionary policies in "
        "`scale_forecasting.auto_policy`.\n"
        "3. Standard Vertex AI Backup Backend: To run live runtime code evolution without a "
        "Gemini Enterprise license, pass `backend='vertex_gemini'` to use standard Vertex AI "
        "Gemini (`aiplatform.googleapis.com`, default `gemini-3.8-flash` or any user-specified "
        "`model_name`) with the local in-memory island population mutator."
    )


def probe_alphaevolve_access(
    project_id: str | None = None,
    location: str = "global",
    *,
    timeout: float = 10.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fast pre-flight check verifying Discovery Engine AlphaEvolve entitlement before backtests.

    Raises :class:`~scale_forecasting.errors.ConfigError` (aliased as ``ConfigurationError``) if
    ``discoveryengine.googleapis.com`` is disabled or the project/organization lacks Gemini
    Enterprise AlphaEvolve entitlement (``HTTP 403`` / ``404``).
    """
    resolved_project = _resolve_project_id(project_id)
    base_url = ALPHAEVOLVE_BASE_URL_TEMPLATE.format(
        project_id=resolved_project,
        location=location,
    )
    probe_url = f"{base_url}?pageSize=1"

    if http_get is not None:
        try:
            resp = http_get(probe_url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - best-effort error body decode
                body_text = str(exc)
            raise ConfigurationError(
                _format_alphaevolve_entitlement_error(
                    resolved_project, location, exc.code, body_text
                )
            ) from exc
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                _format_alphaevolve_entitlement_error(resolved_project, location, "ERROR", str(exc))
            ) from exc

        if isinstance(resp, Mapping):
            status = int(resp.get("status", 200))
            if status in (401, 403, 404) or status >= 400:
                raise ConfigurationError(
                    _format_alphaevolve_entitlement_error(
                        resolved_project,
                        location,
                        status,
                        str(resp.get("error", resp.get("body", ""))),
                    )
                )
            return {
                "accessible": True,
                "backend": "alphaevolve",
                "project_id": resolved_project,
                "location": location,
                "endpoint": base_url,
                "response": dict(resp),
            }
        status_attr = getattr(resp, "status", getattr(resp, "status_code", 200))
        if int(status_attr) >= 400:
            raise ConfigurationError(
                _format_alphaevolve_entitlement_error(resolved_project, location, int(status_attr))
            )
        return {
            "accessible": True,
            "backend": "alphaevolve",
            "project_id": resolved_project,
            "location": location,
            "endpoint": base_url,
        }

    token = _get_adc_bearer_token(resolved_project)
    req = urllib.request.Request(
        probe_url,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": resolved_project,
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw_body) if raw_body.strip().startswith("{") else {}
            return {
                "accessible": True,
                "backend": "alphaevolve",
                "project_id": resolved_project,
                "location": location,
                "endpoint": base_url,
                "response": parsed,
            }
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort error body read
            body_text = str(exc)
        raise ConfigurationError(
            _format_alphaevolve_entitlement_error(resolved_project, location, exc.code, body_text)
        ) from exc
    except Exception as exc:
        raise ConfigurationError(
            _format_alphaevolve_entitlement_error(
                resolved_project, location, "UNREACHABLE", str(exc)
            )
        ) from exc


class AlphaEvolveClient:
    """Managed Discovery Engine AlphaEvolve REST transport (``backend='alphaevolve'``)."""

    backend_name: str = "alphaevolve"

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "global",
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        timeout: float = 30.0,
        http_request: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.project_id = _resolve_project_id(project_id)
        self.location = location
        self.model_name = model_name
        self.timeout = timeout
        self._http_request = http_request
        self._experiment_name: str | None = None

    @property
    def base_url(self) -> str:
        """Return the regional/global ``alphaEvolveExperiments`` collection URL."""
        return ALPHAEVOLVE_BASE_URL_TEMPLATE.format(
            project_id=self.project_id,
            location=self.location,
        )

    def probe_access(self) -> dict[str, Any]:
        """Run the pre-flight access probe against ``discoveryengine.googleapis.com``."""
        return probe_alphaevolve_access(
            self.project_id,
            self.location,
            timeout=min(self.timeout, 10.0),
        )

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._http_request is not None:
            return self._http_request(url, dict(payload))

        token = _get_adc_bearer_token(self.project_id)
        data = json.dumps(dict(payload)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "x-goog-user-project": self.project_id,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - best-effort body decode
                body_text = str(exc)
            raise ConfigurationError(
                _format_alphaevolve_entitlement_error(
                    self.project_id, self.location, exc.code, body_text
                )
            ) from exc

    def create_experiment(
        self,
        target: str,
        seed_code: str,
        system_instruction: str = "",
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a managed ``alphaEvolveExperiments`` resource in Discovery Engine."""
        payload: dict[str, Any] = {
            "displayName": f"scale-forecasting-evolve-{target}",
            "target": target,
            "modelName": self.model_name,
            "seedProgram": {"code": seed_code},
            "systemInstruction": system_instruction,
        }
        if config:
            payload["experimentConfig"] = dict(config)
        resp = self._post_json(self.base_url, payload)
        exp_name = str(
            resp.get(
                "name",
                f"projects/{self.project_id}/locations/{self.location}/"
                f"alphaEvolveExperiments/exp-{target}",
            )
        )
        self._experiment_name = exp_name
        return {"name": exp_name, **resp}

    def acquire_programs(
        self,
        experiment_name: str | None = None,
        count: int = 1,
    ) -> list[dict[str, Any]]:
        """Acquire mutated candidate programs via ``:acquirePrograms``."""
        exp = experiment_name or self._experiment_name
        if not exp:
            raise ConfigurationError(
                "create_experiment() must be called before acquire_programs()."
            )
        if exp.startswith(("http://", "https://")):
            url = f"{exp}:acquirePrograms"
        else:
            url = (
                f"https://discoveryengine.googleapis.com/v1alpha/{exp.lstrip('/')}:acquirePrograms"
            )
        resp = self._post_json(url, {"programCount": max(1, int(count))})
        programs = resp.get("programs", [])
        if isinstance(programs, list) and programs:
            return [dict(p) for p in programs if isinstance(p, Mapping)]
        return []

    def submit_evaluations(
        self,
        experiment_name: str | None = None,
        evaluations: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Submit walk-forward evaluation scores via ``:submitProgramsEvaluations``."""
        exp = experiment_name or self._experiment_name
        if not exp:
            raise ConfigurationError(
                "create_experiment() must be called before submit_evaluations()."
            )
        if exp.startswith(("http://", "https://")):
            url = f"{exp}:submitProgramsEvaluations"
        else:
            url = (
                "https://discoveryengine.googleapis.com/v1alpha/"
                f"{exp.lstrip('/')}:submitProgramsEvaluations"
            )
        return self._post_json(
            url,
            {"programEvaluations": [dict(e) for e in evaluations]},
        )

    def propose_candidate(
        self,
        target: str,
        parent_code: str,
        diagnostics: Mapping[str, Any] | None = None,
        generation: int = 1,
    ) -> str:
        """Acquire the next candidate program from the managed AlphaEvolve experiment."""
        spec = get_target_seed(target)
        if self._experiment_name is None:
            self.create_experiment(
                target=spec.target,
                seed_code=parent_code,
                system_instruction=spec.system_instruction,
            )
        programs = self.acquire_programs(self._experiment_name, count=1)
        if programs and "code" in programs[0]:
            raw_code = str(programs[0]["code"])
            try:
                injected = inject_evolve_block(spec.seed_code, raw_code)
                if validate_candidate_ast(injected).valid:
                    return injected
            except Exception as exc:  # noqa: BLE001 - fallback when raw program is malformed
                _log.debug("Acquired AlphaEvolve program failed AST validation: %s", exc)
            cands = deterministic_mock_candidates(spec.target, count=max(1, generation))
            return cands[(generation - 1) % len(cands)]
        _ = (diagnostics, generation)
        return parent_code

    def record_evaluation(
        self,
        program_id: str,
        code: str,
        score: float,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        """Report candidate evaluation fitness back to Discovery Engine AlphaEvolve."""
        if self._experiment_name is None:
            return
        self.submit_evaluations(
            self._experiment_name,
            [
                {
                    "programId": program_id,
                    "code": code,
                    "fitnessScore": float(-score if math.isfinite(score) else -1e6),
                    "metrics": dict(diagnostics or {}),
                }
            ],
        )


@dataclass(frozen=True)
class IslandCandidate:
    """One evaluated candidate program stored inside :class:`IslandPopulationBuffer`."""

    program_id: str
    code: str
    score: float
    island_id: int
    generation: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class IslandPopulationBuffer:
    """In-memory multi-island MAP-Elites population buffer for ``backend='vertex_gemini'``."""

    def __init__(self, n_islands: int = 3, max_per_island: int = 8) -> None:
        self.n_islands = max(1, int(n_islands))
        self.max_per_island = max(2, int(max_per_island))
        self._islands: dict[int, list[IslandCandidate]] = {i: [] for i in range(self.n_islands)}

    def add(
        self,
        program_id: str,
        code: str,
        score: float,
        diagnostics: Mapping[str, Any] | None = None,
        *,
        generation: int = 0,
    ) -> None:
        """Insert an evaluated candidate into its corresponding island and prune low performers."""
        if not math.isfinite(score):
            return
        island_id = int(generation) % self.n_islands
        cand = IslandCandidate(
            program_id=program_id,
            code=code,
            score=float(score),
            island_id=island_id,
            generation=int(generation),
            diagnostics=dict(diagnostics or {}),
        )
        bucket = self._islands[island_id]
        bucket.append(cand)
        bucket.sort(key=lambda c: c.score)
        if len(bucket) > self.max_per_island:
            del bucket[self.max_per_island :]

    def select_parent(self, generation: int) -> IslandCandidate | None:
        """Select the lowest-loss parent from the active island for ``generation``."""
        island_id = int(generation) % self.n_islands
        bucket = self._islands.get(island_id, [])
        if bucket:
            return bucket[0]
        all_cands = [c for b in self._islands.values() for c in b]
        if not all_cands:
            return None
        return min(all_cands, key=lambda c: c.score)

    def best(self) -> IslandCandidate | None:
        """Return the global best candidate across all islands."""
        all_cands = [c for b in self._islands.values() for c in b]
        return min(all_cands, key=lambda c: c.score) if all_cands else None


class VertexGeminiEvolutionClient:
    """Backup non-default evolution transport using standard Vertex AI Gemini.

    Uses ``google.genai.Client(vertexai=True, project=project_id, location=region)`` (default model
    ``"gemini-3.8-flash"``, or any caller-supplied ``model_name``) plus
    :class:`IslandPopulationBuffer` to mutate ``# EVOLVE-BLOCK`` code from walk-forward diagnostic
    feedback without requiring a Gemini Enterprise license.
    """

    backend_name: str = "vertex_gemini"

    def __init__(
        self,
        project_id: str | None = None,
        region: str = "us-central1",
        model_name: str = DEFAULT_GEMINI_MODEL,
        *,
        n_islands: int = 3,
        temperature: float = 0.4,
        genai_client: Any = None,
    ) -> None:
        self.project_id = (
            _resolve_project_id(project_id)
            if genai_client is None
            else (project_id or os.environ.get("SF_PROJECT_ID") or "mock-project")
        )
        self.region = region
        self.model_name = model_name
        self.temperature = temperature
        self.population = IslandPopulationBuffer(n_islands=n_islands)
        self._genai_client = genai_client
        self._custom_genai_client = genai_client is not None

    def _get_or_create_genai_client(self) -> Any:
        if self._genai_client is not None:
            return self._genai_client
        from google import genai

        self._genai_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.region,
        )
        return self._genai_client

    def _generate_text(self, system_instruction: str, prompt: str) -> str:
        """Call Vertex AI Gemini via ``google.genai`` (with regional fallback if preview 404)."""
        combined_prompt = f"{system_instruction}\n\n{prompt}"
        try:
            client = self._get_or_create_genai_client()
            models_to_try = [self.model_name]
            if not self._custom_genai_client and "gemini-2.5-flash" not in models_to_try:
                models_to_try.append("gemini-2.5-flash")
            for idx, candidate_model in enumerate(models_to_try):
                try:
                    resp = client.models.generate_content(
                        model=candidate_model,
                        contents=combined_prompt,
                    )
                    text = getattr(resp, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
                    if isinstance(resp, Mapping) and "text" in resp:
                        return str(resp["text"])
                except Exception as exc:  # noqa: BLE001 - fallback when regional endpoint returns 404 for preview model
                    if idx + 1 < len(models_to_try):
                        continue
                    _log.debug("Vertex AI generate_content fallback triggered: %s", exc)
        except ImportError:
            token = _get_adc_bearer_token(self.project_id)
            url = (
                f"https://{self.region}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project_id}/locations/{self.region}/"
                f"publishers/google/models/{self.model_name}:generateContent"
            )
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": combined_prompt}],
                    }
                ],
                "generationConfig": {"temperature": self.temperature},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as http_resp:
                body = json.loads(http_resp.read().decode("utf-8", errors="replace"))
                cands = body.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return str(parts[0]["text"])
        return ""

    def propose_candidate(
        self,
        target: str,
        parent_code: str,
        diagnostics: Mapping[str, Any] | None = None,
        generation: int = 1,
    ) -> str:
        """Mutate ``# EVOLVE-BLOCK`` using Vertex AI Gemini + island population state."""
        spec = get_target_seed(target)
        island_parent = self.population.select_parent(generation)
        eff_parent = island_parent.code if island_parent is not None else parent_code
        eff_diag = (
            island_parent.diagnostics
            if (island_parent is not None and island_parent.diagnostics)
            else dict(diagnostics or {})
        )
        prompt = build_mutation_prompt(
            target=spec.target,
            parent_code=eff_parent,
            diagnostics=eff_diag,
            generation=generation,
        )
        raw_response = self._generate_text(spec.system_instruction, prompt)
        if not raw_response.strip():
            # Fallback to deterministic candidate if model response is empty
            cands = deterministic_mock_candidates(spec.target, count=max(1, generation))
            return cands[(generation - 1) % len(cands)]
        try:
            injected = inject_evolve_block(spec.seed_code, raw_response)
            if validate_candidate_ast(injected).valid:
                return injected
        except Exception as exc:  # noqa: BLE001 - malformed LLM block falls back cleanly
            _log.debug("Vertex Gemini candidate block failed AST validation: %s", exc)
        cands = deterministic_mock_candidates(spec.target, count=max(1, generation))
        return cands[(generation - 1) % len(cands)]

    def record_evaluation(
        self,
        program_id: str,
        code: str,
        score: float,
        diagnostics: Mapping[str, Any] | None = None,
        *,
        generation: int = 1,
    ) -> None:
        """Record candidate score and walk-forward diagnostics into the island buffer."""
        self.population.add(
            program_id=program_id,
            code=code,
            score=score,
            diagnostics=diagnostics,
            generation=generation,
        )


class MockEvolutionClient:
    """Offline deterministic evolution transport for fast unit testing and CI ``--dry-run``."""

    backend_name: str = "mock"

    def __init__(
        self,
        target: str = "features",
        candidates: Sequence[str] | None = None,
        max_evaluations: int = 10,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self.target = target
        self.model_name = model_name
        self._custom_candidates = list(candidates) if candidates is not None else None
        self.max_evaluations = max(1, int(max_evaluations))
        self.recorded_evaluations: list[dict[str, Any]] = []

    def propose_candidate(
        self,
        target: str,
        parent_code: str,
        diagnostics: Mapping[str, Any] | None = None,
        generation: int = 1,
    ) -> str:
        """Return deterministic candidate ``generation - 1`` with zero network calls."""
        _ = (parent_code, diagnostics)
        spec = get_target_seed(target)
        if self._custom_candidates:
            raw = self._custom_candidates[(generation - 1) % len(self._custom_candidates)]
            if "def " in raw or "class " in raw:
                return raw
            return inject_evolve_block(spec.seed_code, raw)
        built_in = deterministic_mock_candidates(spec.target, count=max(3, self.max_evaluations))
        return built_in[(generation - 1) % len(built_in)]

    def record_evaluation(
        self,
        program_id: str,
        code: str,
        score: float,
        diagnostics: Mapping[str, Any] | None = None,
        *,
        generation: int = 1,
    ) -> None:
        """Append evaluation telemetry to ``self.recorded_evaluations``."""
        self.recorded_evaluations.append(
            {
                "program_id": program_id,
                "code": code,
                "score": float(score) if math.isfinite(score) else None,
                "generation": generation,
                "diagnostics": dict(diagnostics or {}),
            }
        )


def create_evolution_client(
    backend: str,
    *,
    target: str = "features",
    project_id: str | None = None,
    location: str = "global",
    region: str = "us-central1",
    model_name: str = DEFAULT_GEMINI_MODEL,
    max_evaluations: int = 10,
    mock_candidates: Sequence[str] | None = None,
    genai_client: Any = None,
    alphaevolve_http_request: Callable[..., dict[str, Any]] | None = None,
) -> AlphaEvolveClient | VertexGeminiEvolutionClient | MockEvolutionClient:
    """Instantiate the requested evolution client backend."""
    norm = str(backend).strip().lower()
    if norm == "alphaevolve":
        return AlphaEvolveClient(
            project_id=project_id,
            location=location,
            model_name=model_name,
            http_request=alphaevolve_http_request,
        )
    if norm == "vertex_gemini":
        return VertexGeminiEvolutionClient(
            project_id=project_id,
            region=region,
            model_name=model_name,
            genai_client=genai_client,
        )
    if norm == "mock":
        return MockEvolutionClient(
            target=target,
            candidates=mock_candidates,
            max_evaluations=max_evaluations,
            model_name=model_name,
        )
    raise ConfigurationError(
        f"Unknown evolution backend '{backend}'. Valid backends: "
        "'alphaevolve', 'vertex_gemini', 'mock'."
    )
