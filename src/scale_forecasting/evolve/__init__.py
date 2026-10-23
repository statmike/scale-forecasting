"""Opt-In AlphaEvolve & Vertex AI Gemini Evolutionary Harness (`scale_forecasting.evolve`).

Provides opt-in runtime evolutionary code synthesis built directly on top of
:meth:`~scale_forecasting.sdk.Forecaster.auto` (`auto.py` / `auto_policy.py`) and governed by a
**3-Gate Anti-Overfitting Ratchet**:

* **Dual-Backend Evolution Transport (:mod:`~scale_forecasting.evolve.client`):**
  - ``backend="alphaevolve"`` (**Default**): Managed Gemini Enterprise Discovery Engine AlphaEvolve
    REST API (``discoveryengine.googleapis.com/v1alpha``), guarded by a fast pre-flight access probe
    (:func:`probe_alphaevolve_access`) before any backtests run.
  - ``backend="vertex_gemini"`` (**Backup Non-Default**): Standard Vertex AI Gemini
    (``aiplatform.googleapis.com``, default ``"gemini-3.8-flash"`` or any user-specified
    ``model_name``) with an in-memory multi-island MAP-Elites population buffer
    (:class:`IslandPopulationBuffer`) for projects without Gemini Enterprise licensing.
  - ``backend="mock"`` (**Offline Test Backend**): Deterministic zero-network candidate replay.
* **Five ``# EVOLVE-BLOCK`` Targets (:mod:`~scale_forecasting.evolve.targets`):** ``"features"``
  (:func:`synthesize_evolved_features`), ``"ensemble"`` (:func:`compute_evolved_weights`),
  ``"model"`` (:class:`EvolvedForecaster`), ``"policy"`` (:func:`evolve_policy_weights`), and
  ``"similarity_kernel"`` (:func:`compute_evolved_similarity_kernel`).
* **AST Security & Temporal Causality Sandbox (:mod:`~scale_forecasting.evolve.sandbox`):**
  Static AST whitelist (:func:`validate_candidate_ast`) plus the $+10^6$ future-target perturbation
  test (:func:`verify_no_future_leakage`).
* **3-Gate Anti-Overfitting Ratchet (:mod:`~scale_forecasting.evolve.evaluator` &
  :mod:`~scale_forecasting.evolve.runner`):**
  Gate 1 (Inner Walk-Forward CV Lift), Gate 2 (Diebold–Mariano significance & worst-decile FVA
  guard), and Gate 3 (Held-Out Lockdown Fold never seen by the evolution loop).
"""

from __future__ import annotations

from .client import (
    ALPHAEVOLVE_BASE_URL_TEMPLATE,
    DEFAULT_GEMINI_MODEL,
    AlphaEvolveClient,
    ConfigurationError,
    IslandCandidate,
    IslandPopulationBuffer,
    MockEvolutionClient,
    VertexGeminiEvolutionClient,
    create_evolution_client,
    probe_alphaevolve_access,
)
from .evaluator import (
    CandidateEvaluation,
    EvolutionEvaluator,
    split_inner_and_lockdown_panel,
    verify_oof_pool_compatibility,
)
from .runner import (
    EvolveConfig,
    EvolveResult,
    run_evolution,
)
from .sandbox import (
    ALLOWED_MODULES,
    PROHIBITED_CALLS,
    PROHIBITED_MODULES,
    SandboxValidationResult,
    compile_candidate_symbol,
    execute_in_sandbox,
    validate_candidate_ast,
    verify_no_future_leakage,
)
from .targets import (
    EVOLVE_BLOCK_END,
    EVOLVE_BLOCK_START,
    EVOLVE_TARGETS,
    OVERFIT_LOCKDOWN_CANDIDATES,
    AutoPolicyWeights,
    EvolvedForecaster,
    TargetSpec,
    build_mutation_prompt,
    compute_evolved_similarity_kernel,
    compute_evolved_weights,
    deterministic_mock_candidates,
    evolve_policy_weights,
    extract_evolve_block,
    get_target_seed,
    inject_evolve_block,
    synthesize_evolved_features,
)

__all__ = [
    "ALLOWED_MODULES",
    "ALPHAEVOLVE_BASE_URL_TEMPLATE",
    "DEFAULT_GEMINI_MODEL",
    "EVOLVE_BLOCK_END",
    "EVOLVE_BLOCK_START",
    "EVOLVE_TARGETS",
    "OVERFIT_LOCKDOWN_CANDIDATES",
    "PROHIBITED_CALLS",
    "PROHIBITED_MODULES",
    "AlphaEvolveClient",
    "AutoPolicyWeights",
    "CandidateEvaluation",
    "ConfigurationError",
    "EvolutionEvaluator",
    "EvolveConfig",
    "EvolveResult",
    "EvolvedForecaster",
    "IslandCandidate",
    "IslandPopulationBuffer",
    "MockEvolutionClient",
    "SandboxValidationResult",
    "TargetSpec",
    "VertexGeminiEvolutionClient",
    "build_mutation_prompt",
    "compile_candidate_symbol",
    "compute_evolved_similarity_kernel",
    "compute_evolved_weights",
    "create_evolution_client",
    "deterministic_mock_candidates",
    "evolve_policy_weights",
    "execute_in_sandbox",
    "extract_evolve_block",
    "get_target_seed",
    "inject_evolve_block",
    "main",
    "probe_alphaevolve_access",
    "run_evolution",
    "split_inner_and_lockdown_panel",
    "synthesize_evolved_features",
    "validate_candidate_ast",
    "verify_no_future_leakage",
    "verify_oof_pool_compatibility",
]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point wrapper delegating to :func:`scale_forecasting.evolve.__main__.main`."""
    from .__main__ import main as _cli_main

    return _cli_main(argv)
