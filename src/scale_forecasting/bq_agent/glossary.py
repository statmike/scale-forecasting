"""Forecasting Business Glossary & Knowledge Catalog synchronization.

Defines the canonical forecasting terminology shared by:
1. Inline ``Context.glossaryTerms`` on the registered BigQuery ``DataAgent``, and
2. Google Cloud **Knowledge Catalog** Business Glossaries (`glossaries`, `terms`, and
   `entryGroups/@dataplex/entryLinks` definition links attached directly to the scoped BigQuery
   registry tables and views).

BigQuery Data Agents import at most 10 linked glossary terms per table; every object's link set in
`GLOSSARY_TERMS` is capped strictly at ``<= 8`` terms per table/view so no term is silently
truncated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .rest import KNOWLEDGE_CATALOG_BASE_URL, build_authorized_session, request_json, wait_for_lro

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .context import BQAgentScope

logger = logging.getLogger(__name__)

MAX_LINKS_PER_TABLE = 8


@dataclass(frozen=True)
class GlossaryTermSpec:
    """Canonical definition of a forecasting domain term for Knowledge Catalog & Data Agents."""

    term_id: str
    display_name: str
    description: str
    labels: tuple[str, ...]
    linked_objects: tuple[str, ...]


GLOSSARY_TERMS: tuple[GlossaryTermSpec, ...] = (
    GlossaryTermSpec(
        term_id="wape",
        display_name="WAPE",
        description=(
            "Weighted Absolute Percentage Error: SUM(|y_true - yhat|) / SUM(|y_true|). "
            "Primary volume-weighted panel accuracy metric (lower is better; 0.08 = 8% error). "
            "Query pooled_wape in v_model_leaderboard_comparable for apples-to-apples rankings."
        ),
        labels=("accuracy", "metric"),
        linked_objects=(
            "forecast_metadata",
            "v_model_leaderboard",
            "v_model_leaderboard_comparable",
            "v_run_summary",
            "v_fva_waterfall",
            "v_horizon_decay",
            "v_bias_and_exceptions",
            "v_executive_sop_summary",
        ),
    ),
    GlossaryTermSpec(
        term_id="mase",
        display_name="MASE",
        description=(
            "Mean Absolute Scaled Error: MAE divided by the in-sample one-step naive MAE. "
            "Scale-free metric safe for intermittent or zero-valued series; MASE < 1.0 beats "
            "the naive random walk, while MASE > 1.0 flags WORSE_THAN_NAIVE exceptions."
        ),
        labels=("accuracy", "metric"),
        linked_objects=("forecast_metadata", "v_bias_and_exceptions"),
    ),
    GlossaryTermSpec(
        term_id="smape",
        display_name="sMAPE",
        description=(
            "Symmetric Mean Absolute Percentage Error: AVG(2 * |y_true - yhat| / "
            "(|y_true| + |yhat|)), bounded on [0, 2] and persisted per (ts_id, model_type) "
            "in forecast_metadata.metrics."
        ),
        labels=("accuracy", "metric"),
        linked_objects=("forecast_metadata",),
    ),
    GlossaryTermSpec(
        term_id="rmsse",
        display_name="RMSSE",
        description=(
            "Root Mean Squared Scaled Error (M5 Competition primary metric): squared error "
            "scaled by in-sample seasonal naive MSE, safe for intermittent retail demand."
        ),
        labels=("accuracy", "metric"),
        linked_objects=("forecast_metadata",),
    ),
    GlossaryTermSpec(
        term_id="mae",
        display_name="MAE",
        description=(
            "Mean Absolute Error: AVG(|y_true - yhat|) in native target units. Median-optimal "
            "point error metric surfaced across v_model_leaderboard and v_horizon_decay."
        ),
        labels=("accuracy", "metric"),
        linked_objects=("v_model_leaderboard",),
    ),
    GlossaryTermSpec(
        term_id="rmse",
        display_name="RMSE",
        description=(
            "Root Mean Squared Error: SQRT(AVG((y_true - yhat)^2)) in native target units. "
            "Mean-optimal accuracy metric penalizing large spike misses heavily."
        ),
        labels=("accuracy", "metric"),
        linked_objects=("v_model_leaderboard",),
    ),
    GlossaryTermSpec(
        term_id="winkler-score",
        display_name="Winkler Score",
        description=(
            "Proper scoring rule for prediction intervals [yhat_lower, yhat_upper] balancing "
            "interval sharpness (width) against a (2 / alpha) penalty for observations falling "
            "outside the interval."
        ),
        labels=("uncertainty", "metric"),
        linked_objects=("backtest_oof",),
    ),
    GlossaryTermSpec(
        term_id="pinball-loss",
        display_name="Pinball Loss",
        description=(
            "Quantile check loss evaluating asymmetric over- vs. under-prediction penalties at "
            "probability level q (for example P10, P50, P90 safety-stock quantiles)."
        ),
        labels=("uncertainty", "metric"),
        linked_objects=("backtest_oof",),
    ),
    GlossaryTermSpec(
        term_id="crps",
        display_name="CRPS",
        description=(
            "Continuous Ranked Probability Score measuring full predictive distribution "
            "calibration and sharpness across all forecasted quantiles."
        ),
        labels=("uncertainty", "metric"),
        linked_objects=("v_model_leaderboard",),
    ),
    GlossaryTermSpec(
        term_id="empirical-coverage",
        display_name="Empirical Coverage",
        description=(
            "Realized fraction of holdout observations where y_true falls inside "
            "[yhat_lower, yhat_upper], audited against nominal target coverage (e.g. 0.80) in "
            "v_backtest_coverage and v_horizon_decay."
        ),
        labels=("uncertainty", "calibration"),
        linked_objects=("v_backtest_coverage", "v_horizon_decay"),
    ),
    GlossaryTermSpec(
        term_id="fva",
        display_name="FVA",
        description=(
            "Forecast Value Added: accuracy improvement contributed by each modeling stage "
            "(statistical, covariate ML/DL, ensemble, reconciliation) over a naive baseline. "
            "Positive fva_wape_delta and fva_relative_lift_pct in v_fva_waterfall prove "
            "additional complexity earned its keep."
        ),
        labels=("governance", "value"),
        linked_objects=("v_fva_waterfall", "v_executive_sop_summary"),
    ),
    GlossaryTermSpec(
        term_id="comparable-leaderboard",
        display_name="Comparable Leaderboard",
        description=(
            "Apples-to-apples model ranking in v_model_leaderboard_comparable restricted to the "
            "newest holdout fold (MAX(fold_id)) of backtest_oof and evaluated via panel-pooled "
            "error so ragged series lengths and near-zero denominators never distort the winner."
        ),
        labels=("leaderboard", "evaluation"),
        linked_objects=("backtest_oof", "v_model_leaderboard_comparable", "v_backtest_coverage"),
    ),
    GlossaryTermSpec(
        term_id="pooled-error",
        display_name="Pooled Error",
        description=(
            "Volume-weighted panel error computed from summed absolute errors divided by summed "
            "actuals across all series in the holdout fold (pooled_wape, pooled_mae, pooled_rmse)."
        ),
        labels=("leaderboard", "evaluation"),
        linked_objects=("v_model_leaderboard_comparable",),
    ),
    GlossaryTermSpec(
        term_id="mean-error-rollup",
        display_name="Mean Error Rollup",
        description=(
            "Unweighted arithmetic average of per-series metric values in v_model_leaderboard, "
            "weighting every series equally regardless of unit volume."
        ),
        labels=("leaderboard", "evaluation"),
        linked_objects=("v_model_leaderboard",),
    ),
    GlossaryTermSpec(
        term_id="diebold-mariano",
        display_name="Diebold-Mariano Test",
        description=(
            "HAC-robust statistical significance test on horizon loss differentials verifying "
            "whether an accuracy gain over baseline is statistically distinguishable from noise."
        ),
        labels=("governance", "statistics"),
        linked_objects=("v_model_leaderboard_comparable",),
    ),
    GlossaryTermSpec(
        term_id="exception-status",
        display_name="Exception Status",
        description=(
            "Monday-morning planner triage label in v_bias_and_exceptions classifying each "
            "(ts_id, model_type) as OK, CHRONIC_OVER_FORECAST, CHRONIC_UNDER_FORECAST, "
            "WORSE_THAN_NAIVE, STALE_MODEL_DRIFT, or HIGH_ERROR."
        ),
        labels=("exceptions", "operations"),
        linked_objects=("v_bias_and_exceptions", "v_executive_sop_summary"),
    ),
    GlossaryTermSpec(
        term_id="tracking-signal",
        display_name="Tracking Signal",
        description=(
            "Trigg's cumulative signed forecast error divided by Mean Absolute Deviation: "
            "SUM(yhat - y_true) / AVG(|y_true - yhat|). Values > +4.0 indicate systematic "
            "over-forecasting (CHRONIC_OVER_FORECAST); values < -4.0 indicate systematic "
            "under-forecasting (CHRONIC_UNDER_FORECAST) in v_bias_and_exceptions."
        ),
        labels=("bias", "exceptions"),
        linked_objects=("backtest_oof", "v_bias_and_exceptions", "v_executive_sop_summary"),
    ),
    GlossaryTermSpec(
        term_id="signed-bias",
        display_name="Signed Bias",
        description=(
            "Average directional error AVG(yhat - y_true) at each horizon step; positive values "
            "indicate over-forecasting and negative values indicate under-forecasting."
        ),
        labels=("bias", "diagnostics"),
        linked_objects=("v_horizon_decay",),
    ),
    GlossaryTermSpec(
        term_id="sbc-regime",
        display_name="SBC Regime",
        description=(
            "Syntetos-Boylan demand classification (smooth, erratic, intermittent, lumpy) based "
            "on Average Inter-Demand Interval (ADI) and squared Coefficient of Variation (CV^2), "
            "persisted in forecast_metadata.fit_diagnostics and surfaced in v_bias_and_exceptions."
        ),
        labels=("demand_profile", "segmentation"),
        linked_objects=("forecast_metadata", "v_bias_and_exceptions", "v_anomalies"),
    ),
    GlossaryTermSpec(
        term_id="adi",
        display_name="ADI",
        description=(
            "Average Inter-Demand Interval: average number of periods between non-zero demand "
            "occurrences; ADI >= 1.32 separates intermittent/lumpy series from smooth/erratic."
        ),
        labels=("demand_profile", "segmentation"),
        linked_objects=("v_bias_and_exceptions",),
    ),
    GlossaryTermSpec(
        term_id="cv-squared",
        display_name="CV Squared",
        description=(
            "Squared Coefficient of Variation of non-zero demand sizes; CV^2 >= 0.49 separates "
            "erratic/lumpy demand variability from smooth/intermittent demand."
        ),
        labels=("demand_profile", "segmentation"),
        linked_objects=("v_bias_and_exceptions",),
    ),
    GlossaryTermSpec(
        term_id="staleness-gap",
        display_name="Staleness Gap",
        description=(
            "Accuracy degradation (in decision_metric units) of a frozen never-refit model "
            "compared against per-fold refitting. Quantifies how quickly model parameters go "
            "stale and drives refit cadence in v_model_leaderboard and v_backtest_coverage."
        ),
        labels=("operations", "drift"),
        linked_objects=(
            "forecast_metadata",
            "v_model_leaderboard",
            "v_backtest_coverage",
            "v_bias_and_exceptions",
        ),
    ),
    GlossaryTermSpec(
        term_id="expanding-vs-sliding-cv",
        display_name="Expanding vs Sliding CV",
        description=(
            "Rolling-origin cross-validation windowing policy: expanding grows training history "
            "from t=0 across folds, whereas sliding bounds training history to window_size."
        ),
        labels=("evaluation", "backtest"),
        linked_objects=("backtest_oof", "v_backtest_coverage"),
    ),
    GlossaryTermSpec(
        term_id="champion-ensemble",
        display_name="Champion Ensemble",
        description=(
            "Blended or per-series winning ensemble strategy persisted with model_type starting "
            "with 'ensemble_' (such as ensemble_champion, ensemble_inverse_error, ensemble_nnls, "
            "ensemble_ridge, or ensemble_xgb) and a non-null ensemble_id digest."
        ),
        labels=("ensemble", "models"),
        linked_objects=(
            "forecast_predictions",
            "backtest_oof",
            "v_model_leaderboard_comparable",
            "v_run_summary",
            "v_fva_waterfall",
            "v_executive_sop_summary",
        ),
    ),
    GlossaryTermSpec(
        term_id="learned-stacking",
        display_name="Learned Stacking",
        description=(
            "Out-of-fold meta-learner ensemble (ensemble_nnls, ensemble_ridge, ensemble_xgb) "
            "trained strictly on backtest_oof predictions to learn optimal model weights."
        ),
        labels=("ensemble", "models"),
        linked_objects=("v_model_leaderboard_comparable",),
    ),
    GlossaryTermSpec(
        term_id="calculated-blend",
        display_name="Calculated Blend",
        description=(
            "Zero-training ensemble combiner (ensemble_mean, ensemble_median, "
            "ensemble_inverse_error, ensemble_champion, ensemble_horizon_specialist) "
            "combining candidate models without fitting a meta-learner."
        ),
        labels=("ensemble", "models"),
        linked_objects=("v_model_leaderboard_comparable",),
    ),
    GlossaryTermSpec(
        term_id="covariate-ablation",
        display_name="Covariate Ablation",
        description=(
            "Controlled model variants (@univariate, @future_only, @past_only, @static_only) "
            "isolating whether exogenous features genuinely improve out-of-fold accuracy."
        ),
        labels=("features", "governance"),
        linked_objects=("forecast_metadata", "v_feature_attributions"),
    ),
    GlossaryTermSpec(
        term_id="model-spec-hash",
        display_name="Model Spec Hash",
        description=(
            "Content-addressed digest of (model_type, model_params, data, features, backtest) "
            "in forecast_metadata enabling zero-recompute reuse across identical prior runs."
        ),
        labels=("lineage", "caching"),
        linked_objects=("forecast_metadata",),
    ),
    GlossaryTermSpec(
        term_id="conformal-quantiles",
        display_name="Conformal Quantiles",
        description=(
            "Distribution-free out-of-fold calibrated prediction intervals "
            "[yhat_lower, yhat_upper] and monotone multi-quantile forecast fans stored in "
            "forecast_predictions.quantiles "
            "(read via JSON_VALUE(quantiles, '$.\"0.1\"'), '$.\"0.5\"', '$.\"0.9\"'))."
        ),
        labels=("uncertainty", "calibration"),
        linked_objects=("forecast_predictions", "v_horizon_decay", "v_anomalies"),
    ),
    GlossaryTermSpec(
        term_id="prediction-interval",
        display_name="Prediction Interval",
        description=(
            "Calibrated [yhat_lower, yhat_upper] uncertainty band around point forecast yhat at "
            "nominal coverage (1 - alpha) in forecast_predictions and backtest_oof."
        ),
        labels=("uncertainty", "calibration"),
        linked_objects=("forecast_predictions",),
    ),
    GlossaryTermSpec(
        term_id="adaptive-conformal-inference",
        display_name="Adaptive Conformal Inference",
        description=(
            "Online distribution-free interval calibration (ACI) updating quantile miscoverage "
            "alpha_t dynamically under non-stationary regime shifts without model retraining."
        ),
        labels=("uncertainty", "drift"),
        linked_objects=("v_horizon_decay",),
    ),
    GlossaryTermSpec(
        term_id="horizon-step",
        display_name="Horizon Step",
        description=(
            "1-based lead-time index (h = 1 .. H) ahead of the training cutoff; v_horizon_decay "
            "tracks how error and interval coverage evolve as lead time increases."
        ),
        labels=("horizon", "evaluation"),
        linked_objects=("forecast_predictions", "v_horizon_decay"),
    ),
    GlossaryTermSpec(
        term_id="reconciled-model",
        display_name="Reconciled Model",
        description=(
            "Coherent hierarchical or cross-temporal forecast variant suffixed with _bottom_up, "
            "_wls_struct, _mint_shrink, _mint_nonneg, _thief, or _cross_temporal_mint so leaf "
            "forecasts sum exactly to regional/category and temporal aggregates."
        ),
        labels=("hierarchy", "reconciliation"),
        linked_objects=("forecast_predictions", "v_fva_waterfall"),
    ),
    GlossaryTermSpec(
        term_id="mint-shrinkage",
        display_name="MinT Shrinkage",
        description=(
            "Minimum-Trace optimal hierarchical reconciliation using Ledoit-Wolf shrinkage on "
            "out-of-fold error covariances (_mint_shrink and non-negative QP _mint_nonneg)."
        ),
        labels=("hierarchy", "reconciliation"),
        linked_objects=("v_fva_waterfall",),
    ),
    GlossaryTermSpec(
        term_id="thief-reconciliation",
        display_name="THieF Reconciliation",
        description=(
            "Temporal Hierarchies Forecasting (_thief and _cross_temporal_mint) reconciling "
            "daily, weekly, monthly, and quarterly forecasts into one coherent trajectory."
        ),
        labels=("hierarchy", "reconciliation"),
        linked_objects=("v_fva_waterfall",),
    ),
    GlossaryTermSpec(
        term_id="compositional-simplex",
        display_name="Compositional Simplex",
        description=(
            "Isometric or Centered Log-Ratio transforms (@transform_ilr, @transform_clr) ensuring "
            "forecasted market or category shares stay strictly positive and sum to 1.0."
        ),
        labels=("transforms", "coherence"),
        linked_objects=("v_fva_waterfall",),
    ),
    GlossaryTermSpec(
        term_id="structural-break",
        display_name="Structural Break",
        description=(
            "Detected permanent level shift or regime boundary in historical demand surfaced in "
            "v_anomalies (detection_kind = 'STRUCTURAL_BREAK') from fit_diagnostics."
        ),
        labels=("anomalies", "regime"),
        linked_objects=("v_anomalies",),
    ),
    GlossaryTermSpec(
        term_id="cold-start-npi",
        display_name="Cold-Start NPI",
        description=(
            "New Product Introduction series with limited history forecasted via similarity "
            "donor pooling (@similarity_cold_start), Bass diffusion, or Weibull lifecycle curves."
        ),
        labels=("npi", "similarity"),
        linked_objects=("v_anomalies",),
    ),
    GlossaryTermSpec(
        term_id="similarity-global-pooling",
        display_name="Similarity Global Pooling",
        description=(
            "Cross-series panel training (@similarity_global) clustering analogous time series "
            "across 7 similarity lenses so related items share seasonal and promotional strength."
        ),
        labels=("similarity", "models"),
        linked_objects=("v_model_leaderboard_comparable",),
    ),
    GlossaryTermSpec(
        term_id="feature-attribution",
        display_name="Feature Attribution",
        description=(
            "Two-tier covariate driver importance unpacked in v_feature_attributions: Tier 1 "
            "GLOBAL_FIT training attributions and Tier 2 LOCAL_HORIZON step-level contributions."
        ),
        labels=("explainability", "features"),
        linked_objects=("v_feature_attributions",),
    ),
    GlossaryTermSpec(
        term_id="monotone-guardrail",
        display_name="Monotone Guardrail",
        description=(
            "Economic sign constraint (@monotone) enforcing non-positive price elasticity and "
            "non-negative promotional lift on tree and linear forecasting models."
        ),
        labels=("causal", "explainability"),
        linked_objects=("forecast_predictions", "v_feature_attributions"),
    ),
    GlossaryTermSpec(
        term_id="snapshot-millis",
        display_name="Snapshot Millis",
        description=(
            "BigQuery epoch-millisecond snapshot timestamp recorded on run_registry at launch. "
            "Historical backtest ground truth is locked in backtest_oof.y_true as of this "
            "snapshot, while post-run realized accuracy joins forecast_predictions to fresh "
            "source_table rows arriving after TIMESTAMP_MILLIS(snapshot_millis)."
        ),
        labels=("lineage", "temporal"),
        linked_objects=("run_registry", "forecast_predictions", "backtest_oof", "v_run_summary"),
    ),
    GlossaryTermSpec(
        term_id="device-verdict",
        display_name="Device Verdict",
        description=(
            "GPU FinOps hardware utilization verdict in v_run_jobs (ENGAGED_UTILISED vs. "
            "IDLE_WASTE) auditing whether provisioned GPU accelerators were actively used by "
            "model training."
        ),
        labels=("finops", "compute"),
        linked_objects=("run_jobs", "v_run_jobs", "v_executive_sop_summary"),
    ),
    GlossaryTermSpec(
        term_id="executive-sop-scorecard",
        display_name="Executive S&OP Scorecard",
        description=(
            "One-row per-run executive Sales & Operations Planning summary in "
            "v_executive_sop_summary combining total forecasted units, P10-P90 uncertainty span, "
            "champion WAPE, FVA lift over naive, exception rate, runtime, and GPU FinOps audit."
        ),
        labels=("sop", "executive"),
        linked_objects=("v_executive_sop_summary",),
    ),
)


def _slug_id(raw: str) -> str:
    """Normalize an identifier to Knowledge Catalog's ``[a-z0-9-]+`` resource ID grammar."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"sf-{cleaned}" if cleaned else "sf-glossary"
    return cleaned[:60].rstrip("-")


def default_glossary_id(dataset_id: str) -> str:
    """Return the deterministic Knowledge Catalog glossary ID for ``dataset_id``."""
    return _slug_id(f"sf-glossary-{dataset_id}")


def build_context_glossary_terms(
    terms: Sequence[GlossaryTermSpec] = GLOSSARY_TERMS,
) -> list[dict[str, Any]]:
    """Format `GLOSSARY_TERMS` for `geminidataanalytics` ``Context.glossaryTerms``."""
    return [
        {
            "displayName": term.display_name,
            "description": term.description,
            "labels": list(term.labels),
        }
        for term in terms
    ]


def table_term_links(
    terms: Sequence[GlossaryTermSpec] = GLOSSARY_TERMS,
) -> dict[str, list[GlossaryTermSpec]]:
    """Map each table/view name to its linked `GlossaryTermSpec`s (strictly ``<= 8`` per object)."""
    by_table: dict[str, list[GlossaryTermSpec]] = {}
    for term in terms:
        for obj in term.linked_objects:
            bucket = by_table.setdefault(obj, [])
            if len(bucket) < MAX_LINKS_PER_TABLE:
                bucket.append(term)
    return by_table


def sync_knowledge_catalog_glossary(
    scope: BQAgentScope,
    *,
    glossary_id: str | None = None,
    location: str = "global",
    bq_location: str = "us-central1",
    session: Any = None,
) -> dict[str, Any]:
    """Idempotently provision the Knowledge Catalog Business Glossary, Terms, and EntryLinks.

    Creates or updates ``projects/{project}/locations/{location}/glossaries/{glossary_id}``,
    upserts all `GLOSSARY_TERMS`, and creates ``definition`` ``EntryLink``s under
    ``entryGroups/@dataplex`` strictly for the scoped registry tables and views in ``scope``.
    """
    project_id = scope.project_id
    eff_glossary_id = glossary_id or default_glossary_id(scope.dataset_id)
    sess = session or build_authorized_session(project_id)

    parent = f"projects/{project_id}/locations/{location}"
    glossary_name = f"{parent}/glossaries/{eff_glossary_id}"
    glossary_url = f"{KNOWLEDGE_CATALOG_BASE_URL}/{glossary_name}"
    glossary_body = {
        "displayName": f"scale-forecasting Glossary ({scope.dataset_id})",
        "description": (
            f"Canonical forecasting metrics, FVA stages, bias diagnostics, and lineage terms "
            f"for BigQuery registry {project_id}.{scope.dataset_id}."
        ),
    }

    existing = request_json(
        sess,
        "GET",
        glossary_url,
        allow_not_found=True,
        service_label="Knowledge Catalog",
    )
    if existing is None:
        op = request_json(
            sess,
            "POST",
            f"{KNOWLEDGE_CATALOG_BASE_URL}/{parent}/glossaries",
            params={"glossaryId": eff_glossary_id},
            json_body=glossary_body,
            service_label="Knowledge Catalog",
        )
        wait_for_lro(
            sess, op, base_url=KNOWLEDGE_CATALOG_BASE_URL, service_label="Knowledge Catalog"
        )
    else:
        op = request_json(
            sess,
            "PATCH",
            glossary_url,
            params={"updateMask": "displayName,description"},
            json_body=glossary_body,
            service_label="Knowledge Catalog",
        )
        wait_for_lro(
            sess, op, base_url=KNOWLEDGE_CATALOG_BASE_URL, service_label="Knowledge Catalog"
        )

    synced_terms: list[str] = []
    for spec in GLOSSARY_TERMS:
        term_name = f"{glossary_name}/terms/{spec.term_id}"
        term_url = f"{KNOWLEDGE_CATALOG_BASE_URL}/{term_name}"
        term_body = {
            "parent": glossary_name,
            "displayName": spec.display_name,
            "description": spec.description,
            "labels": dict.fromkeys(spec.labels, "true"),
        }
        existing_term = request_json(
            sess,
            "GET",
            term_url,
            allow_not_found=True,
            service_label="Knowledge Catalog",
        )
        if existing_term is None:
            request_json(
                sess,
                "POST",
                f"{KNOWLEDGE_CATALOG_BASE_URL}/{glossary_name}/terms",
                params={"termId": spec.term_id},
                json_body=term_body,
                service_label="Knowledge Catalog",
            )
        else:
            request_json(
                sess,
                "PATCH",
                term_url,
                params={"updateMask": "displayName,description,labels"},
                json_body=term_body,
                service_label="Knowledge Catalog",
            )
        synced_terms.append(spec.term_id)

    # Attach glossary terms to scoped BigQuery entries via Knowledge Catalog EntryLinks.
    scoped_by_table = {
        ref.table_id: ref
        for ref in (*scope.registry_tables, *scope.registry_views)
        if ref.project_id == project_id and ref.dataset_id == scope.dataset_id
    }
    links_by_table = table_term_links(GLOSSARY_TERMS)
    linked_count = 0
    entry_links_parent = f"{KNOWLEDGE_CATALOG_BASE_URL}/{parent}/entryGroups/@dataplex/entryLinks"

    for table_id, term_specs in links_by_table.items():
        ref = scoped_by_table.get(table_id)
        if ref is None:
            continue
        candidate_bq_locs = [bq_location.lower()]
        if "us" not in candidate_bq_locs:
            candidate_bq_locs.append("us")
        for spec in term_specs:
            link_id = _slug_id(f"sf-{scope.dataset_id}-{table_id}-{spec.term_id}")
            link_url = f"{entry_links_parent}/{link_id}"
            existing_link = request_json(
                sess,
                "GET",
                link_url,
                allow_not_found=True,
                service_label="Knowledge Catalog",
            )
            if existing_link is not None:
                linked_count += 1
                continue
            term_entry = (
                f"projects/{project_id}/locations/{location}/entryGroups/@dataplex/entries/"
                f"projects/{project_id}/locations/{location}/glossaries/{eff_glossary_id}"
                f"/terms/{spec.term_id}"
            )
            for cand_loc in candidate_bq_locs:
                bq_entry = (
                    f"projects/{ref.project_id}/locations/{cand_loc}/entryGroups/@bigquery/"
                    f"entries/bigquery.googleapis.com/projects/{ref.project_id}/datasets/"
                    f"{ref.dataset_id}/tables/{ref.table_id}"
                )
                link_body = {
                    "entryLinkType": (
                        "projects/dataplex-types/locations/global/entryLinkTypes/definition"
                    ),
                    "entryReferences": [
                        {"name": bq_entry, "type": "SOURCE"},
                        {"name": term_entry, "type": "TARGET"},
                    ],
                }
                try:
                    request_json(
                        sess,
                        "POST",
                        entry_links_parent,
                        params={"entryLinkId": link_id},
                        json_body=link_body,
                        service_label="Knowledge Catalog",
                    )
                    linked_count += 1
                    break
                except Exception as exc:  # noqa: BLE001 - regional vs multi-region @bigquery entry fallback
                    logger.debug(
                        "Skipping optional Knowledge Catalog EntryLink %s (%s): %s",
                        link_id,
                        cand_loc,
                        exc,
                    )

    return {
        "glossary_name": glossary_name,
        "glossary_id": eff_glossary_id,
        "terms_synced": synced_terms,
        "entry_links_synced": linked_count,
    }
