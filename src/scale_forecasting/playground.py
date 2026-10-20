"""Local model-dev loop — try any model on sample data, offline.

This is the on-ramp for a data scientist opening the repo: pick a model, get a small
sample dataset, run the *real* worker cell, and read the prediction frame + metric panel —
no GCP, no config files. The notebook (``notebooks/00_model_playground.ipynb``) and the CLI
(``python -m scale_forecasting.playground``) are both thin skins over the functions here,
so the dev loop and production run the identical code path: sample → validate →
`worker.run_cell`.

Adding a model needs no change here — `available_models` reads the factory registry,
so a new file under ``models/`` that ends in ``register(...)`` appears automatically.

Public surface: ``available_models``, ``model_catalog``, ``metric_catalog``, ``sample_data``,
``build_config``, ``run_model``, ``summarize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RunConfig
from .data_gen.generator import GenConfig, generate_panel
from .metrics import METRIC_NAMES, get_metric
from .models import get_model, list_models
from .validation import validate_panel
from .worker import CellResult, run_cell

# Fixed seed so the sample panel is reproducible run-to-run (determinism, not size).
_SAMPLE_SEED = 20260726


def available_models(*, include_bigquery: bool = False, installed_only: bool = False) -> list[str]:
    """Registered model names, sorted (the factory is the source of truth).

    BigQuery-native models run as SQL in BigQuery, not in a local cell, so they're
    excluded by default — the playground runs the Python models offline. Pass
    ``installed_only=True`` to filter out models whose optional Python package is
    not installed in the current environment.
    """
    names = list_models(available_only=installed_only)
    if include_bigquery:
        return names
    return [n for n in names if get_model(n).runtime == "python"]


def model_catalog() -> pd.DataFrame:
    """Every registered model with the runtimes it can run on — the whole suite, one table.

    One row per model (Python, Vertex AI AutoML, *and* BigQuery-native), so the story is
    complete: which model runs where, what covariates it accepts, and whether it emits
    two-tier feature attributions (`explainability`). The flags are derived from the model's
    class metadata, not a hand-maintained list, so a new model file shows up here automatically:

    - **Python-runtime** models run the *identical* cell code (``worker.run_cell``) on every
      Python compute — ``local`` (this playground), ``spark`` (Dataproc fan-out), ``ray``
      (Ray on Vertex or GKE), ``vertex`` (Vertex CustomJob), ``gce`` (GCE Single-VM), and
      ``gke`` (GKE Indexed Jobs).
    - **Vertex AI AutoML** models (``runtime == "vertex_automl"``) run global cross-series
      Tabular Workflows or AutoML training jobs in the cloud (``engines/automl_engine``) and
      also provide an offline panel ridge approximation for local sandbox testing.
    - **BigQuery-native** models run only as SQL in ``bigquery`` (``engines/bigquery_engine``);
      they can't run in a local/Spark/Ray Python cell (their in-process fit/predict raise).

    Columns: ``model, family, runtime, package, package_url, available, local, spark, ray, gpu,
    vertex_automl, bigquery, exog, future_covariates, past_covariates, static_covariates,
    explainability, global_mode, hybrid_mode``.
    """
    rows: list[dict[str, Any]] = []
    for name in list_models():
        cls = get_model(name)
        is_python = cls.runtime == "python"
        rows.append(
            {
                "model": name,
                "family": cls.family,
                "runtime": cls.runtime,
                "package": cls.package,
                "package_url": cls.package_url,
                "available": cls.is_available(),
                "local": is_python,
                "spark": is_python,
                "ray": is_python,
                # The model's own capability flag, not a family guess: "deep_learning" is a
                # scheduling family, "has a tensor library under it" is what a device needs.
                "gpu": cls.gpu_capable,
                "vertex_automl": cls.runtime == "vertex_automl",
                "bigquery": cls.runtime == "bigquery",
                "exog": cls.supports_exog,
                "future_covariates": cls.supports_future_covariates,
                "past_covariates": cls.supports_past_covariates,
                "static_covariates": cls.supports_static_covariates,
                "explainability": cls.supports_explainability,
                "global_mode": cls.supports_global,
                "hybrid_mode": cls.supports_hybrid,
            }
        )
    # runtime then family then name → Python models first, native last; stable and readable.
    df = pd.DataFrame(rows).sort_values(["runtime", "family", "model"]).reset_index(drop=True)
    return df


def metric_catalog() -> pd.DataFrame:
    """Every registered evaluation metric with its direction and input requirements.

    Columns: ``metric, kind, direction, needs_intervals, needs_train_history,
    needs_seasonal_period, mean_optimal``.
    """
    rows: list[dict[str, Any]] = []
    for name in METRIC_NAMES:
        cls = get_metric(name)
        rows.append(
            {
                "metric": name,
                "kind": "interval" if cls.needs_intervals else "point",
                "direction": cls.direction,
                "needs_intervals": cls.needs_intervals,
                "needs_train_history": cls.needs_train_history,
                "needs_seasonal_period": cls.needs_seasonal_period,
                "mean_optimal": cls.mean_optimal,
            }
        )
    return pd.DataFrame(rows)


def sample_data(
    *,
    n_series: int = 3,
    history: int = 730,
    freq: str = "D",
    with_exog: bool = False,
    with_hierarchy: bool = False,
    seed: int = _SAMPLE_SEED,
) -> pd.DataFrame:
    """A small deterministic sample panel from the real generator.

    Same code path as the shipped 100k dataset, so what you see locally is what runs at
    scale. Columns: ``ts_id, archetype, ds, y`` (+ ``region, category`` when ``with_hierarchy``
    or ``with_exog``, and ``promo_flag, price_index, temperature`` when ``with_exog``).
    Defaults to 3 series × 2 years daily — enough to show trend + seasonality + intervals.
    """
    cfg = GenConfig(
        history=history,
        freq=freq,
        with_exog=with_exog,
        with_hierarchy=with_hierarchy,
    )
    return generate_panel(n_series, cfg, seed)


def build_config(
    model: str,
    *,
    freq: str = "D",
    horizon: int = 28,
    backtest: bool = False,
    with_exog: bool = False,
    static_covariates: list[str] | None = None,
    future_covariates: list[str] | None = None,
    past_covariates: list[str] | None = None,
    floor: float | None = None,
    cap: float | None = None,
    integer_rounding: str = "none",
    quantiles: list[float] | None = None,
) -> RunConfig:
    """A minimal `RunConfig` for running ``model`` on `sample_data`.

    Wires the generator's column names (``ts_id``/``ds``/``y``, and ``price_index`` as the
    exog role when ``with_exog=True``) so the config matches the sample panel out of the box.
    ``backtest`` turns on a small 3-fold CV so the metric panel is populated (otherwise metrics
    are NaN by design — full-fit runs don't score themselves).
    """
    exog = ["price_index"] if with_exog else []
    features_cfg: dict[str, Any] = {"exog": exog}
    if static_covariates:
        features_cfg["static_covariates"] = list(static_covariates)
    if future_covariates:
        features_cfg["future_covariates"] = list(future_covariates)
    if past_covariates:
        features_cfg["past_covariates"] = list(past_covariates)
    backtest_cfg: dict[str, Any] = (
        {"enabled": True, "n_folds": 3, "horizon": horizon, "step": horizon}
        if backtest
        else {"enabled": False}
    )
    output_cfg: dict[str, Any] = {"integer_rounding": integer_rounding}
    if floor is not None:
        output_cfg["floor"] = floor
    if cap is not None:
        output_cfg["cap"] = cap
    if quantiles:
        output_cfg["quantiles"] = list(quantiles)
    # Built as a plain dict and validated, mirroring config.load_config — pydantic coerces
    # the nested blocks and applies the same normalization a file-loaded config gets.
    raw: dict[str, Any] = {
        "run_name": f"playground-{model}",
        "data": {"source_table": "playground", "freq": freq, "horizon": horizon},
        "models": [model],
        "features": features_cfg,
        "backtest": backtest_cfg,
        "output": output_cfg,
    }
    return RunConfig.model_validate(raw)


@dataclass(frozen=True)
class PlaygroundRun:
    """One model run on one series — the CellResult plus the series it ran on."""

    result: CellResult
    series: pd.DataFrame
    backtest: bool

    def horizon_decay_df(self, *, metric: str = "wape") -> pd.DataFrame:
        """Compute lead-time error decay across horizon steps for this run's backtest OOF."""
        from .diagnostics import compute_horizon_decay

        if self.result.oof is None or self.result.oof.empty:
            return compute_horizon_decay(pd.DataFrame(), metric=metric)
        oof = self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
        return compute_horizon_decay(oof, metric=metric)

    def bias_report_df(self) -> pd.DataFrame:
        """Compute directional bias and Trigg Tracking Signal for this run's backtest OOF."""
        from .diagnostics import compute_bias_report

        if self.result.oof is None or self.result.oof.empty:
            return compute_bias_report(pd.DataFrame())
        oof = self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
        return compute_bias_report(oof)

    def bias_df(self) -> pd.DataFrame:
        """Alias for :meth:`bias_report_df` matching ``Forecaster.bias_df()``."""
        return self.bias_report_df()

    def rollup_time(self, freq: str = "W", *, level: float = 0.80) -> pd.DataFrame:
        """Aggregate this run's future predictions to coarser calendar buckets with AR(1) bounds."""
        from .diagnostics import rollup_time

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        return rollup_time(preds, freq=freq, oof_df=oof, level=level)

    def inventory_policy_df(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        review_period: float = 0.0,
        service_level: float | None = 0.95,
        fill_rate: float | None = None,
        on_hand: float = 0.0,
        on_order: float = 0.0,
        backorders: float = 0.0,
        holding_cost: float = 1.0,
        stockout_cost: float = 9.0,
        min_order_qty: float = 0.0,
        order_multiple: float = 0.0,
    ) -> pd.DataFrame:
        """Translate this run's calibrated forecasts into an operational inventory policy."""
        from .diagnostics import compute_inventory_policy

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        return compute_inventory_policy(
            preds,
            oof_df=oof,
            inventory_df=inventory_df,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
        )

    def plot_inventory_policy(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        review_period: float = 0.0,
        service_level: float | None = 0.95,
        fill_rate: float | None = None,
        on_hand: float = 0.0,
        on_order: float = 0.0,
        backorders: float = 0.0,
        holding_cost: float = 1.0,
        stockout_cost: float = 9.0,
        min_order_qty: float = 0.0,
        order_multiple: float = 0.0,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot Reorder Point (`ROP`), Safety Stock (`SS`), and Order Quantity for this run."""
        from .diagnostics import plot_inventory_policy

        df = self.inventory_policy_df(
            inventory_df=inventory_df,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
        )
        return plot_inventory_policy(df, ax=ax, title=title)

    def simulate_scenarios(
        self,
        scenarios: dict[str, dict[str, Any]],
        *,
        price_col: str | None = None,
        unit_cost: float | str | dict[str, float] | None = None,
        reconcile: bool = True,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        inventory_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run counterfactual future covariate scenarios on this single-series run."""
        from .scenarios import simulate_scenarios

        fut_covs = [
            c for c in ("promo_flag", "price_index", "is_holiday") if c in self.series.columns
        ]
        cfg = build_config(
            self.result.model_type,
            horizon=len(self.result.predictions) or 14,
            backtest=self.backtest,
            future_covariates=fut_covs or None,
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        return simulate_scenarios(
            self.series,
            cfg,
            scenarios,
            models=[self.result.model_type],
            price_col=price_col,
            unit_cost=unit_cost,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            oof_df=oof,
            inventory_params=inventory_params,
        )

    def promo_elasticity_df(
        self,
        *,
        promo_col: str | None = None,
        price_col: str | None = None,
        confounders: list[str] | None = None,
        summary: bool = False,
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition on this series."""
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        decomp = compute_promo_elasticity_decomposition(
            self.series,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
        )
        return summarize_promo_elasticity(decomp) if summary else decomp

    promo_elasticity = promo_elasticity_df

    def plot_promo_elasticity(
        self,
        *,
        promo_col: str | None = None,
        price_col: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot organic baseline vs. promotional lift and post-promo dip for this series."""
        from .elasticity import plot_promo_elasticity

        decomp = self.promo_elasticity_df(promo_col=promo_col, price_col=price_col, summary=False)
        return plot_promo_elasticity(decomp, ts_id=self.result.ts_id, ax=ax, title=title)

    def exceptions_df(self, *, top_n: int = 50) -> pd.DataFrame:
        """Return the exception triage queue for this playground run."""
        from .diagnostics import compute_exceptions_queue

        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        return compute_exceptions_queue(oof_df=oof, predictions_df=preds, top_n=top_n)

    def export_report(
        self,
        out_path: str | Any,
        *,
        title: str | None = None,
        write_json: bool = False,
    ) -> Any:
        """Export a self-contained `.html` report for this playground run."""
        from .report import export_report

        return export_report(self, out_path, title=title, write_json=write_json)


@dataclass(frozen=True)
class BakeOff:
    """A local multi-model bake-off on one series: the leaderboard + everything to plot it.

    ``leaderboard`` is one row per model — base models *and* the ensemble pseudo-models —
    ranked by the decision metric. ``predictions`` is the long-format future forecast for
    every model (base + ensemble) so the notebook can overlay them. ``series`` is the history
    they were fit on; ``config`` is the frozen `RunConfig` that drove the whole thing; ``oof``
    holds the combined base + ensemble out-of-fold rows for FVA, horizon-decay, and bias audits.
    """

    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    series: pd.DataFrame
    config: RunConfig
    oof: pd.DataFrame | None = None

    def fva_df(
        self,
        *,
        baseline: str | None = None,
        metric: str | None = None,
    ) -> pd.DataFrame:
        """Compute Forecast Value Added (FVA) and Diebold–Mariano p-values across models."""
        from .diagnostics import compute_fva

        eff_metric = metric or self.config.backtest.decision_metric
        return compute_fva(
            self.oof if self.oof is not None else pd.DataFrame(),
            baseline=baseline,
            metric=eff_metric,
            series_df=self.series,
        )

    def horizon_decay_df(
        self,
        *,
        metric: str | None = None,
        baseline: str | None = None,
    ) -> pd.DataFrame:
        """Compute lead-time accuracy decay by horizon step across all bake-off models."""
        from .diagnostics import compute_horizon_decay

        eff_metric = metric or self.config.backtest.decision_metric
        return compute_horizon_decay(
            self.oof if self.oof is not None else pd.DataFrame(),
            metric=eff_metric,
            baseline=baseline,
        )

    def bias_report_df(self) -> pd.DataFrame:
        """Audit each bake-off model for directional bias and Trigg Tracking Signal."""
        from .diagnostics import compute_bias_report

        return compute_bias_report(self.oof if self.oof is not None else pd.DataFrame())

    def bias_df(self) -> pd.DataFrame:
        """Alias for :meth:`bias_report_df` matching ``Forecaster.bias_df()``."""
        return self.bias_report_df()

    def rollup_time(self, freq: str = "W", *, level: float = 0.80) -> pd.DataFrame:
        """Aggregate bake-off future predictions to coarser calendar periods with AR(1) bounds."""
        from .diagnostics import rollup_time

        return rollup_time(self.predictions, freq=freq, oof_df=self.oof, level=level)

    def inventory_policy_df(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        review_period: float = 0.0,
        service_level: float | None = 0.95,
        fill_rate: float | None = None,
        on_hand: float = 0.0,
        on_order: float = 0.0,
        backorders: float = 0.0,
        holding_cost: float = 1.0,
        stockout_cost: float = 9.0,
        min_order_qty: float = 0.0,
        order_multiple: float = 0.0,
    ) -> pd.DataFrame:
        """Translate bake-off forecasts across all models into operational inventory policies."""
        from .diagnostics import compute_inventory_policy

        return compute_inventory_policy(
            self.predictions,
            oof_df=self.oof,
            inventory_df=inventory_df,
            model_name=model_name or model_type,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
        )

    def plot_inventory_policy(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        review_period: float = 0.0,
        service_level: float | None = 0.95,
        fill_rate: float | None = None,
        on_hand: float = 0.0,
        on_order: float = 0.0,
        backorders: float = 0.0,
        holding_cost: float = 1.0,
        stockout_cost: float = 9.0,
        min_order_qty: float = 0.0,
        order_multiple: float = 0.0,
        max_rows: int = 15,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot Reorder Points (`ROP`), Safety Stock (`SS`), and Order Quantities across models."""
        from .diagnostics import plot_inventory_policy

        df = self.inventory_policy_df(
            inventory_df=inventory_df,
            model_name=model_name or model_type,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
        )
        return plot_inventory_policy(df, max_rows=max_rows, ax=ax, title=title)

    def simulate_scenarios(
        self,
        scenarios: dict[str, dict[str, Any]],
        *,
        models: list[str] | None = None,
        price_col: str | None = None,
        unit_cost: float | str | dict[str, float] | None = None,
        reconcile: bool = True,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        inventory_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run counterfactual covariate scenarios across this bake-off's series and models."""
        from .scenarios import simulate_scenarios

        return simulate_scenarios(
            self.series,
            self.config,
            scenarios,
            models=models,
            price_col=price_col,
            unit_cost=unit_cost,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            oof_df=self.oof,
            inventory_params=inventory_params,
        )

    def promo_elasticity_df(
        self,
        *,
        promo_col: str | None = None,
        price_col: str | None = None,
        confounders: list[str] | None = None,
        summary: bool = False,
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition on this bake-off's
        series.
        """
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        decomp = compute_promo_elasticity_decomposition(
            self.series,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
            freq=self.config.data.freq,
            holidays=self.config.features.holidays,
        )
        return summarize_promo_elasticity(decomp) if summary else decomp

    promo_elasticity = promo_elasticity_df

    def plot_promo_elasticity(
        self,
        *,
        ts_id: str | None = None,
        promo_col: str | None = None,
        price_col: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot organic baseline vs. promotional lift and post-promo dip for this bake-off."""
        from .elasticity import plot_promo_elasticity

        decomp = self.promo_elasticity_df(promo_col=promo_col, price_col=price_col, summary=False)
        return plot_promo_elasticity(decomp, ts_id=ts_id, ax=ax, title=title)

    def exceptions_df(self, *, top_n: int = 50) -> pd.DataFrame:
        """Build the Top-N exception triage queue across all models in this bake-off."""
        from .diagnostics import compute_exceptions_queue

        return compute_exceptions_queue(
            oof_df=self.oof,
            predictions_df=self.predictions,
            floor=self.config.output.floor,
            cap=self.config.output.cap,
            metric=self.config.backtest.decision_metric,
            top_n=top_n,
        )

    def export_report(
        self,
        out_path: str | Any,
        *,
        title: str | None = None,
        write_json: bool = False,
    ) -> Any:
        """Export a self-contained interactive `.html` report for this bake-off."""
        from .report import export_report

        return export_report(self, out_path, title=title, write_json=write_json)


def bakeoff(
    data: pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    models: list[str] | None = None,
    freq: str = "D",
    horizon: int = 28,
    n_folds: int = 3,
    calculated: str = "inverse_error",
    learned: str = "nnls",
    with_exog: bool = False,
    future_covariates: list[str] | None = None,
    past_covariates: list[str] | None = None,
    static_covariates: list[str] | None = None,
    include_univariate_baselines: bool = False,
    ablation_modes: list[str] | None = None,
    ablation_models: list[str] | None = None,
    variant_policy: str = "best_per_base_model",
    model_params: dict[str, dict[str, Any]] | None = None,
    floor: float | None = None,
    cap: float | None = None,
    integer_rounding: str = "none",
    quantiles: list[float] | None = None,
) -> BakeOff:
    """Run several models head-to-head on one series and add two ensembles — via the framework.

    The playground's showcase of the *whole* pipeline offline: it runs every model in
    ``models`` on one series with backtesting on (`worker.run_cell`, the same cell a
    100k run executes), then builds **one calculated** and **one learned** ensemble by calling
    the *exact* framework functions the real ensemble stage uses — no reimplementation:

    - **calculated** (default ``inverse_error``): `ensembler.combine_calculated` blends
      the base future forecasts, weighting by ``1/decision_metric`` from the backtest.
    - **learned** (default ``nnls``): `ensembler.fit_learned` trains the meta-learner on
      the base backtest OOF, then `ensemble_run._apply_weights` applies the weights.
    - **scoring**: `ensembler.combine_oof` + `metrics.compute_metrics` /
      `worker.rollup_metrics` score every ensemble on the OOF window — the identical
      path `ensemble_run.run_ensembles` uses, minus the BigQuery I/O.

    Because it drives real framework code, the leaderboard here is what a scaled run would
    produce for the same series. ``models`` defaults to a fast, reliable subset; pass your own
    (see `available_models`) to compare any Python models. Returns a `BakeOff`.
    """
    from .backtest import training_window
    from .ensemble_run import _apply_weights
    from .ensembler import (
        calibrate_ensemble_predictions,
        combine_calculated,
        combine_oof,
        ensemble_oof_bounds,
        fit_learned,
    )
    from .features import variant_diagnostics_for
    from .metrics import METRIC_NAMES, compute_metrics
    from .seasonality import seasonal_period
    from .worker import is_panel_model, rollup_metrics, run_panel

    if models is None:
        # Fast, dependency-light defaults so the bake-off runs anywhere the repo imports.
        models = ["theta", "holtwinters", "stl_bagging", "xgboost"]
    if data is None:
        data = sample_data(
            n_series=3,
            freq=freq,
            with_exog=with_exog or bool(future_covariates),
        )

    # One config for the whole bake-off: all base models, backtest on (ensembles need OOF), and
    # both an ensemble strategy from each family. Built + validated exactly like a file config.
    exog = ["price_index"] if with_exog else []
    features_cfg: dict[str, Any] = {"exog": exog}
    if future_covariates:
        features_cfg["future_covariates"] = list(future_covariates)
    if past_covariates:
        features_cfg["past_covariates"] = list(past_covariates)
    if static_covariates:
        features_cfg["static_covariates"] = list(static_covariates)
    if include_univariate_baselines:
        features_cfg["include_univariate_baselines"] = True
    if ablation_modes:
        features_cfg["ablation_modes"] = list(ablation_modes)
    if ablation_models:
        features_cfg["ablation_models"] = list(ablation_models)
    output_cfg: dict[str, Any] = {"integer_rounding": integer_rounding}
    if floor is not None:
        output_cfg["floor"] = floor
    if cap is not None:
        output_cfg["cap"] = cap
    if quantiles:
        output_cfg["quantiles"] = list(quantiles)
    raw: dict[str, Any] = {
        "run_name": "playground-bakeoff",
        "data": {"source_table": "playground", "freq": freq, "horizon": horizon},
        "models": models,
        "features": features_cfg,
        "backtest": {"enabled": True, "n_folds": n_folds, "horizon": horizon, "step": horizon},
        "output": output_cfg,
        "ensemble": {
            "enabled": True,
            "strategies": [calculated, learned],
            "variant_policy": variant_policy,
        },
    }
    if model_params:
        raw["model_params"] = dict(model_params)
    cfg = RunConfig.model_validate(raw)
    validate_panel(data, cfg)

    ts_col = cfg.data.ts_id_col
    chosen = ts_id if ts_id is not None else str(data[ts_col].iloc[0])
    series = data[data[ts_col] == chosen].reset_index(drop=True)
    if series.empty:
        available = ", ".join(sorted(data[ts_col].unique().astype(str))[:10])
        raise ValueError(f"ts_id '{chosen}' not in data; available: {available}")

    decision = cfg.backtest.decision_metric

    # --- run every base model through the real worker cell -----------------------------------
    base_rows: list[dict[str, Any]] = []  # leaderboard rows
    pred_parts: list[pd.DataFrame] = []  # long-format future forecasts (for the overlay plot)
    base_pred_long: list[pd.DataFrame] = []  # feeds the calculated/learned blends
    oof_long: list[pd.DataFrame] = []  # feeds combine_oof / fit_learned
    metric_long: list[dict[str, Any]] = []  # feeds inverse_error weighting

    for model in cfg.models:
        if is_panel_model(model, cfg):
            panel_results = run_panel(data, model, cfg)
            result = next((r for r in panel_results if r.ts_id == chosen), panel_results[0])
        else:
            result = run_cell(series, model, cfg)
        status = result.status
        vdiag = variant_diagnostics_for(model, cfg)
        row: dict[str, Any] = {
            "model": model,
            "base_model": vdiag["base_model"],
            "scope_capability": vdiag["scope_capability"],
            "training_mode": vdiag["training_mode"],
            "cov_profile": vdiag["cov_profile"],
            "is_univariate": vdiag["is_univariate"],
            "covariates_used": vdiag["covariates_used"],
            "model_spec_hash": vdiag["model_spec_hash"],
            "kind": "base",
            "status": status,
        }
        row.update({m: result.metrics.get(m, float("nan")) for m in METRIC_NAMES})
        base_rows.append(row)
        if status != "ok":
            continue  # an errored model still shows on the board, but can't feed the ensembles

        pred = result.predictions.rename(columns={"ds": "forecast_date"}).assign(
            ts_id=chosen, model_type=model
        )
        pred_parts.append(pred.assign(model=model, kind="base"))
        base_pred_long.append(
            pred[["ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper"]]
        )
        metric_long.append(
            {"ts_id": chosen, "model_type": model, decision: result.metrics.get(decision)}
        )
        if result.oof is not None and not result.oof.empty:
            oof_long.append(
                result.oof.rename(columns={"ds": "forecast_date"}).assign(
                    ts_id=chosen, model_type=model
                )
            )

    base_df = pd.concat(base_pred_long, ignore_index=True) if base_pred_long else pd.DataFrame()
    oof_df = pd.concat(oof_long, ignore_index=True) if oof_long else pd.DataFrame()
    metric_df = pd.DataFrame(metric_long)

    # --- ensembles: calculated + learned, both via the framework's pure functions ------------
    ens_pred_rows: list[dict[str, Any]] = list(combine_calculated(base_df, cfg, metric_df, oof_df))
    learned_weights, _artifacts, _basis = fit_learned(oof_df, cfg)
    for strategy, wmap in learned_weights.items():
        ens_pred_rows.extend(_apply_weights(base_df, wmap, cfg.run_name, strategy))

    ens_oof = combine_oof(oof_df, cfg, learned_weights) if not oof_df.empty else pd.DataFrame()
    ens_pred_rows = calibrate_ensemble_predictions(ens_pred_rows, ens_oof, cfg)

    for r in ens_pred_rows:
        pred_parts.append(
            pd.DataFrame(
                [
                    {
                        "forecast_date": r["forecast_date"],
                        "yhat": r["yhat"],
                        "yhat_lower": r.get("yhat_lower"),
                        "yhat_upper": r.get("yhat_upper"),
                        "ts_id": chosen,
                        "model_type": r["model_type"],
                        "model": r["model_type"],
                        "kind": "ensemble",
                    }
                ]
            )
        )

    # --- score the ensembles on the OOF window (identical to ensemble_run) --------------------
    hist = series.sort_values(cfg.data.date_col)
    hist_ds = pd.to_datetime(hist[cfg.data.date_col]).to_numpy()
    hist_y = hist[cfg.data.target_col].to_numpy()
    for model_type, g in ens_oof.groupby("model_type"):
        fold_panels: list[dict[str, float]] = []
        for _fold, fg in g.sort_values("forecast_date").groupby("fold_id"):
            lo, hi = ensemble_oof_bounds(g, _fold, cfg)
            fold_panels.append(
                compute_metrics(
                    fg["y_true"].to_numpy(),
                    fg["yhat"].to_numpy(),
                    lower=lo,
                    upper=hi,
                    y_train=training_window(
                        hist_ds,
                        hist_y,
                        fg["cutoff_date"].min() if "cutoff_date" in fg.columns else None,
                        cfg,
                    ),
                    seasonal_period=seasonal_period(cfg.data.freq),
                )
            )
        panel = rollup_metrics(fold_panels)
        row = {
            "model": str(model_type),
            "base_model": str(model_type),
            "scope_capability": "ensemble",
            "training_mode": "ensemble",
            "cov_profile": "ensemble",
            "is_univariate": None,
            "covariates_used": None,
            "model_spec_hash": None,
            "kind": "ensemble",
            "status": "ok",
        }
        row.update({m: panel.get(m, float("nan")) for m in METRIC_NAMES})
        base_rows.append(row)

    leaderboard = (
        pd.DataFrame(base_rows).sort_values(decision, na_position="last").reset_index(drop=True)
    )
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    all_oof_parts = [p for p in (oof_df, ens_oof) if not p.empty]
    all_oof = pd.concat(all_oof_parts, ignore_index=True) if all_oof_parts else pd.DataFrame()
    return BakeOff(
        leaderboard=leaderboard,
        predictions=predictions,
        series=series,
        config=cfg,
        oof=all_oof,
    )


def run_model(
    model: str,
    data: pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    freq: str = "D",
    horizon: int = 28,
    backtest: bool = False,
    with_exog: bool = False,
    future_covariates: list[str] | None = None,
    floor: float | None = None,
    cap: float | None = None,
    integer_rounding: str = "none",
    quantiles: list[float] | None = None,
) -> PlaygroundRun:
    """Run one model on one series and return the result (validates first).

    ``data`` defaults to a fresh `sample_data` panel (matching ``freq``/``with_exog``).
    The whole panel is validated up front (`validation.validate_panel`) so a shape
    problem fails fast with a clear message — the same guard a real run uses. Then a single
    series (``ts_id``, or the first one) is handed to `worker.run_cell`, exactly as an
    engine would. ``run_cell`` never raises: a model failure comes back as an error cell.
    """
    if data is None:
        data = sample_data(
            n_series=3,
            freq=freq,
            with_exog=with_exog or bool(future_covariates),
        )

    cfg = build_config(
        model,
        freq=freq,
        horizon=horizon,
        backtest=backtest,
        with_exog=with_exog,
        future_covariates=future_covariates,
        floor=floor,
        cap=cap,
        integer_rounding=integer_rounding,
        quantiles=quantiles,
    )
    validate_panel(data, cfg)

    ts_col = cfg.data.ts_id_col
    chosen = ts_id if ts_id is not None else str(data[ts_col].iloc[0])
    series = data[data[ts_col] == chosen].reset_index(drop=True)
    if series.empty:
        available = ", ".join(sorted(data[ts_col].unique().astype(str))[:10])
        raise ValueError(f"ts_id '{chosen}' not in data; available: {available}")

    result = run_cell(series, model, cfg)
    return PlaygroundRun(result=result, series=series, backtest=backtest)


def summarize(run: PlaygroundRun) -> str:
    """A compact human-readable summary of a run — for the CLI and notebook printout."""
    r = run.result
    lines = [
        f"model      : {r.model_type}",
        f"series     : {r.ts_id}  ({len(run.series)} obs)",
        f"engine     : {r.compute_engine}",
        f"status     : {r.status}",
    ]
    if r.status == "error":
        lines.append(f"error      : {r.error}")
        return "\n".join(lines)

    lines.append(f"forecast   : {len(r.predictions)} steps")
    if not r.predictions.empty:
        first, last = r.predictions["ds"].iloc[0], r.predictions["ds"].iloc[-1]
        lines.append(f"horizon    : {first.date()} → {last.date()}")
    if run.backtest:
        panel = ", ".join(
            f"{k}={v:.3f}"
            for k, v in r.metrics.items()
            if v == v  # skip NaN
        )
        lines.append(f"metrics    : {panel or '(none)'}")
    else:
        lines.append("metrics    : (backtest off — run with backtest=True to score)")
    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    """``python -m scale_forecasting.playground`` — the dev loop from the terminal.

    Secondary to the notebook (which is the first-class dev surface), this exists for CI
    and quick checks. ``--list`` shows the models; otherwise run one and print the summary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="scale_forecasting.playground",
        description="Run one forecasting model on sample data, offline.",
    )
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    parser.add_argument("--model", help="model name to run (see --list)")
    parser.add_argument("--freq", default="D", help="frequency (D, W, MS, ME, h)")
    parser.add_argument("--horizon", type=int, default=28, help="forecast horizon (steps)")
    parser.add_argument("--backtest", action="store_true", help="score with a 3-fold backtest")
    parser.add_argument("--exog", action="store_true", help="include the example exog driver")
    args = parser.parse_args(argv)

    if args.list or not args.model:
        print("available models:")
        for name in available_models():
            print(f"  {name}")
        if not args.model:
            print("\nrun one with: --model <name> [--backtest] [--exog] [--freq D]")
        return 0

    run = run_model(
        args.model,
        freq=args.freq,
        horizon=args.horizon,
        backtest=args.backtest,
        with_exog=args.exog,
    )
    print(summarize(run))
    return 0 if run.result.status == "ok" else 1


if __name__ == "__main__":  # pragma: no cover - exercised via _main in tests
    raise SystemExit(_main())
