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
    explainability, monotonic_constraints, global_mode, hybrid_mode``.
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
                "monotonic_constraints": cls.supports_monotonic_constraints,
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


def _resolve_pg_panel(
    series: pd.DataFrame,
    panel_df: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    *,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Resolve a multi-series panel for `PlaygroundRun` / `BakeOff` similarity methods."""
    import numpy as np

    chosen = (
        data
        if (data is not None and not data.empty)
        else (panel_df if (panel_df is not None and not panel_df.empty) else series)
    )
    if chosen is None or chosen.empty:
        return sample_data(n_series=3, history=90)
    work = chosen.copy()
    if ts_id_col not in work.columns:
        work[ts_id_col] = "series_0"
    if work[ts_id_col].nunique() >= 2:
        return work
    base_id = str(work[ts_id_col].iloc[0])
    peer = work.copy()
    peer[ts_id_col] = f"{base_id}__peer"
    if target_col in peer.columns:
        y_vals = pd.to_numeric(peer[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        peer[target_col] = 0.92 * y_vals + 0.08 * np.roll(y_vals, 1)
    return pd.concat([work, peer], ignore_index=True)


@dataclass(frozen=True)
class PlaygroundRun:
    """One model run on one series — the CellResult plus the series it ran on."""

    result: CellResult
    series: pd.DataFrame
    backtest: bool
    panel_df: pd.DataFrame | None = None

    @property
    def predictions(self) -> pd.DataFrame:
        """Convenience alias for ``self.result.predictions``."""
        return self.result.predictions

    @property
    def oof(self) -> pd.DataFrame:
        """Convenience alias for ``self.result.oof``."""
        return self.result.oof  # type: ignore[return-value]

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

    def similarity_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str = "hybrid",
        top_k: int | None = None,
        max_lag: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute tidy pairwise trajectory similarity across the playground panel
        (`similarity.compute_similarity_matrix`).
        """
        from .similarity import compute_similarity_matrix

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        return compute_similarity_matrix(
            panel,
            method=method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            top_k=int(top_k) if top_k is not None else 10,
            **kwargs,
        ).pairs_df

    def similarity_matrix_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str = "hybrid",
        max_lag: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute the symmetric $(N, N)$ trajectory similarity matrix DataFrame
        (`similarity.compute_similarity_matrix`).
        """
        from .similarity import compute_similarity_matrix

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        return compute_similarity_matrix(
            panel,
            method=method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            **kwargs,
        ).matrix_df

    def similarity_clusters_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        n_clusters: int | str = "auto",
        method: str = "hybrid",
        min_cluster_size: int = 2,
        max_clusters: int = 8,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Partition series into behavioral similarity clusters
        (`similarity.cluster_by_similarity_df`)."""
        from .similarity import cluster_by_similarity_df

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        return cluster_by_similarity_df(
            panel,
            n_clusters=n_clusters,  # type: ignore[arg-type]
            method=method,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            **kwargs,
        )

    def similarity_embeddings_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str = "hybrid",
        n_components: int = 2,
        include_foundation: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Extract per-series physical/econometric invariants and 2D coordinates
        (`similarity.embed_series_df`)."""
        from .similarity import embed_series_df

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = "foundation_latent" if include_foundation and method == "hybrid" else method
        return embed_series_df(
            panel,
            method=eff_method,
            n_components=n_components,
            **kwargs,
        )

    def find_similar_series_df(
        self,
        query_ts_id_or_data: str | pd.DataFrame | None = None,
        *,
        query_ts_id: str | None = None,
        data: pd.DataFrame | None = None,
        top_k: int = 5,
        mode: str = "full_history",
        method: str = "hybrid",
        prefix_len: int | None = None,
        max_lag: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Retrieve Top-$K$ similar peer series (`similarity.find_similar_series_df`)."""
        from .similarity import find_similar_series_df as _find_similar_series_df

        eff_data = data
        eff_query = query_ts_id
        if isinstance(query_ts_id_or_data, pd.DataFrame):
            eff_data = query_ts_id_or_data
        elif isinstance(query_ts_id_or_data, str) and eff_query is None:
            eff_query = query_ts_id_or_data

        panel = _resolve_pg_panel(self.series, self.panel_df, eff_data)
        return _find_similar_series_df(
            panel,
            query_ts_id=eff_query or self.result.ts_id,
            top_k=top_k,
            mode=mode,
            method=method,
            prefix_len=prefix_len,
            max_lag=max_lag,
            **kwargs,
        )

    def plot_similarity_map(
        self,
        sim_or_clusters_or_panel: Any = None,
        *,
        data: pd.DataFrame | None = None,
        query_ts_id: str | None = None,
        n_clusters: int | str = "auto",
        method: str = "hybrid",
        ax: Any = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render the 2-panel trajectory similarity heatmap & 2D MDS cluster map
        (`similarity.plot_similarity_map`).
        """
        from .similarity import plot_similarity_map as _plot_similarity_map

        target = (
            sim_or_clusters_or_panel
            if sim_or_clusters_or_panel is not None
            else _resolve_pg_panel(self.series, self.panel_df, data)
        )
        return _plot_similarity_map(
            target,
            query_ts_id=query_ts_id or self.result.ts_id,
            n_clusters=n_clusters,  # type: ignore[arg-type]
            method=method,
            ax=ax,
            title=title,
            **kwargs,
        )

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
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        target_portfolio_fill_rate: float | None = None,
        unit_cost: float | dict[str, float] | str = 1.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        vendor_moq: float | dict[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Translate this run's calibrated forecasts into an operational inventory policy."""
        if (
            max_budget is not None
            or max_volume is not None
            or max_total_units is not None
            or target_portfolio_fill_rate is not None
            or vendor_moq is not None
            or uncertainty_source == "copula"
            or trajectories_df is not None
            or copula_paths is not None
        ):
            out = self.optimize_inventory_df(
                inventory_df=inventory_df,
                trajectories_df=trajectories_df,
                copula_paths=copula_paths,
                lead_time=lead_time,
                lead_time_std=lead_time_std,
                review_period=review_period,
                service_level=service_level,
                fill_rate=fill_rate,
                target_portfolio_fill_rate=target_portfolio_fill_rate,
                max_budget=max_budget,
                max_volume=max_volume,
                max_total_units=max_total_units,
                holding_cost=holding_cost,
                stockout_cost=stockout_cost,
                unit_cost=unit_cost,
                unit_volume=unit_volume,
                min_order_qty=min_order_qty,
                order_multiple=order_multiple,
                vendor_moq=vendor_moq,
                on_hand=on_hand,
                on_order=on_order,
                backorders=backorders,
                uncertainty_source=uncertainty_source,
                copula_type=copula_type,
                n_paths=n_paths,
                seed=seed,
                return_result=False,
            )
            assert isinstance(out, pd.DataFrame)
            return out

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
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        target_portfolio_fill_rate: float | None = None,
        unit_cost: float | dict[str, float] | str = 1.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        vendor_moq: float | dict[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
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
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
            target_portfolio_fill_rate=target_portfolio_fill_rate,
            unit_cost=unit_cost,
            unit_volume=unit_volume,
            vendor_moq=vendor_moq,
            uncertainty_source=uncertainty_source,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            copula_type=copula_type,
            n_paths=n_paths,
            seed=seed,
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
        use_similarity_priors: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str = "hybrid",
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition on this series."""
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        eff_use_sim = (
            bool(similarity_pooling)
            and (similarity_weight is None or float(similarity_weight) > 0.0)
            if similarity_pooling is not None
            else (
                bool(use_similarity_priors)
                and (similarity_weight is None or float(similarity_weight) > 0.0)
            )
        )
        src_panel = (
            self.panel_df
            if (self.panel_df is not None and not self.panel_df.empty and eff_use_sim)
            else self.series
        )
        decomp = compute_promo_elasticity_decomposition(
            src_panel,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=similarity_matrix,
            similarity_method=similarity_method,
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

    def temporal_reconcile_df(
        self,
        *,
        method: str = "mint_shrink",
        aggregations: list[int] | None = None,
        temporal_aggregations: list[int] | None = None,
        stamp_method: bool = False,
    ) -> pd.DataFrame:
        """Reconcile this run's forecast across temporal aggregations (`THieF`)."""
        from .temporal_reconciliation import reconcile_temporal_forecasts

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_aggs = temporal_aggregations if temporal_aggregations is not None else aggregations
        rec = reconcile_temporal_forecasts(
            preds,
            history_df=self.series,
            oof_df=oof,
            temporal_aggregations=eff_aggs,
            method=method,
            stamp_method=stamp_method,
        )
        assert isinstance(rec, pd.DataFrame)
        return rec

    def sample_coherent_paths(
        self,
        *,
        n_paths: int = 200,
        seed: int = 42,
        method: str = "mint_shrink",
        nonneg: bool = False,
    ) -> pd.DataFrame:
        """Draw ``n_paths`` coherent probabilistic trajectory scenarios for this playground run."""
        from .temporal_reconciliation import sample_coherent_paths as _sample_coherent_paths

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        return _sample_coherent_paths(
            preds,
            history_df=self.series,
            oof_df=oof,
            n_paths=n_paths,
            seed=seed,
            method=method,
            nonneg=nonneg,
        )

    def allocate_target(
        self,
        target_total: Any = None,
        *,
        target: Any = None,
        targets: Any = None,
        target_sum: Any = None,
        locked_nodes: Any = None,
        locks: Any = None,
        lower_bounds: Any = None,
        upper_bounds: Any = None,
        bounds: Any = None,
        floor: float | None = 0.0,
        cap: float | None = None,
        method: str = "mint_shrink",
        weights: Any = None,
        smoothness_weight: float = 0.0,
        price_col: str | dict[str, float] | float | None = None,
        prices: dict[str, float] | float | pd.DataFrame | None = None,
        unit_cost: float | dict[str, float] | str | None = None,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        inventory_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Allocate a top-down target across this series' forecast horizon
        (`goal_seek.allocate_target`).
        """
        from .goal_seek import allocate_target as _allocate_target

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        out = _allocate_target(
            preds,
            target_total=target_total,
            target=target,
            targets=targets,
            target_sum=target_sum,
            locked_nodes=locked_nodes,
            locks=locks,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            bounds=bounds,
            floor=floor,
            cap=cap,
            method=method,
            weights=weights,
            smoothness_weight=smoothness_weight,
            history_df=self.series,
            oof_df=oof,
            price_col=price_col,
            prices=prices,
            unit_cost=unit_cost,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            inventory_params=inventory_params,
        )
        assert isinstance(out, pd.DataFrame)
        return out

    def plot_goal_seek(
        self,
        goal_seek_df: Any,
        *,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot baseline vs. goal-seek allocated trajectory for this playground run."""
        from .goal_seek import plot_goal_seek as _plot_goal_seek

        return _plot_goal_seek(goal_seek_df, ts_id=self.result.ts_id, ax=ax, title=title)

    def cold_start_transfer_df(
        self,
        *,
        min_obs: int | None = None,
        min_history_points: int = 14,
        method: Any = None,
        matching: str | None = None,
        cold_start_matching: str | None = None,
        similarity_method: str | None = None,
        top_k_donors: int = 3,
        donor_pool_df: pd.DataFrame | None = None,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Synthesize cold-start / NPI trajectories via analog donor matching (`cold_start`)."""
        from .cold_start import cold_start_transfer_df as _cold_start_transfer_df

        _ = return_result
        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        panel = (
            donor_pool_df
            if donor_pool_df is not None
            else (self.panel_df if self.panel_df is not None else self.series)
        )
        eff_matching = cold_start_matching if cold_start_matching is not None else matching
        return _cold_start_transfer_df(
            panel,
            forecasts_df=preds,
            oof_df=oof,
            min_obs=min_obs if min_obs is not None else min_history_points,
            method=method,
            matching=eff_matching,  # type: ignore[arg-type]
            similarity_method=similarity_method,  # type: ignore[arg-type]
            top_k_donors=top_k_donors,
            horizon=len(self.result.predictions) or 14,
        )

    def plot_cold_start_transfer(
        self,
        transfer_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot cold-start transferred forecast vs. donor trajectories (`cold_start`)."""
        from .cold_start import plot_cold_start_transfer

        _ = title
        eff_df = transfer_df if transfer_df is not None else self.cold_start_transfer_df()
        return plot_cold_start_transfer(
            eff_df,
            panel_df=self.series,
            ts_id=ts_id or self.result.ts_id,
            ax=ax,
        )

    def aci_df(
        self,
        *,
        actuals_df: pd.DataFrame | None = None,
        target_coverage: float = 0.80,
        target_alpha: float | None = None,
        gamma: float = 0.05,
        normalized: bool = True,
        method: Any = "aci_normalized",
        horizon_delay: bool = True,
        window: int | None = None,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Compute Adaptive Conformal Inference (ACI) online coverage tracking (`aci`)."""
        from .aci import compute_aci_df

        _ = (window, return_result)
        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_cov = (1.0 - float(target_alpha)) if target_alpha is not None else target_coverage
        source_oof = (
            oof
            if (oof is not None and not oof.empty)
            else (
                actuals_df
                if actuals_df is not None
                else self.series.assign(
                    y_true=self.series["y"],
                    yhat=self.series["y"].rolling(3, min_periods=1).mean(),
                    model_type=self.result.model_type,
                )
            )
        )
        return compute_aci_df(
            source_oof,
            predictions_df=preds if not preds.empty else None,
            gamma=gamma,
            target_coverage=eff_cov,
            normalized=normalized,
            method=method,
            horizon_delay=horizon_delay,
        )

    def plot_aci(
        self,
        aci_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot ACI rolling empirical coverage (`aci.plot_aci`)."""
        from .aci import plot_aci

        _ = title
        eff_df = aci_df if aci_df is not None else self.aci_df()
        return plot_aci(eff_df, ts_id=ts_id or self.result.ts_id, ax=ax)

    def sample_trajectories(
        self,
        *,
        n_paths: int = 500,
        copula_type: str = "auto",
        df_t: float = 4.0,
        cross_series: bool = True,
        reconcile: bool = True,
        similarity_shrinkage: bool = True,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str = "hybrid",
        floor: float | None = None,
        cap: float | None = None,
        seed: int = 42,
        nonneg: bool = False,
        return_result: bool = False,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        """Generate joint copula trajectory paths (`copula.sample_trajectories`)."""
        from .copula import sample_trajectories as _sample_trajectories

        _ = return_result
        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_floor = 0.0 if (nonneg and floor is None) else floor
        return _sample_trajectories(
            preds,
            oof_df=oof,
            history_df=self.series,
            n_paths=n_paths,
            copula_type=copula_type,
            df_t=df_t,
            cross_series=cross_series,
            reconcile=reconcile,
            similarity_shrinkage=similarity_shrinkage,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=similarity_matrix,
            similarity_method=similarity_method,
            floor=eff_floor,
            cap=cap,
            seed=seed,
        )

    def tail_risk_df(
        self,
        trajectories_df: pd.DataFrame | None = None,
        *,
        actuals_df: pd.DataFrame | None = None,
        lead_time: int | float | None = None,
        window: int = 7,
        alphas: tuple[float, ...] = (0.90, 0.95, 0.99),
        alpha: float = 0.95,
        threshold: float | dict[str, float] | None = None,
        holding_cost: float | None = None,
        stockout_cost: float | None = None,
        n_paths: int = 300,
        copula_type: str = "auto",
        reconcile: bool = True,
        similarity_shrinkage: bool = True,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str = "hybrid",
        seed: int = 42,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        """Compute VaR and CVaR tail-risk summary (`copula.compute_tail_risk`)."""
        from .copula import compute_tail_risk

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(
                n_paths=n_paths,
                copula_type=copula_type,
                reconcile=reconcile,
                similarity_shrinkage=similarity_shrinkage,
                similarity_weight=similarity_weight,
                similarity_matrix=similarity_matrix,
                similarity_method=similarity_method,
                seed=seed,
            )
        )
        return compute_tail_risk(
            eff_traj,
            predictions_df=preds,
            actuals_df=actuals_df,
            oof_df=oof,
            lead_time=lead_time,
            window=window,
            alphas=alphas,
            alpha=alpha,
            threshold=threshold,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
        )

    def plot_trajectories(
        self,
        trajectories_df: pd.DataFrame | None = None,
        *,
        tail_risk: pd.DataFrame | None = None,
        ts_id: str | None = None,
        max_spaghetti_paths: int = 40,
        max_paths_shown: int | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot joint copula trajectory spaghetti fan (`copula.plot_trajectories`)."""
        from .copula import plot_trajectories

        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(n_paths=200)
        )
        return plot_trajectories(
            eff_traj,
            tail_risk=tail_risk,
            ts_id=ts_id or self.result.ts_id,
            max_spaghetti_paths=(
                max_paths_shown if max_paths_shown is not None else max_spaghetti_paths
            ),
            ax=ax,
            title=title,
        )

    def optimize_inventory_df(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        lead_time: float | dict[str, float] = 7.0,
        lead_time_std: float | dict[str, float] = 0.0,
        review_period: float | dict[str, float] = 0.0,
        service_level: float | str | dict[str, float | str] | None = 0.95,
        fill_rate: float | dict[str, float] | None = None,
        target_portfolio_fill_rate: float | None = None,
        min_portfolio_fill_rate: float | None = None,
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        holding_cost: float | dict[str, float] = 1.0,
        stockout_cost: float | dict[str, float] = 9.0,
        unit_cost: float | dict[str, float] | str = 10.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        min_order_qty: float | dict[str, float] = 0.0,
        order_multiple: float | dict[str, float] | None = None,
        vendor_moq: float | dict[str, float] | None = None,
        on_hand: float | dict[str, float] = 0.0,
        on_order: float | dict[str, float] = 0.0,
        backorders: float | dict[str, float] = 0.0,
        uncertainty_source: str = "copula",
        copula_type: str = "auto",
        similarity_shrinkage: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str = "hybrid",
        n_paths: int = 300,
        seed: int = 42,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Solve portfolio-constrained inventory replenishment (`inventory_optimizer`)."""
        from .inventory_optimizer import optimize_inventory_portfolio

        _ = return_result
        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_source: Any = "analytical" if uncertainty_source == "ar1" else uncertainty_source
        eff_uc = unit_cost if not isinstance(unit_cost, str) else 10.0
        eff_uv = unit_volume if not isinstance(unit_volume, str) else 1.0
        eff_sim_shrink = (
            bool(similarity_pooling)
            if similarity_pooling is not None
            else bool(similarity_shrinkage)
        )
        return optimize_inventory_portfolio(
            preds,
            oof_df=oof,
            history_df=self.series,
            inventory_df=inventory_df,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            target_portfolio_fill_rate=target_portfolio_fill_rate,
            min_portfolio_fill_rate=min_portfolio_fill_rate,
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            unit_cost=eff_uc,
            unit_volume=eff_uv,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
            vendor_moq=vendor_moq,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            uncertainty_source=eff_source,
            copula_type=copula_type,
            similarity_shrinkage=eff_sim_shrink,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=similarity_matrix,
            similarity_method=similarity_method,
            n_paths=n_paths,
            seed=seed,
        )

    def plot_inventory_frontier(
        self,
        opt_result_or_df: Any = None,
        *,
        budgets: list[float] | None = None,
        n_points: int = 8,
        max_rows: int = 15,
        ax: Any = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot working-capital vs. expected fill-rate Pareto frontier (`inventory_optimizer`)."""
        from .inventory_optimizer import plot_inventory_frontier

        target = (
            opt_result_or_df
            if opt_result_or_df is not None
            else self.optimize_inventory_df(**kwargs)
        )
        return plot_inventory_frontier(
            target,
            budgets=budgets,
            n_points=n_points,
            max_rows=max_rows,
            ax=ax,
            title=title,
        )

    def optimize_scenario(
        self,
        *,
        future_covariates_df: pd.DataFrame | None = None,
        trajectories_df: pd.DataFrame | None = None,
        objective: Any = "gross_profit",
        price_col: str | None = None,
        promo_col: str | None = None,
        price_bounds: tuple[float, float] = (-0.15, 0.15),
        price_grid: Any = None,
        promo_bounds: tuple[float, float] = (0.0, 1.0),
        binary_promo: bool = True,
        max_promo_days: int | dict[str, int] | None = None,
        max_promo_periods: int | dict[str, int] | None = None,
        min_cooldown_days: int = 0,
        min_promo_gap: int = 0,
        min_margin_pct: float | None = None,
        min_gross_margin: float | None = None,
        min_total_units: float | None = None,
        max_total_units: float | None = None,
        target_total_units: float | None = None,
        max_promo_budget: float | None = None,
        inventory_cap: float | dict[str, float] | None = None,
        unit_cost: float | dict[str, float] | None = 4.0,
        promo_fixed_cost: float | dict[str, float] = 0.0,
        risk_aversion_lambda: float = 0.0,
        cvar_alpha: float = 0.95,
        pull_forward_lags: int = 3,
        cannibalization_weight: float = 1.0,
        cannibalization_strength: float | None = None,
        use_similarity_priors: bool = True,
        cross_sku_elasticity: bool | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str = "hybrid",
        reconcile: bool = True,
        include_inventory_impact: bool = True,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float | None = 0.95,
        n_paths: int = 250,
        copula_type: str = "auto",
        seed: int = 42,
        **_kwargs: Any,
    ) -> Any:
        """Solve prescriptive price & promotional calendar optimization (`scenario_optimizer`)."""
        from .scenario_optimizer import optimize_scenario as _optimize_scenario

        preds = self.result.predictions.assign(
            ts_id=self.result.ts_id, model_type=self.result.model_type
        )
        oof = (
            self.result.oof.assign(ts_id=self.result.ts_id, model_type=self.result.model_type)
            if self.result.oof is not None and not self.result.oof.empty
            else None
        )
        eff_cannib = (
            float(cannibalization_strength)
            if cannibalization_strength is not None
            else float(cannibalization_weight)
        )
        eff_use_sim = (
            bool(cross_sku_elasticity)
            if cross_sku_elasticity is not None
            else bool(use_similarity_priors)
        )
        return _optimize_scenario(
            self.series,
            predictions_df=preds,
            oof_df=oof,
            history_df=self.series,
            future_covariates_df=future_covariates_df,
            trajectories_df=trajectories_df,
            model_type=self.result.model_type,
            objective=objective,
            price_col=price_col,
            promo_col=promo_col,
            price_bounds=price_bounds,
            price_grid=price_grid,
            promo_bounds=promo_bounds,
            binary_promo=binary_promo,
            max_promo_days=max_promo_days if max_promo_days is not None else max_promo_periods,
            min_cooldown_days=min_cooldown_days if min_cooldown_days else min_promo_gap,
            unit_cost=unit_cost,
            promo_fixed_cost=promo_fixed_cost,
            min_margin_pct=min_margin_pct,
            min_gross_margin=min_gross_margin,
            min_total_units=min_total_units,
            max_total_units=max_total_units,
            target_total_units=target_total_units,
            max_promo_budget=max_promo_budget,
            inventory_cap=inventory_cap,
            risk_aversion_lambda=risk_aversion_lambda,
            cvar_alpha=cvar_alpha,
            pull_forward_lags=pull_forward_lags,
            cannibalization_weight=eff_cannib,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=similarity_matrix,
            similarity_method=similarity_method,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            n_paths=n_paths,
            copula_type=copula_type,
            seed=seed,
            horizon=len(self.result.predictions) or 14,
        )

    def plot_optimal_scenario(
        self,
        opt_result_or_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        """Plot optimal price/promo schedule (`scenario_optimizer.plot_optimal_scenario`)."""
        from .scenario_optimizer import plot_optimal_scenario

        target = opt_result_or_df if opt_result_or_df is not None else self.optimize_scenario()
        return plot_optimal_scenario(
            target,
            ts_id=ts_id or self.result.ts_id,
            ax=ax,
            title=title,
        )

    def export_report(
        self,
        out_path: str | Any,
        *,
        title: str | None = None,
        goal_seek_df: pd.DataFrame | None = None,
        aci_df: pd.DataFrame | None = None,
        tail_risk_df: pd.DataFrame | None = None,
        cold_start_df: pd.DataFrame | None = None,
        inventory_opt_df: pd.DataFrame | None = None,
        scenario_opt_df: pd.DataFrame | None = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_clusters_df: pd.DataFrame | None = None,
        write_json: bool = False,
        include_json_sidecar: bool = False,
    ) -> Any:
        """Export a self-contained `.html` report for this playground run."""
        from .report import export_report

        return export_report(
            self,
            out_path,
            title=title,
            goal_seek_df=goal_seek_df,
            aci_df=aci_df,
            tail_risk_df=tail_risk_df,
            cold_start_df=cold_start_df,
            inventory_opt_df=inventory_opt_df,
            scenario_opt_df=scenario_opt_df,
            similarity_df=similarity_df,
            similarity_clusters_df=similarity_clusters_df,
            write_json=write_json,
            include_json_sidecar=include_json_sidecar,
        )


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
    panel_df: pd.DataFrame | None = None

    def similarity_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        top_k: int | None = None,
        max_lag: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute tidy pairwise time-series similarity edges
        (`similarity.compute_similarity_matrix`)."""
        from .similarity import compute_similarity_matrix

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = method or self.config.features.similarity_method
        res = compute_similarity_matrix(
            panel,
            cfg=self.config,
            method=eff_method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            top_k=int(top_k) if top_k is not None else 10,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            freq=self.config.data.freq,
            **kwargs,
        )
        return res.pairs_df

    def similarity_matrix_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        max_lag: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute square `(N, N)` similarity DataFrame indexed by `ts_id`."""
        from .similarity import compute_similarity_matrix

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = method or self.config.features.similarity_method
        res = compute_similarity_matrix(
            panel,
            cfg=self.config,
            method=eff_method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            freq=self.config.data.freq,
            **kwargs,
        )
        return res.matrix_df

    def similarity_clusters_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        n_clusters: int | str = "auto",
        method: str | None = None,
        min_cluster_size: int = 2,
        max_clusters: int = 8,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Partition series into behavioral similarity clusters
        (`similarity.cluster_by_similarity_df`)."""
        from .similarity import cluster_by_similarity_df

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = method or self.config.features.similarity_method
        eff_k = (
            self.config.features.similarity_n_clusters
            if n_clusters == "auto" and self.config.features.similarity_n_clusters != "auto"
            else n_clusters
        )
        return cluster_by_similarity_df(
            panel,
            cfg=self.config,
            n_clusters=eff_k,  # type: ignore[arg-type]
            method=eff_method,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            freq=self.config.data.freq,
            **kwargs,
        )

    def similarity_embeddings_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        n_components: int = 2,
        include_foundation: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Extract per-series physical & econometric invariants and 2D coordinates."""
        from .similarity import embed_series_df

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = (
            "foundation_latent"
            if include_foundation and method is None
            else (method or self.config.features.similarity_method)
        )
        return embed_series_df(
            panel,
            cfg=self.config,
            method=eff_method,
            n_components=n_components,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            freq=self.config.data.freq,
            **kwargs,
        )

    def find_similar_series_df(
        self,
        query_ts_id_or_data: str | pd.DataFrame | None = None,
        *,
        query_ts_id: str | None = None,
        data: pd.DataFrame | None = None,
        top_k: int = 5,
        mode: str = "full_history",
        method: str | None = None,
        prefix_len: int | None = None,
        max_lag: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Rank top-K most similar peer series to ``query_ts_id``."""
        from .similarity import find_similar_series_df as _find_similar_series_df

        eff_data = data
        eff_query = query_ts_id
        if isinstance(query_ts_id_or_data, pd.DataFrame):
            eff_data = query_ts_id_or_data
        elif isinstance(query_ts_id_or_data, str) and eff_query is None:
            eff_query = query_ts_id_or_data

        panel = _resolve_pg_panel(self.series, self.panel_df, eff_data)
        ts_col = self.config.data.ts_id_col
        eff_query = (
            eff_query
            if eff_query is not None
            else str(
                self.series[ts_col].iloc[0]
                if ts_col in self.series.columns and not self.series.empty
                else panel[ts_col].iloc[0]
            )
        )
        eff_method = method or self.config.features.similarity_method
        return _find_similar_series_df(
            panel,
            query_ts_id=eff_query,
            cfg=self.config,
            top_k=top_k,
            mode=mode,
            method=eff_method,
            prefix_len=prefix_len,
            max_lag=max_lag,
            ts_id_col=ts_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            freq=self.config.data.freq,
            **kwargs,
        )

    def plot_similarity_map(
        self,
        similarity_or_clusters_df: Any = None,
        *,
        data: pd.DataFrame | None = None,
        query_ts_id: str | None = None,
        method: str | None = None,
        n_clusters: int | str = "auto",
        ax: Any = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a 2-panel similarity heatmap and MDS behavioral cluster projection."""
        from .similarity import plot_similarity_map as _plot_similarity_map

        panel = _resolve_pg_panel(self.series, self.panel_df, data)
        eff_method = method or self.config.features.similarity_method
        return _plot_similarity_map(
            similarity_or_clusters_df if similarity_or_clusters_df is not None else panel,
            cfg=self.config,
            query_ts_id=query_ts_id,
            method=eff_method,
            n_clusters=n_clusters,  # type: ignore[arg-type]
            ax=ax,
            title=title,
            **kwargs,
        )

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
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        target_portfolio_fill_rate: float | None = None,
        unit_cost: float | dict[str, float] | str = 1.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        vendor_moq: float | dict[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Translate bake-off forecasts across all models into operational inventory policies."""
        if (
            max_budget is not None
            or max_volume is not None
            or max_total_units is not None
            or target_portfolio_fill_rate is not None
            or vendor_moq is not None
            or uncertainty_source == "copula"
            or trajectories_df is not None
            or copula_paths is not None
        ):
            out = self.optimize_inventory_df(
                inventory_df=inventory_df,
                trajectories_df=trajectories_df,
                copula_paths=copula_paths,
                model_name=model_name or model_type,
                lead_time=lead_time,
                lead_time_std=lead_time_std,
                review_period=review_period,
                service_level=service_level,
                fill_rate=fill_rate,
                target_portfolio_fill_rate=target_portfolio_fill_rate,
                max_budget=max_budget,
                max_volume=max_volume,
                max_total_units=max_total_units,
                holding_cost=holding_cost,
                stockout_cost=stockout_cost,
                unit_cost=unit_cost,
                unit_volume=unit_volume,
                min_order_qty=min_order_qty,
                order_multiple=order_multiple,
                vendor_moq=vendor_moq,
                on_hand=on_hand,
                on_order=on_order,
                backorders=backorders,
                uncertainty_source=uncertainty_source,
                copula_type=copula_type,
                n_paths=n_paths,
                seed=seed,
                return_result=False,
            )
            assert isinstance(out, pd.DataFrame)
            return out

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
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        target_portfolio_fill_rate: float | None = None,
        unit_cost: float | dict[str, float] | str = 1.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        vendor_moq: float | dict[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
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
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
            target_portfolio_fill_rate=target_portfolio_fill_rate,
            unit_cost=unit_cost,
            unit_volume=unit_volume,
            vendor_moq=vendor_moq,
            uncertainty_source=uncertainty_source,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            copula_type=copula_type,
            n_paths=n_paths,
            seed=seed,
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
        use_similarity_priors: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_method: str = "hybrid",
        pooling_strength: float = 0.5,
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition on this bake-off's
        series.
        """
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        _ = pooling_strength
        eff_use_sim = (
            bool(similarity_pooling)
            and (similarity_weight is None or float(similarity_weight) > 0.0)
            if similarity_pooling is not None
            else (
                bool(use_similarity_priors)
                and (similarity_weight is None or float(similarity_weight) > 0.0)
            )
        )
        panel_src = self.panel_df if (eff_use_sim and self.panel_df is not None) else self.series
        decomp = compute_promo_elasticity_decomposition(
            panel_src,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
            freq=self.config.data.freq,
            holidays=self.config.features.holidays,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=(
                similarity_matrix if similarity_matrix is not None else similarity_df
            ),
            similarity_method=similarity_method,
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

    def temporal_reconcile_df(
        self,
        *,
        method: str = "mint_shrink",
        model_type: str | None = None,
        aggregations: list[int] | None = None,
        temporal_aggregations: list[int] | None = None,
    ) -> pd.DataFrame:
        """Reconcile bake-off forecasts across temporal aggregations (`THieF`)."""
        from .temporal_reconciliation import reconcile_temporal_forecasts

        eff_aggs = temporal_aggregations if temporal_aggregations is not None else aggregations
        rec = reconcile_temporal_forecasts(
            self.predictions,
            history_df=self.series,
            oof_df=self.oof,
            cfg=self.config,
            freq=self.config.data.freq,
            horizon=self.config.data.horizon,
            temporal_aggregations=eff_aggs,
            method=method,
            model_type=model_type,
        )
        assert isinstance(rec, pd.DataFrame)
        return rec

    def sample_coherent_paths(
        self,
        *,
        model_type: str | None = None,
        n_paths: int = 200,
        seed: int = 42,
        method: str = "mint_shrink",
        nonneg: bool = False,
    ) -> pd.DataFrame:
        """Draw ``n_paths`` coherent probabilistic trajectory scenarios for this bake-off."""
        from .temporal_reconciliation import sample_coherent_paths as _sample_coherent_paths

        eff_model = model_type
        if eff_model is None and not self.leaderboard.empty:
            for col in ("model_type", "model"):
                if col in self.leaderboard.columns:
                    eff_model = str(self.leaderboard[col].iloc[0])
                    break
        preds = self.predictions
        if eff_model is not None and "model_type" in preds.columns:
            preds = preds[preds["model_type"].astype(str) == eff_model].copy()
        return _sample_coherent_paths(
            preds,
            history_df=self.series,
            oof_df=self.oof,
            cfg=self.config,
            n_paths=n_paths,
            seed=seed,
            method=method,
            nonneg=nonneg,
        )

    def allocate_target(
        self,
        target_total: Any = None,
        *,
        target: Any = None,
        targets: Any = None,
        target_sum: Any = None,
        model_type: str | None = None,
        locked_nodes: Any = None,
        locks: Any = None,
        lower_bounds: Any = None,
        upper_bounds: Any = None,
        bounds: Any = None,
        floor: float | None = 0.0,
        cap: float | None = None,
        method: str = "mint_shrink",
        weights: Any = None,
        smoothness_weight: float = 0.0,
        price_col: str | dict[str, float] | float | None = None,
        prices: dict[str, float] | float | pd.DataFrame | None = None,
        unit_cost: float | dict[str, float] | str | None = None,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        inventory_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Allocate a top-down target across this bake-off's forecasts
        (`goal_seek.allocate_target`).
        """
        from .goal_seek import allocate_target as _allocate_target

        eff_model = model_type
        if eff_model is None and not self.leaderboard.empty:
            for col in ("model_type", "model"):
                if col in self.leaderboard.columns:
                    eff_model = str(self.leaderboard[col].iloc[0])
                    break
        out = _allocate_target(
            self.predictions,
            target_total=target_total,
            target=target,
            targets=targets,
            target_sum=target_sum,
            model_type=eff_model,
            locked_nodes=locked_nodes,
            locks=locks,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            bounds=bounds,
            floor=floor,
            cap=cap,
            method=method,
            weights=weights,
            smoothness_weight=smoothness_weight,
            cfg=self.config,
            history_df=self.series,
            oof_df=self.oof,
            price_col=price_col,
            prices=prices,
            unit_cost=unit_cost,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            inventory_params=inventory_params,
        )
        assert isinstance(out, pd.DataFrame)
        return out

    def plot_goal_seek(
        self,
        goal_seek_df: Any,
        *,
        ts_id: str | None = None,
        top_n: int = 10,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot baseline vs. goal-seek allocated forecasts for this bake-off."""
        from .goal_seek import plot_goal_seek as _plot_goal_seek

        return _plot_goal_seek(
            goal_seek_df,
            ts_id=ts_id,
            top_n=top_n,
            ax=ax,
            title=title,
        )

    def _champion_model(self, model_type: str | None = None) -> str | None:
        if model_type is not None:
            return model_type
        if not self.leaderboard.empty:
            for col in ("model_type", "model"):
                if col in self.leaderboard.columns:
                    return str(self.leaderboard[col].iloc[0])
        return None

    def cold_start_transfer_df(
        self,
        *,
        min_obs: int | None = None,
        min_history_points: int = 14,
        method: Any = None,
        matching: str | None = None,
        similarity_method: str | None = None,
        top_k_donors: int = 3,
        donor_pool_df: pd.DataFrame | None = None,
        model_type: str | None = None,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Synthesize cold-start / NPI trajectories via analog donor matching (`cold_start`)."""
        from .cold_start import cold_start_transfer_df as _cold_start_transfer_df

        _ = return_result
        eff_model = self._champion_model(model_type)
        preds = self.predictions
        if eff_model is not None and "model_type" in preds.columns:
            preds = preds[preds["model_type"].astype(str) == eff_model].copy()
        panel = (
            donor_pool_df
            if donor_pool_df is not None
            else (self.panel_df if self.panel_df is not None else self.series)
        )
        return _cold_start_transfer_df(
            panel,
            cfg=self.config,
            forecasts_df=preds,
            oof_df=self.oof,
            min_obs=min_obs if min_obs is not None else min_history_points,
            method=method,
            matching=matching,  # type: ignore[arg-type]
            similarity_method=similarity_method,  # type: ignore[arg-type]
            static_covariates=self.config.features.static_covariates,
            hierarchy_cols=[c for lvl in self.config.hierarchy.levels for c in lvl],
            top_k_donors=top_k_donors,
            ts_id_col=self.config.data.ts_id_col,
            date_col=self.config.data.date_col,
            target_col=self.config.data.target_col,
            horizon=self.config.data.horizon,
            freq=self.config.data.freq,
        )

    def plot_cold_start_transfer(
        self,
        transfer_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot cold-start transferred forecast vs. donor trajectories (`cold_start`)."""
        from .cold_start import plot_cold_start_transfer

        _ = title
        eff_df = transfer_df if transfer_df is not None else self.cold_start_transfer_df()
        return plot_cold_start_transfer(
            eff_df,
            panel_df=self.series,
            ts_id=ts_id,
            ax=ax,
        )

    def aci_df(
        self,
        *,
        actuals_df: pd.DataFrame | None = None,
        target_coverage: float = 0.80,
        target_alpha: float | None = None,
        gamma: float | None = None,
        normalized: bool = True,
        method: Any = None,
        horizon_delay: bool = True,
        window: int | None = None,
        model_type: str | None = None,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Compute Adaptive Conformal Inference (ACI) online coverage tracking (`aci`)."""
        from .aci import compute_aci_df

        _ = (window, return_result)
        eff_model = self._champion_model(model_type)
        preds = self.predictions
        oof = self.oof
        if eff_model is not None:
            if "model_type" in preds.columns:
                preds = preds[preds["model_type"].astype(str) == eff_model].copy()
            if oof is not None and not oof.empty and "model_type" in oof.columns:
                oof = oof[oof["model_type"].astype(str) == eff_model].copy()
        eff_gamma = gamma if gamma is not None else float(self.config.output.aci_gamma)
        eff_cov = (1.0 - float(target_alpha)) if target_alpha is not None else target_coverage
        eff_method = method or self.config.output.calibration_method
        if eff_method == "conformal":
            eff_method = "aci_normalized" if normalized else "aci"
        source_oof = (
            oof
            if (oof is not None and not oof.empty)
            else (actuals_df if actuals_df is not None else preds)
        )
        return compute_aci_df(
            source_oof,
            predictions_df=preds if not preds.empty else None,
            gamma=eff_gamma,
            target_coverage=eff_cov,
            normalized=normalized,
            method=eff_method,
            horizon_delay=horizon_delay,
            ts_id_col=self.config.data.ts_id_col,
        )

    def plot_aci(
        self,
        aci_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot ACI rolling empirical coverage (`aci.plot_aci`)."""
        from .aci import plot_aci

        _ = title
        eff_df = aci_df if aci_df is not None else self.aci_df()
        return plot_aci(eff_df, ts_id=ts_id, ax=ax)

    def sample_trajectories(
        self,
        *,
        n_paths: int = 500,
        copula_type: str = "auto",
        df_t: float = 4.0,
        cross_series: bool = True,
        similarity_shrinkage: bool = True,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_method: str = "hybrid",
        reconcile: bool = True,
        reconcile_hierarchy: bool = True,
        floor: float | None = None,
        cap: float | None = None,
        seed: int = 42,
        nonneg: bool = False,
        model_type: str | None = None,
        return_result: bool = False,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        """Generate joint copula trajectory paths (`copula.sample_trajectories`)."""
        from .copula import sample_trajectories as _sample_trajectories

        _ = return_result
        eff_model = self._champion_model(model_type)
        preds = self.predictions
        oof = self.oof
        if eff_model is not None:
            if "model_type" in preds.columns:
                preds = preds[preds["model_type"].astype(str) == eff_model].copy()
            if oof is not None and not oof.empty and "model_type" in oof.columns:
                oof = oof[oof["model_type"].astype(str) == eff_model].copy()
        eff_floor = floor if floor is not None else (0.0 if nonneg else self.config.output.floor)
        eff_cap = cap if cap is not None else self.config.output.cap
        return _sample_trajectories(
            preds,
            oof_df=oof,
            history_df=self.panel_df if self.panel_df is not None else self.series,
            cfg=self.config,
            n_paths=n_paths,
            copula_type=copula_type,
            df_t=df_t,
            cross_series=cross_series,
            similarity_shrinkage=similarity_shrinkage,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=(
                similarity_matrix if similarity_matrix is not None else similarity_df
            ),
            similarity_method=similarity_method,
            reconcile=reconcile and reconcile_hierarchy,
            floor=eff_floor,
            cap=eff_cap,
            seed=seed,
        )

    def tail_risk_df(
        self,
        trajectories_df: pd.DataFrame | None = None,
        *,
        actuals_df: pd.DataFrame | None = None,
        lead_time: int | float | None = None,
        window: int = 7,
        alphas: tuple[float, ...] = (0.90, 0.95, 0.99),
        alpha: float = 0.95,
        threshold: float | dict[str, float] | None = None,
        holding_cost: float | None = None,
        stockout_cost: float | None = None,
        n_paths: int = 300,
        copula_type: str = "auto",
        similarity_shrinkage: bool = False,
        similarity_df: pd.DataFrame | None = None,
        seed: int = 42,
        model_type: str | None = None,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        """Compute VaR and CVaR tail-risk summary (`copula.compute_tail_risk`)."""
        from .copula import compute_tail_risk

        eff_model = self._champion_model(model_type)
        preds = self.predictions
        oof = self.oof
        if eff_model is not None:
            if "model_type" in preds.columns:
                preds = preds[preds["model_type"].astype(str) == eff_model].copy()
            if oof is not None and not oof.empty and "model_type" in oof.columns:
                oof = oof[oof["model_type"].astype(str) == eff_model].copy()
        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(
                n_paths=n_paths,
                copula_type=copula_type,
                similarity_shrinkage=similarity_shrinkage,
                similarity_df=similarity_df,
                seed=seed,
                model_type=model_type,
            )
        )
        return compute_tail_risk(
            eff_traj,
            predictions_df=preds,
            actuals_df=actuals_df,
            oof_df=oof,
            lead_time=lead_time,
            window=window,
            alphas=alphas,
            alpha=alpha,
            threshold=threshold,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
        )

    def plot_trajectories(
        self,
        trajectories_df: pd.DataFrame | None = None,
        *,
        tail_risk: pd.DataFrame | None = None,
        ts_id: str | None = None,
        max_spaghetti_paths: int = 40,
        max_paths_shown: int | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot joint copula trajectory spaghetti fan (`copula.plot_trajectories`)."""
        from .copula import plot_trajectories

        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(n_paths=200)
        )
        return plot_trajectories(
            eff_traj,
            tail_risk=tail_risk,
            ts_id=ts_id,
            max_spaghetti_paths=(
                max_paths_shown if max_paths_shown is not None else max_spaghetti_paths
            ),
            ax=ax,
            title=title,
        )

    def optimize_inventory_df(
        self,
        *,
        inventory_df: pd.DataFrame | dict[str, dict[str, Any]] | None = None,
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        lead_time: float | dict[str, float] = 7.0,
        lead_time_std: float | dict[str, float] = 0.0,
        review_period: float | dict[str, float] = 0.0,
        service_level: float | str | dict[str, float | str] | None = 0.95,
        fill_rate: float | dict[str, float] | None = None,
        target_portfolio_fill_rate: float | None = None,
        min_portfolio_fill_rate: float | None = None,
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        holding_cost: float | dict[str, float] = 1.0,
        stockout_cost: float | dict[str, float] = 9.0,
        unit_cost: float | dict[str, float] | str = 10.0,
        unit_volume: float | dict[str, float] | str = 1.0,
        min_order_qty: float | dict[str, float] = 0.0,
        order_multiple: float | dict[str, float] | None = None,
        vendor_moq: float | dict[str, float] | None = None,
        on_hand: float | dict[str, float] = 0.0,
        on_order: float | dict[str, float] = 0.0,
        backorders: float | dict[str, float] = 0.0,
        uncertainty_source: str = "copula",
        copula_type: str = "auto",
        similarity_shrinkage: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_method: str = "hybrid",
        n_paths: int = 300,
        seed: int = 42,
        return_result: bool = False,
    ) -> pd.DataFrame:
        """Solve portfolio-constrained inventory replenishment (`inventory_optimizer`)."""
        from .inventory_optimizer import optimize_inventory_portfolio

        _ = return_result
        eff_model = self._champion_model(model_name or model_type)
        eff_source: Any = "analytical" if uncertainty_source == "ar1" else uncertainty_source
        eff_uc = unit_cost if not isinstance(unit_cost, str) else 10.0
        eff_uv = unit_volume if not isinstance(unit_volume, str) else 1.0
        eff_sim_shrink = (
            bool(similarity_pooling)
            if similarity_pooling is not None
            else bool(similarity_shrinkage)
        )
        return optimize_inventory_portfolio(
            self.predictions,
            oof_df=self.oof,
            inventory_df=inventory_df,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            model_name=eff_model,
            model_type=eff_model,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level,
            fill_rate=fill_rate,
            target_portfolio_fill_rate=target_portfolio_fill_rate,
            min_portfolio_fill_rate=min_portfolio_fill_rate,
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            unit_cost=eff_uc,
            unit_volume=eff_uv,
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
            vendor_moq=vendor_moq,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            uncertainty_source=eff_source,
            copula_type=copula_type,
            similarity_shrinkage=eff_sim_shrink,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=(
                similarity_matrix if similarity_matrix is not None else similarity_df
            ),
            similarity_method=similarity_method,
            history_df=self.panel_df if self.panel_df is not None else self.series,
            n_paths=n_paths,
            seed=seed,
        )

    def plot_inventory_frontier(
        self,
        opt_result_or_df: Any = None,
        *,
        budgets: list[float] | None = None,
        n_points: int = 8,
        max_rows: int = 15,
        ax: Any = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot working-capital vs. expected fill-rate Pareto frontier (`inventory_optimizer`)."""
        from .inventory_optimizer import plot_inventory_frontier

        target = (
            opt_result_or_df
            if opt_result_or_df is not None
            else self.optimize_inventory_df(**kwargs)
        )
        return plot_inventory_frontier(
            target,
            budgets=budgets,
            n_points=n_points,
            max_rows=max_rows,
            ax=ax,
            title=title,
        )

    def optimize_scenario(
        self,
        *,
        future_covariates_df: pd.DataFrame | None = None,
        trajectories_df: pd.DataFrame | None = None,
        objective: Any = "gross_profit",
        price_col: str | None = None,
        promo_col: str | None = None,
        price_bounds: tuple[float, float] = (-0.15, 0.15),
        price_grid: Any = None,
        promo_bounds: tuple[float, float] = (0.0, 1.0),
        binary_promo: bool = True,
        max_promo_days: int | dict[str, int] | None = None,
        max_promo_periods: int | dict[str, int] | None = None,
        min_cooldown_days: int = 0,
        min_promo_gap: int = 0,
        min_margin_pct: float | None = None,
        min_gross_margin: float | None = None,
        min_total_units: float | None = None,
        max_total_units: float | None = None,
        target_total_units: float | None = None,
        max_promo_budget: float | None = None,
        inventory_cap: float | dict[str, float] | None = None,
        unit_cost: float | dict[str, float] | None = 4.0,
        promo_fixed_cost: float | dict[str, float] = 0.0,
        risk_aversion_lambda: float = 0.0,
        cvar_alpha: float = 0.95,
        pull_forward_lags: int = 3,
        cannibalization_weight: float = 1.0,
        cannibalization_strength: float | None = None,
        use_similarity_priors: bool = True,
        cross_sku_elasticity: bool | None = None,
        similarity_cannibalization: bool | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_method: str = "hybrid",
        reconcile: bool = True,
        include_inventory_impact: bool = True,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float | None = 0.95,
        n_paths: int = 250,
        copula_type: str = "auto",
        seed: int = 42,
        model_type: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        """Solve prescriptive price & promotional calendar optimization (`scenario_optimizer`)."""
        from .scenario_optimizer import optimize_scenario as _optimize_scenario

        eff_model = self._champion_model(model_type)
        preds = self.predictions
        oof = self.oof
        if eff_model is not None:
            if "model_type" in preds.columns:
                preds = preds[preds["model_type"].astype(str) == eff_model].copy()
            if oof is not None and not oof.empty and "model_type" in oof.columns:
                oof = oof[oof["model_type"].astype(str) == eff_model].copy()
        eff_cannib = (
            float(cannibalization_strength)
            if cannibalization_strength is not None
            else float(cannibalization_weight)
        )
        eff_use_sim = (
            bool(similarity_cannibalization)
            if similarity_cannibalization is not None
            else (
                bool(cross_sku_elasticity)
                if cross_sku_elasticity is not None
                else bool(use_similarity_priors)
            )
        )
        return _optimize_scenario(
            self.series,
            config=self.config,
            predictions_df=preds,
            oof_df=oof,
            history_df=self.panel_df if self.panel_df is not None else self.series,
            future_covariates_df=future_covariates_df,
            trajectories_df=trajectories_df,
            model_type=eff_model,
            objective=objective,
            price_col=price_col,
            promo_col=promo_col,
            price_bounds=price_bounds,
            price_grid=price_grid,
            promo_bounds=promo_bounds,
            binary_promo=binary_promo,
            max_promo_days=max_promo_days if max_promo_days is not None else max_promo_periods,
            min_cooldown_days=min_cooldown_days if min_cooldown_days else min_promo_gap,
            unit_cost=unit_cost,
            promo_fixed_cost=promo_fixed_cost,
            min_margin_pct=min_margin_pct,
            min_gross_margin=min_gross_margin,
            min_total_units=min_total_units,
            max_total_units=max_total_units,
            target_total_units=target_total_units,
            max_promo_budget=max_promo_budget,
            inventory_cap=inventory_cap,
            risk_aversion_lambda=risk_aversion_lambda,
            cvar_alpha=cvar_alpha,
            pull_forward_lags=pull_forward_lags,
            cannibalization_weight=eff_cannib,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=(
                similarity_matrix if similarity_matrix is not None else similarity_df
            ),
            similarity_method=similarity_method,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            n_paths=n_paths,
            copula_type=copula_type,
            seed=seed,
            horizon=self.config.data.horizon,
            freq=self.config.data.freq,
        )

    def plot_optimal_scenario(
        self,
        opt_result_or_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        """Plot optimal price/promo schedule (`scenario_optimizer.plot_optimal_scenario`)."""
        from .scenario_optimizer import plot_optimal_scenario

        target = opt_result_or_df if opt_result_or_df is not None else self.optimize_scenario()
        return plot_optimal_scenario(
            target,
            ts_id=ts_id,
            ax=ax,
            title=title,
        )

    def export_report(
        self,
        out_path: str | Any,
        *,
        title: str | None = None,
        goal_seek_df: pd.DataFrame | None = None,
        aci_df: pd.DataFrame | None = None,
        tail_risk_df: pd.DataFrame | None = None,
        cold_start_df: pd.DataFrame | None = None,
        inventory_opt_df: pd.DataFrame | None = None,
        scenario_opt_df: pd.DataFrame | None = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_clusters_df: pd.DataFrame | None = None,
        write_json: bool = False,
        include_json_sidecar: bool = False,
    ) -> Any:
        """Export a self-contained interactive `.html` report for this bake-off."""
        from .report import export_report

        return export_report(
            self,
            out_path,
            title=title,
            goal_seek_df=goal_seek_df,
            aci_df=aci_df,
            tail_risk_df=tail_risk_df,
            cold_start_df=cold_start_df,
            inventory_opt_df=inventory_opt_df,
            scenario_opt_df=scenario_opt_df,
            similarity_df=similarity_df,
            similarity_clusters_df=similarity_clusters_df,
            write_json=write_json,
            include_json_sidecar=include_json_sidecar,
        )


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
        panel_df=data,
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
    return PlaygroundRun(result=result, series=series, backtest=backtest, panel_df=data)


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
