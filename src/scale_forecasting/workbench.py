"""Post-run diagnostic, causal, hierarchical, similarity, and prescriptive workbench mixin.

Provides :class:`PlaygroundWorkbenchMixin` (shared across
:class:`~scale_forecasting.sdk.Forecaster`, :class:`~scale_forecasting.playground.PlaygroundRun`,
and :class:`~scale_forecasting.playground.BakeOff`) and :class:`WorkbenchMixin` (the full post-run
interactive & prescriptive workbench mixed into :class:`~scale_forecasting.sdk.Forecaster`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .config import RunConfig
from .registry.tables import bigquery_client

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from .settings import Settings

__all__ = [
    "PlaygroundWorkbenchMixin",
    "WorkbenchMixin",
]


class PlaygroundWorkbenchMixin:
    """Shared similarity & prescriptive plotting surface across `Forecaster`, `PlaygroundRun`, and
    `BakeOff`.
    """

    _config: RunConfig
    _data: pd.DataFrame | None = None
    _auto_session: Any = None
    _settings: Settings | None = None

    def _wb_default_ts_id(self) -> str | None:
        return None

    def _resolve_similarity_panel(
        self,
        data: pd.DataFrame | None = None,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Resolve a panel DataFrame for similarity analysis from explicit ``data``, in-memory
        ``self._data``, ``self._auto_session.data_df``, BigQuery source/history, or synthetic data.
        """
        from . import playground

        panel = data
        if panel is None and self._data is not None:
            panel = self._data
        if (
            panel is None
            and self._auto_session is not None
            and self._auto_session.data_df is not None
        ):
            panel = self._auto_session.data_df
        if panel is None and run_id is not None:
            with contextlib.suppress(Exception):
                from .engines.bigquery_names import source_ref
                from .engines.bigquery_sql import series_filter
                from .registry.reads import resolve_settings

                resolved = resolve_settings(self._settings)
                source = source_ref(self._config, resolved.registry_dataset_ref)
                idc, datec = self._config.data.ts_id_col, self._config.data.date_col
                sfilter = series_filter(self._config, source, idc)
                where_clause = f" WHERE {sfilter}" if sfilter else ""
                sql = f"SELECT * FROM `{source}`{where_clause} ORDER BY {idc}, {datec}"
                client = bigquery_client(resolved.project_id)
                df = client.query(sql).to_dataframe()
                if not df.empty:
                    panel = df
        if panel is None or panel.empty:
            panel = playground.sample_data(
                n_series=min(self._config.data.series_limit or 4, 6),
                freq=self._config.data.freq,
                with_exog=True,
            )
        return panel

    def similarity_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        top_k: int | None = None,
        max_lag: int | None = None,
        text_cols: Sequence[str] | str | None = None,
        attribute_cols: Sequence[str] | None = None,
        hierarchy_cols: Sequence[str] | None = None,
        within_hierarchy_level: str | Sequence[str] | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute tidy pairwise trajectory similarity, lead-lag cross-correlation, and decomposed
        distance metrics across the panel (`similarity.compute_similarity_matrix`).
        """
        from .similarity import compute_similarity_matrix

        panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        eff_method = method or self._config.features.similarity_method
        res = compute_similarity_matrix(
            panel,
            cfg=self._config,
            method=eff_method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            top_k=int(top_k) if top_k is not None else 10,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )
        return res.pairs_df

    def similarity_matrix_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        max_lag: int | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute the symmetric $(N, N)$ trajectory similarity matrix DataFrame indexed and
        columned by ``ts_id`` (`similarity.compute_similarity_matrix`).
        """
        from .similarity import compute_similarity_matrix

        panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        eff_method = method or self._config.features.similarity_method
        res = compute_similarity_matrix(
            panel,
            cfg=self._config,
            method=eff_method,
            max_lag=int(max_lag) if max_lag is not None else 0,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
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
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Partition series into behavioral similarity clusters with medoid exemplars and dynamical
        archetype labels (`similarity.cluster_by_similarity_df`).
        """
        from .similarity import cluster_by_similarity_df

        panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        eff_method = method or self._config.features.similarity_method
        eff_k = (
            self._config.features.similarity_n_clusters
            if n_clusters == "auto" and self._config.features.similarity_n_clusters != "auto"
            else n_clusters
        )
        return cluster_by_similarity_df(
            panel,
            cfg=self._config,
            n_clusters=eff_k,  # type: ignore[arg-type]
            method=eff_method,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )

    cluster_by_similarity_df = similarity_clusters_df

    def similarity_embeddings_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        n_components: int = 2,
        include_foundation: bool = False,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Extract per-series physical & econometric invariants and 2D phase-space coordinates
        (`similarity.embed_series_df`).
        """
        from .similarity import embed_series_df

        panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        eff_method = (
            "foundation_latent"
            if include_foundation and (method is None or method == "hybrid")
            else (method or self._config.features.similarity_method)
        )
        return embed_series_df(
            panel,
            cfg=self._config,
            method=eff_method,
            n_components=n_components,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )

    embed_series_df = similarity_embeddings_df

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
        text_cols: Sequence[str] | str | None = None,
        attribute_cols: Sequence[str] | None = None,
        hierarchy_cols: Sequence[str] | None = None,
        within_hierarchy_level: str | Sequence[str] | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Retrieve the Top-$K$ most similar peer series for ``query_ts_id`` across full history,
        early launch prefix, or recent window (`similarity.find_similar_series_df`).
        """
        from .similarity import find_similar_series_df as _find_similar_series_df

        eff_data = data
        eff_query = query_ts_id
        if isinstance(query_ts_id_or_data, str):
            if eff_query is None:
                eff_query = query_ts_id_or_data
        elif query_ts_id_or_data is not None:
            eff_data = query_ts_id_or_data

        panel = self._resolve_similarity_panel(data=eff_data, run_id=run_id)
        ts_col = self._config.data.ts_id_col
        if eff_query is None:
            eff_query = self._wb_default_ts_id()
        if eff_query is None:
            eff_query = (
                str(panel[ts_col].iloc[0])
                if not panel.empty and ts_col in panel.columns
                else "series_0"
            )
        eff_method = method or self._config.features.similarity_method
        return _find_similar_series_df(
            panel,
            query_ts_id=eff_query,
            cfg=self._config,
            top_k=top_k,
            mode=mode,
            method=eff_method,
            prefix_len=prefix_len,
            max_lag=max_lag,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            ts_id_col=ts_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )

    def plot_similarity_map(
        self,
        sim_or_clusters_or_panel: Any = None,
        *,
        similarity_or_clusters_df: Any = None,
        data: pd.DataFrame | None = None,
        query_ts_id: str | None = None,
        n_clusters: int | str = "auto",
        method: str | None = None,
        ax: Any = None,
        title: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render the 2-panel trajectory similarity heatmap & 2D MDS cluster map
        (`similarity.plot_similarity_map`).
        """
        from .similarity import plot_similarity_map as _plot_similarity_map

        eff_arg = (
            sim_or_clusters_or_panel
            if sim_or_clusters_or_panel is not None
            else similarity_or_clusters_df
        )
        target = (
            eff_arg
            if eff_arg is not None
            else self._resolve_similarity_panel(data=data, run_id=run_id)
        )
        eff_method = method or self._config.features.similarity_method
        return _plot_similarity_map(
            target,
            cfg=self._config,
            query_ts_id=query_ts_id or self._wb_default_ts_id(),
            n_clusters=n_clusters,  # type: ignore[arg-type]
            method=eff_method,
            ax=ax,
            title=title,
            **kwargs,
        )

    def recommend_substitutes_df(
        self,
        query_ts_id_or_data: pd.DataFrame | str | None = None,
        inventory_df: pd.DataFrame | None = None,
        *,
        query_ts_id: str | Sequence[str] | None = None,
        data: pd.DataFrame | None = None,
        top_k: int = 5,
        text_cols: Sequence[str] | str | None = None,
        attribute_cols: Sequence[str] | None = None,
        hierarchy_cols: Sequence[str] | None = None,
        within_hierarchy_level: str | Sequence[str] | None = None,
        require_in_stock: bool = True,
        on_hand_col: str = "on_hand",
        safety_stock_col: str = "safety_stock",
        method: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Recommend stockout-aware assortment substitutes for one or all series
        (`similarity.recommend_substitutes_df`).
        """
        from .similarity import recommend_substitutes_df as _recommend_substitutes_df

        eff_data: pd.DataFrame | None = data
        eff_query = query_ts_id
        if isinstance(query_ts_id_or_data, str):
            if eff_query is None:
                eff_query = query_ts_id_or_data
        elif query_ts_id_or_data is not None:
            eff_data = query_ts_id_or_data
        panel = self._resolve_similarity_panel(data=eff_data, run_id=run_id)
        eff_method = method or self._config.features.similarity_method
        return _recommend_substitutes_df(
            panel,
            inventory_df=inventory_df,
            query_ts_id=eff_query,
            top_k=top_k,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            require_in_stock=require_in_stock,
            on_hand_col=on_hand_col,
            safety_stock_col=safety_stock_col,
            method=eff_method,
            cfg=self._config,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )

    def emerging_trends_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        window: int = 14,
        top_k_related: int = 3,
        text_cols: Sequence[str] | str | None = None,
        attribute_cols: Sequence[str] | None = None,
        hierarchy_cols: Sequence[str] | None = None,
        max_lag: int = 7,
        method: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Detect emerging trends, regime acceleration, and lead-lag propagation across the panel
        (`similarity.detect_emerging_trends_df`).
        """
        from .similarity import detect_emerging_trends_df as _detect_emerging_trends_df

        panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        eff_method = method or self._config.features.similarity_method
        return _detect_emerging_trends_df(
            panel,
            window=window,
            top_k_related=top_k_related,
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            max_lag=max_lag,
            method=eff_method,
            cfg=self._config,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            freq=self._config.data.freq,
            **kwargs,
        )

    detect_emerging_trends_df = emerging_trends_df

    def plot_goal_seek(
        self,
        goal_seek_df: Any,
        *,
        ts_id: str | None = None,
        top_n: int = 10,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot baseline vs. goal-seek allocated forecasts (`goal_seek.plot_goal_seek`)."""
        from .goal_seek import plot_goal_seek as _plot_goal_seek

        return _plot_goal_seek(
            goal_seek_df,
            ts_id=ts_id or self._wb_default_ts_id(),
            top_n=top_n,
            ax=ax,
            title=title,
        )

    def plot_cold_start_transfer(
        self,
        transfer_df: Any = None,
        *,
        data: pd.DataFrame | None = None,
        history_df: pd.DataFrame | None = None,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot cold-start transferred forecast vs. donor trajectories
        (`cold_start.plot_cold_start_transfer`).
        """
        from .cold_start import plot_cold_start_transfer as _plot_cold_start_transfer

        _ = title
        eff_hist = (
            history_df
            if history_df is not None
            else (
                data
                if data is not None
                else (
                    getattr(self, "series", None)
                    if getattr(self, "series", None) is not None
                    else (
                        self._data
                        if self._data is not None
                        else (
                            self._auto_session.data_df if self._auto_session is not None else None
                        )
                    )
                )
            )
        )
        eff_df = (
            transfer_df
            if transfer_df is not None
            else (
                self.cold_start_transfer_df(data=eff_hist)  # type: ignore[attr-defined]
                if hasattr(self, "_predictions_df")
                else self.cold_start_transfer_df()  # type: ignore[attr-defined]
            )
        )
        return _plot_cold_start_transfer(
            eff_df,
            panel_df=eff_hist,
            ts_id=ts_id or self._wb_default_ts_id(),
            ax=ax,
        )

    def plot_aci(
        self,
        aci_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot ACI rolling empirical coverage and adaptive miscoverage rate (`aci.plot_aci`)."""
        from .aci import plot_aci as _plot_aci

        _ = title
        eff_df = aci_df if aci_df is not None else self.aci_df()  # type: ignore[attr-defined]
        return _plot_aci(eff_df, ts_id=ts_id or self._wb_default_ts_id(), ax=ax)

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
        """Plot joint copula trajectory spaghetti fan and VaR/CVaR tail distribution
        (`copula.plot_trajectories`).
        """
        from .copula import plot_trajectories as _plot_trajectories

        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(n_paths=200)  # type: ignore[attr-defined]
        )
        return _plot_trajectories(
            eff_traj,
            tail_risk=tail_risk,
            ts_id=ts_id or self._wb_default_ts_id(),
            max_spaghetti_paths=(
                max_paths_shown if max_paths_shown is not None else max_spaghetti_paths
            ),
            ax=ax,
            title=title,
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
        """Plot working-capital vs. expected fill-rate / cost Pareto frontier and SKU order bars
        (`inventory_optimizer.plot_inventory_frontier`).
        """
        from .inventory_optimizer import plot_inventory_frontier as _plot_inventory_frontier

        target = (
            opt_result_or_df
            if opt_result_or_df is not None
            else self.optimize_inventory_df(**kwargs)  # type: ignore[attr-defined]
        )
        return _plot_inventory_frontier(
            target,
            budgets=budgets,
            n_points=n_points,
            max_rows=max_rows,
            ax=ax,
            title=title,
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
        """Plot optimal price/promo schedule vs. baseline gross margin / revenue / units
        (`scenario_optimizer.plot_optimal_scenario`).
        """
        from .scenario_optimizer import plot_optimal_scenario as _plot_optimal_scenario

        target = (
            opt_result_or_df if opt_result_or_df is not None else self.optimize_scenario()  # type: ignore[attr-defined]
        )
        return _plot_optimal_scenario(
            target,
            ts_id=ts_id or self._wb_default_ts_id(),
            ax=ax,
            title=title,
        )

    def _resolve_clearance_inputs(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Resolve `(predictions_df, oof_df, history_df)` across `Forecaster`, `BakeOff`, and
        `PlaygroundRun`.
        """
        if hasattr(self, "_resolve_track_k_frames"):
            preds, eff_oof, hist = self._resolve_track_k_frames(
                forecasts_df=forecasts_df,
                data=data,
                oof_df=oof_df,
                run_id=run_id,
                model_type=model_type,
            )
            panel = (
                hist
                if hist is not None
                else self._resolve_similarity_panel(data=data, run_id=run_id)
            )
            return (preds if not preds.empty else panel), eff_oof, panel

        panel = data
        if panel is None:
            panel = getattr(self, "panel_df", None)
        if panel is None:
            panel = getattr(self, "series", None)

        preds = forecasts_df
        eff_oof = oof_df
        if preds is None and hasattr(self, "predictions"):
            raw_preds = getattr(self, "predictions", None)
            if raw_preds is not None and not raw_preds.empty:
                preds = raw_preds.copy()
                champ_fn = getattr(self, "_champion_model", None)
                eff_m = champ_fn(model_type) if callable(champ_fn) else model_type
                if eff_m is not None and "model_type" in preds.columns:
                    sub = preds[preds["model_type"].astype(str) == str(eff_m)]
                    if not sub.empty:
                        preds = sub.copy()
            if eff_oof is None:
                eff_oof = getattr(self, "oof", None)
        elif preds is None and hasattr(self, "result"):
            res_cell = getattr(self, "result", None)
            if res_cell is not None and not res_cell.predictions.empty:
                preds = res_cell.predictions.assign(
                    ts_id=res_cell.ts_id, model_type=res_cell.model_type
                )
                if eff_oof is None and res_cell.oof is not None and not res_cell.oof.empty:
                    eff_oof = res_cell.oof.assign(
                        ts_id=res_cell.ts_id, model_type=res_cell.model_type
                    )

        if panel is None:
            panel = self._resolve_similarity_panel(data=data, run_id=run_id)
        return (preds if preds is not None and not preds.empty else panel), eff_oof, panel

    def _default_clearance_covariate_and_hierarchy_cols(
        self,
        kwargs: dict[str, Any],
    ) -> None:
        """Populate default `covariate_cols` and `hierarchy_cols` from `self._config` when unset."""
        if "covariate_cols" not in kwargs:
            covs = list(
                dict.fromkeys(
                    [
                        *getattr(self._config.features, "future_covariates", ()),
                        *getattr(self._config.features, "past_covariates", ()),
                    ]
                )
            )
            if covs:
                kwargs["covariate_cols"] = covs
        if (
            "hierarchy_cols" not in kwargs
            and getattr(self._config, "hierarchy", None) is not None
            and getattr(self._config.hierarchy, "enabled", False)
            and getattr(self._config.hierarchy, "levels", None)
        ):
            kwargs["hierarchy_cols"] = list(self._config.hierarchy.levels)

    def optimize_clearance(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the lifecycle markdown, perishable expiry, and shelf-space reclamation MILP
        (`clearance_optimizer.optimize_clearance`).
        """
        from .clearance_optimizer import optimize_clearance as _optimize_clearance

        preds, eff_oof, hist = self._resolve_clearance_inputs(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        self._default_clearance_covariate_and_hierarchy_cols(kwargs)
        return _optimize_clearance(
            preds,
            history_df=kwargs.pop("history_df", hist),
            oof_df=eff_oof,
            model_type=model_type,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            horizon=self._config.data.horizon,
            freq=self._config.data.freq,
            **kwargs,
        )

    def optimize_clearance_df(
        self,
        *,
        summary: bool = False,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Solve the lifecycle markdown, perishable expiry, and shelf-space reclamation MILP and
        return a tidy schedule or per-SKU summary DataFrame
        (`clearance_optimizer.optimize_clearance_df`).
        """
        res = self.optimize_clearance(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            **kwargs,
        )
        out = res.summary_df.copy() if summary else res.schedule_df.copy()
        out.attrs["clearance_result"] = res
        out.attrs["status"] = res.status
        out.attrs["is_feasible"] = res.is_feasible
        out.attrs["relaxation_hint"] = res.relaxation_hint
        out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
        return out

    def plot_clearance_schedule(
        self,
        result_or_df: Any = None,
        *,
        ts_id: str | None = None,
        ax: Any = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the optimal markdown discount ladder, inventory depletion trajectory, and SKU
        profit summary (`clearance_optimizer.plot_clearance_schedule`).
        """
        from .clearance_optimizer import plot_clearance_schedule as _plot_clearance_schedule

        target = result_or_df if result_or_df is not None else self.optimize_clearance(**kwargs)
        return _plot_clearance_schedule(
            target,
            ts_id=ts_id or self._wb_default_ts_id(),
            ax=ax,
            title=title,
        )

    def optimize_contract_volume(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve multi-SKU vendor contract volume, tiered rebate, and Take-or-Pay optimization
        (`clearance_optimizer.optimize_contract_volume`).
        """
        from .clearance_optimizer import optimize_contract_volume as _optimize_contract_volume

        preds, eff_oof, hist = self._resolve_clearance_inputs(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        self._default_clearance_covariate_and_hierarchy_cols(kwargs)
        return _optimize_contract_volume(
            preds,
            history_df=kwargs.pop("history_df", hist),
            oof_df=eff_oof,
            model_type=model_type,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            horizon=self._config.data.horizon,
            freq=self._config.data.freq,
            **kwargs,
        )

    def optimize_contract_volume_df(
        self,
        *,
        summary: bool = False,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Solve multi-SKU vendor contract volume, tiered rebate, and Take-or-Pay optimization and
        return a tidy schedule or per-SKU summary DataFrame
        (`clearance_optimizer.optimize_contract_volume_df`).
        """
        res = self.optimize_contract_volume(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            **kwargs,
        )
        out = res.summary_df.copy() if summary else res.schedule_df.copy()
        out.attrs["contract_result"] = res
        out.attrs["status"] = res.status
        out.attrs["is_feasible"] = res.is_feasible
        out.attrs["relaxation_hint"] = res.relaxation_hint
        out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
        return out

    def optimize_assortment(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the assortment rationalization & delisting demand-transference MILP
        (`clearance_optimizer.optimize_assortment`).
        """
        from .clearance_optimizer import optimize_assortment as _optimize_assortment

        preds, eff_oof, hist = self._resolve_clearance_inputs(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        self._default_clearance_covariate_and_hierarchy_cols(kwargs)
        return _optimize_assortment(
            preds,
            history_df=kwargs.pop("history_df", hist),
            oof_df=eff_oof,
            model_type=model_type,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            horizon=self._config.data.horizon,
            freq=self._config.data.freq,
            **kwargs,
        )

    def optimize_assortment_df(
        self,
        *,
        transference: bool = False,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Solve the assortment rationalization & delisting demand-transference MILP and return a
        tidy SKU decision or pairwise transference DataFrame
        (`clearance_optimizer.optimize_assortment_df`).
        """
        res = self.optimize_assortment(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            **kwargs,
        )
        out = res.transference_df.copy() if transference else res.assortment_df.copy()
        out.attrs["assortment_result"] = res
        out.attrs["status"] = res.status
        out.attrs["is_feasible"] = res.is_feasible
        out.attrs["relaxation_hint"] = res.relaxation_hint
        out.attrs["hierarchy_rollup_df"] = res.hierarchy_rollup_df
        return out

    def optimize_prepacks(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the Simplex Size-Curve & Integer Pre-Pack Allocation MILP
        (`inventory_optimizer.optimize_prepacks`).
        """
        from .inventory_optimizer import optimize_prepacks as _optimize_prepacks

        preds, eff_oof, hist = self._resolve_clearance_inputs(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        return _optimize_prepacks(
            preds,
            history_df=kwargs.pop("history_df", hist),
            oof_df=eff_oof,
            model_type=model_type,
            ts_id_col=self._config.data.ts_id_col,
            **kwargs,
        )

    def optimize_prepacks_df(
        self,
        *,
        summary: bool = False,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Solve the Simplex Size-Curve & Integer Pre-Pack Allocation MILP and return a tidy
        allocation or pack-order DataFrame (`inventory_optimizer.optimize_prepacks_df`).
        """
        res = self.optimize_prepacks(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            **kwargs,
        )
        out = res.pack_orders_df.copy() if summary else res.allocation_df.copy()
        out.attrs["prepack_result"] = res
        out.attrs["status"] = res.status
        out.attrs["is_feasible"] = res.is_feasible
        out.attrs["relaxation_hint"] = res.relaxation_hint
        out.attrs["size_curves_df"] = res.size_curves_df
        return out

    def optimize_rebalancing(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the Hitchcock–Koopmans Lateral Store-to-Store / Depot Inventory Rebalancing LP
        (`inventory_optimizer.optimize_rebalancing`).
        """
        from .inventory_optimizer import optimize_rebalancing as _optimize_rebalancing

        preds, eff_oof, _hist = self._resolve_clearance_inputs(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        return _optimize_rebalancing(
            preds,
            oof_df=eff_oof,
            model_type=model_type,
            ts_id_col=self._config.data.ts_id_col,
            **kwargs,
        )

    def optimize_rebalancing_df(
        self,
        *,
        summary: bool = False,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Solve the Hitchcock–Koopmans Lateral Store-to-Store / Depot Inventory Rebalancing LP and
        return a tidy lane transfer or node summary DataFrame
        (`inventory_optimizer.optimize_rebalancing_df`).
        """
        res = self.optimize_rebalancing(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            **kwargs,
        )
        out = res.node_summary_df.copy() if summary else res.transfers_df.copy()
        out.attrs["rebalancing_result"] = res
        out.attrs["status"] = res.status
        out.attrs["is_feasible"] = res.is_feasible
        out.attrs["relaxation_hint"] = res.relaxation_hint
        return out

    def sop_summary_df(
        self,
        run_id: str | None = None,
        *,
        limit: int = 25,
    ) -> pd.DataFrame:
        """Return an offline Executive S&OP Scorecard DataFrame mirroring `v_executive_sop_summary`
        for this `PlaygroundRun` or `BakeOff`.
        """
        from .bq_agent.offline import build_offline_sop_summary_df

        _ = limit
        return build_offline_sop_summary_df(self, run_id=run_id, default_n_series=1)

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Answer a natural-language forecasting question over this offline `PlaygroundRun` or
        `BakeOff` (`bq_agent.offline.ask_offline`).
        """
        from .bq_agent.offline import ask_offline

        return ask_offline(self, question, conversation_id=conversation_id)


class WorkbenchMixin(PlaygroundWorkbenchMixin):
    """Post-run diagnostic, causal, hierarchical, and prescriptive optimization workbench methods
    mixed into :class:`~scale_forecasting.sdk.Forecaster`.
    """

    _predictions_df: pd.DataFrame | None = None
    _oof_df: pd.DataFrame | None = None

    @property
    def run_id(self) -> str:
        raise NotImplementedError

    def predictions_df(
        self,
        run_id: str | None = None,
        *,
        ts_id: str | None = None,
        ts_ids: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        include_history: bool = False,
        include_oof: bool = False,
        history_tail: int = 180,
        series_limit: int | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def leaderboard_df(
        self, run_id: str | None = None, *, all_metrics: bool = False
    ) -> pd.DataFrame:
        raise NotImplementedError

    def autofocus_ledger_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def registry(self) -> Any:
        raise NotImplementedError

    def fva_df(
        self,
        run_id: str | None = None,
        *,
        baseline: str | None = None,
        metric: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Compute Forecast Value Added (FVA) & Diebold–Mariano statistical significance for
        ``run_id`` (`diagnostics.compute_fva`).
        """
        import pandas as pd

        from .diagnostics import compute_fva
        from .registry.reads import read_oof

        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            if baseline is None and metric is None:
                return self._auto_session.fva_df.copy()
            eff_metric = metric or self._config.backtest.decision_metric
            return compute_fva(self._auto_session.oof_df, baseline=baseline, metric=eff_metric)

        if (
            self._auto_session is None
            and (self._oof_df is not None or self._data is not None)
            and (run_id is None or run_id == self.run_id)
        ):
            _, loc_oof, loc_hist = self._resolve_track_k_frames(limit=limit)
            eff_metric = metric or self._config.backtest.decision_metric
            return compute_fva(
                loc_oof if loc_oof is not None else pd.DataFrame(),
                baseline=baseline,
                metric=eff_metric,
                series_df=loc_hist,
            )

        rid = run_id or self.run_id
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
        eff_metric = metric or self._config.backtest.decision_metric
        return compute_fva(oof_df, baseline=baseline, metric=eff_metric)

    def fva_cascade_df(
        self,
        run_id: str | None = None,
        *,
        baseline: str | None = None,
        metric: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Return the Forecast Value Added (FVA) cascade or baseline value ladder DataFrame."""
        if self._auto_session is not None and (run_id is None or run_id == self.run_id):
            bv = self._auto_session.baseline_value_df()
            if not bv.empty:
                return bv
        return self.fva_df(run_id, baseline=baseline, metric=metric, limit=limit)

    def plot_fva(
        self,
        run_id: str | None = None,
        *,
        baseline: str | None = None,
        metric: str | None = None,
        limit: int = 20000,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot the Forecast Value Added (FVA) accuracy waterfall for ``run_id``."""
        from .diagnostics import plot_fva

        rid = run_id or self.run_id
        df = self.fva_df(rid, baseline=baseline, metric=metric, limit=limit)
        return plot_fva(df, ax=ax, title=title or f"{rid} — Forecast Value Added")

    def horizon_decay_df(
        self,
        run_id: str | None = None,
        *,
        metric: str | None = None,
        baseline: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Compute lead-time accuracy decay across horizon steps ($h = 1 \\dots H$)."""
        import pandas as pd

        from .diagnostics import compute_horizon_decay
        from .registry.reads import read_oof

        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            if baseline is None and metric is None:
                return self._auto_session.horizon_decay_df.copy()
            eff_metric = metric or self._config.backtest.decision_metric
            return compute_horizon_decay(
                self._auto_session.oof_df, metric=eff_metric, baseline=baseline
            )

        if (
            self._auto_session is None
            and (self._oof_df is not None or self._data is not None)
            and (run_id is None or run_id == self.run_id)
        ):
            _, loc_oof, _ = self._resolve_track_k_frames(limit=limit)
            eff_metric = metric or self._config.backtest.decision_metric
            return compute_horizon_decay(
                loc_oof if loc_oof is not None else pd.DataFrame(),
                metric=eff_metric,
                baseline=baseline,
            )

        rid = run_id or self.run_id
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
        eff_metric = metric or self._config.backtest.decision_metric
        return compute_horizon_decay(oof_df, metric=eff_metric, baseline=baseline)

    def plot_horizon_decay(
        self,
        run_id: str | None = None,
        *,
        metric: str | None = None,
        baseline: str | None = None,
        limit: int = 20000,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot lead-time accuracy decay by horizon step ($h = 1 \\dots H$) for ``run_id``."""
        from .diagnostics import plot_horizon_decay

        rid = run_id or self.run_id
        df = self.horizon_decay_df(rid, metric=metric, baseline=baseline, limit=limit)
        return plot_horizon_decay(df, ax=ax, title=title or f"{rid} — Horizon Accuracy Decay")

    def bias_report_df(
        self,
        run_id: str | None = None,
        *,
        threshold_pct: float = 0.05,
        tracking_signal_limit: float = 4.0,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Audit every ``(ts_id, model_type)`` cell for directional bias and Trigg Tracking Signal
        on ``run_id``.
        """
        import pandas as pd

        from .diagnostics import compute_bias_report
        from .registry.reads import read_oof

        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            return compute_bias_report(
                self._auto_session.oof_df,
                threshold_pct=threshold_pct,
                tracking_signal_limit=tracking_signal_limit,
            )

        if (
            self._auto_session is None
            and (self._oof_df is not None or self._data is not None)
            and (run_id is None or run_id == self.run_id)
        ):
            _, loc_oof, _ = self._resolve_track_k_frames(limit=limit)
            return compute_bias_report(
                loc_oof if loc_oof is not None else pd.DataFrame(),
                threshold_pct=threshold_pct,
                tracking_signal_limit=tracking_signal_limit,
            )

        rid = run_id or self.run_id
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
        return compute_bias_report(
            oof_df,
            threshold_pct=threshold_pct,
            tracking_signal_limit=tracking_signal_limit,
        )

    def bias_df(
        self,
        run_id: str | None = None,
        *,
        threshold_pct: float = 0.05,
        tracking_signal_limit: float = 4.0,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Alias for :meth:`bias_report_df` matching :meth:`BakeOff.bias_df`."""
        return self.bias_report_df(
            run_id,
            threshold_pct=threshold_pct,
            tracking_signal_limit=tracking_signal_limit,
            limit=limit,
        )

    def rollup_time(
        self,
        freq: str = "W",
        *,
        run_id: str | None = None,
        level: float = 0.80,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Aggregate ``run_id`` future predictions to Weekly (`"W"`), Monthly (`"MS"`/`"M"`), or
        Quarterly (`"QS"`/`"Q"`) buckets with autocorrelation-aware prediction intervals.
        """
        import pandas as pd

        from .diagnostics import rollup_time
        from .registry.reads import read_oof, read_predictions

        rid = run_id or self.run_id
        pred_rows = read_predictions(rid, limit=limit, settings=self._settings)
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else None
        return rollup_time(pred_df, freq=freq, oof_df=oof_df, level=level)

    def inventory_policy_df(
        self,
        run_id: str | None = None,
        *,
        inventory_df: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
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
        unit_cost: float | Mapping[str, float] | str = 1.0,
        unit_volume: float | Mapping[str, float] | str = 1.0,
        vendor_moq: float | Mapping[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Translate ``run_id`` calibrated forecasts into operational inventory & replenishment
        policies (`diagnostics.compute_inventory_policy`).

        Computes continuous-review $(s, S)$ and periodic-review $(R, S)$ **Reorder Points
        (`ROP`)**, **Safety Stock (`SS`)**, **Order-Up-To Levels ($S$)**, **Recommended Order
        Quantities ($Q$)**, and **Expected Holding vs. Stockout Cost** tables using AR(1)
        multi-step forecast error covariance ($\\hat{\\rho}_1$ from ``backtest_oof``) and supplier
        lead-time variability ($\\mu_L, \\sigma_L$). When portfolio constraints (`max_budget`,
        `max_volume`, `max_total_units`, `target_portfolio_fill_rate`, or `vendor_moq`) are
        supplied, delegates to :func:`inventory_optimizer.optimize_inventory_portfolio`.
        """
        import pandas as pd

        from .diagnostics import compute_inventory_policy
        from .registry.reads import read_oof, read_predictions

        has_portfolio_constraints = (
            max_budget is not None
            or max_volume is not None
            or max_total_units is not None
            or target_portfolio_fill_rate is not None
            or vendor_moq is not None
            or uncertainty_source == "copula"
            or trajectories_df is not None
            or copula_paths is not None
        )
        if has_portfolio_constraints:
            return self.optimize_inventory_df(
                run_id=run_id,
                inventory_df=inventory_df,
                trajectories_df=trajectories_df,
                copula_paths=copula_paths,
                model_name=model_name,
                model_type=model_type,
                lead_time=lead_time,
                lead_time_std=lead_time_std,
                review_period=review_period,
                service_level=service_level if service_level is not None else 0.95,
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
                limit=limit,
            )

        rid = run_id or self.run_id
        eff_model = model_name or model_type
        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            pred_df = self._auto_session.raw_predictions_df.rename(columns={"forecast_date": "ds"})
            oof_df = self._auto_session.oof_df
            return compute_inventory_policy(
                pred_df,
                oof_df=oof_df,
                inventory_df=inventory_df,
                model_name=eff_model,
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
        if (
            self._auto_session is None
            and (self._predictions_df is not None or self._data is not None)
            and (run_id is None or run_id == self.run_id)
        ):
            loc_pred, loc_oof, _ = self._resolve_track_k_frames(model_type=eff_model, limit=limit)
            return compute_inventory_policy(
                loc_pred,
                oof_df=loc_oof,
                inventory_df=inventory_df,
                model_name=eff_model,
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
        model_filter = [eff_model] if eff_model and eff_model not in ("all", "*") else None
        pred_rows = read_predictions(rid, models=model_filter, limit=limit, settings=self._settings)
        oof_rows = read_oof(rid, models=model_filter, limit=limit, settings=self._settings)
        pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else None
        return compute_inventory_policy(
            pred_df,
            oof_df=oof_df,
            inventory_df=inventory_df,
            model_name=eff_model,
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
        run_id: str | None = None,
        *,
        inventory_df: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
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
        unit_cost: float | Mapping[str, float] | str = 1.0,
        unit_volume: float | Mapping[str, float] | str = 1.0,
        vendor_moq: float | Mapping[str, float] | None = None,
        uncertainty_source: str = "ar1",
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        copula_type: str = "gaussian",
        n_paths: int = 300,
        seed: int = 42,
        max_rows: int = 15,
        limit: int = 20000,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot Reorder Points (`ROP`), Safety Stock (`SS`), and Recommended Order Quantities ($Q$)
        for ``run_id``.
        """
        from .diagnostics import plot_inventory_policy

        rid = run_id or self.run_id
        df = self.inventory_policy_df(
            rid,
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
            limit=limit,
        )
        return plot_inventory_policy(
            df,
            max_rows=max_rows,
            ax=ax,
            title=title or f"{rid} — Inventory & Replenishment Policy",
        )

    def simulate_scenarios(
        self,
        scenarios: Mapping[str, Mapping[str, Any]],
        *,
        data: pd.DataFrame | None = None,
        models: Sequence[str] | None = None,
        future_covariates_df: pd.DataFrame | None = None,
        price_col: str | None = None,
        unit_cost: float | str | Mapping[str, float] | None = None,
        reconcile: bool = True,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        oof_df: pd.DataFrame | None = None,
        inventory_params: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Run full-chain counterfactual covariate shocks (`scenarios.simulate_scenarios`) across
        ``data`` (or in-memory `auto_session.data_df`, or BigQuery source series when ``run_id`` is
        provided, or synthetic sample data when ``data`` is omitted) using this config's settings.
        """
        from . import playground
        from .scenarios import simulate_scenarios

        panel = data
        fut_df = future_covariates_df
        eff_oof = oof_df
        if (
            panel is None
            and self._auto_session is not None
            and self._auto_session.data_df is not None
        ):
            panel = self._auto_session.data_df
            if fut_df is None:
                fut_df = self._auto_session.future_df
            if eff_oof is None and not self._auto_session.oof_df.empty:
                eff_oof = self._auto_session.oof_df
        if panel is None and self._data is not None:
            panel = self._data
            if eff_oof is None and self._oof_df is not None:
                eff_oof = self._oof_df
        if panel is None and run_id is not None:
            try:
                from .engines.bigquery_names import source_ref
                from .engines.bigquery_sql import series_filter
                from .registry.reads import resolve_settings

                resolved = resolve_settings(self._settings)
                source = source_ref(self._config, resolved.registry_dataset_ref)
                idc, datec = self._config.data.ts_id_col, self._config.data.date_col
                sfilter = series_filter(self._config, source, idc)
                where_clause = f" WHERE {sfilter}" if sfilter else ""
                sql = f"SELECT * FROM `{source}`{where_clause} ORDER BY {idc}, {datec}"
                client = bigquery_client(resolved.project_id)
                df = client.query(sql).to_dataframe()
                if not df.empty:
                    panel = df
            except Exception:  # noqa: BLE001 - fall back to synthetic sample data when offline
                panel = None
        if panel is None:
            panel = playground.sample_data(
                n_series=min(self._config.data.series_limit or 3, 5),
                freq=self._config.data.freq,
                with_exog=True,
            )
        return simulate_scenarios(
            panel,
            self._config,
            scenarios,
            models=models,
            future_covariates_df=fut_df,
            price_col=price_col,
            unit_cost=unit_cost,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            oof_df=eff_oof,
            inventory_params=inventory_params,
        )

    def plot_scenarios(
        self,
        sim_df: pd.DataFrame,
        *,
        ts_id: str | None = None,
        model_type: str | None = None,
        show_revenue: bool = False,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot baseline vs. counterfactual scenario trajectories from `simulate_scenarios`."""
        from .scenarios import plot_scenarios

        return plot_scenarios(
            sim_df,
            ts_id=ts_id,
            model_type=model_type,
            show_revenue=show_revenue,
            ax=ax,
            title=title,
        )

    def _resolve_track_k_frames(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        limit: int = 20000,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Resolve `(predictions_df, oof_df, history_df)` from session, local data, or BQ."""
        import pandas as pd

        hist_df = data if data is not None else self._data
        eff_oof = oof_df
        preds_df = forecasts_df

        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            if hist_df is None:
                hist_df = self._auto_session.data_df
            if eff_oof is None and not self._auto_session.oof_df.empty:
                eff_oof = self._auto_session.oof_df
            if preds_df is None and not self._auto_session.raw_predictions_df.empty:
                preds_df = self._auto_session.raw_predictions_df.copy()

        if (
            preds_df is None
            and self._predictions_df is not None
            and (run_id is None or run_id == self.run_id)
        ):
            preds_df = self._predictions_df.copy()
            if "segment" in preds_df.columns:
                preds_df = preds_df[preds_df["segment"] == "forecast"].copy()
            if eff_oof is None and self._oof_df is not None:
                eff_oof = self._oof_df.copy()

        if preds_df is None and hist_df is not None and run_id is None:
            from .engines.spark_io import run_group

            models_to_run = (
                [model_type] if model_type is not None else (list(self._config.models) or ["theta"])
            )
            run_cfg = self._config
            id_col = run_cfg.data.ts_id_col
            if not hist_df.empty and id_col in hist_df.columns:
                max_len = int(hist_df.groupby(id_col).size().max())
                bt = run_cfg.backtest
                needed = bt.min_train + bt.horizon + max(0, bt.n_folds - 1) * bt.step + bt.gap
                if max_len < needed and max_len >= 6:
                    eff_h = min(bt.horizon, max(1, max_len // 3))
                    eff_n_folds = max(1, bt.n_folds)
                    eff_step = min(bt.step, eff_h)
                    eff_min_train = min(
                        bt.min_train,
                        max(2, max_len - eff_h - (eff_n_folds - 1) * eff_step - bt.gap),
                    )
                    run_cfg = run_cfg.model_copy(
                        update={
                            "backtest": bt.model_copy(
                                update={
                                    "enabled": True,
                                    "horizon": eff_h,
                                    "step": eff_step,
                                    "min_train": eff_min_train,
                                }
                            )
                        }
                    )
            cells, _ = run_group(hist_df, run_cfg, models=models_to_run)
            ok_cells = [c for c in cells if c.status == "ok" and not c.predictions.empty]
            if ok_cells:
                preds_df = pd.concat(
                    [
                        c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type)
                        for c in ok_cells
                    ],
                    ignore_index=True,
                )
            oof_cells = [
                c for c in cells if c.status == "ok" and c.oof is not None and not c.oof.empty
            ]
            if eff_oof is None and oof_cells:
                eff_oof = pd.concat(
                    [
                        c.oof.assign(ts_id=c.ts_id, model_type=c.model_type)
                        for c in oof_cells
                        if c.oof is not None
                    ],
                    ignore_index=True,
                )
            if data is None and model_type is None:
                if preds_df is not None and self._predictions_df is None:
                    self._predictions_df = preds_df.copy()
                if eff_oof is not None and self._oof_df is None:
                    self._oof_df = eff_oof.copy()

        if preds_df is None:
            from .registry.reads import read_oof, read_predictions

            rid = run_id or self.run_id
            model_filter = [model_type] if model_type and model_type not in ("all", "*") else None
            pred_rows = read_predictions(
                rid, models=model_filter, limit=limit, settings=self._settings
            )
            preds_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
            if eff_oof is None:
                oof_rows = read_oof(rid, models=model_filter, limit=limit, settings=self._settings)
                eff_oof = pd.DataFrame.from_records(oof_rows) if oof_rows else None
            if hist_df is None:
                with contextlib.suppress(Exception):
                    from .engines.bigquery_names import source_ref
                    from .engines.bigquery_sql import series_filter
                    from .registry.reads import resolve_settings

                    resolved = resolve_settings(self._settings)
                    source = source_ref(self._config, resolved.registry_dataset_ref)
                    idc, datec = self._config.data.ts_id_col, self._config.data.date_col
                    sfilter = series_filter(self._config, source, idc)
                    where_clause = f" WHERE {sfilter}" if sfilter else ""
                    sql = f"SELECT * FROM `{source}`{where_clause} ORDER BY {idc}, {datec}"
                    client = bigquery_client(resolved.project_id)
                    df_src = client.query(sql).to_dataframe()
                    if not df_src.empty:
                        hist_df = df_src

        if preds_df is not None and "segment" in preds_df.columns:
            fc_mask = preds_df["segment"] == "forecast"
            if fc_mask.any():
                preds_df = preds_df[fc_mask].copy()

        return (
            preds_df if preds_df is not None else pd.DataFrame(),
            eff_oof,
            hist_df,
        )

    def temporal_reconcile_df(
        self,
        *,
        data: pd.DataFrame | None = None,
        forecasts_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        method: str | None = None,
        model_type: str | None = None,
        aggregations: Sequence[int] | None = None,
        temporal_aggregations: Sequence[int] | None = None,
        cross_sectional: bool | None = None,
        spatial_temporal_weight: float | str | None = None,
        return_aggregations: bool = False,
        limit: int = 20000,
    ) -> Any:
        """Reconcile forecasts across temporal aggregations (`THieF`) or simultaneously across
        cross-sectional and temporal hierarchies (`cross_temporal_mint`).
        """
        from .reconciliation import build_hierarchy
        from .temporal_reconciliation import (
            reconcile_cross_temporal,
            reconcile_temporal_forecasts,
        )

        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        if forecasts_df is None and not preds_df.empty and "model_type" in preds_df.columns:
            target_models = (
                {model_type}
                if model_type is not None and model_type not in ("all", "*")
                else set(self._config.models)
            )
            if target_models and any(
                m in target_models for m in preds_df["model_type"].astype(str).unique()
            ):
                preds_df = preds_df[preds_df["model_type"].astype(str).isin(target_models)].copy()
                if eff_oof is not None and not eff_oof.empty and "model_type" in eff_oof.columns:
                    eff_oof = eff_oof[eff_oof["model_type"].astype(str).isin(target_models)].copy()
        eff_aggs = (
            temporal_aggregations
            if temporal_aggregations is not None
            else (
                aggregations
                if aggregations is not None
                else self._config.hierarchy.temporal_aggregations
            )
        )
        eff_method = method or (
            self._config.hierarchy.reconciliation_methods[0]
            if self._config.hierarchy.reconciliation_methods
            else "mint_shrink"
        )
        has_hier = bool(
            self._config.hierarchy.levels or self._config.hierarchy.similarity_hierarchy
        )
        use_cross_sectional = (
            cross_sectional
            if cross_sectional is not None
            else bool(
                self._config.hierarchy.cross_temporal
                or eff_method == "cross_temporal_mint"
                or (
                    self._config.hierarchy.enabled
                    and has_hier
                    and hist_df is not None
                    and not hist_df.empty
                )
            )
        )
        if use_cross_sectional and has_hier and hist_df is not None and not hist_df.empty:
            needed = {c for lvl in self._config.hierarchy.levels for c in lvl}
            if needed.issubset(hist_df.columns) or self._config.hierarchy.similarity_hierarchy:
                panel_df, hier_spec = build_hierarchy(hist_df, self._config)
                return reconcile_cross_temporal(
                    preds_df,
                    hier_spec,
                    history_df=panel_df,
                    oof_df=eff_oof,
                    cfg=self._config,
                    freq=self._config.data.freq,
                    temporal_aggregations=eff_aggs,
                    method=eff_method,
                    middle_level=self._config.hierarchy.middle_level,
                    spatial_temporal_weight=spatial_temporal_weight,  # type: ignore[arg-type]
                    return_aggregations=return_aggregations,
                    ts_id_col=self._config.data.ts_id_col,
                    target_col=self._config.data.target_col,
                )

        return reconcile_temporal_forecasts(
            preds_df,
            history_df=hist_df,
            oof_df=eff_oof,
            cfg=self._config,
            freq=self._config.data.freq,
            horizon=self._config.data.horizon,
            temporal_aggregations=eff_aggs,
            method=eff_method if eff_method != "cross_temporal_mint" else "mint_shrink",
            model_type=model_type,
            return_aggregations=return_aggregations,
            ts_id_col=self._config.data.ts_id_col,
            target_col=self._config.data.target_col,
        )

    temporal_reconcile = temporal_reconcile_df

    def sample_coherent_paths(
        self,
        *,
        data: pd.DataFrame | None = None,
        forecasts_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        spec: Any = None,
        hier_spec: Any = None,
        model_type: str | None = None,
        n_paths: int = 200,
        seed: int = 42,
        method: str | None = None,
        nonneg: bool = False,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Draw ``n_paths`` coherent probabilistic trajectory scenarios satisfying cross-sectional
        and/or temporal aggregation constraints (`temporal_reconciliation.sample_coherent_paths`).
        """
        from .reconciliation import build_hierarchy
        from .temporal_reconciliation import sample_coherent_paths as _sample_coherent_paths

        eff_fc = forecasts_df if forecasts_df is not None else predictions_df
        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=eff_fc,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        eff_spec = spec if spec is not None else hier_spec
        if (
            eff_spec is None
            and (self._config.hierarchy.levels or self._config.hierarchy.similarity_hierarchy)
            and hist_df is not None
            and not hist_df.empty
        ):
            needed = {c for lvl in self._config.hierarchy.levels for c in lvl}
            if needed.issubset(hist_df.columns) or self._config.hierarchy.similarity_hierarchy:
                hist_df, eff_spec = build_hierarchy(hist_df, self._config)
        eff_method = method or (
            self._config.hierarchy.reconciliation_methods[0]
            if self._config.hierarchy.reconciliation_methods
            else "mint_shrink"
        )
        return _sample_coherent_paths(
            preds_df,
            eff_spec,
            history_df=hist_df,
            oof_df=eff_oof,
            cfg=self._config,
            n_paths=n_paths,
            seed=seed,
            method=eff_method,
            nonneg=nonneg,
        )

    def allocate_target(
        self,
        target_total: Any = None,
        _second_arg: Any = None,
        *,
        target: Any = None,
        targets: Any = None,
        target_sum: Any = None,
        horizon_total: float | None = None,
        forecasts_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        history_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        target_node: str = "__total__",
        period: Sequence[Any] | str | None = None,
        target_dates: Sequence[Any] | str | None = None,
        target_mode: str = "auto",
        target_metric: str = "auto",
        locked_nodes: Sequence[str] | Mapping[str, float] | None = None,
        locks: Sequence[str] | Mapping[str, float] | None = None,
        lower_bounds: Mapping[str, float] | Sequence[float] | Any = None,
        upper_bounds: Mapping[str, float] | Sequence[float] | Any = None,
        bounds: Any = None,
        floor: float | None = 0.0,
        cap: float | None = None,
        ceiling: float | None = None,
        nonneg: bool = True,
        method: str = "mint_shrink",
        weights: Any = None,
        smoothness_weight: float = 0.0,
        hier_spec: Any = None,
        spec: Any = None,
        model_type: str | None = None,
        scenario: str | None = None,
        price_col: str | Mapping[str, float] | float | None = None,
        prices: Mapping[str, float] | float | pd.DataFrame | None = None,
        unit_cost: float | Mapping[str, float] | str | None = None,
        include_inventory_impact: bool = False,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float | None = 0.95,
        inventory_params: Mapping[str, Any] | None = None,
        return_result: bool = False,
        limit: int = 20000,
    ) -> Any:
        """Optimally disaggregate a top-down or regional target across leaf series and dates via
        constrained Minimum-Trace Quadratic Programming (`goal_seek.allocate_target`).
        """
        import pandas as pd

        from .goal_seek import allocate_target as _allocate_target

        eff_forecasts = forecasts_df if forecasts_df is not None else predictions_df
        eff_target = target_total
        if (
            isinstance(target_total, pd.DataFrame)
            and ("yhat" in target_total.columns or "scenario_yhat" in target_total.columns)
            and eff_forecasts is None
        ):
            eff_forecasts = target_total
            eff_target = _second_arg

        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=eff_forecasts,
            data=data if data is not None else history_df,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        return _allocate_target(
            preds_df,
            target_total=eff_target,
            target=target,
            targets=targets,
            target_sum=target_sum,
            horizon_total=horizon_total,
            node_id=node_id,
            target_node=target_node,
            period=period,
            target_dates=target_dates,
            target_mode=target_mode,  # type: ignore[arg-type]
            target_metric=target_metric,  # type: ignore[arg-type]
            locked_nodes=locked_nodes,
            locks=locks,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            bounds=bounds,
            floor=floor if floor is not None else self._config.output.floor,
            cap=cap
            if cap is not None
            else (ceiling if ceiling is not None else self._config.output.cap),
            nonneg=nonneg,
            method=method,
            weights=weights,
            smoothness_weight=smoothness_weight,
            hier_spec=hier_spec,
            spec=spec,
            cfg=self._config,
            history_df=hist_df,
            oof_df=eff_oof,
            price_col=price_col,
            prices=prices,
            unit_cost=unit_cost,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            inventory_params=inventory_params,
            model_type=model_type,
            scenario=scenario,
            date_col=self._config.data.date_col,
            ts_id_col=self._config.data.ts_id_col,
            target_col=self._config.data.target_col,
            return_result=return_result,
        )

    def audit_monotonicity_df(
        self,
        sim_df: pd.DataFrame,
        *,
        constraints: Mapping[str, int] | None = None,
        scenarios: Mapping[str, Mapping[str, Any]] | None = None,
        tol: float = 1e-6,
    ) -> pd.DataFrame:
        """Audit counterfactual ``simulate_scenarios`` predictions for directional monotonicity
        compliance (`monotonicity.audit_scenario_monotonicity`).
        """
        from .monotonicity import audit_scenario_monotonicity

        eff_constraints = (
            constraints
            if constraints is not None
            else dict(self._config.features.monotonic_constraints)
        )
        return audit_scenario_monotonicity(
            sim_df,
            constraints=eff_constraints,
            scenarios=scenarios,
            tol=tol,
        )

    audit_scenario_monotonicity = audit_monotonicity_df

    def promo_elasticity_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        promo_col: str | None = None,
        price_col: str | None = None,
        confounders: Sequence[str] | None = None,
        summary: bool = False,
        use_similarity_priors: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str | None = None,
        adstock_decay: float | str = 0.0,
        include_halo: bool = True,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Compute cross-fitted Double-ML promo & price elasticity decomposition
        (`elasticity.compute_promo_elasticity_decomposition`).
        """
        from . import playground
        from .elasticity import compute_promo_elasticity_decomposition, summarize_promo_elasticity

        panel = data
        if panel is None and self._data is not None:
            panel = self._data
        if (
            panel is None
            and self._auto_session is not None
            and self._auto_session.data_df is not None
        ):
            panel = self._auto_session.data_df
        if panel is None and run_id is not None:
            try:
                from .engines.bigquery_names import source_ref
                from .engines.bigquery_sql import series_filter
                from .registry.reads import resolve_settings

                resolved = resolve_settings(self._settings)
                source = source_ref(self._config, resolved.registry_dataset_ref)
                idc, datec = self._config.data.ts_id_col, self._config.data.date_col
                sfilter = series_filter(self._config, source, idc)
                where_clause = f" WHERE {sfilter}" if sfilter else ""
                sql = f"SELECT * FROM `{source}`{where_clause} ORDER BY {idc}, {datec}"
                client = bigquery_client(resolved.project_id)
                df = client.query(sql).to_dataframe()
                if not df.empty:
                    panel = df
            except Exception:  # noqa: BLE001 - fall back to synthetic sample data when offline
                panel = None
        if panel is None:
            panel = playground.sample_data(
                n_series=min(self._config.data.series_limit or 3, 5),
                freq=self._config.data.freq,
                with_exog=True,
            )
        eff_use_sim = (
            bool(similarity_pooling)
            and (similarity_weight is None or float(similarity_weight) > 0.0)
            if similarity_pooling is not None
            else (
                bool(use_similarity_priors)
                and (similarity_weight is None or float(similarity_weight) > 0.0)
            )
        )
        eff_sim_method = similarity_method or self._config.features.similarity_method
        decomp = compute_promo_elasticity_decomposition(
            panel,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            promo_col=promo_col,
            price_col=price_col,
            confounders=confounders,
            freq=self._config.data.freq,
            holidays=self._config.features.holidays,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=similarity_matrix,
            similarity_method=eff_sim_method,
            adstock_decay=adstock_decay,
            include_halo=include_halo,
        )
        return summarize_promo_elasticity(decomp) if summary else decomp

    promo_elasticity = promo_elasticity_df

    def plot_promo_elasticity(
        self,
        data: pd.DataFrame | None = None,
        *,
        ts_id: str | None = None,
        promo_col: str | None = None,
        price_col: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot organic baseline vs. promotional lift and post-promo dip."""
        from .elasticity import PROMO_ELASTICITY_COLUMNS, plot_promo_elasticity

        if data is not None and set(PROMO_ELASTICITY_COLUMNS).issubset(data.columns):
            decomp = data
        else:
            decomp = self.promo_elasticity_df(
                data, promo_col=promo_col, price_col=price_col, summary=False
            )
        return plot_promo_elasticity(decomp, ts_id=ts_id, ax=ax, title=title)

    def cold_start_transfer_df(
        self,
        data: pd.DataFrame | None = None,
        *,
        forecasts_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        min_obs: int | None = None,
        min_history_points: int = 14,
        method: str | None = None,
        matching: str | None = None,
        cold_start_matching: str | None = None,
        similarity_method: str | None = None,
        text_cols: Sequence[str] | str | None = None,
        attribute_cols: Sequence[str] | None = None,
        hierarchy_cols: Sequence[str] | None = None,
        top_k_donors: int = 3,
        model_type: str | None = None,
        limit: int = 20000,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Synthesize cold-start / NPI trajectories via analog donor matching & seasonal shape
        transfer (`cold_start.cold_start_transfer_df`).
        """
        from .cold_start import cold_start_transfer_df as _cold_start_transfer_df

        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        panel = hist_df if hist_df is not None else data
        if panel is None or panel.empty:
            from . import playground

            panel = playground.sample_data(
                n_series=min(self._config.data.series_limit or 3, 5),
                freq=self._config.data.freq,
                with_exog=True,
            )
        eff_matching = cold_start_matching if cold_start_matching is not None else matching
        eff_hier_cols = (
            list(hierarchy_cols)
            if hierarchy_cols is not None
            else [c for lvl in self._config.hierarchy.levels for c in lvl]
        )
        return _cold_start_transfer_df(
            panel,
            cfg=self._config,
            forecasts_df=preds_df if not preds_df.empty else None,
            oof_df=eff_oof,
            min_obs=min_obs if min_obs is not None else min_history_points,
            method=method,  # type: ignore[arg-type]
            matching=eff_matching,  # type: ignore[arg-type]
            similarity_method=similarity_method,  # type: ignore[arg-type]
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            static_covariates=self._config.features.static_covariates,
            hierarchy_cols=eff_hier_cols,
            top_k_donors=top_k_donors,
            ts_id_col=self._config.data.ts_id_col,
            date_col=self._config.data.date_col,
            target_col=self._config.data.target_col,
            horizon=self._config.data.horizon,
            freq=self._config.data.freq,
            **kwargs,
        )

    def aci_df(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        actuals_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        run_id: str | None = None,
        gamma: float | None = None,
        target_coverage: float = 0.80,
        target_alpha: float | None = None,
        normalized: bool = True,
        method: str | None = None,
        horizon_delay: bool = True,
        model_type: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Compute Adaptive Conformal Inference (ACI) online coverage tracking and recalibrated
        forward intervals (`aci.compute_aci_df`).
        """
        from .aci import compute_aci_df

        preds_df, eff_oof, _ = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        eff_gamma = gamma if gamma is not None else float(self._config.output.aci_gamma)
        eff_cov = (1.0 - float(target_alpha)) if target_alpha is not None else target_coverage
        eff_method = method or self._config.output.calibration_method
        if eff_method == "conformal":
            eff_method = "aci_normalized" if normalized else "aci"
        source_oof = (
            eff_oof
            if (eff_oof is not None and not eff_oof.empty)
            else (actuals_df if actuals_df is not None else preds_df)
        )
        return compute_aci_df(
            source_oof,
            predictions_df=preds_df if not preds_df.empty else None,
            gamma=eff_gamma,
            target_coverage=eff_cov,
            normalized=normalized,
            method=eff_method,  # type: ignore[arg-type]
            horizon_delay=horizon_delay,
            ts_id_col=self._config.data.ts_id_col,
        )

    def sample_trajectories(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        run_id: str | None = None,
        n_paths: int = 500,
        copula_type: str = "auto",
        df_t: float = 4.0,
        cross_series: bool = True,
        reconcile: bool = True,
        reconcile_hierarchy: bool = True,
        similarity_shrinkage: bool = True,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str | None = None,
        floor: float | None = None,
        cap: float | None = None,
        seed: int = 42,
        model_type: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Generate joint cross-series & cross-horizon copula trajectory paths
        (`copula.sample_trajectories`).
        """
        from .copula import sample_trajectories as _sample_trajectories
        from .reconciliation import build_hierarchy

        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        eff_reconcile = reconcile and reconcile_hierarchy
        hier_spec = None
        if (
            eff_reconcile
            and (self._config.hierarchy.levels or self._config.hierarchy.similarity_hierarchy)
            and hist_df is not None
            and not hist_df.empty
        ):
            needed = {c for lvl in self._config.hierarchy.levels for c in lvl}
            if needed.issubset(hist_df.columns) or self._config.hierarchy.similarity_hierarchy:
                _, hier_spec = build_hierarchy(hist_df, self._config)
        eff_sim_method = similarity_method or self._config.features.similarity_method
        return _sample_trajectories(
            preds_df,
            oof_df=eff_oof,
            history_df=hist_df,
            cfg=self._config,
            hier_spec=hier_spec,
            n_paths=n_paths,
            copula_type=copula_type,
            df_t=df_t,
            cross_series=cross_series,
            reconcile=eff_reconcile,
            simplex_sum=self._config.features.simplex_sum,
            floor=floor if floor is not None else self._config.output.floor,
            cap=cap if cap is not None else self._config.output.cap,
            similarity_shrinkage=similarity_shrinkage,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=similarity_matrix,
            similarity_method=eff_sim_method,
            seed=seed,
        )

    def tail_risk_df(
        self,
        trajectories_df: pd.DataFrame | None = None,
        *,
        forecasts_df: pd.DataFrame | None = None,
        actuals_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        run_id: str | None = None,
        lead_time: int | None = None,
        window: int = 7,
        alphas: Sequence[float] = (0.90, 0.95, 0.99),
        alpha: float = 0.95,
        threshold: float | Mapping[str, float] | None = None,
        holding_cost: float | Mapping[str, float] | None = None,
        stockout_cost: float | Mapping[str, float] | None = None,
        n_paths: int = 300,
        copula_type: str = "auto",
        reconcile: bool = True,
        similarity_shrinkage: bool = True,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str | None = None,
        seed: int = 42,
        model_type: str | None = None,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Compute portfolio Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected
        Shortfall) across joint copula trajectories (`copula.compute_tail_risk`).
        """
        from .copula import compute_tail_risk

        preds_df, eff_oof, _ = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
            limit=limit,
        )
        eff_traj = (
            trajectories_df
            if trajectories_df is not None
            else self.sample_trajectories(
                forecasts_df=preds_df,
                oof_df=eff_oof,
                data=data,
                run_id=run_id,
                n_paths=n_paths,
                copula_type=copula_type,
                reconcile=reconcile,
                similarity_shrinkage=similarity_shrinkage,
                similarity_weight=similarity_weight,
                similarity_matrix=similarity_matrix,
                similarity_method=similarity_method,
                seed=seed,
                model_type=model_type,
                limit=limit,
            )
        )
        return compute_tail_risk(
            eff_traj,
            predictions_df=preds_df if not preds_df.empty else None,
            actuals_df=actuals_df,
            oof_df=eff_oof,
            lead_time=lead_time,
            window=window,
            alphas=alphas,
            alpha=alpha,
            threshold=threshold,
            holding_cost=holding_cost,  # type: ignore[arg-type]
            stockout_cost=stockout_cost,  # type: ignore[arg-type]
        )

    def optimize_inventory_df(
        self,
        *,
        forecasts_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        run_id: str | None = None,
        inventory_df: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
        trajectories_df: pd.DataFrame | None = None,
        copula_paths: pd.DataFrame | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        review_period: float = 0.0,
        service_level: float = 0.95,
        fill_rate: float | None = None,
        target_portfolio_fill_rate: float | None = None,
        max_budget: float | None = None,
        max_volume: float | None = None,
        max_total_units: float | None = None,
        holding_cost: float = 1.0,
        stockout_cost: float = 9.0,
        unit_cost: float | Mapping[str, float] | str = 10.0,
        unit_volume: float | Mapping[str, float] | str = 1.0,
        min_order_qty: float = 0.0,
        order_multiple: float = 0.0,
        vendor_moq: float | Mapping[str, float] | str | None = None,
        on_hand: float = 0.0,
        on_order: float = 0.0,
        backorders: float = 0.0,
        uncertainty_source: str = "analytical",
        copula_type: str = "auto",
        similarity_shrinkage: bool = True,
        similarity_pooling: bool | None = None,
        similarity_weight: float | str = "auto",
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str | None = None,
        n_paths: int = 300,
        seed: int = 42,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Solve multi-SKU portfolio-constrained inventory replenishment under working-capital,
        warehouse-volume, or aggregate service-level targets
        (`inventory_optimizer.optimize_inventory_portfolio`).
        """
        from .inventory_optimizer import optimize_inventory_portfolio

        eff_model = model_name or model_type
        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            forecasts_df=forecasts_df,
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=eff_model,
            limit=limit,
        )
        ts_col = self._config.data.ts_id_col
        if (
            forecasts_df is None
            and hist_df is not None
            and not hist_df.empty
            and ts_col in hist_df.columns
            and "ts_id" in preds_df.columns
        ):
            leaf_ids = set(hist_df[ts_col].astype(str).unique())
            if preds_df["ts_id"].astype(str).isin(leaf_ids).any():
                preds_df = preds_df[preds_df["ts_id"].astype(str).isin(leaf_ids)].copy()
                if eff_oof is not None and not eff_oof.empty and "ts_id" in eff_oof.columns:
                    eff_oof = eff_oof[eff_oof["ts_id"].astype(str).isin(leaf_ids)].copy()
        eff_unc = "copula" if uncertainty_source == "copula" else "analytical"
        eff_sim_shrink = (
            bool(similarity_pooling)
            if similarity_pooling is not None
            else bool(similarity_shrinkage)
        )
        eff_sim_method = similarity_method or self._config.features.similarity_method
        return optimize_inventory_portfolio(
            preds_df,
            oof_df=eff_oof,
            history_df=hist_df,
            inventory_df=inventory_df,
            trajectories_df=trajectories_df,
            copula_paths=copula_paths,
            model_name=eff_model,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            review_period=review_period,
            service_level=service_level if service_level is not None else 0.95,
            fill_rate=fill_rate,
            target_portfolio_fill_rate=target_portfolio_fill_rate,
            max_budget=max_budget,
            max_volume=max_volume,
            max_total_units=max_total_units,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            unit_cost=unit_cost,  # type: ignore[arg-type]
            unit_volume=unit_volume,  # type: ignore[arg-type]
            min_order_qty=min_order_qty,
            order_multiple=order_multiple,
            vendor_moq=vendor_moq if vendor_moq is not None else min_order_qty,  # type: ignore[arg-type]
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            uncertainty_source=eff_unc,  # type: ignore[arg-type]
            copula_type=copula_type,
            similarity_shrinkage=eff_sim_shrink,
            similarity_weight=similarity_weight,  # type: ignore[arg-type]
            similarity_matrix=similarity_matrix,
            similarity_method=eff_sim_method,
            n_paths=n_paths,
            seed=seed,
        )

    def optimize_scenario(
        self,
        *,
        data: pd.DataFrame | None = None,
        future_covariates_df: pd.DataFrame | None = None,
        run_id: str | None = None,
        objective: str = "gross_profit",
        price_col: str | None = None,
        promo_col: str | None = None,
        price_bounds: tuple[float, float] = (-0.15, 0.15),
        price_grid: Sequence[float] | None = None,
        max_promo_days: int | None = None,
        max_promo_periods: int | None = None,
        min_cooldown_days: int = 0,
        min_promo_gap: int = 0,
        min_margin_pct: float | None = None,
        min_total_units: float | None = None,
        max_total_units: float | None = None,
        max_promo_budget: float | None = None,
        inventory_cap: float | Mapping[str, float] | None = None,
        unit_cost: float | str | Mapping[str, float] | None = None,
        promo_fixed_cost: float = 0.0,
        risk_aversion_lambda: float = 0.0,
        cvar_alpha: float = 0.95,
        cannibalization_weight: float = 1.0,
        cannibalization_strength: float | None = None,
        use_similarity_priors: bool = True,
        cross_sku_elasticity: bool | None = None,
        similarity_matrix: pd.DataFrame | Any = None,
        similarity_method: str | None = None,
        reconcile: bool = True,
        include_inventory_impact: bool = True,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        n_paths: int = 250,
        copula_type: str = "auto",
        seed: int = 42,
        oof_df: pd.DataFrame | None = None,
        model_type: str | None = None,
        strict_feasibility: bool = False,
    ) -> Any:
        """Solve prescriptive price & promotional calendar optimization over Double-ML elasticity
        surfaces (`scenario_optimizer.optimize_scenario`).
        """
        from . import playground
        from .scenario_optimizer import optimize_scenario as _optimize_scenario

        preds_df, eff_oof, hist_df = self._resolve_track_k_frames(
            data=data,
            oof_df=oof_df,
            run_id=run_id,
            model_type=model_type,
        )
        panel = hist_df if hist_df is not None else data
        if panel is None or panel.empty:
            panel = playground.sample_data(
                n_series=min(self._config.data.series_limit or 3, 5),
                freq=self._config.data.freq,
                with_exog=True,
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
        eff_sim_method = similarity_method or self._config.features.similarity_method
        return _optimize_scenario(
            panel,
            config=self._config,
            predictions_df=preds_df if not preds_df.empty else None,
            oof_df=eff_oof,
            history_df=panel,
            future_covariates_df=future_covariates_df,
            model_type=model_type,
            objective=objective,  # type: ignore[arg-type]
            price_col=price_col,
            promo_col=promo_col,
            price_bounds=price_bounds,
            price_grid=price_grid,
            max_promo_days=max_promo_days if max_promo_days is not None else max_promo_periods,
            min_cooldown_days=min_cooldown_days if min_cooldown_days else min_promo_gap,
            unit_cost=unit_cost,  # type: ignore[arg-type]
            promo_fixed_cost=promo_fixed_cost,
            min_margin_pct=min_margin_pct,
            min_total_units=min_total_units,
            max_total_units=max_total_units,
            max_promo_budget=max_promo_budget,
            inventory_cap=inventory_cap,
            risk_aversion_lambda=risk_aversion_lambda,
            cvar_alpha=cvar_alpha,
            cannibalization_weight=eff_cannib,
            use_similarity_priors=eff_use_sim,
            similarity_matrix=similarity_matrix,
            similarity_method=eff_sim_method,
            reconcile=reconcile,
            include_inventory_impact=include_inventory_impact,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level,
            n_paths=n_paths,
            copula_type=copula_type,
            seed=seed,
            strict_feasibility=strict_feasibility,
        )

    def score_realized(
        self,
        actuals_df: pd.DataFrame,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        drift_threshold: float = 1.5,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Score published forward predictions for ``run_id`` against newly arrived ``actuals_df``
        and flag series whose live error exceeds ``drift_threshold`` times backtest expectation.
        """
        import pandas as pd

        from .diagnostics import compute_realized_accuracy
        from .registry.reads import read_oof, read_predictions

        rid = run_id or self.run_id
        pred_rows = read_predictions(rid, limit=limit, settings=self._settings)
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else None
        eff_metric = metric or self._config.backtest.decision_metric
        return compute_realized_accuracy(
            pred_df,
            actuals_df,
            oof_df=oof_df,
            metric=eff_metric,
            drift_threshold=drift_threshold,
        )

    def revision_stability_df(
        self,
        run_ids: Sequence[str],
        *,
        jump_threshold_pct: float = 0.30,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Measure cross-run forecast revision stability ("whipsaw") across ``run_ids``."""
        return self.registry().revision_stability_df(
            run_ids, jump_threshold_pct=jump_threshold_pct, limit=limit
        )

    def _effective_config_for_run(self, run_id: str) -> RunConfig:
        """Return the stored `RunConfig` for ``run_id`` when inspecting a past run, falling back
        to ``self._config``.
        """
        if run_id == self.run_id:
            return self._config
        with contextlib.suppress(Exception):
            from .registry.reads import read_run_config

            raw = read_run_config(run_id, settings=self._settings)
            if raw:
                try:
                    return RunConfig.model_validate(raw)
                except Exception:  # noqa: BLE001 - legacy past run config fallback
                    data_blk = raw.get("data") if isinstance(raw.get("data"), dict) else {}
                    if data_blk and "source_table" in data_blk:
                        return RunConfig.model_validate(
                            {
                                "run_name": str(raw.get("run_name") or "past-run"),
                                "data": {
                                    k: v
                                    for k, v in data_blk.items()
                                    if k
                                    in (
                                        "source_table",
                                        "ts_id_col",
                                        "date_col",
                                        "target_col",
                                        "freq",
                                        "horizon",
                                        "series_limit",
                                    )
                                },
                                "models": ["naive_seasonal"],
                            }
                        )
        return self._config

    def detect_anomalies_df(
        self,
        run_id: str | None = None,
        *,
        model_name: str | None = None,
        anomaly_prob_threshold: float = 0.95,
        cutoff_back_steps: int | None = None,
        anomalies_only: bool = False,
        actuals_df: pd.DataFrame | None = None,
        engine: str = "auto",
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Detect time-series anomalies across any of the 37 models, 9 ensembles, or past runs.

        Supports four complementary execution paths controlled by ``model_name`` and ``engine``:

        1. **In-database BigQuery ML (`engine="auto"` or `"bqml"` with `arima_plus` / `timesfm`):**
           Runs `ML.DETECT_ANOMALIES(MODEL ...)` for `arima_plus` / `arima_plus@xreg` or serverless
           `AI.DETECT_ANOMALIES` (`TimesFM`) directly in BigQuery. If inspecting an older past
           ``run_id`` whose BQML model object was already dropped and ``engine="auto"``,
           automatically falls back to the stored ``backtest_oof`` intervals.
        2. **Universal Model / Ensemble OOF (`model_name="<any model>"`, `"ensemble_*"`, `"all"`,
           or `engine="oof"`):** Scores stored ``backtest_oof`` out-of-fold predictions and
           calibrated prediction intervals for **any** statistical, ML, deep-learning, Vertex AI
           AutoML, BigQuery ML, or ensemble model on current or past runs via
           :func:`diagnostics.compute_anomalies` — zero re-fitting required.
        3. **Forward Forecast Realized Actuals (`actuals_df=...`):** Joins published
           ``forecast_predictions`` intervals against newly arrived ``actuals_df`` to flag live
           post-cutoff anomalies.
        4. **Model-Free STL + Robust MAD (`model_name="stl"` or `engine="stl"`):** Decomposes raw
           historical series when no backtest OOF intervals are available.

        Returns a tidy ``DataFrame`` with columns ``["ts_id", "model_type", "ds", "y", "yhat",
        "lower_bound", "upper_bound", "anomaly_probability", "is_anomaly"]``.
        """
        import pandas as pd

        from .diagnostics import compute_anomalies
        from .registry.reads import (
            read_oof,
            read_predictions,
            read_series_history,
            resolve_settings,
        )

        rid = run_id or self.run_id
        eff_cfg = self._effective_config_for_run(rid)
        conf_level = 0.80
        freq = str(eff_cfg.data.freq)

        # Mode 3: Forward predictions scored against newly arrived actuals_df.
        if actuals_df is not None:
            model_filter = [model_name] if model_name and model_name not in ("all", "stl") else None
            pred_rows = read_predictions(
                rid, models=model_filter, limit=limit, settings=self._settings
            )
            oof_rows = read_oof(rid, models=model_filter, limit=limit, settings=self._settings)
            pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
            oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else None
            return compute_anomalies(
                predictions_df=pred_df,
                actuals_df=actuals_df,
                oof_df=oof_df,
                model_name=model_name,
                anomaly_prob_threshold=anomaly_prob_threshold,
                confidence_level=conf_level,
                anomalies_only=anomalies_only,
                freq=freq,
            )

        # Mode 4: Explicit model-free STL decomposition over historical series.
        if engine == "stl" or model_name == "stl":
            hist_rows = read_series_history(
                rid, history_tail=cutoff_back_steps or 1000, settings=self._settings
            )
            hist_df = pd.DataFrame.from_records(hist_rows) if hist_rows else pd.DataFrame()
            return compute_anomalies(
                series_df=hist_df,
                model_name="stl",
                anomaly_prob_threshold=anomaly_prob_threshold,
                confidence_level=conf_level,
                anomalies_only=anomalies_only,
                freq=freq,
            )

        # Resolve default model_name when omitted.
        eff_model = model_name
        if eff_model is None:
            if engine == "bqml":
                native_candidates = [
                    m
                    for m in eff_cfg.models
                    if m.split("@", 1)[0] in ("arima_plus", "arima_plus_xreg", "timesfm")
                ]
                eff_model = native_candidates[0] if native_candidates else "arima_plus"
            elif (
                engine == "auto"
                and len(eff_cfg.models) == 1
                and eff_cfg.models[0].split("@", 1)[0]
                in ("arima_plus", "arima_plus_xreg", "timesfm")
                and not eff_cfg.ensemble.enabled
            ):
                eff_model = eff_cfg.models[0]

        base_requested = eff_model.split("@", 1)[0] if eff_model else ""
        if base_requested == "arima_plus_xreg":
            eff_model = "arima_plus@xreg"
            base_requested = "arima_plus"

        # Mode 1: Native BigQuery ML.DETECT_ANOMALIES / AI.DETECT_ANOMALIES when requested.
        if (
            engine in ("auto", "bqml")
            and eff_model is not None
            and base_requested in ("arima_plus", "timesfm")
        ):
            from google.cloud import bigquery

            from .engines.bigquery_sql import build_detect_anomalies_query

            resolved = resolve_settings(self._settings)
            sql = build_detect_anomalies_query(
                eff_cfg,
                model_name=eff_model,
                dataset=resolved.dataset_ref,
                registry_dataset=resolved.registry_dataset_ref,
                anomaly_prob_threshold=anomaly_prob_threshold,
                cutoff_back_steps=cutoff_back_steps,
                run_id=rid,
            )
            client = bigquery_client(resolved.project_id)
            try:
                df = client.query(
                    sql, job_config=bigquery.QueryJobConfig(use_query_cache=False)
                ).to_dataframe()
                if not df.empty:
                    if "model_type" not in df.columns:
                        df.insert(1, "model_type", eff_model)
                    if (
                        "yhat" not in df.columns
                        and "lower_bound" in df.columns
                        and "upper_bound" in df.columns
                    ):
                        df.insert(
                            4,
                            "yhat",
                            0.5
                            * (
                                pd.to_numeric(df["lower_bound"], errors="coerce")
                                + pd.to_numeric(df["upper_bound"], errors="coerce")
                            ),
                        )
                    ordered = [
                        c
                        for c in (
                            "ts_id",
                            "model_type",
                            "ds",
                            "y",
                            "yhat",
                            "lower_bound",
                            "upper_bound",
                            "anomaly_probability",
                            "is_anomaly",
                        )
                        if c in df.columns
                    ]
                    df = df[ordered + [c for c in df.columns if c not in ordered]]
                if anomalies_only and not df.empty and "is_anomaly" in df.columns:
                    return df.loc[df["is_anomaly"] == True].reset_index(drop=True)  # noqa: E712 - nullable boolean series
                return df
            except Exception:
                if engine == "bqml":
                    raise
                # Under engine="auto", if a past run's BQML model artifact was dropped, fall
                # through to stored backtest_oof below.

        # Mode 2: Universal backtest_oof residual & interval scoring across any model / ensemble.
        model_filter = [eff_model] if eff_model and eff_model != "all" else None
        oof_rows = read_oof(rid, models=model_filter, limit=limit, settings=self._settings)
        if oof_rows:
            oof_df = pd.DataFrame.from_records(oof_rows)
            if cutoff_back_steps is not None and cutoff_back_steps > 0:
                oof_df = (
                    oof_df.sort_values(["ts_id", "model_type", "forecast_date"])
                    .groupby(["ts_id", "model_type"], as_index=False)
                    .tail(int(cutoff_back_steps))
                )
            return compute_anomalies(
                oof_df=oof_df,
                model_name=eff_model,
                anomaly_prob_threshold=anomaly_prob_threshold,
                confidence_level=conf_level,
                anomalies_only=anomalies_only,
                freq=freq,
            )

        # Fallback when a run had backtest.enabled=False: use STL over historical series.
        hist_rows = read_series_history(
            rid, history_tail=cutoff_back_steps or 1000, settings=self._settings
        )
        hist_df = pd.DataFrame.from_records(hist_rows) if hist_rows else pd.DataFrame()
        return compute_anomalies(
            series_df=hist_df,
            model_name=eff_model or "stl",
            anomaly_prob_threshold=anomaly_prob_threshold,
            confidence_level=conf_level,
            anomalies_only=anomalies_only,
            freq=freq,
        )

    def detect_change_points_df(
        self,
        run_id: str | None = None,
        *,
        model_name: str | None = None,
        exclude_gap_edges: bool = True,
        min_gap_steps: int = 1,
        sigma_threshold: float = 3.0,
        engine: str = "auto",
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Detect structural regime change points across raw series or any model's residuals on
        current or past runs.

        * **Raw series mode (`model_name=None` or `"raw"`):** Runs BigQuery's native
          `ML.DETECT_CHANGE_POINTS` with gap-edge artifact filtering when ``engine in ("auto",
          "bqml")``, or pure-Python binary segmentation (:func:`diagnostics.compute_change_points`)
          when ``engine="oof"``.
        * **Model-residual mode (`model_name="<model>"`, `"ensemble_*"`, or `"all"`):** Reads
          stored ``backtest_oof`` trajectories for ``run_id`` and segments the model's residuals
          $e_t = y_t - \\hat{y}_t$ via :func:`diagnostics.compute_change_points` to pinpoint
          structural regime shifts that the model failed to anticipate (distinguishing unmodeled
          structural breaks from seasonal or promotional jumps the model already explained).

        Returns a tidy ``DataFrame`` with columns ``["ts_id", "model_type", "begin_date",
        "end_date", "window_count", "window_avg", "window_min", "window_max", "window_stddev",
        "gap_edge_missing_steps", "is_gap_edge", "status"]``.
        """
        import pandas as pd

        from .diagnostics import compute_change_points
        from .registry.reads import read_oof, read_series_history, resolve_settings

        rid = run_id or self.run_id
        eff_cfg = self._effective_config_for_run(rid)
        freq = str(eff_cfg.data.freq)

        # Model-residual mode: segment backtest_oof residuals for any model or ensemble.
        if model_name is not None and model_name != "raw":
            eff_model = "arima_plus@xreg" if model_name == "arima_plus_xreg" else model_name
            model_filter = [eff_model] if eff_model != "all" else None
            oof_rows = read_oof(rid, models=model_filter, limit=limit, settings=self._settings)
            oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
            return compute_change_points(
                oof_df=oof_df,
                model_name=eff_model,
                exclude_gap_edges=exclude_gap_edges,
                min_gap_steps=min_gap_steps,
                freq=freq,
                sigma_threshold=sigma_threshold,
            )

        # Raw series mode: BigQuery ML.DETECT_CHANGE_POINTS when engine in ("auto", "bqml").
        if engine in ("auto", "bqml"):
            from google.cloud import bigquery

            from .engines.bigquery_sql import build_detect_change_points_query

            resolved = resolve_settings(self._settings)
            sql = build_detect_change_points_query(
                eff_cfg,
                dataset=resolved.dataset_ref,
                exclude_gap_edges=exclude_gap_edges,
                min_gap_steps=min_gap_steps,
            )
            client = bigquery_client(resolved.project_id)
            try:
                df = client.query(
                    sql, job_config=bigquery.QueryJobConfig(use_query_cache=False)
                ).to_dataframe()
                if "model_type" not in df.columns:
                    df.insert(1 if "ts_id" in df.columns else 0, "model_type", "raw")
                return df
            except Exception:
                if engine == "bqml":
                    raise

        # Offline / fallback raw series change-point segmentation from stored history.
        hist_rows = read_series_history(rid, history_tail=2000, settings=self._settings)
        hist_df = pd.DataFrame.from_records(hist_rows) if hist_rows else pd.DataFrame()
        return compute_change_points(
            series_df=hist_df,
            model_name="raw",
            exclude_gap_edges=exclude_gap_edges,
            min_gap_steps=min_gap_steps,
            freq=freq,
            sigma_threshold=sigma_threshold,
        )

    def exceptions_df(
        self,
        run_id: str | None = None,
        *,
        actuals_df: pd.DataFrame | None = None,
        prior_run_ids: Sequence[str] | None = None,
        top_n: int = 50,
        limit: int = 20000,
    ) -> pd.DataFrame:
        """Build the Top-N operational exception triage queue for ``run_id``."""
        import pandas as pd

        from .diagnostics import compute_exceptions_queue
        from .registry.reads import read_oof, read_predictions

        rid = run_id or self.run_id
        if (
            self._auto_session is not None
            and self._auto_session.where == "local"
            and (run_id is None or run_id == self.run_id)
        ):
            return compute_exceptions_queue(
                oof_df=self._auto_session.oof_df,
                predictions_df=self._auto_session.raw_predictions_df.rename(
                    columns={"forecast_date": "ds"}
                ),
                floor=self._config.output.floor,
                cap=self._config.output.cap,
                metric=self._config.backtest.decision_metric,
                top_n=top_n,
            )
        if (
            self._auto_session is None
            and (
                self._predictions_df is not None
                or self._oof_df is not None
                or self._data is not None
            )
            and (run_id is None or run_id == self.run_id)
        ):
            loc_pred, loc_oof, _ = self._resolve_track_k_frames(limit=limit)
            return compute_exceptions_queue(
                oof_df=loc_oof,
                predictions_df=loc_pred.rename(columns={"forecast_date": "ds"}),
                floor=self._config.output.floor,
                cap=self._config.output.cap,
                metric=self._config.backtest.decision_metric,
                top_n=top_n,
            )
        pred_rows = read_predictions(rid, limit=limit, settings=self._settings)
        oof_rows = read_oof(rid, limit=limit, settings=self._settings)
        pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
        oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()

        real_df = (
            self.score_realized(actuals_df, run_id=rid, limit=limit)
            if actuals_df is not None and not actuals_df.empty
            else None
        )
        stab_df = (
            self.revision_stability_df([*prior_run_ids, rid], limit=limit)
            if prior_run_ids
            else None
        )
        return compute_exceptions_queue(
            oof_df=oof_df,
            predictions_df=pred_df,
            realized_df=real_df,
            stability_df=stab_df,
            floor=self._config.output.floor,
            cap=self._config.output.cap,
            metric=self._config.backtest.decision_metric,
            top_n=top_n,
        )

    def export_report(
        self,
        out_path: str | Any,
        *,
        run_id: str | None = None,
        title: str | None = None,
        predictions_df: pd.DataFrame | None = None,
        oof_df: pd.DataFrame | None = None,
        leaderboard_df: pd.DataFrame | None = None,
        promo_elasticity_df: pd.DataFrame | None = None,
        scenarios_df: pd.DataFrame | None = None,
        goal_seek_df: pd.DataFrame | None = None,
        aci_df: pd.DataFrame | None = None,
        tail_risk_df: pd.DataFrame | None = None,
        cold_start_df: pd.DataFrame | None = None,
        inventory_opt_df: pd.DataFrame | None = None,
        scenario_opt_df: pd.DataFrame | None = None,
        similarity_df: pd.DataFrame | None = None,
        similarity_clusters_df: pd.DataFrame | None = None,
        substitutes_df: pd.DataFrame | None = None,
        emerging_trends_df: pd.DataFrame | None = None,
        write_json: bool = False,
        include_json_sidecar: bool = False,
    ) -> Any:
        """Export a self-contained `.html` report for ``run_id``."""
        import pandas as pd

        from .report import export_report

        rid = run_id or self.run_id

        def _safe_df(fn: Any) -> pd.DataFrame:
            try:
                return fn(rid)
            except Exception:  # noqa: BLE001 - allow offline report export without SF_* env vars
                return pd.DataFrame()

        eff_preds = (
            predictions_df
            if predictions_df is not None
            else (
                self._predictions_df
                if self._predictions_df is not None
                else _safe_df(self.predictions_df)
            )
        )
        eff_oof = oof_df if oof_df is not None else self._oof_df
        eff_sim = similarity_df
        if eff_sim is None:
            try:
                eff_sim = self.similarity_df(run_id=rid)
            except Exception:  # noqa: BLE001 - offline report fallback
                eff_sim = pd.DataFrame()
        eff_sim_clusters = similarity_clusters_df
        if eff_sim_clusters is None:
            try:
                eff_sim_clusters = self.similarity_clusters_df(run_id=rid)
            except Exception:  # noqa: BLE001 - offline report fallback
                eff_sim_clusters = pd.DataFrame()
        eff_subs = substitutes_df
        eff_trends = emerging_trends_df
        if eff_trends is None:
            try:
                eff_trends = self.emerging_trends_df(run_id=rid)
            except Exception:  # noqa: BLE001 - offline report fallback
                eff_trends = pd.DataFrame()
        payload_dict: dict[str, Any] = {
            "run_id": rid,
            "leaderboard_df": (
                leaderboard_df if leaderboard_df is not None else _safe_df(self.leaderboard_df)
            ),
            "fva_df": _safe_df(self.fva_df),
            "horizon_decay_df": _safe_df(self.horizon_decay_df),
            "bias_df": _safe_df(self.bias_report_df),
            "exceptions_df": _safe_df(self.exceptions_df),
            "inventory_policy_df": _safe_df(self.inventory_policy_df),
            "predictions_df": eff_preds,
            "oof_df": eff_oof,
            "promo_elasticity_df": promo_elasticity_df,
            "scenarios_df": scenarios_df,
            "goal_seek_df": goal_seek_df,
            "aci_df": aci_df,
            "tail_risk_df": tail_risk_df,
            "cold_start_df": cold_start_df,
            "inventory_opt_df": inventory_opt_df,
            "scenario_opt_df": scenario_opt_df,
            "similarity_df": eff_sim,
            "similarity_clusters_df": eff_sim_clusters,
            "substitutes_df": eff_subs,
            "emerging_trends_df": eff_trends,
        }
        if self._auto_session is not None:
            payload_dict["autofocus_ledger_df"] = self.autofocus_ledger_df()
            payload_dict["blueprint_rationale_df"] = self._auto_session.blueprint.rationale_df
            payload_dict["history_df"] = self._auto_session.data_df
        elif self._data is not None:
            payload_dict["history_df"] = self._data
        return export_report(
            payload_dict,
            out_path,
            title=title,
            metric=self._config.backtest.decision_metric,
            promo_elasticity_df=promo_elasticity_df,
            scenarios_df=scenarios_df,
            goal_seek_df=goal_seek_df,
            aci_df=aci_df,
            tail_risk_df=tail_risk_df,
            cold_start_df=cold_start_df,
            inventory_opt_df=inventory_opt_df,
            scenario_opt_df=scenario_opt_df,
            similarity_df=eff_sim,
            similarity_clusters_df=eff_sim_clusters,
            substitutes_df=eff_subs,
            emerging_trends_df=eff_trends,
            write_json=write_json,
            include_json_sidecar=include_json_sidecar,
        )

    def sop_summary_df(
        self,
        run_id: str | None = None,
        *,
        limit: int = 25,
    ) -> pd.DataFrame:
        """Return the Executive S&OP Rollup (`v_executive_sop_summary` in BigQuery or computed
        offline for local in-memory `Forecaster` sessions).
        """
        import pandas as pd

        is_local = (run_id is None or run_id == self.run_id) and (
            (self._auto_session is not None and self._auto_session.where == "local")
            or (
                self._auto_session is None
                and (
                    self._predictions_df is not None
                    or self._oof_df is not None
                    or self._data is not None
                )
            )
        )
        if is_local:
            from .bq_agent.offline import build_offline_sop_summary_df

            return build_offline_sop_summary_df(self, run_id=self.run_id, default_n_series=0)

        from .registry.reads import read_executive_sop_summary

        rows = read_executive_sop_summary(
            run_id=run_id or self.run_id,
            limit=limit,
            settings=self._settings,
        )
        return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

    def probe_bq_agent(
        self,
        *,
        agent_id: str | None = None,
        location: str = "global",
    ) -> Any:
        """Check whether BigQuery Conversational Analytics & **Knowledge Catalog** APIs are
        enabled and authorized for this forecaster's registry without raising
        (`bq_agent.probe.probe_bq_agent_access`).
        """
        from .bq_agent.probe import probe_bq_agent_access

        return probe_bq_agent_access(
            agent_id=agent_id,
            location=location,
            settings=self._settings,
        )

    def ensure_bq_agent(self, **kwargs: Any) -> Any:
        """Idempotently provision or update the stateful BigQuery Conversational Analytics
        `DataAgent` and **Knowledge Catalog** glossary/scans for this forecaster's registry
        (`bq_agent.client.ensure_registry_agent`).
        """
        from .bq_agent.client import ensure_registry_agent

        return ensure_registry_agent(settings=self._settings, **kwargs)

    def ask(
        self,
        question: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        stateful: bool = True,
        agent_id: str | None = None,
        auto_ensure_agent: bool = True,
        offline: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Ask a natural-language question against this forecaster's results.

        Routes automatically to the offline deterministic router (`bq_agent.offline.ask_offline`)
        when called on an in-memory local `Forecaster` session (or when ``offline=True``), and to
        the stateful BigQuery Conversational Analytics `DataAgent` (`bq_agent.client.ask_registry`)
        scoped to ``run_id or self.run_id`` for BigQuery-backed runs.
        """
        use_offline = offline is True or (
            offline is None
            and run_id is None
            and (
                (self._auto_session is not None and self._auto_session.where == "local")
                or (
                    self._auto_session is None
                    and (
                        self._predictions_df is not None
                        or self._oof_df is not None
                        or self._data is not None
                    )
                )
            )
        )
        if use_offline:
            from .bq_agent.offline import ask_offline

            return ask_offline(self, question, conversation_id=conversation_id)

        from .bq_agent.client import ask_registry

        return ask_registry(
            question,
            run_id=run_id or self.run_id,
            conversation_id=conversation_id,
            stateful=stateful,
            auto_ensure_agent=auto_ensure_agent,
            agent_id=agent_id,
            settings=self._settings,
            **kwargs,
        )
