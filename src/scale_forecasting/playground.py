"""Local model-dev loop — try any model on sample data, offline (DESIGN §13.2).

This is the on-ramp for a data scientist opening the repo: pick a model, get a small
sample dataset, run the *real* worker cell, and read the prediction frame + metric panel —
no GCP, no config files. The notebook (``notebooks/model_playground.ipynb``) and the CLI
(``python -m scale_forecasting.playground``) are both thin skins over the functions here,
so the dev loop and production run the identical code path (G1): sample → validate →
:func:`worker.run_cell`.

Adding a model needs no change here — :func:`available_models` reads the factory registry,
so a new file under ``models/`` that ends in ``register(...)`` appears automatically.

Public surface: ``available_models``, ``sample_data``, ``build_config``, ``run_model``,
``summarize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RunConfig
from .data_gen.generator import GenConfig, generate_panel
from .models import get_model, list_models
from .validation import validate_panel
from .worker import CellResult, run_cell

# Fixed seed so the sample panel is reproducible run-to-run (determinism, not size).
_SAMPLE_SEED = 20260726


def available_models(*, include_bigquery: bool = False) -> list[str]:
    """Registered model names, sorted (the factory is the source of truth).

    BigQuery-native models run as SQL in BigQuery (Arc B), not in a local cell, so they're
    excluded by default — the playground runs the Python models offline.
    """
    names = list_models()
    if include_bigquery:
        return names
    return [n for n in names if get_model(n).runtime != "bigquery"]


def model_catalog() -> pd.DataFrame:
    """Every registered model with the runtimes it can run on — the whole suite, one table.

    One row per model (Python *and* BigQuery-native), so the story is complete: which model
    runs where. The where-it-runs flags are derived from the model's class metadata, not a
    hand-maintained list, so a new model file shows up here automatically:

    - **Python-runtime** models run the *identical* cell code (``worker.run_cell``) on every
      Python compute — ``local`` (this playground), ``spark`` (Dataproc fan-out), and ``ray``
      (Ray on Vertex). The one ``deep_learning`` model additionally uses the Ray **GPU** pool.
    - **BigQuery-native** models run only as SQL in ``bigquery`` (``engines/bigquery_engine``);
      they can't run in a local/Spark/Ray Python cell (their in-process fit/predict raise).

    Columns: ``model, family, runtime, local, spark, ray, gpu, bigquery, exog``. Booleans, so
    ``df`` renders as a tidy capability matrix in the notebook.
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
                "local": is_python,
                "spark": is_python,
                "ray": is_python,
                "gpu": is_python and cls.family == "deep_learning",  # only the GPU pool needs it
                "bigquery": cls.runtime == "bigquery",
                "exog": cls.supports_exog,
            }
        )
    # runtime then family then name → Python models first, native last; stable and readable.
    df = pd.DataFrame(rows).sort_values(["runtime", "family", "model"]).reset_index(drop=True)
    return df


def sample_data(
    *,
    n_series: int = 3,
    history: int = 730,
    freq: str = "D",
    with_exog: bool = False,
    seed: int = _SAMPLE_SEED,
) -> pd.DataFrame:
    """A small deterministic sample panel from the real generator (DESIGN §13.1).

    Same code path as the shipped 100k dataset, so what you see locally is what runs at
    scale. Columns: ``ts_id, archetype, ds, y`` (+ ``price_index`` when ``with_exog``).
    Defaults to 3 series × 2 years daily — enough to show trend + seasonality + intervals.
    """
    cfg = GenConfig(history=history, freq=freq, with_exog=with_exog)
    return generate_panel(n_series, cfg, seed)


def build_config(
    model: str,
    *,
    freq: str = "D",
    horizon: int = 28,
    backtest: bool = False,
    with_exog: bool = False,
) -> RunConfig:
    """A minimal :class:`RunConfig` for running ``model`` on :func:`sample_data`.

    Wires the generator's column names (``ts_id``/``ds``/``y``, and ``price_index`` as the
    exog role) so the config matches the sample panel out of the box. ``backtest`` turns on
    a small 3-fold CV so the metric panel is populated (otherwise metrics are NaN by design
    — full-fit runs don't score themselves).
    """
    exog = ["price_index"] if with_exog else []
    backtest_cfg: dict[str, Any] = (
        {"enabled": True, "n_folds": 3, "horizon": horizon, "step": horizon}
        if backtest
        else {"enabled": False}
    )
    # Built as a plain dict and validated, mirroring config.load_config — pydantic coerces
    # the nested blocks and applies the same normalization a file-loaded config gets.
    raw: dict[str, Any] = {
        "run_name": f"playground-{model}",
        "data": {"source_table": "playground", "freq": freq, "horizon": horizon},
        "models": [model],
        "features": {"exog": exog},
        "backtest": backtest_cfg,
    }
    return RunConfig.model_validate(raw)


@dataclass(frozen=True)
class PlaygroundRun:
    """One model run on one series — the CellResult plus the series it ran on."""

    result: CellResult
    series: pd.DataFrame
    backtest: bool


@dataclass(frozen=True)
class BakeOff:
    """A local multi-model bake-off on one series: the leaderboard + everything to plot it.

    ``leaderboard`` is one row per model — base models *and* the ensemble pseudo-models —
    ranked by the decision metric. ``predictions`` is the long-format future forecast for
    every model (base + ensemble) so the notebook can overlay them. ``series`` is the history
    they were fit on; ``config`` is the frozen :class:`RunConfig` that drove the whole thing.
    """

    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    series: pd.DataFrame
    config: RunConfig


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
) -> BakeOff:
    """Run several models head-to-head on one series and add two ensembles — via the framework.

    The playground's showcase of the *whole* pipeline offline: it runs every model in
    ``models`` on one series with backtesting on (:func:`worker.run_cell`, the same cell a
    100k run executes), then builds **one calculated** and **one learned** ensemble by calling
    the *exact* framework functions the real ensemble stage uses — no reimplementation:

    - **calculated** (default ``inverse_error``): :func:`ensembler.combine_calculated` blends
      the base future forecasts, weighting by ``1/decision_metric`` from the backtest.
    - **learned** (default ``nnls``): :func:`ensembler.fit_learned` trains the meta-learner on
      the base backtest OOF, then :func:`ensemble_run._apply_weights` applies the weights.
    - **scoring**: :func:`ensembler.combine_oof` + :func:`metrics.compute_metrics` /
      :func:`worker._rollup_metrics` score every ensemble on the OOF window — the identical
      path :func:`ensemble_run.run_ensembles` uses, minus the BigQuery I/O.

    Because it drives real framework code, the leaderboard here is what a scaled run would
    produce for the same series. ``models`` defaults to a fast, reliable subset; pass your own
    (see :func:`available_models`) to compare any Python models. Returns a :class:`BakeOff`.
    """
    from .ensemble_run import _apply_weights
    from .ensembler import combine_calculated, combine_oof, fit_learned
    from .metrics import METRIC_NAMES, compute_metrics
    from .worker import _rollup_metrics

    if models is None:
        # Fast, dependency-light defaults so the bake-off runs anywhere the repo imports.
        models = ["theta", "holtwinters", "stl_bagging", "xgboost"]
    if data is None:
        data = sample_data(n_series=3, freq=freq, with_exog=with_exog)

    # One config for the whole bake-off: all base models, backtest on (ensembles need OOF), and
    # both an ensemble strategy from each family. Built + validated exactly like a file config.
    exog = ["price_index"] if with_exog else []
    raw: dict[str, Any] = {
        "run_name": "playground-bakeoff",
        "data": {"source_table": "playground", "freq": freq, "horizon": horizon},
        "models": models,
        "features": {"exog": exog},
        "backtest": {"enabled": True, "n_folds": n_folds, "horizon": horizon, "step": horizon},
        "ensemble": {"enabled": True, "strategies": [calculated, learned]},
    }
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

    for model in models:
        result = run_cell(series, model, cfg)
        status = result.status
        row: dict[str, Any] = {"model": model, "kind": "base", "status": status}
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
    ens_pred_rows: list[dict[str, Any]] = []
    for r in combine_calculated(base_df, cfg, metric_df):
        ens_pred_rows.append(r)
    learned_weights, _artifacts = fit_learned(oof_df, cfg)
    for strategy, wmap in learned_weights.items():
        ens_pred_rows.extend(_apply_weights(base_df, wmap, cfg.run_name, strategy))

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
    y_train = series[cfg.data.target_col].to_numpy()
    ens_oof = combine_oof(oof_df, cfg, learned_weights) if not oof_df.empty else pd.DataFrame()
    for model_type, g in ens_oof.groupby("model_type"):
        fold_panels = [
            compute_metrics(fg["y_true"].to_numpy(), fg["yhat"].to_numpy(), y_train=y_train)
            for _fold, fg in g.sort_values("forecast_date").groupby("fold_id")
        ]
        panel = _rollup_metrics(fold_panels)
        row = {"model": str(model_type), "kind": "ensemble", "status": "ok"}
        row.update({m: panel.get(m, float("nan")) for m in METRIC_NAMES})
        base_rows.append(row)

    leaderboard = (
        pd.DataFrame(base_rows).sort_values(decision, na_position="last").reset_index(drop=True)
    )
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return BakeOff(leaderboard=leaderboard, predictions=predictions, series=series, config=cfg)


def run_model(
    model: str,
    data: pd.DataFrame | None = None,
    *,
    ts_id: str | None = None,
    freq: str = "D",
    horizon: int = 28,
    backtest: bool = False,
    with_exog: bool = False,
) -> PlaygroundRun:
    """Run one model on one series and return the result (validates first).

    ``data`` defaults to a fresh :func:`sample_data` panel (matching ``freq``/``with_exog``).
    The whole panel is validated up front (:func:`validation.validate_panel`) so a shape
    problem fails fast with a clear message — the same guard a real run uses. Then a single
    series (``ts_id``, or the first one) is handed to :func:`worker.run_cell`, exactly as an
    engine would. ``run_cell`` never raises: a model failure comes back as an error cell.
    """
    if data is None:
        data = sample_data(n_series=3, freq=freq, with_exog=with_exog)

    cfg = build_config(model, freq=freq, horizon=horizon, backtest=backtest, with_exog=with_exog)
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
