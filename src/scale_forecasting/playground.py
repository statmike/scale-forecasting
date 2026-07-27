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

# A small, fast default: enough history for yearly seasonality, cheap to fit interactively.
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
    cfg = GenConfig(n_series=n_series, history=history, freq=freq, with_exog=with_exog)
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

    cfg = build_config(
        model, freq=freq, horizon=horizon, backtest=backtest, with_exog=with_exog
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
            f"{k}={v:.3f}" for k, v in r.metrics.items() if v == v  # skip NaN
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
