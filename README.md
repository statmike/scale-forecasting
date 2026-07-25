# scale-forecasting

**Massively-parallel time-series forecasting on Google Cloud — Spark, Ray, and BigQuery, one config away.**

Forecast tens of thousands of time series in parallel, backtest and ensemble many
methods, and capture every run's lineage in BigQuery — from a local notebook or from
Airflow, with the *same* code. Deploy the whole thing into a fresh project with one
`terraform apply`, complete with 100k example series to run against immediately.

> ⚠️ **Status: under construction.** This README is a skeleton; a 5-minute quickstart,
> architecture diagram, and cost/quota notes land in the final polish phase.

---

## Why it exists

Large-scale forecasting usually means a pile of bespoke cluster code that only its
author understands. This project is the opposite bet: a small, readable codebase —
**one capability per file** — that still scales to 100k+ series, so a data scientist
can open any file, understand it in one read, and fork it to their needs.

## What it does

- **Two Python runtimes** — Dataproc Serverless (**Spark**) and **Ray** on Vertex AI —
  run the identical per-series unit of work; pick one per run via config.
- **BigQuery-native models** (ARIMA_PLUS, ARIMA_PLUS_XREG, TimesFM) run **in parallel**,
  SQL-only, no Python compute.
- **Backtesting** (expanding/sliding folds, full metric panel) and **ensembling**
  (calculated + learned) out of the box.
- **BigQuery lineage** — every run's config, per-model metrics, forecasts, and artifact
  links are captured in managed Iceberg tables.
- **Config-driven** — one JSON file describes the whole run; the same file runs locally
  and under Composer.

## Architecture

A run is one validated JSON config. It selects the data, the model list, one Python
runtime, backtest and ensemble settings — and is persisted verbatim into the run
registry, so the config *is* the experiment record.

- **Runtimes.** The Python model suite runs on **Spark _xor_ Ray** (one per run).
  BigQuery-native models are additive and run **in parallel** with either. So a run is
  "Spark + optional BQ models" or "Ray + optional BQ models" — never Spark + Ray.
- **The one unit of work.** `worker.run_cell(series, model, cfg) -> CellResult` fits,
  (optionally) backtests, and predicts one `(series, model)` cell. The *same* function
  runs locally, inside a Spark Pandas UDF, and inside a Ray task. Engines differ only in
  how they fan it out and collect results — that's what makes "same code everywhere" real.
- **Scale without a bottleneck.** Workers return data, not RPCs; results are written to
  BigQuery in bulk (Storage Write API). Parallelism is bounded by compute, not a tracking
  server's QPS.
- **Lineage.** Three managed-Iceberg tiers — `run_registry` (config) →
  `forecast_metadata` (metrics + GCS artifact links) → `forecast_predictions` (values),
  plus `backtest_oof` for learned ensembling.

Read `src/scale_forecasting/config.py` (the run contract) and
`src/scale_forecasting/worker.py` (the unit of work) to see the whole shape.

## Quickstart

_Coming in the polish phase._ In brief:

```bash
uv sync
python -c "import scale_forecasting; print(scale_forecasting.__version__)"
```

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
