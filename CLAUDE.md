# CLAUDE.md — working in this repo

Guidance for any AI/dev contributor working in `scale-forecasting`.

## What this project is

A modular, Terraform-deployable system for massively-parallel time-series forecasting on
Google Cloud. Two Python runtimes (Spark **xor** Ray per run) plus BigQuery-native models
running in parallel, with backtesting, ensembling, and full BigQuery lineage. Read
`DESIGN.md` for the architecture.

## The one rule that overrides everything else

**Elegance.** This system does something large and complex, but any data scientist must be
able to open a file and understand it in one read. **No bloat** — no speculative
abstractions, no dead parameters, no defensive cruft, no re-implementing a library. If a
helper isn't used in ≥2 places, inline or delete it.

## How the code is organized

- **One capability per file.** Each file has a small, named public surface; everything
  else is private (`_prefixed`). Don't merge files or add files outside the documented tree.
- **Pure vs. I/O split.** `models/`, `features.py`, `backtest.py`, `metrics.py`,
  `ensembler.py`, `worker.run_cell`, `config.py`, `registry/ids.py` are **pure** (no GCP,
  unit-testable offline). Only `registry/`, `engines/`, `main.py`, `data_gen/seed_spark.py`
  touch the network.
- **The G1 seam:** `worker.run_cell(series, model, cfg) -> CellResult` is the single unit
  of work that runs identically local, in a Spark UDF, and in a Ray task. Engines only
  differ in *how they call it and collect results*.
- **Adding a model** = one new file under `models/` implementing `BaseModel` + one
  `register(...)` call. No other edits.

## Conventions

- Python 3.11, type hints on every public function, `from __future__ import annotations`.
- Arrow-friendly dtypes at boundaries; pandas DataFrames at the model/worker seam.
- Errors: raise `ScaleForecastError` subclasses; a worker cell **captures** errors into
  its result, it never crashes the batch.
- Determinism: any randomness takes an explicit seed.

## Working discipline

- **Test-gated.** A capability is done only when its test command is green and the output
  is shown — never on assertion alone.
- **ADC everywhere**, no service-account keys.
- Environment: `uv sync`; run tests with `pytest`; lint with `ruff`; types with `mypy`.
