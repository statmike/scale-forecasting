# API Reference

This reference is generated directly from the source docstrings, so it always matches the code.

`scale_forecasting` exposes three doors onto the same forecasting core, plus one for managing what
those runs leave behind:

- **The easy path** — [`Forecaster`](sdk.md): point it at a config, call `dry_run()` / `run()` /
  `review()`.
- **The orchestration entrypoint** — [`run`](main.md): the single function every runtime dispatches
  through, used identically locally and in the cloud.
- **The direct path** — the cell primitives ([`run_cell`](worker.md), and the group/chunk runners in
  [Spark core](engines_spark_io.md) and [Ray core](engines_ray_io.md)) for embedding the model
  machinery in your own Spark `applyInPandas` or Ray tasks.
- **The manage path** — [`registry.ops`](registry_ops.md) (and the `Registry` SDK class on the
  [same page as `Forecaster`](sdk.md)): inspect a registry, drop a run across every tier, sweep
  orphaned artifacts, snapshot or export.

Supporting surfaces: [configuration](config.md) and [settings](settings.md), the
[error taxonomy](errors.md), the [model factory](models.md) and [`BaseModel`](models_base_model.md)
contract, [metrics](metrics.md), [backtesting](backtest.md), the [router](router.md), and the
analyst [registry views](registry_views.md).

The package front door lazy-loads the heavy names via `__getattr__`, so `import scale_forecasting`
stays fast; each page below documents the concrete module a name resolves to.
