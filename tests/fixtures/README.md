# Test fixtures

`golden.py` (BUILD Phase 2) builds the deterministic 12-series golden panel via the
real `data_gen.generator`, so tests and shipped data share one code path (TESTING §1).
`tiny_config.json` is a `series_limit`-subset config exercising each config branch.
