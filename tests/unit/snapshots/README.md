# Snapshots

Two kinds of file live here, and they are regenerated differently.

**SQL snapshots** — `ddl_deployment.sql`, `ddl_drop.sql`, `views.sql`, `bigquery_native.sql`. Each is
the rendered output of a generator, compared verbatim by `test_ddl.py`, `test_views.py` and
`test_bigquery_sql.py`. They exist so a change to generated SQL shows up as a reviewable diff rather
than as a silent deployment difference.

**Identity and output snapshots** — `run_ids_prebreak.json`, `golden_panel_prebreak.json`. Written by
`uv run python tests/unit/test_prebreak_snapshots.py --write`, read by that module's tests. They pin
what the `run_id` digests and the model outputs were *before* the planned identity break, which is
the only point at which "every id moved" and "no number changed" are checkable claims. Read that
module's docstring before regenerating either one — the failures they produce are the deliverable,
and regenerating to clear a red gate throws away the thing being measured.
