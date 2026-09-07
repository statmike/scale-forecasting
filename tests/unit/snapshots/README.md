# Snapshots

Two kinds of file live here, and they are regenerated differently.

**SQL snapshots** — `ddl_deployment.sql`, `ddl_drop.sql`, `views.sql`, `bigquery_native.sql`. Each is
the rendered output of a generator, compared verbatim by `test_ddl.py`, `test_views.py` and
`test_bigquery_sql.py`. They exist so a change to generated SQL shows up as a reviewable diff rather
than as a silent deployment difference.

**Identity and output snapshots** — four files, in two pairs, read by `test_prebreak_snapshots.py`.

`run_ids_prebreak.json` and `golden_panel_prebreak.json` pin what the `run_id` digests and the model
outputs were *before* the planned identity break, which is the only point at which "every id moved"
and "no number changed" are checkable claims. **They are a historical record and `--write` does not
touch them.** Restore them from git if one is ever lost.

`run_ids.json` and `golden_panel.json` pin the same two things as of *today*, and they are the pair
that keeps working: the pre-break claims stop discriminating once they are settled, while these fail
in the commit that moves something. Regenerate with
`uv run python tests/unit/test_prebreak_snapshots.py --write` — but read that module's docstring
first. The failures these produce are the deliverable, and regenerating to clear a red gate throws
away the thing being measured.
