#!/usr/bin/env python3
"""Prepend a one-click "open this notebook" header to every notebook in ``notebooks/``.

Each notebook gets a scrubbed markdown header with a single **functional badge** — Run in Colab
Enterprise (one-click import) — plus a one-line runbook naming the Colab Enterprise template to pick
(the templates carry the ``SF_*`` env, so it's open → pick template → Run all). No tracking pixel,
no social/author links (generic-product charter). Idempotent: the header is delimited by an HTML
marker, so re-running replaces it in place rather than stacking duplicates.

Run from the repo root: ``python scripts/add_notebook_headers.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import nbformat

_REPO = "statmike/scale-forecasting"
_BRANCH = "main"
_MARKER = "<!--- sf-header --->"

# Which Colab Enterprise template each notebook wants (mirrors notebook_acceptance.REGISTRY).
_SPARK_CONNECT = {"01_spark_via_connect"}


def _template_note(stem: str) -> str:
    if stem in _SPARK_CONNECT:
        return (
            "Runs on the **`sf-spark-connect`** runtime template (Python 3.12) for the interactive "
            "Spark Connect path. From `sf-main` (3.11) it still runs, via the remote-batch "
            "fallback."
        )
    if stem == "model_playground":
        return (
            "Runs on the **`sf-main`** runtime template (Python 3.11) — or fully locally with just "
            "ADC + a clone; the playground needs no cloud at all."
        )
    return "Runs on the **`sf-main`** runtime template (Python 3.11)."


def _header_markdown(stem: str) -> str:
    path = f"notebooks/{stem}.ipynb"
    raw = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/{path}"
    colab_ent = f"https://console.cloud.google.com/vertex-ai/colab/import/{quote(raw, safe='')}"

    colab_ent_logo = "https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN"

    return f"""{_MARKER}
<table align="left">
<tr>
  <td style="text-align: center">
    <a href="{colab_ent}">
      <img width="32px" src="{colab_ent_logo}" alt="Colab Enterprise logo">
      <br>Run in<br>Colab Enterprise
    </a>
  </td>
</tr>
</table>
<br clear="left"/>

> **Run in Colab Enterprise:** click the badge to import this notebook, pick a runtime, and
> **Run all**. The Terraform-deployed templates already carry the `SF_*` run identity in their env,
> so there's no environment cell to fill in. {_template_note(stem)} See
> [`docs/notebook_runtimes.md`](https://github.com/{_REPO}/blob/{_BRANCH}/docs/notebook_runtimes.md)
> for the per-notebook template mapping and the headless acceptance harness.
"""


def _apply(notebook_path: Path) -> str:
    """Insert or replace the header cell in one notebook. Returns 'added' | 'replaced'."""
    nb = nbformat.read(notebook_path, as_version=4)
    header = _header_markdown(notebook_path.stem)
    first = nb.cells[0] if nb.cells else None
    if first is not None and first.cell_type == "markdown" and _MARKER in "".join(first.source):
        first.source = header
        action = "replaced"
    else:
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(header))
        action = "added"
    nbformat.write(nb, notebook_path)
    return action


def main() -> int:
    notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
    paths = sorted(notebooks_dir.glob("*.ipynb"))
    if not paths:
        print(f"no notebooks found in {notebooks_dir}", file=sys.stderr)
        return 1
    for path in paths:
        action = _apply(path)
        print(f"{action:>8}  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
