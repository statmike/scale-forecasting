"""Persisted notebook content must carry nothing personal or corporate — AGENTS.md §1, rule 4.

Why this file exists: the notebooks commit their outputs so GitHub and the docs site render results
without execution, and an output is whatever the kernel printed — including a library warning
whose first token is the absolute path of the site-packages file that raised it. On a workstation
that path starts with a corporate home directory and an OS username, and on 2026-10-07 four cells
of notebook 00 and one of notebook 09 were found carrying exactly that. Nothing in the existing
gates reads notebook outputs, so nothing could have caught it.

What it checks: every markdown source, code source, and persisted text output (stream text and any
``text/*`` or JSON display payload; images are skipped) in ``notebooks/*.ipynb``, against the
patterns that identify a person or an internal network rather than the demo project:

* e-mail addresses, except service-account identities (``*.gserviceaccount.com``) and the RFC 2606
  documentation domains (``example.com`` and friends);
* absolute paths under a personal or corporate home directory — ``/usr/local/google/home/<name>``,
  ``/Users/<name>``, ``/home/<name>`` — except the service users a managed notebook runs as
  (``jupyter`` on Vertex AI Workbench, ``runner`` on GitHub Actions, and the like);
* internal Google hostnames and short links (``*.corp.google.com``, ``*.googlers.com``,
  ``*.googleplex.com``, ``go/<link>``);
* credential material: OAuth access tokens, API keys, private-key blocks, service-account key
  fields.

What it deliberately allows, because AGENTS.md decided it: the demo project ID, bucket names,
dataset names and service-account e-mails. Those are identifiers Google Cloud itself prints in
console URLs and samples; knowing them grants nothing without IAM.

On failure the message names the notebook, the cell, and the pattern — **never the matched text**,
because this test runs in a public CI log and a tripwire that republishes what it caught is worse
than none. Fix the notebook (silence the warning at its source, clear the cell output, or re-run the
cell), and do not widen the allowlists for a one-off.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOKS = sorted((_REPO_ROOT / "notebooks").glob("*.ipynb"))

# Home directories a managed or CI notebook legitimately runs under. Anything else under /home is a
# person.
_SERVICE_USERS = r"(?:jupyter|runner|user|root|spark|ray|dataproc|yarn|hadoop|airflow|colab)"

_FORBIDDEN: dict[str, re.Pattern[str]] = {
    "e-mail address": re.compile(
        r"\b[A-Za-z0-9._%+-]+@(?!(?:[A-Za-z0-9-]+\.)*(?:gserviceaccount\.com|example\.(?:com|org|net))\b)"
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
    ),
    "corporate home path": re.compile(r"/usr/local/google/home/[A-Za-z0-9._-]+"),
    "macOS home path": re.compile(r"/Users/[A-Za-z0-9._-]+(?=/|\b)"),
    "Linux home path": re.compile(
        rf"(?<!/google)/home/(?!{_SERVICE_USERS}\b)[A-Za-z0-9._-]+(?=/|\b)"
    ),
    "internal hostname": re.compile(
        r"\b[A-Za-z0-9.-]+\.(?:corp\.google\.com|googlers\.com|googleplex\.com)\b", re.IGNORECASE
    ),
    "internal short link": re.compile(r"(?<![A-Za-z0-9/.])go/[A-Za-z0-9_-]{2,}"),
    "OAuth access token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),
    "API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "service-account key field": re.compile(r'"private_key(?:_id)?"\s*:'),
}


def _texts(cell: dict) -> list[tuple[str, str]]:
    """``(where, text)`` for everything in a cell a reader could see: source and text outputs."""
    found = [("source", "".join(cell.get("source", [])))]
    for out in cell.get("outputs", []) or []:
        if "text" in out:
            found.append(
                (f"{out.get('output_type')}/{out.get('name', 'text')}", "".join(out["text"]))
            )
        for mime, payload in (out.get("data") or {}).items():
            if mime.startswith("text/") or mime.endswith("json"):
                text = "".join(payload) if isinstance(payload, list) else json.dumps(payload)
                found.append((f"{out.get('output_type')}/{mime}", text))
        if out.get("output_type") == "error":
            found.append(("error/traceback", "\n".join(out.get("traceback", []))))
    return found


def _violations(path: Path) -> list[str]:
    """One line per ``(cell, stream, pattern)`` with the summed match count — never the text.

    A kernel writes stderr in chunks and each chunk is its own output entry, so counts are summed
    per cell rather than reported per chunk.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[tuple[int, str, str], int] = {}
    for index, cell in enumerate(nb.get("cells", [])):
        for where, text in _texts(cell):
            for label, pattern in _FORBIDDEN.items():
                n = len(pattern.findall(text))
                if n:
                    counts[index, where, label] = counts.get((index, where, label), 0) + n
    return [
        f"{path.name} cell {i} ({where}): {n} × {label}" for (i, where, label), n in counts.items()
    ]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=[p.stem for p in _NOTEBOOKS])
def test_persisted_notebook_content_carries_no_personal_or_internal_identifiers(path: Path) -> None:
    """Sources and persisted outputs are clean under AGENTS.md §1.4 (values are never printed)."""
    hits = _violations(path)
    assert not hits, (
        "persisted notebook content matches a forbidden pattern (matched text withheld — this log "
        "is public):\n  "
        + "\n  ".join(hits)
        + "\nSilence the warning at its source or clear the cell "
        "output, then re-run; do not widen the allowlists in tests/unit/test_notebook_hygiene.py."
    )


def test_the_notebook_set_is_the_one_the_ledger_tracks() -> None:
    """Eleven notebooks today; a new one is covered automatically, a vanished one is noticed."""
    assert len(_NOTEBOOKS) >= 11, [p.name for p in _NOTEBOOKS]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=[p.stem for p in _NOTEBOOKS])
def test_notebook_at_a_glance_and_closing_cells(path: Path) -> None:
    """Every notebook opens with an 'At a glance' table and closes with a 'Where next' cell."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    assert len(cells) >= 3, f"{path.name}: expected at least 3 cells"

    intro_src = "".join(cells[1].get("source", []))
    for required in (
        "### At a glance",
        "**Learning outcomes**",
        "**Prerequisites**",
        "**Estimated time**",
        "**Estimated run cost**",
        "**Ongoing cost**",
    ):
        assert required in intro_src, f"{path.name} cell 1 missing {required!r}"

    closing_src = "".join(cells[-1].get("source", []))
    assert cells[-1].get("cell_type") == "markdown", f"{path.name} final cell must be markdown"
    assert "## Where next" in closing_src, f"{path.name} final cell missing '## Where next'"


@pytest.mark.parametrize(
    ("label", "sample", "should_match"),
    [
        ("e-mail address", "contact someone@corp-mail.example.io today", True),
        ("e-mail address", "sa: sf-worker@my-project.iam.gserviceaccount.com", False),
        ("e-mail address", "user_id: <the launching user email>", False),
        ("e-mail address", "placeholder owner@example.com", False),
        (
            "corporate home path",
            "/usr/local/google/home/someone/.venv/lib/x.py:44: UserWarning",
            True,
        ),
        ("Linux home path", "/home/someone/proj/x.py", True),
        ("Linux home path", "/home/jupyter/scale-forecasting/notebooks", False),
        ("macOS home path", "/Users/someone/code", True),
        ("internal hostname", "host ws1.c.googlers.com", True),
        ("internal hostname", "https://cloud.google.com/vertex-ai", False),
        ("internal short link", "see go/some-doc for details", True),
        ("internal short link", "https://example.com/go/path", False),
        ("API key", "AIza" + "A" * 35, True),
        ("private key block", "-----BEGIN PRIVATE KEY-----", True),
    ],
)
def test_the_patterns_mean_what_the_docstring_says(
    label: str, sample: str, should_match: bool
) -> None:
    """Pin each pattern's intent so a regex edit cannot quietly widen or narrow the rule."""
    assert bool(_FORBIDDEN[label].search(sample)) is should_match
