"""Tripwire for the live-validation ledger (`docs/validation.md`).

The ledger is the only record that something was proven on live infrastructure. Its value depends
entirely on nobody being able to leave it quietly wrong, so these tests enforce the invariants a
human would otherwise have to remember:

- every smoke config has exactly one row, and every row names a real config (no ghosts, no gaps);
- every axis a row depends on is a declared axis;
- **a `CURRENT` row's axis values still match the current architecture.** This is the one that
  matters. When a code change moves an axis (as `822ae25` moved `ray_deps`), every row pinned to the
  old value fails here until someone re-runs it or honestly marks it `STALE`.

The ledger is parsed rather than generated: one human-readable file, no second copy to drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "docs" / "validation.md"
_SMOKE_CONFIGS = _REPO_ROOT / "configs" / "smokes"

_VALID_STATUSES = frozenset({"CURRENT", "STALE", "NEVER_RUN", "NEEDS_RECHECK"})
_UNSET = frozenset({"—", "-", "", "not recorded"})


def _table_rows(markdown: str, header_starts_with: str) -> list[list[str]]:
    """Cells of every body row of the first table whose header starts with ``header_starts_with``.

    A markdown table is a run of consecutive ``|``-delimited lines; the first is the header and the
    second the ``---`` separator, so the body is everything from the third line on.
    """
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"| {header_starts_with}"):
            body = []
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                body.append([c.strip() for c in row.strip().strip("|").split("|")])
            return body
    raise AssertionError(f"no table found with a header starting {header_starts_with!r}")


def _parse_axes(cell: str) -> dict[str, str]:
    """The ``axis=value`` pairs in an "Axes at proof" cell, stripped of backtick formatting."""
    if cell in _UNSET:
        return {}
    pairs = {}
    for chunk in cell.split(","):
        item = chunk.strip().strip("`")
        if not item:
            continue
        axis, _, value = item.partition("=")
        pairs[axis.strip()] = value.strip()
    return pairs


@pytest.fixture(scope="module")
def ledger() -> str:
    return _LEDGER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def current_axes(ledger: str) -> dict[str, str]:
    """The architecture-axes table as ``{axis: current_value}``."""
    return {
        row[0].strip("`"): row[1].strip("`")
        for row in _table_rows(ledger, "Axis ")
    }


@pytest.fixture(scope="module")
def entries(ledger: str) -> list[dict[str, str]]:
    """Smoke rows as dicts. Column order is fixed by the ledger's own header."""
    rows = _table_rows(ledger, "# ")
    return [
        {
            "id": r[0],
            "config": r[1].strip("`"),
            "status": r[3],
            "date": r[4],
            "run_id": r[5].strip("`"),
            "axes": r[6],
        }
        for r in rows
    ]


def test_ledger_exists() -> None:
    assert _LEDGER.is_file(), f"the validation ledger is missing at {_LEDGER}"


def test_every_smoke_config_has_exactly_one_entry(entries: list[dict[str, str]]) -> None:
    """A new smoke config must be entered in the ledger — even if only as NEVER_RUN."""
    on_disk = {p.name for p in _SMOKE_CONFIGS.glob("*.json")}
    in_ledger = [e["config"] for e in entries]

    assert len(in_ledger) == len(set(in_ledger)), "a config is listed twice in the ledger"
    assert set(in_ledger) == on_disk, (
        "the ledger and configs/smokes/ disagree.\n"
        f"  missing from the ledger: {sorted(on_disk - set(in_ledger))}\n"
        f"  ghost rows (no config):  {sorted(set(in_ledger) - on_disk)}"
    )


def test_statuses_are_recognised(entries: list[dict[str, str]]) -> None:
    for entry in entries:
        assert entry["status"] in _VALID_STATUSES, (
            f"smoke {entry['id']} has unknown status {entry['status']!r}; "
            f"expected one of {sorted(_VALID_STATUSES)}"
        )


def test_declared_axes_are_real_axes(
    entries: list[dict[str, str]], current_axes: dict[str, str]
) -> None:
    for entry in entries:
        for axis in _parse_axes(entry["axes"]):
            assert axis in current_axes, (
                f"smoke {entry['id']} depends on axis {axis!r}, which is not in the "
                f"architecture-axes table. Add the axis or fix the typo."
            )


def test_current_entries_match_the_current_architecture(
    entries: list[dict[str, str]], current_axes: dict[str, str]
) -> None:
    """The load-bearing check: a CURRENT claim must rest on today's architecture.

    If this fails, a code change moved an axis out from under a passing result. Re-run that
    validation, or mark the row STALE — do not edit the axis value to make the test pass.
    """
    for entry in entries:
        if entry["status"] != "CURRENT":
            continue
        for axis, proof_value in _parse_axes(entry["axes"]).items():
            assert proof_value == current_axes[axis], (
                f"smoke {entry['id']} claims CURRENT but was proven on "
                f"{axis}={proof_value!r}, and {axis} is now {current_axes[axis]!r}. "
                f"Re-run it, or mark the row STALE."
            )


def test_stale_entries_name_the_axis_that_moved(
    entries: list[dict[str, str]], current_axes: dict[str, str]
) -> None:
    """STALE must be justified — else it is a CURRENT row downgraded to dodge a failing check."""
    for entry in entries:
        if entry["status"] != "STALE":
            continue
        proof_axes = _parse_axes(entry["axes"])
        drifted = [a for a, v in proof_axes.items() if v != current_axes[a]]
        assert drifted, (
            f"smoke {entry['id']} is marked STALE but every axis it declares still holds its "
            f"proof-time value. Either it is actually CURRENT, or it depends on an axis it "
            f"does not declare."
        )


def test_run_entries_are_dated_and_never_run_entries_are_not(
    entries: list[dict[str, str]],
) -> None:
    for entry in entries:
        has_date = entry["date"] not in _UNSET
        if entry["status"] == "NEVER_RUN":
            assert not has_date, (
                f"smoke {entry['id']} is NEVER_RUN but carries a date {entry['date']!r}"
            )
        else:
            assert has_date, f"smoke {entry['id']} is {entry['status']} but has no date"


def test_known_gaps_section_is_present(ledger: str) -> None:
    """The gaps list is the ledger's honesty valve; losing it would hide what is untested."""
    assert re.search(r"^## Known validation gaps", ledger, re.MULTILINE), (
        "the 'Known validation gaps' section is missing from the ledger"
    )
