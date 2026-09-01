"""Tripwire for the live-validation ledger (`docs/validation.md`).

The ledger is the only record that something was proven on live infrastructure. Its value depends
entirely on nobody being able to leave it quietly wrong, so these tests enforce the invariants a
human would otherwise have to remember:

- every artefact that can be run live has exactly one row, and every row names a real artefact
  (no ghosts, no gaps) — across all three surfaces: the smoke suite, the demonstration and scale
  configs, and the notebooks;
- every axis a row depends on is a declared axis;
- **a `CURRENT` row's axis values still match the current architecture.** This is the one that
  matters. When a code change moves an axis (as `822ae25` moved `ray_deps`), every row pinned to the
  old value fails here until someone re-runs it or honestly marks it `STALE`.

The per-row invariants are enforced over **every** surface rather than the smoke table alone. That
was the original shape of this file and it left the notebooks unguarded: their rows carry statuses
and axes in the same vocabulary, and nothing checked them, so a notebook could go on claiming
`CURRENT` across an axis change that the smoke rows were correctly downgraded for.

The ledger is parsed rather than generated: one human-readable file, no second copy to drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "docs" / "validation.md"
_SMOKE_CONFIGS = _REPO_ROOT / "configs" / "smokes"
_DEMO_CONFIGS = _REPO_ROOT / "configs"
_NOTEBOOKS = _REPO_ROOT / "notebooks"

# Not a run config: a zone/region failover map read at submit time. It has no `run_name`, cannot be
# handed to `main`, and so cannot have a live result.
_NOT_A_RUN_CONFIG = frozenset({"compute_fallback.json"})

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
    return {row[0].strip("`"): row[1].strip("`") for row in _table_rows(ledger, "Axis ")}


@pytest.fixture(scope="module")
def smoke_entries(ledger: str) -> list[dict[str, str]]:
    """Smoke rows. Column order is fixed by the ledger's own header."""
    return [
        {
            "surface": "smoke",
            "id": f"smoke {r[0]}",
            "artefact": r[1].strip("`"),
            "status": r[3],
            "date": r[4],
            "run_id": r[5].strip("`"),
            "axes": r[6],
        }
        for r in _table_rows(ledger, "# ")
    ]


@pytest.fixture(scope="module")
def demo_entries(ledger: str) -> list[dict[str, str]]:
    """Demonstration and scale config rows — the day-one surface and the 100k headline."""
    return [
        {
            "surface": "demo",
            "id": f"config {r[0].strip('`')}",
            "artefact": r[0].strip("`"),
            "status": r[2],
            "date": r[3],
            "run_id": r[4].strip("`"),
            "axes": r[5],
        }
        for r in _table_rows(ledger, "Config ")
    ]


@pytest.fixture(scope="module")
def notebook_entries(ledger: str) -> list[dict[str, str]]:
    """Notebook rows. No `run_id` column — a notebook is a session, not a single run."""
    return [
        {
            "surface": "notebook",
            "id": f"notebook {r[0].strip('`')}",
            "artefact": r[0].strip("`"),
            "status": r[1],
            "date": r[2],
            "run_id": "",
            "axes": r[3],
        }
        for r in _table_rows(ledger, "Notebook ")
    ]


@pytest.fixture(scope="module")
def entries(
    smoke_entries: list[dict[str, str]],
    demo_entries: list[dict[str, str]],
    notebook_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Every row on every surface, in one list. The per-row invariants apply to all of them."""
    return [*smoke_entries, *demo_entries, *notebook_entries]


def test_ledger_exists() -> None:
    assert _LEDGER.is_file(), f"the validation ledger is missing at {_LEDGER}"


def _assert_one_row_each(entries: list[dict[str, str]], on_disk: set[str], surface: str) -> None:
    """The no-ghosts, no-gaps check, shared by the three surfaces."""
    listed = [e["artefact"] for e in entries]
    assert len(listed) == len(set(listed)), f"a {surface} is listed twice in the ledger"
    assert set(listed) == on_disk, (
        f"the ledger and the {surface} files on disk disagree.\n"
        f"  missing from the ledger: {sorted(on_disk - set(listed))}\n"
        f"  ghost rows (no file):    {sorted(set(listed) - on_disk)}"
    )


def test_every_smoke_config_has_exactly_one_entry(smoke_entries: list[dict[str, str]]) -> None:
    """A new smoke config must be entered in the ledger — even if only as NEVER_RUN."""
    _assert_one_row_each(
        smoke_entries, {p.name for p in _SMOKE_CONFIGS.glob("*.json")}, "smoke config"
    )


def test_every_demonstration_config_has_exactly_one_entry(
    demo_entries: list[dict[str, str]],
) -> None:
    """Same discipline for the surface a user actually runs, which had no rows at all until now."""
    on_disk = {p.name for p in _DEMO_CONFIGS.glob("*.json")} - _NOT_A_RUN_CONFIG
    _assert_one_row_each(demo_entries, on_disk, "demonstration config")


def test_every_notebook_has_exactly_one_entry(notebook_entries: list[dict[str, str]]) -> None:
    """A notebook ships with its output cells, so an unlisted one is an unrecorded live result."""
    _assert_one_row_each(notebook_entries, {p.name for p in _NOTEBOOKS.glob("*.ipynb")}, "notebook")


def test_statuses_are_recognised(entries: list[dict[str, str]]) -> None:
    for entry in entries:
        assert entry["status"] in _VALID_STATUSES, (
            f"{entry['id']} has unknown status {entry['status']!r}; "
            f"expected one of {sorted(_VALID_STATUSES)}"
        )


def test_the_capabilities_table_uses_the_same_status_vocabulary(ledger: str) -> None:
    """It has no axes or dates, so only its statuses can be checked — but they must be real ones."""
    for row in _table_rows(ledger, "Capability "):
        assert row[1] in _VALID_STATUSES, (
            f"capability {row[0]!r} has unknown status {row[1]!r}; "
            f"expected one of {sorted(_VALID_STATUSES)}"
        )


def test_declared_axes_are_real_axes(
    entries: list[dict[str, str]], current_axes: dict[str, str]
) -> None:
    for entry in entries:
        for axis in _parse_axes(entry["axes"]):
            assert axis in current_axes, (
                f"{entry['id']} depends on axis {axis!r}, which is not in the "
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
                f"{entry['id']} claims CURRENT but was proven on "
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
            f"{entry['id']} is marked STALE but every axis it declares still holds its "
            f"proof-time value. Either it is actually CURRENT, or it depends on an axis it "
            f"does not declare."
        )


def test_run_entries_are_dated_and_never_run_entries_are_not(
    entries: list[dict[str, str]],
) -> None:
    for entry in entries:
        has_date = entry["date"] not in _UNSET
        if entry["status"] == "NEVER_RUN":
            assert not has_date, f"{entry['id']} is NEVER_RUN but carries a date {entry['date']!r}"
        else:
            assert has_date, f"{entry['id']} is {entry['status']} but has no date"


def test_a_never_run_entry_cannot_carry_evidence(entries: list[dict[str, str]]) -> None:
    """NEVER_RUN with a `run_id` or a declared axis is a row someone half-updated after a run."""
    for entry in entries:
        if entry["status"] != "NEVER_RUN":
            continue
        assert entry["run_id"] in _UNSET, (
            f"{entry['id']} is NEVER_RUN but records run_id {entry['run_id']!r}"
        )
        assert not _parse_axes(entry["axes"]), (
            f"{entry['id']} is NEVER_RUN but declares axes {entry['axes']!r}; "
            f"axes are what a result was proven on, so an unrun row has none"
        )


def test_a_proven_entry_declares_at_least_one_axis(entries: list[dict[str, str]]) -> None:
    """Every live result rests on *something*. A bare CURRENT row is a claim with no expiry.

    `python` is the floor: any run at all executes our code on some interpreter, so a row that
    declares nothing has not been thought about rather than genuinely depending on nothing.
    """
    for entry in entries:
        if entry["status"] not in {"CURRENT", "STALE"}:
            continue
        assert _parse_axes(entry["axes"]), (
            f"{entry['id']} is {entry['status']} but declares no architecture axes. "
            f"Name what it depends on, at minimum the interpreter."
        )


def test_known_gaps_section_is_present(ledger: str) -> None:
    """The gaps list is the ledger's honesty valve; losing it would hide what is untested."""
    assert re.search(r"^## Known validation gaps", ledger, re.MULTILINE), (
        "the 'Known validation gaps' section is missing from the ledger"
    )
