"""Automated documentation, Mermaid, table, link, and config example tripwire.

Verifies across every Markdown file in the repository (root ``README.md``, ``AGENTS.md``,
``docs/**/*.md``, and subdirectory ``README.md`` files):

1. Every Markdown table has matching unescaped column counts across header, separator, and body.
2. Every ``mermaid`` diagram block uses a supported diagram type, balances ``subgraph``/``end``
   blocks, and double-quotes node labels containing parentheses or ``<br/>`` tags.
3. Every complete ``RunConfig`` JSON snippet in ``README.md`` and ``docs/**/*.md`` validates cleanly
   through ``RunConfig.model_validate`` without dropping learned ensemble strategies.
4. Every relative file link and every ``configs/*.json`` / ``configs/smokes/*.json`` reference
   resolves to a real file on disk.
5. Every ``python -m scale_forecasting.<module>`` CLI invocation references an existing module.
6. Canonical counts (models, metrics, analytical views, smoke configs) and current field names stay
   synchronized with the code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.models import list_models
from scale_forecasting.registry.views import VIEW_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "site",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
)

_SUPPORTED_MERMAID_HEADERS = (
    "flowchart ",
    "graph ",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "xychart-beta",
)

_FORBIDDEN_DOC_TOKENS = (
    "covariate_policy",
    '"limit_series"',
    "quickstart_100.json",
    "new_ensemble.json",
    "12 analytical SQL views",
    "12 Analytical SQL Views",
    "4 Analytical SQL Views",
    "4 analytical SQL views",
    "v_residual_distribution",
    "v_forecast_results",
    "scale_forecasting.sdk.Registry.doctor",
    "`global` (zero-shot)",
)


def _all_markdown_files() -> list[Path]:
    files: list[Path] = []
    for top in ("README.md", "AGENTS.md"):
        p = _REPO_ROOT / top
        if p.exists():
            files.append(p)
    for sub in ("docs", "src", "configs", "notebooks", "tests", "docker", "terraform"):
        sub_dir = _REPO_ROOT / sub
        if not sub_dir.exists():
            continue
        for path in sorted(sub_dir.rglob("*.md")):
            rel_parts = path.relative_to(_REPO_ROOT).parts
            if any(part in _EXCLUDED_DIRS for part in rel_parts):
                continue
            # Skip docs/notebooks symlink duplicates (checked via notebooks/ directly).
            if len(rel_parts) >= 2 and rel_parts[0] == "docs" and rel_parts[1] == "notebooks":
                continue
            files.append(path)
    return sorted(files)


def _all_notebook_markdown_sources() -> list[tuple[str, str]]:
    """Return ``(label, markdown_text)`` for every markdown cell in ``notebooks/*.ipynb``."""
    sources: list[tuple[str, str]] = []
    nb_dir = _REPO_ROOT / "notebooks"
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        rel = nb_path.relative_to(_REPO_ROOT)
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                sources.append((f"{rel}#cell{idx}", src))
    return sources


def _strip_fenced_code(markdown: str) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` outside fenced code blocks."""
    out: list[tuple[int, str]] = []
    in_fence = False
    fence_marker = ""
    for idx, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        m = re.match(r"^(`{3,}|~{3,})", stripped)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0] * len(marker)
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if not in_fence:
            out.append((idx, line))
    return out


def _count_table_cols(line: str) -> int:
    """Count table columns by splitting on unescaped ``|`` outside inline code spans."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]
    in_code = False
    cols = 1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        elif ch == "|" and not in_code:
            cols += 1
        i += 1
    return cols


@pytest.fixture(scope="module")
def markdown_files() -> list[Path]:
    return _all_markdown_files()


def test_markdown_tables_column_parity(markdown_files: list[Path]) -> None:
    """Every Markdown table row has the same column count as its header row."""
    errors: list[str] = []
    sources: list[tuple[str, str]] = [
        (str(path.relative_to(_REPO_ROOT)), path.read_text(encoding="utf-8"))
        for path in markdown_files
    ] + _all_notebook_markdown_sources()
    for rel, text in sources:
        non_code_lines = _strip_fenced_code(text)
        i = 0
        while i < len(non_code_lines):
            line_no, line = non_code_lines[i]
            if line.strip().startswith("|") and i + 1 < len(non_code_lines):
                sep_no, sep_line = non_code_lines[i + 1]
                if sep_no == line_no + 1 and re.match(r"^\s*\|[\s:\-\|]+\|\s*$", sep_line):
                    header_cols = _count_table_cols(line)
                    sep_cols = _count_table_cols(sep_line)
                    if header_cols != sep_cols:
                        errors.append(
                            f"{rel}:{line_no}: header has {header_cols} cols, "
                            f"separator has {sep_cols} cols"
                        )
                    j = i + 2
                    prev_no = sep_no
                    while j < len(non_code_lines):
                        row_no, row_line = non_code_lines[j]
                        if row_no != prev_no + 1 or not row_line.strip().startswith("|"):
                            break
                        row_cols = _count_table_cols(row_line)
                        if row_cols != header_cols:
                            errors.append(
                                f"{rel}:{row_no}: table row has {row_cols} cols, "
                                f"expected {header_cols} (header at line {line_no})"
                            )
                        prev_no = row_no
                        j += 1
                    i = j
                    continue
            i += 1
    assert not errors, "Markdown table column parity errors:\n" + "\n".join(errors)


def test_mermaid_blocks_syntax_and_quoting(markdown_files: list[Path]) -> None:
    """Every Mermaid block has a supported header, balanced subgraphs, quoted labels, and <br/>."""
    errors: list[str] = []
    unquoted_node_re = re.compile(r"\b[A-Za-z0-9_]+\[(?![\"\[/\(])([^\]\"\n]*[()<][^\]\"\n]*)\]")
    sources: list[tuple[str, str]] = [
        (str(path.relative_to(_REPO_ROOT)), path.read_text(encoding="utf-8"))
        for path in markdown_files
    ] + _all_notebook_markdown_sources()
    for rel, text in sources:
        lines = text.splitlines()
        in_mermaid = False
        start_line = 0
        block: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not in_mermaid and stripped == "```mermaid":
                in_mermaid = True
                start_line = idx
                block = []
                continue
            if in_mermaid and stripped == "```":
                in_mermaid = False
                non_empty = [
                    ln.strip() for _, ln in block if ln.strip() and not ln.strip().startswith("%%")
                ]
                if not non_empty:
                    errors.append(f"{rel}:{start_line}: empty mermaid block")
                    continue
                first = non_empty[0]
                if not first.startswith(_SUPPORTED_MERMAID_HEADERS):
                    errors.append(f"{rel}:{start_line}: unsupported mermaid header {first!r}")
                subgraphs = sum(1 for _, ln in block if re.match(r"^\s*subgraph\b", ln))
                ends = sum(1 for _, ln in block if re.match(r"^\s*end\s*$", ln))
                if subgraphs != ends:
                    errors.append(
                        f"{rel}:{start_line}: unbalanced mermaid subgraphs "
                        f"({subgraphs} subgraph vs {ends} end)"
                    )
                for b_no, b_line in block:
                    if b_line.strip().startswith("%%"):
                        continue
                    if r"\n" in b_line:
                        errors.append(
                            f"{rel}:{b_no}: literal \\n in mermaid label (use <br/> instead)"
                        )
                    for match in unquoted_node_re.finditer(b_line):
                        errors.append(
                            f"{rel}:{b_no}: unquoted special chars in mermaid label "
                            f"[{match.group(1)}]"
                        )
                continue
            if in_mermaid:
                block.append((idx, line))
    assert not errors, "Mermaid diagram syntax errors:\n" + "\n".join(errors)


def test_embedded_runconfig_json_examples_validate(markdown_files: list[Path]) -> None:
    """Every full RunConfig JSON block in documentation validates cleanly via RunConfig."""
    errors: list[str] = []
    learned = {"nnls", "ridge", "xgb"}
    for path in markdown_files:
        rel = path.relative_to(_REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"```json\n(.*?)```", text, flags=re.DOTALL):
            line_no = text[: match.start()].count("\n") + 1
            raw = match.group(1).strip()
            if not (
                raw.startswith("{")
                and '"run_name"' in raw
                and '"data"' in raw
                and '"models"' in raw
            ):
                continue
            clean = re.sub(r"//.*", "", raw)
            try:
                payload = json.loads(clean)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel}:{line_no}: invalid JSON in RunConfig block: {exc}")
                continue
            try:
                cfg = RunConfig.model_validate(payload)
            except Exception as exc:
                errors.append(f"{rel}:{line_no}: RunConfig validation failed: {exc}")
                continue
            authored_strategies = set((payload.get("ensemble") or {}).get("strategies") or [])
            if authored_strategies & learned and not cfg.backtest.enabled:
                errors.append(
                    f"{rel}:{line_no}: RunConfig example specifies learned ensemble strategies "
                    f"{sorted(authored_strategies & learned)} without backtest.enabled=true"
                )
    assert not errors, "Embedded RunConfig JSON errors:\n" + "\n".join(errors)


def _markdown_anchors(text: str) -> set[str]:
    """Return all heading slugs (GitHub/MkDocs slugify) and explicit ``<a id="...">`` anchors."""
    import pymdownx.slugs

    slugify = pymdownx.slugs.slugify(case="lower")
    anchors = set(re.findall(r'<a\s+id=["\']([^"\']+)["\']', text))
    counts: dict[str, int] = {}
    for _, line in _strip_fenced_code(text):
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if not m:
            continue
        raw = re.sub(r"`([^`]*)`", r"\1", m.group(1))
        raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
        slug = slugify(raw, "-")
        if slug in counts:
            counts[slug] += 1
            anchors.add(f"{slug}_{counts[slug]}")
            anchors.add(f"{slug}-{counts[slug]}")
        else:
            counts[slug] = 0
            anchors.add(slug)
    return anchors


def test_markdown_relative_links_and_config_paths_exist(markdown_files: list[Path]) -> None:
    """Every relative Markdown link, ``#anchor``, and ``configs/*.json`` reference resolves."""
    errors: list[str] = []
    link_re = re.compile(r"\[[^\]]+\]\(([^)#\s]*)(?:#([^)\s]+))?\)")
    cfg_re = re.compile(r"`((?:configs/)?(?:smokes/)?[0-9a-z_]+\.json)`")
    all_json_names = {p.name for p in (_REPO_ROOT / "configs").rglob("*.json")} | {
        p.name for p in (_REPO_ROOT / "tests").rglob("*.json")
    }
    anchor_cache: dict[Path, set[str]] = {}
    for path in markdown_files:
        rel = path.relative_to(_REPO_ROOT)
        for line_no, line in _strip_fenced_code(path.read_text(encoding="utf-8")):
            for match in link_re.finditer(line):
                target, frag = match.group(1), match.group(2)
                if target.startswith(
                    ("http://", "https://", "mailto:", "conversation://", "file://")
                ):
                    continue
                if not target:
                    resolved = path
                else:
                    resolved = (path.parent / target).resolve()
                    # notebooks/README.md is mounted at docs/notebooks/README.md in MkDocs.
                    if not resolved.exists() and path == _REPO_ROOT / "notebooks" / "README.md":
                        resolved = Path(
                            os.path.normpath(_REPO_ROOT / "docs" / "notebooks" / target)
                        )
                if not resolved.exists():
                    errors.append(f"{rel}:{line_no}: broken relative link {target!r}")
                    continue
                if frag and resolved.suffix == ".md":
                    if resolved not in anchor_cache:
                        anchor_cache[resolved] = _markdown_anchors(
                            resolved.read_text(encoding="utf-8")
                        )
                    if frag not in anchor_cache[resolved]:
                        errors.append(
                            f"{rel}:{line_no}: broken anchor #{frag} in "
                            f"{resolved.relative_to(_REPO_ROOT)}"
                        )
            for match in cfg_re.finditer(line):
                ref = match.group(1)
                if ref in {
                    "tiny_config.json",
                    "my_run.json",
                    "config.json",
                    "run_ids_prebreak.json",
                    "golden_panel_prebreak.json",
                }:
                    continue
                if "/" in ref:
                    full = _REPO_ROOT / (ref if ref.startswith("configs/") else f"configs/{ref}")
                    if not full.exists():
                        errors.append(f"{rel}:{line_no}: non-existent config path `{ref}`")
                elif ref not in all_json_names:
                    errors.append(f"{rel}:{line_no}: non-existent config filename `{ref}`")
    assert not errors, "Broken documentation links or config references:\n" + "\n".join(errors)


def test_markdown_cli_module_invocations_exist(markdown_files: list[Path]) -> None:
    """Every ``python -m scale_forecasting.<module>`` reference in Markdown resolves on disk."""
    errors: list[str] = []
    mod_re = re.compile(r"python(?:3)?\s+-m\s+(scale_forecasting(?:\.[A-Za-z0-9_]+)+)")
    for path in markdown_files:
        rel = path.relative_to(_REPO_ROOT)
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in mod_re.finditer(line):
                mod = match.group(1)
                rel_mod = mod.replace(".", "/")
                py_file = _REPO_ROOT / "src" / f"{rel_mod}.py"
                pkg_init = _REPO_ROOT / "src" / rel_mod / "__init__.py"
                if not (py_file.exists() or pkg_init.exists()):
                    errors.append(f"{rel}:{idx}: non-existent CLI module `{mod}`")
    assert not errors, "Broken CLI module references:\n" + "\n".join(errors)


def test_no_stale_config_names_or_counts_in_docs(markdown_files: list[Path]) -> None:
    """No deprecated parameter names appear in docs, and headline counts match code."""
    errors: list[str] = []
    for path in markdown_files:
        rel = path.relative_to(_REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_DOC_TOKENS:
            if token in text:
                errors.append(f"{rel}: contains stale/forbidden documentation token {token!r}")

    n_models = len(list_models())
    n_metrics = len(METRIC_NAMES)
    n_views = len(VIEW_NAMES)
    n_smokes = len(list((_REPO_ROOT / "configs" / "smokes").glob("*.json")))

    root_readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pkg_readme = (_REPO_ROOT / "src" / "scale_forecasting" / "README.md").read_text(
        encoding="utf-8"
    )

    if f"{n_models} models" not in root_readme:
        errors.append(f"README.md missing '{n_models} models'")
    if f"{n_metrics} evaluation metrics" not in root_readme:
        errors.append(f"README.md missing '{n_metrics} evaluation metrics'")
    if f"{n_views} Analytical SQL Views" not in root_readme:
        errors.append(f"README.md missing '{n_views} Analytical SQL Views'")
    if f"{n_models} models" not in pkg_readme:
        errors.append(f"src/scale_forecasting/README.md missing '{n_models} models'")
    if f"{n_metrics} metrics" not in pkg_readme:
        errors.append(f"src/scale_forecasting/README.md missing '{n_metrics} metrics'")
    if n_smokes != 42:
        errors.append(f"Expected 42 smoke configs on disk, found {n_smokes}")

    assert not errors, "Documentation inventory drift:\n" + "\n".join(errors)
