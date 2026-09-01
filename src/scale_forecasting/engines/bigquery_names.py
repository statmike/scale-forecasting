"""What a run's BigQuery objects are called, and where they live.

Two families of reference, and the distinction is not recoverable later, which is why every
builder makes it at the call site: **source tables** are read-only input, qualified against the
source ``dataset`` (`_source_ref`); **model objects** are run *outputs*, keyed by ``run_id`` exactly
like the registry rows and living in the registry dataset (`_model_ref` / `_registry_of`).

`model_object_matches_run` is the inverse of `_model_ref`'s naming rule, and it lives here for that
reason — the two must never drift. It has an outside consumer: a per-run teardown
(`registry.ops.drop_run`) has no other way to find a run's BQML objects, because nothing in the
registry records their names. The only handle is the name itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..registry.ids import make_run_id

if TYPE_CHECKING:
    from ..config import RunConfig


# Every persisted BQML model object this engine creates starts here — the handle a per-run teardown
# matches on (`model_object_matches_run`).
_MODEL_PREFIX = "sf_model_"

def _sanitize_identifier(text: str) -> str:
    """Coerce arbitrary text into a valid BigQuery identifier fragment.

    Custom holiday names must be valid column names — no spaces — because ``ML.EXPLAIN_FORECAST``
    surfaces them as columns. Model object names must likewise be plain identifiers, so the
    hyphenated ``run_id`` is folded here too. Non-alphanumerics collapse to ``_``.
    """
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return out.strip("_") or "x"

def _source_ref(cfg: RunConfig, dataset: str) -> str:
    """Fully-qualify the source table: pass through a dotted name, else qualify against ``dataset``.

    Mirrors ``spark_io._resolve_source_table`` so both runtimes read the identical table.
    """
    src = cfg.data.source_table
    return src if "." in src else f"{dataset}.{src}"

def model_object_matches_run(model_id: str, run_id: str) -> bool:
    """Does a BQML model object name belong to ``run_id`` (the final model or any of its folds)?

    The inverse of `_model_ref`'s naming rule, kept beside it so the two cannot drift. A run's
    persisted model objects are the fourth thing a per-run teardown has to delete — they are
    invisible to the registry tables (nothing records their names), so the only way to find them is
    to list the dataset's models and match the name back to the run. ``model_id`` is the bare object
    id as BigQuery lists it, not a qualified ref.
    """
    if not model_id.startswith(_MODEL_PREFIX):
        return False
    tail = f"_{_sanitize_identifier(run_id)}"
    rest = model_id[len(_MODEL_PREFIX) :]
    if rest.endswith(tail):
        return True
    # A fold object appends _f{k} after the run id.
    head, sep, fold = rest.rpartition("_f")
    return bool(sep) and fold.isdigit() and head.endswith(tail)

def _registry_of(dataset: str, registry_dataset: str | None) -> str:
    """The dataset that owns a run's *outputs* — ``registry_dataset``, else ``dataset``.

    Every builder takes the source ``dataset`` positionally and an optional keyword
    ``registry_dataset``. They are the same string in every deployment that has not set
    ``SF_REGISTRY_DATASET_ID``, so the default keeps existing callers and rendered SQL
    byte-identical — but the *distinction* has to be made at each call site, because which family
    a name belongs to is not recoverable later. See `settings.Settings.registry_dataset_ref`.
    """
    return registry_dataset or dataset

def _model_ref(
    cfg: RunConfig, model_name: str, registry_dataset: str, *, fold_id: int | None = None
) -> str:
    """The backtick-quoted BQML model object path for one ``(model, run[, fold])`` (persisted).

    The object lives in the **registry** dataset, not the source dataset: a ``sf_model_*_{run_id}``
    is a run *output*, keyed by ``run_id`` exactly like the registry rows, and a per-run teardown
    has to be able to enumerate it. Source data is read-only input with a different lifetime.

    The object name embeds the config-pinned ``run_id`` so re-running the same config targets the
    *same* model (``CREATE OR REPLACE`` is idempotent) while a different config gets a distinct
    object. ``fold_id`` (when set) appends an ``_f{k}`` suffix so each backtest fold trains its own
    model object without clobbering the final (true-future) model or the other folds. A model object
    name is an identifier — it cannot be a bound query parameter — so the ``run_id`` is interpolated
    here; it is a pure function of ``cfg`` (`make_run_id`), keeping every builder pure.
    """
    run_id = make_run_id(cfg)
    suffix = f"_f{fold_id}" if fold_id is not None else ""
    stem = f"{_MODEL_PREFIX}{model_name}_{_sanitize_identifier(run_id)}{suffix}"
    return f"`{registry_dataset}.{stem}`"
