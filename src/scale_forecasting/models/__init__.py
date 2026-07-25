"""Model factory (CONTRACTS §1, §6).

Owned by BUILD step 2.5. Importing this package registers every model file by name;
``get_model(name)`` returns the class, ``list_models()`` lists them. Adding a model is
one new file + one ``register`` call — no edits here.
"""

from __future__ import annotations


def get_model(name: str) -> type:  # pragma: no cover - stub, see BUILD 2.5
    raise NotImplementedError("models.get_model — BUILD step 2.5")


def list_models() -> list[str]:  # pragma: no cover - stub, see BUILD 2.5
    raise NotImplementedError("models.list_models — BUILD step 2.5")
