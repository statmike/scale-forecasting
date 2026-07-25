"""Pure 100k-series generator — no I/O (CONTRACTS §6, DESIGN §13.1).

Deterministic math: trend + Fourier multi-seasonality + holidays + AR(1) noise +
intermittency/level-shift/outliers, across ~5 archetype buckets. The golden test
fixture is a tiny call into this same code, so tests and shipped data share one path.

Owned by BUILD step 2.5a. Public surface: ``generate_partition``, ``generate_panel``.
"""

from __future__ import annotations


def generate_partition(id_range: object, cfg: object, seed: int) -> object:  # pragma: no cover
    raise NotImplementedError("data_gen.generator.generate_partition — BUILD step 2.5a")


def generate_panel(n: int, cfg: object, seed: int) -> object:  # pragma: no cover - stub
    raise NotImplementedError("data_gen.generator.generate_panel — BUILD step 2.5a")
