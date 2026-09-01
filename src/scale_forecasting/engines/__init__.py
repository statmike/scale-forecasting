"""Compute engines. Each exposes ``run(cfg)`` with the same shape: build cells →
execute via ``worker.run_cell`` → hand CellResults to ``registry.cells.write_cells``.
That symmetry is what makes "same code everywhere" real.
"""

from __future__ import annotations
