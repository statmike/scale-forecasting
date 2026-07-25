"""Compute engines. Each exposes ``run(cfg)`` with the same shape: build cells →
execute via ``worker.run_cell`` → hand CellResults to ``registry.bq.write_cells``
(CONTRACTS §6). That symmetry is what makes "same code everywhere" (G1) real.
"""

from __future__ import annotations
